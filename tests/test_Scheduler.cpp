// Copyright (C) Jason Lynch

// Tests the queue as a pure function of fixed readings: a bench that answers every call with a pseudo-measurement and
// advances a pretend clock, so that the whole schedule -- the anchor, the order the entries are measured in, the
// interleaving, publication and resumption -- is replayed with no GPU and compared against what it has to be.

#include "Scheduler.h"

#include "Anchor.h"
#include "Emit.h"
#include "Primes.h"
#include "Selection.h"

#include "test.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace tune;

namespace {

bool near(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(std::abs(a), std::abs(b)); }

Env nvidia() {
  Env env;
  env.isNvidia = true;
  env.computeCapability = 806;
  return env;
}

// Four shapes that hold the probe: two FP64 shapes with eight variants each, the second twice the size of the first and
// in its long carry, and two hybrids with one variant each.
std::vector<FFTShape> shapes() {
  return {FFTShape{"512:15:512"}, FFTShape{"1K:8:1K"}, FFTShape{"1:512:8:512"}, FFTShape{"3:1K:8:512"}};
}

RunScope scope() { return makeScope(ScopeArgs{.lo = 110'000'000, .hi = 135'000'000, .probe = 118'063'003}, {}); }

// What the pretend device says each shape costs, in microseconds per iteration, before the variant's own share. The
// broadcast digit is a little slower, as nVidia cards read it.
double pseudoCost(const FFTConfig& fft) {
  static const std::map<std::string, double> byShape{
    {"512:15:512", 1700}, {"1K:8:1K", 3000}, {"1:512:8:512", 1450}, {"3:1K:8:512", 2200}};
  bool const broadcast = variant_W(fft.variant) == 0 || variant_H(fft.variant) == 0;
  return byShape.at(fft.shape.spec()) * (1 + 0.004 * (fft.variant % 7)) * (broadcast ? 1.05 : 1);
}

// A device that answers at once: every call is the pseudo-cost, with a spread a thousandth of it, and takes what a
// call of that cost would, a first build of a configuration paying a compile on top.  The anchor, where there is one,
// is due first and then every ANCHOR_EVERY_SEC of pretend time.
class FakeBench final : public Bench {
public:
  FakeBench(TuneDB& db, u32 sess, bool anchored = true, u32 stopAfter = ~0u) :
    db_{db}, sess_{sess}, anchored_{anchored}, stopAfter_{stopAfter} {}

  std::vector<std::string> order;

  [[nodiscard]] bool anchorDue() const override {
    return anchored_ && !stopped_ && (!anchorTimed_ || clock_ - lastAnchor_ >= ANCHOR_EVERY_SEC);
  }

  // Readings the first anchor item records, as a race for the env's anchor does: spec to cost.
  std::map<std::string, double> race;

  void timeAnchor() override {
    if (!anchorTimed_) {
      for (const auto& [spec, cost] : race) { record(FFTConfig{spec}, TestKind::PRP, 118'063'003, cost); }
    }
    order.push_back("anchor");
    anchorTimed_ = true;
    lastAnchor_ = clock_;
    clock_ += 5;
  }

  [[nodiscard]] Result run(const FFTConfig& fft, TestKind kind, u64 exponent) override {
    // A stop in the middle of a call, as Ctrl-C would make one: nothing is recorded.
    if (calls_ == stopAfter_) {
      stopped_ = true;
      return {};
    }
    ++calls_;

    std::string const spec = fft.spec();
    order.push_back(spec + "@" + std::to_string(exponent));

    double const cost = pseudoCost(fft);
    bool const fresh = built_.insert(spec).second;
    double const seconds = 5 * 1000 * cost * 1e-6 + 1.5 + (fresh ? 12 : 0);
    clock_ += seconds;

    record(fft, kind, exponent, cost);
    return {.completed = true, .seconds = seconds, .usPerIt = cost};
  }

  [[nodiscard]] bool stopped() const override { return stopped_; }

private:
  void record(const FFTConfig& fft, TestKind kind, u64 exponent, double cost) {
    CHECK(db_.add(RunRow{.sess = sess_,
                         .fft = fft.spec(),
                         .kind = kind,
                         .exponent = exponent,
                         .regime = regimeOf(fft, exponent),
                         .cfg = db_.internCfg({}),
                         .m = {.mean = cost,
                               .stddev = cost * 0.001,
                               .blocks = BLOCKS_PER_CALL,
                               .calls = 1,
                               .drift = 1,
                               .status = Status::Ok,
                               .ts = u64(clock_)}}));
  }

  TuneDB& db_;
  u32 sess_;
  bool anchored_;
  u32 stopAfter_;

  u32 calls_ = 0;
  bool stopped_ = false;
  bool anchorTimed_ = false;
  double lastAnchor_ = 0;
  double clock_ = 0;
  std::set<std::string> built_;
};

// A database with the env and one session on it, as Session::begin leaves one.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;

  Fixture() {
    env = db.internEnv(dbEnvOf(nvidia()));
    sess = db.beginSession(env, "512:15:512:212@118063003", 0, 1'753'471'200);
    CHECK(env && sess);
  }

  // A later process on the same database.
  void newSession() { sess = db.beginSession(env, "512:15:512:212@118063003", 0, 1'753'481'200); }
};

std::vector<std::string> runAll(Fixture& f, FakeBench& bench, u32* published = nullptr) {
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes())};
  (void)runQueue(scheduler, f.db, f.env, bench, [&](const Objective&) {
    if (published) { ++*published; }
  });
  return bench.order;
}

// Calls recorded against one spec at one exponent, as the database folds them.
u32 callsOn(const TuneDB& db, const std::string& spec, u64 exponent) {
  u32 n = 0;
  for (const RunRow& row : db.mergedRuns()) {
    if (row.fft == spec && row.exponent == exponent) { n += row.m.calls; }
  }
  return n;
}

// A scratch directory of its own, since publication writes a file.
struct Dir {
  fs::path path;

  explicit Dir(const char* name) : path{fs::temp_directory_path() / name} {
    fs::remove_all(path);
    fs::create_directories(path);
  }

  ~Dir() { fs::remove_all(path); }

  Dir(const Dir&) = delete;
  Dir& operator=(const Dir&) = delete;
};

}  // namespace

TEST(the_gain_prior_values_a_candidate_off_the_pace_above_zero) {
  // At the frontier the saving is the expected gain itself: sum of p * g over the prior, times the cost.
  double expected = 0;
  for (const GainBin& bin : GAIN_PRIOR) { expected += bin.p * bin.gain; }
  CHECK(near(expectedSaving(100, 100), 100 * expected));

  // 40% off the pace: only the 32% bin reaches under the frontier, and a point estimate of the gain would give exactly
  // zero here.
  CHECK(near(expectedSaving(100, 140), 0.01 * (100 - 140 * 0.68)));
  CHECK(expectedSaving(100, 140) > 0);

  // Past the prior's tail nothing is expected.
  CHECK_EQ(expectedSaving(100, 150), 0.0);

  // Cheaper than the frontier as estimated: the whole difference, plus the gain on top of it.
  CHECK(expectedSaving(100, 80) > 20);
}

TEST(a_call_is_expected_to_take_what_calls_have_taken) {
  CallClock clock{1000};

  // Five blocks of 1000 iterations at 2000 us is 10 s of iterations, before anything is learnt about the rest.
  CHECK(near(clock.seconds(2000, false), 10 + CALL_OVERHEAD_SEC));
  CHECK(near(clock.seconds(2000, true), 10 + CALL_OVERHEAD_SEC + COMPILE_ESTIMATE_SEC));

  // A warm call teaches the overhead, and a first build what a compile costs over and above it.
  clock.observe(13, 2000, false);
  CHECK(near(clock.overhead(), 3));
  clock.observe(10 + 3 + 7, 2000, true);
  CHECK(near(clock.compile(), 7));
  CHECK(near(clock.seconds(1000, true), 5 + 3 + 7));

  // A call faster than its own iterations says the overhead is nothing, never less.
  CallClock fast{1000};
  fast.observe(1, 2000, false);
  CHECK_EQ(fast.overhead(), 0.0);
}

TEST(a_baseline_is_offered_for_each_band_the_workload_weighs) {
  std::vector<Baseline> const all = baselines(nvidia(), scope(), shapes());

  // 512:15:512 and 1K:8:1K have eighteen variants each, the hybrids one each; each holds the probe in exactly one
  // regime, and is timed there.
  CHECK_EQ(all.size(), size_t(38));
  for (const Baseline& b : all) {
    CHECK_EQ(b.exponent, u64(118'063'003));
    CHECK(b.band.contains(b.exponent));
    CHECK(b.fft.carry == CARRY_AUTO);
    CHECK(b.kind == TestKind::PRP);
  }
  CHECK_EQ(all.front().label(), std::string{"512:15:512:000 prp short32"});
  CHECK_EQ(all[18].label(), std::string{"1K:8:1K:000 prp long32"});

  // A band the probe is outside is timed at the largest prime it reaches, in the regime being measured: 512:15:512
  // runs short32 up to 143413744, so over a workload above the probe its baseline is timed at its own top.
  RunScope const high = makeScope(ScopeArgs{.lo = 140'000'000, .hi = 160'000'000, .probe = 150'000'001}, {});
  std::vector<Baseline> const top = baselines(nvidia(), high, {FFTShape{"512:15:512"}});
  CHECK(!top.empty());
  for (const Baseline& b : top) {
    CHECK(b.exponent <= b.band.hi && b.exponent >= b.band.lo);
    CHECK(b.exponent < u64(150'000'001));
  }
  CHECK_EQ(top.back().exponent, u64(143'413'741));

  // A shape no grid point falls in is not offered at all.
  RunScope const low = makeScope(ScopeArgs{.lo = 20'000'000, .hi = 21'000'000, .probe = 20'000'003}, {});
  CHECK(baselines(nvidia(), low, {FFTShape{"512:15:512"}}).empty());
}

TEST(fixed_readings_give_a_fixed_schedule) {
  Fixture f;
  FakeBench bench{f.db, f.sess};
  u32 published = 0;
  std::vector<std::string> const order = runAll(f, bench, &published);

  // Anchor first.  Then the shape the stated prior prices cheapest, 512:15:512, at its first variant (every variant of
  // a shape is priced alike, and the order breaks the tie).  That is the broadcast digit, which reads above the prior,
  // and once one FP64 reading exists the fit prices the rest of 512:15:512 at 90% of it, which puts the FFT3161
  // hybrid's stated prior below them: the hybrid is next, ahead of even 000's second call.  From there each entry's
  // second call comes straight after its first, because a configuration already built skips the compile and so costs
  // a fraction of anything unbuilt.  The other seventeen variants follow, each needing a gain of 4% or more over the
  // hybrid to pay, and the anchor falls due once among them and again as they end.  3:1K:8:512 (FFT61) is priced 35%
  // above the hybrid, which only the prior's 32% bin comes near: worth little, but not nothing, so it is measured
  // last.  1K:8:1K is never measured: at twice the size of a measured FP64 shape its fitted prior is past anything the
  // gain prior reaches, and the queue stops with all eighteen of its variants unmeasured.
  std::vector<std::string> const expected{
    "anchor",
    "512:15:512:000@118063003",
    "1:512:8:512:202@118063003",
    "1:512:8:512:202@118063003",
    "512:15:512:000@118063003",
    "512:15:512:001@118063003",
    "512:15:512:001@118063003",
    "512:15:512:002@118063003",
    "512:15:512:002@118063003",
    "512:15:512:010@118063003",
    "512:15:512:010@118063003",
    "512:15:512:011@118063003",
    "512:15:512:011@118063003",
    "512:15:512:012@118063003",
    "512:15:512:012@118063003",
    "512:15:512:100@118063003",
    "512:15:512:100@118063003",
    "512:15:512:101@118063003",
    "512:15:512:101@118063003",
    "512:15:512:102@118063003",
    "anchor",
    "512:15:512:102@118063003",
    "512:15:512:110@118063003",
    "512:15:512:110@118063003",
    "512:15:512:111@118063003",
    "512:15:512:111@118063003",
    "512:15:512:112@118063003",
    "512:15:512:112@118063003",
    "512:15:512:200@118063003",
    "512:15:512:200@118063003",
    "512:15:512:201@118063003",
    "512:15:512:201@118063003",
    "512:15:512:202@118063003",
    "512:15:512:202@118063003",
    "512:15:512:210@118063003",
    "512:15:512:210@118063003",
    "512:15:512:211@118063003",
    "512:15:512:211@118063003",
    "512:15:512:212@118063003",
    "512:15:512:212@118063003",
    "anchor",
    "3:1K:8:512:202@118063003",
    "3:1K:8:512:202@118063003",
  };

  CHECK(order == expected);

  // Once before the first item and once after each.
  CHECK_EQ(published, u32(1 + std::ranges::count_if(order, [](const std::string& s) { return s != "anchor"; })));

  // And the same readings give the same schedule every time.
  Fixture again;
  FakeBench second{again.db, again.sess};
  CHECK(runAll(again, second) == order);
}

TEST(the_same_configuration_is_not_run_twice_while_another_is_close) {
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes())};

  auto item = [](size_t index, double value, double seconds) {
    return Item{
      .kind = ItemKind::Baseline, .index = index, .exponent = 118'063'003, .value = value, .seconds = seconds};
  };
  auto picked = [&](const std::vector<Item>& ranked) {
    std::optional<Item> const next = scheduler.pick(ranked);
    CHECK(next.has_value());
    return next ? next->index : ~size_t(0);
  };

  std::vector<Item> const ranked{item(0, 100, 10), item(1, 95, 10), item(2, 50, 10)};
  CHECK_EQ(picked(ranked), size_t(0));

  // Having just run the top item, the runner-up within 10% of it goes first...
  scheduler.ran(ranked[0], 10, 1700);
  CHECK_EQ(picked(ranked), size_t(1));

  // ...but not one further off, which would be trading value for a tie-break.
  std::vector<Item> const clear{item(0, 100, 10), item(1, 85, 10)};
  CHECK_EQ(picked(clear), size_t(0));

  // Nothing of any value is nothing to run.
  CHECK(!scheduler.pick({item(0, 0, 10)}).has_value());
  CHECK(!scheduler.pick({}).has_value());

  // The anchor is not an entry, so it does not count as the one just run.
  scheduler.ran({.kind = ItemKind::Anchor}, 0, 0);
  CHECK_EQ(picked(ranked), size_t(0));
}

TEST(an_interrupted_run_leaves_a_valid_selection_file_and_a_rerun_resumes) {
  Dir const dir{"prpll-test-scheduler-resume"};
  fs::path const out = dir.path / "selection.txt";

  auto publisher = [&](const TuneDB& db, u32 env) {
    return [&db, env, &out](const Objective& objective) {
      Provenance const from{
        .ts = 1'753'471'200, .db = "tunedb.txt", .env = env, .T = objective.T(), .workloadLo = 1, .workloadHi = 2};
      CHECK(publish(out, db, {}, from));
    };
  };

  // Uninterrupted, for what a whole run concludes.
  Fixture whole;
  FakeBench wholeBench{whole.db, whole.sess};
  std::vector<std::string> const all = runAll(whole, wholeBench);

  // Stopped in the middle of the fourth call: the hybrid is concluded, 512:15:512:000 has one call of two, and the call
  // that was cut short recorded nothing.
  Fixture f;
  FakeBench first{f.db, f.sess, true, 3};
  Scheduler one{scope(), baselines(nvidia(), scope(), shapes())};
  QueueReport const stopped = runQueue(one, f.db, f.env, first, publisher(f.db, f.env));
  CHECK(stopped.stopped);
  CHECK_EQ(stopped.items, 3u);
  CHECK_EQ(callsOn(f.db, "512:15:512:000", 118'063'003), 1u);

  // What was published is a file production reads, holding exactly what had concluded.
  std::optional<SelectionFile> const file = readSelection(out);
  CHECK(file.has_value());
  CHECK_EQ(file->entries.size(), size_t(1));
  CHECK_EQ(file->entries.front().fft, std::string{"1:512:8:512:202"});
  CHECK(file->provenance.find("T=1487.4") != std::string::npos);

  // A later process on the same database finishes 512:15:512:000 first -- one call left is the cheapest thing on offer,
  // though this process has never built it -- at the exponent it was started at, and repeats nothing concluded.
  f.newSession();
  FakeBench second{f.db, f.sess};
  Scheduler two{scope(), baselines(nvidia(), scope(), shapes())};
  QueueReport const resumed = runQueue(two, f.db, f.env, second, publisher(f.db, f.env));
  CHECK(!resumed.stopped);
  CHECK(second.order.size() >= 2);
  CHECK_EQ(second.order[1], std::string{"512:15:512:000@118063003"});
  CHECK(std::ranges::count(second.order, std::string{"1:512:8:512:202@118063003"}) == 0);
  CHECK_EQ(callsOn(f.db, "512:15:512:000", 118'063'003), MIN_CALLS);

  // Between them the two runs measured what one whole run does, each entry the calls it needs and no more.
  std::map<std::string, u32> split;
  std::map<std::string, u32> single;
  for (const std::string& s : first.order) { split[s] += s != "anchor"; }
  for (const std::string& s : second.order) { split[s] += s != "anchor"; }
  for (const std::string& s : all) { single[s] += s != "anchor"; }
  split.erase("anchor");
  single.erase("anchor");
  CHECK(split == single);

  // Every run republishes the whole frontier: the hybrid, which covers every exponent 512:15:512 does, for less.
  std::optional<SelectionFile> const last = readSelection(out);
  CHECK(last.has_value());
  CHECK_EQ(last->entries.size(), size_t(1));
  CHECK_EQ(last->entries.front().fft, std::string{"1:512:8:512:202"});
}

TEST(what_failed_or_what_an_earlier_generation_died_on_is_not_offered_again) {
  Fixture f;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes())};
  u32 const defaults = f.db.internCfg({});

  auto offered = [&](const std::string& spec) {
    Objective const objective{f.db, f.env, scheduler.scope()};
    for (const Item& item : scheduler.admissible(f.db, f.env, objective)) {
      if (scheduler.baselines()[item.index].fft.spec() == spec) { return true; }
    }
    return false;
  };

  CHECK(offered("512:15:512:101"));
  CHECK(offered("512:15:512:102"));
  CHECK(offered("1:512:8:512:202"));

  // A build that failed is a verdict; repeating it would only repeat it.
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = "512:15:512:101",
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(FFTConfig{"512:15:512:101"}, 118'063'003),
                        .cfg = defaults,
                        .m = {.status = Status::NoCompile, .ts = 1}}));
  CHECK(!offered("512:15:512:101"));

  // An attempt an earlier process was holding when it died is not made again.
  u32 const dead = f.db.beginSession(f.env, "512:15:512:212@118063003", 1, 1'753'471'300);
  CHECK(f.db.add(TryRow{
    .sess = dead, .fft = "512:15:512:102", .kind = TestKind::PRP, .exponent = 118'063'003, .cfg = defaults, .ts = 2}));
  f.db.sealSession(dead);
  CHECK(!offered("512:15:512:102"));

  // Another env's rows say nothing about this one.
  u32 const other = f.db.internEnv(dbEnvOf(Env{.isAmd = true}));
  u32 const there = f.db.beginSession(other, "", 0, 1'753'471'400);
  CHECK(f.db.add(RunRow{.sess = there,
                        .fft = "1:512:8:512:202",
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(FFTConfig{"1:512:8:512:202"}, 118'063'003),
                        .cfg = defaults,
                        .m = {.status = Status::Err, .ts = 3}}));
  CHECK(offered("1:512:8:512:202"));
}

TEST(an_unanchored_session_starts_measuring_at_once) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  std::vector<std::string> const order = runAll(f, bench);
  CHECK(!order.empty());
  CHECK(std::ranges::count(order, std::string{"anchor"}) == 0);
  CHECK_EQ(order.front(), std::string{"512:15:512:000@118063003"});
}

TEST(a_started_entry_is_finished_where_it_was_started) {
  // Started by a run whose probe was elsewhere in the band: the call still to make pools with the one made only at the
  // same exponent, so that is where it is made, whatever this run's probe is.
  Fixture f;
  u64 const elsewhere = Primes{}.prevPrime(125'000'000);
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = "512:15:512:101",
                        .kind = TestKind::PRP,
                        .exponent = elsewhere,
                        .regime = regimeOf(FFTConfig{"512:15:512:101"}, elsewhere),
                        .cfg = f.db.internCfg({}),
                        .m = {.mean = 1720, .stddev = 1, .blocks = 4, .calls = 1, .status = Status::Ok, .ts = 1}}));

  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes())};
  Objective const objective{f.db, f.env, scheduler.scope()};
  std::vector<Item> const ranked = scheduler.admissible(f.db, f.env, objective);

  auto const at = std::ranges::find_if(
    ranked, [&](const Item& item) { return scheduler.baselines()[item.index].fft.spec() == "512:15:512:101"; });
  CHECK(at != ranked.end());
  CHECK_EQ(at->exponent, elsewhere);
  CHECK_EQ(at->calls, 1u);

  // Its neighbours, never started, are timed at the probe.
  for (const Item& item : ranked) {
    if (item.calls == 0) { CHECK_EQ(item.exponent, u64(118'063'003)); }
  }
}

TEST(the_first_item_follows_what_the_anchor_race_read) {
  // A card slow at FP64: the race for its anchor reads 512:15:512 at 5000 us/it and the hybrid at 1450.  The stated
  // prior would start on 512:15:512; what the race read says the hybrid, whose one call is then finished first.
  Fixture f;
  FakeBench bench{f.db, f.sess};
  bench.race = {{"512:15:512:212", 5000}, {"1:512:8:512:202", 1450}};
  std::vector<std::string> const order = runAll(f, bench);

  CHECK(order.size() >= 2);
  CHECK_EQ(order[0], std::string{"anchor"});
  CHECK_EQ(order[1], std::string{"1:512:8:512:202@118063003"});
}
