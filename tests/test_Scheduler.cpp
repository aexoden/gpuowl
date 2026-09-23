// Copyright (C) Jason Lynch

// Tests the queue as a pure function of fixed readings: a bench that answers every call with a pseudo-measurement and
// advances a pretend clock, so that the whole schedule -- the anchor, the order the entries are measured in, the
// interleaving, publication and resumption -- is replayed with no GPU and compared against what it has to be.

#include "Scheduler.h"

#include "Anchor.h"
#include "Emit.h"
#include "FFTVariants.h"
#include "Primes.h"
#include "Selection.h"

#include "test.h"

#include <cmath>
#include <cstdio>
#include <functional>
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

  // What the device builds when asked for an option set: what was asked, unless a test says the host sets part of it
  // aside.
  std::function<UseConfig(const UseConfig&)> builtAs = [](const UseConfig& asked) { return asked; };

  // What a first build of a configuration costs on top of its call.
  double compileSeconds = 12;

  // What an option set does to a configuration's cost, as a factor; none by default.
  std::function<double(const FFTConfig&, const UseConfig&)> optionFactor = [](const FFTConfig&, const UseConfig&) {
    return 1.0;
  };

  [[nodiscard]] Result run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                           const std::string&) override {
    // A stop in the middle of a call, as Ctrl-C would make one: nothing is recorded.
    if (calls_ == stopAfter_) {
      stopped_ = true;
      return {};
    }
    ++calls_;

    std::string const spec = fft.spec();
    order.push_back(spec + "@" + std::to_string(exponent) + (options.empty() ? "" : " " + configText(options)));

    double const cost = pseudoCost(fft) * optionFactor(fft, options);
    bool const fresh = built_.insert(spec + " " + configText(options)).second;
    double const seconds = 5 * 1000 * cost * 1e-6 + 1.5 + (fresh ? compileSeconds : 0);
    clock_ += seconds;

    UseConfig const built = builtAs(options);
    record(fft, kind, exponent, cost, built);
    return {.completed = true, .seconds = seconds, .usPerIt = cost, .ran = built};
  }

  [[nodiscard]] bool stopped() const override { return stopped_; }

private:
  void record(const FFTConfig& fft, TestKind kind, u64 exponent, double cost, const UseConfig& options = {}) {
    CHECK(db_.add(RunRow{.sess = sess_,
                         .fft = fft.spec(),
                         .kind = kind,
                         .exponent = exponent,
                         .regime = regimeOf(fft, exponent),
                         .cfg = db_.internCfg(options),
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
  (void)runQueue(scheduler, f.db, f.env, bench, [&](const Objective&, const Defaults&) {
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
  // a shape is priced alike, and the order breaks the tie); each entry's second call straight after its first, because
  // a configuration already built skips the compile and so costs a fraction of anything unbuilt.  Once one FP64 reading
  // exists the fit prices the rest of 512:15:512 at 90% of it, which puts the FFT3161 hybrid's stated prior below
  // them, and it is next.  The other seventeen variants follow, each needing a gain of 4% or more over the hybrid to
  // pay, and the anchor falls due once among them and again as they end.  3:1K:8:512 (FFT61) is priced 35% above the
  // hybrid, which only the prior's 32% bin comes near: worth little, but not nothing, so it is measured next.  1K:8:1K
  // comes last: at twice the size of a measured FP64 shape its fitted prior is about twice the hybrid's, which only the
  // prior's 64% bin reaches under, and the queue stops once all eighteen of its variants are measured.  The anchor
  // falls due twice more among them.
  std::vector<std::string> const expected{
    "anchor",
    "512:15:512:000@118063003",
    "512:15:512:000@118063003",
    "1:512:8:512:202@118063003",
    "1:512:8:512:202@118063003",
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
    "1K:8:1K:000@118063003",
    "1K:8:1K:000@118063003",
    "1K:8:1K:001@118063003",
    "1K:8:1K:001@118063003",
    "1K:8:1K:002@118063003",
    "1K:8:1K:002@118063003",
    "1K:8:1K:010@118063003",
    "1K:8:1K:010@118063003",
    "1K:8:1K:011@118063003",
    "1K:8:1K:011@118063003",
    "1K:8:1K:012@118063003",
    "anchor",
    "1K:8:1K:012@118063003",
    "1K:8:1K:100@118063003",
    "1K:8:1K:100@118063003",
    "1K:8:1K:101@118063003",
    "1K:8:1K:101@118063003",
    "1K:8:1K:102@118063003",
    "1K:8:1K:102@118063003",
    "1K:8:1K:110@118063003",
    "1K:8:1K:110@118063003",
    "1K:8:1K:111@118063003",
    "1K:8:1K:111@118063003",
    "1K:8:1K:112@118063003",
    "1K:8:1K:112@118063003",
    "1K:8:1K:200@118063003",
    "anchor",
    "1K:8:1K:200@118063003",
    "1K:8:1K:201@118063003",
    "1K:8:1K:201@118063003",
    "1K:8:1K:202@118063003",
    "1K:8:1K:202@118063003",
    "1K:8:1K:210@118063003",
    "1K:8:1K:210@118063003",
    "1K:8:1K:211@118063003",
    "1K:8:1K:211@118063003",
    "1K:8:1K:212@118063003",
    "1K:8:1K:212@118063003",
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
    return [&db, env, &out](const Objective& objective, const Defaults& defaults) {
      Provenance const from{
        .ts = 1'753'471'200, .db = "tunedb.txt", .env = env, .T = objective.T(), .workloadLo = 1, .workloadHi = 2};
      CHECK(publish(out, db, defaults, from));
    };
  };

  // Uninterrupted, for what a whole run concludes.
  Fixture whole;
  FakeBench wholeBench{whole.db, whole.sess};
  std::vector<std::string> const all = runAll(whole, wholeBench);

  // Stopped in the middle of the fourth call: 512:15:512:000 is concluded, the hybrid has one call of two, and the call
  // that was cut short recorded nothing.
  Fixture f;
  FakeBench first{f.db, f.sess, true, 3};
  Scheduler one{scope(), baselines(nvidia(), scope(), shapes())};
  QueueReport const stopped = runQueue(one, f.db, f.env, first, publisher(f.db, f.env));
  CHECK(stopped.stopped);
  CHECK_EQ(stopped.items, 3u);
  CHECK_EQ(callsOn(f.db, "1:512:8:512:202", 118'063'003), 1u);

  // What was published is a file production reads, holding exactly what had concluded.
  std::optional<SelectionFile> const file = readSelection(out);
  CHECK(file.has_value());
  CHECK_EQ(file->entries.size(), size_t(1));
  CHECK_EQ(file->entries.front().fft, std::string{"512:15:512:000"});
  CHECK(file->provenance.find("T=1788.1") != std::string::npos);

  // A later process on the same database finishes the hybrid first -- one call left is the cheapest thing on offer,
  // though this process has never built it -- at the exponent it was started at, and repeats nothing concluded.
  f.newSession();
  FakeBench second{f.db, f.sess};
  Scheduler two{scope(), baselines(nvidia(), scope(), shapes())};
  QueueReport const resumed = runQueue(two, f.db, f.env, second, publisher(f.db, f.env));
  CHECK(!resumed.stopped);
  CHECK(second.order.size() >= 2);
  CHECK_EQ(second.order[1], std::string{"1:512:8:512:202@118063003"});
  CHECK(std::ranges::count(second.order, std::string{"512:15:512:000@118063003"}) == 0);
  CHECK_EQ(callsOn(f.db, "1:512:8:512:202", 118'063'003), MIN_CALLS);

  // Between them the two runs measured what one whole run does, each entry the calls it needs and no more.
  std::map<std::string, u32> split;
  std::map<std::string, u32> single;
  for (const std::string& s : first.order) { split[s] += s != "anchor"; }
  for (const std::string& s : second.order) { split[s] += s != "anchor"; }
  for (const std::string& s : all) { single[s] += s != "anchor"; }
  split.erase("anchor");
  single.erase("anchor");
  CHECK(split == single);

  // Every run republishes the whole frontier: the hybrid now covers every exponent 512:15:512:000 did, for less.
  std::optional<SelectionFile> const last = readSelection(out);
  CHECK(last.has_value());
  // And 1K:8:1K:112, the cheapest of the variants the gain prior's tail was worth measuring at twice the hybrid's cost,
  // for the exponents past the hybrid's reach that its long carry still serves.
  CHECK(last.has_value());
  CHECK_EQ(last->entries.size(), size_t(2));
  CHECK_EQ(last->entries.front().fft, std::string{"1:512:8:512:202"});
  CHECK_EQ(last->entries.back().fft, std::string{"1K:8:1K:112"});
  CHECK(last->entries.back().reach > last->entries.front().reach);
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

namespace {

// Two families of the four shapes: FP64 on 512:15:512 and FFT3161 on 1:512:8:512, each at its default variant.
Bootstrap twoFamilies(bool enabled = true) {
  std::vector<Family> families;
  for (const char* spec : {"512:15:512", "1:512:8:512"}) {
    FFTShape const shape{spec};
    families.push_back({.type = shape.fft_type, .fft = FFTConfig{shape, defaultVariant(shape), CARRY_AUTO}});
  }
  return Bootstrap{nvidia(), 118'063'003, families, enabled};
}

// WMUL=1 saves 2% everywhere and TAIL_KERNELS=3 saves 2% on FP64 only; every other move costs 1%.
double planted(const FFTConfig& fft, const UseConfig& options) {
  double factor = 1;
  for (const auto& [key, value] : options) {
    bool const wins =
      (key == "WMUL" && value == "1") || (key == "TAIL_KERNELS" && value == "3" && fft.shape.fft_type == FFT64);
    factor *= wins ? 0.98 : 1.01;
  }
  return factor;
}

struct BootstrapRun {
  std::vector<std::string> order;
  Defaults defaults;
  QueueReport report;
};

BootstrapRun runBootstrapped(Fixture& f, u32 stopAfter = ~0u,
                             const std::function<UseConfig(const UseConfig&)>& builtAs = {}, bool enabled = true) {
  FakeBench bench{f.db, f.sess, false, stopAfter};
  bench.optionFactor = planted;
  if (builtAs) { bench.builtAs = builtAs; }
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(enabled)};

  BootstrapRun out;
  out.report = runQueue(scheduler, f.db, f.env, bench,
                        [&](const Objective&, const Defaults& defaults) { out.defaults = defaults; });
  out.order = bench.order;
  return out;
}

bool isFamilyCall(const std::string& call) {
  return call.starts_with(twoFamilies().families()[0].fft.spec() + "@") ||
    call.starts_with(twoFamilies().families()[1].fft.spec() + "@");
}

}  // namespace

TEST(the_bootstrap_runs_first_and_every_baseline_runs_at_what_it_decided) {
  Fixture f;
  BootstrapRun const run = runBootstrapped(f);
  const std::vector<std::string>& order = run.order;

  // Each family read at its defaults before anything is raced.
  CHECK(order.size() > 2);
  CHECK(isFamilyCall(order[0]) && order[0].find(' ') == std::string::npos);
  CHECK(isFamilyCall(order[1]) && order[1].find(' ') == std::string::npos);

  // Every bootstrap call comes before every baseline: the baselines run at what the races decide.
  auto const firstBaseline = std::ranges::find_if(order, [](const std::string& s) { return !isFamilyCall(s); });
  CHECK(firstBaseline != order.end());
  CHECK(std::none_of(firstBaseline, order.end(), isFamilyCall));

  // A race calls its candidates turn about: never the same one twice running.
  for (auto it = order.begin(); std::next(it) < firstBaseline; ++it) { CHECK(*it != *std::next(it)); }

  // What the families agreed on is the global line, and the key only FP64 moved is FP64's.
  CHECK_EQ(configText(run.defaults.global), std::string{"WMUL=1"});
  CHECK_EQ(run.defaults.family.size(), size_t(1));
  if (run.defaults.family.size() == 1) {
    std::vector<std::pair<std::string, std::string>> const tailKernels{{"TAIL_KERNELS", "3"}};
    CHECK(run.defaults.family[0].selector.type == FFT64);
    CHECK(run.defaults.family[0].uses == tailKernels);
  }

  // Every baseline ran at the lines as production resolves them for its type.
  for (auto it = firstBaseline; it != order.end(); ++it) {
    FFTConfig const fft{it->substr(0, it->find('@'))};
    std::string const want = fft.shape.fft_type == FFT64 ? "TAIL_KERNELS=3,WMUL=1" : "WMUL=1";
    CHECK_EQ(it->substr(it->find(' ') + 1), want);
  }
}

TEST(a_bootstrap_interrupted_carries_on_from_its_rows) {
  Fixture whole;
  BootstrapRun const all = runBootstrapped(whole);

  // Stopped part way through the first family's races, and carried on by a later process.
  Fixture f;
  BootstrapRun const first = runBootstrapped(f, 40);
  CHECK(first.report.stopped);
  CHECK(first.defaults.global.empty() && first.defaults.family.empty());

  f.newSession();
  BootstrapRun const second = runBootstrapped(f);
  CHECK(!second.report.stopped);

  // Between them the two runs made exactly the calls one whole run does, and came to the same lines.
  std::map<std::string, u32> split;
  std::map<std::string, u32> single;
  for (const std::string& s : first.order) { ++split[s]; }
  for (const std::string& s : second.order) { ++split[s]; }
  for (const std::string& s : all.order) { ++single[s]; }
  CHECK(split == single);
  CHECK_EQ(configText(second.defaults.global), configText(all.defaults.global));
  CHECK_EQ(second.defaults.family.size(), all.defaults.family.size());
}

TEST(with_the_bootstrap_off_the_baselines_run_at_the_built_in_defaults) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = planted;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(false)};
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});

  // The same schedule as a scheduler that has no bootstrap at all.
  Fixture g;
  FakeBench plain{g.db, g.sess, false};
  CHECK(runAll(g, plain) == bench.order);
}

TEST(a_candidate_the_host_builds_otherwise_drops_out_of_its_race) {
  // A host that sets every WMUL aside, as it sets OLD_FENCE=0 aside off AMD and nVidia: the calls land on the
  // incumbent, never on the candidate, and without a limit the race would ask for it for ever.
  // Stopped after 3000 calls, far more than the run needs, so that a race that never ends fails here rather than hangs.
  Fixture f;
  BootstrapRun const run = runBootstrapped(f, 3000, [](const UseConfig& asked) {
    UseConfig built = asked;
    built.erase("WMUL");
    return built;
  });

  CHECK(!run.report.stopped);
  std::map<std::string, u32> asked;
  for (const std::string& s : run.order) {
    if (s.find("WMUL=") != std::string::npos && isFamilyCall(s)) { ++asked[s]; }
  }
  CHECK(!asked.empty());
  for (const auto& [call, n] : asked) { CHECK(n <= 2 * MIN_CALLS); }

  // It decides nothing about WMUL, and TAIL_KERNELS=3 still wins on FP64.
  CHECK(run.defaults.global.find("WMUL") == run.defaults.global.end());
  CHECK_EQ(run.defaults.family.size(), size_t(1));
}

TEST(baselines_measured_under_earlier_defaults_are_measured_again_under_new_ones) {
  // A run with the bootstrap off measures every baseline at the built-in defaults; turned on, the races decide lines
  // those rows do not match, emission would drop them, and so the baselines are owed again at the lines.
  Fixture f;
  BootstrapRun const off = runBootstrapped(f, ~0u, {}, false);
  CHECK(std::ranges::none_of(off.order, [](const std::string& s) { return s.find(' ') != std::string::npos; }));

  f.newSession();
  BootstrapRun const on = runBootstrapped(f);
  CHECK_EQ(configText(on.defaults.global), std::string{"WMUL=1"});
  CHECK(std::ranges::count(on.order, std::string{"512:15:512:101@118063003 TAIL_KERNELS=3,WMUL=1"}) == MIN_CALLS);
  CHECK(std::ranges::count(on.order, std::string{"3:1K:8:512:202@118063003 WMUL=1"}) == MIN_CALLS);
}

namespace {

constexpr const char* PROBED = "512:15:512:212";

// WMUL=1 saves 5%.  LDSPAD_W=0 costs 2% at the default WMUL and saves 3% at WMUL=1, so it only pays once WMUL has
// moved -- and LDSPAD_W depends on WMUL, so its probe is owed again once WMUL=1 is the best set.  Every other key
// costs 1%.
double interacting(const FFTConfig&, const UseConfig& options) {
  bool const wmul1 = useValue(options, "WMUL", 2) == 1;
  double factor = wmul1 ? 0.95 : 1;
  if (useValue(options, "LDSPAD_W", 1) == 0) { factor *= wmul1 ? 0.97 : 1.02; }
  for (const auto& [key, value] : options) {
    if (key != "WMUL" && key != "LDSPAD_W") { factor *= 1.01; }
  }
  return factor;
}

struct ProbeRun {
  std::vector<std::string> order;
  QueueReport report;
  std::string best;
};

ProbeRun runProbed(Fixture& f, u32 stopAfter = ~0u, const std::function<UseConfig(const UseConfig&)>& builtAs = {}) {
  FakeBench bench{f.db, f.sess, false, stopAfter};
  bench.optionFactor = interacting;
  if (builtAs) { bench.builtAs = builtAs; }

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  CHECK_EQ(one.size(), size_t(1));

  Scheduler scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false},
                      Strategy{.kind = Strategy::Kind::Single}};
  ProbeRun out;
  out.report = runQueue(scheduler, f.db, f.env, bench, [&](const Objective& objective, const Defaults&) {
    for (const SelectionEntry& e : objective.entries()) {
      if (e.fft == PROBED) {
        out.best = configText(canonicalConfig(nvidia(), FFTConfig{PROBED}, e.opts));
        break;
      }
    }
  });
  out.order = bench.order;
  return out;
}

std::map<std::string, u32> tally(const std::vector<std::string>& order) {
  std::map<std::string, u32> out;
  for (const std::string& s : order) { ++out[s]; }
  return out;
}

}  // namespace

TEST(a_measured_entry_is_probed_until_no_step_improves_it) {
  Fixture f;
  ProbeRun const run = runProbed(f);
  CHECK(!run.report.stopped);

  // WMUL=1 wins the first round of one-step probes, which re-offers LDSPAD_W=0 (it depends on WMUL) and nothing
  // else; that wins too, and opens LDSSWIZ_W, which does not.
  CHECK_EQ(run.best, std::string{"LDSPAD_W=0,WMUL=1"});

  std::string const at = std::string{PROBED} + "@118063003";
  std::map<std::string, u32> const calls = tally(run.order);
  CHECK_EQ(calls.at(at), MIN_CALLS);
  CHECK_EQ(calls.at(at + " WMUL=1"), MIN_CALLS);
  CHECK_EQ(calls.at(at + " LDSPAD_W=0"), MIN_CALLS);
  CHECK_EQ(calls.at(at + " LDSPAD_W=0,WMUL=1"), MIN_CALLS);
  CHECK_EQ(calls.at(at + " LDSPAD_W=0,LDSSWIZ_W=1,WMUL=1"), MIN_CALLS);

  // ZEROHACK_W depends on nothing that moved, so its first reading stands.
  CHECK_EQ(calls.at(at + " ZEROHACK_W=0"), MIN_CALLS);
  CHECK(!calls.contains(at + " WMUL=1,ZEROHACK_W=0"));

  // Every configuration is measured exactly as often as a row needs: the baseline, the first round's one-step moves,
  // and one probe in each of the two rounds after it.
  ProbeList const first = probesOf(nvidia(), FFTConfig{PROBED}, {}, {.kind = Strategy::Kind::Single});
  CHECK_EQ(calls.size(), 1 + first.probes.size() + 2);
  for (const auto& [call, n] : calls) { CHECK_EQ(n, MIN_CALLS); }
}

TEST(probing_interrupted_carries_on_from_its_rows) {
  Fixture whole;
  ProbeRun const all = runProbed(whole);

  Fixture f;
  ProbeRun const first = runProbed(f, 31);
  CHECK(first.report.stopped);
  f.newSession();
  ProbeRun const second = runProbed(f);
  CHECK(!second.report.stopped);

  std::map<std::string, u32> split = tally(first.order);
  for (const auto& [call, n] : tally(second.order)) { split[call] += n; }
  CHECK(split == tally(all.order));
  CHECK_EQ(second.best, all.best);
}

TEST(a_probe_the_host_builds_otherwise_is_given_up) {
  // A host that sets every WMUL aside: those probes' rows land on other configurations, so they are tried at most
  // MAX_ATTEMPTS times and the search carries on without them.  Bounded, so that a probe asked for for ever fails here
  // rather than hangs.
  Fixture f;
  ProbeRun const run = runProbed(f, 3000, [](const UseConfig& asked) {
    UseConfig built = asked;
    built.erase("WMUL");
    return built;
  });
  CHECK(!run.report.stopped);

  u32 wmul = 0;
  for (const auto& [call, n] : tally(run.order)) {
    if (call.find("WMUL=") == std::string::npos) { continue; }
    ++wmul;
    CHECK(n <= 2 * MIN_CALLS);
  }
  CHECK(wmul > 0);
  CHECK(run.best.find("WMUL") == std::string::npos);
}

TEST(the_bootstrap_still_runs_first_when_entries_are_probed) {
  Fixture plain;
  BootstrapRun const without = runBootstrapped(plain);
  auto const firstBaseline = std::ranges::find_if(without.order, [](const std::string& s) { return !isFamilyCall(s); });
  size_t const bootstrapCalls = size_t(firstBaseline - without.order.begin());

  Fixture f;
  FakeBench bench{f.db, f.sess, false, u32(bootstrapCalls + 60)};
  bench.optionFactor = planted;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(),
                      Strategy{.kind = Strategy::Kind::Single}};
  Defaults lines;
  (void)runQueue(scheduler, f.db, f.env, bench, [&](const Objective&, const Defaults& d) { lines = d; });

  // Call for call the same bootstrap, and probes only after it: a probe measures a step from what the bootstrap
  // decides, so one taken earlier measures a step from something production is not going to run.
  CHECK(bench.order.size() > bootstrapCalls);
  CHECK(std::equal(without.order.begin(), firstBaseline, bench.order.begin()));

  // Once it is complete, entries are probed: some call runs at neither the lines nor the family's own line.
  CHECK(std::any_of(bench.order.begin() + ptrdiff_t(bootstrapCalls), bench.order.end(), [](const std::string& s) {
    std::string const opts = s.substr(s.find(' ') + 1);
    return s.find(' ') != std::string::npos && opts != "WMUL=1" && opts != "TAIL_KERNELS=3,WMUL=1";
  }));

  // A probe that moves a key back to its built-in value beside a line that sets it otherwise names it, since a row
  // that left it out would be one the line shadows, and emission would never publish it.
  CHECK_EQ(configText(lines.global), std::string{"WMUL=1"});
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), lines});
  bool wmulBack = false;
  for (const Item& item : items) {
    if (item.kind != ItemKind::Probe) { continue; }
    const Baseline& b = scheduler.baselines()[item.index];
    CHECK(!shadowedBy(lines, nvidia(), b.fft, b.kind, item.options));
    wmulBack = wmulBack || (item.what.ends_with("WMUL=2") && item.options.at("WMUL") == "2");
  }
  CHECK(wmulBack);
}

TEST(a_probe_names_every_key_a_line_would_set_once_its_own_keys_are_in_place) {
  // FFT3161 at width 512 offers L2_STRIPING up to 512/64 = 8 alone and 512/128 = 4 beside MULTI_Q=1, so under these
  // lines L2_STRIPING=8 is fitted away.  A probe back to MULTI_Q=0 makes it legal again: the probe must name
  // L2_STRIPING at its own value, or the line shadows its row and emission never publishes it.
  FFTConfig const fft{"1:512:8:512:202"};
  Defaults const lines{.global = {{"L2_STRIPING", "8"}, {"MULTI_Q", "1"}}, .family = {}};

  UseConfig const probe = besideLines(nvidia(), fft, TestKind::PRP, lines, {});
  CHECK_EQ(configText(probe), std::string{"L2_STRIPING=0,MULTI_Q=0"});
  CHECK(!shadowedBy(lines, nvidia(), fft, TestKind::PRP, probe));

  // Beside MULTI_Q=1 the line is fitted away, and nothing needs naming but the probe's own key.
  UseConfig const kept = besideLines(nvidia(), fft, TestKind::PRP, lines, {{"MULTI_Q", "1"}});
  CHECK_EQ(configText(kept), std::string{"MULTI_Q=1"});
  CHECK(!shadowedBy(lines, nvidia(), fft, TestKind::PRP, kept));
}

TEST(two_contested_entries_are_alternated_not_batched) {
  // Two variants of 512:15:512, the shape the stated prior prices cheapest, 0.4% apart, on a device where a build costs
  // no more than a call.  The prior prices a shape and not its variants, so the two are worth the same until both have
  // concluded, and each is within INTERLEAVE_EPS of the other throughout.
  std::vector<Baseline> pair;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.variant == 101 || b.fft.variant == 102) { pair.push_back(b); }
  }
  CHECK_EQ(pair.size(), size_t(2));

  auto run = [&](double compileSeconds) {
    Fixture f;
    FakeBench bench{f.db, f.sess, false};
    bench.compileSeconds = compileSeconds;
    Scheduler scheduler{scope(), pair};
    (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});
    return bench.order;
  };

  std::vector<std::string> const alternated{
    "512:15:512:101@118063003",
    "512:15:512:102@118063003",
    "512:15:512:101@118063003",
    "512:15:512:102@118063003",
  };
  CHECK(run(0) == alternated);

  // Where a build costs 12 s, a configuration already built is cheaper to call again by more than INTERLEAVE_EPS: the
  // two are no longer close, and value per second, not a tie-break, orders them.
  std::vector<std::string> const batched{
    "512:15:512:101@118063003",
    "512:15:512:101@118063003",
    "512:15:512:102@118063003",
    "512:15:512:102@118063003",
  };
  CHECK(run(12) == batched);
}

TEST(an_entry_whose_moves_have_paid_is_probed_ahead_of_one_whose_have_not) {
  Fixture f;
  std::vector<FFTShape> const one{FFTShape{"512:15:512"}};
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), one), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};

  auto add = [&](const std::string& fft, const UseConfig& opts, double mean) {
    CHECK(f.db.add(RunRow{.sess = f.sess,
                          .fft = fft,
                          .kind = TestKind::PRP,
                          .exponent = 118'063'003,
                          .regime = regimeOf(FFTConfig{fft}, 118'063'003),
                          .cfg = f.db.internCfg(opts),
                          .m = {.mean = mean,
                                .stddev = 0.1,
                                .blocks = 4 * MIN_CALLS,
                                .calls = MIN_CALLS,
                                .drift = 1,
                                .status = Status::Ok,
                                .ts = 0}}));
  };

  // Both end at 1700 us/it, so each probe of either would save the same were the gains they have shown not counted:
  // one got there by a 9% move, the other has had two moves that found nothing.
  add("512:15:512:211", {}, 1870);
  add("512:15:512:211", {{"WMUL", "1"}}, 1700);
  add("512:15:512:212", {}, 1700);
  add("512:15:512:212", {{"ZEROHACK_W", "0"}}, 1710);
  add("512:15:512:212", {{"LDSPAD_W", "0"}}, 1705);

  Objective const objective{f.db, f.env, scheduler.scope()};
  double paid = -1;
  double flat = -1;
  for (const Item& item : scheduler.admissible(f.db, f.env, objective)) {
    if (item.kind != ItemKind::Probe) { continue; }
    std::string const spec = scheduler.baselines()[item.index].fft.spec();
    double& at = spec == "512:15:512:211" ? paid : flat;
    CHECK(at < 0 || at == item.value);
    at = item.value;
  }

  CHECK(flat > 0);
  // Both are what production runs across the band, so each probe is worth W * c * the mean gain of its entry's
  // distribution, and the two differ by exactly that.  One 9% move against the prior's 8 pseudo-observations, mixed
  // half back into the device's, is worth ~30% more a probe than two moves that found nothing.
  GainModel const gains = gainsOf(f.db, f.env);
  double const own = gains.forEntry({"512:15:512:211", TestKind::PRP, "short32"}).mean();
  double const other = gains.forEntry({"512:15:512:212", TestKind::PRP, "short32"}).mean();
  CHECK(std::abs(paid / flat - own / other) < 1e-9);
  CHECK(paid > 1.25 * flat);
}

TEST(an_unmeasured_entry_is_valued_under_the_gains_the_device_has_shown) {
  Fixture f;
  std::vector<FFTShape> const one{FFTShape{"512:15:512"}};
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), one)};

  // One entry whose every move took 16% off: a device on which the defaults are badly placed.
  double cost = 2400;
  u32 moves = 0;
  for (const UseConfig& opts :
       std::vector<UseConfig>{{}, {{"WMUL", "1"}}, {{"LDSPAD_W", "0"}}, {{"ZEROHACK_W", "0"}}, {{"LOADS", "3"}}}) {
    CHECK(f.db.add(RunRow{.sess = f.sess,
                          .fft = "512:15:512:211",
                          .kind = TestKind::PRP,
                          .exponent = 118'063'003,
                          .regime = regimeOf(FFTConfig{"512:15:512:211"}, 118'063'003),
                          .cfg = f.db.internCfg(opts),
                          .m = {.mean = cost,
                                .stddev = 0.1,
                                .blocks = 4 * MIN_CALLS,
                                .calls = MIN_CALLS,
                                .drift = 1,
                                .status = Status::Ok,
                                .ts = 0}}));
    moves += !opts.empty();
    cost *= 0.84;
  }
  GainDist const shown = gainsOf(f.db, f.env).global();
  CHECK(near(gainsOf(f.db, f.env).all().n(), moves));
  CHECK(shown.mean() > 2 * GAIN_PRIOR.mean());

  Objective const objective{f.db, f.env, scheduler.scope()};
  u32 checked = 0;
  for (const Item& item : scheduler.admissible(f.db, f.env, objective)) {
    const Baseline& b = scheduler.baselines()[item.index];
    double const estimate = objective.priorModel().cost(b.fft.shape);
    CHECK(near(item.value, expectedSaving(objective.points(), b.kind, b.band, estimate, shown)));
    CHECK(item.value > expectedSaving(objective.points(), b.kind, b.band, estimate, GAIN_PRIOR));
    ++checked;
  }
  CHECK(checked > 0);
}
