// Copyright (C) Jason Lynch

// Tests the queue as a pure function of fixed readings: a bench that answers every call with a pseudo-measurement and
// advances a pretend clock, so that the whole schedule -- the anchor, the order the entries are measured in, the
// interleaving, publication and resumption -- is replayed with no GPU and compared against what it has to be.

#include "Scheduler.h"

#include "Anchor.h"
#include "Args.h"
#include "Emit.h"
#include "FFTVariants.h"
#include "Primes.h"
#include "Production.h"
#include "Progress.h"
#include "Selection.h"
#include "Status.h"
#include "Summary.h"

#include "test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
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

  void declareRestart(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 k) override {
    CHECK(db_.add(JumpRow{.sess = sess_,
                          .fft = fft.spec(),
                          .kind = kind,
                          .regime = regimeOf(fft, exponent),
                          .cfg = db_.internCfg(options),
                          .k = k,
                          .ts = u64(clock_)}));
  }

  void declareCombo(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 tier) override {
    CHECK(db_.add(ComboRow{.sess = sess_,
                           .fft = fft.spec(),
                           .kind = kind,
                           .regime = regimeOf(fft, exponent),
                           .cfg = db_.internCfg(options),
                           .tier = tier,
                           .ts = u64(clock_)}));
  }

  void declareBootstrap(const FFTConfig& fft, u64 probe) override {
    CHECK(db_.add(BootRow{.sess = sess_, .fft = fft.spec(), .probe = probe, .ts = u64(clock_)}));
  }

  void declareRound(const RoundRow& round) override {
    RoundRow row = round;
    row.sess = sess_;
    row.ts = u64(clock_);
    CHECK(db_.add(row));
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

  // What an accuracy check reads for an option set at an exponent: a z the floor of every type is clear of, by default.
  std::function<double(const FFTConfig&, const UseConfig&, u64)> zOf = [](const FFTConfig&, const UseConfig&, u64) {
    return 24.0;
  };

  [[nodiscard]] Reading gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) override {
    if (calls_ == stopAfter_) {
      stopped_ = true;
      return {};
    }
    ++calls_;

    order.push_back("gate " + fft.spec() + "@" + std::to_string(exponent) +
                    (options.empty() ? "" : " " + configText(options)));
    double const seconds = ROE_ITERATIONS * pseudoCost(fft) * 1e-6 + 1.5 + compileSeconds;
    clock_ += seconds;

    UseConfig const built = builtAs(options);
    double const z = zOf(fft, built, exponent);
    CHECK(db_.add(RoeRow{.sess = sess_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = db_.internCfg(built),
                         .z = z,
                         .n = 2000,
                         .maxRoe = 0.3,
                         .checkOk = true,
                         .ts = u64(clock_)}));
    return {.completed = true, .seconds = seconds, .z = z, .n = 2000, .checkOk = true, .ran = built};
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

std::vector<std::string> runAll(Fixture& f, FakeBench& bench, u32* published = nullptr, bool gate = false) {
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, gate};
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

  // Nor is anything worth less than the floor, however quickly it would be done; what is worth the floor runs in its
  // place, and interleaves only with what is worth the floor as well.
  std::vector<Item> const cheap{item(0, 2, 1), item(1, 10, 100), item(2, 9, 100)};
  std::optional<Item> const worth = scheduler.pick(cheap, 5);
  CHECK(worth && worth->index == 1);
  CHECK(!scheduler.pick(cheap, 20).has_value());

  // A gate reading and a bootstrap call run by rule, whatever they are valued at.
  Item gate = item(0, 1, 10);
  gate.kind = ItemKind::Gate;
  std::optional<Item> const next = scheduler.pick({gate}, 1000);
  CHECK(next && next->kind == ItemKind::Gate);
  Item call = item(0, 1, 10);
  call.kind = ItemKind::Bootstrap;
  CHECK(worthRunning(call, 1000));
  CHECK(!worthRunning(item(0, 1, 10), 1000));

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
  std::vector<std::string> const all = runAll(whole, wholeBench, nullptr, true);

  // Stopped in the middle of the fifth call: 512:15:512:212, the default variant, taken first since nothing covered the
  // workload, is concluded and its accuracy read, the hybrid has one call of two, and the call that was cut short
  // recorded nothing.
  Fixture f;
  FakeBench first{f.db, f.sess, true, 4};
  Scheduler one{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  QueueReport const stopped = runQueue(one, f.db, f.env, first, publisher(f.db, f.env));
  CHECK(stopped.stopped);
  CHECK_EQ(stopped.items, 4u);
  CHECK_EQ(callsOn(f.db, "1:512:8:512:202", 118'063'003), 1u);

  // What was published is a file production reads, holding exactly what had concluded.
  std::optional<SelectionFile> const file = readSelection(out);
  CHECK(file.has_value());
  CHECK_EQ(file->entries.size(), size_t(1));
  CHECK_EQ(file->entries.front().fft, std::string{"512:15:512:212"});
  CHECK(file->provenance.find("T=1716.6") != std::string::npos);

  // A later process on the same database finishes the hybrid first -- one call left is the cheapest thing on offer,
  // though this process has never built it -- at the exponent it was started at, and repeats nothing concluded.
  f.newSession();
  FakeBench second{f.db, f.sess};
  Scheduler two{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  QueueReport const resumed = runQueue(two, f.db, f.env, second, publisher(f.db, f.env));
  CHECK(!resumed.stopped);
  CHECK(second.order.size() >= 2);
  CHECK_EQ(second.order[1], std::string{"1:512:8:512:202@118063003"});
  CHECK(std::ranges::count(second.order, std::string{"512:15:512:212@118063003"}) == 0);
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

  // Every run republishes the whole frontier: the hybrid now covers every exponent 512:15:512:212 did, for less.
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
    Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
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
  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
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

TEST(the_bootstrap_runs_first_and_every_baseline_runs_at_the_built_in_defaults) {
  Fixture f;
  BootstrapRun const run = runBootstrapped(f);
  const std::vector<std::string>& order = run.order;

  // Each family read at its defaults before anything is raced.
  CHECK(order.size() > 2);
  CHECK(isFamilyCall(order[0]) && order[0].find(' ') == std::string::npos);
  CHECK(isFamilyCall(order[1]) && order[1].find(' ') == std::string::npos);

  // With no search to choose contenders for there is no defaults sweep, so every bootstrap call comes before every
  // baseline.
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

  // Every baseline ran at the built-in defaults: what the entry costs untuned, whatever the lines say.
  for (auto it = firstBaseline; it != order.end(); ++it) { CHECK(it->find(' ') == std::string::npos); }
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

TEST(a_baseline_measured_before_the_bootstrap_is_not_measured_again) {
  // A run with the bootstrap off measures every baseline at the built-in defaults; turned on, the races decide lines
  // those rows do not match.  A row names everything it ran, so it is still a measurement of its entry: the second run
  // makes its bootstrap calls and nothing else, each family racing on its type's cheapest reading at the probe, and
  // what the first measured is published beside the new lines, naming what they would change.
  Fixture f;
  BootstrapRun const off = runBootstrapped(f, ~0u, {}, false);
  CHECK(std::ranges::none_of(off.order, [](const std::string& s) { return s.find(' ') != std::string::npos; }));

  f.newSession();
  BootstrapRun const on = runBootstrapped(f);
  std::vector<Family> const families = twoFamilies().familiesIn(f.db, f.env);
  CHECK(!on.order.empty());
  CHECK(std::ranges::all_of(on.order, [&](const std::string& call) {
    return std::ranges::any_of(families, [&](const Family& fam) { return call.starts_with(fam.fft.spec() + "@"); });
  }));
  CHECK(std::ranges::none_of(families, [](const Family& fam) { return fam.fft.spec() == "512:15:512:212"; }));
  CHECK(!on.defaults.global.empty() || !on.defaults.family.empty());

  auto const file = emit(f.db, on.defaults, Provenance{.ts = 0, .db = {}, .env = f.env});
  CHECK(file.has_value());
  if (!file) { return; }
  CHECK(!file->entries.empty());
  for (const SelectionEntry& e : file->entries) {
    CHECK(shadowedKeys(Args{true}, nvidia(), *file, e, FFTConfig{e.fft}).empty());
  }
}

TEST(a_baseline_is_taken_at_the_built_in_defaults_whatever_the_lines_say) {
  // With the bootstrap off the lines start empty.  FFT61 rounds nothing, so a row of it is published as soon as it
  // concludes: here one at the probe under WMUL=1, the set a search of it found.  The lines then carry WMUL=1, but each
  // baseline still owed is taken at the built-in defaults: what the entry costs untuned, which is what everything its
  // search finds is a gain over.
  Fixture f;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(false)};
  std::string const spec = "3:1K:8:512:202";
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = spec,
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(FFTConfig{spec}, 118'063'003),
                        .cfg = f.db.internCfg({{"WMUL", "1"}}),
                        .m = {.mean = 2000,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));

  Defaults const lines = scheduler.lines(f.db, f.env, scheduler.bootstrapState(f.db, f.env));
  CHECK_EQ(linesText(lines), std::string{"WMUL=1"});

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  u32 underIt = 0;
  for (const Item& item : scheduler.admissible(f.db, f.env, objective)) {
    if (item.kind != ItemKind::Baseline) { continue; }
    const Baseline& b = scheduler.baselines()[item.index];
    CHECK(!(b.fft.spec() == spec && b.band.contains(118'063'003)));
    CHECK(item.options.empty());
    underIt += underDefaults(nvidia(), b.fft, b.kind, lines).contains("WMUL");
  }
  CHECK(underIt > 0);
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

// The one entry PROBED, searched one axis at a time with the bootstrap off.
Scheduler probedScheduler(bool restarts) {
  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  CHECK_EQ(one.size(), size_t(1));

  return Scheduler{
    scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false}, Strategy{.kind = Strategy::Kind::Single},
    restarts};
}

struct ProbeRun {
  std::vector<std::string> order;
  QueueReport report;
  std::string best;
  GainModel gains;
};

ProbeRun runProbed(Fixture& f, u32 stopAfter = ~0u, const std::function<UseConfig(const UseConfig&)>& builtAs = {},
                   bool restarts = false, double stop = 0) {
  FakeBench bench{f.db, f.sess, false, stopAfter};
  bench.optionFactor = interacting;
  if (builtAs) { bench.builtAs = builtAs; }

  Scheduler scheduler = probedScheduler(restarts);
  ProbeRun out;
  // As the search sees it: with the gate turned off nothing is published but what rounds nothing.
  out.report = runQueue(
    scheduler, f.db, f.env, bench,
    [&](const Objective&, const Defaults&) {
      for (const SelectionEntry& e : entriesFor(f.db, f.env, Gating::Assumed)) {
        if (e.fft == PROBED) {
          out.best = configText(canonicalConfig(nvidia(), FFTConfig{PROBED}, e.opts));
          break;
        }
      }
    },
    stop);
  out.order = bench.order;
  out.gains = gainsOf(f.db, f.env);
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

namespace {

// The kind and the label of every item a queue finished, in order.
class Record final : public Watch {
public:
  std::vector<std::pair<ItemKind, std::string>> done;

  void measuring(const std::string&) override {}
  void progress(const RunProgress&) override {}
  void finished(const Finished& f) override { done.emplace_back(f.kind, f.label); }
};

}  // namespace

TEST(the_defaults_sweep_then_the_bootstrap_run_before_any_entry_is_searched) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false, 800};
  bench.optionFactor = planted;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(),
                      Strategy{.kind = Strategy::Kind::Single}};
  Defaults lines;
  Record record;
  (void)runQueue(scheduler, f.db, f.env, bench, [&](const Objective&, const Defaults& d) { lines = d; }, 0, &record);
  const auto& done = record.done;

  // First every contender at the built-in defaults: the hybrid, the cheapest, is one; 3:1K:8:512, half as dear
  // again, is not.
  auto const sweepEnd = std::ranges::find_if(done, [](const auto& i) {
    return i.first != ItemKind::Baseline || !i.second.ends_with("at the built-in defaults");
  });
  CHECK(std::any_of(done.begin(), sweepEnd, [](const auto& i) { return i.second.starts_with("1:512:8:512:202 "); }));
  CHECK(std::none_of(done.begin(), sweepEnd, [](const auto& i) { return i.second.starts_with("3:1K:8:512:"); }));

  // Then the bootstrap, whole, on the families the run recorded once the sweep was done; a search step measures a
  // step from what the bootstrap decides, so one taken earlier measures a step from lines that are about to move.
  CHECK(scheduler.bootstrap().chosen(f.db, f.env));
  auto const bootstrapEnd =
    std::find_if(sweepEnd, done.end(), [](const auto& i) { return i.first != ItemKind::Bootstrap; });
  CHECK(bootstrapEnd != sweepEnd);
  CHECK(std::none_of(bootstrapEnd, done.end(), [](const auto& i) { return i.first == ItemKind::Bootstrap; }));
  CHECK(std::any_of(bootstrapEnd, done.end(), [](const auto& i) { return i.first == ItemKind::Probe; }));

  // A step from a set at a key's built-in value, beside a line that sets it otherwise, records the set it ran, and what
  // it would publish names the key, so that it runs as it was measured.
  CHECK_EQ(configText(lines.global), std::string{"WMUL=1"});
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  bool builtIn = false;
  for (const Item& item : items) {
    const Baseline& b = scheduler.baselines()[item.index];
    if (item.kind != ItemKind::Probe || item.options.contains("WMUL") ||
        !underDefaults(nvidia(), b.fft, b.kind, lines).contains("WMUL")) {
      continue;
    }
    builtIn = true;
    UseConfig const published = besideLines(nvidia(), b.fft, b.kind, lines, item.options);
    CHECK_EQ(published.at("WMUL"), std::string{"2"});
  }
  CHECK(builtIn);
}

namespace {

// A concluded row of `spec` at the probe under `options`.
void concludedAt(Fixture& f, const std::string& spec, const UseConfig& options, double mean) {
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = spec,
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(FFTConfig{spec}, 118'063'003),
                        .cfg = f.db.internCfg(options),
                        .m = {.mean = mean,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));
}

std::vector<Baseline> only(std::initializer_list<const char*> specs) {
  std::vector<Baseline> out;
  for (const Baseline& b : baselines(nvidia(), scope(), shapes())) {
    if (std::ranges::any_of(specs, [&](const char* s) { return b.fft.spec() == s; })) { out.push_back(b); }
  }
  return out;
}

}  // namespace

TEST(the_sweep_reads_every_entry_within_the_margin_at_the_built_in_defaults) {
  Fixture f;
  Scheduler const scheduler{
    scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};
  auto sweep = [&] {
    std::set<std::string> out;
    for (const Item& item : scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed})) {
      if (item.sweep) {
        CHECK(item.options.empty());
        out.insert(scheduler.baselines()[item.index].fft.spec());
      }
    }
    return out;
  };

  // The hybrid at 1450 is the fastest; 512:15:512, read at 1700, is 17% behind, and its other variants, priced from
  // that reading, with it.  Nothing is owed.
  concludedAt(f, "1:512:8:512:202", {}, 1450);
  concludedAt(f, "512:15:512:212", {}, 1700);
  CHECK(sweep().empty());
  CHECK(scheduler.swept(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed}));

  // A variant read under other options at 1500 is within 10%, and owes its reading at the defaults whatever else it
  // has been read at; the shape's prior is its cheapest reading, so its other variants come within the margin with it.
  // The one already read at the defaults does not, nor the hybrid, nor the NTT half as dear again.
  concludedAt(f, "512:15:512:101", {{"WMUL", "1"}}, 1500);
  std::set<std::string> const owed = sweep();
  CHECK(owed.contains("512:15:512:101") && owed.contains("512:15:512:000") && owed.contains("512:15:512:211"));
  CHECK(!owed.contains("512:15:512:212") && !owed.contains("1:512:8:512:202") && !owed.contains("3:1K:8:512:202"));
  CHECK(!scheduler.swept(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed}));

  // Read there, it is swept.
  concludedAt(f, "512:15:512:101", {}, 1620);
  CHECK(!sweep().contains("512:15:512:101"));
}

TEST(a_failed_reading_at_the_built_in_defaults_is_not_swept_again) {
  Fixture f;
  FFTConfig const failed{"512:15:512:101"};
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = failed.spec(),
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(failed, 118'063'003),
                        .cfg = f.db.internCfg({}),
                        .m = {.status = Status::Err}}));
  Scheduler const scheduler{scope(), only({"512:15:512:101"}), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};

  CHECK(std::ranges::none_of(scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed}),
                             [](const Item& i) { return i.sweep; }));
  CHECK(scheduler.swept(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed}));
}

TEST(a_prior_that_cannot_be_read_does_not_keep_measured_entries_from_contending) {
  // Two variants of 1K:8:1K read 3% apart, and the smaller 512:15:512, whose prior is well below both, failing at the
  // built-in defaults: the prior will never be replaced by a reading, and the two measured entries are what production
  // runs.
  Fixture f;
  concludedAt(f, "1K:8:1K:101", {}, 3000);
  concludedAt(f, "1K:8:1K:102", {}, 3090);
  FFTConfig const failed{"512:15:512:101"};
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = failed.spec(),
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(failed, 118'063'003),
                        .cfg = f.db.internCfg({}),
                        .m = {.status = Status::Err}}));
  std::vector<Baseline> entries = only({"1K:8:1K:101", "1K:8:1K:102", "512:15:512:101"});
  std::erase_if(entries, [](const Baseline& b) { return !b.band.contains(118'063'003); });
  Scheduler const scheduler{scope(),
                            entries,
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 2, .roundCalls = 16}};

  (void)scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(scheduler.lastHalving().active);
  CHECK_EQ(scheduler.lastHalving().pool.size(), size_t{2});
}

TEST(a_cheaper_entry_still_to_be_read_is_read_before_the_margin_takes_in_dearer_ones) {
  // 512:15:512 read at 1700 prices 512:16:512 about 7% dearer, within 10% of it; but the hybrid, not read yet, is
  // priced well below both.  It is read first, and only its reading says whether 512:16:512 is still within the margin.
  Fixture f;
  concludedAt(f, "512:15:512:212", {}, 1700);
  std::vector<Baseline> entries;
  for (const Baseline& b :
       baselines(nvidia(), scope(), {FFTShape{"512:15:512"}, FFTShape{"512:16:512"}, FFTShape{"1:512:8:512"}})) {
    if (b.band.contains(118'063'003) &&
        (b.fft.spec() == "512:15:512:212" || b.fft.spec() == "512:16:512:101" || b.fft.spec() == "1:512:8:512:202")) {
      entries.push_back(b);
    }
  }
  CHECK_EQ(entries.size(), size_t{3});
  Scheduler const scheduler{scope(), entries, 1000, {}, Strategy{.kind = Strategy::Kind::Single}};
  auto sweep = [&] {
    std::set<std::string> out;
    for (const Item& item : scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed})) {
      if (item.sweep) { out.insert(scheduler.baselines()[item.index].fft.spec()); }
    }
    return out;
  };

  CHECK(sweep() == std::set<std::string>{"1:512:8:512:202"});
  concludedAt(f, "1:512:8:512:202", {}, 1800);
  CHECK(sweep() == std::set<std::string>{"512:16:512:101"});
}

TEST(where_nothing_measured_serves_a_point_an_entry_that_cannot_be_read_does_not_set_its_margin) {
  // Nothing measured anywhere, and the cheapest shape by its prior failing at the built-in defaults: the NTT, half as
  // dear again, is the cheapest thing left that could serve the workload, and is read.
  Fixture f;
  FFTConfig const failed{"512:15:512:101"};
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = failed.spec(),
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(failed, 118'063'003),
                        .cfg = f.db.internCfg({}),
                        .m = {.status = Status::Err}}));
  Scheduler const scheduler{
    scope(), only({"512:15:512:101", "3:1K:8:512:202"}), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};

  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(std::ranges::any_of(
    items, [&](const Item& i) { return i.sweep && scheduler.baselines()[i.index].fft.spec() == "3:1K:8:512:202"; }));
}

TEST(a_sweep_reading_is_resumed_where_it_was_started) {
  // One call at the prime below the probe, as a run over an earlier probe would have left it.
  Fixture f;
  FFTConfig const fft{"512:15:512:101"};
  u64 const started = Primes{}.prevPrime(118'063'003);
  CHECK(f.db.add(
    RunRow{.sess = f.sess,
           .fft = fft.spec(),
           .kind = TestKind::PRP,
           .exponent = started,
           .regime = regimeOf(fft, started),
           .cfg = f.db.internCfg({}),
           .m = {.mean = 1700, .stddev = 0.1, .blocks = 4, .calls = 1, .drift = 1, .status = Status::Ok, .ts = 0}}));
  Scheduler const scheduler{scope(), only({"512:15:512:101"}), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};

  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(!items.empty() && items.front().sweep);
  if (items.empty()) { return; }
  CHECK_EQ(items.front().exponent, started);
  CHECK_EQ(items.front().calls, 1u);
}

TEST(an_entry_at_the_built_in_defaults_tries_the_lines_first) {
  // FFT61 rounds nothing, so its row under WMUL=1 is published with the gate off, and the lines carry WMUL=1.  The
  // hybrid is measured only at the defaults.
  Fixture f;
  Scheduler const scheduler{
    scope(), only({"1:512:8:512:202", "3:1K:8:512:202"}), 1000, {}, Strategy{.kind = Strategy::Kind::Single}};
  concludedAt(f, "3:1K:8:512:202", {{"WMUL", "1"}}, 2000);
  concludedAt(f, "1:512:8:512:202", {}, 1450);
  CHECK_EQ(linesText(scheduler.lines(f.db, f.env, scheduler.bootstrapState(f.db, f.env))), std::string{"WMUL=1"});

  auto firstOf = [&](const std::string& spec) -> std::optional<Item> {
    for (const Item& item : scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed})) {
      if (item.kind == ItemKind::Probe && scheduler.baselines()[item.index].fft.spec() == spec) { return item; }
    }
    return {};
  };

  // The hybrid's first step is the lines, all at once; FFT61's best set is already its own, so it has none.
  std::optional<Item> const hybrid = firstOf("1:512:8:512:202");
  CHECK(hybrid && hybrid->what == "the default lines WMUL=1" && configText(hybrid->options) == "WMUL=1");
  std::optional<Item> const ntt = firstOf("3:1K:8:512:202");
  CHECK(ntt && ntt->what.find("the default lines") == std::string::npos);

  // Answered, it is not offered again, and the steps go on from whichever set is now the best.
  concludedAt(f, "1:512:8:512:202", {{"WMUL", "1"}}, 1400);
  std::optional<Item> const next = firstOf("1:512:8:512:202");
  CHECK(next && next->what.find("the default lines") == std::string::npos);
}

TEST(a_bootstrap_completed_before_its_choice_was_recorded_is_not_raced_again) {
  // A database written before the choice was recorded: the same run, with the boot rows it now writes taken out.
  Fixture whole;
  (void)runBootstrapped(whole);
  std::string legacy;
  for (const std::string& line : split(whole.db.text(), '\n')) {
    if (!line.empty() && !line.starts_with("boot ")) { legacy += line + "\n"; }
  }
  Fixture f;
  CHECK(f.db.parse(legacy, "legacy"));
  CHECK(f.db.boots().empty());
  f.newSession();

  FakeBench bench{f.db, f.sess, false, 60};
  bench.optionFactor = planted;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(),
                      Strategy{.kind = Strategy::Kind::Single}};
  Record record;
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, 0, &record);

  // Recorded on the families it raced on, and nothing of it raced again.
  CHECK(scheduler.bootstrap().chosen(f.db, f.env));
  CHECK(scheduler.bootstrap().familiesIn(f.db, f.env).front().fft.spec() ==
        twoFamilies().families().front().fft.spec());
  CHECK(scheduler.bootstrapState(f.db, f.env).complete);
  CHECK(!record.done.empty());
  CHECK(std::ranges::none_of(record.done, [](const auto& i) { return i.first == ItemKind::Bootstrap; }));
}

TEST(each_phase_says_how_far_through_it_the_run_is_and_never_goes_back) {
  // The whole of a small run: the sweep, the bootstrap of two families, the halving and the search after it.
  class Phases final : public Watch {
  public:
    std::vector<Phase> seen;
    void measuring(const std::string&) override {}
    void progress(const RunProgress& p) override { seen.push_back(p.phase); }
  };
  // The variants of one shape, within a few percent of one another, so that several contend.
  Fixture f;
  FakeBench bench{f.db, f.sess, false, 600};
  bench.optionFactor = planted;
  FFTShape const shape{"512:15:512"};
  Scheduler scheduler{scope(),
                      baselines(nvidia(), scope(), {shape}),
                      1000,
                      Bootstrap{nvidia(), 118'063'003, {{.type = FFT64, .fft = FFTConfig{"512:15:512:212"}}}},
                      Strategy{.kind = Strategy::Kind::Single},
                      false,
                      false,
                      Halving{.contenders = 4, .roundCalls = 2}};
  Phases phases;
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, 0, &phases);

  // "<a> of <b>" after `prefix` in `text`.
  auto fraction = [](const std::string& text, const std::string& prefix) -> std::optional<std::pair<u32, u32>> {
    size_t const at = text.find(prefix);
    if (at == std::string::npos) { return {}; }
    u32 a = 0;
    u32 b = 0;
    if (sscanf(text.c_str() + at + prefix.size(), "%u of %u", &a, &b) != 2) { return {}; }
    return std::pair{a, b};
  };

  std::set<std::string> kinds;
  std::optional<std::pair<u32, u32>> lastSweep;
  std::map<std::string, u32> lastGroup;
  for (const Phase& p : phases.seen) {
    CHECK(!p.text.empty() && !p.brief.empty());
    kinds.insert(p.text.substr(0, p.text.find(':')));
    if (auto const s = fraction(p.text, "defaults sweep: ")) {
      CHECK(s->first <= s->second);
      CHECK(!lastSweep || s->first >= lastSweep->first);
      lastSweep = s;
    }
    if (auto const g = fraction(p.text, ", group ")) {
      std::string const family = p.text.substr(0, p.text.find(", group "));
      CHECK(g->first >= 1 && g->first <= g->second);
      CHECK(g->first >= lastGroup[family]);
      lastGroup[family] = g->first;
    }
  }
  CHECK(kinds.contains("defaults sweep"));
  CHECK(kinds.contains("bootstrap"));
  CHECK(kinds.contains("halving"));
  CHECK(lastGroup.size() == 1);
}

TEST(an_ll_only_run_measures_and_publishes_nothing_in_prp) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false, 400};
  bench.optionFactor = planted;
  RunScope const ll =
    makeScope(ScopeArgs{.lo = 110'000'000, .hi = 135'000'000, .probe = 118'063'003, .kinds = {TestKind::LL}}, {});
  std::vector<Family> families;
  for (const char* spec : {"512:15:512", "1:512:8:512"}) {
    FFTShape const shape{spec};
    families.push_back({.type = shape.fft_type, .fft = FFTConfig{shape, defaultVariant(shape), CARRY_AUTO}});
  }
  Scheduler scheduler{ll,
                      baselines(nvidia(), ll, shapes()),
                      1000,
                      Bootstrap{nvidia(), 118'063'003, families, true, COMBO_TIERS, TestKind::LL},
                      Strategy{.kind = Strategy::Kind::Single},
                      false,
                      false,
                      Halving{.contenders = 4, .roundCalls = 2}};
  Record record;
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, 0, &record);

  CHECK(std::ranges::any_of(record.done, [](const auto& i) { return i.first == ItemKind::Bootstrap; }));
  CHECK(std::ranges::any_of(record.done, [](const auto& i) {
    return i.first == ItemKind::Bootstrap && i.second.find(" ll ") != std::string::npos;
  }));
  CHECK(std::ranges::none_of(f.db.mergedRuns(), [](const RunRow& r) { return r.kind == TestKind::PRP; }));
  auto const file = emit(f.db, scheduler.lines(f.db, f.env, scheduler.bootstrapState(f.db, f.env)),
                         Provenance{.ts = 0, .db = {}, .env = f.env});
  CHECK(file.has_value());
  if (file) {
    CHECK(std::ranges::none_of(file->entries, [](const SelectionEntry& e) { return e.kind == TestKind::PRP; }));
  }
}

TEST(the_first_round_takes_one_variant_of_each_shape_before_a_second_of_any) {
  // Three variants of 512:15:512 ahead of the hybrid, which is still within the margin.
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  concludedAt(f, "512:15:512:102", {}, 1455);
  concludedAt(f, "512:15:512:110", {}, 1460);
  concludedAt(f, "1:512:8:512:202", {}, 1500);
  Scheduler const scheduler{scope(),
                            only({"512:15:512:101", "512:15:512:102", "512:15:512:110", "1:512:8:512:202"}),
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 2, .roundCalls = 4}};
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});

  const HalvingState& h = scheduler.lastHalving();
  CHECK(h.active);
  CHECK_EQ(h.round, 0u);
  CHECK_EQ(h.budget, u64(4));
  std::vector<std::string> pool;
  for (size_t const i : h.pool) { pool.push_back(scheduler.baselines()[i].fft.spec()); }
  CHECK(pool == (std::vector<std::string>{"512:15:512:101", "1:512:8:512:202"}));

  // Its round is all there is, by rule, and only its members' steps.
  CHECK(!items.empty());
  for (const Item& item : items) {
    CHECK(item.halving && byRule(item));
    CHECK(std::ranges::find(h.pool, item.index) != h.pool.end());
  }
}

namespace {

// Calls of search on an entry: rows at a set other than the built-in defaults, dearer than its defaults so that its gap
// stays where it was.
void searched(Fixture& f, const char* spec, u32 calls, double mean = 1600) {
  for (u32 c = 0; c < calls; ++c) {
    CHECK(f.db.add(
      RunRow{.sess = f.sess,
             .fft = spec,
             .kind = TestKind::PRP,
             .exponent = 118'063'003,
             .regime = regimeOf(FFTConfig{spec}, 118'063'003),
             .cfg = f.db.internCfg({{"ZEROHACK_W", "0"}}),
             .m = {.mean = mean, .stddev = 0.1, .blocks = 4, .calls = 1, .drift = 1, .status = Status::Ok, .ts = 0}}));
  }
}

// Ranks the queue, and records the rounds that begins, as a run does before its next call; then where the halving
// stands, its pool by spec.
std::tuple<bool, u32, u64, std::vector<std::string>> halvingAfter(Fixture& f, const Scheduler& scheduler) {
  (void)scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  const HalvingState& h = scheduler.lastHalving();
  for (RoundRow row : h.unrecorded) {
    row.sess = f.sess;
    CHECK(f.db.add(row));
  }
  std::vector<std::string> pool;
  for (size_t const i : h.pool) { pool.push_back(scheduler.baselines()[i].fft.spec()); }
  return {h.active, h.round, h.budget, pool};
}

}  // namespace

TEST(each_round_keeps_the_faster_half_and_doubles_the_calls_until_one_is_left) {
  Fixture f;
  std::vector<const char*> const specs{"512:15:512:101", "512:15:512:102", "512:15:512:110", "512:15:512:111"};
  double cost = 1450;
  for (const char* spec : specs) { concludedAt(f, spec, {}, cost += 5); }
  Scheduler const scheduler{scope(),
                            only({"512:15:512:101", "512:15:512:102", "512:15:512:110", "512:15:512:111"}),
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 4, .roundCalls = 4}};

  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 0u, u64(4), std::vector<std::string>(specs.begin(), specs.end())));

  // Round 2: all four had their 4 calls; the two nearest the fastest go on, to 8 more each.
  for (const char* spec : specs) { searched(f, spec, 4); }
  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 1u, u64(8), std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));

  // One is short of its calls, and it alone is offered.
  searched(f, "512:15:512:101", 8);
  for (const Item& item : scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed})) {
    CHECK_EQ(scheduler.baselines()[item.index].fft.spec(), std::string{"512:15:512:102"});
  }

  // Both had theirs: one is left, and the search is ranked by value again, whatever it offers.
  searched(f, "512:15:512:102", 8);
  auto const [active, round, budget, pool] = halvingAfter(f, scheduler);
  CHECK(!active);
  CHECK(pool == std::vector<std::string>{"512:15:512:101"});
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(std::ranges::any_of(
    items, [&](const Item& i) { return scheduler.baselines()[i.index].fft.spec() == "512:15:512:111"; }));
  CHECK(std::ranges::none_of(items, [](const Item& i) { return i.halving; }));
}

TEST(a_round_keeps_its_entries_until_each_has_had_its_calls) {
  // 101 starts at 1450 and 102 at 1495, 3% behind.  101's first two calls find 15%: 102 is now 22% behind it, outside
  // the margin, but the round began with it and it is owed its calls before anything is decided.
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  concludedAt(f, "512:15:512:102", {}, 1495);
  Scheduler const scheduler{scope(), only({"512:15:512:101", "512:15:512:102"}), 1000,
                            {},      Strategy{.kind = Strategy::Kind::Single},   false,
                            false,   Halving{.contenders = 2, .roundCalls = 16}};

  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 0u, u64(16), std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));
  concludedAt(f, "512:15:512:101", {{"INPLACE", "1"}}, 1230);

  auto const [active, round, budget, pool] = halvingAfter(f, scheduler);
  CHECK(active);
  CHECK(pool == (std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(!items.empty() && scheduler.baselines()[items.front().index].fft.spec() == "512:15:512:102");

  // A later process reads the same round from the rows, whatever the gaps say now.
  f.newSession();
  Scheduler const again{scope(), only({"512:15:512:101", "512:15:512:102"}), 1000,
                        {},      Strategy{.kind = Strategy::Kind::Single},   false,
                        false,   Halving{.contenders = 2, .roundCalls = 16}};
  CHECK(std::get<3>(halvingAfter(f, again)) == (std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));

  // Once each has had its calls, the round is decided by the gaps as they stand then.
  searched(f, "512:15:512:101", 14);
  searched(f, "512:15:512:102", 16);
  auto const [over, last, calls, left] = halvingAfter(f, again);
  CHECK(!over);
  CHECK(left == std::vector<std::string>{"512:15:512:101"});
}

TEST(a_round_waiting_on_readings_taken_by_rule_is_still_under_way) {
  // 110, 17% behind under other options, is not swept; a round begins over 101 and 102.  A step of 110's then puts it
  // within the margin, and its reading at the defaults is taken first -- which leaves the round where it was.
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  concludedAt(f, "512:15:512:102", {}, 1460);
  concludedAt(f, "512:15:512:110", {{"INPLACE", "1"}}, 1700);
  Scheduler const scheduler{scope(),
                            only({"512:15:512:101", "512:15:512:102", "512:15:512:110"}),
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 4, .roundCalls = 4}};
  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 0u, u64(4), std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));

  concludedAt(f, "512:15:512:110", {{"WMUL", "1"}}, 1470);
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(!items.empty() && items.front().sweep);
  CHECK(scheduler.lastHalving().active);
  CHECK(scheduler.lastHalving().unrecorded.empty());
}

TEST(calls_made_before_a_round_do_not_count_towards_it) {
  // 101 alone is read at first, and has 40 calls at other sets -- a bootstrap's, say -- before 102 is read and a round
  // can begin.  Those do not count towards its calls in the round.
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  Scheduler const scheduler{scope(), only({"512:15:512:101", "512:15:512:102"}), 1000,
                            {},      Strategy{.kind = Strategy::Kind::Single},   false,
                            false,   Halving{.contenders = 2, .roundCalls = 16}};
  CHECK(!std::get<0>(halvingAfter(f, scheduler)));
  searched(f, "512:15:512:101", 40);
  concludedAt(f, "512:15:512:102", {}, 1460);

  auto const [active, round, budget, pool] = halvingAfter(f, scheduler);
  CHECK(active && round == 0 && budget == 16);
  CHECK(scheduler.lastHalving().calls == (std::vector<u64>{0, 0}));
}

TEST(an_entry_that_has_had_its_calls_finishes_the_step_it_began) {
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  concludedAt(f, "512:15:512:102", {}, 1460);
  Scheduler const scheduler{scope(), only({"512:15:512:101", "512:15:512:102"}), 1000,
                            {},      Strategy{.kind = Strategy::Kind::Single},   false,
                            false,   Halving{.contenders = 2, .roundCalls = 4}};
  (void)halvingAfter(f, scheduler);
  searched(f, "512:15:512:101", 4);
  searched(f, "512:15:512:102", 4);

  // 101's last call began a step one call short of concluding.
  CHECK(f.db.add(
    RunRow{.sess = f.sess,
           .fft = "512:15:512:101",
           .kind = TestKind::PRP,
           .exponent = 118'063'003,
           .regime = regimeOf(FFTConfig{"512:15:512:101"}, 118'063'003),
           .cfg = f.db.internCfg({{"INPLACE", "1"}}),
           .m = {.mean = 1440, .stddev = 0.1, .blocks = 4, .calls = 1, .drift = 1, .status = Status::Ok, .ts = 0}}));

  CHECK(std::get<0>(halvingAfter(f, scheduler)));
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK_EQ(items.size(), size_t{1});
  for (const Item& item : items) {
    CHECK(item.halving && item.calls == 1 && configText(item.options) == "INPLACE=1");
    CHECK_EQ(scheduler.baselines()[item.index].fft.spec(), std::string{"512:15:512:101"});
  }

  concludedAt(f, "512:15:512:101", {{"INPLACE", "1"}}, 1440);
  CHECK(!std::get<0>(halvingAfter(f, scheduler)));
}

TEST(a_database_with_no_rounds_takes_up_the_halving_where_the_calls_left_it) {
  // Four contenders under the rule before rounds were recorded: all past round 1's 4 calls, and 101 past round 2's
  // 4 + 8.  The halving goes on in round 2, 102 owed the rest of its 8, not begun again.
  Fixture f;
  std::vector<const char*> const specs{"512:15:512:101", "512:15:512:102", "512:15:512:110", "512:15:512:111"};
  double cost = 1450;
  for (const char* spec : specs) { concludedAt(f, spec, {}, cost += 5); }
  searched(f, "512:15:512:101", 12);
  searched(f, "512:15:512:102", 7);
  searched(f, "512:15:512:110", 4);
  searched(f, "512:15:512:111", 4);
  Scheduler const scheduler{scope(),
                            only({"512:15:512:101", "512:15:512:102", "512:15:512:110", "512:15:512:111"}),
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 4, .roundCalls = 4}};

  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 1u, u64(8), std::vector<std::string>{"512:15:512:101", "512:15:512:102"}));
  CHECK(scheduler.lastHalving().calls == (std::vector<u64>{8, 3}));

  // Over, it stays over; and those that took part are not newcomers.
  searched(f, "512:15:512:102", 5);
  CHECK(!std::get<0>(halvingAfter(f, scheduler)));
  CHECK(!std::get<0>(halvingAfter(f, scheduler)));
  CHECK(std::get<3>(halvingAfter(f, scheduler)) == std::vector<std::string>{"512:15:512:101"});
}

TEST(an_entry_that_comes_within_the_margin_later_is_halved_with_the_one_left) {
  // 110 is 17% behind at its defaults, and the first halving is over before a step of its search is found to put it
  // within the margin.
  Fixture f;
  concludedAt(f, "512:15:512:101", {}, 1450);
  concludedAt(f, "512:15:512:102", {}, 1460);
  concludedAt(f, "512:15:512:110", {}, 1700);
  Scheduler const scheduler{scope(),
                            only({"512:15:512:101", "512:15:512:102", "512:15:512:110"}),
                            1000,
                            {},
                            Strategy{.kind = Strategy::Kind::Single},
                            false,
                            false,
                            Halving{.contenders = 4, .roundCalls = 4}};
  (void)halvingAfter(f, scheduler);
  searched(f, "512:15:512:101", 4);
  searched(f, "512:15:512:102", 4);
  CHECK(!std::get<0>(halvingAfter(f, scheduler)));

  // Tuned to within the margin, it and the one left are halved.
  concludedAt(f, "512:15:512:110", {{"INPLACE", "1"}}, 1470);
  CHECK(halvingAfter(f, scheduler) ==
        std::tuple(true, 0u, u64(4), std::vector<std::string>{"512:15:512:101", "512:15:512:110"}));
}

TEST(the_search_is_spread_over_the_contenders_before_it_settles_on_one) {
  // The variants of 512:15:512 within a few percent of one another at their defaults, :101 3% further behind but 15%
  // faster in place (INPLACE=1, off by default here); every other move costs 1%.  Ranked by value alone the search
  // spends its calls on the variants cheapest at their defaults, since a single step of :101 would have to gain what
  // they are ahead by, and the move that pays is never reached.
  auto run = [](u32 contenders) {
    Fixture f;
    FakeBench bench{f.db, f.sess, false, 200};
    bench.optionFactor = [](const FFTConfig& fft, const UseConfig& options) {
      double factor = fft.variant == 101 ? 1.03 : 1;
      for (const auto& [key, value] : options) {
        factor *= fft.variant == 101 && key == "INPLACE" && value == "1" ? 0.85 : 1.01;
      }
      return factor;
    };
    Scheduler scheduler{scope(),
                        baselines(nvidia(), scope(), {FFTShape{"512:15:512"}}),
                        1000,
                        {},
                        Strategy{.kind = Strategy::Kind::Single},
                        false,
                        false,
                        Halving{.contenders = contenders, .roundCalls = 4}};
    (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});
    return std::ranges::count(bench.order, std::string{"512:15:512:101@118063003 INPLACE=1"});
  };

  CHECK(run(16) >= 2);
  CHECK_EQ(run(0), 0);
}

TEST(an_entry_names_every_key_a_line_would_set_once_its_own_keys_are_in_place) {
  // FFT3161 at width 512 offers L2_STRIPING up to 512/64 = 8 alone and 512/128 = 4 beside MULTI_Q=1, so under these
  // lines L2_STRIPING=8 is fitted away.  A set back at MULTI_Q=0 makes it legal again: the entry must name L2_STRIPING
  // at its own value, or production would run it at 8.
  FFTConfig const fft{"1:512:8:512:202"};
  Defaults const lines{.global = {{"L2_STRIPING", "8"}, {"MULTI_Q", "1"}}, .family = {}};

  CHECK_EQ(configText(besideLines(nvidia(), fft, TestKind::PRP, lines, {})), std::string{"L2_STRIPING=0,MULTI_Q=0"});

  // Beside MULTI_Q=1 the line is fitted away, and nothing needs naming but the set's own key.
  CHECK_EQ(configText(besideLines(nvidia(), fft, TestKind::PRP, lines, {{"MULTI_Q", "1"}})), std::string{"MULTI_Q=1"});
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
  std::vector<Baseline> two;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.variant == 211 || b.fft.variant == 212) { two.push_back(b); }
  }
  Scheduler scheduler{scope(), two, 1000, {}, Strategy{.kind = Strategy::Kind::Single}};

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

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
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

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
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

TEST(a_restart_is_offered_once_no_probe_is_left_and_once_each_period_of_sets_before) {
  Fixture plain;
  ProbeRun const descent = runProbed(plain);
  std::set<std::string> const stepped(descent.order.begin(), descent.order.end());
  CHECK(stepped.size() > RESTART_PERIOD);

  // Restarts never run dry, so the run is stopped four draws in.
  Fixture f;
  u32 const calls = u32(descent.order.size()) + 4 * MIN_CALLS;
  ProbeRun const run = runProbed(f, calls, {}, true);
  CHECK(run.report.stopped);
  CHECK_EQ(run.order.size(), size_t(calls));

  // Call for call the same descent around the jumps, which are the entry's own sequence in order, each measured as a
  // row needs, none of them cheaper than where the descent ended, so the best set stays put.
  std::vector<std::string> steps;
  std::vector<size_t> jumps;
  for (size_t i = 0; i < run.order.size(); ++i) {
    if (stepped.contains(run.order[i])) {
      steps.push_back(run.order[i]);
    } else {
      jumps.push_back(i);
    }
  }
  CHECK(steps == descent.order);
  CHECK_EQ(run.best, descent.best);
  CHECK_EQ(jumps.size(), size_t(4 * MIN_CALLS));
  if (jumps.size() != 4 * MIN_CALLS) { return; }

  FFTConfig const fft{PROBED};
  std::string const at = std::string{PROBED} + "@118063003 ";
  for (u32 k = 0; k < 4; ++k) {
    std::string const drawn = at + configText(restartOf(nvidia(), fft, std::string{PROBED} + " prp short32", k));
    for (u32 c = 0; c < MIN_CALLS; ++c) { CHECK_EQ(run.order[jumps[k * MIN_CALLS + c]], drawn); }
  }

  // The first while probes were still left, once the entry had measured a period of sets; the rest at the end.
  std::set<std::string> const before(run.order.begin(), run.order.begin() + ptrdiff_t(jumps.front()));
  CHECK_EQ(before.size(), size_t(RESTART_PERIOD));
  CHECK(jumps.front() < descent.order.size());
  CHECK_EQ(jumps[MIN_CALLS], descent.order.size() + MIN_CALLS);

  // Each jump taught the entry that its search is spent, and the device nothing.
  EntryKey const entry{PROBED, TestKind::PRP, "short32"};
  CHECK(near(run.gains.all().n(), descent.gains.all().n()));
  CHECK(run.gains.forEntry(entry).mean() < descent.gains.forEntry(entry).mean());
}

TEST(a_restart_waits_while_a_probe_is_left_until_a_period_of_sets_has_been_measured) {
  Fixture plain;
  u32 const descent = u32(runProbed(plain).order.size());

  auto offered = [](Fixture& f) {
    std::vector<ItemKind> out;
    for (const Item& item : probedScheduler(true).admissible(f.db, f.env, Objective{f.db, f.env, scope()})) {
      out.push_back(item.kind);
    }
    return out;
  };
  auto count = [](const std::vector<ItemKind>& kinds, ItemKind kind) { return std::ranges::count(kinds, kind); };

  // A few sets in, probes and no jump.
  Fixture f;
  (void)runProbed(f, 10);
  std::vector<ItemKind> const midway = offered(f);
  CHECK(count(midway, ItemKind::Probe) > 0);
  CHECK_EQ(count(midway, ItemKind::Restart), 0);

  // A period of sets in, the jump is due, and goes before the probes still left.
  f.newSession();
  (void)runProbed(f, 2 * RESTART_PERIOD - 10);
  std::vector<ItemKind> const due = offered(f);
  CHECK(count(due, ItemKind::Probe) > 0);
  CHECK_EQ(count(due, ItemKind::Restart), 1);
  CHECK(!due.empty() && due.front() == ItemKind::Restart);

  // The descent over, one jump and nothing else.
  f.newSession();
  (void)runProbed(f, descent - 2 * RESTART_PERIOD);
  std::vector<ItemKind> const spent = offered(f);
  CHECK_EQ(count(spent, ItemKind::Probe), 0);
  CHECK_EQ(count(spent, ItemKind::Restart), 1);
}

TEST(restarting_interrupted_draws_the_same_sequence) {
  Fixture plain;
  u32 const descent = u32(runProbed(plain).order.size());
  u32 const calls = descent + 3 * MIN_CALLS;

  Fixture whole;
  ProbeRun const all = runProbed(whole, calls, {}, true);

  // Stopped between a restart's two calls, and resumed by a later process.
  Fixture f;
  ProbeRun const first = runProbed(f, descent + MIN_CALLS + 1, {}, true);
  f.newSession();
  ProbeRun const second = runProbed(f, calls - (descent + MIN_CALLS + 1), {}, true);

  std::vector<std::string> joined = first.order;
  joined.insert(joined.end(), second.order.begin(), second.order.end());
  CHECK(joined == all.order);
}

namespace {

// Two pairs of FP64 variants, each pair 0.3% apart: 512:15:512 reaches ~142M, and 1K:8:1K serves what lies past it up
// to 160M.  Each pair's rows already hold two calls, whose intervals overlap -- one more call on each side separates
// them -- and neither pair is within RACE_MARGIN.
constexpr u64 TIE_LO = 110'000'000;
constexpr u64 TIE_HI = 160'000'000;

// 0 for a variant outside the pairs.
double tieCost(const FFTConfig& fft) {
  static const std::map<std::string, double> bySpec{
    {"512:15:512:201", 1700}, {"512:15:512:202", 1705.1}, {"1K:8:1K:202", 3000}, {"1K:8:1K:212", 3009}};
  auto const at = bySpec.find(fft.spec());
  return at != bySpec.end() ? at->second : 0;
}

// The order the refines are called in, as specs, with the probe exponent at `probe`.
std::vector<std::string> refineOrder(u64 probe) {
  RunScope const tied = makeScope(ScopeArgs{.lo = TIE_LO, .hi = TIE_HI, .probe = probe}, {});

  std::vector<Baseline> pairs;
  for (const Baseline& b : baselines(nvidia(), tied, {FFTShape{"512:15:512"}, FFTShape{"1K:8:1K"}})) {
    if (tieCost(b.fft) > 0) { pairs.push_back(b); }
  }
  CHECK_EQ(pairs.size(), size_t(4));

  Fixture f;
  for (const Baseline& b : pairs) {
    double const cost = tieCost(b.fft);
    CHECK(f.db.add(RunRow{.sess = f.sess,
                          .fft = b.fft.spec(),
                          .kind = b.kind,
                          .exponent = 118'063'003,
                          .regime = regimeOf(b.fft, 118'063'003),
                          .cfg = f.db.internCfg({}),
                          .m = {.mean = cost,
                                .stddev = cost * 0.001,
                                .blocks = 2 * BLOCKS_PER_CALL,
                                .calls = MIN_CALLS,
                                .drift = 1,
                                .status = Status::Ok,
                                .ts = 1}}));
  }

  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = [](const FFTConfig& fft, const UseConfig&) { return tieCost(fft) / pseudoCost(fft); };

  // A strategy with no step to offer an FP64 entry, so that nothing competes with the refines.
  Scheduler scheduler{tied, pairs, 1000, {}, Strategy{.kind = Strategy::Kind::Permute, .keys = {"MULTI_Q"}}};
  QueueReport const report = runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});
  CHECK(!report.stopped);

  // Dry because every contest is decided, not because anything gave up.
  std::vector<OptionSet> const sets = optionSetsFor(f.db, f.env);
  for (const Contest& c : contests(sets, Objective{f.db, f.env, tied}.points())) { CHECK(!undecided(c, sets)); }

  std::vector<std::string> out;
  for (const std::string& call : bench.order) { out.push_back(call.substr(0, call.find('@'))); }
  return out;
}

}  // namespace

TEST(a_near_tie_at_a_high_traffic_boundary_is_resolved_before_a_low_traffic_one) {
  // With the probe at 118M, half the workload's weight and every point below ~142M decide between the 512:15:512
  // pair; the 1K:8:1K pair decides only the twenty points above.  Each side of the heavy contest is called, which
  // settles it, before either side of the light one.  Within a pair the dearer side first: its error bar, being a
  // fraction of its cost, is the wider.
  std::vector<std::string> const probeLow{"512:15:512:202", "512:15:512:201", "1K:8:1K:212", "1K:8:1K:202"};
  CHECK(refineOrder(118'063'003) == probeLow);

  // The same rows with the probe at 150M, where only the 1K:8:1K pair is eligible: the traffic has moved, and the
  // order with it -- though that pair's calls take longer.
  std::vector<std::string> const probeHigh{"1K:8:1K:212", "1K:8:1K:202", "512:15:512:202", "512:15:512:201"};
  CHECK(refineOrder(150'000'001) == probeHigh);
}

TEST(a_restart_is_found_past_a_long_stretch_of_draws_that_cannot_run) {
  Fixture f;
  (void)runProbed(f);
  FFTConfig const fft{PROBED};
  std::string const entry = std::string{PROBED} + " prp short32";

  // Every value but the default of these keys will not build, which rules out nearly every draw.
  for (const char* key : {"TAIL_KERNELS", "TAIL_TRIGS", "SHUFL_BYTES_W", "SHUFL_BYTES_H", "WMUL", "INPLACE"}) {
    const Option* const option = findOption(key);
    int const fallback = option->defaultFor(nvidia(), fft, {});
    for (int const value : option->valuesFor(nvidia(), fft, {})) {
      if (value == fallback) { continue; }
      CHECK(f.db.add(NogoRow{.sess = f.sess, .fft = PROBED, .key = key, .val = std::to_string(value), .ts = 2}));
    }
  }
  u32 first = 0;
  while (f.db.isNogo(f.env, PROBED, restartOf(nvidia(), fft, entry, first))) { ++first; }

  // Far past the entry's measured sets and a stage's worth of draws beyond them, which is where a bounded walk from
  // the start of the sequence gave up.
  CHECK(first > 256);

  std::vector<Item> const items = probedScheduler(true).admissible(f.db, f.env, Objective{f.db, f.env, scope()});
  auto const restart = std::ranges::find_if(items, [](const Item& i) { return i.kind == ItemKind::Restart; });
  CHECK(restart != items.end() && restart->draw == first);
}

TEST(a_restart_is_known_by_its_declaration_not_by_the_workload) {
  Fixture plain;
  ProbeRun const descent = runProbed(plain);

  Fixture f;
  (void)runProbed(f, u32(descent.order.size()) + 4 * MIN_CALLS, {}, true);
  // Four measured, and a fifth declared before the call the stop cut short, which the next process resumes.
  CHECK_EQ(f.db.jumps().size(), size_t(5));

  // What the device learns is a function of the rows alone -- no workload, no baselines -- and the jump rows are what
  // keep the four restarts out of it.  Without them the same readings would count as four moves that found nothing.
  CHECK(near(gainsOf(f.db, f.env).all().n(), descent.gains.all().n()));

  std::string undeclared;
  for (const std::string& line : split(f.db.text(), '\n')) {
    if (!line.empty() && !line.starts_with("jump")) { undeclared += line + '\n'; }
  }
  TuneDB stripped;
  CHECK(stripped.parse(undeclared, "stripped"));
  CHECK(near(gainsOf(stripped, f.env).all().n(), descent.gains.all().n() + 4));
}

TEST(a_restart_space_is_spent_only_once_its_draws_keep_repeating) {
  auto nothingRuns = [](u32) { return false; };

  // Three configurations, none runnable: the scan stops, RESTART_REPEATS draws past the last new one, and stays
  // stopped.
  RestartScan small;
  CHECK(!nextRunnable(small, [](u32 k) { return std::to_string(k % 3); }, nothingRuns));
  CHECK(small.exhausted);
  CHECK_EQ(small.next, 3 + RESTART_REPEATS);
  CHECK(!nextRunnable(small, [](u32) { return std::string{"new"}; }, [](u32) { return true; }));

  // A thousand new configurations that cannot run are the space still being explored, not a spent one.
  RestartScan large;
  auto const found = nextRunnable(large, [](u32 k) { return std::to_string(k); }, [](u32 k) { return k == 999; });
  CHECK(found && *found == 999u);
  CHECK(!large.exhausted);

  // And the scan resumes from where it stood, not from the start.
  CHECK(nextRunnable(large, [](u32 k) { return std::to_string(k); }, [](u32 k) { return k >= 999; }) == 999u);
  CHECK(nextRunnable(large, [](u32 k) { return std::to_string(k); }, [](u32 k) { return k > 999; }) == 1000u);
}

namespace {

// Every key set away from its default costs 1%, as in interacting(), except at the values named.
double offDefault(const UseConfig& options, const UseConfig& planted) {
  double factor = 1;
  for (const auto& [key, value] : options) {
    if (auto const at = planted.find(key); at == planted.end() || at->second != value) { factor *= 1.01; }
  }
  return factor;
}

// SHUFL_BYTES_W=16 costs 1% at the default WMUL but saves 3% at WMUL=1, which alone costs 1%: the structural step
// loses, and only a search of its own branch finds what it opens.
double structuralPair(const FFTConfig&, const UseConfig& options) {
  UseConfig const planted{{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}};
  bool const wide = useValue(options, "SHUFL_BYTES_W", 8) == 16;
  bool const wmul1 = useValue(options, "WMUL", 2) == 1;
  double const pair = wide && wmul1 ? 0.97 : wide || wmul1 ? 1.01 : 1;
  return pair * offDefault(options, planted);
}

// TAIL_KERNELS=3 and ZEROHACK_H=0, in two groups that share the tail kernels, each cost a little alone -- less than
// anything else in their groups does -- and save 3% together.
double crossGroupPair(const FFTConfig&, const UseConfig& options) {
  UseConfig const planted{{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}};
  bool const tail = useValue(options, "TAIL_KERNELS", 2) == 3;
  bool const height = useValue(options, "ZEROHACK_H", 1) == 0;
  double const pair = tail && height ? 0.97 : tail ? 1.004 : height ? 1.002 : 1;
  return pair * offDefault(options, planted);
}

struct SearchRun {
  std::vector<std::string> order;
  std::string best;
};

SearchRun runSearched(const Strategy& strategy, double (*factor)(const FFTConfig&, const UseConfig&),
                      u32 stopAfter = ~0u) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false, stopAfter};
  bench.optionFactor = factor;

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  Scheduler scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false}, strategy};

  SearchRun out;
  QueueReport const report = runQueue(scheduler, f.db, f.env, bench, [&](const Objective&, const Defaults&) {
    for (const SelectionEntry& e : entriesFor(f.db, f.env, Gating::Assumed)) {
      if (e.fft == PROBED) { out.best = configText(canonicalConfig(nvidia(), FFTConfig{PROBED}, e.opts)); }
    }
  });
  CHECK(report.stopped == (stopAfter != ~0u));
  out.order = bench.order;
  return out;
}

// Enough calls for the first 152 probes of the entry and the first combination above them, a little over 300 calls
// in; searching it until nothing is left takes several hundred more.
constexpr u32 SEARCH_CALLS = 350;

// Groups over crossGroupPair, which two tests read.
SearchRun groupsSearched() {
  static SearchRun const run = runSearched({.kind = Strategy::Kind::Groups}, crossGroupPair, SEARCH_CALLS);
  return run;
}

}  // namespace

TEST(combo_tiers_1_schedules_exactly_what_groups_does) {
  SearchRun const groups = groupsSearched();
  SearchRun const one = runSearched({.kind = Strategy::Kind::Hybrid, .comboTiers = 1}, crossGroupPair, SEARCH_CALLS);
  CHECK_EQ(groups.order.size(), size_t(SEARCH_CALLS));
  CHECK(groups.order == one.order);
}

TEST(a_cross_group_pair_is_found_by_hybrid_and_not_by_groups) {
  SearchRun const groups = groupsSearched();
  SearchRun const hybrid = runSearched({.kind = Strategy::Kind::Hybrid}, crossGroupPair, SEARCH_CALLS);
  CHECK_EQ(groups.best, std::string{"-"});
  CHECK_EQ(hybrid.best, std::string{"TAIL_KERNELS=3,ZEROHACK_H=0"});

  // Found by the combination of the two groups' best answers, measured as any configuration is.  No stage of groups
  // moves two groups at once, so it never offers the pair.
  std::string const at = std::string{PROBED} + "@118063003";
  CHECK(std::ranges::count(hybrid.order, at + " TAIL_KERNELS=3,ZEROHACK_H=0") == MIN_CALLS);
  CHECK(std::ranges::count(groups.order, at + " TAIL_KERNELS=3,ZEROHACK_H=0") == 0);
}

TEST(a_structural_value_that_loses_its_step_is_searched_in_its_own_branch) {
  // Single steps from the best set alone, as before branches: SHUFL_BYTES_W=16 loses, so WMUL is never tried beside
  // it.  Groups searches that branch too, and WMUL depends on SHUFL_BYTES_W, so the readings at the default width
  // shuffle do not answer for it there.
  SearchRun const single = runSearched({.kind = Strategy::Kind::Single}, structuralPair);
  SearchRun const groups = runSearched({.kind = Strategy::Kind::Groups}, structuralPair, SEARCH_CALLS);
  CHECK_EQ(single.best, std::string{"-"});
  CHECK_EQ(groups.best, std::string{"SHUFL_BYTES_W=16,WMUL=1"});

  // A key that depends on nothing structural is answered in every branch by its one reading in the first, so no
  // configuration of another branch sets one.  Nothing is measured more than a row needs.
  auto const structural = {"INPLACE", "SHUFL_BYTES_W", "LDSPAD_W", "SHUFL_BYTES_H", "LDSPAD_H"};
  auto const free = {"TAIL_KERNELS", "TAIL_TRIGS", "LOADS",          "STORES",
                     "FAST_BARRIER", "OLD_FENCE",  "ENABLE_BARSYNC", "L2_STRIPING"};
  auto const has = [](const std::string& call, auto keys) {
    return std::ranges::any_of(keys, [&](const char* key) { return call.find(std::string{key} + "=") != call.npos; });
  };
  for (const auto& [call, n] : tally(groups.order)) {
    CHECK(n <= MIN_CALLS);
    CHECK(!(has(call, structural) && has(call, free)));

    // Only the best set steps into another branch, so every branch searched is one structural step from a best set:
    // the defaults, or SHUFL_BYTES_W=16 once it has won.  A step out of any other branch would be a second.
    auto const moved = std::ranges::count_if(
      structural, [&](const char* key) { return call.find(std::string{key} + "=") != call.npos; });
    CHECK(moved <= (call.find("SHUFL_BYTES_W=16") != call.npos ? 2 : 1));
  }
}

TEST(only_the_best_set_steps_into_another_branch) {
  // The defaults and SHUFL_BYTES_W=16, each read twice: two branches, the second searched from its own best set but
  // never stepped out of.
  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = structuralPair;
  FFTConfig const fft{PROBED};
  for (const UseConfig& options : {UseConfig{}, UseConfig{{"SHUFL_BYTES_W", "16"}}}) {
    for (u32 call = 0; call < MIN_CALLS; ++call) { (void)bench.run(fft, TestKind::PRP, 118'063'003, options, {}); }
  }

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  Scheduler const scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false},
                            Strategy{.kind = Strategy::Kind::Groups}};
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope()});

  // What an item sets, not branchOf(): WMUL=4 beside SHUFL_BYTES_W=16 fills the LDS, so LDSPAD_W no longer applies.
  // Each branch's items are worth what a move from its own best set is, and this one's is 1% dearer.
  u32 inBranch = 0;
  double const best = items.empty() ? 0 : items.front().value;
  for (const Item& item : items) {
    if (useValue(item.options, "SHUFL_BYTES_W", 8) != 16) {
      CHECK_EQ(item.value, best);
      continue;
    }
    ++inBranch;
    CHECK(item.value < best);
    for (const auto& [key, value] : item.options) { CHECK(key == "SHUFL_BYTES_W" || !findOption(key)->structural); }
  }

  // Its WMUL and LDSSWIZ_W probes, which depend on SHUFL_BYTES_W; and nothing else, the rest being answered.
  CHECK(inBranch > 0);
  CHECK(std::ranges::any_of(items, [](const Item& i) { return i.what == "Width WMUL=1" && i.options.size() == 2; }));
}

TEST(a_probe_answered_in_one_regime_is_still_owed_in_another) {
  // The same FFT in two regimes is two entries with the same best set.  WMUL=1 measured in one says nothing about the
  // other, however often admissible() is asked.
  RunScope const wide = makeScope(ScopeArgs{.lo = 70'000'000, .hi = 90'000'000, .probe = 80'000'023}, {});
  std::vector<Baseline> both;
  for (const Baseline& b : baselines(nvidia(), wide, {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { both.push_back(b); }
  }
  CHECK_EQ(both.size(), size_t(2));
  if (both.size() != 2) { return; }

  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = [](const FFTConfig&, const UseConfig& options) { return options.empty() ? 1.0 : 1.01; };
  for (const Baseline& b : both) {
    for (u32 call = 0; call < MIN_CALLS; ++call) { (void)bench.run(b.fft, b.kind, b.exponent, {}, {}); }
  }
  for (u32 call = 0; call < MIN_CALLS; ++call) {
    (void)bench.run(both[1].fft, both[1].kind, both[1].exponent, {{"WMUL", "1"}}, {});
  }

  Scheduler const scheduler{wide, both, 1000, Bootstrap{nvidia(), 80'000'023, {}, false},
                            Strategy{.kind = Strategy::Kind::Single}};
  Objective const objective{f.db, f.env, wide, Gating::Assumed};
  for (int round = 0; round < 2; ++round) {
    std::vector<Item> const items = scheduler.admissible(f.db, f.env, objective);
    auto owed = [&](size_t index) {
      return std::ranges::any_of(items, [&](const Item& i) { return i.index == index && i.what == "single WMUL=1"; });
    };
    CHECK(owed(0));
    CHECK(!owed(1));
  }
}

namespace {

// The calls of `order` that a combo row declares, as their places in it.
std::vector<size_t> comboCalls(const TuneDB& db, const std::vector<std::string>& order) {
  std::set<std::string> declared;
  for (const ComboRow& row : db.combos()) {
    if (const UseConfig* const opts = db.findCfg(row.cfg)) { declared.insert(row.fft + " " + configText(*opts)); }
  }
  std::vector<size_t> out;
  for (size_t i = 0; i < order.size(); ++i) {
    const std::string& call = order[i];
    size_t const space = call.find(' ');
    if (space == std::string::npos) { continue; }
    if (declared.contains(call.substr(0, call.find('@')) + call.substr(space))) { out.push_back(i); }
  }
  return out;
}

}  // namespace

TEST(a_combination_waits_for_the_tier_below_and_is_valued_by_the_combination_gains) {
  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = crossGroupPair;
  FFTConfig const fft{PROBED};
  auto measure = [&](const UseConfig& options) {
    for (u32 call = 0; call < MIN_CALLS; ++call) { (void)bench.run(fft, TestKind::PRP, 118'063'003, options, {}); }
  };
  measure({});
  measure({{"TAIL_KERNELS", "3"}});
  measure({{"ZEROHACK_H", "0"}});

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  CHECK_EQ(one.size(), size_t(1));
  if (one.size() != 1) { return; }
  Scheduler const scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false}, Strategy{}};
  auto kindsOf = [&](const std::vector<Item>& items, ItemKind kind) {
    return std::ranges::count_if(items, [&](const Item& i) { return i.kind == kind; });
  };

  // Tail and Height each have an answer to combine, but their groups' probes are not all answered yet.
  std::vector<Item> const early = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope()});
  CHECK(kindsOf(early, ItemKind::Probe) > 0);
  CHECK_EQ(kindsOf(early, ItemKind::Combo), 0);

  // Every step from the defaults answered, the structural ones among them, and none of them better than the defaults.
  // The branches those open have probes of their own left, which hold back only their own combinations.
  for (const Probe& p : probesOf(nvidia(), fft, {}, Strategy{}).probes) {
    if (p.tier == 1) { measure(p.config); }
  }

  Objective const objective{f.db, f.env, scope(), Gating::Assumed};
  std::vector<Item> const late = scheduler.admissible(f.db, f.env, objective);
  CHECK(kindsOf(late, ItemKind::Combo) > 0);

  // What the combinations the device has shown expect a combination from the entry's best set to save, which here --
  // with none shown -- is the prior's, and not what a move is expected to.
  const Baseline& b = one.front();
  EntryKey const key{PROBED, b.kind, b.band.regime.label()};
  double best = 0;
  for (const OptionSet& s : optionSetsFor(f.db, f.env)) {
    if (s.entry.fft == PROBED && (!best || s.entry.cost < best)) { best = s.entry.cost; }
  }
  GainModel const gains = gainsOf(f.db, f.env);
  double const combo = expectedSaving(objective.points(), b.kind, b.band, best, gains.comboForEntry(key));
  double const move = expectedSaving(objective.points(), b.kind, b.band, best, gains.forEntry(key));
  CHECK(combo > 0 && !near(combo, move));
  for (const Item& item : late) {
    if (item.kind == ItemKind::Combo) {
      CHECK(near(item.value, combo));
      CHECK(item.tier == 2 || item.tier == 3);
    }
  }
}

TEST(a_stage_listed_in_part_offers_its_next_points_once_its_first_are_answered) {
  // Both limits lifted, and every option set a little dearer than the defaults, so that the best set stays where it is
  // while its probes are answered.
  Fixture f;
  FakeBench bench{f.db, f.sess, false};
  bench.optionFactor = [](const FFTConfig&, const UseConfig& options) { return 1 + 0.01 * double(options.size()); };
  FFTConfig const fft{PROBED};
  auto measure = [&](const UseConfig& options) {
    for (u32 call = 0; call < MIN_CALLS; ++call) { (void)bench.run(fft, TestKind::PRP, 118'063'003, options, {}); }
  };
  measure({});

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
    if (b.fft.spec() == PROBED) { one.push_back(b); }
  }
  CHECK_EQ(one.size(), size_t(1));
  if (one.size() != 1) { return; }
  Strategy const lifted{.maxPermute = NO_LIMIT, .maxPoints = NO_LIMIT};
  Scheduler const scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false}, lifted};

  // The largest part of the list, and its points in the order it lists them whole.
  ProbeList const windowed = probesOf(nvidia(), fft, {}, lifted, {}, true, PROBE_WINDOW);
  CHECK(!windowed.unlisted.empty());
  if (windowed.unlisted.empty()) { return; }
  ProbeList::Unlisted const big = *std::ranges::max_element(windowed.unlisted, {}, &ProbeList::Unlisted::most);
  std::vector<std::string> whole;
  for (const Probe& p : probesOf(nvidia(), fft, {}, lifted).probes) {
    if (p.part == big.part) { whole.push_back(configText(p.config)); }
  }
  CHECK(whole.size() > 2 * PROBE_WINDOW);
  if (whole.size() <= 2 * PROBE_WINDOW) { return; }
  std::set<std::string> const inBig{whole.begin(), whole.end()};

  auto offered = [&](const std::vector<Item>& items) {
    std::vector<std::string> out;
    for (const Item& item : items) {
      if (item.kind == ItemKind::Probe && inBig.contains(configText(item.options))) {
        out.push_back(configText(item.options));
      }
    }
    return out;
  };
  auto combos = [](const std::vector<Item>& items) {
    return std::ranges::count_if(items, [](const Item& i) { return i.kind == ItemKind::Combo; });
  };

  // One window at first, the head of the whole part, its first item saying how much of the part is left.
  std::vector<Item> const first = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope()});
  CHECK(offered(first) == std::vector<std::string>(whole.begin(), whole.begin() + PROBE_WINDOW));
  auto const head = std::ranges::find_if(first, [&](const Item& i) { return configText(i.options) == whole.front(); });
  CHECK(head != first.end());
  if (head != first.end()) { CHECK(head->unlisted >= whole.size() - PROBE_WINDOW); }

  // Every probe offered answered: the part's next points follow, and the combinations of what was read wait for them.
  for (const Item& item : first) {
    if (item.kind == ItemKind::Probe) { measure(item.options); }
  }
  std::vector<Item> const second = scheduler.admissible(f.db, f.env, Objective{f.db, f.env, scope(), Gating::Assumed});
  CHECK(offered(second) == std::vector<std::string>(whole.begin() + PROBE_WINDOW, whole.begin() + 2 * PROBE_WINDOW));
  CHECK_EQ(combos(second), 0);
}

TEST(choosing_the_next_item_with_both_limits_lifted_costs_what_is_listed) {
  // The run that stalled: an FFT6431 entry measured on a P100 under CUDA, searched with both limits lifted.  Its Cuda
  // group was 2 x 4 x 14^7 points before its register caps were searched a key at a time; Memory's 4499 are the most
  // any stage has now.
  Env const p100{.isNvidia = true, .cudaBackend = true, .computeCapability = 600, .pdlLaunch = true};
  TuneDB db;
  u32 const env = db.internEnv(dbEnvOf(p100));
  u32 const sess = db.beginSession(env, "51:1K:4:256:212@67513549", 0, 1'790'452'892);
  RunScope const workload = makeScope(ScopeArgs{.lo = 67'000'000, .hi = 80'000'000, .probe = 67'513'549}, {});

  std::vector<Baseline> one;
  for (const Baseline& b : baselines(p100, workload, {FFTShape{"51:1K:4:256"}})) {
    if (b.fft.spec() == "51:1K:4:256:212") { one.push_back(b); }
  }
  CHECK_EQ(one.size(), size_t(1));
  if (one.size() != 1) { return; }
  FFTConfig const fft = one.front().fft;
  CHECK(db.add(RunRow{.sess = sess,
                      .fft = fft.spec(),
                      .kind = TestKind::PRP,
                      .exponent = 67'513'549,
                      .regime = regimeOf(fft, 67'513'549),
                      .cfg = db.internCfg({}),
                      .m = {.mean = 652.604,
                            .stddev = 0.141,
                            .blocks = MIN_CALLS * BLOCKS_PER_CALL,
                            .calls = MIN_CALLS,
                            .drift = 1,
                            .status = Status::Ok,
                            .ts = 1'790'452'979}}));

  Scheduler const scheduler{workload, one, 1000, Bootstrap{p100, 67'513'549, {}, false},
                            Strategy{.maxPermute = NO_LIMIT, .maxPoints = NO_LIMIT}};
  std::vector<Item> const items = scheduler.admissible(db, env, Objective{db, env, workload, Gating::Assumed});

  // A window of each part, and the rest of the space said rather than listed.
  u64 unlisted = 0;
  for (const Item& item : items) { unlisted += item.unlisted; }
  CHECK(unlisted >= 4499 - PROBE_WINDOW);
  CHECK(unlisted < 10'000);
  CHECK(!items.empty());
  CHECK(items.size() < 2000);
}

TEST(each_combination_is_declared_once_before_its_first_call) {
  // A whole search, and the same search stopped between the two calls of its first combination and carried on by a
  // later process: the second resumes it without declaring it again.
  Fixture whole;
  FakeBench all{whole.db, whole.sess, false, SEARCH_CALLS + 60};
  all.optionFactor = crossGroupPair;
  auto searchOne = [] {
    std::vector<Baseline> one;
    for (const Baseline& b : baselines(nvidia(), scope(), {FFTShape{"512:15:512"}})) {
      if (b.fft.spec() == PROBED) { one.push_back(b); }
    }
    return Scheduler{scope(), one, 1000, Bootstrap{nvidia(), 118'063'003, {}, false}, Strategy{}};
  };
  Scheduler s0 = searchOne();
  (void)runQueue(s0, whole.db, whole.env, all, [](const Objective&, const Defaults&) {});

  std::vector<size_t> const combos = comboCalls(whole.db, all.order);
  CHECK(!combos.empty());
  if (combos.empty()) { return; }
  CHECK(std::ranges::all_of(whole.db.combos(), [](const ComboRow& r) { return r.tier == 2 || r.tier == 3; }));

  // One declaration per combination, and each measured as any configuration is.
  std::set<u32> cfgs;
  for (const ComboRow& r : whole.db.combos()) { CHECK(cfgs.insert(r.cfg).second); }
  CHECK(std::ranges::any_of(whole.db.combos(), [&](const ComboRow& r) {
    return r.tier == 2 && configText(*whole.db.findCfg(r.cfg)) == "TAIL_KERNELS=3,ZEROHACK_H=0";
  }));

  Fixture f;
  u32 const stop = u32(combos.front()) + 1;
  FakeBench first{f.db, f.sess, false, stop};
  first.optionFactor = crossGroupPair;
  Scheduler s1 = searchOne();
  (void)runQueue(s1, f.db, f.env, first, [](const Objective&, const Defaults&) {});
  CHECK_EQ(f.db.combos().size(), size_t(1));

  f.newSession();
  FakeBench second{f.db, f.sess, false, SEARCH_CALLS + 60 - stop};
  second.optionFactor = crossGroupPair;
  Scheduler s2 = searchOne();
  (void)runQueue(s2, f.db, f.env, second, [](const Objective&, const Defaults&) {});

  std::vector<std::string> split = first.order;
  split.insert(split.end(), second.order.begin(), second.order.end());
  CHECK(tally(split) == tally(all.order));
  CHECK_EQ(f.db.combos().size(), whole.db.combos().size());

  // Each concluded combination is one observation of the combination gains, and none of the move gains.
  GainModel const gains = gainsOf(f.db, f.env);
  GainModel const plain = gainsOf(whole.db, whole.env);
  CHECK(gains.combos().n() > 0 && gains.combos().n() <= double(f.db.combos().size()));
  CHECK(near(gains.combos().n(), plain.combos().n()));
  CHECK(near(gains.all().n(), plain.all().n()));
}

TEST(a_bootstrap_interrupted_among_its_combinations_carries_on_from_its_rows) {
  Fixture whole;
  BootstrapRun const all = runBootstrapped(whole);
  std::vector<size_t> const combos = comboCalls(whole.db, all.order);
  CHECK(!combos.empty());
  if (combos.empty()) { return; }

  // Every bootstrap combination is called before the first baseline, as every bootstrap call is.
  auto const firstBaseline = std::ranges::find_if(all.order, [](const std::string& s) { return !isFamilyCall(s); });
  CHECK(combos.back() < size_t(firstBaseline - all.order.begin()));

  Fixture f;
  BootstrapRun const first = runBootstrapped(f, u32(combos.front()) + 1);
  CHECK(first.report.stopped);
  f.newSession();
  BootstrapRun const second = runBootstrapped(f);
  CHECK(!second.report.stopped);

  std::map<std::string, u32> split = tally(first.order);
  for (const auto& [call, n] : tally(second.order)) { split[call] += n; }
  CHECK(split == tally(all.order));
  CHECK_EQ(configText(second.defaults.global), configText(all.defaults.global));
  CHECK_EQ(f.db.combos().size(), whole.db.combos().size());
}

namespace {

// Whether every entry `objective` publishes is one the accuracy gate has passed, up to where it is published.
bool allGated(const TuneDB& db, u32 env, const Objective& objective) {
  Gates const gates{db, env, nvidia()};
  return std::ranges::all_of(objective.entries(), [&](const SelectionEntry& e) {
    FFTConfig const fft{e.fft};
    GateVerdict const v = gates(fft, interval(fft, e.emin), e.opts);
    return v.state == GateState::Passed && e.reach <= v.reach;
  });
}

// A row concluding one baseline at the built-in defaults, as two calls leave it.
void conclude(Fixture& f, const std::string& spec, const UseConfig& opts, double mean) {
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = spec,
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .regime = regimeOf(FFTConfig{spec}, 118'063'003),
                        .cfg = f.db.internCfg(opts),
                        .m = {.mean = mean,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));
}

}  // namespace

TEST(nothing_is_published_that_the_gate_has_not_passed) {
  // 1K:8:1K:112 is the cheapest 1K variant, and reads below the FP64 floor at the top of its interval and everywhere
  // under it.
  Fixture f;
  FakeBench bench{f.db, f.sess};
  bench.zOf = [](const FFTConfig& fft, const UseConfig&, u64) { return fft.spec() == "1K:8:1K:112" ? 17.0 : 24.0; };

  u32 publications = 0;
  std::set<std::string> ever;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  (void)runQueue(scheduler, f.db, f.env, bench, [&](const Objective& objective, const Defaults&) {
    ++publications;
    CHECK(allGated(f.db, f.env, objective));
    for (const SelectionEntry& e : objective.entries()) { ever.insert(e.fft); }
  });
  CHECK(publications > 1);

  // It was read at the top, a reach was derived for it that no reading bore out, and it was never published; the next
  // cheapest 1K variant was read and published in its place.
  CHECK_EQ(std::ranges::count(bench.order, std::string{"gate 1K:8:1K:112@167772107"}), 1);
  CHECK_EQ(std::ranges::count_if(bench.order, [](const std::string& s) { return s.starts_with("gate 1K:8:1K:112@"); }),
           long(MAX_DERIVE_READINGS));
  CHECK(!ever.contains("1K:8:1K:112"));
  std::vector<SelectionEntry> const last = entriesFor(f.db, f.env);
  CHECK(std::ranges::any_of(last, [](const SelectionEntry& e) { return e.fft.starts_with("1K:8:1K:"); }));

  // The exact-arithmetic hybrid was never read at all.
  CHECK(std::ranges::none_of(bench.order, [](const std::string& s) { return s.starts_with("gate 1:512:8:512"); }));
}

TEST(a_set_short_of_the_floor_at_its_top_is_published_up_to_the_reach_derived_for_it) {
  // 1K:8:1K:112 again, now reading 17 at the top of its interval and a unit more for every 0.012 bits per word under
  // it: its reach is derived where it reads 28, by readings the queue takes by rule before it publishes the entry.
  Fixture f;
  FakeBench bench{f.db, f.sess};
  FFTConfig const cheapest{"1K:8:1K:112"};
  u64 const top = 167'772'107;
  auto const zOf = [=](u64 E) { return 17.0 + (double(top) - double(E)) / double(cheapest.size()) / 0.012; };
  bench.zOf = [&](const FFTConfig& fft, const UseConfig&, u64 E) {
    return fft.spec() == cheapest.spec() ? zOf(E) : 24.0;
  };

  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  (void)runQueue(scheduler, f.db, f.env, bench,
                 [&](const Objective& objective, const Defaults&) { CHECK(allGated(f.db, f.env, objective)); });

  u32 const readings =
    u32(std::ranges::count_if(bench.order, [](const std::string& s) { return s.starts_with("gate 1K:8:1K:112@"); }));
  CHECK(readings >= 2 && readings <= 5);

  std::vector<SelectionEntry> const last = entriesFor(f.db, f.env);
  auto const entry = std::ranges::find(last, cheapest.spec(), &SelectionEntry::fft);
  CHECK(entry != last.end());
  if (entry == last.end()) { return; }
  CHECK(entry->evidence == Evidence::Confirmed);
  CHECK(entry->reach < top);
  CHECK(zOf(entry->reach) >= 28);
  CHECK(double(entry->reach) / cheapest.size() > double(top) / cheapest.size() - (28 - 17) * 0.012 - REACH_GUARD_BPW);

  // Above it, another 1K variant, which reaches the table's top, serves what it cannot.
  CHECK(std::ranges::any_of(last, [&](const SelectionEntry& e) {
    return e.fft.starts_with("1K:8:1K:") && e.fft != cheapest.spec() && e.reach > entry->reach;
  }));
}

TEST(a_reading_the_gate_owes_is_taken_before_anything_valued) {
  Fixture f;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  conclude(f, "1K:8:1K:112", {}, 2900);

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  std::vector<Item> const owed = scheduler.admissible(f.db, f.env, objective);
  CHECK_EQ(owed.size(), size_t{1});
  CHECK(owed.at(0).kind == ItemKind::Gate);
  CHECK_EQ(owed.at(0).exponent, u64(167'772'107));
  CHECK(owed.at(0).options.empty() && owed.at(0).subject.empty());

  // Answered, the baselines are what is left.
  CHECK(f.db.add(RoeRow{.sess = f.sess,
                        .fft = "1K:8:1K:112",
                        .exponent = 167'772'107,
                        .cfg = f.db.internCfg({}),
                        .z = 24,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = true,
                        .ts = 1}));
  std::vector<Item> const next = scheduler.admissible(f.db, f.env, objective);
  CHECK(!next.empty());
  CHECK(std::ranges::none_of(next, [](const Item& i) { return i.kind == ItemKind::Gate; }));

  // Without the gate there is never a reading to take.
  Fixture g;
  Scheduler ungated{scope(), baselines(nvidia(), scope(), shapes())};
  conclude(g, "1K:8:1K:112", {}, 2900);
  CHECK(std::ranges::none_of(ungated.admissible(g.db, g.env, objective),
                             [](const Item& i) { return i.kind == ItemKind::Gate; }));
}

namespace {

// A workload 512:15:512 serves only the bottom of, the probe and so half the weight included: past 143413741, the top
// of its table, something larger has to be measured, and nothing about it is worth a call by value.
RunScope straddling() { return makeScope(ScopeArgs{.lo = 140'000'000, .hi = 150'000'000, .probe = 141'000'000}, {}); }

bool covered(const Objective& objective) {
  return std::ranges::all_of(objective.points(),
                             [](const ObjectivePoint& p) { return p.weight <= 0 || !p.cost || p.cost->measured(); });
}

}  // namespace

TEST(every_exponent_the_workload_weighs_is_covered_before_anything_valued) {
  Fixture f;
  FakeBench bench{f.db, f.sess};
  Scheduler scheduler{straddling(), baselines(nvidia(), straddling(), shapes()), 1000, {}, {}, false, true};

  // A stop fraction no measurement could clear: the run is left with only what runs by rule.
  QueueReport const report = runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, 0.5);
  CHECK(report.end == QueueEnd::BelowStop);
  CHECK(covered(Objective{f.db, f.env, scheduler.scope(), {}}));

  // Each shape it measured, it measured first at the variant production's shape scan would run.
  std::set<std::string> seen;
  for (const std::string& call : bench.order) {
    if (call == "anchor" || call.starts_with("gate ")) { continue; }
    FFTConfig const fft{call.substr(0, call.find('@'))};
    if (seen.insert(fft.shape.spec()).second) { CHECK_EQ(fft.variant, defaultVariant(fft.shape)); }
  }
  CHECK(seen.size() >= 2);

  // Nothing it left is by rule, and nothing valued was worth the stop fraction.
  CHECK(!report.left.empty());
  for (const Item& item : report.left) {
    CHECK(!byRule(item));
    CHECK(item.value < report.floor);
  }
}

TEST(a_gap_is_offered_only_to_the_baselines_that_can_fill_it) {
  Fixture f;
  Scheduler scheduler{straddling(), baselines(nvidia(), straddling(), shapes()), 1000, {}, {}, false, true};
  FFTConfig const bottom{"512:15:512:212"};
  u64 const top = 143'413'741;
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = bottom.spec(),
                        .kind = TestKind::PRP,
                        .exponent = scheduler.scope().probe,
                        .regime = regimeOf(bottom, scheduler.scope().probe),
                        .cfg = f.db.internCfg({}),
                        .m = {.mean = 1700,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));
  CHECK(f.db.add(RoeRow{.sess = f.sess,
                        .fft = bottom.spec(),
                        .exponent = top,
                        .cfg = f.db.internCfg({}),
                        .z = 24,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = true,
                        .ts = 1}));

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  CHECK(!covered(objective));
  std::vector<Item> const offered = scheduler.admissible(f.db, f.env, objective);
  CHECK(!offered.empty());
  for (const Item& item : offered) {
    const Baseline& b = scheduler.baselines()[item.index];
    CHECK(item.kind == ItemKind::Baseline && item.cover);
    CHECK(b.band.hi > top);
    CHECK(worthRunning(item, 1e9));
    CHECK(item.what.starts_with("covering "));
  }
  const FFTConfig& first = scheduler.baselines()[offered.front().index].fft;
  CHECK_EQ(first.variant, defaultVariant(first.shape));

  // Stopped here, the summary says what the rest is waiting on, and that what is left runs by rule.
  QueueReport report;
  report.left = offered;
  RunSummary const s = summarize(scheduler, f.db, f.env, f.sess, report, STOP);
  CHECK_EQ(s.heldBy, std::string{"the workload being covered"});
  CHECK(std::ranges::any_of(s.remaining,
                            [](const RunSummary::Remaining& r) { return r.kind == ItemKind::Baseline && r.byRule; }));
}

TEST(a_set_that_spends_accuracy_has_its_defaults_read_as_well) {
  // Any 1K variant on which a middle chain length can move away from its default.
  Env const env = nvidia();
  std::optional<Baseline> entry;
  UseConfig moved;
  for (const Baseline& b : baselines(env, scope(), shapes())) {
    for (const char* key : {"MM_CHAIN", "MM2_CHAIN"}) {
      const Option& option = *findOption(key);
      for (int v : option.valuesFor(env, b.fft, {})) {
        if (!entry && b.fft.shape.spec() == "1K:8:1K" && v != option.defaultFor(env, b.fft, {})) {
          entry = b;
          moved = {{key, std::to_string(v)}};
        }
      }
    }
  }
  CHECK(entry.has_value());
  if (!entry) { return; }
  CHECK(movesAccuracy(env, entry->fft, moved));
  std::string const spec = entry->fft.spec();
  u64 const top = gateExponent(interval(entry->fft, 118'063'003));

  Fixture f;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  conclude(f, spec, moved, 2900);
  CHECK(f.db.add(RoeRow{.sess = f.sess,
                        .fft = spec,
                        .exponent = top,
                        .cfg = f.db.internCfg(moved),
                        .z = 23,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = true,
                        .ts = 1}));

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  std::vector<Item> const owed = scheduler.admissible(f.db, f.env, objective);
  CHECK_EQ(owed.size(), size_t{1});
  if (owed.empty()) { return; }
  CHECK(owed.at(0).kind == ItemKind::Gate);
  CHECK_EQ(owed.at(0).exponent, top);
  CHECK(owed.at(0).subject == moved);
  CHECK(!movesAccuracy(env, entry->fft, owed.at(0).options));
  CHECK(canonicalConfig(env, entry->fft, owed.at(0).options).empty());
}


namespace {

// 1K:8:1K:112, whose 64-bit carry runs on past its table reach, and a workload that goes on past that reach too.
FFTConfig raisable() { return FFTConfig{"1K:8:1K:112"}; }
u64 raisableTop() { return gateExponent(interval(raisable(), maxExp(raisable()))); }

RunScope pastTheTop(u64 past) {
  u64 const top = raisableTop();
  return makeScope(ScopeArgs{.lo = top - 10'000'000, .hi = top + past, .probe = top - 1'000'000}, {});
}

// z falling by one for every 0.012 bits per word above the table's top, where it reads 32.
double raisableZ(u64 E) { return 32.0 - (double(E) - double(raisableTop())) / double(raisable().size()) / 0.012; }

void readAt(Fixture& f, const FFTConfig& fft, u64 exponent, double z, u64 ts) {
  CHECK(f.db.add(RoeRow{.sess = f.sess,
                        .fft = fft.spec(),
                        .exponent = exponent,
                        .cfg = f.db.internCfg({}),
                        .z = z,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = true,
                        .ts = ts}));
}

}  // namespace

TEST(a_reach_item_is_worth_what_the_set_would_save_between_its_reach_and_the_reading) {
  FFTConfig const fft = raisable();
  u64 const top = raisableTop();
  Interval const span = interval(fft, top);
  CHECK_EQ(span.hi, maxExp(fft));
  CHECK(span.regime.carry64);

  // Past the table's top, a larger FFT that costs more serves the workload.
  FFTConfig const larger{"1K:9:1K:202"};

  for (u64 const past : {u64(10'000'000), u64(0)}) {
    RunScope const scope = pastTheTop(past);
    Fixture f;
    Scheduler scheduler{scope, baselines(nvidia(), scope, {fft.shape}), 1000, {}, {}, false, true};
    for (const auto& [c, mean] : {std::pair{fft, 2900.0}, std::pair{larger, 3300.0}}) {
      CHECK(f.db.add(RunRow{.sess = f.sess,
                            .fft = c.spec(),
                            .kind = TestKind::PRP,
                            .exponent = scope.probe,
                            .regime = regimeOf(c, scope.probe),
                            .cfg = f.db.internCfg({}),
                            .m = {.mean = mean,
                                  .stddev = 0.1,
                                  .blocks = 4 * MIN_CALLS,
                                  .calls = MIN_CALLS,
                                  .drift = 1,
                                  .status = Status::Ok,
                                  .ts = 0}}));
    }
    readAt(f, fft, top, 32, 1);
    readAt(f, larger, gateExponent(interval(larger, scope.probe)), 24, 2);

    Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
    std::vector<Item> const items = scheduler.admissible(f.db, f.env, objective);
    auto const reach = std::ranges::find(items, ItemKind::Reach, &Item::kind);

    // A workload that ends at the table's top has nothing for a raise to save.
    if (!past) {
      CHECK(reach == items.end());
      continue;
    }

    CHECK(reach != items.end());
    if (reach == items.end()) { continue; }
    CHECK(reach->exponent > span.hi);
    CHECK(reach->options.empty() && reach->subject.empty());
    CHECK(reach->span == span);

    std::vector<OptionSet> const sets = optionSetsFor(f.db, f.env);
    auto const set = std::ranges::find_if(sets, [&](const OptionSet& o) { return o.entry.fft == fft.spec(); });
    CHECK(set != sets.end());
    if (set == sets.end()) { continue; }
    double const expected = saving(objective.points(), TestKind::PRP,
                                   {.lo = span.hi + 1, .hi = reach->exponent, .regime = span.regime}, set->entry.cost);
    CHECK(expected > 0);
    CHECK(near(reach->value, expected));
    CHECK(reach->seconds > 0);

    // Once the reading there is the standard's, the reach is raised to it and no item is left.
    readAt(f, fft, reach->exponent, 28.5, 3);
    std::vector<SelectionEntry> const published = entriesFor(f.db, f.env);
    auto const entry = std::ranges::find(published, fft.spec(), &SelectionEntry::fft);
    CHECK(entry != published.end());
    if (entry == published.end()) { continue; }
    CHECK_EQ(entry->reach, reach->exponent);
    CHECK(entry->evidence == Evidence::Confirmed);
    Objective const after{f.db, f.env, scheduler.scope(), Gating::Assumed};
    CHECK(after.T() < objective.T());
    CHECK(std::ranges::none_of(scheduler.admissible(f.db, f.env, after),
                               [](const Item& i) { return i.kind == ItemKind::Reach; }));
  }
}

namespace {

// A concluded row of `fft` at `exponent`, in `kind`, at the built-in defaults.
void concludeAt(Fixture& f, const FFTConfig& fft, TestKind kind, u64 exponent, double mean) {
  CHECK(f.db.add(RunRow{.sess = f.sess,
                        .fft = fft.spec(),
                        .kind = kind,
                        .exponent = exponent,
                        .regime = regimeOf(fft, exponent),
                        .cfg = f.db.internCfg({}),
                        .m = {.mean = mean,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));
}

}  // namespace

TEST(a_raise_is_offered_for_a_workload_wholly_past_the_band_that_holds_its_set) {
  // Measured and read by an earlier run over a wider workload; this one weighs only what the raise would serve, so no
  // baseline holds the set.
  FFTConfig const fft = raisable();
  u64 const top = raisableTop();
  Interval const span = interval(fft, top);
  FFTConfig const larger{"1K:9:1K:202"};
  RunScope const scope = makeScope(ScopeArgs{.lo = span.hi + 1, .hi = span.hi + 2'000'000, .probe = 0}, {});

  Scheduler scheduler{scope, baselines(nvidia(), scope, {fft.shape}), 1000, {}, {}, false, true};
  CHECK(std::ranges::none_of(scheduler.baselines(), [&](const Baseline& b) { return b.fft.spec() == fft.spec(); }));

  Fixture f;
  concludeAt(f, fft, TestKind::PRP, top - 1'000'000, 2900);
  concludeAt(f, larger, TestKind::PRP, span.hi + 1'000'000, 3300);
  readAt(f, fft, top, 32, 1);
  readAt(f, larger, gateExponent(interval(larger, span.hi + 1'000'000)), 24, 2);

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, objective);
  auto const reach = std::ranges::find(items, ItemKind::Reach, &Item::kind);
  CHECK(reach != items.end());
  if (reach == items.end()) { return; }
  CHECK(reach->fft.has_value() && reach->fft->spec() == fft.spec());
  CHECK(reach->exponent > span.hi);
  CHECK(reach->value > 0);

  // And the queue takes it, with no baseline to say what it reads.
  FakeBench bench{f.db, f.sess, false};
  bench.zOf = [&](const FFTConfig& c, const UseConfig&, u64 E) { return c.spec() == fft.spec() ? raisableZ(E) : 24.0; };
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});
  CHECK(std::ranges::count(bench.order, "gate " + fft.spec() + "@" + std::to_string(reach->exponent)) == 1);
  std::vector<SelectionEntry> const last = entriesFor(f.db, f.env);
  CHECK(std::ranges::any_of(last, [&](const SelectionEntry& e) { return e.fft == fft.spec() && e.reach > span.hi; }));
}

TEST(one_reading_that_raises_several_kinds_is_worth_what_it_saves_in_each) {
  FFTConfig const fft = raisable();
  u64 const top = raisableTop();
  Interval const span = interval(fft, top);
  FFTConfig const larger{"1K:9:1K:202"};
  RunScope const scope = makeScope(
    ScopeArgs{.lo = top - 10'000'000, .hi = top + 10'000'000, .probe = 0, .kinds = {TestKind::PRP, TestKind::LL}}, {});
  Scheduler scheduler{scope, baselines(nvidia(), scope, {fft.shape}), 1000, {}, {}, false, true};

  Fixture f;
  for (TestKind const kind : {TestKind::PRP, TestKind::LL}) {
    concludeAt(f, fft, kind, top - 1'000'000, 2900);
    concludeAt(f, larger, kind, top - 1'000'000, 3300);
  }
  readAt(f, fft, top, 32, 1);
  readAt(f, larger, gateExponent(interval(larger, top - 1'000'000)), 24, 2);

  Objective const objective{f.db, f.env, scheduler.scope(), Gating::Assumed};
  std::vector<Item> const items = scheduler.admissible(f.db, f.env, objective);
  CHECK_EQ(std::ranges::count(items, ItemKind::Reach, &Item::kind), 1);
  auto const reach = std::ranges::find(items, ItemKind::Reach, &Item::kind);
  if (reach == items.end()) { return; }

  double each[2]{};
  for (const OptionSet& set : optionSetsFor(f.db, f.env)) {
    if (set.entry.fft != fft.spec() || set.entry.regime != span.regime) { continue; }
    Interval const beyond{.lo = span.hi + 1, .hi = reach->exponent, .regime = span.regime};
    each[set.entry.kind == TestKind::LL] = saving(objective.points(), set.entry.kind, beyond, set.entry.cost);
  }
  CHECK(each[0] > 0 && each[1] > 0);
  CHECK(near(reach->value, each[0] + each[1]));

  // Worth running where the fraction of T a run stops at lies between either kind's saving and their sum.
  double const floor = (std::max(each[0], each[1]) + each[0] + each[1]) / 2;
  CHECK(worthRunning(*reach, floor));
}

TEST(a_set_better_than_the_standard_at_the_top_is_raised_by_the_queue_where_the_workload_goes_on) {
  Fixture f;
  FakeBench bench{f.db, f.sess};
  FFTConfig const fft = raisable();
  u64 const top = raisableTop();
  bench.zOf = [&](const FFTConfig& c, const UseConfig&, u64 E) { return c.spec() == fft.spec() ? raisableZ(E) : 24.0; };

  RunScope const scope = pastTheTop(10'000'000);
  Scheduler scheduler{scope, baselines(nvidia(), scope, shapes()), 1000, {}, {}, false, true};
  (void)runQueue(scheduler, f.db, f.env, bench,
                 [&](const Objective& objective, const Defaults&) { CHECK(allGated(f.db, f.env, objective)); });

  // In the band the table ends in, read at the top by rule, then above it by value, until a reading confirmed the
  // raise.
  std::string const prefix = "gate " + fft.spec() + "@";
  std::vector<u64> read;
  for (const std::string& s : bench.order) {
    if (!s.starts_with(prefix)) { continue; }
    if (u64 const E = std::stoull(s.substr(prefix.size())); E >= interval(fft, top).lo) { read.push_back(E); }
  }
  CHECK(!read.empty() && read.front() == top);
  CHECK(read.size() >= 2 && read.size() <= 4);
  CHECK(std::ranges::all_of(read, [&](u64 E) { return E == top || E > maxExp(fft); }));

  std::vector<SelectionEntry> const last = entriesFor(f.db, f.env);
  auto const entry = std::ranges::find_if(
    last, [&](const SelectionEntry& e) { return e.fft == fft.spec() && e.regime == interval(fft, top).regime; });
  CHECK(entry != last.end());
  if (entry == last.end()) { return; }
  CHECK(entry->reach > maxExp(fft));
  CHECK_EQ(entry->reach, read.back());
  CHECK(raisableZ(entry->reach) >= 28);
  CHECK(entry->evidence == Evidence::Confirmed);
}

TEST(a_run_that_restarts_stops_once_nothing_is_worth_the_stop_fraction) {
  Fixture plain;
  ProbeRun const descent = runProbed(plain);

  // Restarts are always worth something, so without a stop fraction only their space running out ends the run.
  Fixture forever;
  ProbeRun const unstopped = runProbed(forever, u32(descent.order.size()) + 40 * MIN_CALLS, {}, true, 0);
  CHECK(unstopped.report.stopped);
  CHECK(unstopped.report.end == QueueEnd::Stopped);

  // With one, the run ends by itself: the descent, then jumps until the ones that keep finding nothing have taught the
  // entry that its next is worth less than 0.1% of T.
  Fixture f;
  ProbeRun const run = runProbed(f, ~0u, {}, true, STOP);
  CHECK(!run.report.stopped);
  CHECK(run.report.end == QueueEnd::BelowStop);
  std::set<std::string> const stepped(descent.order.begin(), descent.order.end());
  std::vector<std::string> steps;
  std::ranges::copy_if(run.order, std::back_inserter(steps), [&](const std::string& s) { return stepped.contains(s); });
  CHECK(steps == descent.order);
  CHECK(run.report.spent.contains(ItemKind::Restart));
  CHECK(run.order.size() < unstopped.order.size());

  CHECK(near(run.report.floor, STOP * run.report.valuedT));
  CHECK(!run.report.left.empty());
  for (const Item& item : run.report.left) {
    CHECK(item.value > 0);
    CHECK(item.value < run.report.floor);
  }
}

TEST(a_run_with_nothing_worth_the_stop_fraction_reads_no_anchor) {
  Fixture f;
  ProbeRun const done = runProbed(f, ~0u, {}, true, STOP);
  CHECK(done.report.end == QueueEnd::BelowStop);

  // A later process on the same database finds nothing to do, and so nothing to divide by an anchor reading.
  f.newSession();
  FakeBench bench{f.db, f.sess, true};
  Scheduler scheduler = probedScheduler(true);
  QueueReport const again = runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, STOP);
  CHECK(again.end == QueueEnd::BelowStop);
  CHECK_EQ(again.items, 0u);
  CHECK_EQ(again.anchors, 0u);
  CHECK(bench.order.empty());
}

namespace {

// Everything shapes() offers, searched one key at a time, jumping once searched, and gated, the way a run is.
Scheduler summaryQueue() {
  return Scheduler{scope(),
                   baselines(nvidia(), scope(), shapes()),
                   1000,
                   Bootstrap{nvidia(), 118'063'003, {}, false},
                   Strategy{.kind = Strategy::Kind::Single},
                   true,
                   true};
}

struct SummaryRun {
  Fixture f;
  Scheduler scheduler = summaryQueue();
  QueueReport report;
  RunSummary summary;

  SummaryRun(u32 stopAfter, double stop) {
    FakeBench bench{f.db, f.sess, true, stopAfter};
    bench.optionFactor = interacting;
    bench.race = {{"512:15:512:212", 1700}};
    report = runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, stop);
    summary = summarize(scheduler, f.db, f.env, f.sess, report, stop);
  }
};

// What the summary says about a run, checked against the queue's own ranking of what it left, taken again.
void checkSummary(SummaryRun& run, double stop) {
  const RunSummary& s = run.summary;
  const TuneDB& db = run.f.db;
  u32 const env = run.f.env;

  Objective const valuing{db, env, scope(), Gating::Assumed};
  std::vector<Item> const left = run.scheduler.admissible(db, env, valuing);
  CHECK(!left.empty());

  CHECK(near(s.T, valuing.T()));
  CHECK(near(s.floor, stop * valuing.T()));
  CHECK_EQ(s.items, run.report.items);

  // Every kind the queue still offers, how many, and the most any one of them is worth.
  std::map<ItemKind, std::pair<u32, double>> byKind;
  for (const Item& item : left) {
    auto& [n, best] = byKind[item.kind];
    ++n;
    best = std::max(best, item.value);
  }
  CHECK_EQ(s.remaining.size(), byKind.size());
  for (const RunSummary::Remaining& r : s.remaining) {
    CHECK(byKind.contains(r.kind));
    CHECK_EQ(r.count, byKind[r.kind].first);
    CHECK(near(r.value, byKind[r.kind].second));
  }

  // Each type's entries, and of those the queue still offers a first measurement of, the one the least gain would
  // justify: a baseline the queue values at what the summary says, whose saving at that gain is exactly what an item
  // had to be worth, and which no other baseline of its type needs less gain than.
  std::map<enum FFT_TYPES, u32> entries;
  std::set<std::string> seen;
  for (const Baseline& b : run.scheduler.baselines()) {
    if (seen.insert(b.label()).second) { ++entries[b.fft.shape.fft_type]; }
  }
  u32 unmeasured = 0;
  for (const RunSummary::Family& fam : s.families) {
    CHECK_EQ(fam.entries, entries[fam.type]);
    CHECK(fam.measured + fam.unmeasured <= fam.entries);
    unmeasured += fam.unmeasured;
    if (!fam.unmeasured) { continue; }

    std::optional<double> least;
    const Item* closest = nullptr;
    for (const Item& item : left) {
      const Baseline& b = run.scheduler.baselines()[item.index];
      if (item.kind != ItemKind::Baseline || b.fft.shape.fft_type != fam.type) { continue; }
      std::optional<double> const g = requiredGain(valuing.points(), b.kind, b.band, item.cost, s.floor);
      if (g && (!least || *g < *least)) { least = g; }
      if (b.label() == fam.closest) { closest = &item; }
    }
    CHECK(closest != nullptr);
    if (!closest) { continue; }
    CHECK(near(fam.closestValue, closest->value));
    CHECK_EQ(fam.gain.has_value(), least.has_value());
    if (!fam.gain) { continue; }

    const Baseline& b = run.scheduler.baselines()[closest->index];
    CHECK(near(*fam.gain, *least));
    double const saved = saving(valuing.points(), b.kind, b.band, closest->cost * (1 - *fam.gain));
    CHECK(*fam.gain == 0 || std::abs(saved - s.floor) <= 1e-9 * std::max(1.0, s.floor));
    CHECK(near(fam.chance, gainsOf(db, env).global().chanceOfAtLeast(*fam.gain)));
  }
  CHECK_EQ(unmeasured, byKind.contains(ItemKind::Baseline) ? byKind[ItemKind::Baseline].first : 0u);

  GainModel const gains = gainsOf(db, env);
  CHECK(near(s.moves, gains.all().n()));
  CHECK(near(s.combos, gains.combos().n()));
  for (size_t i = 0; i < GAIN_BINS; ++i) {
    CHECK(near(s.moveGains.p[i], gains.global().p[i]));
    CHECK(near(s.comboGains.p[i], gains.globalCombo().p[i]));
  }

  CHECK_EQ(s.anchors, run.report.anchors);
}

}  // namespace

TEST(the_summary_numbers_are_the_queues_own_scores) {
  // Stopped part-way, with baselines, probes and refines all still offered.
  SummaryRun early{60, STOP};
  CHECK(early.summary.end == QueueEnd::Stopped);
  checkSummary(early, STOP);
  CHECK(std::ranges::any_of(early.summary.families, [](const RunSummary::Family& f) { return f.unmeasured > 0; }));

  // And run to its own end, where everything left is worth less than the stop fraction.
  SummaryRun done{~0u, STOP};
  CHECK(done.summary.end == QueueEnd::BelowStop);
  checkSummary(done, STOP);
  for (const RunSummary::Remaining& r : done.summary.remaining) { CHECK(r.value < done.summary.floor); }

  // With no stop fraction, the gain a family needed is the gain at which it would save anything at all.
  SummaryRun zero{60, 0};
  checkSummary(zero, 0);
}

TEST(the_drift_record_is_the_sessions_own_anchor_readings) {
  Fixture f;
  u32 const cfg = f.db.internCfg({});
  auto anchor = [&](u32 sess, double ratio, u64 ts) {
    CHECK(f.db.add(AnchorRow{.sess = sess,
                             .fft = "512:15:512:212",
                             .exponent = 118'063'003,
                             .cfg = cfg,
                             .mean = 1700 * ratio,
                             .ratio = ratio,
                             .ts = ts}));
  };
  anchor(f.sess, 1.0, 1);
  anchor(f.sess, 1.03, 2);
  anchor(f.sess, 0.975, 3);
  u32 const first = f.sess;
  f.newSession();
  anchor(f.sess, 1.12, 4);
  anchor(first, 1.01, 5);

  Scheduler const scheduler{scope(), {}};
  RunSummary const s = summarize(scheduler, f.db, f.env, first, QueueReport{}, STOP);
  CHECK_EQ(s.drift.anchor, std::string{"512:15:512:212@118063003"});
  CHECK_EQ(s.drift.readings, 4u);
  CHECK(near(s.drift.first, 1.0));
  CHECK(near(s.drift.last, 1.01));
  CHECK(near(s.drift.lo, 0.975));
  CHECK(near(s.drift.hi, 1.03));
  CHECK(s.drift.level == DriftLevel::Warn);
  CHECK(!s.drift.alarmed);

  RunSummary const later = summarize(scheduler, f.db, f.env, f.sess, QueueReport{}, STOP);
  CHECK_EQ(later.drift.readings, 1u);
  CHECK(later.drift.level == DriftLevel::Alarm);
  CHECK(f.db.add(AlarmRow{.sess = f.sess, .ts = 6}));
  CHECK(summarize(scheduler, f.db, f.env, f.sess, QueueReport{}, STOP).drift.alarmed);

  f.newSession();
  CHECK_EQ(summarize(scheduler, f.db, f.env, f.sess, QueueReport{}, STOP).drift.readings, 0u);
}

TEST(entries_held_back_by_rule_are_waiting_not_ruled_out) {
  // Stopped before the first call: the workload's coverage comes before the bootstrap, so what is owed is covering
  // baselines -- at a cold start every entry covers something -- nothing is measured, and nothing is ruled out either.
  Fixture f;
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, twoFamilies(), {}, false, true};
  FakeBench bench{f.db, f.sess, false, 0};
  QueueReport const report = runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, STOP);
  RunSummary const s = summarize(scheduler, f.db, f.env, f.sess, report, STOP);
  CHECK(!s.families.empty());
  CHECK(std::ranges::all_of(report.left, [](const Item& i) { return i.cover; }));
  for (const RunSummary::Family& fam : s.families) {
    CHECK_EQ(fam.measured, 0u);
    CHECK_EQ(fam.unmeasured + fam.waiting, fam.entries);
  }

  // Stopped with a gate reading owed: the entry it reads is measured, and every other is waiting on the gate.
  Fixture g;
  Scheduler gated{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  FakeBench first{g.db, g.sess, false, MIN_CALLS};
  QueueReport const held = runQueue(gated, g.db, g.env, first, [](const Objective&, const Defaults&) {}, STOP);
  CHECK(std::ranges::all_of(held.left, [](const Item& i) { return i.kind == ItemKind::Gate; }));
  RunSummary const t = summarize(gated, g.db, g.env, g.sess, held, STOP);
  CHECK_EQ(t.heldBy, std::string{"the accuracy gate"});
  u32 measured = 0;
  for (const RunSummary::Family& fam : t.families) {
    CHECK_EQ(fam.measured + fam.waiting, fam.entries);
    measured += fam.measured;
  }
  CHECK_EQ(measured, 1u);
}

// ---------------------------------------------------------------------------------------------------------------
// Status: the same queue, asked from outside the run.

namespace {

// What a bench records for `item`'s call, as it spells its order.
std::string called(const Scheduler& scheduler, const Item& item) {
  const FFTConfig& fft = item.fft ? *item.fft : scheduler.baselines()[item.index].fft;
  bool const reads = item.kind == ItemKind::Gate || item.kind == ItemKind::Reach;
  return (reads ? "gate " : "") + fft.spec() + "@" + std::to_string(item.exponent) +
    (item.options.empty() ? "" : " " + configText(item.options));
}

// The database as a status reads it, from outside every process that wrote it.
TuneDB readBack(const TuneDB& db) {
  TuneDB out;
  CHECK(out.parse(db.text(), "status"));
  return out;
}

}  // namespace

TEST(a_status_is_the_queues_own_view_of_the_database_it_reads) {
  // A run stopped part-way, its file read while another row is half written.
  SummaryRun run{60, STOP};
  TuneDB db;
  CHECK(db.parse(completeLines(run.f.db.text() + "run   1 512:15:512:212 prp 1180"), "status"));
  CHECK_EQ(db.text(), run.f.db.text());

  Scheduler const fresh = summaryQueue();
  TuneStatus const s = statusOf(fresh, db, run.f.env, STOP, {});

  // Its counts are the run's own summary's.
  CHECK_EQ(s.families.size(), run.summary.families.size());
  for (size_t i = 0; i < std::min(s.families.size(), run.summary.families.size()); ++i) {
    const RunSummary::Family& mine = s.families[i];
    const RunSummary::Family& theirs = run.summary.families[i];
    CHECK(mine.type == theirs.type);
    CHECK_EQ(mine.entries, theirs.entries);
    CHECK_EQ(mine.measured, theirs.measured);
    CHECK_EQ(mine.unmeasured, theirs.unmeasured);
    CHECK_EQ(mine.waiting, theirs.waiting);
  }
  // To the precision the file holds a mean to, which is all a later process has of it.
  CHECK(std::abs(s.valuedT - run.summary.T) <= 1e-6 * run.summary.T);
  CHECK(std::abs(s.floor - run.summary.floor) <= 1e-6 * run.summary.floor);

  // Every item the queue offers is accounted for, and those named are worth running, best rate first.
  Objective const valuing{db, run.f.env, scope(), Gating::Assumed};
  std::vector<Item> const left = fresh.admissible(db, run.f.env, valuing);
  CHECK_EQ(s.next.size(), STATUS_NEXT);
  CHECK_EQ(s.next.size() + s.moreWorth + s.notWorth, left.size());
  for (size_t i = 0; i < s.next.size(); ++i) {
    CHECK(worthRunning(s.next[i].item, s.floor));
    CHECK(i == 0 || s.next[i].item.rate() <= s.next[i - 1].item.rate());
  }

  // And the first is what the next process runs first.
  u32 const sess = db.beginSession(run.f.env, "512:15:512:212@118063003", 0, 1'753'491'200);
  FakeBench bench{db, sess, false, 1};
  Scheduler again = summaryQueue();
  (void)runQueue(again, db, run.f.env, bench, [](const Objective&, const Defaults&) {}, STOP);
  CHECK_EQ(bench.order.size(), size_t{1});
  if (!bench.order.empty()) { CHECK_EQ(bench.order.front(), called(fresh, s.next.front().item)); }
}

TEST(a_status_counts_what_the_gate_made_of_what_is_published) {
  // 1K:8:1K:112 reads 17 at its top, so its reach is derived below the table's (as in the test above that derives it).
  Fixture f;
  FakeBench bench{f.db, f.sess};
  FFTConfig const cheapest{"1K:8:1K:112"};
  u64 const top = 167'772'107;
  bench.zOf = [&](const FFTConfig& fft, const UseConfig&, u64 E) {
    return fft.spec() == cheapest.spec() ? 17.0 + (double(top) - double(E)) / double(cheapest.size()) / 0.012 : 24.0;
  };
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  (void)runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {});

  Scheduler const fresh{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  TuneDB readF = readBack(f.db);
  TuneStatus const s = statusOf(fresh, readF, f.env, 0, {});
  const TuneStatus::Accuracy& a = s.accuracy;
  CHECK(a.entries > 0);
  CHECK_EQ(a.exact + a.confirmed + a.unvalidated, a.entries);
  CHECK(a.exact > 0);  // the hybrids
  CHECK(a.confirmed > 0);
  CHECK_EQ(a.belowTable, 1u);
  CHECK_EQ(a.aboveTable, 0u);
  CHECK_EQ(a.owed, 0u);
  CHECK_EQ(a.rejected, 0u);
  CHECK(s.next.empty());

  // A set measured and not yet read is owed, and its reading is the next thing a run takes, by rule.
  Fixture g;
  conclude(g, "1K:8:1K:112", {}, 2900);
  TuneDB readG = readBack(g.db);
  TuneStatus const owed = statusOf(fresh, readG, g.env, STOP, {});
  CHECK_EQ(owed.accuracy.owed, 1u);
  CHECK_EQ(owed.accuracy.entries, 0u);
  CHECK_EQ(owed.next.size(), size_t{1});
  CHECK(!owed.next.empty() && owed.next.front().item.kind == ItemKind::Gate);
  CHECK_EQ(owed.heldBy, std::string{"the accuracy gate"});

  // One that reads below the floor wherever it is read is rejected, and never published.
  Fixture h;
  FakeBench below{h.db, h.sess};
  below.zOf = [&](const FFTConfig& fft, const UseConfig&, u64) { return fft.spec() == cheapest.spec() ? 17.0 : 24.0; };
  Scheduler queue{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  (void)runQueue(queue, h.db, h.env, below, [](const Objective&, const Defaults&) {});
  TuneDB readH = readBack(h.db);
  TuneStatus const rejected = statusOf(fresh, readH, h.env, 0, {});
  CHECK_EQ(rejected.accuracy.rejected, 1u);
  CHECK_EQ(rejected.accuracy.owed, 0u);
}

TEST(an_attempt_is_being_measured_while_its_process_holds_the_database_and_a_fault_once_it_is_gone) {
  Fixture f;
  CHECK(f.db.add(TryRow{.sess = f.sess,
                        .fft = "1K:8:1K:212",
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .cfg = f.db.internCfg({{"TAIL_KERNELS", "3"}}),
                        .ts = 100}));
  u32 const died = f.sess;

  // Another card's fault is that card's.
  u32 const other = f.db.internEnv(DbEnv{.gpu = "another card", .name = "another card", .driver = "1.0"});
  u32 const otherSess = f.db.beginSession(other, "", 0, 1'753'400'000);
  CHECK(f.db.add(TryRow{.sess = otherSess,
                        .fft = "1K:8:1K:202",
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .cfg = f.db.internCfg({}),
                        .ts = 50}));

  f.newSession();
  CHECK(f.db.add(TryRow{.sess = f.sess,
                        .fft = "512:15:512:212",
                        .kind = TestKind::PRP,
                        .exponent = 118'063'003,
                        .cfg = f.db.internCfg({}),
                        .ts = 200}));
  TuneDB db = readBack(f.db);
  Scheduler const scheduler{scope(), {}};

  // Held: the newest session is the process holding the database, and its open attempt is what it is measuring.
  TuneStatus const held = statusOf(scheduler, db, f.env, STOP, {.held = true, .written = 250, .now = 260});
  CHECK(held.measuring.has_value());
  if (held.measuring) {
    CHECK_EQ(held.measuring->row.fft, std::string{"512:15:512:212"});
    CHECK_EQ(held.measuring->options, std::string{"-"});
  }
  CHECK_EQ(held.faults.size(), size_t{1});
  if (!held.faults.empty()) {
    CHECK_EQ(held.faults.front().row.sess, died);
    CHECK_EQ(held.faults.front().options, std::string{"TAIL_KERNELS=3"});
  }
  CHECK(held.latest.has_value() && held.latest->id == f.sess);

  // What it is measuring is not held against it, as the process holding it will answer it.
  CHECK(!db.diedOn(f.env, db.findCfgId({}), TestKind::PRP, "512:15:512:212", 118'063'003));

  // Nothing holding it, the newest session died holding its attempt too.
  TuneDB again = readBack(f.db);
  TuneStatus const gone = statusOf(scheduler, again, f.env, STOP, {.held = false, .written = 250, .now = 260});
  CHECK(!gone.measuring.has_value());
  CHECK_EQ(gone.faults.size(), size_t{2});
}

TEST(a_watch_is_told_each_call_as_the_log_names_it_and_where_the_run_stands_after_it) {
  // What the queue tells a watch, in the order it tells it: "?" before a call, "=" for where the run stands.
  struct Recorder final : Watch {
    std::vector<std::string> events;
    std::vector<RunProgress> stands;
    std::vector<double> stateT;
    std::vector<size_t> ranked;
    std::vector<Finished> done;

    void measuring(const std::string& what) override { events.push_back("? " + what); }
    void progress(const RunProgress& p) override {
      events.push_back("=");
      stands.push_back(p);
    }
    void state(const QueueState& q) override {
      events.push_back("s");
      stateT.push_back(q.objective.T());
      ranked.push_back(q.ranked.size());
    }
    void finished(const Finished& f) override {
      events.push_back("!");
      done.push_back(f);
    }
  };

  Fixture f;
  FakeBench bench{f.db, f.sess};
  Scheduler scheduler{scope(), baselines(nvidia(), scope(), shapes()), 1000, {}, {}, false, true};
  Recorder watch;
  QueueReport const report =
    runQueue(scheduler, f.db, f.env, bench, [](const Objective&, const Defaults&) {}, STOP, &watch);

  // Where the run stands comes first, and again after every call, so a call is never left as the last word; the state
  // it was ranked from straight after it; and each call's end before the next is named.
  CHECK(!watch.events.empty() && watch.events.front() == "=");
  CHECK(watch.events.size() > 1 && watch.events.back() == "s" && watch.events[watch.events.size() - 2] == "=");
  for (size_t i = 1; i < watch.events.size(); ++i) {
    CHECK(!(watch.events[i].starts_with("?") && watch.events[i - 1].starts_with("?")));
    CHECK((watch.events[i] == "s") == (watch.events[i - 1] == "="));
  }
  CHECK_EQ(watch.stateT.size(), watch.stands.size());
  for (size_t i = 0; i < std::min(watch.stateT.size(), watch.stands.size()); ++i) {
    CHECK(near(watch.stateT[i], watch.stands[i].T));
  }
  CHECK(!watch.ranked.empty() && watch.ranked.front() > 0);

  // Each call is named as the bench was asked for it, numbered as the log numbers it, and the anchor by name.
  std::vector<std::string> calls;
  u32 anchors = 0;
  for (const std::string& e : watch.events) {
    if (e == "? the drift anchor") {
      ++anchors;
    } else if (e.starts_with("? ")) {
      calls.push_back(e.substr(2));
    }
  }
  CHECK_EQ(anchors, report.anchors);
  CHECK_EQ(calls.size(), size_t{report.items});
  std::vector<std::string> asked;
  for (const std::string& o : bench.order) {
    if (o != "anchor") { asked.push_back(o); }
  }
  CHECK_EQ(asked.size(), calls.size());

  // Each call's end names it as its start did, numbered as the log numbers it, with T as the run's own figures moved.
  CHECK_EQ(watch.done.size(), calls.size());
  for (size_t i = 0; i < std::min(watch.done.size(), calls.size()); ++i) {
    const Finished& f = watch.done[i];
    CHECK_EQ(f.n, u32(i + 1));
    CHECK_EQ(std::to_string(f.n) + ". " + toString(f.kind) + " " + f.label + " at " + std::to_string(f.exponent) +
               f.call,
             calls[i]);
    CHECK(f.completed);
    CHECK(f.reads ? f.usPerIt == 0 && f.z > 0 : f.usPerIt > 0);
    if (i > 0) { CHECK(near(f.before, watch.done[i - 1].after)); }
  }
  if (!watch.done.empty()) { CHECK(near(watch.done.back().after, report.endT)); }

  for (size_t i = 0; i < std::min(asked.size(), calls.size()); ++i) {
    CHECK(calls[i].starts_with(std::to_string(i + 1) + ". "));
    std::string const at = asked[i].starts_with("gate ") ? asked[i].substr(5) : asked[i];
    std::string const spec = at.substr(0, at.find('@'));
    std::string const exponent = at.substr(at.find('@') + 1, at.find(' ') - at.find('@') - 1);
    CHECK(calls[i].find(" " + spec) != std::string::npos);
    CHECK(calls[i].find(" at " + exponent) != std::string::npos);
  }

  // The last word is the run's own: what it ran and the T it ended on.
  CHECK(!watch.stands.empty());
  if (!watch.stands.empty()) {
    const RunProgress& last = watch.stands.back();
    CHECK_EQ(last.items, report.items);
    CHECK_EQ(last.anchors, report.anchors);
    CHECK(near(last.T, report.endT));
    CHECK(near(last.startT, report.startT));
    CHECK(last.measured > 0 && last.measured <= 1);
    CHECK(last.spent.size() == report.spent.size());
    CHECK_EQ(last.worthRunning, 0u);
    CHECK(near(last.floor, report.floor));
  }
}
