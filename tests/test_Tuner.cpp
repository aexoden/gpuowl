// Copyright (C) Jason Lynch

// The tuning loop against a simulated card: costs from a hidden function of the configuration, with noise, recorded
// in a real database exactly as the device's would be.

#include "test.h"

#include "Audit.h"
#include "Explore.h"
#include "FFTVariants.h"
#include "File.h"
#include "Prior.h"
#include "Report.h"
#include "timeutil.h"
#include "Tuner.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>

#include <unistd.h>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

// A card on which cost follows the stated per-type prior, the FP64 types are 30% dearer than it, and ZEROHACK_W=0
// saves 4% wherever it applies.
class SimDevice final : public Device {
public:
  SimDevice(TuneDB& db, u32 sess) : db_{db}, sess_{sess} {}

  static double truth(const Subject& s, const UseConfig& options) {
    double us = statedPrior(s.fft.shape) * (s.fft.FFT_FP64 ? 1.3 : 1.0);
    if (useValue(options, "ZEROHACK_W", 1) == 0) { us *= 0.96; }
    return us;
  }

  Call call(const Candidate& c, const UseConfig& options) override {
    double const us = truth(c.subject, options) * (1 + 0.003 * noise_(rng_));
    Call out;
    out.core.stats = {.mean = us, .sd = us * 0.001, .n = 4};
    u32 const cfg = db_.internCfg(options);
    CHECK(db_.add(CallRow{.sess = sess_,
                          .fft = c.subject.fft.spec(),
                          .kind = c.subject.kind,
                          .exponent = c.timing,
                          .cfg = cfg,
                          .mean = us,
                          .sd = us * 0.001,
                          .blocks = 4,
                          .ts = clock_ += 8}));
    ++calls;
    return out;
  }

  RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) override {
    RoeCheck out{.applicable = true, .checkOk = true, .z = 30, .n = 9000, .exponent = exponent};
    CHECK(db_.add(RoeRow{.sess = sess_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = db_.internCfg(options),
                         .checkOk = true,
                         .z = 30,
                         .n = 9000,
                         .ts = clock_ += 3}));
    return out;
  }

  [[nodiscard]] std::string held(const Candidate&, const UseConfig&) const override { return {}; }
  [[nodiscard]] bool stopped() const override { return false; }
  [[nodiscard]] u32 session() const override { return sess_; }
  // A call takes 8 s and a gate reading 3, whatever the machine running the test is doing.
  [[nodiscard]] u64 now() const override { return clock_; }

  u32 calls = 0;

private:
  TuneDB& db_;
  u32 sess_;
  std::mt19937_64 rng_{3};
  u64 clock_ = 1000;
  std::normal_distribution<double> noise_{0, 1};
};

}  // namespace

TEST(the_loop_publishes_what_the_simulated_card_rewards) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  u32 const sess = db.beginSession(env, "", 0, "");
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});

  SimDevice device{db, sess};
  Timer t;
  LoopResult const r = runLoop(device, db, env, nvidia(), scope,
                               // The model's loop alone: what the other lanes add has a test of its own.
                               {.stop = 0.002,
                                .lanes = *parseTimetable("model"),
                                .seed = u64(getenv("SEED") ? atoi(getenv("SEED")) : 7),
                                .regretEvery = 8,
                                .maxItems = u32(getenv("ITEMS") ? atoi(getenv("ITEMS")) : 80)});
  double const seconds = t.at();
  printf("loop: %u calls, %u gate readings, regret %.4f, T %.1f, %.1f s (%.0f ms an item)\n", r.calls, r.gateReadings,
         r.regret, r.T, seconds, 1000 * seconds / std::max(1u, r.calls + r.gateReadings));

  // What production would run at the probe uses the saving.
  Readings const readings{db, env, nvidia()};
  std::vector<std::vector<Candidate>> const candidates{candidatesOf(nvidia(), scope, TestKind::PRP)};
  Publication const pub = publish(scope, candidates, readings);
  const Line* const line = pub.of(TestKind::PRP)->at(scope.probe);
  CHECK(line != nullptr);
  if (!line) { return; }
  printf("loop: at the probe %s -use %s, %.1f us/it\n", line->candidate->subject.fft.spec().c_str(),
         configText(line->reading->options).c_str(), line->reading->mean);

  // The cheapest true cost at the probe, among every candidate at its defaults with the saving, against what is
  // published there.
  double best = std::numeric_limits<double>::infinity();
  for (const Candidate& c : candidates.front()) {
    if (!c.interval.contains(scope.probe)) { continue; }
    best = std::min(best, SimDevice::truth(c.subject, {{"ZEROHACK_W", "0"}}));
  }
  double const published = SimDevice::truth(line->candidate->subject, line->reading->options);
  printf("loop: true cost there %.1f against the best %.1f (%+.2f%%)\n", published, best, (published / best - 1) * 100);
  CHECK(published <= best * 1.01);
}

TEST(the_lanes_measure_legal_sets_in_packets_of_their_own) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  u32 const sess = db.beginSession(env, "", 0, "");
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});
  SimDevice device{db, sess};
  (void)runLoop(device, db, env, nvidia(), scope,
                {.stop = 0, .lanes = *parseTimetable("30/25/25/20"), .seed = 11, .regretEvery = 1000, .maxItems = 60});

  std::map<std::string, u32> lanes;
  u32 branches = 0;
  std::map<u32, std::vector<const ItemRow*>> restarts;
  for (const ItemRow& r : db.items()) {
    ++lanes[r.lane];
    const UseConfig* const options = db.findCfg(r.cfg);
    CHECK(options != nullptr);
    if (!options) { continue; }
    FFTConfig const fft{r.fft};
    // What a lane proposes is what the host would build.
    CHECK(fitted(nvidia(), fft, *options) == *options);
    if (r.what == "first-look") { CHECK(options->empty()); }
    if (r.lane == "restarts") { restarts[r.packet].push_back(&r); }
    // A branch's set stays in a branch its entry was already read in.
    if (r.what == "branch") {
      UseConfig const branch = branchOf(nvidia(), fft, *options);
      bool before = false;
      for (const ItemRow& earlier : db.items()) {
        if (&earlier == &r) { break; }
        const UseConfig* const o = db.findCfg(earlier.cfg);
        before = before || (earlier.fft == r.fft && o && branchOf(nvidia(), fft, *o) == branch);
      }
      CHECK(before);
      ++branches;
    }
  }
  CHECK(branches > 0);
  CHECK(lanes["model"] > 0);
  CHECK(lanes["coverage"] > 0);
  CHECK(lanes["restarts"] > 0);
  CHECK(lanes["interactions"] > 0);
  for (const auto& [packet, steps] : restarts) {
    CHECK(packet > 0);
    CHECK(steps.size() <= RESTART_MOST);
    for (size_t i = 0; i < steps.size(); ++i) {
      CHECK_EQ(steps[i]->ordinal, u32(i + 1));
      CHECK(steps[i]->fft == steps.front()->fft);
    }
  }
}

TEST(a_budget_ends_the_campaign_and_a_larger_one_resumes_it) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});

  // A simulated call takes 8 s: 10 minutes is about 75 of them.
  fs::path const dir = fs::temp_directory_path() / ("prpll-test-budget-" + std::to_string(getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  SimDevice first{db, db.beginSession(env, "", 0, "")};
  LoopResult const a = runLoop(
    first, db, env, nvidia(), scope,
    {.stop = 0, .budget = 600, .seed = 3, .regretEvery = 1000, .maxItems = 500, .publishTo = dir, .settings = "sim"});
  CHECK(a.budgetSpent);
  // The report beside the tune files says how the run ended.
  std::string const report = File::openRead(dir / REPORT_FILE).readAll();
  CHECK(report.find("\"state\": \"budget spent\"") != std::string::npos);
  CHECK(report.find("\"settings\": \"sim\"") != std::string::npos);
  CHECK(report.find("\"published\": [\n") != std::string::npos);
  CHECK(std::ranges::count(report, '{') == std::ranges::count(report, '}'));
  CHECK(std::ranges::count(report, '[') == std::ranges::count(report, ']'));
  fs::remove_all(dir);
  CHECK(a.elapsed >= 600 && a.elapsed < 640);
  CHECK(first.calls > 50 && first.calls < 90);

  // Resumed with 15 minutes: about five more.
  SimDevice second{db, db.beginSession(env, "", 1, "")};
  LoopResult const b = runLoop(second, db, env, nvidia(), scope,
                               {.stop = 0, .budget = 900, .seed = 4, .regretEvery = 1000, .maxItems = 500});
  CHECK(b.budgetSpent);
  CHECK(b.elapsed >= 900 && b.elapsed < 940);
  CHECK(second.calls > 25 && second.calls < 50);
}

TEST(the_end_of_a_budget_confirms_then_audits) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});
  SimDevice device{db, db.beginSession(env, "", 0, "")};
  LoopResult const r =
    runLoop(device, db, env, nvidia(), scope,
            {.stop = 0, .budget = 1800, .reserve = 0.4, .seed = 5, .regretEvery = 1000, .maxItems = 900});
  CHECK(r.budgetSpent);

  // Every published line read afresh CONFIRM_CALLS times before the audit was declared.
  CHECK_EQ(db.audits().size(), size_t{1});
  const AuditRow& audit = db.audits().front();
  std::map<std::string, u32> confirmed;
  for (const ItemRow& item : db.items()) {
    if (item.lane == "confirm") {
      CHECK(item.ts <= audit.ts);
      ++confirmed[item.fft + '|' + std::to_string(item.cfg)];
    }
  }
  CHECK(!confirmed.empty());
  for (const auto& [line, calls] : confirmed) { CHECK(calls <= CONFIRM_CALLS); }

  // No more trials than declared, each no longer than a restart's packet.
  CHECK(audit.trials > 0);
  std::map<u32, u32> trials;
  for (const ItemRow& item : db.items()) {
    if (item.lane == "audit") { ++trials[item.packet]; }
  }
  CHECK(!trials.empty() && trials.size() <= audit.trials);
  for (const auto& [packet, steps] : trials) { CHECK(steps <= RESTART_PACKET); }

  Readings const readings{db, env, nvidia()};
  std::vector<std::vector<Candidate>> const candidates{candidatesOf(nvidia(), scope, TestKind::PRP)};
  Publication const published = publish(scope, candidates, readings);
  std::vector<std::string> const report = auditReport(auditSummary(db, env, readings, published, true), true);
  CHECK(report.size() >= 3);
  for (const std::string& line : report) { printf("%s\n", line.c_str()); }
}

TEST(a_register_key_is_offered_the_plan_read_nearest_the_best_set) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  u32 const sess = db.beginSession(env, "", 0, "");
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});
  std::vector<std::vector<Candidate>> const candidates{candidatesOf(nvidia(), scope, TestKind::PRP)};
  const Candidate& c = candidates.front().front();
  std::string const fft = c.subject.fft.spec();

  // The first build, at the defaults, and one in the INPLACE=0 branch that needed more registers.
  RegisterPlan const first{.k0 = 3, .occupancies = {{3, 80}, {4, 64}}};
  RegisterPlan const notInPlace{.k0 = 2, .occupancies = {{2, 128}, {3, 80}}};
  KernelResources const resources{.regs = 80, .localBytes = 0, .sharedBytes = 0, .threads = 256};
  CHECK(db.add(RegsRow{.sess = sess,
                       .fft = fft,
                       .kind = TestKind::PRP,
                       .key = "REGCF64",
                       .resources = resources,
                       .plan = first,
                       .ts = 1}));
  u32 const off = db.internCfg({{"INPLACE", "0"}});
  CHECK(db.add(KresRow{.sess = sess,
                       .fft = fft,
                       .kind = TestKind::PRP,
                       .key = "REGCF64",
                       .resources = {.regs = 128, .localBytes = 0, .sharedBytes = 0, .threads = 256},
                       .plan = notInPlace,
                       .cfg = off,
                       .ts = 2}));

  {
    Readings const readings{db, env, nvidia()};
    auto const offers = registerOffers(db, env, nvidia(), candidates, readings);
    CHECK(offers.at(&c).at("REGCF64") == first.values());
  }
  // Once the best set is INPLACE=0, its plan.
  CHECK(db.add(CallRow{.sess = sess,
                       .fft = fft,
                       .kind = TestKind::PRP,
                       .exponent = c.timing,
                       .cfg = off,
                       .mean = 500,
                       .sd = 0.5,
                       .blocks = 4,
                       .ts = 3}));
  Readings const readings{db, env, nvidia()};
  auto const offers = registerOffers(db, env, nvidia(), candidates, readings);
  CHECK(offers.at(&c).at("REGCF64") == notInPlace.values());
}

TEST(an_audit_interrupted_is_resumed_not_declared_again) {
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});
  LoopSettings settings{.stop = 0, .budget = 1800, .reserve = 0.4, .seed = 5, .regretEvery = 1000, .maxItems = 900};
  auto fresh = [](TuneDB& db) {
    CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "sim"));
    return db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  };

  // Where an uninterrupted campaign declares its audit: the simulated card is the same every time.
  u32 declaredAt = 0;
  {
    TuneDB db;
    u32 const env = fresh(db);
    SimDevice device{db, db.beginSession(env, "", 0, "")};
    (void)runLoop(device, db, env, nvidia(), scope, settings);
    CHECK_EQ(db.audits().size(), size_t{1});
    for (const ItemRow& item : db.items()) {
      declaredAt += item.ts < db.audits().front().ts || (item.ts == db.audits().front().ts && item.lane != "audit");
    }
  }

  // The same campaign stopped a few items into its audit, then resumed in a session of its own.
  TuneDB db;
  u32 const env = fresh(db);
  SimDevice first{db, db.beginSession(env, "", 0, "")};
  LoopSettings stopped = settings;
  stopped.maxItems = declaredAt + 5;
  (void)runLoop(first, db, env, nvidia(), scope, stopped);
  CHECK_EQ(db.audits().size(), size_t{1});
  u32 const declared = db.audits().front().trials;
  {
    Readings const readings{db, env, nvidia()};
    AuditState const midway = auditState(db, env, readings);
    CHECK(midway.started > 0 && midway.done < declared);
  }

  SimDevice second{db, db.beginSession(env, "", 1, "")};
  settings.seed = 6;
  LoopResult const r = runLoop(second, db, env, nvidia(), scope, settings);
  CHECK(r.budgetSpent);
  CHECK_EQ(db.audits().size(), size_t{1});
  Readings const readings{db, env, nvidia()};
  AuditState const state = auditState(db, env, readings);
  CHECK_EQ(state.started, declared);
  CHECK_EQ(state.done, declared);
}
