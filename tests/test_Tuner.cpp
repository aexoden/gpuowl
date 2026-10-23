// Copyright (C) Jason Lynch

// The tuning loop against a simulated card: costs from a hidden function of the configuration, with noise, recorded
// in a real database exactly as the device's would be.

#include "test.h"

#include "Explore.h"
#include "FFTVariants.h"
#include "Prior.h"
#include "timeutil.h"
#include "Tuner.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>

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
                {.stop = 0, .lanes = *parseTimetable("40/30/30/0"), .seed = 11, .regretEvery = 1000, .maxItems = 60});

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
  for (const auto& [packet, steps] : restarts) {
    CHECK(packet > 0);
    CHECK(steps.size() <= RESTART_MOST);
    for (size_t i = 0; i < steps.size(); ++i) {
      CHECK_EQ(steps[i]->ordinal, u32(i + 1));
      CHECK(steps[i]->fft == steps.front()->fft);
    }
  }
}
