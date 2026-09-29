// Copyright (C) Jason Lynch

#include "Landscape.h"

#include "Emit.h"
#include "Gate.h"
#include "History.h"

#include <algorithm>
#include <set>

namespace landscape {

using namespace tune;

namespace {

// What a first build of a configuration costs on top of its call, and what a call costs beside its iterations.
constexpr double COMPILE_SECONDS = 12;
constexpr double OVERHEAD_SECONDS = 1.5;

class SimBench final : public Bench {
public:
  SimBench(TuneDB& db, u32 sess, const Landscape& landscape, double budget, u64 epoch) :
    db_{db}, sess_{sess}, landscape_{landscape}, budget_{budget}, epoch_{epoch} {}

  std::map<std::string, u32> calls;

  [[nodiscard]] double clock() const { return clock_; }

  [[nodiscard]] bool anchorDue() const override { return false; }
  void timeAnchor() override {}

  void declareRestart(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 k) override {
    (void)db_.add(JumpRow{.sess = sess_,
                          .fft = fft.spec(),
                          .kind = kind,
                          .regime = regimeOf(fft, exponent),
                          .cfg = db_.internCfg(options),
                          .k = k,
                          .ts = ts()});
  }

  void declareCombo(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 tier) override {
    (void)db_.add(ComboRow{.sess = sess_,
                           .fft = fft.spec(),
                           .kind = kind,
                           .regime = regimeOf(fft, exponent),
                           .cfg = db_.internCfg(options),
                           .tier = tier,
                           .ts = ts()});
  }

  void declareBootstrap(const FFTConfig& fft, u64 probe) override {
    (void)db_.add(BootRow{.sess = sess_, .fft = fft.spec(), .probe = probe, .ts = ts()});
  }

  void declareRound(const RoundRow& round) override {
    RoundRow row = round;
    row.sess = sess_;
    row.ts = ts();
    (void)db_.add(row);
  }

  [[nodiscard]] Result run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                           const std::string& moved) override {
    bool const fresh = built_.insert(fft.spec() + " " + configText(options)).second;
    ++calls[fft.spec() + " " + toString(kind) + " " + regimeOf(fft, exponent).label()];

    if (landscape_.fails(fft, options)) {
      clock_ += OVERHEAD_SECONDS + (fresh ? COMPILE_SECONDS : 0);
      record(fft, kind, exponent, options, {.drift = 1, .status = Status::NoCompile, .ts = ts()});
      if (auto const at = options.find(moved); at != options.end()) {
        (void)db_.add(NogoRow{.sess = sess_, .fft = fft.spec(), .key = moved, .val = at->second, .ts = ts()});
      }
      return {.completed = false, .ran = options, .status = Status::NoCompile};
    }

    double const cost = landscape_.cost(fft, kind, options);
    double const seconds =
      (1 + BLOCKS_PER_CALL) * 1000 * cost * 1e-6 + OVERHEAD_SECONDS + (fresh ? COMPILE_SECONDS : 0);
    clock_ += seconds;
    record(fft, kind, exponent, options,
           {.mean = cost,
            .stddev = cost * 0.001,
            .blocks = BLOCKS_PER_CALL,
            .calls = 1,
            .drift = 1,
            .status = Status::Ok,
            .ts = ts()});
    return {.completed = true, .seconds = seconds, .usPerIt = cost, .ran = options};
  }

  // Every set reads a z the floor of every type is clear of.
  [[nodiscard]] Reading gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) override {
    double const seconds =
      ROE_ITERATIONS * landscape_.cost(fft, TestKind::PRP, options) * 1e-6 + OVERHEAD_SECONDS + COMPILE_SECONDS;
    clock_ += seconds;
    (void)db_.add(RoeRow{.sess = sess_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = db_.internCfg(options),
                         .z = 24,
                         .n = 2000,
                         .maxRoe = 0.3,
                         .checkOk = true,
                         .ts = ts()});
    return {.completed = true, .seconds = seconds, .z = 24, .n = 2000, .checkOk = true, .ran = options};
  }

  [[nodiscard]] bool stopped() const override { return clock_ >= budget_; }

private:
  [[nodiscard]] u64 ts() const { return epoch_ + u64(clock_); }

  void record(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, const Measurement& m) {
    (void)db_.add(RunRow{.sess = sess_,
                         .fft = fft.spec(),
                         .kind = kind,
                         .exponent = exponent,
                         .regime = regimeOf(fft, exponent),
                         .cfg = db_.internCfg(options),
                         .m = m});
  }

  TuneDB& db_;
  u32 sess_;
  const Landscape& landscape_;
  double budget_;
  u64 epoch_;
  double clock_ = 0;
  std::set<std::string> built_;
};

// Past every row `db` has, so that what this run records is later than anything before it.
u64 epochOf(const TuneDB& db) {
  u64 out = 1'753'471'200;
  for (const RunRow& row : db.mergedRuns()) { out = std::max(out, row.m.ts + 1); }
  for (const JumpRow& row : db.jumps()) { out = std::max(out, row.ts + 1); }
  for (const RoundRow& row : db.rounds()) { out = std::max(out, row.ts + 1); }
  return out;
}

std::optional<double> regretOf(const Objective& objective, const std::vector<Baseline>& entries,
                               const Landscape& landscape) {
  double excess = 0;
  double least = 0;
  for (const ObjectivePoint& point : objective.points()) {
    if (point.weight <= 0) { continue; }
    if (!point.cost || !point.cost->measured()) { return {}; }

    std::optional<double> best;
    for (const Baseline& b : entries) {
      if (b.kind == point.kind && b.band.contains(point.exponent)) {
        best = std::min(best.value_or(landscape.best(b.fft, b.kind)), landscape.best(b.fft, b.kind));
      }
    }
    if (!best) { continue; }
    excess += point.weight * std::max(0.0, point.cost->us - *best);
    least += point.weight * *best;
  }
  return least > 0 ? excess / least : 0;
}

}  // namespace

Env device() {
  Env env;
  env.isNvidia = true;
  env.computeCapability = 806;
  return env;
}

std::optional<double> Outcome::settled(double within) const {
  std::optional<double> out;
  for (const Moment& m : trace) {
    if (m.regret && *m.regret <= within) {
      if (!out) { out = m.seconds; }
    } else {
      out.reset();
    }
  }
  return out;
}

Outcome simulate(const Scenario& scenario, const Policy& policy, TuneDB* given) {
  TuneDB fresh;
  TuneDB& db = given ? *given : fresh;
  Env const env = device();
  u32 const id = db.internEnv(dbEnvOf(env));
  u64 const epoch = epochOf(db);
  u32 const sess = db.beginSession(id, "", 0, epoch);

  std::vector<FFTShape> shapes;
  for (const std::string& spec : scenario.ffts) {
    FFTShape const shape = FFTConfig{spec}.shape;
    if (std::ranges::none_of(shapes, [&](const FFTShape& s) { return s.spec() == shape.spec(); })) {
      shapes.push_back(shape);
    }
  }
  std::vector<Baseline> entries;
  std::vector<FFTConfig> inScope;
  for (const Baseline& b : baselines(env, scenario.scope, shapes)) {
    if (std::ranges::find(scenario.ffts, b.fft.spec()) == scenario.ffts.end()) { continue; }
    entries.push_back(b);
    inScope.push_back(b.fft);
  }

  Bootstrap bootstrap{env, scenario.scope.probe, bootstrapFamilies(env, scenario.scope.probe, inScope),
                      policy.bootstrap, probeKind(scenario.scope)};
  Scheduler scheduler{scenario.scope,  entries,         1000,        std::move(bootstrap),
                      policy.strategy, policy.restarts, policy.gate, policy.halving};

  SimBench bench{db, sess, scenario.landscape, scenario.budget, epoch};
  Outcome out;
  u32 items = 0;
  auto const publish = [&](const Objective& objective, const Defaults&) {
    out.trace.push_back(
      {.seconds = bench.clock(), .items = items++, .regret = regretOf(objective, entries, scenario.landscape)});
  };
  QueueReport const report = runQueue(scheduler, db, id, bench, publish, policy.stop);

  out.seconds = bench.clock();
  out.items = report.items;
  out.calls = bench.calls;
  for (const RoundRow& r : db.rounds()) {
    if (db.envOf(r.sess) != id) { continue; }
    std::string text = std::to_string(r.round) + " " + std::to_string(r.calls) + ":";
    for (const RoundMember& m : r.members) { text += " " + m.fft; }
    out.rounds.push_back(std::move(text));
  }
  for (const BootRow& b : db.boots()) {
    if (db.envOf(b.sess) == id) { out.bootstrapped.push_back(b.fft); }
  }
  for (const SelectionEntry& e : entriesFor(db, id)) {
    auto const fft = parseFft(e.fft);
    out.published[e.fft + " " + toString(e.kind) + " " + e.regime.label()] =
      configText(fft ? canonicalConfig(env, *fft, e.opts) : e.opts);
  }
  return out;
}

}  // namespace landscape
