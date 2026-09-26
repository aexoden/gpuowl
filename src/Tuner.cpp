// Copyright (C) Jason Lynch

#include "Tuner.h"

#include "Accuracy.h"
#include "Args.h"
#include "BuildId.h"
#include "Emit.h"
#include "File.h"
#include "GpuCommon.h"
#include "log.h"
#include "Measure.h"
#include "Objective.h"
#include "Primes.h"
#include "Progress.h"
#include "Restart.h"
#include "Scheduler.h"
#include "Signal.h"
#include "Status.h"
#include "Summary.h"
#include "Task.h"
#include "TuneDB.h"
#include "Worktodo.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>
#include <vector>

namespace tune {

namespace {

constexpr const char* SELECTION_NAME = "selection.txt";
constexpr const char* TUNE_TXT_NAME = "tune.txt";

// How many grid points a report prints in full before it falls back to the heaviest few.
constexpr size_t GRID_SHOWN = 12;

// Upstream's own tuner is known by its option words (tune.cpp), and a -tune whose first word is one of them is its.
[[nodiscard]] bool isUpstreamWord(std::string_view token) {
  std::string_view const key = token.substr(0, token.find('='));
  for (std::string_view const word :
       {"noconfig", "inplace", "fp64", "ntt", "nofp32", "fp6431", "quick", "minexp", "maxexp"}) {
    if (key == word) { return true; }
  }
  return false;
}

[[nodiscard]] u32 asEnvId(std::string_view text, const char* what) {
  std::optional<u32> const id = parseInt<u32>(text);
  if (!id || *id == 0) { throw "-tune: " + std::string{what} + " takes the number of an env in the database"; }
  return *id;
}

[[nodiscard]] std::string hex16(u64 value) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return buf;
}

// Enough to tell one env from another by eye: which card it is, where that card sits, and which kernels its rows were
// measured against.
[[nodiscard]] std::string describe(const DbEnv& env) {
  return "  env " + std::to_string(env.id) + ": " + (env.gpu.empty() ? "unnamed device" : env.gpu) +
    (env.machine.empty() ? "" : " at " + env.machine) + ", " + (env.cudaBackend ? "cuda" : "ocl") + ", kernels " +
    hex16(env.build) + "\n";
}

void listEnvs(const std::vector<const DbEnv*>& envs) {
  for (const DbEnv* const env : envs) { log("%s", describe(*env).c_str()); }
}

// Everything the scope is expressed in has to be an exponent the rest of it can work with: one a primality test is
// answered for, since the probe is a prime, and one a worktodo could carry.
[[nodiscard]] std::string outsideBand(const char* what, const std::string& shown) {
  return "-tune: " + std::string{what} + shown + " is outside the exponents this can work with (" +
    std::to_string(MIN_EXPONENT) + " to " + std::to_string(MAX_EXPONENT) + ")";
}

void requireExponent(u64 E, const char* what) {
  if (E < MIN_EXPONENT || E > MAX_EXPONENT) { throw outsideBand(what, std::to_string(E)); }
}

// An exponent with the K/M/G suffix the ranges are usually written with.
[[nodiscard]] u64 asExponent(std::string_view text, const char* what) {
  u64 scale = 1;
  if (!text.empty()) {
    switch (text.back()) {
    case 'K':
    case 'k': scale = 1000; break;
    case 'M':
    case 'm': scale = 1'000'000; break;
    case 'G':
    case 'g': scale = 1'000'000'000; break;
    default: break;
    }
  }
  std::optional<u64> const value = parseInt<u64>(text.substr(0, text.size() - (scale > 1)));
  if (!value || *value == 0) { throw "-tune: " + std::string{what} + " takes an exponent, such as 400M or 400000000"; }
  // Checked before the multiplication rather than after it, which would wrap and land back in range.
  if (*value > MAX_EXPONENT / scale) { throw outsideBand(what, std::string{text}); }

  u64 const E = *value * scale;
  requireExponent(E, what);
  return E;
}

void parseWorkload(std::string_view text, ScopeArgs& out) {
  size_t const dash = text.find('-');
  out.lo = asExponent(text.substr(0, dash), "workload=");
  out.hi = dash == std::string_view::npos ? out.lo : asExponent(text.substr(dash + 1), "workload=");
  if (out.hi < out.lo) { throw std::string{"-tune: workload= is empty: its first exponent is the larger"}; }
}

std::vector<Group> parseGroups(std::string_view text) {
  std::vector<Group> out;
  for (size_t at = 0; at <= text.size();) {
    size_t const plus = text.find('+', at);
    std::string_view const one = text.substr(at, plus == std::string_view::npos ? plus : plus - at);
    at = plus == std::string_view::npos ? text.size() + 1 : plus + 1;

    auto const group = std::ranges::find_if(allGroups(), [&](Group g) { return one == toString(g); });
    if (group == allGroups().end()) {
      std::string names;
      for (Group const g : allGroups()) { names += (names.empty() ? "" : ", ") + std::string{toString(g)}; }
      throw "-tune accuracy: groups= does not know '" + std::string{one} + "'. Accepted: " + names;
    }
    if (std::ranges::find(out, *group) == out.end()) { out.push_back(*group); }
  }
  return out;
}

void parseKinds(std::string_view text, ScopeArgs& out) {
  out.kinds.clear();
  for (size_t at = 0; at <= text.size();) {
    size_t const plus = text.find('+', at);
    std::string_view const one = text.substr(at, plus == std::string_view::npos ? plus : plus - at);
    at = plus == std::string_view::npos ? text.size() + 1 : plus + 1;

    std::optional<TestKind> const kind = parseTestKind(one);
    if (!kind) { throw "-tune: kinds= does not know '" + std::string{one} + "'. Accepted: prp, ll, prp+ll"; }
    if (std::ranges::find(out.kinds, *kind) == out.kinds.end()) { out.kinds.push_back(*kind); }
  }
  if (out.kinds.empty()) { throw std::string{"-tune: kinds= takes at least one test kind"}; }
}

// A percentage of T, or a bare 0.  A bare fraction is refused rather than guessed at: 0.1 could mean either 0.1% or
// 10%.
[[nodiscard]] double parseStop(std::string_view text) {
  bool const percent = text.ends_with('%');
  std::optional<double> const value = parseNonNegative(percent ? text.substr(0, text.size() - 1) : text);
  if (!value || *value >= 100 || (!percent && *value != 0)) {
    throw std::string{"-tune: stop= takes a percentage of T below 100%, such as stop=0.1%, or 0 to run until stopped"};
  }
  return *value / 100;
}

// The prime at or below `E`, which is what the tuner can actually time, raised back into the range where there is no
// prime below `E` in it. Nothing at all where the range holds no prime: it then names no exponent that could be timed.
[[nodiscard]] std::optional<u64> primeAtOrBelow(const Primes& primes, u64 E, u64 lo, u64 hi) {
  u64 const below = primes.isPrime(E) ? E : primes.prevPrime(E);
  if (below >= lo) { return below; }

  u64 const above = primes.nextPrime(lo - 1);
  if (above <= hi) { return above; }
  return {};
}

// The most-populated bin of `PROBE_BIN` relative width, represented by the mean of the exponents in it. Ties go to the
// lower bin, so that the same worktodo always names the same probe.
[[nodiscard]] u64 modeOf(const std::vector<u64>& exponents, u64 lo) {
  struct Bin {
    u32 count = 0;
    double sum = 0;
  };
  std::map<i64, Bin> bins;

  double const width = std::log1p(PROBE_BIN);
  for (u64 const E : exponents) {
    Bin& bin = bins[i64(std::floor(std::log(double(E) / double(lo)) / width))];
    ++bin.count;
    bin.sum += double(E);
  }

  const Bin* best = nullptr;
  for (const auto& [index, bin] : bins) {
    if (!best || bin.count > best->count) { best = &bin; }
  }
  return u64(std::llround(best->sum / best->count));
}

// The points a kind with no pending work is spread over: `GRID_POINTS` log-spaced across the range, which samples the
// cost curve evenly in the variable it actually bends in.
[[nodiscard]] std::vector<u64> spreadOver(u64 lo, u64 hi) {
  if (hi <= lo) { return {lo}; }

  std::vector<u64> out;
  double const ratio = std::log(double(hi) / double(lo));
  for (u32 i = 0; i < GRID_POINTS; ++i) {
    u64 const E = std::clamp(u64(std::llround(double(lo) * std::exp(ratio * i / (GRID_POINTS - 1)))), lo, hi);
    if (out.empty() || out.back() != E) { out.push_back(E); }
  }
  return out;
}

[[nodiscard]] Grid gridFor(TestKind kind, const std::vector<PendingWork>& pending, u64 lo, u64 hi, u64 probe,
                           double probeWeight) {
  std::vector<u64> mine;
  for (const PendingWork& work : pending) {
    if (work.kind == kind && lo <= work.exponent && work.exponent <= hi) { mine.push_back(work.exponent); }
  }

  Grid out{.kind = kind, .fromWorktodo = !mine.empty(), .points = {}};

  std::map<u64, double> weights;
  if (out.fromWorktodo) {
    for (u64 const E : mine) { weights[E] += (1 - probeWeight) / double(mine.size()); }
  } else {
    std::vector<u64> const spread = spreadOver(lo, hi);
    for (u64 const E : spread) { weights[E] += (1 - probeWeight) / double(spread.size()); }
  }
  weights[probe] += probeWeight;

  for (const auto& [E, weight] : weights) { out.points.push_back({.exponent = E, .weight = weight}); }
  return out;
}

// The queue's items, run through a measurement session, which warms the device on whatever it builds first and races
// the env's anchor where it has none.
class SessionBench final : public Bench {
public:
  SessionBench(Session& session, u32 blockSize) : session_{session}, blockSize_{blockSize} {}

  [[nodiscard]] bool anchorDue() const override { return session_.anchorDue(); }

  void timeAnchor() override { session_.keepAnchor(); }

  void declareRestart(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 k) override {
    session_.declareRestart(fft, kind, exponent, options, k);
  }

  void declareCombo(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 tier) override {
    session_.declareCombo(fft, kind, exponent, options, tier);
  }

  [[nodiscard]] Result run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                           const std::string& moved) override {
    session_.varying(moved.empty() ? std::vector<std::string>{} : std::vector<std::string>{moved});
    Call const c = session_.run(fft, kind, exponent, options, BLOCKS_PER_CALL, blockSize_);
    session_.varying({});
    return {
      .completed = c.measurement.ok(), .seconds = c.buildSec + c.timedSec, .usPerIt = c.measurement.mean, .ran = c.ran};
  }

  [[nodiscard]] Reading gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) override {
    Timer timer;
    RoeCheck const roe = session_.checkRoe(fft, options, exponent);
    return {.completed = roe.status == Status::Ok && roe.applicable && !session_.stopped(),
            .seconds = timer.at(),
            .z = roe.z,
            .n = roe.n,
            .checkOk = roe.checkOk,
            .ran = roe.ran};
  }

  [[nodiscard]] bool stopped() const override { return session_.stopped() || Signal::stopRequested(); }

private:
  Session& session_;
  u32 blockSize_;
};

// The bootstrap a run over `scope` on `device` works through: one family per FFT type the workload reaches.
[[nodiscard]] Bootstrap bootstrapFor(const Env& device, const RunScope& scope, const std::vector<Baseline>& entries,
                                     bool enabled) {
  std::vector<FFTConfig> inScope;
  for (const Baseline& b : entries) { inScope.push_back(b.fft); }
  return Bootstrap{device, scope.probe, bootstrapFamilies(device, scope.probe, inScope), enabled};
}

// The lines what `env` has measured supports, for a command that has no run of its own to take them from.
[[nodiscard]] Defaults defaultsOf(const TuneDB& db, u32 env, const RunScope& scope) {
  const DbEnv* const row = db.findEnv(env);
  if (!row) { return {}; }
  Env const device = row->toEnv();
  return bootstrapFor(device, scope, baselines(device, scope), true).state(db, env).defaults;
}

// The queue a run with `command`'s settings works through on `device`.
[[nodiscard]] Scheduler schedulerFor(const Env& device, const RunScope& scope, const TuneCommand& command,
                                     u32 blockSize) {
  std::vector<Baseline> entries = baselines(device, scope);
  Bootstrap bootstrap = bootstrapFor(device, scope, entries, command.bootstrap);
  return Scheduler{scope, std::move(entries), blockSize, std::move(bootstrap), command.strategy, true, true};
}

// The shortest text that reads back as `value`.
[[nodiscard]] std::string shortest(double value) {
  char buf[32];
  auto const [end, ec] = std::to_chars(buf, buf + sizeof(buf), value);
  return ec == std::errc{} ? std::string(buf, end) : std::to_string(value);
}

[[nodiscard]] std::vector<std::string_view> settingTokens(std::string_view text) {
  std::vector<std::string_view> out;
  for (size_t at = 0; at < text.size();) {
    size_t const comma = std::min(text.find(',', at), text.size());
    if (comma > at) { out.push_back(text.substr(at, comma - at)); }
    at = comma + 1;
  }
  return out;
}

[[nodiscard]] std::string_view settingKey(std::string_view token) { return token.substr(0, token.find('=')); }

}  // namespace

bool ScopeArgs::wantsKind(TestKind kind) const { return std::ranges::find(kinds, kind) != kinds.end(); }

double Grid::weight(u64 E) const {
  auto const at = std::ranges::find_if(points, [E](const GridPoint& p) { return p.exponent == E; });
  return at == points.end() ? 0 : at->weight;
}

const Grid* RunScope::grid(TestKind kind) const {
  auto const at = std::ranges::find_if(grids, [kind](const Grid& g) { return g.kind == kind; });
  return at == grids.end() ? nullptr : &*at;
}

RunScope makeScope(const ScopeArgs& args, const std::vector<PendingWork>& pending) {
  if (args.kinds.empty()) { throw std::string{"-tune: there is no test kind to tune for"}; }

  // The settings the command line validates, validated again: this is the entry point, and not everything reaching it
  // has been through the parser.
  if (args.lo) {
    requireExponent(args.lo, "workload=");
    requireExponent(args.hi, "workload=");
  }
  if (args.probe) { requireExponent(args.probe, "probe="); }

  // An assignment for an exponent nothing here can place is not work this can be scoped to.
  std::vector<u64> wanted;
  for (const PendingWork& work : pending) {
    if (args.wantsKind(work.kind) && MIN_EXPONENT <= work.exponent && work.exponent <= MAX_EXPONENT) {
      wanted.push_back(work.exponent);
    }
  }

  RunScope out;
  out.probeWeight = args.probeWeight;

  if (args.lo) {
    out.lo = args.lo;
    out.hi = args.hi;
    out.rangeSource = "named on the command line";
  } else if (!wanted.empty()) {
    auto const [lo, hi] = std::ranges::minmax(wanted);
    out.lo = u64(double(lo) * (1 - WORKLOAD_PAD));
    out.hi = u64(double(hi) * (1 + WORKLOAD_PAD));
    out.rangeSource = std::to_string(wanted.size()) + (wanted.size() == 1 ? " assignment" : " assignments") +
      " pending, padded " + std::to_string(u32(WORKLOAD_PAD * 100)) + "% at each end";
  } else {
    out.lo = DEFAULT_WORKLOAD_LO;
    out.hi = DEFAULT_WORKLOAD_HI;
    out.rangeSource = "the default range, there being no pending work";
  }

  // A single exponent is a legitimate thing to tune for, but a range of no width leaves the grid with one point and
  // nothing to say about the exponents either side of it.
  if (out.hi == out.lo) {
    out.lo = u64(double(out.lo) * (1 - WORKLOAD_PAD));
    out.hi = u64(double(out.hi) * (1 + WORKLOAD_PAD));
  }

  // A named probe is where the user is headed, so a range that was derived rather than named reaches it.
  if (args.probe && !args.lo && (args.probe < out.lo || args.probe > out.hi)) {
    out.lo = std::min(out.lo, u64(double(args.probe) * (1 - WORKLOAD_PAD)));
    out.hi = std::max(out.hi, u64(double(args.probe) * (1 + WORKLOAD_PAD)));
    out.rangeSource += ", widened to the probe";
  }

  // The padding above is the only thing that can leave the band, every exponent it was computed from being inside it.
  out.lo = std::max(out.lo, MIN_EXPONENT);
  out.hi = std::min(out.hi, MAX_EXPONENT);

  std::vector<u64> inRange;
  for (u64 const E : wanted) {
    if (out.lo <= E && E <= out.hi) { inRange.push_back(E); }
  }

  Primes const primes;
  u64 asked = 0;
  if (args.probe) {
    asked = args.probe;
    if (asked < out.lo || asked > out.hi) {
      throw "-tune: probe=" + std::to_string(asked) + " is outside workload=" + std::to_string(out.lo) + "-" +
        std::to_string(out.hi);
    }
    out.probeSource = "named on the command line";
  } else if (!inRange.empty()) {
    asked = modeOf(inRange, out.lo);
    out.probeSource = "the most-populated " + std::to_string(u32(PROBE_BIN * 100)) + "% bin of the pending work";
  } else {
    asked = u64(std::llround(std::sqrt(double(out.lo) * double(out.hi))));
    out.probeSource = "the geometric centre of the range";
  }

  std::optional<u64> const probe = primeAtOrBelow(primes, asked, out.lo, out.hi);
  if (!probe) {
    throw "-tune: workload=" + std::to_string(out.lo) + "-" + std::to_string(out.hi) +
      " holds no prime, so there is no exponent in it the tuner could time";
  }

  out.probe = *probe;
  if (out.probe != asked) { out.probeSource += ", the prime at or below " + std::to_string(asked); }

  for (TestKind const kind : args.kinds) {
    out.grids.push_back(gridFor(kind, pending, out.lo, out.hi, out.probe, out.probeWeight));
  }
  return out;
}

std::vector<fs::path> worktodoFiles(const Args& args, const fs::path& dir) {
  std::vector<fs::path> out;

  // By file identity rather than by spelling, since a pool given as `.` of the run directory names the same file
  // twice and would count its assignments twice.
  auto take = [&out](const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) { return; }
    for (const fs::path& seen : out) {
      if (fs::equivalent(seen, path, ec) && !ec) { return; }
    }
    out.push_back(path);
  };

  for (u32 i = 0; i == 0 || i < args.workers; ++i) { take(dir / ("worktodo-" + std::to_string(i) + ".txt")); }

  // Not a file a run takes work from unless it is the pool's, but a user who has left one here means it as their
  // workload, and reading it costs a scope nothing.
  take(dir / "worktodo.txt");
  if (!args.masterDir.empty()) { take(args.masterDir / "worktodo.txt"); }
  return out;
}

std::vector<PendingWork> scanWorktodo(const std::vector<fs::path>& files) {
  std::vector<PendingWork> out;
  for (const Task& task : Worktodo::pending(files)) {
    switch (task.kind) {
    case Task::PRP:
    case Task::CERT: out.push_back({.kind = TestKind::PRP, .exponent = task.exponent}); break;
    case Task::LL: out.push_back({.kind = TestKind::LL, .exponent = task.exponent}); break;
    case Task::VERIFY: break;
    }
  }
  return out;
}

void reportScope(const RunScope& scope, const std::vector<fs::path>& files, const Objective& objective,
                 const std::string& against) {
  std::string read;
  for (const fs::path& file : files) { read += (read.empty() ? "" : ", ") + file.filename().string(); }
  log("tune: pending work read from %s\n", read.empty() ? "no worktodo file" : read.c_str());

  log("tune: workload %" PRIu64 "-%" PRIu64 " (%s)\n", scope.lo, scope.hi, scope.rangeSource.c_str());
  log("tune: probe %" PRIu64 " (%s), carrying %.0f%% of the weight\n", scope.probe, scope.probeSource.c_str(),
      scope.probeWeight * 100);

  for (const Grid& grid : scope.grids) {
    log("tune: %s grid: %zu %s, %s\n", toString(grid.kind), grid.points.size(),
        grid.points.size() == 1 ? "exponent" : "exponents",
        grid.fromWorktodo ? "from the pending work" : "spread across the range");

    // The whole grid where it is short enough to read, and otherwise the points carrying the most weight, which are
    // the ones a ranking is going to turn on.
    std::vector<GridPoint> shown = grid.points;
    bool const all = shown.size() <= GRID_SHOWN;
    if (!all) {
      // Exponent order breaks the ties, which a spread grid is almost entirely made of, so that what is printed is a
      // readable run of points rather than an arbitrary selection of equals.
      std::ranges::partial_sort(shown, shown.begin() + GRID_SHOWN, [](const GridPoint& a, const GridPoint& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.exponent < b.exponent;
      });
      shown.resize(GRID_SHOWN);
      std::ranges::sort(shown, [](const GridPoint& a, const GridPoint& b) { return a.exponent < b.exponent; });
    }
    for (const GridPoint& point : shown) {
      std::optional<Cost> const cost = objective.cStar(grid.kind, point.exponent);
      std::string source = "no FFT can run it";
      if (cost) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%9.3f us/it  ", cost->us);
        source = buf + (cost->measured() ? "entry " + cost->entry + ", " + cost->fft : "prior, from " + cost->fft);
      }
      log("tune:   %" PRIu64 " %5.1f%%  %s%s\n", point.exponent, point.weight * 100, source.c_str(),
          point.exponent == scope.probe ? "  (probe)" : "");
    }
    if (!all) { log("tune:   ... and %zu more\n", grid.points.size() - shown.size()); }
  }

  log("tune: T = %.3f us/it %s, %.1f%% of the weight on measured entries\n", objective.T(), against.c_str(),
      objective.measured() * 100);
  if (double const lost = objective.unservable(); lost > 0) {
    log("tune: %.1f%% of the weight is on exponents no FFT can run, and is left out of T\n", lost * 100);
  }
}

const char* toString(TuneVerb verb) {
  switch (verb) {
  case TuneVerb::Emit: return "emit";
  case TuneVerb::Reset: return "reset";
  case TuneVerb::Adopt: return "adopt";
  case TuneVerb::Compact: return "compact";
  case TuneVerb::Scope: return "scope";
  case TuneVerb::Run: return "run";
  case TuneVerb::Accuracy: return "accuracy";
  case TuneVerb::Status: return "status";
  }
  return "?";
}

bool opensDevice(TuneVerb verb) { return verb == TuneVerb::Run || verb == TuneVerb::Accuracy; }

std::optional<TuneCommand> parseTuneCommand(std::string_view text) {
  size_t const firstComma = text.find(',');
  std::string_view const verb = text.substr(0, firstComma);
  if (isUpstreamWord(verb)) { return {}; }

  TuneCommand out;
  if (verb == "emit") {
    out.verb = TuneVerb::Emit;
  } else if (verb == "reset") {
    out.verb = TuneVerb::Reset;
  } else if (verb == "adopt") {
    out.verb = TuneVerb::Adopt;
  } else if (verb == "compact") {
    out.verb = TuneVerb::Compact;
  } else if (verb == "scope") {
    out.verb = TuneVerb::Scope;
  } else if (verb == "accuracy") {
    out.verb = TuneVerb::Accuracy;
  } else if (verb == "status") {
    out.verb = TuneVerb::Status;
  } else {
    // Settings alone, or none: the first word is one of them rather than a subcommand.
    out.verb = TuneVerb::Run;
    if (text.empty()) { return out; }
  }

  bool const isRun = out.verb == TuneVerb::Run;
  std::string const who = isRun ? "-tune" : "-tune " + std::string{verb};
  bool const isAccuracy = out.verb == TuneVerb::Accuracy;
  bool const isStatus = out.verb == TuneVerb::Status;
  bool const takesScope = out.verb == TuneVerb::Scope || out.verb == TuneVerb::Emit || isRun || isAccuracy || isStatus;
  // A status values what a run would do next, so it takes whatever shapes the run's queue.
  bool const queues = isRun || isStatus;

  // Applied once the strategy they belong to is known, whichever order the settings come in.
  std::optional<u32> comboTop;
  std::optional<u32> comboTiers;

  size_t const settingsFrom = isRun ? 0 : firstComma == std::string_view::npos ? text.size() + 1 : firstComma + 1;
  for (size_t at = settingsFrom; at <= text.size();) {
    size_t const comma = text.find(',', at);
    std::string_view const token = text.substr(at, comma == std::string_view::npos ? comma : comma - at);
    at = comma == std::string_view::npos ? text.size() + 1 : comma + 1;

    size_t const eq = token.find('=');
    if (eq == std::string_view::npos) {
      throw who + ": '" + std::string{token} + "' is not " + (isRun ? "a subcommand or " : "") +
        "a <key>=<value> setting";
    }

    std::string_view const key = token.substr(0, eq);
    std::string_view const val = token.substr(eq + 1);

    // A run's env is the device it opens.
    bool const wantsEnv = out.verb != TuneVerb::Compact && !opensDevice(out.verb);

    if (key == "env" && wantsEnv) {
      out.env = asEnvId(val, "env=");
    } else if (key == "into" && out.verb == TuneVerb::Adopt) {
      out.env = asEnvId(val, "into=");
    } else if (key == "from" && out.verb == TuneVerb::Adopt) {
      out.from = asEnvId(val, "from=");
    } else if (key == "fft" && (out.verb == TuneVerb::Reset || isAccuracy)) {
      if (val.empty()) { throw who + ": fft= takes an FFT specification"; }
      out.fft = std::string{val};
    } else if (key == "groups" && isAccuracy) {
      out.groups = parseGroups(val);
    } else if (key == "workload" && takesScope) {
      parseWorkload(val, out.scope);
    } else if (key == "probe" && takesScope) {
      out.scope.probe = asExponent(val, "probe=");
    } else if (key == "probeWeight" && takesScope && !isAccuracy) {
      std::optional<double> const weight = parseNonNegative(val);
      if (!weight || *weight > 1) { throw std::string{"-tune: probeWeight= takes a fraction between 0 and 1"}; }
      out.scope.probeWeight = *weight;
    } else if (key == "kinds" && takesScope && !isAccuracy) {
      parseKinds(val, out.scope);
    } else if (key == "bootstrap" && queues) {
      if (val != "0" && val != "1") { throw std::string{"-tune: bootstrap= takes 0 or 1"}; }
      out.bootstrap = val == "1";
    } else if (key == "strategy" && queues) {
      out.strategy = parseStrategy(val);
    } else if (key == "stop" && queues) {
      out.stop = parseStop(val);
    } else if (key == "tunetxt" && (isRun || out.verb == TuneVerb::Emit)) {
      if (val != "0" && val != "1") { throw who + ": tunetxt= takes 0 or 1"; }
      out.tuneTxt = val == "1";
    } else if (key == "comboTop" && queues) {
      comboTop = parseInt<u32>(val);
      if (!comboTop || *comboTop < 1) { throw std::string{"-tune: comboTop= takes a count of 1 or more"}; }
    } else if (key == "comboTiers" && queues) {
      comboTiers = parseInt<u32>(val);
      if (!comboTiers || *comboTiers < 1 || *comboTiers > COMBO_TIERS) {
        throw std::string{"-tune: comboTiers= takes 1, 2 or 3"};
      }
    } else {
      std::string accepted = "nothing";
      switch (out.verb) {
      case TuneVerb::Emit:
        accepted = "env=<id>, tunetxt=0|1, and the run's workload=<lo>-<hi>, probe=<E>, probeWeight=<0..1>,"
                   " kinds=prp|ll|prp+ll";
        break;
      case TuneVerb::Reset: accepted = "env=<id>, fft=<spec>"; break;
      case TuneVerb::Adopt: accepted = "into=<id> (or env=<id>), from=<id>"; break;
      case TuneVerb::Compact: break;
      case TuneVerb::Scope:
        accepted = "workload=<lo>-<hi>, probe=<E>, probeWeight=<0..1>, kinds=prp|ll|prp+ll, env=<id>";
        break;
      case TuneVerb::Run:
        accepted = "workload=<lo>-<hi>, probe=<E>, probeWeight=<0..1>, kinds=prp|ll|prp+ll, bootstrap=0|1,"
                   " strategy=hybrid|single|groups|permute:<KEY>+<KEY>..., comboTop=<N>, comboTiers=1|2|3, stop=<P>%|0,"
                   " tunetxt=0|1,"
                   " or a subcommand: emit, reset, adopt, compact, scope, status, accuracy";
        break;
      case TuneVerb::Status:
        accepted = "env=<id>, and a run's workload=<lo>-<hi>, probe=<E>, probeWeight=<0..1>, kinds=prp|ll|prp+ll,"
                   " bootstrap=0|1, strategy=<S>, comboTop=<N>, comboTiers=1|2|3, stop=<P>%|0";
        break;
      case TuneVerb::Accuracy: accepted = "workload=<lo>-<hi>, probe=<E>, fft=<spec>, groups=<Group>+<Group>..."; break;
      }
      throw who + ": '" + std::string{key} + "=' is not understood. Accepted: " + accepted;
    }

    if (isStatus && key != "env") { out.settings += (out.settings.empty() ? "" : ",") + std::string{token}; }
  }

  if (comboTop || comboTiers) {
    if (out.strategy.kind != Strategy::Kind::Hybrid) {
      throw "-tune: comboTop= and comboTiers= combine the groups of strategy=hybrid, not of strategy=" +
        out.strategy.text();
    }
    out.strategy.comboTop = comboTop.value_or(COMBO_TOP);
    out.strategy.comboTiers = comboTiers.value_or(COMBO_TIERS);
  }

  // Both named and contradictory is a mistyped command rather than a scope: caught here, so that it is refused
  // before anything else happens.  A probe named against a range derived from the worktodo is not contradictory --
  // it says where the user is going, and makeScope widens the range to it.
  if (takesScope && out.scope.lo && out.scope.probe &&
      (out.scope.probe < out.scope.lo || out.scope.probe > out.scope.hi)) {
    throw "-tune: probe=" + std::to_string(out.scope.probe) + " is outside workload=" + std::to_string(out.scope.lo) +
      "-" + std::to_string(out.scope.hi);
  }

  return out;
}

std::string runSettings(const RunScope& scope, const TuneCommand& command) {
  std::string kinds;
  for (TestKind const kind : command.scope.kinds) { kinds += (kinds.empty() ? "" : "+") + std::string{toString(kind)}; }

  std::string out = "workload=" + std::to_string(scope.lo) + "-" + std::to_string(scope.hi) +
    ",probe=" + std::to_string(scope.probe) + ",probeWeight=" + shortest(scope.probeWeight) + ",kinds=" + kinds +
    ",bootstrap=" + (command.bootstrap ? "1" : "0") + ",strategy=" + command.strategy.text();
  if (command.strategy.kind == Strategy::Kind::Hybrid) {
    out += ",comboTop=" + std::to_string(command.strategy.comboTop) +
      ",comboTiers=" + std::to_string(command.strategy.comboTiers);
  }
  return out + ",stop=" + (command.stop > 0 ? shortest(command.stop * 100) + "%" : "0");
}

std::vector<WorkRow> workRows(u32 sess, const std::vector<PendingWork>& pending) {
  std::map<std::pair<TestKind, u64>, u32> counts;
  for (const PendingWork& work : pending) { ++counts[{work.kind, work.exponent}]; }

  std::vector<WorkRow> out;
  for (const auto& [at, count] : counts) {
    out.push_back({.sess = sess, .kind = at.first, .exponent = at.second, .count = count});
  }
  return out;
}

std::vector<PendingWork> pendingOf(const TuneDB& db, u32 sess) {
  std::vector<PendingWork> out;
  for (const WorkRow& row : db.works()) {
    if (row.sess != sess) { continue; }
    out.insert(out.end(), row.count, PendingWork{.kind = row.kind, .exponent = row.exponent});
  }
  return out;
}

std::vector<PendingWork> statusWork(const TuneDB& db, const SessRow* run, const Args& args, const fs::path& dir) {
  return run ? pendingOf(db, run->id) : scanWorktodo(worktodoFiles(args, dir));
}

std::string statusSettings(std::string_view run, std::string_view asked) {
  std::set<std::string_view> named;
  for (std::string_view const token : settingTokens(asked)) { named.insert(settingKey(token)); }
  bool const rescoped = named.contains("workload") || named.contains("probe");
  bool const restrategied = named.contains("strategy");

  std::string out;
  auto add = [&out](std::string_view token) { out += (out.empty() ? "" : ",") + std::string{token}; };
  for (std::string_view const token : settingTokens(run)) {
    std::string_view const key = settingKey(token);
    if (named.contains(key) || (rescoped && (key == "workload" || key == "probe")) ||
        (restrategied && (key == "comboTop" || key == "comboTiers"))) {
      continue;
    }
    add(token);
  }
  for (std::string_view const token : settingTokens(asked)) { add(token); }
  return out;
}

u32 commandEnv(const TuneDB& db, const TuneCommand& command, u64 build) {
  if (command.env) {
    if (!db.findEnv(command.env)) {
      log("tune: there is no env %u in the database\n", command.env);
      return 0;
    }
    return command.env;
  }

  std::vector<const DbEnv*> matching;
  std::vector<const DbEnv*> all;
  for (const DbEnv& env : db.envs()) {
    all.push_back(&env);
    if (env.build == build) { matching.push_back(&env); }
  }

  if (matching.size() == 1) { return matching.front()->id; }

  if (matching.empty()) {
    log("tune: nothing in the database was measured against the kernels this binary carries (%s), so there is no env"
        " to work on; name one with env=<id>, or adopt it into one\n",
        hex16(build).c_str());
    listEnvs(all);
  } else {
    log("tune: %zu envs were measured against the kernels this binary carries, and nothing here opens a device to"
        " tell their cards apart; name one with env=<id>\n",
        matching.size());
    listEnvs(matching);
  }

  return 0;
}

bool rewriteFor(TuneDB& db, const TuneCommand& command, u32 env) {
  switch (command.verb) {
  case TuneVerb::Emit: return true;

  // None rewrites anything, and they are here only so that the switch is complete.
  case TuneVerb::Scope:
  case TuneVerb::Status:
  case TuneVerb::Run:
  case TuneVerb::Accuracy: return false;

  case TuneVerb::Compact: return db.compact();

  case TuneVerb::Reset: return db.reset(env, command.fft);

  case TuneVerb::Adopt: {
    u32 const from = command.from ? command.from : db.adoptCandidate(env);
    if (!from) {
      log("tune: env %u has no earlier env of the same card to adopt; name one with from=<id>\n", env);
      return false;
    }
    if (from == env) {
      log("tune: env %u cannot adopt itself\n", env);
      return false;
    }
    if (!db.adopt(from, env)) { return false; }

    log("tune: env %u has taken over the rows of env %u\n", env, from);
    return true;
  }
  }

  return false;
}

namespace {

// Where the database in `dir` stands.  Read without the lock, so that a run holding it can be asked about, and taking
// only the lines already complete, since that run may be part-way through appending one.
[[nodiscard]] bool reportStatus(const TuneCommand& command, const Args& args, const fs::path& dir) {
  fs::path const dbPath = dir / TuneDB::DEFAULT_NAME;
  std::error_code ec;
  if (!fs::exists(dbPath, ec)) {
    log("tune: status: nothing has been measured in %s, which holds no %s\n", dir.string().c_str(),
        TuneDB::DEFAULT_NAME);
    return true;
  }

  Activity activity{.held = TuneDB::writerHolds(dbPath), .written = 0, .now = u64(std::time(nullptr))};
  if (auto const when = fs::last_write_time(dbPath, ec); !ec) {
    auto const since = std::chrono::file_clock::to_sys(when).time_since_epoch();
    activity.written = u64(std::chrono::duration_cast<std::chrono::seconds>(since).count());
  }

  std::string text;
  {
    File file = File::openRead(dbPath);
    if (!file) {
      log("Can't read '%s'\n", dbPath.string().c_str());
      return false;
    }
    text = file.readAll();
  }
  TuneDB db;
  if (!db.parse(completeLines(text), dbPath.string())) { return false; }

  u32 const env = commandEnv(db, command, buildFingerprint());
  if (!env) { return false; }
  const DbEnv& row = *db.findEnv(env);

  const SessRow* run = nullptr;
  for (const SessRow& s : db.sessions()) {
    if (s.env == env && !s.tune.empty() && (!run || s.id > run->id)) { run = &s; }
  }

  std::string const word = statusSettings(run ? run->tune : "", command.settings);
  std::optional<TuneCommand> settings;
  try {
    settings = parseTuneCommand(word);
  } catch (const std::string& why) {
    log("tune: status: '%s' is not a run's settings: %s\n", word.c_str(), why.c_str());
    return false;
  }
  if (!settings || settings->verb != TuneVerb::Run) {
    log("tune: status: '%s' is not a run's settings\n", word.c_str());
    return false;
  }

  std::vector<PendingWork> const pending = statusWork(db, run, args, dir);
  RunScope const scope = makeScope(settings->scope, pending);
  Scheduler const scheduler = schedulerFor(row.toEnv(), scope, *settings, args.blockSize);

  std::string const work = std::to_string(pending.size()) + (pending.size() == 1 ? " assignment" : " assignments");
  std::string const whose = !run
    ? "a run started here now, there being none on env " + std::to_string(env) + ", over the " + work + " pending"
    : "session " + std::to_string(run->id) + "'s run" + (command.settings.empty() ? "" : " with " + command.settings) +
      ", over the " + work + " pending when it started";
  logStatus(statusOf(scheduler, db, env, settings->stop, activity), row, whose + ": " + runSettings(scope, *settings));
  return true;
}

}  // namespace

bool runTuneCommand(const TuneCommand& command, const Args& args, const fs::path& dir) {
  if (command.verb == TuneVerb::Status) { return reportStatus(command, args, dir); }

  fs::path const dbPath = dir / TuneDB::DEFAULT_NAME;

  std::vector<fs::path> files;
  std::optional<RunScope> scope;
  if (command.verb == TuneVerb::Scope || command.verb == TuneVerb::Emit) {
    files = worktodoFiles(args, dir);
    scope = makeScope(command.scope, scanWorktodo(files));

    // Nothing to read, so nothing to lock: a scope checked before the first run leaves the directory as it found it.
    if (command.verb == TuneVerb::Scope && !command.env && !fs::exists(dbPath)) {
      reportScope(*scope, files, Objective{Env{}, *scope}, "with nothing measured");
      return true;
    }
  }

  TuneDB db;
  // Before the load and held for the rest of the command: a session appending beside this would be writing rows into
  // a file about to be rewritten from what was read here.
  if (!db.lockForWriting(dbPath)) { return false; }
  if (!db.load(dbPath)) { return false; }

  u32 env = 0;
  if (command.verb != TuneVerb::Compact) {
    env = commandEnv(db, command, buildFingerprint());
    if (!env) { return false; }
  }

  if (command.verb == TuneVerb::Scope) {
    reportScope(*scope, files, Objective{db, env, *scope, defaultsOf(db, env, *scope)},
                "against env " + std::to_string(env));
    return true;
  }

  if (command.verb == TuneVerb::Emit) {
    Provenance const from{.ts = u64(time(nullptr)), .db = TuneDB::DEFAULT_NAME, .env = env};

    std::optional<SelectionFile> const file = emit(db, defaultsOf(db, env, *scope), from);
    if (!file) {
      log("tune: emit: the database gave nothing that could be published\n");
      return false;
    }

    fs::path const out = dir / SELECTION_NAME;
    writeSelection(out, *file);
    size_t const excluded = file->excluded.size();
    std::string const exclusions =
      excluded ? " and " + std::to_string(excluded) + (excluded == 1 ? " exclusion" : " exclusions") : "";
    log("tune: published %zu %s%s of env %u to %s\n", file->entries.size(),
        file->entries.size() == 1 ? "entry" : "entries", exclusions.c_str(), env, out.string().c_str());

    if (command.tuneTxt) {
      const DbEnv* const row = db.findEnv(env);
      fs::path const compat = dir / TUNE_TXT_NAME;
      size_t const lines = writeCompatibility(compat, *file, row ? row->toEnv() : Env{});
      log("tune: wrote %zu %s to %s for binaries that read no %s\n", lines, lines == 1 ? "line" : "lines",
          compat.string().c_str(), SELECTION_NAME);
    }
    return true;
  }

  if (!rewriteFor(db, command, env)) { return false; }

  db.save(dbPath);
  log("tune: %s rewrote %s\n", toString(command.verb), dbPath.string().c_str());
  return true;
}

MeasureOutcome runTune(const GpuCommon& shared, const TuneCommand& command) {
  if (command.verb == TuneVerb::Accuracy) { return runAccuracy(shared, command); }

  Args& args = *shared.args;
  fs::path const dir = fs::current_path();

  // Before anything is built, so that every measurement depends on the built-in defaults and the tuner's own choices.
  Takeover const takeover = takeOverConfig(args);
  log("tune: %s\n", describe(takeover).c_str());
  if (!args.fftSpec.empty()) { log("tune: -fft is not used by the tuner, which chooses among every FFT itself\n"); }

  std::vector<fs::path> const files = worktodoFiles(args, dir);
  std::vector<PendingWork> const pending = scanWorktodo(files);
  RunScope const scope = makeScope(command.scope, pending);
  Env const env = detectEnv(*shared.context, args);

  fs::path const dbPath = dir / TuneDB::DEFAULT_NAME;
  TuneDB db;
  // Before the load, and held for the rest of the run: every id this writes is allocated from what it read.
  if (!db.lockForWriting(dbPath) || !db.load(dbPath)) { return MeasureOutcome::Failed; }
  db.attach(dbPath);

  Session session{shared, db, env};
  if (!session.begin(scope.probe, runSettings(scope, command))) {
    log("tune: '%s' would not take a session\n", dbPath.string().c_str());
    return MeasureOutcome::Failed;
  }
  for (const WorkRow& row : workRows(session.id(), pending)) {
    if (!db.add(row)) {
      log("tune: the pending work was not recorded, so -tune status cannot weight this run as it does\n");
      break;
    }
  }
  if (u32 const gen = restart::generation()) { log("tune: generation %u\n", gen); }

  u32 const envId = session.envId();
  reportScope(scope, files, Objective{db, envId, scope}, "against env " + std::to_string(envId));

  Scheduler scheduler = schedulerFor(env, scope, command, args.blockSize);
  std::string const combo = command.strategy.kind == Strategy::Kind::Hybrid
    ? " (comboTop=" + std::to_string(command.strategy.comboTop) +
      ", comboTiers=" + std::to_string(command.strategy.comboTiers) + ")"
    : "";
  char until[96];
  if (command.stop > 0) {
    snprintf(until, sizeof(until), "until nothing is expected to lower T by %g%% of it", command.stop * 100);
  } else {
    snprintf(until, sizeof(until), "until stopped (stop=0)");
  }
  log("tune: %zu entries could serve the workload; each measured one is searched by strategy=%s%s, then by random "
      "restarts; the run goes on %s\n",
      scheduler.baselines().size(), command.strategy.text().c_str(), combo.c_str(), until);
  if (command.bootstrap) {
    std::string names;
    for (const Family& f : scheduler.bootstrap().families()) {
      names += (names.empty() ? "" : ", ") + std::string{typeName(f.type)} + " " + f.fft.spec();
    }
    log("tune: bootstrap at %" PRIu64 " over %s\n", scope.probe, names.empty() ? "no family" : names.c_str());
  }

  fs::path const out = dir / SELECTION_NAME;
  std::optional<fs::path> const compat =
    command.tuneTxt ? std::optional<fs::path>{dir / TUNE_TXT_NAME} : std::optional<fs::path>{};
  auto publishNow = [&](const Objective& objective, const Defaults& defaults) {
    Provenance const from{.ts = u64(time(nullptr)),
                          .db = TuneDB::DEFAULT_NAME,
                          .env = envId,
                          .T = objective.T(),
                          .workloadLo = scope.lo,
                          .workloadHi = scope.hi};
    if (!publish(out, db, defaults, from, compat)) { log("tune: %s could not be published\n", out.string().c_str()); }
  };

  SessionBench bench{session, args.blockSize};
  QueueReport const report = [&] {
    // Gone before the summary, which is the run's last word and belongs on a terminal with nothing drawn over it.
    RunView view{liveTerminal()};
    return runQueue(scheduler, db, envId, bench, publishNow, command.stop, &view);
  }();

  logSummary(summarize(scheduler, db, envId, session.id(), report, command.stop));
  log("tune: T %.3f -> %.3f us/it%s\n", report.startT, report.endT,
      report.stopped ? ", stopped before the queue was done" : "");
  log("tune: published %s%s\n", out.string().c_str(), compat ? (" and " + compat->string()).c_str() : "");
  if (!takeover.configKeys.empty()) {
    std::string keys;
    for (const std::string& k : takeover.configKeys) { keys += (keys.empty() ? "" : ", ") + k; }
    log("tune: the config files set %s, which production puts ahead of %s; remove them to run what was tuned\n",
        keys.c_str(), out.filename().string().c_str());
  }

  if (session.deviceLost()) { return MeasureOutcome::DeviceLost; }
  session.end();
  return session.cannotRecord() ? MeasureOutcome::Failed : MeasureOutcome::Ok;
}

}  // namespace tune
