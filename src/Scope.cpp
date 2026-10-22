// Copyright (C) Jason Lynch

#include "Scope.h"

#include "Args.h"
#include "FFTVariants.h"
#include "Primes.h"
#include "Task.h"
#include "TuneDB.h"
#include "Worktodo.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>

namespace tune {

namespace {

[[nodiscard]] std::string outsideBand(const char* what, const std::string& shown) {
  return "-tune: " + std::string{what} + shown + " is outside the exponents this can work with (" +
    std::to_string(MIN_EXPONENT) + " to " + std::to_string(MAX_EXPONENT) + ")";
}

void requireExponent(u64 E, const char* what) {
  if (E < MIN_EXPONENT || E > MAX_EXPONENT) { throw outsideBand(what, std::to_string(E)); }
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

// Spread evenly across the range whatever is pending: a worktodo is a few days of the months of work a tune serves, so
// weighting its exponents would tune for this week's assignments and leave the rest of the range unmeasured.
[[nodiscard]] Grid gridFor(TestKind kind, u64 lo, u64 hi, u64 probe, double probeWeight) {
  Grid out{.kind = kind, .points = {}};

  std::map<u64, double> weights;
  std::vector<u64> const spread = spreadOver(lo, hi);
  for (u64 const E : spread) { weights[E] += (1 - probeWeight) / double(spread.size()); }
  weights[probe] += probeWeight;

  for (const auto& [E, weight] : weights) { out.points.push_back({.exponent = E, .weight = weight}); }
  return out;
}

}  // namespace

bool ScopeArgs::wantsKind(TestKind kind) const { return std::ranges::find(kinds, kind) != kinds.end(); }

// An exponent with the K/M/G suffix the ranges are usually written with.
u64 parseExponent(std::string_view text, const char* what) {
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

double Grid::weight(u64 E) const {
  auto const at = std::ranges::find(points, E, &GridPoint::exponent);
  return at == points.end() ? 0 : at->weight;
}

const Grid* RunScope::grid(TestKind kind) const {
  auto const at = std::ranges::find_if(grids, [kind](const Grid& g) { return g.kind == kind; });
  return at == grids.end() ? nullptr : &*at;
}

bool RunScope::admits(const FFTShape& shape) const {
  if (!shape.isPfa()) { return true; }
  FFTParts const parts = fftParts(shape.fft_type);
  bool const hybrid = parts.fp64 || parts.fp32;
  return pfaShapes == PfaShapes::All || (pfaShapes == PfaShapes::Ntt && !hybrid);
}

const char* toString(PfaShapes shapes) {
  switch (shapes) {
  case PfaShapes::None: return "none";
  case PfaShapes::Ntt: return "ntt";
  case PfaShapes::All: return "all";
  }
  return "?";
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
  out.pfaShapes = args.pfaShapes;

  if (args.lo) {
    out.lo = args.lo;
    out.hi = args.hi;
    out.rangeSource = "named on the command line";
  } else if (!wanted.empty()) {
    auto const [lo, hi] = std::ranges::minmax(wanted);
    out.lo = u64(double(lo) * (1 - WORKLOAD_PAD));
    out.hi = u64(double(hi) * (1 + WORKLOAD_AHEAD));
    out.rangeSource = std::to_string(wanted.size()) + (wanted.size() == 1 ? " assignment" : " assignments") +
      " pending, from " + std::to_string(u32(WORKLOAD_PAD * 100)) + "% below to " +
      std::to_string(u32(WORKLOAD_AHEAD * 100)) + "% above";
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
    out.grids.push_back(gridFor(kind, out.lo, out.hi, out.probe, out.probeWeight));
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

}  // namespace tune
