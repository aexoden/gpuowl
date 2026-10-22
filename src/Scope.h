// Copyright (C) Jason Lynch

// What a tuning run works within: the exponent range the user's work covers, the one exponent within it that matters
// most, and the grid of exponents the objective is summed over, one per test kind.  Everything the tuner values is
// weighted by them, so that a configuration nobody will run is never paid for.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "TestKind.h"

#include <filesystem>
#include <string>
#include <vector>

class Args;

namespace tune {

// The range used where there is no pending work to derive one from.
inline constexpr u64 DEFAULT_WORKLOAD_LO = 100'000'000;
inline constexpr u64 DEFAULT_WORKLOAD_HI = 400'000'000;

// How far past the pending work a derived range reaches: below the lowest, and above the highest.  A worktodo holds a
// few days of work, and a tune serves months of assignments, which move upward.
inline constexpr double WORKLOAD_PAD = 0.05;
inline constexpr double WORKLOAD_AHEAD = 0.25;

// How much of the workload weight sits on the probe exponent alone: 0 optimises the range evenly, 1 exactly the
// exponent the user runs.
inline constexpr double PROBE_WEIGHT = 0.5;

// The exponents a scope can be expressed in: upstream's own worktodo floor, and the ten billion Primes.h is sized for.
inline constexpr u64 MIN_EXPONENT = 1000;
inline constexpr u64 MAX_EXPONENT = 10'000'000'000;

// Log-spaced grid points across the range, besides the probe.
inline constexpr u32 GRID_POINTS = 64;

// The width of the bins the worktodo's most populated one is found among, relative to the exponent.
inline constexpr double PROBE_BIN = 0.02;

// Which prime-factor middles (MIDDLE 3, 7, 9 or 11 times a power of two) a run considers: the NTT types', every
// type's, or none.  A hybrid's runs out of place with a two-kernel FP tail, which rarely makes it the fastest.
enum class PfaShapes : u8 { None, Ntt, All };

[[nodiscard]] const char* toString(PfaShapes shapes);

// What the settings ask for; zero where a setting is to be derived from the pending work.
struct ScopeArgs {
  u64 lo = 0;
  u64 hi = 0;
  u64 probe = 0;
  double probeWeight = PROBE_WEIGHT;
  std::vector<TestKind> kinds{TestKind::PRP};
  PfaShapes pfaShapes = PfaShapes::Ntt;

  [[nodiscard]] bool wantsKind(TestKind kind) const;
};

struct PendingWork {
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  [[nodiscard]] bool operator==(const PendingWork&) const = default;
};

struct GridPoint {
  u64 exponent = 0;
  double weight = 0;
  [[nodiscard]] bool operator==(const GridPoint&) const = default;
};

// One test kind's points, ascending, with weights summing to 1.
struct Grid {
  TestKind kind = TestKind::PRP;
  std::vector<GridPoint> points;

  // The weight standing on `E`; 0 for an exponent not on the grid.
  [[nodiscard]] double weight(u64 E) const;
};

struct RunScope {
  u64 lo = 0;
  u64 hi = 0;
  std::string rangeSource{};
  u64 probe = 0;
  std::string probeSource{};
  double probeWeight = PROBE_WEIGHT;
  std::vector<Grid> grids;
  PfaShapes pfaShapes = PfaShapes::Ntt;

  [[nodiscard]] const Grid* grid(TestKind kind) const;

  // Whether a run considers `shape` at all.
  [[nodiscard]] bool admits(const FFTShape& shape) const;
};

// The scope the settings and the pending work imply.  Pure.  Throws a message for settings that describe no scope.
[[nodiscard]] RunScope makeScope(const ScopeArgs& args, const std::vector<PendingWork>& pending);

// The worktodo files a run in `dir` draws work from, those that exist: one per worker, the file a single-instance run
// leaves beside them, and the pool's own where there is a pool.
[[nodiscard]] std::vector<fs::path> worktodoFiles(const Args& args, const fs::path& dir);

// The pending work those files hold.  A Cert is PRP work: it runs the PRP kernels.
[[nodiscard]] std::vector<PendingWork> scanWorktodo(const std::vector<fs::path>& files);

// An exponent with the K, M or G suffix ranges are usually written with, checked against the band above.  Throws a
// message naming `what` for anything else.
[[nodiscard]] u64 parseExponent(std::string_view text, const char* what);

}  // namespace tune
