// Copyright (C) Jason Lynch

// The -tune command line, and the scope a tuning run works within.
//
// Five of its subcommands open no device, build no kernels and take no readings: four read and rewrite the measurement
// database, and the fifth reports the scope. They run wherever the files are rather than only on the card that was
// measured.
//
// The scope is two things: the exponent range the user's work covers, and the one exponent within it that matters
// most. Everything downstream is weighted by them, so a configuration nobody will run is never paid for.

#pragma once

#include "common.h"
#include "UseResolve.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class Args;

namespace tune {

class TuneDB;

// The exponent range used where there is no worktodo to derive one from: the span current Mersenne work covers.
inline constexpr u64 DEFAULT_WORKLOAD_LO = 100'000'000;
inline constexpr u64 DEFAULT_WORKLOAD_HI = 400'000'000;

// How far past the pending work a derived range reaches, at each end. The point is to cover the assignments that will
// arrive as well as the ones in hand.
inline constexpr double WORKLOAD_PAD = 0.05;

// How much of the workload weight sits on the probe exponent alone. 0 optimises the range evenly, 1 optimises exactly
// the exponent the user runs.
inline constexpr double PROBE_WEIGHT = 0.5;

// The exponents a scope can be expressed in. The floor is upstream's own worktodo floor; the ceiling is the ten billion
// Primes.h is sized for, about twice what the largest FFT reaches.
inline constexpr u64 MIN_EXPONENT = 1000;
inline constexpr u64 MAX_EXPONENT = 10'000'000'000;

// Grid points across the range where there is no worktodo to take them from.
inline constexpr u32 GRID_POINTS = 64;

// The width of the bins the worktodo mode is the most populated of, as a fraction of the exponent.
inline constexpr double PROBE_BIN = 0.02;

enum class TuneVerb : u8 {
  Emit,     // publish the selection file the database supports
  Reset,    // drop what was measured on an env, or on one shape of it
  Adopt,    // restamp another env's rows as this one's
  Compact,  // fold duplicate rows and drop the option sets nothing names
  Scope,    // report the range, the probe and the grid a tuning run would work within
};

[[nodiscard]] const char* toString(TuneVerb verb);

// What `-tune` was asked to work within. Zero where the setting was not given and is to be derived from the worktodo.
struct ScopeArgs {
  u64 lo = 0;
  u64 hi = 0;
  u64 probe = 0;
  double probeWeight = PROBE_WEIGHT;
  std::vector<TestKind> kinds{TestKind::PRP};

  [[nodiscard]] bool wantsKind(TestKind kind) const;
};

struct TuneCommand {
  TuneVerb verb = TuneVerb::Emit;

  // The env the command acts on; 0 to take the one this binary's kernels were measured on.
  u32 env = 0;

  // `adopt` only: the env whose rows are being taken over.  0 to take the most recent env that is this card under
  // other kernels.
  u32 from = 0;

  // `reset` only: one shape of the env rather than all of it.
  std::string fft;

  // `scope` only.
  ScopeArgs scope;
};

// One pending assignment, reduced to what the scope cares about.
struct PendingWork {
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;

  [[nodiscard]] bool operator==(const PendingWork&) const = default;
};

// One exponent `T` is summed over, and the share of the workload weight standing on it.
struct GridPoint {
  u64 exponent = 0;
  double weight = 0;

  [[nodiscard]] bool operator==(const GridPoint&) const = default;
};

// The grid of one test kind: its points in ascending order, with weights summing to 1.
struct Grid {
  TestKind kind = TestKind::PRP;

  // False where the kind had no pending work and the points are spread across the range instead.
  bool fromWorktodo = false;

  std::vector<GridPoint> points;

  [[nodiscard]] double weight(u64 E) const;
};

// The range, the probe, and one grid per test kind, each recording where it came from so that a run can say what it
// is optimising before it spends anything on it.  Not `Scope`, which an option already has one of.
struct RunScope {
  u64 lo = 0;
  u64 hi = 0;
  std::string rangeSource;

  u64 probe = 0;
  std::string probeSource;
  double probeWeight = PROBE_WEIGHT;

  std::vector<Grid> grids;

  [[nodiscard]] const Grid* grid(TestKind kind) const;
};

// The scope the settings and the pending work imply. Pure: every file has been read by the time it is called, and the
// same arguments give the same answer on any machine. Throws a message for settings that cannot describe a scope.
[[nodiscard]] RunScope makeScope(const ScopeArgs& args, const std::vector<PendingWork>& pending);

// The worktodo files a run in `dir` would draw work from, those that exist and no others: one per worker, the file a
// single-instance run leaves beside them, and the pool's own where there is a pool.
[[nodiscard]] std::vector<fs::path> worktodoFiles(const Args& args, const fs::path& dir);

// The pending work those files hold. A Cert is PRP work: it runs the PRP kernels, so a separate grid for it would say
// what the PRP grid already says.
[[nodiscard]] std::vector<PendingWork> scanWorktodo(const std::vector<fs::path>& files);

// What the scope is, in the log, before anything is spent against it.
void reportScope(const RunScope& scope, const std::vector<fs::path>& files);

// The device-free subcommand `text` asks for, or nothing where it asks for something else -- upstream's own tuner
// takes the same flag, and its option words are not these.  Throws a message for a subcommand it recognises and then
// cannot read, so a mistyped setting is a usage error rather than a silent fall-through to the other tuner.
[[nodiscard]] std::optional<TuneCommand> parseTuneCommand(std::string_view text);

// Which env the command runs on: the one it names, or the single env whose rows were measured against `build`.  0
// where there is no such env or more than one, having said which envs there are -- with no device open there is
// nothing here that can tell two cards apart, and publishing the wrong card's measurements is worse than asking.
[[nodiscard]] u32 commandEnv(const TuneDB& db, const TuneCommand& command, u64 build);

// Everything `command` changes about a loaded database, `emit` aside -- that one reads.  False with the reason logged.
[[nodiscard]] bool rewriteFor(TuneDB& db, const TuneCommand& command, u32 env);

// Runs one device-free subcommand against the files in `dir`, reporting whether it did what it was asked.
[[nodiscard]] bool runTuneCommand(const TuneCommand& command, const Args& args, const fs::path& dir);

}  // namespace tune
