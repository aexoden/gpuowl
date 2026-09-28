// Copyright (C) Jason Lynch

// The -tune command line, the scope a tuning run works within, and the run itself.
//
// Six of its subcommands open no device, build no kernels and take no readings: four read and rewrite the measurement
// database, one reports the scope, and one where the database stands. They run wherever the files are rather than only
// on the card that was measured.  `accuracy` reads which option keys change the rounding, on the device.  Everything
// else -- settings alone, or nothing at all -- is a tuning run on the device.
//
// The scope is two things: the exponent range the user's work covers, and the one exponent within it that matters
// most. Everything downstream is weighted by them, so a configuration nobody will run is never paid for.

#pragma once

#include "common.h"
#include "Probe.h"
#include "UseResolve.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class Args;
class GpuCommon;

namespace tune {

enum class MeasureOutcome : u8;

class Objective;
class TuneDB;
struct SessRow;
struct WorkRow;

// The exponent range used where there is no worktodo to derive one from: the span current Mersenne work covers.
inline constexpr u64 DEFAULT_WORKLOAD_LO = 100'000'000;
inline constexpr u64 DEFAULT_WORKLOAD_HI = 400'000'000;

// How far past the pending work a derived range reaches: below the lowest, and above the highest.  A worktodo holds a
// few days of work, and a tune serves for months of assignments, which move upward.
inline constexpr double WORKLOAD_PAD = 0.05;
inline constexpr double WORKLOAD_AHEAD = 0.25;

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

// A run stops once nothing it could measure is expected to lower T by this fraction of it.  Measured against what is
// left to gain rather than against a clock, so that a run ends where the next hour would buy little, however long the
// last one took.
inline constexpr double STOP = 0.001;

enum class TuneVerb : u8 {
  Emit,      // publish the selection file the database supports
  Reset,     // drop what was measured on an env, or on one shape of it
  Adopt,     // restamp another env's rows as this one's
  Compact,   // fold duplicate rows and drop the option sets nothing names
  Scope,     // report the range, the probe and the grid a tuning run would work within
  Run,       // tune: measure what is worth measuring, publishing after every item
  Accuracy,  // read which option keys change the rounding
  Status,    // report where the database stands: what is measured, what is not, and what a run would take next
};

[[nodiscard]] const char* toString(TuneVerb verb);

// Whether the verb builds kernels on a device; the others work on the files alone.
[[nodiscard]] bool opensDevice(TuneVerb verb);

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

  // `reset`: one shape of the env rather than all of it.  `accuracy`: the one FFT to read rather than each family's.
  std::string fft;

  // `accuracy` only: the groups whose keys are read; empty for every group.
  std::vector<Group> groups;

  // `scope`, `run` and `emit`: the scope the bootstrap's races were run at the probe of.
  ScopeArgs scope;

  // `run` only: false to leave every family at the built-in defaults rather than racing its options first.
  bool bootstrap = true;

  // `run` only: what counts as one step from an entry's best option set.
  Strategy strategy{};

  // `run` only: how the search is first spread over the entries worth searching.
  Halving halving{.contenders = CONTENDERS, .roundCalls = ROUND_CALLS};

  // `run` only: the fraction of T an item has to be expected to remove to be worth running; 0 to run until stopped.
  double stop = STOP;

  // `run` and `emit`: whether to write the tune.txt an older binary reads, beside every selection file published.
  bool tuneTxt = false;

  // `run` only: a full-screen view of the run in place of the lines it writes to a terminal.  Display only: not part
  // of the settings a run records, and the log file is the same either way.
  bool dashboard = false;

  // `status`: the settings named, as a run takes them, which replace those of the latest run.
  std::string settings;
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

// What the scope is, in the log, before anything is spent against it, and what the objective over it stands at.
// `against` says what that objective was taken over.
void reportScope(const RunScope& scope, const std::vector<fs::path>& files, const Objective& objective,
                 const std::string& against);

// How far `strategy` searches from one best set of each of `ffts` on `env`, as the log says it: a line per FFT, the
// steps of each group and what they come to, and with `groups` a line per group, its bins and what each offers.
[[nodiscard]] std::vector<std::string> searchReport(const Env& env, const std::vector<FFTConfig>& ffts,
                                                    const Strategy& strategy, bool groups);

// What `text` asks this tuner for.  Throws a message for anything it cannot read, naming -oldtune for an option word of
// upstream's own tuner, which took this flag before this one did.
[[nodiscard]] TuneCommand parseTuneCommand(std::string_view text);

// A run's settings as one word a run takes: its scope resolved, and every other setting named, so that it values its
// items the same way whatever the worktodo says by then.
[[nodiscard]] std::string runSettings(const RunScope& scope, const TuneCommand& command);

// `pending`, as the rows session `sess` records it by: one per kind and exponent, with how many assignments it has.
[[nodiscard]] std::vector<WorkRow> workRows(u32 sess, const std::vector<PendingWork>& pending);

// The pending work session `sess` recorded, as makeScope() takes it.
[[nodiscard]] std::vector<PendingWork> pendingOf(const TuneDB& db, u32 sess);

// The pending work a status weights `run` by: what it recorded when it started, not what the worktodo in `dir` says
// now; the worktodo only where there is no run.
[[nodiscard]] std::vector<PendingWork> statusWork(const TuneDB& db, const SessRow* run, const Args& args,
                                                  const fs::path& dir);

// A run's settings word with `asked`'s settings in place of its own.  A range or a probe replaces both, since the run's
// were resolved together, and a strategy the combination settings of the run's.
[[nodiscard]] std::string statusSettings(std::string_view run, std::string_view asked);

// Which env the command runs on: the one it names, or the single env whose rows were measured against `build`.  0
// where there is no such env or more than one, having said which envs there are -- with no device open there is
// nothing here that can tell two cards apart, and publishing the wrong card's measurements is worse than asking.
[[nodiscard]] u32 commandEnv(const TuneDB& db, const TuneCommand& command, u64 build);

// The env `adopt` folds into: the one named; else, for a from= env, that card's env under `build`; else the single env
// measured against `build`.  Where no env was measured against `build` -- straight after an update, before anything
// has been timed with the new kernels -- the card's env under them is added to `db` from the env being adopted, which
// is from= or, where the database holds a single card, its most recent env.  0 with the reason logged.
[[nodiscard]] u32 adoptTarget(TuneDB& db, const TuneCommand& command, u64 build);

// Everything `command` changes about a loaded database, `emit` aside -- that one reads.  False with the reason logged.
[[nodiscard]] bool rewriteFor(TuneDB& db, const TuneCommand& command, u32 env);

// Runs one device-free subcommand against the files in `dir`, reporting whether it did what it was asked.
[[nodiscard]] bool runTuneCommand(const TuneCommand& command, const Args& args, const fs::path& dir);

// The tuning run, in the current directory: every -use setting the config files and the command line make is set
// aside, and the queue measures what is worth measuring until nothing is or it is stopped, publishing the selection
// file after every item.
[[nodiscard]] MeasureOutcome runTune(const GpuCommon& shared, const TuneCommand& command);

}  // namespace tune
