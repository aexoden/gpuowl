// Copyright (C) Jason Lynch

// The tuning queue: one priority queue of work items, re-scored from the database after every item, with no passes and
// no phases.  Each item is ranked by how much of the objective T it is expected to remove per second it costs.
//
// The value model is only as good as its gain distribution, and a bad one misallocates time without ever making a wrong
// decision: what is published is chosen by ranking measured rows, never by these scores.  That asymmetry is what lets
// the scores be estimates.
//
// Everything here is pure but the loop, and the loop reaches the device only through a Bench, so a schedule can be
// replayed from fixed readings with no GPU.

#pragma once

#include "Bootstrap.h"
#include "common.h"
#include "Eligibility.h"
#include "FFTConfig.h"
#include "Objective.h"
#include "OptionSpace.h"
#include "TuneDB.h"
#include "Tuner.h"
#include "UseResolve.h"

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace tune {

// While another item scores within this fraction of the top one, the same configuration is not run twice in a row: a
// comparison between two configurations measured one after the other is also a comparison between two moments.
inline constexpr double INTERLEAVE_EPS = 0.10;

// What building a configuration's kernels is assumed to cost until one build has been timed.
inline constexpr double COMPILE_ESTIMATE_SEC = 20;

// What a call spends outside the iterations it runs -- buffers, trig tables, the first block's setup -- until one call
// has been timed.
inline constexpr double CALL_OVERHEAD_SEC = 1;

// Blocks a call runs before the ones it times (Gpu::timeIters).
inline constexpr u32 WARMUP_BLOCKS = 1;

// The prior over the relative gain a configuration not yet measured turns out to have over the estimate it was scored
// at.  A point estimate would silently prune: a candidate 40% off the pace scores exactly zero against any gain short
// of 40%, which is exclusion in all but name.  The tail is what a badly-defaulted family looks like.
struct GainBin {
  double gain;
  double p;
};
inline constexpr std::array<GainBin, 8> GAIN_PRIOR{{
  {0.000, 0.40},
  {0.005, 0.20},
  {0.010, 0.15},
  {0.020, 0.10},
  {0.040, 0.07},
  {0.080, 0.04},
  {0.160, 0.03},
  {0.320, 0.01},
}};

// The expected fall in cost at one exponent, in the cost's unit, from a configuration estimated at `estimate` where the
// best available now costs `best`: the expectation over GAIN_PRIOR of max(0, best - estimate * (1 - g)).
[[nodiscard]] double expectedSaving(double best, double estimate);

// What a call is expected to take, in wall-clock seconds, learnt from the calls this process has made.
class CallClock {
public:
  explicit CallClock(u32 blockSize = 1000) : blockSize_{blockSize} {}

  // `fresh` for a configuration whose kernels this process has not built before.
  [[nodiscard]] double seconds(double usPerIt, bool fresh) const;

  // A completed call: how long it took and what it measured per iteration.
  void observe(double seconds, double usPerIt, bool fresh);

  [[nodiscard]] double overhead() const;
  [[nodiscard]] double compile() const;

private:
  [[nodiscard]] double iterSeconds(double usPerIt) const;

  u32 blockSize_;
  double overheadSum_ = 0;
  u32 overheadN_ = 0;
  double compileSum_ = 0;
  u32 compileN_ = 0;
};

// An entry nothing has measured yet: one configuration at the built-in defaults, in one test kind and one regime band.
struct Baseline {
  FFTConfig fft;
  TestKind kind;

  // Where the entry would be eligible, and so the band whose workload weight its value is taken over.
  Interval band;

  // Where it is timed when nothing has been started on it: the probe where the band holds it, and otherwise the largest
  // prime the band reaches.
  u64 exponent;

  // "<spec> <kind> <regime>"
  [[nodiscard]] std::string label() const;
};

// Every entry `env` could publish that the workload gives any weight to: each shape, at each variant `env` can compile,
// in each regime band of the automatic carry that holds a grid point.
[[nodiscard]] std::vector<Baseline> baselines(const Env& env, const RunScope& scope,
                                              const std::vector<FFTShape>& shapes = FFTShape::allShapes());

enum class ItemKind : u8 { Anchor, Bootstrap, Baseline };

[[nodiscard]] const char* toString(ItemKind kind);

struct Item {
  ItemKind kind = ItemKind::Baseline;

  // Into Scheduler::baselines(), or for a bootstrap call into the bootstrap's families.
  size_t index = 0;

  // What the configuration is built with: a bootstrap candidate, or for a baseline the defaults the bootstrap decided.
  UseConfig options{};

  // A bootstrap call's: the key it moved, and what it is, for the log.
  std::string moved{};
  std::string what{};

  u64 exponent = 0;

  // The expected fall in T, in microseconds per iteration, and what the item is expected to take.
  double value = 0;
  double seconds = 0;

  bool fresh = true;

  // Calls already recorded against the entry, which is what makes this item a resumption.
  u32 calls = 0;

  [[nodiscard]] double rate() const { return seconds > 0 ? value / seconds : 0; }
};

class Scheduler {
public:
  // With the default `bootstrap`, which is turned off, every baseline is admissible at once and runs at the built-in
  // defaults.
  Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize = 1000, Bootstrap bootstrap = {});

  [[nodiscard]] const RunScope& scope() const { return scope_; }
  [[nodiscard]] const std::vector<Baseline>& baselines() const { return baselines_; }
  [[nodiscard]] const Bootstrap& bootstrap() const { return bootstrap_; }

  [[nodiscard]] BootstrapState bootstrapState(const TuneDB& db, u32 env) const;

  // Every item that may run now, scored against what `env` has measured.  While a family still has a bootstrap call to
  // make, those calls are all there is, most wanted first: every other configuration runs at what the bootstrap
  // decides, so measuring one earlier would measure something production is not going to run.  After that the
  // baselines, best rate first; an entry is left out once a row has concluded it or recorded a failure of it, while an
  // earlier generation's death or an unbuildable key holds it, and once this process has tried it more often than any
  // entry needs.
  [[nodiscard]] std::vector<Item> admissible(const TuneDB& db, u32 env, const Objective& objective) const;

  // The item to run next from a ranking admissible() gave, or nothing where none is worth anything.  The top item,
  // unless it would repeat the previous one while another is within INTERLEAVE_EPS of it -- or for a bootstrap call,
  // unless it would repeat the previous one at all while the race has another candidate to call.
  [[nodiscard]] std::optional<Item> pick(const std::vector<Item>& ranked) const;

  // Records that `item` ran, taking `seconds` and measuring `usPerIt` (0 where it measured nothing).  `recorded` is
  // false where the call completed but its row cannot count for the item, because its kernels were built otherwise.
  void ran(const Item& item, double seconds, double usPerIt, bool recorded = true);

  [[nodiscard]] const CallClock& clock() const { return clock_; }

private:
  [[nodiscard]] std::string keyOf(const Item& item) const;

  [[nodiscard]] std::vector<Item> bootstrapItems(const BootstrapState& state, const Objective& objective) const;

  RunScope scope_;
  std::vector<Baseline> baselines_;
  CallClock clock_;
  Bootstrap bootstrap_;

  // The configurations this process has built, whose next build finds its kernels compiled.
  std::set<std::string> built_;

  std::map<size_t, u32> attempts_;

  // Bootstrap candidates by keyOf(), and how often a call of one recorded nothing; past MAX_ATTEMPTS they are out.
  std::map<std::string, u32> unrecorded_;
  std::string last_;
};

// What the queue runs its items on.
class Bench {
public:
  struct Result {
    bool completed = false;
    double seconds = 0;
    double usPerIt = 0;

    // The options the kernels were built with, after whatever the host set aside.
    UseConfig ran{};
  };

  virtual ~Bench() = default;

  // Always false for a session with no anchor.
  [[nodiscard]] virtual bool anchorDue() const = 0;
  virtual void timeAnchor() = 0;

  // Records its own rows.  `moved` is the one key that differs from the configuration this one is compared with, if
  // there is one, so that a build failure can be pinned on it.
  [[nodiscard]] virtual Result run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                                   const std::string& moved) = 0;

  // On request, or because the device can do nothing more.
  [[nodiscard]] virtual bool stopped() const = 0;
};

struct QueueReport {
  u32 items = 0;
  u32 anchors = 0;
  bool stopped = false;

  // What T stood at before the first item and after the last.
  double startT = 0;
  double endT = 0;
};

// What a run publishes: the objective the entries give, and the default lines the bootstrap has decided so far.
using Publisher = std::function<void(const Objective&, const Defaults&)>;

// Runs items until none is worth anything or the bench stops, re-scoring from the database after each.  `publish`
// is given the objective once before the first item and again after every one, so that whatever interrupts the run
// finds a selection file describing everything measured before it.
QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench, const Publisher& publish);

}  // namespace tune
