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
#include "Probe.h"
#include "TuneDB.h"
#include "Tuner.h"
#include "UseResolve.h"
#include "Value.h"

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
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

// How far behind what production runs at some exponent an entry may be and still be worth searching: the gap tuning
// has been seen to close between one FFT and another at their defaults.  Entries this close are all measured at the
// built-in defaults before the bootstrap (the defaults sweep).
inline constexpr double CONTEND_MARGIN = 0.10;


// What a run is doing now, and how far through it it is: "defaults sweep: 23 of 70 FFTs read", with a brief form for
// the terminal's bottom line ("sweep 23/70").
struct Phase {
  std::string text;
  std::string brief;
};

// Where the halving stands, as the rows say: which round, which entries are still in it, and how many calls of search
// each is to have had by the round's end.
struct HalvingState {
  bool active = false;
  u32 round = 0;
  u64 budget = 0;

  // Into Scheduler::baselines(), the smallest gap to what production runs first; and the calls of search each has had.
  std::vector<size_t> pool;
  std::vector<u64> calls;

  // How many contenders the first round took.
  u32 contenders = 0;
};

// What a call is expected to take, in wall-clock seconds, learnt from the calls this process has made.
class CallClock {
public:
  explicit CallClock(u32 blockSize = 1000) : blockSize_{blockSize} {}

  // `fresh` for a configuration whose kernels this process has not built before.
  [[nodiscard]] double seconds(double usPerIt, bool fresh) const;

  // What an accuracy check is expected to take.  Always a build of its own: the kernels are compiled for the exponent,
  // and the gate reads a set where it was not timed.
  [[nodiscard]] double gateSeconds(double usPerIt) const;

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

enum class ItemKind : u8 { Anchor, Bootstrap, Baseline, Probe, Combo, Refine, Restart, Gate, Reach };

[[nodiscard]] const char* toString(ItemKind kind);

struct Item {
  ItemKind kind = ItemKind::Baseline;

  // Into Scheduler::baselines() -- for anything but a baseline, the baseline of the entry it measures -- or for a
  // bootstrap call into the bootstrap's families.  Unused by a reach, which carries its FFT itself.
  size_t index = 0;

  // A reach's: the FFT it reads.  Not a baseline's, since what a raise is worth lies past the band that holds its set,
  // which the workload need not weigh at all, and one reading raises every kind of the set at once.
  std::optional<FFTConfig> fft{};

  // What the configuration is built with: a bootstrap candidate, for a baseline the lines as they stand, for a probe, a
  // combo or a restart its option set, for a refine the option set its row was recorded under, for a gate the set it
  // reads: the one waiting on it, or that set's accuracy reference, and for a reach the set whose reach it raises.
  UseConfig options{};

  // A gate's or a reach's: the option set the reading is for, and the interval the fitted table gives it, which the
  // gate reads it at the top of, derives a lower reach inside and raises a reach above.
  UseConfig subject{};
  Interval span{};

  // A bootstrap call's or a probe's: the key it moved; and what it is, for the log.
  std::string moved{};
  std::string what{};

  u64 exponent = 0;

  // The expected fall in T, in microseconds per iteration.
  double value = 0;

  // For a baseline, a probe, a combo or a restart, the cost the gain it is valued by is a gain on: what the entry is
  // estimated or measured to cost now.
  double cost = 0;

  // What the item is expected to take.
  double seconds = 0;

  bool fresh = true;

  // Calls already recorded against the entry, which is what makes this item a resumption.
  u32 calls = 0;

  // A restart's: which draw of its entry's sequence it is.
  u32 draw = 0;

  // 2 or 3 for a point of a combination, a combo's or a bootstrap race's, and 1 for anything else.
  u32 tier = 1;

  // A baseline's: run by rule, because it may serve an exponent the workload weighs that nothing measured serves.
  bool cover = false;

  // A baseline's: run by rule, at the built-in defaults, because the entry is within CONTEND_MARGIN of what
  // production runs somewhere the workload weighs.
  bool sweep = false;

  // A probe's, a combo's or a restart's: run by rule, as a round of the halving.
  bool halving = false;

  // The first probe or combo offered from a stage listed only in part: at most how many more points the stage has,
  // which later windows list.
  u64 unlisted = 0;

  [[nodiscard]] double rate() const { return seconds > 0 ? value / seconds : 0; }
};

// Where each entry stands, from the rows of an env.
struct Progress;

// Draws in a row that repeat a configuration already seen, after which a restart space is taken to be spent.  A space
// with one point left unseen among n survives this many draws with probability (1 - 1/n)^256: for n = 64, under 2%,
// and the next process starts the count again.
inline constexpr u32 RESTART_REPEATS = 256;

// How far one entry's restart sequence has been read.  Only what can never become runnable again is passed -- a draw a
// row answers, a hold, a draw given up on, one the lines shadow, which once the bootstrap is complete they keep doing
// -- so the cursor never goes back, and a long stretch of such draws is read once rather than on every re-score.
struct RestartScan {
  u32 next = 0;
  u32 repeats = 0;
  bool exhausted = false;

  // The configurations drawn or measured so far, as their canonical text.
  std::set<std::string> seen;
};

// Moves `scan` to the first draw from where it stands that `runnable` accepts, and returns it; `text` names the
// configuration a draw is.  Nothing once RESTART_REPEATS draws in a row have repeated configurations in `seen`, which
// is what a space whose every point has been measured or ruled out looks like: a long run of new configurations that
// cannot run is the space still being explored, however long it is.
[[nodiscard]] std::optional<u32> nextRunnable(RestartScan& scan, const std::function<std::string(u32)>& text,
                                              const std::function<bool(u32)>& runnable);

// Whether `item` runs by rule rather than by value: a bootstrap call, a gate reading, a baseline covering the workload
// or taken in the defaults sweep, or a step of a halving round.
[[nodiscard]] bool byRule(const Item& item);

// Whether `item` is worth a call where anything expected to lower T by less than `floor` is not: one that runs by rule
// always is; anything else once it is worth something and at least `floor`.
[[nodiscard]] bool worthRunning(const Item& item, double floor);

class Scheduler {
public:
  // With the default `bootstrap`, which is turned off, every baseline is admissible at once.  Baselines run at the
  // built-in defaults whatever the bootstrap decides.  Without a `strategy` nothing measured is measured further: no
  // probes, and no refines.  Without `restarts` an entry whose probes are all answered is left there, so the queue can
  // run dry; with them it never does, since a jump is always worth a little.  Without `gate` the accuracy gate reads
  // nothing, so nothing but exact arithmetic is ever published; the search is the same either way.
  Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize = 1000, Bootstrap bootstrap = {},
            std::optional<Strategy> strategy = {}, bool restarts = false, bool gate = false, Halving halving = {});

  [[nodiscard]] const RunScope& scope() const { return scope_; }
  [[nodiscard]] const std::vector<Baseline>& baselines() const { return baselines_; }
  [[nodiscard]] const Bootstrap& bootstrap() const { return bootstrap_; }
  [[nodiscard]] const std::optional<Strategy>& strategy() const { return strategy_; }
  [[nodiscard]] const Halving& halving() const { return halving_; }

  // Where the halving stood when admissible() last ranked the queue.
  [[nodiscard]] const HalvingState& lastHalving() const { return lastHalving_; }

  // The phase `ranked` is in, as admissible() just gave it from `state`, with its own totals: how many races of the
  // family being raced are decided, how many contenders the sweep has read, how far the halving's round has got.
  [[nodiscard]] Phase phase(const BootstrapState& state, const std::vector<Item>& ranked, const Objective& objective,
                            double floor) const;

  [[nodiscard]] BootstrapState bootstrapState(const TuneDB& db, u32 env) const;

  // The default lines as they stand -- what the selection file publishes beside its entries, and what a baseline is
  // taken under, so that it measures what production runs an FFT nothing has been published for at: publishedLines()
  // over the sets the gate has passed, at the probe, from `state`.
  [[nodiscard]] Defaults lines(const TuneDB& db, u32 env, const BootstrapState& state) const;

  // Every item that may run now, scored against what `env` has measured, by `objective` -- which should count the sets
  // the gate still owes a reading, Gating::Assumed, since those readings are taken first.  Once the workload is covered
  // and, with a strategy, the defaults sweep is done, while a family still has a bootstrap call to make those calls
  // are all there is, most wanted first: the search tries the lines they decide first, so a step taken earlier is a
  // step from lines about to move.  Otherwise, while the table would publish a set the accuracy gate owes a reading --
  // of the set, or of its accuracy reference -- those readings are all there is, quickest first: they are what stands
  // between what has been found and what production runs, and the objective cannot price them, since its prior is
  // below what the entries they publish cost.  Then, with the gate, while an exponent the workload weighs has no entry,
  // the baselines whose bands hold one: nothing would be published there otherwise, and the value of a first
  // measurement is only the gain it might show over the prior, which is its own shape's.  Then, with a strategy, the
  // defaults sweep (sweepItems()).  After that, together and best rate first: the baselines, at the built-in defaults;
  // for an entry still best at the built-in defaults, the lines as one step, first; the probes and combos of
  // every entry with a row emission could publish, from its best set or, under a strategy that searches by group, from
  // the best set of each of its cheapest MAX_BRANCHES structural branches, each valued at that branch's cost -- a probe
  // under the entry's move gains and a combo under its combination gains -- and a combo only once its branch has
  // nothing of a lower tier left to offer, since it combines what those found; one more call on each side of every
  // contest production decides that the race rule leaves undecided (refineValues()); for an entry with no probe or
  // combo left, the next draw of its restart sequence; and with the gate, the next reading of each passed set whose
  // reach may be raised above the table, worth what that set would save over the exponents between its reach and that
  // reading, at what it costs.  A baseline is left out once a row has concluded it or recorded a failure of it, and a
  // probe, a combo or a restart once a row answers it or recorded a failure of it; any of them while an earlier
  // generation's death or an unbuildable key holds it, and once this process has tried it more often than any entry
  // needs.
  [[nodiscard]] std::vector<Item> admissible(const TuneDB& db, u32 env, const Objective& objective) const;

  // Whether nothing is left of the workload's coverage or, with a strategy, of the defaults sweep: what the bootstrap
  // waits for.
  [[nodiscard]] bool swept(const TuneDB& db, u32 env, const Objective& objective) const;

  // The halving as the rows stand.  The contenders are the entries with a publishable reading within CONTEND_MARGIN
  // of the fastest thing at some exponent of their band the workload weighs, the best `contenders` of them taken one
  // variant of each shape before a second of any, since what the search is spread over is which shape tunes best.  A
  // round ends once every contender still in it has had its calls of search or has no step left to take in
  // `offering`; the pool then keeps its better half, by gap, and the calls double.  It is over once one is left.
  [[nodiscard]] HalvingState halvingState(const TuneDB& db, u32 env,
                                          const std::map<EntryKey, std::vector<Reading>>& readings,
                                          const Objective& objective, const std::set<size_t>& offering) const;


  // The first measurements admissible() offers once nothing runs ahead of them by rule, whether or not a bootstrap
  // call or a gate reading is holding them back now.
  [[nodiscard]] std::vector<Item> baselineItems(const TuneDB& db, u32 env, const Objective& objective) const;

  // The item to run next from a ranking admissible() gave, or nothing where none is worth running (worthRunning()).
  // The top item worth running, unless it would repeat the previous one while another is within INTERLEAVE_EPS of it
  // -- or for a bootstrap call, unless it would repeat the previous one at all while the race has another candidate to
  // call.
  [[nodiscard]] std::optional<Item> pick(const std::vector<Item>& ranked, double floor = 0) const;

  // Records that `item` ran, taking `seconds` and measuring `usPerIt` (0 where it measured nothing).  `recorded` is
  // false where the call completed but its row cannot count for the item, because its kernels were built otherwise.
  void ran(const Item& item, double seconds, double usPerIt, bool recorded = true);

  [[nodiscard]] const CallClock& clock() const { return clock_; }

private:
  [[nodiscard]] std::string keyOf(const Item& item) const;

  [[nodiscard]] std::vector<Item> bootstrapItems(const BootstrapState& state, const Objective& objective) const;

  [[nodiscard]] std::vector<Item> baselineItems(const TuneDB& db, u32 env, const Progress& progress,
                                                const GainModel& gains, const Objective& objective) const;

  // The defaults sweep: every entry with no reading at the built-in defaults whose cheapest reading, or where it has
  // none its prior, is within CONTEND_MARGIN of c* at some exponent of its band the workload weighs.
  [[nodiscard]] std::vector<Item> sweepItems(const TuneDB& db, u32 env, const Progress& progress,
                                             const std::map<EntryKey, std::vector<Reading>>& readings,
                                             const GainModel& gains, const Objective& objective) const;

  // The fastest thing at each weighted point of `objective`: what production runs there where that is measured, and
  // where it is not, the cheapest of `estimates` among the entries that serve it.  Nothing at a point with no weight.
  [[nodiscard]] std::vector<std::optional<double>> fastestAt(const Objective& objective,
                                                             const std::vector<double>& estimates) const;

  // Best rate first by shape, the variants of one shape together.  The variants of a shape share a prior, so nothing
  // yet tells them apart but the edges of their bands: the variant production's own shape scan runs where nothing is
  // published goes first.
  void rankByShape(std::vector<Item>& items) const;

  [[nodiscard]] std::vector<Item> gateItems(const TuneDB& db, u32 env) const;

  [[nodiscard]] std::vector<Item> coverItems(std::span<const Item> baselines, const Objective& objective) const;

  [[nodiscard]] std::vector<Item> reachItems(const TuneDB& db, u32 env, std::span<const OptionSet> sets,
                                             const Objective& objective) const;

  // Into baselines_, by the entry each is of.
  [[nodiscard]] std::map<EntryKey, size_t> entryIndex() const;

  // "<spec> <canonical options>", which is what makes a later build of the same configuration find it compiled.
  [[nodiscard]] std::string builtKey(const FFTConfig& fft, const UseConfig& options) const;

  // probesOf(), which is pure, for one (entry, best set, whether it steps into other branches), listing at most
  // `listed` points of each stage, and which of its probes the entry's rows checked against it so far answer.  A row
  // that answers a probe always will, so each is checked once.  Per entry, not per FFT: another regime or kind of the
  // same FFT has rows of its own.
  struct ListMemo {
    std::string from;
    u32 listed = PROBE_WINDOW;
    ProbeList list;
    std::vector<bool> answered;
    std::set<std::string> checked;
  };

  // The memo for `best`, rebuilt when `from` -- what the readings of its branch say, which only the combo tiers read
  // -- changes: a best set changes rarely, and each re-score asks again for every entry.  With `widen`, rebuilt
  // listing twice as many points of each stage.  A probe the rebuilt list shares with the old one keeps what the rows
  // answered it.
  [[nodiscard]] ListMemo& probeList(const Baseline& entry, const UseConfig& best, std::span<const Reading> readings,
                                    bool structuralSteps, std::string from, bool widen = false) const;

  // restartOf() for the entry of `baselines_[index]`, canonical, likewise: a draw takes one enumeration of the axes per
  // axis.
  [[nodiscard]] const UseConfig& draw(size_t index, u32 k) const;

  // The restart `baselines_[index]` would make next, if its space has anything left to offer.
  [[nodiscard]] std::optional<Item> nextRestart(const TuneDB& db, u32 env, const Progress& progress,
                                                size_t index) const;

  RunScope scope_;
  std::vector<Baseline> baselines_;
  CallClock clock_;
  Bootstrap bootstrap_;
  std::optional<Strategy> strategy_;
  bool restarts_;
  bool gate_;
  Halving halving_;
  mutable HalvingState lastHalving_;

  // How many entries the last defaults sweep found within the margin or already read there, and how many it still owed.
  mutable u32 sweepWithin_ = 0;
  mutable u32 sweepOwed_ = 0;

  // The configurations this process has built, whose next build finds its kernels compiled.
  std::set<std::string> built_;

  std::map<size_t, u32> attempts_;
  std::map<std::string, u32> probeAttempts_;
  mutable std::map<std::string, ListMemo> probeLists_;
  mutable std::map<std::pair<size_t, u32>, UseConfig> draws_;

  // How far each entry's restart sequence has been read in this process.
  mutable std::map<size_t, RestartScan> scans_;

  // Gate and reach readings by keyOf(), however they ended: one that recorded a reading is not owed again unless its
  // kernels were built otherwise, and asking again would only repeat that.
  std::map<std::string, u32> gateAttempts_;

  // Refine calls by keyOf() that recorded nothing; a refine is bounded by its row's calls otherwise.
  std::map<std::string, u32> unrecordedRefines_;

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

    // What was recorded where the call did not complete.
    Status status = Status::Ok;
  };

  virtual ~Bench() = default;

  // Always false for a session with no anchor.
  [[nodiscard]] virtual bool anchorDue() const = 0;
  virtual void timeAnchor() = 0;

  // Declares that the next call is the `k`th draw of its entry's restart sequence, before it is made.
  virtual void declareRestart(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 k) = 0;

  // Declares that the next call is a point of a combination of tier `tier`, before it is made.
  virtual void declareCombo(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 tier) = 0;

  // Records that the bootstrap family of `fft`'s type races on `fft` at `probe`.
  virtual void declareBootstrap(const FFTConfig& fft, u64 probe) = 0;

  struct Reading {
    bool completed = false;
    double seconds = 0;

    double z = 0;
    u32 n = 0;
    bool checkOk = true;

    UseConfig ran{};
  };

  // Reads the rounding error of one configuration at `exponent`, recording the reading as its own row.
  [[nodiscard]] virtual Reading gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) = 0;

  // Records its own rows.  `moved` is the one key that differs from the configuration this one is compared with, if
  // there is one, so that a build failure can be pinned on it.
  [[nodiscard]] virtual Result run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                                   const std::string& moved) = 0;

  // On request, or because the device can do nothing more.
  [[nodiscard]] virtual bool stopped() const = 0;
};

// Why a run ended.
enum class QueueEnd : u8 {
  Stopped,    // on request, or because the device could do nothing more
  BelowStop,  // something was still worth a little, but nothing the stop fraction of T
  Dry,        // nothing was worth anything
};

struct QueueReport {
  u32 items = 0;
  u32 anchors = 0;
  bool stopped = false;
  QueueEnd end = QueueEnd::Dry;

  // What T stood at before the first item and after the last.
  double startT = 0;
  double endT = 0;

  // The items run, and the seconds they took, by kind.
  struct Spent {
    u32 items = 0;
    double seconds = 0;
  };
  std::map<ItemKind, Spent> spent;

  // Where the run was left: what admissible() gave against the last objective items were valued by, the T of that
  // objective, and what an item had to be worth to run.
  std::vector<Item> left;
  double valuedT = 0;
  double floor = 0;
};

// The lines as the log says them: the global line, then each family's after a "; ! <type>".
[[nodiscard]] std::string linesText(const Defaults& lines);

// What a run publishes: the objective the entries give, and the default lines as they stand (Scheduler::lines()).
using Publisher = std::function<void(const Objective&, const Defaults&)>;

class Watch;

// Runs items until none is worth `stop` of T, or none is worth anything where `stop` is 0, or the bench stops,
// re-scoring from the database after each.  `publish` is given the objective of what is published once before the
// first item and again after every one, so that whatever interrupts the run finds a selection file describing
// everything measured and gated before it.  `watch`, where there is one, is told what each call is before it is made
// and where the run stands after it.
QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench, const Publisher& publish, double stop = 0,
                     Watch* watch = nullptr);

}  // namespace tune
