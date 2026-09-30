// Copyright (C) Jason Lynch

// The tuning queue: every item that may run, re-scored from the database after every item.  The readings the accuracy
// gate owes, and baselines where the workload has nothing measured, go first, by rule.  Then the defaults sweep, the
// bootstrap and the search take turns a call at a time.  The search is spread over the contenders by successive
// halving, which is begun again, for longer, each time the one it left has had its share: the first few times whatever
// the value model says, then for as long as a step of one of them is worth the stop fraction; otherwise, and between
// halvings, each item is ranked by how much of the objective T it is expected to remove per second it costs.
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
#include "Search.h"
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

// Where the halving stands, as the rows say: which round, which entries are in it, and how many calls of search each
// is to have in it.  Between two halvings, the one the last left, and the calls it has before the next begins.
struct HalvingState {
  bool active = false;

  // Which of the env's halvings, from 1.
  u32 halving = 0;

  // Between two halvings: `pool` is the one the last left, `budget` the calls of search it has before the next
  // begins, and `calls` how many of them it has had.
  bool leading = false;

  // The round's place in its halving, from 0, and among the env's rounds (RoundRow::n).
  u32 round = 0;
  u32 n = 0;

  u64 budget = 0;

  // Into Scheduler::baselines(), the round's entries this workload weighs, and the calls of search each has had in it.
  // Once the halving is over, the one it left.
  std::vector<size_t> pool;
  std::vector<u64> calls;

  // How many entries the halving's first round took.
  u32 contenders = 0;

  // Rounds the rows do not hold yet, which a run records before its next call: the rounds just begun, and in a
  // database no round was recorded in, where an earlier build's halving stood.
  std::vector<RoundRow> unrecorded;
};

// What the halving asks of each entry's search, by its index into Scheduler::baselines().
struct Explored {
  // Whether the search has a step to take, whatever the value model makes of it; and whether one of those is a
  // measurement begun and not finished.
  std::function<bool(size_t)> offers;
  std::function<bool(size_t)> resumes;

  // Whether it has a step worth a call on its value alone, which under a stop fraction is one worth that much of T.
  std::function<bool(size_t)> worth;
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

// Every entry `env` could publish that the workload gives any weight to: each shape, at each variant `env` can compile,
// in each regime band of the automatic carry that holds a grid point.
[[nodiscard]] std::vector<Baseline> baselines(const Env& env, const RunScope& scope,
                                              const std::vector<FFTShape>& shapes = FFTShape::allShapes());

enum class ItemKind : u8 { Anchor, Baseline, Probe, Combo, Refine, Restart, Gate, Reach };

[[nodiscard]] const char* toString(ItemKind kind);

struct Item {
  ItemKind kind = ItemKind::Baseline;

  // Into Scheduler::baselines(): for anything but a baseline, the baseline of the entry it measures.  Unused by a
  // reach, which carries its FFT itself.
  size_t index = 0;

  // A reach's: the FFT it reads.  Not a baseline's, since what a raise is worth lies past the band that holds its set,
  // which the workload need not weigh at all, and one reading raises every kind of the set at once.
  std::optional<FFTConfig> fft{};

  // What the configuration is built with: for a baseline the built-in defaults, for a probe, a combo or a restart its
  // option set, for a refine the option set its row was recorded under, for a gate the set it
  // reads: the one waiting on it, or that set's accuracy reference, and for a reach the set whose reach it raises.
  UseConfig options{};

  // A gate's or a reach's: the option set the reading is for, and the interval the fitted table gives it, which the
  // gate reads it at the top of, derives a lower reach inside and raises a reach above.
  UseConfig subject{};
  Interval span{};

  // A probe's: the key it moved; and what it is, for the log.
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

  // 2 or 3 for a point of a combination, and 1 for anything else.
  u32 tier = 1;

  // A baseline's: run by rule, because it may serve an exponent the workload weighs that nothing measured serves.
  bool cover = false;

  // A baseline's: run by rule, at the built-in defaults, because the entry is within CONTEND_MARGIN of what
  // production runs somewhere the workload weighs.
  bool sweep = false;

  // A probe's, a combo's or a restart's: a step of a halving's round, run by rule ahead of anything else.
  bool halving = false;

  // A baseline's, a probe's, a combo's or a restart's: run by rule, as a step of a family's bootstrap, whose worth is
  // the lines it gives every entry of its type rather than what it saves on its own.
  bool bootstrap = false;

  // The first probe or combo offered from a stage listed only in part: at most how many more points the stage has,
  // which later windows list.
  u64 unlisted = 0;

  // A probe's, a combo's or a restart's: where its entry's search put it among what it offered.
  u32 order = 0;

  [[nodiscard]] double rate() const { return seconds > 0 ? value / seconds : 0; }
};

// Whether `item` runs by rule rather than by value: a step of the bootstrap, a gate reading, a baseline covering the
// workload or taken in the defaults sweep, a step of a halving's round, or a search's measurement begun, which one more
// call concludes and whose call so far counts for nothing until it does.
[[nodiscard]] bool byRule(const Item& item);

// Whether `item` is worth a call where anything expected to lower T by less than `floor` is not: one that runs by rule
// always is; anything else once it is worth something and at least `floor`.
[[nodiscard]] bool worthRunning(const Item& item, double floor);

class Scheduler {
public:
  // With the default `bootstrap`, which is turned off, no family is searched ahead of the rest.  Baselines run at the
  // built-in defaults whatever the lines say.  Without a `strategy` nothing measured is measured further: no probes, no
  // refines, and no bootstrap.  Without `restarts` an entry whose probes are all answered is left there, so the queue
  // can run dry; with them it never does, since a jump is always worth a little, and an entry also jumps once each
  // RESTART_PERIOD option sets it measures.  Without `gate` the accuracy gate reads nothing, so nothing but exact
  // arithmetic is ever published; the search is the same either way.
  Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize = 1000, Bootstrap bootstrap = {},
            std::optional<Strategy> strategy = {}, bool restarts = false, bool gate = false, Halving halving = {});

  [[nodiscard]] const RunScope& scope() const { return scope_; }
  [[nodiscard]] const std::vector<Baseline>& baselines() const { return baselines_; }
  [[nodiscard]] const Bootstrap& bootstrap() const { return bootstrap_; }
  [[nodiscard]] const std::optional<Strategy>& strategy() const { return strategy_; }
  [[nodiscard]] const Halving& halving() const { return halving_; }

  // Where the halving stood when admissible() last ranked the queue.
  [[nodiscard]] const HalvingState& lastHalving() const { return lastHalving_; }

  // The phase `ranked` is in, as admissible() just gave it from `state`, with its own totals: how many contenders the
  // sweep has read, how many calls of search the family being bootstrapped has had, how far the halving's round has
  // got.  What takes turns is said together, joined by " + ".
  [[nodiscard]] Phase phase(const BootstrapState& state, const std::vector<Item>& ranked, const Objective& objective,
                            double floor) const;

  // Where the bootstrap stands, each family given BOOTSTRAP_ROUNDS rounds of the halving's calls.  A family that cannot
  // be searched here -- without a strategy, or whose configuration the workload gives no weight to -- is skipped, and
  // one this process has tried to read too often without a reading is held.
  [[nodiscard]] BootstrapState bootstrapState(const TuneDB& db, u32 env) const;

  // The default lines as they stand -- what the selection file publishes beside its entries, and what every entry
  // tries as a step of its own: publishedLines() over the sets the gate has passed, at the probe.
  [[nodiscard]] Defaults lines(const TuneDB& db, u32 env) const;

  // Every item that may run now, scored against what `env` has measured, by `objective` -- which should count the sets
  // the gate still owes a reading, Gating::Assumed, since those readings are taken first.  While the table would
  // publish a set the accuracy gate owes a reading -- of the set, or of its accuracy reference -- those readings are
  // all there is, quickest first: they are what stands between what has been found and what production runs, and the
  // objective cannot price them, since its prior is below what the entries they publish cost.  Then, with the gate,
  // while an exponent the workload weighs has no entry, the baselines whose bands hold one: nothing would be published
  // there otherwise, and the value of a first measurement is only the gain it might show over the prior, which is its
  // own shape's.  After that three kinds of work take turns, a call at a time, starting after the kind the last call
  // was: with a strategy, the defaults sweep (sweepItems()), the entries whose bands hold the probe first, since each
  // family is searched on its type's cheapest reading there; the bootstrap, once the configuration each family is on
  // is recorded -- each family still to be read at the built-in defaults, cheapest first, and once every one is, what
  // the search of the cheapest family still owed its calls offers, in the search's own order and whatever it is
  // priced at; and the search.  The search is the halving's round while one is under way (halvingState()): what the
  // search of each of its entries still short of its calls offers, priced by rule, and whether a later halving begins
  // is decided by `floor`, what a step must be worth to be run on its value (pick()).  Otherwise, together and best
  // rate first: the baselines, at the built-in defaults; what the search of every entry with a row emission could
  // publish offers (EntrySearch::offers()), the lines and each probe valued under the entry's move gains, each combo
  // under its combination gains and a restart under its restart gains, from the cost of the best set it is a step from;
  // one more call on each side of every contest production decides that the race rule leaves undecided
  // (refineValues()); and with the gate, the next reading of each passed set whose reach may be raised above the table,
  // worth what that set would save over the exponents between its reach and that reading, at what it costs.  A baseline
  // is left out once a row has concluded it or recorded a failure of it, while an earlier generation's death holds it,
  // and once this process has tried it more often than any entry needs.
  [[nodiscard]] std::vector<Item> admissible(const TuneDB& db, u32 env, const Objective& objective,
                                             double floor = 0) const;

  // Whether nothing is left of the workload's coverage or, with a strategy, of the defaults sweep at the probe: what
  // the bootstrap's choice of configurations waits for.
  [[nodiscard]] bool swept(const TuneDB& db, u32 env, const Objective& objective) const;

  // The halving as the rows stand.  Its first round takes the contenders: the entries with a publishable reading within
  // CONTEND_MARGIN of what production is measured to run at some exponent of their band the workload weighs, or within
  // it at the built-in defaults of the cheapest reading there at the built-in defaults -- an entry behind only because
  // the one ahead of it has been searched is compared as the two stood before either was -- the best `contenders` of
  // them by gap now, taken one variant of each shape before a second of any, since what the search is spread over is
  // which shape tunes best.  A round's entries are recorded as it begins, and stay in it until each has had its
  // calls of search, counted from then, or has no step left to take whatever it is worth, and has finished any step
  // begun; the better half by gap then go on to a round of twice the calls.  It is over once one is left, which then
  // has as many calls of search as the halving's rounds gave out, under the ranking by value, or until it has no step
  // worth a call, before the next halving begins: over every contender as they stand then, those an earlier halving
  // left behind among them, its first round twice as long as the last halving's.  So an entry whose gains lie further
  // down its search than one round reaches is searched again, for longer each time, and the one ahead keeps at least
  // half of the calls.  A halving once begun runs to its end, since what a round is for is what the value model cannot
  // see coming.  For the same reason the first Halving::halvings begin whatever it makes of their steps; whether a
  // later one is worth running is decided as it would begin, by whether one of its contenders has a step worth a call
  // on its value alone.  That keeps a run with a stop fraction finite: each such halving takes such a step in its
  // first round.  The first halving does not begin while `sweeping`, the defaults sweep still owing readings at the
  // probe, so that it takes every contender at once.
  [[nodiscard]] HalvingState halvingState(const TuneDB& db, u32 env,
                                          const std::map<EntryKey, std::vector<Reading>>& readings,
                                          const Objective& objective, const Explored& explored, bool sweeping) const;

  // The first measurements admissible() offers once nothing runs ahead of them by rule, whether or not a bootstrap
  // call or a gate reading is holding them back now.
  [[nodiscard]] std::vector<Item> baselineItems(const TuneDB& db, u32 env, const Objective& objective) const;

  // The item to run next from a ranking admissible() gave, or nothing where none is worth running (worthRunning()).
  // The top item worth running, unless it would repeat the previous one while another is within INTERLEAVE_EPS of it.
  [[nodiscard]] std::optional<Item> pick(const std::vector<Item>& ranked, double floor = 0) const;

  // Records that `item` ran, taking `seconds` and measuring `usPerIt` (0 where it measured nothing).  `recorded` is
  // false where the call completed but its row cannot count for the item, because its kernels were built otherwise.
  void ran(const Item& item, double seconds, double usPerIt, bool recorded = true);

  [[nodiscard]] const CallClock& clock() const { return clock_; }

private:
  [[nodiscard]] std::string keyOf(const Item& item) const;

  // The readings at the built-in defaults the bootstrap owes: one for each family not read yet, cheapest first.
  [[nodiscard]] std::vector<Item> bootstrapReads(const BootstrapState& state, const TuneDB& db, u32 env,
                                                 const Progress& progress, const Objective& objective) const;

  // Into baselines_, the entry a family's bootstrap searches: its configuration in `kind`, in the band that holds the
  // probe.  Nothing where the workload gives that entry no weight.
  [[nodiscard]] std::optional<size_t> entryOf(const Family& family, TestKind kind) const;

  // The kinds of work that take turns, in the order the turns go round.
  enum class Turn : u8 { Sweep, Bootstrap, Search };
  static constexpr size_t TURNS = 3;

  // Each kind's items taken a turn at a time, the kind after the one the last call was first, then round from there,
  // each in its own order.
  [[nodiscard]] std::vector<Item> inTurn(std::array<std::vector<Item>, TURNS> turns) const;

  [[nodiscard]] std::vector<Item> baselineItems(const TuneDB& db, u32 env, const Progress& progress,
                                                const GainModel& gains, const Objective& objective) const;

  // The defaults sweep: every entry with no reading at the built-in defaults whose cheapest reading, or where it has
  // none its prior, is within CONTEND_MARGIN of the fastest thing at some exponent of its band the workload weighs:
  // what production is measured to run there, or the prior of an entry that can still be read there, if cheaper.
  [[nodiscard]] std::vector<Item> sweepItems(const TuneDB& db, u32 env, const Progress& progress,
                                             const std::map<EntryKey, std::vector<Reading>>& readings,
                                             const GainModel& gains, const Objective& objective) const;

  // What production runs at each weighted point of `objective`, where that is measured; nothing elsewhere.
  [[nodiscard]] std::vector<std::optional<double>> measuredAt(const Objective& objective) const;

  // Best rate first by shape, the variants of one shape together.  The variants of a shape share a prior, so nothing
  // yet tells them apart but the edges of their bands: the variant production's own shape scan runs where nothing is
  // published goes first.
  void rankByShape(std::vector<Item>& items) const;

  [[nodiscard]] std::vector<Item> gateItems(const TuneDB& db, u32 env) const;

  [[nodiscard]] std::vector<Item> coverItems(std::span<const Item> baselines, const Objective& objective) const;

  [[nodiscard]] std::vector<Item> reachItems(const TuneDB& db, u32 env, std::span<const OptionSet> sets,
                                             const Objective& objective) const;

  // Where each entry stands against the others, by its index into baselines_.
  struct Standing {
    // How far behind what production is measured to run it is, at the best of its points; nothing for an entry with
    // no publishable reading.
    std::vector<std::optional<double>> gap;

    // How far its reading at the built-in defaults is behind the cheapest such reading at the best of its points:
    // where it would stand had nothing been searched.  Nothing where either reading is missing.
    std::vector<std::optional<double>> atDefaults;

    // The weight of the workload production runs it at.
    std::vector<double> weight;

    // Within CONTEND_MARGIN now, or at the built-in defaults.
    [[nodiscard]] bool contends(size_t i) const;

    // Nearest the fastest first, the one production runs over more of the workload first where two are as near.
    [[nodiscard]] bool ahead(size_t a, size_t b) const;
  };

  [[nodiscard]] Standing standingOf(const std::map<EntryKey, std::vector<Reading>>& readings,
                                    const Objective& objective) const;

  // The calls of search each entry has had: every call at an option set other than the built-in defaults.
  [[nodiscard]] std::vector<u64> searchCalls(const TuneDB& db, u32 env) const;

  // Up to `limit` of the entries with a gap that `eligible` accepts, one variant of each shape before a second of any,
  // ahead first.
  [[nodiscard]] std::vector<size_t> poolOf(const Standing& standing, const std::function<bool(size_t)>& eligible,
                                           u32 limit) const;

  // For a database no round was recorded in, the rounds that say where the halving stood by the rule before rounds
  // were recorded, which counted every call of search an entry had ever had: over, or in a later round, as that rule
  // has it.  A first round is begun afresh, since it cannot be told from a bootstrap's calls, which that rule also
  // counted; and so is a database whose halving has not begun.
  [[nodiscard]] std::vector<RoundRow> adoption(const Standing& standing, const std::vector<u64>& calls) const;

  [[nodiscard]] RoundMember memberOf(size_t index, u64 from) const;

  // Into baselines_, by the entry each is of.
  [[nodiscard]] std::map<EntryKey, size_t> entryIndex() const;

  // "<spec> <canonical options>", which is what makes a later build of the same configuration find it compiled.
  [[nodiscard]] std::string builtKey(const FFTConfig& fft, const UseConfig& options) const;

  // What `baselines_[index]`'s search offered, as an item: what it is expected to take, and whether its kernels are
  // still to be built.
  [[nodiscard]] Item itemOf(size_t index, Candidate candidate) const;

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

  // Each entry's search, as baselines_ has them.
  mutable std::vector<EntrySearch> searches_;

  // Gate and reach readings by keyOf(), however they ended: one that recorded a reading is not owed again unless its
  // kernels were built otherwise, and asking again would only repeat that.
  std::map<std::string, u32> gateAttempts_;

  // Refine calls by keyOf() that recorded nothing; a refine is bounded by its row's calls otherwise.
  std::map<std::string, u32> unrecordedRefines_;

  std::string last_;

  // The kind of work the last call was, of those that take turns.
  std::optional<Turn> lastTurn_;
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

  // Records that the bootstrap family of `fft`'s type is searched on `fft` at `probe`.
  virtual void declareBootstrap(const FFTConfig& fft, u64 probe) = 0;

  // Records a round of the halving as `round` has it, in this session and at this time.
  virtual void declareRound(const RoundRow& round) = 0;

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
