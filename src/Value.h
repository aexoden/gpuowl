// Copyright (C) Jason Lynch

// What an item of tuning work is worth: the expected fall in T it buys, before the seconds it costs are divided in.
//
// Two kinds of worth, because there are two ways a measurement can lower T.  A new configuration might be cheaper than
// what production runs now, which is worth the saving it would bring times the chance of it -- and that chance is a
// distribution over gains, learnt from the gains this device has actually shown, because a point estimate of the gain
// silently prunes everything it puts beyond reach.  Or one more call on a configuration already measured might show
// that production is running the wrong one of two near-equals, which is worth the expected regret that call removes.
//
// A pure function of the database and the scope's grid.

#pragma once

#include "common.h"
#include "Emit.h"
#include "Objective.h"
#include "TuneDB.h"

#include <array>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tune {

// What an entry is: an FFT, a test kind and a regime label.  Its option sets are ways of running it.
using EntryKey = std::tuple<std::string, TestKind, std::string>;

inline constexpr size_t GAIN_BINS = 9;

// The relative gains the distribution is held at.  Roughly doubling, so that the tail is as finely described as the
// head, and ending at 64% so that nothing within ~2.8x of the frontier is valued at exactly zero.
inline constexpr std::array<double, GAIN_BINS> GAIN_AT{0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64};

// How many observations the prior is worth: the posterior moves halfway to what the device shows after this many.
inline constexpr double GAIN_PRIOR_WEIGHT = 8;

// How much an entry's own gains count against the device's, once the entry has as many as the prior is worth: an
// entry whose probes keep finding nothing sinks, but only so far, so that a change of defaults or a restart can
// float it again.
inline constexpr double ENTRY_MIX = 0.5;

// A distribution over the relative gain a configuration turns out to have over the estimate it was valued at.
struct GainDist {
  std::array<double, GAIN_BINS> p{};

  // Where the best available now costs `best`: the expectation of max(0, best - estimate * (1 - g)).
  [[nodiscard]] double expectedSaving(double best, double estimate) const;

  [[nodiscard]] double mean() const;

  // The probability of a gain of at least `gain`.
  [[nodiscard]] double chanceOfAtLeast(double gain) const;
};

// Mostly nothing, with a genuine right tail: gains of 30-60% are rare, but they are what a badly-defaulted family looks
// like, and a tail that stopped short of them would exclude it in all but name.
inline constexpr GainDist GAIN_PRIOR{{0.398, 0.20, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01, 0.002}};

// A combination of several groups' answers lands less often than a move within one group, and pays more when it does.
inline constexpr GainDist GAIN_COMBO_PRIOR{{0.598, 0.12, 0.08, 0.07, 0.06, 0.04, 0.02, 0.01, 0.002}};

// Observed gains, as counts over the bins.
class GainCounts {
public:
  // Split between the two bins either side of it, in proportion to how near it is to each, so that the counts' mean
  // gain is the observed mean.  A loss is a gain of zero; a gain past the last bin counts there.
  void observe(double gain);

  [[nodiscard]] double n() const { return n_; }
  [[nodiscard]] const std::array<double, GAIN_BINS>& counts() const { return counts_; }

  // Dirichlet-multinomial: `prior` worth GAIN_PRIOR_WEIGHT observations, and these on top.
  [[nodiscard]] GainDist posterior(const GainDist& prior) const;

private:
  std::array<double, GAIN_BINS> counts_{};
  double n_ = 0;
};

// What a gain was observed on.
enum class GainSource : u8 {
  Move,     // a move within one group, or one key: teaches the device and its entry
  Restart,  // a random jump: says the entry's search is spent, and nothing about the device
  Combo,    // a combination of several groups' answers: teaches the device's and its entry's combination gains
};

// What the gains a device has shown say about the next one.  Two distributions, each learnt only from its own kind of
// step, since folding them together would misprice both: moves from the move prior, combinations from theirs.
class GainModel {
public:
  void observe(const EntryKey& entry, double gain, GainSource source = GainSource::Move);

  // For a configuration of no entry in particular.
  [[nodiscard]] GainDist global() const { return all_.posterior(GAIN_PRIOR); }

  // For one of `entry`'s: its own counts over the global posterior, mixed back with that posterior at ENTRY_MIX.
  [[nodiscard]] GainDist forEntry(const EntryKey& entry) const;

  // The same two for a combination.
  [[nodiscard]] GainDist globalCombo() const { return combos_.posterior(GAIN_COMBO_PRIOR); }
  [[nodiscard]] GainDist comboForEntry(const EntryKey& entry) const;

  // For a restart of `entry`: its own restarts over forEntry(), with no floor.  The floor forEntry() keeps is what
  // lets an entry whose moves found nothing be floated again, but a jump is drawn from the same space however often
  // that space has come up empty, so an entry whose jumps keep finding nothing is worth jumping from less and less --
  // and a run that is to stop on its own must be able to stop jumping.
  [[nodiscard]] GainDist restartForEntry(const EntryKey& entry) const;

  [[nodiscard]] const GainCounts& all() const { return all_; }
  [[nodiscard]] const GainCounts& combos() const { return combos_; }

private:
  GainCounts all_;
  std::map<EntryKey, GainCounts> entries_;
  GainCounts combos_;
  std::map<EntryKey, GainCounts> entryCombos_;
  std::map<EntryKey, GainCounts> restarts_;
};

// An option set of an entry, as its canonical text.
using EntrySet = std::pair<EntryKey, std::string>;

// Every gain `env`'s rows show, entry by entry and in the order the rows were first taken: each option set of an entry
// after its first is a move tried against the entry's best so far, and gained max(0, 1 - cost / best), its cost pooled
// over every concluded row of it.  A race's calls are that as much as a probe's are.  A failure gained nothing it could
// be measured by, so is not one.
//
// An option set a jump row declares was a random restart rather than a move, and almost all of them gain nothing: it
// teaches its own entry, whose search it says is exhausted, but not the device's distribution, which values every
// other entry.
//
// An option set a combo row declares was a combination of several groups' answers, and teaches the combination gains
// alone, device and entry, and never the move gains.
[[nodiscard]] GainModel gainsOf(const TuneDB& db, u32 env);

// The fall in T, in microseconds per iteration, were a configuration eligible over `band` to cost `cost`:
// sum over the points of `kind` in `band` of W(E) * max(0, c*(E) - cost).
[[nodiscard]] double saving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band, double cost);

// Its expectation over `gains`, for a configuration valued at `cost` before it is measured.
[[nodiscard]] double expectedSaving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band,
                                    double cost, const GainDist& gains);

// The smallest relative gain on `cost` at which saving() reaches `worth`: 0 where it already does, and nothing where
// no gain could, since even a configuration costing nothing would save less.  With `worth` 0, the gain at which it
// would start to save anything at all.
[[nodiscard]] std::optional<double> requiredGain(std::span<const ObjectivePoint> points, TestKind kind,
                                                 const Interval& band, double cost, double worth);

// The expected cost of deciding now between two configurations whose difference in cost is distributed N(mu, sigma^2):
// sigma * phi(mu / sigma) - |mu| * Phi(-|mu| / sigma).
[[nodiscard]] double regret(double mu, double sigma);

// What one more call on a configuration is expected to remove from that regret, weighted by `weight`: its standard
// error `se`, over `calls` calls, narrows to se * sqrt(calls / (calls + 1)), and the difference's mean may move by as
// much, which is worth regret(mu, se / sqrt(calls + 1)).
[[nodiscard]] double refineValue(double weight, double mu, double se, u32 calls);

// Two option sets production chooses between at some exponents: the one it runs there, the runner-up, and the
// workload weight of the points where they are the two.  Indices into the option sets they were found among.
struct Contest {
  size_t chosen = 0;
  size_t runnerUp = 0;
  double weight = 0;
};

// Two readings are told apart when their intervals, this many standard errors wide on each side, do not overlap.
inline constexpr double RACE_CONFIDENCE = 2.0;

// Or are close enough to call a tie when within this fraction of each other: closer than that, which of the two is
// cheaper is not worth the calls it would take to say.
inline constexpr double RACE_MARGIN = 0.0025;

// A reading that has had this many calls without separating from its rival is tied with it.
inline constexpr u32 RACE_MAX_CALLS = 16;

// At each point of `points`, the set production runs there, ranked as it ranks them, and the one eligible there most
// likely to be cheaper in truth: the fewest standard errors of their difference behind it, by mean.  Not the second in
// production's ranking, which adds standard errors to every mean and so puts a set with few calls behind ones it is
// probably faster than -- the set a further call could most change the answer for.  A pair no point weighs is not a
// contest at all, whatever their readings: no decision between them changes T.
[[nodiscard]] std::vector<Contest> contests(std::span<const OptionSet> sets, std::span<const ObjectivePoint> points);

// What one more call on `sets[side]`, one of the contest's two, is worth.
[[nodiscard]] double refineValue(const Contest& contest, std::span<const OptionSet> sets, size_t side);

// Whether the contest still needs calls: not while the two intervals, RACE_CONFIDENCE standard errors either side, are
// apart, nor while the two are within RACE_MARGIN of each other.
[[nodiscard]] bool undecided(const Contest& contest, std::span<const OptionSet> sets);

// What one more call on each of `sets` is worth, summed over every undecided contest it is one side of; 0 for a set in
// none, and for a side that has had RACE_MAX_CALLS calls, which is tied rather than unresolved.
[[nodiscard]] std::vector<double> refineValues(std::span<const OptionSet> sets, std::span<const ObjectivePoint> points);

}  // namespace tune
