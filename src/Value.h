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
#include <span>
#include <string>
#include <tuple>
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
};

// Mostly nothing, with a genuine right tail: gains of 30-60% are rare, but they are what a badly-defaulted family looks
// like, and a tail that stopped short of them would exclude it in all but name.
inline constexpr GainDist GAIN_PRIOR{{0.398, 0.20, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01, 0.002}};

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

// What the gains a device has shown say about the next one.
class GainModel {
public:
  void observe(const EntryKey& entry, double gain);

  // For a configuration of no entry in particular.
  [[nodiscard]] GainDist global() const { return all_.posterior(GAIN_PRIOR); }

  // For one of `entry`'s: its own counts over the global posterior, mixed back with that posterior at ENTRY_MIX.
  [[nodiscard]] GainDist forEntry(const EntryKey& entry) const;

  [[nodiscard]] const GainCounts& all() const { return all_; }

private:
  GainCounts all_;
  std::map<EntryKey, GainCounts> entries_;
};

// Every gain `env`'s rows show, entry by entry and in the order the rows were first taken: each option set of an entry
// after its first is a move tried against the entry's best so far, and gained max(0, 1 - cost / best), its cost pooled
// over every concluded row of it.  A race's calls are that as much as a probe's are.  A failure gained nothing it could
// be measured by, so is not one.
[[nodiscard]] GainModel gainsOf(const TuneDB& db, u32 env);

// The fall in T, in microseconds per iteration, were a configuration eligible over `band` to cost `cost`:
// sum over the points of `kind` in `band` of W(E) * max(0, c*(E) - cost).
[[nodiscard]] double saving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band, double cost);

// Its expectation over `gains`, for a configuration valued at `cost` before it is measured.
[[nodiscard]] double expectedSaving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band,
                                    double cost, const GainDist& gains);

// The expected cost of deciding now between two configurations whose difference in cost is distributed N(mu, sigma^2):
// sigma * phi(mu / sigma) - |mu| * Phi(-|mu| / sigma).
[[nodiscard]] double regret(double mu, double sigma);

// What one more call on a configuration is expected to remove from that regret, weighted by `weight`: its standard
// error `se`, over `calls` calls, narrows to se * sqrt(calls / (calls + 1)), and the difference's mean may move by as
// much, which is worth regret(mu, se / sqrt(calls + 1)).
[[nodiscard]] double refineValue(double weight, double mu, double se, u32 calls);

// Two option sets production chooses between at some exponents: the one it runs there, the runner-up, and the
// workload weight of the points where they are the cheapest two.  Indices into the option sets they were found among.
struct Contest {
  size_t chosen = 0;
  size_t runnerUp = 0;
  double weight = 0;
};

// Every pair of `sets` that is the cheapest two eligible at some point of `points`, ranked as production ranks them.
// A pair no point weighs is not a contest at all, whatever their readings: no decision between them changes T.
[[nodiscard]] std::vector<Contest> contests(std::span<const OptionSet> sets, std::span<const ObjectivePoint> points);

// What one more call on `sets[side]`, one of the contest's two, is worth.
[[nodiscard]] double refineValue(const Contest& contest, std::span<const OptionSet> sets, size_t side);

}  // namespace tune
