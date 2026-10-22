// Copyright (C) Jason Lynch

// What to measure next, and how far from the best the published choice is likely to be.
//
// Top-two Thompson sampling over the objective.  A grid point is drawn by weight; under a posterior draw of the model
// each candidate holding it is valued at the cheapest of what is proposed for it, and the cheapest candidate wins.
// What is proposed for a candidate:
//
//   - the optimum of its options under the posterior mean (Optimize.h), moving at most MEAN_STEP_AXES axes from its
//     best measured set: what everything learnt says, transfer from other FFTs included, with no prior noise to
//     exploit, one measured step at a time;
//   - the optimum under the draw over a few axes chosen at random, everything else held where the candidate's best
//     measured set has it: exploration that moves several axes together, so a pair that loses alone can be found,
//     while staying near enough to what was measured for its reading to teach something;
//   - every configuration of it already measured.
//
// The draw's optimum over every axis at once is not proposed: with forty-odd untried axes it takes every one at its
// luckiest draw, and predicts a cost no configuration has.  With probability TOP_TWO_BETA, and unless that
// configuration is already well measured, it is what is measured; otherwise fresh draws are taken until one names a
// different winner, the challenger, which is measured instead.
//
// The draw's departure from the posterior mean is scaled by a Student-t factor, so that the search sometimes looks
// much further than the model's own error bars say: the model is a ranking device, and where it is wrong only
// measuring can say so.

#pragma once

#include "Candidates.h"
#include "Model.h"
#include "Optimize.h"
#include "Publish.h"
#include "Readings.h"

#include <array>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace tune {

// The chance the draw's own winner is measured rather than a challenger.
inline constexpr double TOP_TWO_BETA = 0.5;

// Calls after which a configuration is measured again only as a challenger: more calls of it buy little.
inline constexpr u32 WELL_MEASURED = 6;

// Degrees of freedom of the Student-t factor draws are scaled by.
inline constexpr double DRAW_TAIL_DOF = 8;

// How likely an exploring proposal is to free one, two, three or four axes.
inline constexpr std::array<double, 4> FREE_AXES_WEIGHTS{0.5, 0.3, 0.12, 0.08};

// The most axes the mean's proposal may move from a candidate's best measured set in one step.
inline constexpr size_t MEAN_STEP_AXES = 4;

// How much more likely an exploring proposal is to free an axis all of whose positions are untried on the FFT's type.
inline constexpr double UNTRIED_WEIGHT = 4;

// At one grid point, how many candidates are optimised in full: those the posterior mean values best at their best
// measured set (or their defaults), and a few others at random, so that none is passed over for good.
inline constexpr size_t SCREENED = 24;
inline constexpr size_t SCREEN_EXTRA = 4;

// What one fit of the model says of each candidate under its posterior mean, worked out once per fit: its optimum,
// and what it costs at its best measured set (or its defaults), which screens it.
class MeanOptima {
public:
  [[nodiscard]] const Optimum* find(const Candidate* c) const;
  void put(const Candidate* c, Optimum o);

  // The screening value of every candidate, worked out on first use across the machine's threads.
  [[nodiscard]] double screenValue(const Candidate* c) const;
  void screen(const std::vector<const Candidate*>& candidates, const std::function<double(const Candidate*)>& value);

private:
  mutable std::mutex mutex_;
  std::map<const Candidate*, Optimum> optima_;
  std::map<const Candidate*, double> screened_;
};

struct AcquireContext {
  const Model& model;
  const RunScope& scope;
  const std::vector<std::vector<Candidate>>& candidates;  // per kind, as scope.grids
  const Readings& readings;

  // What a candidate's search may offer: the occupancy model's register limits for its kernels, and the tail queues.
  std::function<AxisFilter(const Candidate&)> filter;

  // Whether a configuration is never to be measured: it failed, or a process died holding it.
  std::function<bool(const Candidate&, const UseConfig&)> blocked;

  // Multiplies every draw's departure from the mean, for calibration.
  double scale = 1;

  // The mean optima of this fit; filled as candidates are first asked about.
  MeanOptima* means = nullptr;
};

struct Choice {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  double predictedUs = 0;  // under the draw that chose it
  std::string why{};
};

// The configuration to measure next; nothing where every draw at every point tried names one already well measured.
[[nodiscard]] std::optional<Choice> choose(const AcquireContext& context, std::mt19937_64& rng);

// The relative regret of the publication, the median over `draws` posterior draws: the weight-averaged amount by which
// what production would run costs more, under each draw, than the cheapest the draw sees, relative to its cost.  The
// median, since one draw scaled far out by its heavy tail would otherwise decide it.
// Infinite where some grid point has no line.
[[nodiscard]] double expectedRegret(const AcquireContext& context, const Publication& publication, std::mt19937_64& rng,
                                    u32 draws);

}  // namespace tune
