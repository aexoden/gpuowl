// Copyright (C) Jason Lynch

// The options a draw of the model says are cheapest for one FFT: the search the tuner does in its head before deciding
// what to measure.

#pragma once

#include "Axes.h"
#include "Model.h"

namespace tune {

struct Optimum {
  UseConfig options{};   // canonical
  double predicted = 0;  // the draw's log cost of them, the configuration's residual included

  // Whether `options` are the draw's minimum over every legal configuration the subspace leaves free, every structural
  // branch included; false where the problem was too entangled to eliminate and a local search from several starts
  // answered instead, or where an option depends on too much to be searched (ConditionalSpace::unsearched()).
  bool exact = true;
};

// Where a search may move: the named axes, and every axis whose options depend on one of them -- a free L2PERSIST frees
// L2PERSISTPCT, a free INPLACE the keys of the layout it chooses -- everything else held where `around` has it, or at
// its default where `around` does not name it.
struct Subspace {
  UseConfig around{};
  std::vector<std::string> free{};  // axis names (Axis::name)
};

// The draw's energy, the configuration's residual aside, minimised over every structural branch at once
// (Branches.h): a dependent's effect is weighed under each value of what it depends on, not only under the value the
// search starts from.  The residual is added to the minimum found, not searched over, since it is a term of one
// configuration only.
[[nodiscard]] Optimum optimize(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                               const Subspace* subspace = nullptr);

// What the problem optimize() solves says `options` cost, the residual aside: predict() without it, give or take the
// tie-breaking cost of each axis away from its reference.  For tests.
[[nodiscard]] double conditionalEnergy(const Model& model, const Draw& draw, const Subject& subject,
                                       const AxisFilter& filter, const UseConfig& options);

}  // namespace tune
