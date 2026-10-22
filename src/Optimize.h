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
  bool exact = true;     // false where some branch's minimisation fell back to a local search
};

// Where a search may move: only the named axes, everything else held where `around` has it.
struct Subspace {
  UseConfig around{};
  std::vector<std::string> free{};  // axis names (Axis::name)
};

// Each INPLACE branch is minimised on its own, since it decides which other keys apply.  Within a branch the draw's
// energy is minimised exactly over the axes that apply; the result is fitted to the option table, and where that moves
// which axes apply or what they offer, minimised again from there until it stays put.  With a subspace, only its free
// axes move, INPLACE among them only where it is named.
[[nodiscard]] Optimum optimize(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                               const Subspace* subspace = nullptr);

}  // namespace tune
