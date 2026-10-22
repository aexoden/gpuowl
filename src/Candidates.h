// Copyright (C) Jason Lynch

// What a run could publish: every FFT the scope admits, at every variant the device compiles and the automatic carry,
// over each carry regime of it that holds a grid point with weight.

#pragma once

#include "Eligibility.h"
#include "Model.h"
#include "Scope.h"

#include <vector>

namespace tune {

struct Candidate {
  Subject subject;
  Interval interval;              // within the workload and the FFT's table, one regime
  std::vector<GridPoint> points;  // the grid points it holds, each with its weight
  u64 timing = 0;                 // the exponent it is called at: the probe where it holds it, else its top point

  [[nodiscard]] double weight() const;
};

[[nodiscard]] std::vector<Candidate> candidatesOf(const Env& env, const RunScope& scope, TestKind kind);

}  // namespace tune
