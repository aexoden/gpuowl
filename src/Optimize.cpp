// Copyright (C) Jason Lynch

#include "Optimize.h"

#include "MinSum.h"

#include <algorithm>

namespace tune {

namespace {

// How many times a branch is minimised again after fitting moved it.
constexpr u32 BRANCH_ROUNDS = 6;

// What leaving an axis's reference costs on top of the draw: nothing a reading could resolve, only enough that a tie --
// which the mean of the model is everywhere nothing was measured -- goes to the default.
constexpr double MOVE_COST = 1e-9;

Optimum minimizeBranch(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                       UseConfig background, const Subspace* subspace) {
  const Env& env = model.env();
  Optimum best{.options = {}, .predicted = FORBIDDEN, .exact = true};
  bool exact = true;

  for (u32 round = 0; round < BRANCH_ROUNDS; ++round) {
    std::vector<Axis> axes = axesOf(env, subject.fft, background, filter);
    std::erase_if(axes, [](const Axis& a) { return a.option->key == "INPLACE"; });

    Energy const e = model.energy(draw, subject, background, axes);
    MinSum problem{.unary = e.unary, .pairs = {}};
    for (size_t a = 0; a < axes.size(); ++a) {
      bool const held = subspace && std::ranges::find(subspace->free, axes[a].name) == subspace->free.end();
      std::pair<int, int> const at =
        held ? positionOf(env, subject.fft, subspace->around, axes[a]) : std::pair<int, int>{};
      for (size_t p = 0; p < problem.unary[a].size(); ++p) {
        if (held && axes[a].values[p] != at) {
          problem.unary[a][p] = FORBIDDEN;
        } else if (p != axes[a].reference) {
          problem.unary[a][p] += MOVE_COST;
        }
      }
      // A held axis whose position the background does not offer stays where the background has it.
      if (held && std::ranges::all_of(problem.unary[a], [](double v) { return v == FORBIDDEN; })) {
        problem.unary[a] = e.unary[a];
        for (size_t p = 0; p < problem.unary[a].size(); ++p) {
          if (p != axes[a].current) { problem.unary[a][p] = FORBIDDEN; }
        }
      }
    }
    for (const Energy::Pair& p : e.pairs) { problem.pairs.push_back({.a = p.a, .b = p.b, .table = p.table}); }
    MinSumResult const r = minimize(problem);
    exact = exact && r.exact;

    UseConfig placed = background;
    for (size_t a = 0; a < axes.size(); ++a) { place(placed, axes[a], r.assignment[a]); }
    UseConfig const next = fitted(env, subject.fft, placed);

    double const predicted = model.predict(draw, subject, next);
    if (predicted < best.predicted) { best = {.options = next, .predicted = predicted, .exact = exact}; }
    if (next == background) { break; }
    background = next;
  }
  best.exact = exact;
  return best;
}

}  // namespace

Optimum optimize(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                 const Subspace* subspace) {
  const Env& env = model.env();
  UseConfig const start = subspace ? fitted(env, subject.fft, subspace->around) : UseConfig{};
  bool const branching = !subspace || std::ranges::find(subspace->free, "INPLACE") != subspace->free.end();
  std::vector<UseConfig> branches;
  for (const Axis& axis : axesOf(env, subject.fft, start, filter)) {
    if (axis.option->key != "INPLACE" || !branching) { continue; }
    for (size_t i = 0; i < axis.values.size(); ++i) {
      UseConfig branch = start;
      place(branch, axis, i);
      branches.push_back(fitted(env, subject.fft, branch));
    }
  }
  if (branches.empty()) { branches.push_back(start); }

  Optimum best{.options = {}, .predicted = FORBIDDEN, .exact = true};
  bool exact = true;
  for (const UseConfig& branch : branches) {
    Optimum const o = minimizeBranch(model, draw, subject, filter, branch, subspace);
    exact = exact && o.exact;
    if (o.predicted < best.predicted) { best = o; }
  }
  best.exact = exact;
  return best;
}

}  // namespace tune
