// Copyright (C) Jason Lynch

// Minimising a sum of terms over discrete variables, each term reading one, two or a few of them.

#pragma once

#include "common.h"

#include <limits>
#include <vector>

namespace tune {

inline constexpr double FORBIDDEN = std::numeric_limits<double>::infinity();

struct MinSum {
  // Per variable, the cost of each of its values; FORBIDDEN rules a value out.
  std::vector<std::vector<double>> unary;

  struct Pair {
    size_t a = 0;
    size_t b = 0;
    std::vector<double> table{};  // [value of a * |b| + value of b]
  };
  std::vector<Pair> pairs;

  // A term over any few distinct variables, as a table in the order of `scope`, the last variable fastest.
  struct Factor {
    std::vector<size_t> scope;
    std::vector<double> table;
  };
  std::vector<Factor> factors{};

  // Where the local search starts, besides its own: assignments known to be allowed, which a problem whose FORBIDDEN
  // values encode constraints needs, since a start that breaks one rarely descends to anything allowed.
  std::vector<std::vector<size_t>> starts{};
};

struct MinSumResult {
  std::vector<size_t> assignment;
  double value = FORBIDDEN;
  bool exact = false;  // false where the problem was too entangled to eliminate and a local search answered instead
};

// The cheapest assignment.  Exact by variable elimination wherever no intermediate table exceeds `tableLimit` entries;
// otherwise coordinate descent from several starts, which is a local minimum only.
[[nodiscard]] MinSumResult minimize(const MinSum& problem, size_t tableLimit = size_t{1} << 18);

// The cost of `assignment`.
[[nodiscard]] double evaluate(const MinSum& problem, const std::vector<size_t>& assignment);

}  // namespace tune
