// Copyright (C) Jason Lynch

#include "test.h"

#include "MinSum.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace tune;

namespace {

// A small reproducible generator.
struct Lcg {
  u64 state;
  double uniform() {
    state = state * 6'364'136'223'846'793'005ull + 1'442'695'040'888'963'407ull;
    return double(state >> 11) * 0x1.0p-53;
  }
  size_t below(size_t n) { return size_t(uniform() * double(n)) % n; }
};

MinSum randomProblem(Lcg& rng, size_t vars, double pairChance, double forbidChance) {
  MinSum p;
  for (size_t v = 0; v < vars; ++v) {
    size_t const d = 2 + rng.below(3);
    std::vector<double> u(d);
    for (double& x : u) { x = rng.uniform() < forbidChance ? FORBIDDEN : rng.uniform(); }
    u[rng.below(d)] = rng.uniform();  // never all forbidden
    p.unary.push_back(std::move(u));
  }
  for (size_t a = 0; a < vars; ++a) {
    for (size_t b = 0; b < vars; ++b) {
      if (a == b || rng.uniform() >= pairChance) { continue; }
      MinSum::Pair pair{.a = a, .b = b, .table = {}};
      for (size_t i = 0; i < p.unary[a].size() * p.unary[b].size(); ++i) { pair.table.push_back(rng.uniform() - 0.5); }
      p.pairs.push_back(std::move(pair));
    }
  }
  return p;
}

double bruteForce(const MinSum& p) {
  std::vector<size_t> at(p.unary.size(), 0);
  double best = FORBIDDEN;
  while (true) {
    best = std::min(best, evaluate(p, at));
    size_t v = 0;
    while (v < at.size() && ++at[v] == p.unary[v].size()) { at[v++] = 0; }
    if (v == at.size()) { break; }
  }
  return best;
}

}  // namespace

TEST(min_sum_is_exact_against_brute_force) {
  Lcg rng{12'345};
  for (int trial = 0; trial < 60; ++trial) {
    MinSum const p = randomProblem(rng, 3 + rng.below(6), 0.3, 0.1);
    MinSumResult const r = minimize(p);
    CHECK(r.exact);
    double const truth = bruteForce(p);
    CHECK(std::abs(r.value - truth) < 1e-9 || (std::isinf(truth) && std::isinf(r.value)));
    CHECK(std::abs(evaluate(p, r.assignment) - r.value) < 1e-9 || std::isinf(r.value));
  }
}

TEST(min_sum_falls_back_to_a_local_search_when_entangled) {
  Lcg rng{777};
  MinSum const p = randomProblem(rng, 9, 1.0, 0.0);
  MinSumResult const r = minimize(p, 16);
  CHECK(!r.exact);
  CHECK(std::abs(evaluate(p, r.assignment) - r.value) < 1e-9);
  CHECK(r.value <= evaluate(p, std::vector<size_t>(9, 0)) + 1e-12);
  CHECK(r.value >= bruteForce(p) - 1e-12);
}

TEST(min_sum_respects_forbidden_values) {
  MinSum p;
  p.unary = {{0.0, -1.0}, {0.0, 0.0}};
  // b may not be 0 when a is 1.
  p.pairs.push_back({.a = 0, .b = 1, .table = {0.0, 0.0, FORBIDDEN, 0.5}});
  MinSumResult const r = minimize(p);
  CHECK_EQ(r.assignment[0], size_t{1});
  CHECK_EQ(r.assignment[1], size_t{1});
  CHECK(std::abs(r.value + 0.5) < 1e-12);
}

TEST(min_sum_with_factors_is_exact_against_brute_force) {
  Lcg rng{4242};
  for (int trial = 0; trial < 60; ++trial) {
    MinSum p = randomProblem(rng, 3 + rng.below(5), 0.2, 0.0);
    size_t const n = p.unary.size();
    for (size_t f = 0; f < 1 + rng.below(3); ++f) {
      // A factor over three distinct variables, in no particular order, forbidding some of its combinations.
      std::vector<size_t> scope;
      while (scope.size() < 3) {
        size_t const v = rng.below(n);
        if (std::ranges::find(scope, v) == scope.end()) { scope.push_back(v); }
      }
      size_t size = 1;
      for (size_t const v : scope) { size *= p.unary[v].size(); }
      std::vector<double> table(size);
      for (double& x : table) { x = rng.uniform() < 0.3 ? FORBIDDEN : rng.uniform() - 0.5; }
      table[rng.below(size)] = 0;
      p.factors.push_back({.scope = std::move(scope), .table = std::move(table)});
    }
    MinSumResult const r = minimize(p);
    CHECK(r.exact);
    double const truth = bruteForce(p);
    CHECK(std::abs(r.value - truth) < 1e-9 || (std::isinf(truth) && std::isinf(r.value)));
    CHECK(std::abs(evaluate(p, r.assignment) - r.value) < 1e-9 || std::isinf(r.value));
  }
}

TEST(min_sum_local_search_starts_where_it_is_told) {
  // Only (1, 1, 1) is allowed, and a search starting anywhere else stays there: every step from it is forbidden too.
  MinSum p;
  p.unary = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  std::vector<double> table(8, FORBIDDEN);
  table[7] = 1;
  p.factors.push_back({.scope = {0, 1, 2}, .table = std::move(table)});
  p.starts.push_back({1, 1, 1});
  MinSumResult const r = minimize(p, 1);
  CHECK(!r.exact);
  CHECK(std::abs(r.value - 1) < 1e-12);
}
