// Copyright (C) Jason Lynch

#include "MinSum.h"

#include <algorithm>
#include <numeric>
#include <optional>

namespace tune {

namespace {

using Factor = MinSum::Factor;

// Where `assignment` reads `f`'s table.
size_t indexIn(const Factor& f, const std::vector<size_t>& sizes, const std::vector<size_t>& assignment) {
  size_t at = 0;
  for (size_t const u : f.scope) { at = at * sizes[u] + assignment[u]; }
  return at;
}

// What eliminating one variable left behind: its best value for each assignment of the variables it was tied to.
struct Elimination {
  size_t var = 0;
  std::vector<size_t> scope;
  std::vector<u32> best;
};

// Odometer over the joint values of `scope`.
class Odometer {
public:
  Odometer(const std::vector<size_t>& scope, const std::vector<size_t>& sizes) : scope_{scope}, sizes_{sizes} {
    digits_.assign(scope.size(), 0);
  }

  [[nodiscard]] const std::vector<size_t>& digits() const { return digits_; }

  // False once every combination has been visited.
  bool next() {
    for (size_t i = digits_.size(); i-- > 0;) {
      if (++digits_[i] < sizes_[scope_[i]]) { return true; }
      digits_[i] = 0;
    }
    return false;
  }

private:
  const std::vector<size_t>& scope_;
  const std::vector<size_t>& sizes_;
  std::vector<size_t> digits_;
};

std::optional<MinSumResult> eliminate(const MinSum& problem, size_t tableLimit) {
  size_t const n = problem.unary.size();
  std::vector<size_t> sizes(n);
  for (size_t v = 0; v < n; ++v) { sizes[v] = problem.unary[v].size(); }

  std::vector<Factor> factors;
  for (size_t v = 0; v < n; ++v) { factors.push_back({.scope = {v}, .table = problem.unary[v]}); }
  for (const MinSum::Pair& p : problem.pairs) {
    if (p.a < p.b) {
      factors.push_back({.scope = {p.a, p.b}, .table = p.table});
    } else {
      Factor f{.scope = {p.b, p.a}, .table = std::vector<double>(p.table.size())};
      for (size_t i = 0; i < sizes[p.a]; ++i) {
        for (size_t j = 0; j < sizes[p.b]; ++j) { f.table[j * sizes[p.a] + i] = p.table[i * sizes[p.b] + j]; }
      }
      factors.push_back(std::move(f));
    }
  }
  factors.insert(factors.end(), problem.factors.begin(), problem.factors.end());

  std::vector<bool> alive(n, true);
  std::vector<Elimination> eliminated;
  double constant = 0;

  auto neighbours = [&](size_t v) {
    std::vector<size_t> out;
    for (const Factor& f : factors) {
      if (std::ranges::find(f.scope, v) == f.scope.end()) { continue; }
      for (size_t const u : f.scope) {
        if (u != v) { out.push_back(u); }
      }
    }
    std::ranges::sort(out);
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  };
  auto tableSize = [&](const std::vector<size_t>& scope) {
    size_t size = 1;
    for (size_t const u : scope) {
      if (size > tableLimit) { break; }
      size *= sizes[u];
    }
    return size;
  };

  for (size_t step = 0; step < n; ++step) {
    // The variable whose elimination makes the smallest table.
    size_t pick = n;
    size_t pickSize = 0;
    for (size_t v = 0; v < n; ++v) {
      if (!alive[v]) { continue; }
      size_t const size = tableSize(neighbours(v)) * sizes[v];
      if (pick == n || size < pickSize) {
        pick = v;
        pickSize = size;
      }
    }
    if (pickSize > tableLimit) { return {}; }

    size_t const v = pick;
    std::vector<size_t> const scope = neighbours(v);
    std::vector<Factor> touching;
    std::erase_if(factors, [&](Factor& f) {
      if (std::ranges::find(f.scope, v) == f.scope.end()) { return false; }
      touching.push_back(std::move(f));
      return true;
    });

    // Each touching factor's stride for each variable it reads.
    struct Reader {
      const Factor* factor;
      std::vector<std::pair<size_t, size_t>> strides;  // (position in scope or SIZE_MAX for v, stride)
    };
    std::vector<Reader> readers;
    for (const Factor& f : touching) {
      Reader r{.factor = &f, .strides = {}};
      size_t stride = 1;
      for (size_t k = f.scope.size(); k-- > 0;) {
        size_t const u = f.scope[k];
        size_t const where = u == v ? SIZE_MAX : size_t(std::ranges::find(scope, u) - scope.begin());
        r.strides.emplace_back(where, stride);
        stride *= sizes[u];
      }
      readers.push_back(std::move(r));
    }

    Factor result{.scope = scope, .table = {}};
    Elimination record{.var = v, .scope = scope, .best = {}};
    Odometer odometer{scope, sizes};
    do {
      const std::vector<size_t>& digits = odometer.digits();
      double best = FORBIDDEN;
      u32 bestValue = 0;
      for (size_t x = 0; x < sizes[v]; ++x) {
        double sum = 0;
        for (const Reader& r : readers) {
          size_t at = 0;
          for (auto const& [where, stride] : r.strides) { at += (where == SIZE_MAX ? x : digits[where]) * stride; }
          sum += r.factor->table[at];
        }
        if (sum < best) {
          best = sum;
          bestValue = u32(x);
        }
      }
      result.table.push_back(best);
      record.best.push_back(bestValue);
    } while (!scope.empty() && odometer.next());

    alive[v] = false;
    eliminated.push_back(std::move(record));
    if (scope.empty()) {
      constant += result.table.front();
    } else {
      factors.push_back(std::move(result));
    }
  }

  MinSumResult out{.assignment = std::vector<size_t>(n, 0), .value = constant, .exact = true};
  for (size_t k = eliminated.size(); k-- > 0;) {
    const Elimination& e = eliminated[k];
    size_t at = 0;
    for (size_t const u : e.scope) { at = at * sizes[u] + out.assignment[u]; }
    out.assignment[e.var] = e.best[at];
  }
  return out;
}

// Coordinate descent from several starts.
MinSumResult descend(const MinSum& problem) {
  size_t const n = problem.unary.size();
  std::vector<size_t> sizes(n);
  for (size_t v = 0; v < n; ++v) { sizes[v] = problem.unary[v].size(); }
  std::vector<std::vector<const MinSum::Pair*>> touching(n);
  for (const MinSum::Pair& p : problem.pairs) {
    touching[p.a].push_back(&p);
    touching[p.b].push_back(&p);
  }
  std::vector<std::vector<const Factor*>> reading(n);
  for (const Factor& f : problem.factors) {
    for (size_t const u : f.scope) { reading[u].push_back(&f); }
  }

  auto local = [&](size_t v, size_t x, std::vector<size_t>& at) {
    double sum = problem.unary[v][x];
    for (const MinSum::Pair* p : touching[v]) {
      size_t const nb = problem.unary[p->b].size();
      sum += p->a == v ? p->table[x * nb + at[p->b]] : p->table[at[p->a] * nb + x];
    }
    size_t const was = at[v];
    at[v] = x;
    for (const Factor* f : reading[v]) { sum += f->table[indexIn(*f, sizes, at)]; }
    at[v] = was;
    return sum;
  };

  std::vector<std::vector<size_t>> starts = problem.starts;
  starts.emplace_back(n, 0);
  std::vector<size_t> greedy(n);
  for (size_t v = 0; v < n; ++v) {
    greedy[v] = size_t(std::ranges::min_element(problem.unary[v]) - problem.unary[v].begin());
  }
  starts.push_back(greedy);
  for (u64 seed = 1; seed <= 6; ++seed) {
    std::vector<size_t> s(n);
    u64 state = seed * 0x9e3779b97f4a7c15ull;
    for (size_t v = 0; v < n; ++v) {
      state = state * 6'364'136'223'846'793'005ull + 1'442'695'040'888'963'407ull;
      s[v] = size_t((state >> 33) % problem.unary[v].size());
    }
    starts.push_back(std::move(s));
  }

  MinSumResult best{};
  for (std::vector<size_t> at : starts) {
    for (bool moved = true; moved;) {
      moved = false;
      for (size_t v = 0; v < n; ++v) {
        size_t bestX = at[v];
        double bestCost = local(v, at[v], at);
        for (size_t x = 0; x < problem.unary[v].size(); ++x) {
          if (double const c = local(v, x, at); c < bestCost) {
            bestCost = c;
            bestX = x;
          }
        }
        if (bestX != at[v]) {
          at[v] = bestX;
          moved = true;
        }
      }
    }
    if (double const value = evaluate(problem, at); value < best.value) { best = {.assignment = at, .value = value}; }
  }
  return best;
}

}  // namespace

double evaluate(const MinSum& problem, const std::vector<size_t>& assignment) {
  double sum = 0;
  for (size_t v = 0; v < problem.unary.size(); ++v) { sum += problem.unary[v][assignment[v]]; }
  for (const MinSum::Pair& p : problem.pairs) {
    sum += p.table[assignment[p.a] * problem.unary[p.b].size() + assignment[p.b]];
  }
  std::vector<size_t> sizes(problem.unary.size());
  for (size_t v = 0; v < sizes.size(); ++v) { sizes[v] = problem.unary[v].size(); }
  for (const Factor& f : problem.factors) { sum += f.table[indexIn(f, sizes, assignment)]; }
  return sum;
}

MinSumResult minimize(const MinSum& problem, size_t tableLimit) {
  if (problem.unary.empty()) { return {.assignment = {}, .value = 0, .exact = true}; }
  if (std::optional<MinSumResult> exact = eliminate(problem, tableLimit)) { return *exact; }
  return descend(problem);
}

}  // namespace tune
