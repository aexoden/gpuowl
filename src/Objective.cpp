// Copyright (C) Jason Lynch

#include "Objective.h"

#include "FFTVariants.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace tune {

namespace {

constexpr double PS_PER_US = 1e6;

Env envOf(const TuneDB& db, u32 env) {
  const DbEnv* const row = db.findEnv(env);
  return row ? row->toEnv() : Env{};
}

}  // namespace

double priorWork(u32 size) { return double(size) * std::log2(double(size)); }

double statedPriorK(enum FFT_TYPES type) {
  // The lower of two cards' readings at 118063003, rounded down: an RTX A4000 (fast FP32 and integer, FP64 at 1/32)
  // and a Tesla P100 (FP64 at 1/2), on the smallest shape of each type that holds that exponent.  Which type is
  // cheapest depends on the card by up to a factor of eight, so the constants are only an ordering for a cold start;
  // the first row of a type replaces its constant.  The last three types are not ones production chooses among, so no
  // prior ever prices them, and they borrow from their nearest relative.
  switch (type) {
  case FFT64: return 6.25;
  case FFT3161: return 17.7;
  case FFT3261: return 17.0;
  case FFT61: return 11.6;
  case FFT323161: return 27.5;
  case FFT6431: return 13.9;
  case FFT3231: return 17.0;
  case FFT31: return 11.6;
  case FFT32: return 17.0;
  }
  return 6.25;
}

Prior::Prior(const TuneDB& db, u32 env) {
  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || !row.m.ok()) { continue; }
    if (auto const fft = parseFft(row.fft)) { add(fft->shape, row.m.cost()); }
  }
}

void Prior::add(const FFTShape& shape, double cost) {
  double const k = cost * PS_PER_US / priorWork(shape.size());
  auto const [at, fresh] = measured_[shape.fft_type].try_emplace(shape.size(), k);
  if (!fresh) { at->second = std::min(at->second, k); }
}

double Prior::k(const FFTShape& shape) const {
  auto const type = measured_.find(shape.fft_type);
  if (type == measured_.end()) { return statedPriorK(shape.fft_type); }

  // Not its own size's reading alone: a size read a few times may have been read only at its slowest configuration --
  // one call of a broadcast variant once priced every shape of its size 60% above the best of the size below -- and
  // every other configuration of that size would then be valued as if it could not beat it.
  const std::map<u32, double>& bySize = type->second;
  auto at = bySize.lower_bound(shape.size());

  double out = std::numeric_limits<double>::infinity();
  if (at != bySize.begin()) { out = std::prev(at)->second; }
  if (at != bySize.end() && at->first == shape.size()) {
    out = std::min(out, at->second);
    ++at;
  }
  if (at != bySize.end()) { out = std::min(out, at->second); }
  return out;
}

double Prior::cost(const FFTShape& shape) const {
  return PRIOR_OPTIMISM * k(shape) * priorWork(shape.size()) / PS_PER_US;
}

bool Prior::fitted(enum FFT_TYPES type) const { return measured_.contains(type); }

Objective::Objective(const TuneDB& db, u32 env, const RunScope& scope, Gating gating) :
  Objective(envOf(db, env), entriesFor(db, env, gating), Prior{db, env}, scope) {}

Objective::Objective(const Env& env, const RunScope& scope) : Objective(env, {}, {}, scope) {}

Objective::Objective(const Env& env, std::vector<SelectionEntry> entries, Prior prior, const RunScope& scope) :
  entries_{std::move(entries)}, prior_{std::move(prior)} {
  for (const FFTShape& shape : FFTShape::allShapes()) {
    u64 hi = 0;
    for (u32 variant : runnableVariants(env, shape)) {
      hi = std::max(hi, maxExp(FFTConfig{shape, variant, CARRY_AUTO}));
    }
    if (!hi) { continue; }

    u64 const lo = minExp(FFTConfig{shape, defaultVariant(shape), CARRY_AUTO});
    if (lo <= hi) { candidates_.push_back({shape.spec(), lo, hi, prior_.cost(shape)}); }
  }
  std::ranges::stable_sort(candidates_, [](const Candidate& a, const Candidate& b) { return a.us < b.us; });

  for (const Grid& grid : scope.grids) {
    for (const GridPoint& point : grid.points) {
      ObjectivePoint const out{grid.kind, point.exponent, point.weight, cStar(grid.kind, point.exponent)};
      if (out.cost) { T_ += out.weight * out.cost->us; }
      points_.push_back(out);
    }
  }
}

double Objective::shareOf(const std::function<bool(const ObjectivePoint&)>& which) const {
  // Each kind's grid carries a whole weight of its own, so a run over two kinds carries two.
  double out = 0;
  double total = 0;
  for (const ObjectivePoint& point : points_) {
    total += point.weight;
    out += which(point) ? point.weight : 0;
  }
  return total > 0 ? out / total : 0;
}

double Objective::unservable() const {
  return shareOf([](const ObjectivePoint& point) { return !point.cost; });
}

double Objective::measured() const {
  return shareOf([](const ObjectivePoint& point) { return point.cost && point.cost->measured(); });
}

std::optional<Cost> Objective::cStar(TestKind kind, u64 E) const {
  // Cheapest first, which is the order production walks them in, so the first that covers E is the one it would run.
  for (const SelectionEntry& entry : entries_) {
    if (entry.kind == kind && entry.emin <= E && E <= entry.reach) { return Cost{entry.cost, entry.id, entry.fft}; }
  }

  return prior(E);
}

std::optional<Cost> Objective::prior(u64 E) const {
  for (const Candidate& candidate : candidates_) {
    if (candidate.lo <= E && E <= candidate.hi) { return Cost{candidate.us, {}, candidate.fft}; }
  }

  return {};
}

}  // namespace tune
