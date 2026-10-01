// Copyright (C) Jason Lynch

#include "Value.h"

#include "Bootstrap.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numbers>
#include <tuple>
#include <utility>

namespace tune {

double GainDist::expectedSaving(double best, double estimate) const {
  double out = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) { out += p[i] * std::max(0.0, best - estimate * (1 - GAIN_AT[i])); }
  return out;
}

double GainDist::mean() const {
  double out = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) { out += p[i] * GAIN_AT[i]; }
  return out;
}

double GainDist::chanceOfAtLeast(double gain) const {
  double out = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) { out += GAIN_AT[i] >= gain ? p[i] : 0; }
  return out;
}

void GainCounts::observe(double gain) {
  n_ += 1;

  double const g = std::max(0.0, gain);
  if (g >= GAIN_AT.back()) {
    counts_.back() += 1;
    return;
  }

  size_t const hi = size_t(std::ranges::upper_bound(GAIN_AT, g) - GAIN_AT.begin());
  size_t const lo = hi - 1;
  double const toHi = (g - GAIN_AT[lo]) / (GAIN_AT[hi] - GAIN_AT[lo]);
  counts_[lo] += 1 - toHi;
  counts_[hi] += toHi;
}

GainDist GainCounts::posterior(const GainDist& prior) const {
  GainDist out;
  for (size_t i = 0; i < GAIN_BINS; ++i) {
    out.p[i] = (GAIN_PRIOR_WEIGHT * prior.p[i] + counts_[i]) / (GAIN_PRIOR_WEIGHT + n_);
  }
  return out;
}

namespace {

// An entry's own counts over the device's posterior, mixed back with it.
[[nodiscard]] GainDist mixed(const GainDist& device, const std::map<EntryKey, GainCounts>& entries,
                             const EntryKey& entry) {
  auto const at = entries.find(entry);
  if (at == entries.end()) { return device; }

  GainDist const own = at->second.posterior(device);
  GainDist out;
  for (size_t i = 0; i < GAIN_BINS; ++i) { out.p[i] = (1 - ENTRY_MIX) * device.p[i] + ENTRY_MIX * own.p[i]; }
  return out;
}

}  // namespace

void GainModel::observe(const EntryKey& entry, double gain, GainSource source) {
  switch (source) {
  case GainSource::Move:
    all_.observe(gain);
    entries_[entry].observe(gain);
    return;
  case GainSource::Restart:
    entries_[entry].observe(gain);
    restarts_[entry].observe(gain);
    return;
  case GainSource::Combo:
    combos_.observe(gain);
    entryCombos_[entry].observe(gain);
    return;
  }
}

GainDist GainModel::forEntry(const EntryKey& entry) const { return mixed(global(), entries_, entry); }

GainDist GainModel::comboForEntry(const EntryKey& entry) const { return mixed(globalCombo(), entryCombos_, entry); }

GainDist GainModel::restartForEntry(const EntryKey& entry) const {
  auto const at = restarts_.find(entry);
  return at == restarts_.end() ? forEntry(entry) : at->second.posterior(forEntry(entry));
}

GainModel gainsOf(const TuneDB& db, u32 env) {
  const DbEnv* const dbEnv = db.findEnv(env);
  if (!dbEnv) { return {}; }
  Env const device = dbEnv->toEnv();

  std::map<EntrySet, GainSource> declared;
  auto declare = [&](const auto& row, GainSource source) {
    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (db.envOf(row.sess) != env || !opts || !fft) { return; }
    declared.try_emplace({{row.fft, row.kind, row.regime.label()}, configText(canonicalConfig(device, *fft, *opts))},
                         source);
  };
  for (const JumpRow& row : db.jumps()) { declare(row, GainSource::Restart); }
  for (const ComboRow& row : db.combos()) { declare(row, GainSource::Combo); }

  // One option set is one move however many rows it has: rows at other exponents, or spelt otherwise, are more readings
  // of it, and counting them again would read as moves that found nothing.  Pooled by calls, in the place its first
  // concluded row gives it.
  struct Pooled {
    EntryKey entry;
    GainSource source = GainSource::Move;
    double weighted = 0;
    double calls = 0;
  };
  std::vector<Pooled> sets;
  std::map<EntrySet, size_t> index;

  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || !concluded(row.m)) { continue; }

    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (!opts || !fft) { continue; }

    EntryKey const entry{row.fft, row.kind, row.regime.label()};
    EntrySet const key{entry, configText(canonicalConfig(device, *fft, *opts))};
    auto const [at, fresh] = index.try_emplace(key, sets.size());
    if (fresh) {
      auto const source = declared.find(key);
      sets.push_back({.entry = entry, .source = source != declared.end() ? source->second : GainSource::Move});
    }

    Pooled& set = sets[at->second];
    set.weighted += row.m.cost() * row.m.calls;
    set.calls += row.m.calls;
  }

  GainModel out;
  std::map<EntryKey, double> best;
  for (const Pooled& set : sets) {
    double const cost = set.weighted / set.calls;
    auto const [at, first] = best.try_emplace(set.entry, cost);
    if (first) { continue; }

    out.observe(set.entry, 1 - cost / at->second, set.source);
    at->second = std::min(at->second, cost);
  }
  return out;
}

double saving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band, double cost) {
  double out = 0;
  for (const ObjectivePoint& point : points) {
    if (point.kind != kind || !point.cost || !band.contains(point.exponent)) { continue; }
    out += point.weight * std::max(0.0, point.cost->us - cost);
  }
  return out;
}

double expectedSaving(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band, double cost,
                      const GainDist& gains) {
  double out = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) {
    if (gains.p[i] > 0) { out += gains.p[i] * saving(points, kind, band, cost * (1 - GAIN_AT[i])); }
  }
  return out;
}

std::optional<double> requiredGain(std::span<const ObjectivePoint> points, TestKind kind, const Interval& band,
                                   double cost, double worth) {
  if (cost <= 0) { return {}; }

  // saving() is piecewise linear in the cost, falling with it: between two neighbouring c*, the points dearer than the
  // cost save their weight's worth for every microsecond it drops.
  std::vector<std::pair<double, double>> dearest;
  for (const ObjectivePoint& point : points) {
    if (point.kind != kind || !point.cost || point.weight <= 0 || !band.contains(point.exponent)) { continue; }
    dearest.emplace_back(point.cost->us, point.weight);
  }
  if (dearest.empty()) { return {}; }
  std::ranges::sort(dearest, std::greater{});

  // The cost at which saving() is exactly `worth`, or for 0 the cost below which it is anything at all.
  std::optional<double> at;
  if (worth <= 0) {
    at = dearest.front().first;
  } else {
    double sum = 0;
    double weight = 0;
    for (size_t i = 0; i < dearest.size() && !at; ++i) {
      sum += dearest[i].first * dearest[i].second;
      weight += dearest[i].second;
      double const next = i + 1 < dearest.size() ? dearest[i + 1].first : 0;
      if (sum - weight * next >= worth) { at = (sum - worth) / weight; }
    }
  }
  if (!at) { return {}; }
  return std::max(0.0, 1 - *at / cost);
}

double regret(double mu, double sigma) {
  if (sigma <= 0) { return 0; }

  double const z = std::abs(mu) / sigma;
  double const phi = std::exp(-0.5 * z * z) / std::sqrt(2 * std::numbers::pi);
  double const tail = 0.5 * std::erfc(z / std::numbers::sqrt2);
  return sigma * phi - std::abs(mu) * tail;
}

double refineValue(double weight, double mu, double se, u32 calls) {
  if (weight <= 0 || calls == 0) { return 0; }

  // One more call moves the mean difference by as much as it narrows the error: the posterior mean after it is
  // distributed N(mu, se^2 - se'^2), and the regret it leaves, averaged over that, falls by exactly the regret of a
  // decision whose uncertainty is that spread.  The other side's error is in both and cancels.
  return weight * regret(mu, se / std::sqrt(double(calls) + 1));
}

std::vector<Contest> contests(std::span<const OptionSet> sets, std::span<const ObjectivePoint> points) {
  auto const cheaper = [&](size_t a, size_t b) {
    return std::tuple{sets[a].entry.cost, sets[a].entry.id} < std::tuple{sets[b].entry.cost, sets[b].entry.id};
  };

  // How many standard errors of the difference `i`'s mean stands above `chosen`'s; below 0 where it is cheaper.
  auto const behind = [&](size_t i, size_t chosen) {
    const Measurement& a = sets[i].m;
    const Measurement& c = sets[chosen].m;
    double const gap = a.cost() - c.cost();
    double const se = std::hypot(standardError(a), standardError(c));
    if (se > 0) { return gap / se; }
    return gap == 0 ? 0.0 : std::copysign(std::numeric_limits<double>::infinity(), gap);
  };

  std::map<std::pair<size_t, size_t>, double> weights;
  std::vector<size_t> eligible;
  for (const ObjectivePoint& point : points) {
    if (point.weight <= 0) { continue; }

    eligible.clear();
    for (size_t i = 0; i < sets.size(); ++i) {
      const SelectionEntry& e = sets[i].entry;
      if (e.kind == point.kind && e.emin <= point.exponent && point.exponent <= e.reach) { eligible.push_back(i); }
    }
    if (eligible.size() < 2) { continue; }

    size_t const chosen = *std::ranges::min_element(eligible, cheaper);
    auto const rival = [&](size_t i) {
      return std::tuple<double, double, const std::string&>{behind(i, chosen), sets[i].entry.cost, sets[i].entry.id};
    };
    size_t runnerUp = chosen;
    for (size_t const i : eligible) {
      if (i != chosen && (runnerUp == chosen || rival(i) < rival(runnerUp))) { runnerUp = i; }
    }
    weights[{chosen, runnerUp}] += point.weight;
  }

  std::vector<Contest> out;
  for (const auto& [pair, weight] : weights) {
    out.push_back({.chosen = pair.first, .runnerUp = pair.second, .weight = weight});
  }
  return out;
}

double refineValue(const Contest& contest, std::span<const OptionSet> sets, size_t side) {
  const Measurement& chosen = sets[contest.chosen].m;
  const Measurement& runnerUp = sets[contest.runnerUp].m;
  const Measurement& refined = side == contest.chosen ? chosen : runnerUp;

  return refineValue(contest.weight, chosen.cost() - runnerUp.cost(), standardError(refined), refined.calls);
}

bool undecided(const Contest& contest, std::span<const OptionSet> sets) {
  const Measurement& a = sets[contest.chosen].m;
  const Measurement& b = sets[contest.runnerUp].m;
  const Measurement& lo = a.cost() <= b.cost() ? a : b;
  const Measurement& hi = a.cost() <= b.cost() ? b : a;

  bool const apart = lo.cost() + RACE_CONFIDENCE * standardError(lo) < hi.cost() - RACE_CONFIDENCE * standardError(hi);
  return !apart && hi.cost() - lo.cost() > RACE_MARGIN * lo.cost();
}

std::vector<double> refineValues(std::span<const OptionSet> sets, std::span<const ObjectivePoint> points) {
  std::vector<double> out(sets.size(), 0.0);
  for (const Contest& contest : contests(sets, points)) {
    if (!undecided(contest, sets)) { continue; }
    for (size_t const side : {contest.chosen, contest.runnerUp}) {
      if (sets[side].m.calls < RACE_MAX_CALLS) { out[side] += refineValue(contest, sets, side); }
    }
  }
  return out;
}

}  // namespace tune
