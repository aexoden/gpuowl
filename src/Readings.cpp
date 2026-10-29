// Copyright (C) Jason Lynch

#include "Readings.h"

#include <algorithm>
#include <cmath>

namespace tune {

namespace {

// What one call says of a configuration's cost, relative to it, where a single call is all there is: calls scatter
// about 2.5 times more than the blocks inside one, and about 0.5-1% on the cards measured.
constexpr double SINGLE_CALL_SE = 0.01;

// No reading is more exact than this, relative to its mean.
constexpr double SE_FLOOR = 0.001;

// A call this much above another call of the same configuration was disturbed -- by another process wanting the CPU
// the timed loop busy-waits on, or by the card itself -- and says nothing of the configuration.  Calls scatter by well
// under 1%.
constexpr double DISTURBED = 1.15;

}  // namespace

double Reading::pessimistic() const { return mean + PESSIMISM_SIGMA * se; }

ConfigKey Readings::keyOf(const Subject& subject, const UseConfig& options) {
  return ConfigKey{
    .fft = subject.fft.spec(), .kind = subject.kind, .regime = subject.regime.label(), .options = configText(options)};
}

Readings::Readings(const TuneDB& db, u32 env, const Env& device, double powerHold) {
  struct Taken {
    double cost;
    double relSem;
  };
  std::map<ConfigKey, std::vector<Taken>> costs;

  // What each call drew, by what it shares with its call record.
  using CallOf = std::tuple<u32, std::string, TestKind, u64, u32, u64>;
  std::map<CallOf, const PowerRow*> powers;
  for (const PowerRow& p : db.powers()) {
    if (db.envOf(p.sess) == env) { powers[{p.sess, p.fft, p.kind, p.exponent, p.cfg, p.ts}] = &p; }
  }
  struct Drawn {
    double mj = 0;
    double watts = 0;
    u32 n = 0;
    u32 bound = 0;
  };
  std::map<ConfigKey, Drawn> drawn;

  for (const CallRow& row : db.calls()) {
    if (db.envOf(row.sess) != env) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (!fft || !recorded) { continue; }

    Subject const subject{.fft = *fft, .kind = row.kind, .regime = regimeOf(*fft, row.exponent)};
    UseConfig const options = canonicalConfig(device, *fft, *recorded);
    ConfigKey const key = keyOf(subject, options);

    auto [it, fresh] =
      readings_.try_emplace(key, Reading{.fft = *fft, .kind = row.kind, .regime = subject.regime, .options = options});
    Reading& r = it->second;
    r.exponent = row.exponent;

    switch (row.status) {
    case Status::Ok:
      if (row.ok()) {
        double cost = row.cost();
        auto const p = powers.find({row.sess, row.fft, row.kind, row.exponent, row.cfg, row.ts});
        if (p != powers.end()) {
          Drawn& d = drawn[key];
          d.mj += p->second->mjPerIt;
          d.watts += p->second->watts;
          ++d.n;
          // Millijoules per iteration over watts is milliseconds per iteration.
          // Only instantaneous readings set a cost: the counter and averaged readings lag behind a call (PowerSource).
          if (powerHold > 0 && p->second->capW > 0 && p->second->source == PowerSource::Instant) {
            double const bound = p->second->mjPerIt * 1000 / (powerHold * p->second->capW);
            if (bound > cost) {
              cost = bound;
              ++d.bound;
            }
          }
        }
        costs[key].push_back(
          {.cost = cost, .relSem = row.mean > 0 ? row.sd / row.mean / std::sqrt(double(row.blocks)) : 0});
      }
      break;
    case Status::Lost: break;  // says nothing of the configuration
    default: r.failed = true;
    }
  }

  // Which builds spilled: by FFT, kind and canonical set, whatever the exponent.
  std::map<std::tuple<std::string, TestKind, std::string>, std::vector<std::string>> spilled;
  for (const SpillRow& row : db.spills()) {
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (db.envOf(row.sess) != env || !fft || !recorded) { continue; }
    spilled[{fft->spec(), row.kind, configText(canonicalConfig(device, *fft, *recorded))}] = row.keys;
  }

  for (auto& [key, taken] : costs) {
    double const least = std::ranges::min(taken, {}, &Taken::cost).cost;
    std::erase_if(taken, [&](const Taken& t) { return t.cost > least * DISTURBED; });
    const Reading& r = readings_.at(key);
    for (const Taken& t : taken) {
      auto const spill = spilled.find({r.fft.spec(), r.kind, key.options});
      observations_.push_back({.fft = r.fft,
                               .kind = r.kind,
                               .regime = r.regime,
                               .options = r.options,
                               .cost = t.cost,
                               .relSem = t.relSem,
                               .spilled = spill == spilled.end() ? std::vector<std::string>{} : spill->second});
    }
  }

  for (auto& [key, r] : readings_) {
    auto const it = costs.find(key);
    if (it == costs.end()) { continue; }
    std::vector<double> c;
    for (const Taken& t : it->second) { c.push_back(t.cost); }
    r.calls = u32(c.size());
    double sum = 0;
    for (double const x : c) { sum += x; }
    r.mean = sum / double(c.size());
    double se = r.mean * SINGLE_CALL_SE;
    if (c.size() >= 2) {
      double ss = 0;
      for (double const x : c) { ss += (x - r.mean) * (x - r.mean); }
      se = std::sqrt(ss / double(c.size() - 1) / double(c.size()));
    }
    r.se = std::max(se, r.mean * SE_FLOOR);
  }

  for (auto const& [key, d] : drawn) {
    Reading& r = readings_.at(key);
    r.mjPerIt = d.mj / d.n;
    r.watts = d.watts / d.n;
    r.powerBound = d.bound;
  }

  // The soaks: each configuration's slices after the warm-ups, in the regime of the exponent it was soaked at.
  std::map<ConfigKey, std::vector<double>> slices;
  for (const SoakRow& row : db.soaks()) {
    if (db.envOf(row.sess) != env || row.round == 0) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (!fft || !recorded) { continue; }
    Subject const subject{.fft = *fft, .kind = row.kind, .regime = regimeOf(*fft, row.exponent)};
    slices[keyOf(subject, canonicalConfig(device, *fft, *recorded))].push_back(row.mean);
  }
  std::vector<double> ratios;
  for (auto const& [key, s] : slices) {
    auto const it = readings_.find(key);
    if (it == readings_.end() || it->second.calls == 0 || it->second.failed) { continue; }
    Reading& r = it->second;
    double sum = 0;
    for (double const x : s) { sum += x; }
    r.soaked = sum / double(s.size());
    r.soakSlices = u32(s.size());
    ratios.push_back(r.soaked / r.mean);
  }
  if (!ratios.empty()) {
    std::ranges::sort(ratios);
    size_t const n = ratios.size();
    calibration_ = n % 2 ? ratios[n / 2] : (ratios[n / 2 - 1] + ratios[n / 2]) / 2;
  }
  for (auto& [key, r] : readings_) {
    r.timed = r.mean;
    if (r.calls == 0) { continue; }
    if (!r.isSoaked()) {
      r.mean *= calibration_;
      r.se *= calibration_;
      continue;
    }
    const std::vector<double>& s = slices.at(key);
    double ss = 0;
    for (double const x : s) { ss += (x - r.soaked) * (x - r.soaked); }
    double const se = s.size() >= 2 ? std::sqrt(ss / double(s.size() - 1) / double(s.size())) : 0;
    r.mean = r.soaked;
    r.se = std::max(se, r.mean * SE_FLOOR);
  }
  for (Observation& o : observations_) { o.cost *= calibration_; }

  for (const RoeRow& row : db.roes()) {
    if (db.envOf(row.sess) != env) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (!fft || !recorded) { continue; }
    roes_[{fft->spec(), configText(canonicalConfig(device, *fft, *recorded))}].push_back(
      {.exponent = row.exponent, .checkOk = row.checkOk, .z = row.z, .n = row.n});
  }
}

std::vector<const Reading*> Readings::of(const Subject& subject) const {
  std::vector<const Reading*> out;
  ConfigKey const from{
    .fft = subject.fft.spec(), .kind = subject.kind, .regime = subject.regime.label(), .options = {}};
  for (auto it = readings_.lower_bound(from); it != readings_.end(); ++it) {
    if (it->first.fft != from.fft || it->first.kind != from.kind || it->first.regime != from.regime) { break; }
    out.push_back(&it->second);
  }
  return out;
}

const Reading* Readings::find(const Subject& subject, const UseConfig& options) const {
  auto const it = readings_.find(keyOf(subject, options));
  return it == readings_.end() ? nullptr : &it->second;
}

std::vector<RoeReading> Readings::roes(const FFTConfig& fft, const UseConfig& options) const {
  auto const it = roes_.find({fft.spec(), configText(options)});
  return it == roes_.end() ? std::vector<RoeReading>{} : it->second;
}

}  // namespace tune
