// Copyright (C) Jason Lynch

#include "Readings.h"

#include <cmath>

namespace tune {

namespace {

// What one call says of a configuration's cost, relative to it, where a single call is all there is: calls scatter
// about 2.5 times more than the blocks inside one, and about 0.5-1% on the cards measured.
constexpr double SINGLE_CALL_SE = 0.01;

// No reading is more exact than this, relative to its mean.
constexpr double SE_FLOOR = 0.001;

}  // namespace

double Reading::pessimistic() const { return mean + PESSIMISM_SIGMA * se; }

ConfigKey Readings::keyOf(const Subject& subject, const UseConfig& options) {
  return ConfigKey{
    .fft = subject.fft.spec(), .kind = subject.kind, .regime = subject.regime.label(), .options = configText(options)};
}

Readings::Readings(const TuneDB& db, u32 env, const Env& device) {
  std::map<ConfigKey, std::vector<double>> costs;

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
        costs[key].push_back(row.cost());
        observations_.push_back({.fft = *fft,
                                 .kind = row.kind,
                                 .regime = subject.regime,
                                 .options = options,
                                 .cost = row.cost(),
                                 .relSem = row.mean > 0 ? row.sd / row.mean / std::sqrt(double(row.blocks)) : 0});
      }
      break;
    case Status::Lost: break;  // says nothing of the configuration
    default: r.failed = true;
    }
  }

  for (auto& [key, r] : readings_) {
    auto const it = costs.find(key);
    if (it == costs.end()) { continue; }
    const std::vector<double>& c = it->second;
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
