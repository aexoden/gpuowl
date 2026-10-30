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

// How much dearer than its winner a duel's loser is set: nothing a cost can show, but enough that no sort puts the
// loser first.
constexpr double DUEL_HAIR = 1e-9;

}  // namespace

double callCost(const CallRow& call, const PowerRow* power, double powerHold) {
  double const timed = call.cost();
  // Only instantaneous readings set a cost: the counter and averaged readings lag behind a call (PowerSource).
  if (!power || powerHold <= 0 || power->capW <= 0 || power->source != PowerSource::Instant) { return timed; }
  // Millijoules per iteration over watts is milliseconds per iteration.
  return std::max(timed, power->mjPerIt * 1000 / (powerHold * power->capW));
}

const PowerRow* powerOf(const TuneDB& db, const CallRow& call) {
  for (auto it = db.powers().rbegin(); it != db.powers().rend(); ++it) {
    if (it->ts < call.ts) { break; }
    if (it->sess == call.sess && it->ts == call.ts && it->cfg == call.cfg && it->fft == call.fft &&
        it->kind == call.kind && it->exponent == call.exponent) {
      return &*it;
    }
  }
  return nullptr;
}

double Reading::pessimistic() const { return std::max(mean + PESSIMISM_SIGMA * se, beaten); }

ConfigKey Readings::keyOf(const Subject& subject, const UseConfig& options) {
  return ConfigKey{
    .fft = subject.fft.spec(), .kind = subject.kind, .regime = subject.regime.label(), .options = configText(options)};
}

Readings::Readings(const TuneDB& db, u32 env, const Env& device, double powerHold) :
  powerHold_{powerHold}, device_{device} {
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

  for (size_t at = 0; at < db.calls().size(); ++at) {
    const CallRow& row = db.calls()[at];
    if (db.envOf(row.sess) != env) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (!fft || !recorded) { continue; }

    Subject const subject{.fft = *fft, .kind = row.kind, .regime = regimeOf(*fft, row.exponent)};
    UseConfig const options = canonicalConfig(device, *fft, *recorded);
    ConfigKey const key = keyOf(subject, options);

    auto [it, fresh] = readings_.try_emplace(
      key, Reading{.fft = *fft, .kind = row.kind, .regime = subject.regime, .options = options, .first = at});
    Reading& r = it->second;
    r.exponent = row.exponent;

    switch (row.status) {
    case Status::Ok:
      if (row.ok()) {
        auto const p = powers.find({row.sess, row.fft, row.kind, row.exponent, row.cfg, row.ts});
        const PowerRow* const power = p == powers.end() ? nullptr : p->second;
        double const cost = callCost(row, power, powerHold);
        if (power) {
          Drawn& d = drawn[key];
          d.mj += power->mjPerIt;
          d.watts += power->watts;
          ++d.n;
          if (cost > row.cost()) { ++d.bound; }
        }
        costs[key].push_back(
          {.cost = cost, .relSem = row.mean > 0 ? row.sd / row.mean / std::sqrt(double(row.blocks)) : 0});
      }
      break;
    case Status::Lost: break;  // says nothing of the configuration
    default: r.failed = true;
    }
  }

  // What a process died holding is failed, at every exponent and kind: what took the device down once is never built or
  // published again.
  for (const TryRow& t : db.deaths()) {
    std::optional<FFTConfig> const fft = parseFft(t.fft);
    const UseConfig* const recorded = db.findCfg(t.cfg);
    if (db.envOf(t.sess) != env || !fft || !recorded) { continue; }
    died_.insert({fft->spec(), configText(canonicalConfig(device, *fft, *recorded))});
  }
  for (auto& [key, r] : readings_) {
    if (died_.contains({key.fft, key.options})) { r.failed = true; }
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

  // The duels: the latest between any two configurations stands.  Its loser -- the incumbent wherever the challenger
  // did not win -- costs at least what its winner does, through any chain of duels, unless both have been soaked: what
  // they sustained then settles it.
  struct Standing {
    ConfigKey incumbent;
    ConfigKey challenger;
    DuelOutcome outcome;
  };
  std::map<std::pair<ConfigKey, ConfigKey>, Standing> latest;
  auto keyAt = [&](const std::string& spec, TestKind kind, u64 exponent, u32 cfg) -> std::optional<ConfigKey> {
    std::optional<FFTConfig> const fft = parseFft(spec);
    const UseConfig* const recorded = db.findCfg(cfg);
    if (!fft || !recorded) { return {}; }
    return keyOf({.fft = *fft, .kind = kind, .regime = regimeOf(*fft, exponent)},
                 canonicalConfig(device, *fft, *recorded));
  };
  for (const DuelRow& row : db.duels()) {
    if (db.envOf(row.sess) != env) { continue; }
    std::optional<ConfigKey> const a = keyAt(row.fft, row.kind, row.exponent, row.cfg);
    std::optional<ConfigKey> const b = keyAt(row.challengerFft, row.kind, row.challengerExponent, row.challengerCfg);
    if (!a || !b || *a == *b) { continue; }
    auto pair = *a < *b ? std::pair{*a, *b} : std::pair{*b, *a};
    dueled_.insert(pair);
    latest.insert_or_assign(std::move(pair), Standing{.incumbent = *a, .challenger = *b, .outcome = row.outcome});
  }
  std::vector<std::pair<const Reading*, Reading*>> wins;  // winner, loser
  for (auto const& [pair, s] : latest) {
    auto const i = readings_.find(s.incumbent);
    auto const c = readings_.find(s.challenger);
    if (i == readings_.end() || c == readings_.end()) { continue; }
    Reading& incumbent = i->second;
    Reading& challenger = c->second;
    if (incumbent.calls == 0 || challenger.calls == 0 || incumbent.failed || challenger.failed) { continue; }
    if (incumbent.isSoaked() && challenger.isSoaked()) { continue; }
    if (s.outcome == DuelOutcome::Challenger) {
      wins.emplace_back(&challenger, &incumbent);
    } else {
      wins.emplace_back(&incumbent, &challenger);
    }
  }
  // A cycle of duels would raise its members a hair a round for ever; as many rounds as there are duels settle any
  // chain.
  for (size_t round = 0; round <= wins.size(); ++round) {
    bool moved = false;
    for (auto& [winner, loser] : wins) {
      double const bar = winner->pessimistic() * (1 + DUEL_HAIR);
      if (loser->pessimistic() < bar) {
        loser->beaten = bar;
        moved = true;
      }
    }
    if (!moved) { break; }
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

bool Readings::died(const FFTConfig& fft, const UseConfig& options) const {
  return died_.contains({fft.spec(), configText(options)});
}

bool Readings::dueled(const ConfigKey& a, const ConfigKey& b) const {
  return dueled_.contains(a < b ? std::pair{a, b} : std::pair{b, a});
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
