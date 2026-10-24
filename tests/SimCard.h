// Copyright (C) Jason Lynch

// A simulated card for the tuning loop: a cost for every configuration of every FFT, built from terms whose best
// combination is known, and a Device that measures it with noise, recording in a real database as the device would.
//
// The cost of a configuration is the FFT's base cost times the factor of every term that holds: a term holds on the
// FFTs it names where every option value it needs is in effect.  Every option no term needs is neutral, so an entry's
// best is among its defaults and the needs of every set of its terms taken together -- which is how the landscape's
// optimum is known without searching it.

#pragma once

#include "Candidates.h"
#include "Prior.h"
#include "Publish.h"
#include "Tuner.h"

#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace sim {

using namespace tune;

struct Term {
  std::string name;
  std::function<bool(const FFTConfig&)> where;  // the FFTs it holds on
  UseConfig needs;                              // every key at this value, in effect
  double factor = 1;
};

struct Landscape {
  std::string name;
  std::function<double(const FFTConfig&)> base;  // microseconds per iteration at the defaults
  std::vector<Term> terms;
  double noise = 0.003;  // per call, relative
};

// The stated prior, FP64 30% dearer: the loop's own prior is right about the shapes and wrong about the types.
inline double priorBase(const FFTConfig& fft) { return statedPrior(fft.shape) * (fft.FFT_FP64 ? 1.3 : 1.0); }

// Whether `key` is in effect at `value` in `options`.
inline bool inEffect(const Env& env, const FFTConfig& fft, const UseConfig& options, const std::string& key,
                     const std::string& value) {
  UseConfig const canonical = canonicalConfig(env, fft, options);
  const Option* const o = findOption(key);
  if (!o || !o->appliesTo(env, fft, canonical) || o->isInert(env, fft, canonical)) { return false; }
  return useValue(canonical, key, o->defaultFor(env, fft, canonical)) == std::stoi(value);
}

inline double truth(const Landscape& l, const Env& env, const FFTConfig& fft, const UseConfig& options) {
  double us = l.base(fft);
  for (const Term& t : l.terms) {
    if (!t.where(fft)) { continue; }
    bool holds = true;
    for (const auto& [key, value] : t.needs) { holds = holds && inEffect(env, fft, options, key, value); }
    if (holds) { us *= t.factor; }
  }
  return us;
}

// An FFT's best: its defaults, or the needs of some set of the terms that hold on it, together.
inline std::pair<double, UseConfig> best(const Landscape& l, const Env& env, const FFTConfig& fft) {
  std::vector<const Term*> here;
  for (const Term& t : l.terms) {
    if (t.where(fft)) { here.push_back(&t); }
  }
  std::pair<double, UseConfig> out{truth(l, env, fft, {}), {}};
  for (u32 mask = 1; mask < (1u << here.size()); ++mask) {
    UseConfig options;
    bool consistent = true;
    for (size_t i = 0; i < here.size(); ++i) {
      if (!(mask >> i & 1)) { continue; }
      for (const auto& [key, value] : here[i]->needs) {
        auto const [it, added] = options.try_emplace(key, value);
        consistent = consistent && it->second == value;
      }
    }
    if (!consistent) { continue; }
    if (double const us = truth(l, env, fft, options); us < out.first) { out = {us, options}; }
  }
  return out;
}

// The workload's cost: what production would run at each grid point, weighted.  For the publication, as the card truly
// costs it; for the optimum, the cheapest any candidate holding the point can do.
inline double trueT(const Landscape& l, const Env& env, const RunScope& scope, const Publication& publication) {
  double T = 0;
  for (const Grid& g : scope.grids) {
    const KindPublication* const kind = publication.of(g.kind);
    for (const GridPoint& p : g.points) {
      const Line* const line = kind ? kind->at(p.exponent) : nullptr;
      if (!line) { return std::numeric_limits<double>::infinity(); }
      T += p.weight * truth(l, env, line->candidate->subject.fft, line->reading->options);
    }
  }
  return T;
}

inline double optimalT(const Landscape& l, const Env& env, const RunScope& scope,
                       const std::vector<std::vector<Candidate>>& candidates) {
  double T = 0;
  for (size_t k = 0; k < scope.grids.size(); ++k) {
    std::map<std::string, double> bestOf;
    for (const GridPoint& p : scope.grids[k].points) {
      double cheapest = std::numeric_limits<double>::infinity();
      for (const Candidate& c : candidates[k]) {
        if (!c.interval.contains(p.exponent)) { continue; }
        std::string const spec = c.subject.fft.spec();
        auto it = bestOf.find(spec);
        if (it == bestOf.end()) { it = bestOf.emplace(spec, best(l, env, c.subject.fft).first).first; }
        cheapest = std::min(cheapest, it->second);
      }
      T += scope.grids[k].points.empty() ? 0 : p.weight * cheapest;
    }
  }
  return T;
}

// The card itself: each call 8 s and each gate reading 3 s on its own clock, whatever the machine running it does.
class Card final : public Device {
public:
  Card(const Landscape& landscape, const Env& env, TuneDB& db, u32 sess, u64 seed) :
    landscape_{landscape}, env_{env}, db_{db}, sess_{sess}, rng_{seed} {}

  Call call(const Candidate& c, const UseConfig& options) override {
    double const us = truth(landscape_, env_, c.subject.fft, options) * (1 + landscape_.noise * noise_(rng_));
    Call out;
    out.core.stats = {.mean = us, .sd = us * 0.001, .n = 4};
    (void)db_.add(CallRow{.sess = sess_,
                          .fft = c.subject.fft.spec(),
                          .kind = c.subject.kind,
                          .exponent = c.timing,
                          .cfg = db_.internCfg(options),
                          .mean = us,
                          .sd = us * 0.001,
                          .blocks = 4,
                          .ts = clock_ += 8});
    ++calls;
    return out;
  }

  RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) override {
    (void)db_.add(RoeRow{.sess = sess_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = db_.internCfg(options),
                         .checkOk = true,
                         .z = 30,
                         .n = 9000,
                         .ts = clock_ += 3});
    return {.applicable = true, .checkOk = true, .z = 30, .n = 9000, .exponent = exponent};
  }

  [[nodiscard]] std::string held(const Candidate&, const UseConfig&) const override { return {}; }
  [[nodiscard]] bool stopped() const override { return false; }
  [[nodiscard]] u32 session() const override { return sess_; }
  [[nodiscard]] u64 now() const override { return clock_; }

  u32 calls = 0;

private:
  const Landscape& landscape_;
  Env env_;
  TuneDB& db_;
  u32 sess_;
  std::mt19937_64 rng_;
  std::normal_distribution<double> noise_{0, 1};
  u64 clock_ = 1000;
};

}  // namespace sim
