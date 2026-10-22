// Copyright (C) Jason Lynch

#include "Model.h"

#include "Prior.h"
#include "TuneDB.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>

namespace tune {

namespace {

// Where the solver stops.  The nested terms (an entry within its type's shape terms, an effect at three levels) leave
// directions that only the priors determine, which conjugate gradients converge along very slowly; they move a
// constant between levels and leave predictions alone.  On the A4000's 8300 readings (20000 terms), held-out error was
// 0.90% at 100 iterations and at 5000 alike, at 1 s against 19 s a fit.  The cost is that a draw explores those
// directions less than it should, which the caller's calibration makes up for.
constexpr double SOLVE_TOLERANCE = 1e-6;
constexpr u32 SOLVE_MAX_ITERATIONS = 300;

// A group's spread is learnt from its terms read by at least SPREAD_READINGS observations, once there are
// SPREAD_TERMS of them: the root mean square of those terms, a fitted term being shrunk toward zero by its prior by
// about what SHRINK_UNDO undoes for a well-read one.  It stays within SPREAD_RANGE of the stated prior either way.
constexpr u32 SPREAD_READINGS = 3;
constexpr size_t SPREAD_TERMS = 5;
constexpr double SHRINK_UNDO = 1.25;
constexpr double SPREAD_RANGE = 10;

constexpr u64 FNV_OFFSET = 0xcbf29ce484222325ull;
constexpr u64 FNV_PRIME = 0x100000001b3ull;

u64 splitmix(u64 x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// A standard normal variate determined by `key`, the same on every platform.
double gaussian(u64 key) {
  u64 const a = splitmix(key);
  u64 const b = splitmix(a);
  double const u1 = (double(a >> 11) + 0.5) * 0x1.0p-53;
  double const u2 = double(b >> 11) * 0x1.0p-53;
  return std::sqrt(-2 * std::log(u1)) * std::cos(2 * std::numbers::pi * u2);
}

// A stream of standard normal variates.
class Normal {
public:
  explicit Normal(u64 seed) : state_{seed} {}
  double operator()() { return gaussian(state_++ * 0x2545f4914f6cdd1dull); }

private:
  u64 state_;
};

// The stated prior's cost: what an FFT is expected to cost before this card has read anything like it.
double offsetOf(const FFTConfig& fft) { return std::log(statedPrior(fft.shape)); }

std::string digit(u32 d) { return std::to_string(d); }

}  // namespace

u64 termHash(std::string_view text) {
  u64 h = FNV_OFFSET;
  for (unsigned char c : text) { h = (h ^ c) * FNV_PRIME; }
  return h;
}

Model::Model(Env env, ModelPrior prior) : env_{std::move(env)}, prior_{prior} {}

double Model::spreadOf(Group group, double stated) const {
  double const learnt = learnt_[size_t(group)];
  return learnt > 0 ? learnt : stated;
}

bool Model::interact(const Env& env, const FFTConfig& fft, const UseConfig& background, const Axis& a, const Axis& b) {
  const Option& oa = *a.option;
  const Option& ob = *b.option;
  // A key searched alone caps one kernel, launched on its own, so its effect adds to the rest's.
  if (oa.alone || ob.alone) { return false; }
  if (oa.group == ob.group) { return true; }
  u32 const ta = oa.touchesFor(env, fft, background);
  u32 const tb = ob.touchesFor(env, fft, background);
  // Keys that reach every kernel would join everything to everything; across groups they act through their own
  // effects and the entry's departures from them.
  if ((ta | tb) & KG_GLOBAL) { return false; }
  return (ta & tb) != 0;
}

std::vector<Model::Term> Model::termsOf(const Subject& s, const UseConfig& options) const {
  const FFTShape& shape = s.fft.shape;
  std::string const type = std::to_string(int(shape.fft_type));
  std::string const regime = s.regime.label();
  std::string const entry = s.fft.spec() + '|' + toString(s.kind) + '|' + regime;

  std::vector<Term> out;
  auto add = [&](const std::string& text, Group group, double stated) {
    out.push_back({.hash = termHash(text), .group = group, .sd = spreadOf(group, stated)});
  };

  add("T|" + type, Group::Type, prior_.type);
  add("Tk|" + type + '|' + toString(s.kind), Group::Kind, prior_.kind);
  add("W|" + type + '|' + std::to_string(shape.width), Group::Shape, prior_.shape);
  add("M|" + type + '|' + std::to_string(shape.middle), Group::Shape, prior_.shape);
  add("H|" + type + '|' + std::to_string(shape.height), Group::Shape, prior_.shape);
  add("Vw|" + type + '|' + digit(variant_W(s.fft.variant)), Group::Variant, prior_.variant);
  add("Vm|" + type + '|' + digit(variant_M(s.fft.variant)), Group::Variant, prior_.variant);
  add("Vh|" + type + '|' + digit(variant_H(s.fft.variant)), Group::Variant, prior_.variant);
  add("Rg|" + type + '|' + regime, Group::Variant, prior_.variant);
  add("E|" + entry, Group::Entry, prior_.entry);

  UseConfig const canonical = canonicalConfig(env_, s.fft, options);
  std::vector<Axis> const axes = axesOf(env_, s.fft, canonical);
  // An axis at its reference has no term: a position's effect is against the FFT's own default, so a position never
  // measured is, before any reading, as good as the default give or take its prior.
  std::vector<std::string> labels;
  labels.reserve(axes.size());
  for (const Axis& axis : axes) {
    std::string const label = axis.label(axis.current);
    labels.push_back(label);
    if (axis.current == axis.reference) { continue; }
    add("G|" + label, Group::Effect, prior_.effect);
    add("GT|" + type + '|' + label, Group::EffectType, prior_.effectType);
    add("GS|" + entry + '|' + label, Group::EffectEntry, prior_.effectEntry);
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i].current == axes[i].reference) { continue; }
    for (size_t j = i + 1; j < axes.size(); ++j) {
      if (axes[j].current == axes[j].reference || !interact(env_, s.fft, canonical, axes[i], axes[j])) { continue; }
      auto const& [lo, hi] = std::minmax(labels[i], labels[j]);
      add("P|" + type + '|' + lo + '|' + hi, Group::Pair, prior_.pair);
      out.back().drawn = axes[i].option->group == axes[j].option->group;
    }
  }

  add("R|" + entry + '|' + configText(canonical), Group::Residual, prior_.residual);
  return out;
}

void Model::fit(std::span<const Observation> observations) {
  std::unordered_map<u64, double> previous;
  for (const auto& [hash, at] : index_) { previous.emplace(hash, beta_[at]); }

  index_.clear();
  lambda_.clear();
  rowStart_.assign(1, 0);
  cols_.clear();
  y_.clear();
  w_.clear();

  std::vector<Group> groups;
  std::vector<u32> readings;
  for (const Observation& o : observations) {
    if (!(o.cost > 0)) { continue; }
    for (const Term& t : termsOf(Subject{.fft = o.fft, .kind = o.kind, .regime = o.regime}, o.options)) {
      auto const [it, fresh] = index_.try_emplace(t.hash, u32(lambda_.size()));
      if (fresh) {
        lambda_.push_back(1 / (t.sd * t.sd));
        groups.push_back(t.group);
        readings.push_back(0);
      }
      cols_.push_back(it->second);
      ++readings[it->second];
    }
    rowStart_.push_back(u32(cols_.size()));
    y_.push_back(std::log(o.cost) - offsetOf(o.fft));
    double const var = prior_.callNoise * prior_.callNoise + o.relSem * o.relSem;
    w_.push_back(1 / std::max(var, prior_.noiseFloor * prior_.noiseFloor));
  }

  centre_ = 0;
  if (!y_.empty()) {
    for (double const y : y_) { centre_ += y; }
    centre_ /= double(y_.size());
    for (double& y : y_) { y -= centre_; }
  }

  std::vector<double> start(lambda_.size(), 0.0);
  for (const auto& [hash, at] : index_) {
    if (auto const it = previous.find(hash); it != previous.end()) { start[at] = it->second; }
  }
  beta_ = solve(y_, std::vector<double>(lambda_.size(), 0.0), std::move(start));

  learnSpreads(groups, readings);
}

void Model::learnSpreads(const std::vector<Group>& groups, const std::vector<u32>& readings) {
  auto stated = [&](Group g) {
    switch (g) {
    case Group::Type: return prior_.type;
    case Group::Kind: return prior_.kind;
    case Group::Shape: return prior_.shape;
    case Group::Variant: return prior_.variant;
    case Group::Entry: return prior_.entry;
    case Group::Effect: return prior_.effect;
    case Group::EffectType: return prior_.effectType;
    case Group::EffectEntry: return prior_.effectEntry;
    case Group::Pair: return prior_.pair;
    case Group::Residual: return prior_.residual;
    default: return 0.0;
    }
  };

  for (u32 round = 0; round < 3; ++round) {
    bool moved = false;
    for (size_t g = 0; g < size_t(Group::Count); ++g) {
      double ss = 0;
      size_t n = 0;
      for (size_t j = 0; j < groups.size(); ++j) {
        if (size_t(groups[j]) == g && readings[j] >= SPREAD_READINGS) {
          ss += beta_[j] * beta_[j];
          ++n;
        }
      }
      if (n < SPREAD_TERMS) { continue; }
      double const base = stated(Group(g));
      double const sd = std::clamp(SHRINK_UNDO * std::sqrt(ss / double(n)), base / SPREAD_RANGE, base * SPREAD_RANGE);
      if (std::abs(sd / spreadOf(Group(g), base) - 1) < 0.1) { continue; }
      learnt_[g] = sd;
      for (size_t j = 0; j < groups.size(); ++j) {
        if (size_t(groups[j]) == g) { lambda_[j] = 1 / (sd * sd); }
      }
      moved = true;
    }
    if (!moved) { break; }
    beta_ = solve(y_, std::vector<double>(lambda_.size(), 0.0), beta_);
  }
}

void Model::apply(const std::vector<double>& b, std::vector<double>& out) const {
  out.assign(b.size(), 0.0);
  for (size_t i = 0; i + 1 < rowStart_.size(); ++i) {
    double t = 0;
    for (u32 k = rowStart_[i]; k < rowStart_[i + 1]; ++k) { t += b[cols_[k]]; }
    t *= w_[i];
    for (u32 k = rowStart_[i]; k < rowStart_[i + 1]; ++k) { out[cols_[k]] += t; }
  }
  for (size_t j = 0; j < b.size(); ++j) { out[j] += lambda_[j] * b[j]; }
}

std::vector<double> Model::solve(const std::vector<double>& y, const std::vector<double>& m,
                                 std::vector<double> x) const {
  size_t const n = lambda_.size();
  std::vector<double> rhs(n, 0.0);
  std::vector<double> diag(n, 0.0);
  for (size_t i = 0; i + 1 < rowStart_.size(); ++i) {
    for (u32 k = rowStart_[i]; k < rowStart_[i + 1]; ++k) {
      rhs[cols_[k]] += w_[i] * y[i];
      diag[cols_[k]] += w_[i];
    }
  }
  for (size_t j = 0; j < n; ++j) {
    rhs[j] += lambda_[j] * m[j];
    diag[j] += lambda_[j];
  }

  // Preconditioned conjugate gradients, Jacobi preconditioner.
  std::vector<double> ax;
  apply(x, ax);
  std::vector<double> r(n), z(n), p(n), ap;
  double rhsNorm = 0;
  for (size_t j = 0; j < n; ++j) {
    r[j] = rhs[j] - ax[j];
    z[j] = r[j] / diag[j];
    p[j] = z[j];
    rhsNorm += rhs[j] * rhs[j];
  }
  double rz = 0;
  for (size_t j = 0; j < n; ++j) { rz += r[j] * z[j]; }

  double const tolerance = SOLVE_TOLERANCE * SOLVE_TOLERANCE * std::max(rhsNorm, 1e-30);
  u32 iter = 0;
  for (; iter < SOLVE_MAX_ITERATIONS; ++iter) {
    double rr = 0;
    for (size_t j = 0; j < n; ++j) { rr += r[j] * r[j]; }
    if (rr <= tolerance) { break; }

    apply(p, ap);
    double pap = 0;
    for (size_t j = 0; j < n; ++j) { pap += p[j] * ap[j]; }
    if (!(pap > 0)) { break; }
    double const alpha = rz / pap;
    for (size_t j = 0; j < n; ++j) {
      x[j] += alpha * p[j];
      r[j] -= alpha * ap[j];
      z[j] = r[j] / diag[j];
    }
    double rz2 = 0;
    for (size_t j = 0; j < n; ++j) { rz2 += r[j] * z[j]; }
    double const beta = rz2 / rz;
    rz = rz2;
    for (size_t j = 0; j < n; ++j) { p[j] = z[j] + beta * p[j]; }
  }
  lastIterations_ = iter;
  return x;
}

Draw Model::mean() const { return Draw{.beta = beta_, .seed = 0, .scale = 1}; }

Draw Model::draw(u64 seed, double scale) const {
  Normal normal{seed};
  std::vector<double> y = y_;
  for (size_t i = 0; i < y.size(); ++i) { y[i] += normal() / std::sqrt(w_[i]); }
  std::vector<double> m(lambda_.size());
  for (size_t j = 0; j < m.size(); ++j) { m[j] = normal() / std::sqrt(lambda_[j]); }

  std::vector<double> sample = solve(y, m, beta_);
  for (size_t j = 0; j < sample.size(); ++j) { sample[j] = beta_[j] + scale * (sample[j] - beta_[j]); }
  return Draw{.beta = std::move(sample), .seed = seed ? seed : 1, .scale = scale};
}

double Model::value(const Draw& draw, const Term& term) const {
  if (auto const it = index_.find(term.hash); it != index_.end()) { return draw.beta[it->second]; }
  if (!draw.seed || !term.drawn) { return 0; }
  return draw.scale * term.sd * gaussian(term.hash ^ splitmix(draw.seed));
}

double Model::predict(const Draw& draw, const Subject& subject, const UseConfig& options) const {
  double out = offsetOf(subject.fft) + centre_;
  for (const Term& t : termsOf(subject, options)) { out += value(draw, t); }
  return out;
}

Energy Model::energy(const Draw& draw, const Subject& s, const UseConfig& background,
                     const std::vector<Axis>& axes) const {
  const FFTShape& shape = s.fft.shape;
  std::string const type = std::to_string(int(shape.fft_type));
  std::string const regime = s.regime.label();
  std::string const entry = s.fft.spec() + '|' + toString(s.kind) + '|' + regime;
  auto term = [&](const std::string& text, Group group, double stated, bool drawn = true) {
    return value(draw, Term{.hash = termHash(text), .group = group, .sd = spreadOf(group, stated), .drawn = drawn});
  };

  Energy out;
  out.constant = offsetOf(s.fft) + centre_ + term("T|" + type, Group::Type, prior_.type) +
    term("Tk|" + type + '|' + toString(s.kind), Group::Kind, prior_.kind) +
    term("W|" + type + '|' + std::to_string(shape.width), Group::Shape, prior_.shape) +
    term("M|" + type + '|' + std::to_string(shape.middle), Group::Shape, prior_.shape) +
    term("H|" + type + '|' + std::to_string(shape.height), Group::Shape, prior_.shape) +
    term("Vw|" + type + '|' + digit(variant_W(s.fft.variant)), Group::Variant, prior_.variant) +
    term("Vm|" + type + '|' + digit(variant_M(s.fft.variant)), Group::Variant, prior_.variant) +
    term("Vh|" + type + '|' + digit(variant_H(s.fft.variant)), Group::Variant, prior_.variant) +
    term("Rg|" + type + '|' + regime, Group::Variant, prior_.variant) + term("E|" + entry, Group::Entry, prior_.entry);

  std::vector<std::vector<std::string>> labels(axes.size());
  out.unary.resize(axes.size());
  for (size_t a = 0; a < axes.size(); ++a) {
    for (size_t p = 0; p < axes[a].values.size(); ++p) {
      std::string const label = axes[a].label(p);
      labels[a].push_back(label);
      out.unary[a].push_back(p == axes[a].reference ? 0.0
                                                    : term("G|" + label, Group::Effect, prior_.effect) +
                                 term("GT|" + type + '|' + label, Group::EffectType, prior_.effectType) +
                                 term("GS|" + entry + '|' + label, Group::EffectEntry, prior_.effectEntry));
    }
  }

  for (size_t a = 0; a < axes.size(); ++a) {
    for (size_t b = a + 1; b < axes.size(); ++b) {
      if (!interact(env_, s.fft, background, axes[a], axes[b])) { continue; }
      bool const drawn = axes[a].option->group == axes[b].option->group;
      Energy::Pair pair{.a = a, .b = b, .table = {}};
      for (size_t pa = 0; pa < labels[a].size(); ++pa) {
        for (size_t pb = 0; pb < labels[b].size(); ++pb) {
          if (pa == axes[a].reference || pb == axes[b].reference) {
            pair.table.push_back(0);
            continue;
          }
          auto const& [lo, hi] = std::minmax(labels[a][pa], labels[b][pb]);
          pair.table.push_back(term("P|" + type + '|' + lo + '|' + hi, Group::Pair, prior_.pair, drawn));
        }
      }
      // A pair the draw says nothing of is no edge for the optimiser to eliminate.
      if (std::ranges::any_of(pair.table, [](double v) { return v != 0; })) { out.pairs.push_back(std::move(pair)); }
    }
  }
  return out;
}

}  // namespace tune
