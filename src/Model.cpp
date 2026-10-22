// Copyright (C) Jason Lynch

#include "Model.h"

#include "Prior.h"
#include "TuneDB.h"

#include <algorithm>
#include <cmath>
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

double offsetOf(const FFTConfig& fft) { return std::log(priorWork(fft.shape.size())); }

std::string digit(u32 d) { return std::to_string(d); }

}  // namespace

u64 termHash(std::string_view text) {
  u64 h = FNV_OFFSET;
  for (unsigned char c : text) { h = (h ^ c) * FNV_PRIME; }
  return h;
}

Model::Model(Env env, ModelPrior prior) : env_{std::move(env)}, prior_{prior} {}

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
  auto add = [&](const std::string& text, double sd) { out.push_back({.hash = termHash(text), .sd = sd}); };

  add("T|" + type, prior_.type);
  add("Tk|" + type + '|' + toString(s.kind), prior_.kind);
  add("W|" + type + '|' + std::to_string(shape.width), prior_.shape);
  add("M|" + type + '|' + std::to_string(shape.middle), prior_.shape);
  add("H|" + type + '|' + std::to_string(shape.height), prior_.shape);
  add("Vw|" + type + '|' + digit(variant_W(s.fft.variant)), prior_.variant);
  add("Vm|" + type + '|' + digit(variant_M(s.fft.variant)), prior_.variant);
  add("Vh|" + type + '|' + digit(variant_H(s.fft.variant)), prior_.variant);
  add("Rg|" + type + '|' + regime, prior_.variant);
  add("E|" + entry, prior_.entry);

  UseConfig const canonical = canonicalConfig(env_, s.fft, options);
  std::vector<Axis> const axes = axesOf(env_, s.fft, canonical);
  std::vector<std::string> labels;
  labels.reserve(axes.size());
  for (const Axis& axis : axes) {
    std::string const label = axis.label(axis.current);
    add("G|" + label, prior_.effect);
    add("GT|" + type + '|' + label, prior_.effectType);
    add("GS|" + entry + '|' + label, prior_.effectEntry);
    labels.push_back(label);
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    for (size_t j = i + 1; j < axes.size(); ++j) {
      if (!interact(env_, s.fft, canonical, axes[i], axes[j])) { continue; }
      auto const& [lo, hi] = std::minmax(labels[i], labels[j]);
      add("P|" + type + '|' + lo + '|' + hi, prior_.pair);
    }
  }

  add("R|" + entry + '|' + configText(canonical), prior_.residual);
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

  for (const Observation& o : observations) {
    if (!(o.cost > 0)) { continue; }
    for (const Term& t : termsOf(Subject{.fft = o.fft, .kind = o.kind, .regime = o.regime}, o.options)) {
      auto const [it, fresh] = index_.try_emplace(t.hash, u32(lambda_.size()));
      if (fresh) { lambda_.push_back(1 / (t.sd * t.sd)); }
      cols_.push_back(it->second);
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
  if (!draw.seed) { return 0; }
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
  auto term = [&](const std::string& text, double sd) { return value(draw, Term{.hash = termHash(text), .sd = sd}); };

  Energy out;
  out.constant = offsetOf(s.fft) + centre_ + term("T|" + type, prior_.type) +
    term("Tk|" + type + '|' + toString(s.kind), prior_.kind) +
    term("W|" + type + '|' + std::to_string(shape.width), prior_.shape) +
    term("M|" + type + '|' + std::to_string(shape.middle), prior_.shape) +
    term("H|" + type + '|' + std::to_string(shape.height), prior_.shape) +
    term("Vw|" + type + '|' + digit(variant_W(s.fft.variant)), prior_.variant) +
    term("Vm|" + type + '|' + digit(variant_M(s.fft.variant)), prior_.variant) +
    term("Vh|" + type + '|' + digit(variant_H(s.fft.variant)), prior_.variant) +
    term("Rg|" + type + '|' + regime, prior_.variant) + term("E|" + entry, prior_.entry);

  std::vector<std::vector<std::string>> labels(axes.size());
  out.unary.resize(axes.size());
  for (size_t a = 0; a < axes.size(); ++a) {
    for (size_t p = 0; p < axes[a].values.size(); ++p) {
      std::string const label = axes[a].label(p);
      labels[a].push_back(label);
      out.unary[a].push_back(term("G|" + label, prior_.effect) + term("GT|" + type + '|' + label, prior_.effectType) +
                             term("GS|" + entry + '|' + label, prior_.effectEntry));
    }
  }

  for (size_t a = 0; a < axes.size(); ++a) {
    for (size_t b = a + 1; b < axes.size(); ++b) {
      if (!interact(env_, s.fft, background, axes[a], axes[b])) { continue; }
      Energy::Pair pair{.a = a, .b = b, .table = {}};
      for (const std::string& la : labels[a]) {
        for (const std::string& lb : labels[b]) {
          auto const& [lo, hi] = std::minmax(la, lb);
          pair.table.push_back(term("P|" + type + '|' + lo + '|' + hi, prior_.pair));
        }
      }
      out.pairs.push_back(std::move(pair));
    }
  }
  return out;
}

}  // namespace tune
