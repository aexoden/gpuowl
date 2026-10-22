// Copyright (C) Jason Lynch

// One statistical model of what every configuration costs on one card, learnt from every call the card has made.
//
// The log of a call's cost is a sum of terms:
//
//   an offset           the stated prior's cost (Prior.h), plus the readings' mean departure from it
//   an entry intercept  per (FFT, test kind, carry regime), with a prior from the FFT's structure: its type, the type's
//                       test kind, width, middle, height, variant digits and regime, each a term of its own
//   option effects      per axis at its position (Axes.h), at four levels: the card, the FFT type, the shape, the
//                       entry
//   interactions        per pair of axes that share a kernel (or a group, for the groups that reach every kernel), per
//                       FFT type; a pair within a group takes its prior in a draw before anything has measured it, a
//                       pair across groups only once something has, which keeps a draw's optimisation tractable
//   a residual          per configuration, so that what was measured is what the model says of it
//   noise               per call
//
// Each term has a Gaussian prior around zero, so the posterior is Gaussian too: its mean is a regularised
// least-squares fit, and a draw from it is the same fit with the readings and the prior means perturbed by their own
// noise (perturb-then-optimise).  A term no reading touches has its prior in a draw, seeded by the draw and the term,
// so the model is honestly uncertain of what it has never seen.
//
// What the model is for is ranking what to measure next.  It never decides what is published: measured readings do.

#pragma once

#include "Axes.h"
#include "common.h"
#include "Eligibility.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "TestKind.h"

#include <array>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace tune {

// The prior standard deviation of each kind of term, in natural log units (0.01 is 1%).
struct ModelPrior {
  double type = 0.5;          // per FFT type
  double kind = 0.1;          // per type and test kind
  double shape = 0.2;         // per type and width, middle or height
  double variant = 0.1;       // per type and variant digit, and per type and regime
  double entry = 0.1;         // per entry, against what its structure says
  double effect = 0.1;        // an axis position, on this card
  double effectType = 0.03;   // its departure on one FFT type
  double effectShape = 0.02;  // its departure on one shape of that type, whatever the variant
  double effectEntry = 0.01;  // its departure on one entry
  double pair = 0.02;         // an interaction, per FFT type
  double residual = 0.005;    // one configuration, against everything above
  double callNoise = 0.008;   // one call, against the configuration's cost
  double noiseFloor = 0.002;  // whatever a call's own blocks say
};

// One call as the model learns from it.
struct Observation {
  FFTConfig fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  UseConfig options{};  // canonical
  double cost = 0;      // microseconds per iteration, drift-corrected
  double relSem = 0;    // the call's own standard error relative to its mean, from its blocks
};

// What the model is asked about.
struct Subject {
  FFTConfig fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
};

// A posterior draw, or the posterior mean.
struct Draw {
  std::vector<double> beta{};
  u64 seed = 0;      // 0 for the mean: then a term nothing measured contributes nothing
  double scale = 1;  // multiplies the draw's departure from the mean, for calibration and heavy tails
};

// The model's view of one subject's options under one draw: a constant, a term per axis position, and a term per
// position pair of the axes that interact.  The predicted log cost of an assignment is their sum.
struct Energy {
  double constant = 0;
  std::vector<std::vector<double>> unary;  // [axis][position]
  struct Pair {
    size_t a = 0;
    size_t b = 0;
    std::vector<double> table;  // [position of a * |b| + position of b]
  };
  std::vector<Pair> pairs;
};

class Model {
public:
  explicit Model(Env env, ModelPrior prior = {});

  // Fits the posterior mean to the observations, starting from the last fit where its terms carry over.
  void fit(std::span<const Observation> observations);

  [[nodiscard]] Draw mean() const;

  // A draw from the posterior, its departure from the mean multiplied by `scale`.
  [[nodiscard]] Draw draw(u64 seed, double scale = 1) const;

  // The predicted log cost (natural log of microseconds per iteration).
  [[nodiscard]] double predict(const Draw& draw, const Subject& subject, const UseConfig& options) const;

  // The energy of `axes`, the axes of `subject` under `background` (axesOf()), whose structural keys fix them.  The
  // residual is left out: it is a term of one configuration, which the caller can ask predict() for.
  [[nodiscard]] Energy energy(const Draw& draw, const Subject& subject, const UseConfig& background,
                              const std::vector<Axis>& axes) const;

  // Whether the two axes interact in the model.
  [[nodiscard]] static bool interact(const Env& env, const FFTConfig& fft, const UseConfig& background, const Axis& a,
                                     const Axis& b);

  [[nodiscard]] const Env& env() const { return env_; }

  // Whether an axis position (Axis::label) has been measured on any FFT of the subject's type.
  [[nodiscard]] bool measured(const Subject& subject, const std::string& label) const;
  // Whether it has been measured on any variant of the subject's shape, and on the subject itself.
  [[nodiscard]] bool measuredOnShape(const Subject& subject, const std::string& label) const;
  [[nodiscard]] bool measuredAt(const Subject& subject, const std::string& label) const;
  // The spread an entry's departure from its structure was learnt to have: how much better than its structure an FFT
  // never read is taken to possibly be.
  [[nodiscard]] double entrySpread() const { return spreadOf(Group::Entry, prior_.entry); }

  // The same for an option effect on this card.
  [[nodiscard]] double effectSpread() const { return spreadOf(Group::Effect, prior_.effect); }

  // How far an axis's effect was learnt to depart, on one shape from its effect on the type, and on one entry from
  // its effect on the shape.
  [[nodiscard]] double shapeDeparture(const std::string& axis) const {
    return spreadOf(Group::EffectShape, prior_.effectShape, axis);
  }
  [[nodiscard]] double entryDeparture(const std::string& axis) const {
    return spreadOf(Group::EffectEntry, prior_.effectEntry, axis);
  }

  // Every group's spread as now in force, for reports: "type 0.500 kind 0.100 ...".
  [[nodiscard]] std::string spreads() const;
  [[nodiscard]] size_t terms() const { return index_.size(); }
  [[nodiscard]] size_t observations() const { return y_.size(); }
  [[nodiscard]] u32 lastIterations() const { return lastIterations_; }

  // The solver's iteration cap, for experiments; SOLVE_MAX_ITERATIONS by default.
  void setMaxIterations(u32 n) { maxIterations_ = n; }

private:
  // What kind of departure a term is, for its prior: each kind's spread is learnt from its terms the readings
  // determine well.
  enum class Group : u8 {
    Type,
    Kind,
    Shape,
    Variant,
    Entry,
    Effect,
    EffectType,
    EffectShape,
    EffectEntry,
    Pair,
    Residual,
    Count
  };

  struct Term {
    u64 hash = 0;
    Group group = Group::Type;
    double sd = 0;
    bool drawn = true;   // whether, unmeasured, it takes its prior in a draw
    std::string axis{};  // an effect's axis, for its departures' spread
  };

  // The prior spread of a kind of term: as stated, or as learnt; for an effect's departure on a type or an entry, as
  // learnt for its axis where that axis has been read widely enough.
  [[nodiscard]] double spreadOf(Group group, double stated, const std::string& axis = {}) const;

  // The terms of an observation or a prediction: the structural and option terms, then the residual.
  [[nodiscard]] std::vector<Term> termsOf(const Subject& subject, const UseConfig& options) const;

  // The coefficient of a term under a draw: fitted, or for a term nothing measured, its prior.
  [[nodiscard]] double value(const Draw& draw, const Term& term) const;

  // Learns each group's spread from its well-read terms, and each effect axis's departures, and fits again where one
  // moved.
  void learnSpreads(const std::vector<Group>& groups, const std::vector<std::string>& axes,
                    const std::vector<u32>& readings);

  // (X^T W X + Lambda) b
  void apply(const std::vector<double>& b, std::vector<double>& out) const;

  // Solves (X^T W X + Lambda) beta = X^T W y + Lambda m from `start`.
  [[nodiscard]] std::vector<double> solve(const std::vector<double>& y, const std::vector<double>& m,
                                          std::vector<double> start) const;

  Env env_;
  ModelPrior prior_;

  // Terms by hash, and their prior precisions.
  std::unordered_map<u64, u32> index_;
  std::vector<double> lambda_;

  // The observations: rows of term indices (CSR), centred targets and weights.
  std::vector<u32> rowStart_;
  std::vector<u32> cols_;
  std::vector<double> y_;
  std::vector<double> w_;
  double centre_ = 0;

  // Learnt spreads, per group; 0 where the stated prior stands.
  std::array<double, size_t(Group::Count)> learnt_{};

  // Learnt per axis, for the departures some options have far more of than others: L2_STRIPING's effect depends on
  // the shape, where most options do much the same on every FFT of a type.
  std::map<std::pair<Group, std::string>, double> axisLearnt_;

  std::vector<double> beta_;
  mutable u32 lastIterations_ = 0;
  u32 maxIterations_ = 0;
};

// The model's coefficient for `term` text, hashed as the model hashes it; for tests.
[[nodiscard]] u64 termHash(std::string_view text);

}  // namespace tune
