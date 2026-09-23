// Copyright (C) Jason Lynch

// The one number a tuning run descends: the expected iteration time of the declared workload.
//
//   c*(E) = the cost of the entry production would run at E, where some entry covers E
//         = prior(E), the optimistic estimate of the cheapest configuration that could, where none does
//   T     = sum over the grid of W(E) * c*(E)
//
// A pure function of the database, the env and the scope.  The entries are the ones the selection file would publish,
// at the cost it would rank them by, so c* describes what production would actually run rather than a second opinion
// of it.  The prior is what keeps T finite at cold start and wherever coverage stops: it never says a band is
// infinitely expensive, only that it has not been measured.

#pragma once

#include "common.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Selection.h"
#include "TuneDB.h"
#include "Tuner.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tune {

// Scales every prior below the estimate it comes from, so that an unmeasured band is valued as if it were a little
// cheaper than the evidence says.  Erring low costs one measurement that turns out not to pay; erring high can leave a
// band nobody ever measures.
inline constexpr double PRIOR_OPTIMISM = 0.9;

// The work of one iteration in the unit the prior scales by: size * log2(size).
[[nodiscard]] double priorWork(u32 size);

// A type's k, in picoseconds per unit of priorWork(), where nothing of the type has been measured.
[[nodiscard]] double statedPriorK(enum FFT_TYPES type);

// What a configuration nobody has measured is estimated to cost: k * size * log2(size), with k fitted to the measured
// row of the same type nearest in size, or the stated constant where the type has none.  The variant, the carry and
// the option set are not in the estimate -- it prices a shape, and the cheapest thing that shape could be.
class Prior {
public:
  Prior() = default;

  // Fitted to every reading `env` took that is a cost, concluded or not: the prior only orders what is measured next,
  // so one call is evidence enough, and waiting for a second would leave a type priced by the stated constant after
  // the card has already said otherwise.
  Prior(const TuneDB& db, u32 env);

  // One measured cost, in microseconds per iteration.  At a size measured more than once the cheapest reading is what
  // the fit uses, since the prior is of the cheapest configuration and not of a typical one.
  void add(const FFTShape& shape, double cost);

  // In picoseconds per unit of priorWork().  Between two measured sizes equally far away, the lower k.
  [[nodiscard]] double k(const FFTShape& shape) const;

  // Microseconds per iteration, optimism included.
  [[nodiscard]] double cost(const FFTShape& shape) const;

  // Whether anything of this type has been measured, so that k() is a fit rather than the stated constant.
  [[nodiscard]] bool fitted(enum FFT_TYPES type) const;

private:
  // Type, then size, to the lowest k measured there.
  std::map<enum FFT_TYPES, std::map<u32, double>> measured_;
};

// What c* is at one exponent, and where it came from.
struct Cost {
  double us = 0;

  // The entry production would run; empty where the prior stands in.
  std::string entry;

  // The entry's spec, or for a prior the shape it priced.
  std::string fft;

  [[nodiscard]] bool measured() const { return !entry.empty(); }
};

struct ObjectivePoint {
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  double weight = 0;

  // Nothing where no FFT this build has can run the exponent at all, whatever it costs.
  std::optional<Cost> cost;
};

class Objective {
public:
  // Against what `env` has measured, with `defaults` the lines its entries would be published beside.
  Objective(const TuneDB& db, u32 env, const RunScope& scope, const Defaults& defaults = {});

  // With nothing measured, on a device described by `env`.
  Objective(const Env& env, const RunScope& scope);

  // Microseconds per iteration.  The grid points no FFT can run carry no cost and add nothing: no measurement could
  // change what they contribute, so leaving them in could only make T infinite.
  [[nodiscard]] double T() const { return T_; }

  // The share of the weight on points no FFT can run.
  [[nodiscard]] double unservable() const;

  // The share of the weight on points a measured entry covers.
  [[nodiscard]] double measured() const;

  [[nodiscard]] const std::vector<ObjectivePoint>& points() const { return points_; }

  // At any exponent, on the grid or off it.
  [[nodiscard]] std::optional<Cost> cStar(TestKind kind, u64 E) const;
  [[nodiscard]] std::optional<Cost> prior(u64 E) const;

  [[nodiscard]] const std::vector<SelectionEntry>& entries() const { return entries_; }
  [[nodiscard]] const Prior& priorModel() const { return prior_; }

private:
  // One shape, the exponents some variant of it the device can compile is eligible for, and what the prior says it
  // costs.
  struct Candidate {
    std::string fft;
    u64 lo = 0;
    u64 hi = 0;
    double us = 0;
  };

  Objective(const Env& env, std::vector<SelectionEntry> entries, Prior prior, const RunScope& scope);

  std::vector<SelectionEntry> entries_;
  Prior prior_;

  // Cheapest first, so the first that is eligible is the prior.
  std::vector<Candidate> candidates_;

  std::vector<ObjectivePoint> points_;
  double T_ = 0;
};

}  // namespace tune
