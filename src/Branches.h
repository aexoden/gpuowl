// Copyright (C) Jason Lynch

// One FFT's legal configurations as a single space of variables, whichever structural branch they lie in.
//
// Whether an option applies, what it offers and what it defaults to depend only on the options it depends on
// (Option::dependsOn).  Here every axis that some legal configuration has is a variable, with every position it takes
// anywhere and, where some configuration goes without it, the state "inactive".  An assignment of states is a legal
// configuration exactly when each variable's state is one its ancestors' states allow it, which is a constraint on a
// few variables at a time.  A search subject to those constraints covers every branch at once without listing them,
// of which a real FFT has thousands: L2PERSISTPCT is searched with L2PERSIST=1 even where the search starts from 0.

#pragma once

#include "Axes.h"

#include <optional>
#include <vector>

namespace tune {

// The most assignments of its ancestors' states an option is searched under.  SMEM_SPILL depends on every register
// limit, so with the whole register ladder offered it would have 257^4 of them; with the occupancy model's few
// candidates, a few thousand.
inline constexpr size_t MAX_CONTEXTS = size_t{1} << 16;

struct BranchVariable {
  // Every position the variable takes anywhere, ascending; `current` and `reference` are not used.
  Axis axis{};
  bool canBeInactive = false;

  // The variables whose states decide what this one may be, ascending; every one comes before it.
  std::vector<size_t> ancestors{};

  // What the variable may be under one assignment of its ancestors' states.
  struct Context {
    std::vector<bool> offered{};        // per position
    std::optional<size_t> reference{};  // where nothing sets it; none where it is inactive
    u32 touches = 0;                    // Option::touchesFor
  };
  // Per assignment of the ancestors' states, the last ancestor fastest.
  std::vector<Context> contexts{};

  // Whether its reference and the kernels it touches are the same wherever it is active: if so, what it contributes
  // does not depend on its ancestors beyond whether it is active.
  bool steady = true;

  // Positions, then the inactive state where there is one.
  [[nodiscard]] size_t states() const { return axis.values.size() + (canBeInactive ? 1 : 0); }
  [[nodiscard]] size_t inactive() const { return axis.values.size(); }
};

class ConditionalSpace {
public:
  ConditionalSpace(const Env& env, const FFTConfig& fft, const AxisFilter& filter);

  [[nodiscard]] const std::vector<BranchVariable>& variables() const { return variables_; }

  // The options left at their defaults because they depend on more than MAX_CONTEXTS assignments: a search of the
  // space is exact only where this is empty.
  [[nodiscard]] const std::vector<std::string>& unsearched() const { return unsearched_; }

  // The context of `v` under `states`, which need hold only its ancestors' states.
  [[nodiscard]] size_t contextOf(size_t v, const std::vector<size_t>& states) const;

  // Whether every variable's state is one its ancestors allow it.
  [[nodiscard]] bool legal(const std::vector<size_t>& states) const;

  // The configuration `states` describe, fitted.
  [[nodiscard]] UseConfig configOf(const std::vector<size_t>& states) const;

  // The states of `config`, a legal assignment: a position it gives that is not offered where it lies reads as the
  // reference there, as fitted() would take it.
  [[nodiscard]] std::vector<size_t> statesOf(const UseConfig& config) const;

  // Where a search that holds `v` keeps it in `context`, which must be one where it is active: at the position
  // `canonical` gives it where `canonical` names it and that position is offered there, else at the reference.
  [[nodiscard]] size_t heldAt(size_t v, size_t context, const UseConfig& canonical) const;

  [[nodiscard]] const Env& env() const { return env_; }
  [[nodiscard]] const FFTConfig& fft() const { return fft_; }

private:
  Env env_;
  FFTConfig fft_;
  std::vector<BranchVariable> variables_;
  std::vector<std::string> unsearched_;
};

}  // namespace tune
