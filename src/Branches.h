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

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace tune {

// The most assignments of its ancestors' states an option is searched under.  SMEM_SPILL depends on every register
// limit, so with the whole register ladder offered it would have 257^4 of them; with the occupancy model's few
// candidates, a few thousand.
inline constexpr size_t MAX_CONTEXTS = size_t{1} << 16;

// How many random contexts an option with more than MAX_CONTEXTS is probed at, besides its defaults and each ancestor
// moved alone.
inline constexpr size_t CONTEXT_PROBES = 4096;

// How many leaves legalBranches() walks before it reports a lower bound.
inline constexpr size_t MAX_BRANCH_WALK = size_t{1} << 20;

// How many structural branches an entry has: exactly, or where the walk stopped short, at least.
struct BranchCount {
  size_t legal = 0;
  bool complete = true;
};

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

    [[nodiscard]] bool operator==(const Context&) const = default;
  };
  // The distinct contexts, and which of them each assignment of the ancestors' states is, the last ancestor fastest:
  // thousands of assignments share a few dozen contexts.
  std::vector<Context> kinds{};
  std::vector<u32> contextKinds{};

  [[nodiscard]] size_t contexts() const { return contextKinds.size(); }
  [[nodiscard]] const Context& context(size_t i) const { return kinds[contextKinds[i]]; }

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

  // The options left at their defaults because they depend on more than MAX_CONTEXTS assignments, and were seen to
  // apply in some of them: a search of the space is exact only where this is empty.  One seen in none of its probed
  // assignments (CONTEXT_PROBES) is taken not to apply on this device at all.
  [[nodiscard]] const std::vector<std::string>& unsearched() const { return unsearched_; }

  // The context of `v` under `states`, which need hold only its ancestors' states.
  [[nodiscard]] size_t contextOf(size_t v, const std::vector<size_t>& states) const;

  // Whether every variable's state is one its ancestors allow it.
  [[nodiscard]] bool legal(const std::vector<size_t>& states) const;

  // Whether the ancestors' states `context` stands for are legal among themselves: whether some configuration has
  // that context for `v`.
  [[nodiscard]] bool legalContext(size_t v, size_t context) const;

  // The distinct assignments of the structural variables' states that legal configurations have; worked out once.
  [[nodiscard]] BranchCount legalBranches() const;

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

  mutable std::once_flag branchesOnce_;
  mutable BranchCount branches_;
};

// The space of `fft` under `filter`, built once and shared.
[[nodiscard]] std::shared_ptr<const ConditionalSpace> spaceOf(const Env& env, const FFTConfig& fft,
                                                              const AxisFilter& filter);

}  // namespace tune
