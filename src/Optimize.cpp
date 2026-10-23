// Copyright (C) Jason Lynch

#include "Optimize.h"

#include "Branches.h"
#include "MinSum.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>

namespace tune {

namespace {

// What leaving an axis's reference costs on top of the draw: nothing a reading could resolve, only enough that a tie --
// which the mean of the model is everywhere nothing was measured -- goes to the default.
constexpr double MOVE_COST = 1e-9;

// How many FFTs' spaces are kept; a run asks about far fewer, with a few register offers each.
constexpr size_t SPACES_KEPT = 4096;

// The space of `fft` under `filter`, built once.
std::shared_ptr<const ConditionalSpace> spaceOf(const Env& env, const FFTConfig& fft, const AxisFilter& filter) {
  std::string key = env.label() + ",wg" + std::to_string(env.maxWorkGroupSize) + ",wf" +
    std::to_string(env.amdWavefront) + ',' + env.deviceName + '|' + fft.spec() + '|' + toString(filter.tailQueues);
  if (filter.registers) {
    for (auto const& [reg, values] : *filter.registers) {
      key += '|' + reg;
      for (int const v : values) { key += ',' + std::to_string(v); }
    }
  } else {
    key += "|*";
  }

  static std::mutex mutex;
  static std::map<std::string, std::shared_ptr<const ConditionalSpace>> spaces;
  {
    std::lock_guard const lock{mutex};
    if (auto const it = spaces.find(key); it != spaces.end()) { return it->second; }
  }
  auto space = std::make_shared<const ConditionalSpace>(env, fft, filter);
  std::lock_guard const lock{mutex};
  if (spaces.size() >= SPACES_KEPT) { spaces.clear(); }
  return spaces.try_emplace(key, std::move(space)).first->second;
}

// The reference and kernel groups of a variable whose contributions do not depend on its ancestors.
const BranchVariable::Context* steadyContext(const BranchVariable& v) {
  for (const BranchVariable::Context& c : v.contexts) {
    if (c.reference) { return &c; }
  }
  return nullptr;
}

// The draw's energy over the space, as a problem for minimize(): a factor per variable over its ancestors carrying its
// effects and what they allow it, and one per interacting pair.  Where only a subspace is free, a held variable keeps
// its constraint but no terms, since what it adds is the same wherever the search goes; predict() costs the result.
//
// A pair's term reads each side's reference and kernels, which for a variable that is not steady depend on its
// ancestors.  Rather than tie the pair to all of them -- SWIZ_RECOMPUTE has eight, which entangles the problem past
// elimination -- such a variable gets a companion after the space's own variables: which of its few distinct
// (reference, kernels) its context has, or that it is inactive, held to its ancestors by one factor.
struct Problem {
  std::shared_ptr<const ConditionalSpace> space;
  UseConfig around;
  std::vector<bool> free;
  MinSum minsum;

  struct Companion {
    size_t var = 0;                                     // its index in the problem
    std::vector<size_t> classOf;                        // per context of the space's variable
    std::vector<const BranchVariable::Context*> kinds;  // per class, a context of it; nullptr for inactive
  };
  std::vector<std::optional<Companion>> companions;  // per variable of the space

  // `states` of the space's variables, with their companions' states after them.
  [[nodiscard]] std::vector<size_t> assignment(std::vector<size_t> states) const {
    states.resize(minsum.unary.size(), 0);
    for (size_t v = 0; v < companions.size(); ++v) {
      if (companions[v]) { states[companions[v]->var] = companions[v]->classOf[space->contextOf(v, states)]; }
    }
    return states;
  }
};

Problem problemOf(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                  const Subspace* subspace) {
  const Env& env = model.env();
  Problem out{.space = spaceOf(env, subject.fft, filter),
              .around =
                subspace ? canonicalConfig(env, subject.fft, fitted(env, subject.fft, subspace->around)) : UseConfig{},
              .free = {},
              .minsum = {},
              .companions = {}};
  const ConditionalSpace& space = *out.space;
  const std::vector<BranchVariable>& vars = space.variables();
  size_t const n = vars.size();

  // A variable freed only because an ancestor is -- what a free SHUFL_BYTES_W or INPLACE opens -- is valued under the
  // posterior mean, not the draw, as is any pair it is part of: it moves where readings say it helps, here or on
  // other FFTs, and otherwise ties to its default.  A draw over all of them would take the luckiest prior of each
  // and predict a cost no configuration has.
  out.free.assign(n, !subspace);
  std::vector<bool> follows(n, false);
  if (subspace) {
    for (size_t v = 0; v < n; ++v) {
      bool const named = std::ranges::find(subspace->free, vars[v].axis.name) != subspace->free.end();
      follows[v] = !named && std::ranges::any_of(vars[v].ancestors, [&](size_t a) { return bool(out.free[a]); });
      out.free[v] = named || follows[v];
    }
  }
  bool const anyFollows = std::ranges::any_of(follows, [](bool f) { return f; });
  Draw const mean = anyFollows ? model.mean() : Draw{};
  auto drawFor = [&](bool follower) -> const Draw& { return follower ? mean : draw; };

  MinSum& problem = out.minsum;
  problem.unary.resize(n);
  for (size_t v = 0; v < n; ++v) {
    const BranchVariable& var = vars[v];
    std::vector<double> effect(var.axis.values.size(), 0);
    if (out.free[v]) {
      for (size_t p = 0; p < effect.size(); ++p) {
        effect[p] = model.effectOf(drawFor(follows[v]), subject, var.axis.name, var.axis.label(p)) + MOVE_COST;
      }
    }

    std::vector<double> table;
    table.reserve(var.contexts.size() * var.states());
    for (size_t context = 0; context < var.contexts.size(); ++context) {
      const BranchVariable::Context& c = var.contexts[context];
      std::optional<size_t> const held =
        !out.free[v] && c.reference ? space.heldAt(v, context, out.around) : std::optional<size_t>{};
      for (size_t s = 0; s < var.states(); ++s) {
        if (s == var.inactive()) {
          table.push_back(c.reference ? FORBIDDEN : 0);
        } else if (!c.reference || !c.offered[s] || (held && s != *held)) {
          table.push_back(FORBIDDEN);
        } else {
          table.push_back(s == *c.reference ? 0 : effect[s]);
        }
      }
    }
    if (var.ancestors.empty()) {
      problem.unary[v] = std::move(table);
    } else {
      problem.unary[v].assign(var.states(), 0);
      std::vector<size_t> scope = var.ancestors;
      scope.push_back(v);
      problem.factors.push_back({.scope = std::move(scope), .table = std::move(table)});
    }
  }

  out.companions.resize(n);
  for (size_t v = 0; v < n; ++v) {
    if (vars[v].steady) { continue; }
    Problem::Companion c{.var = problem.unary.size(), .classOf = {}, .kinds = {}};
    for (const BranchVariable::Context& context : vars[v].contexts) {
      auto const same = [&](const BranchVariable::Context* k) {
        return k ? context.reference && k->reference == context.reference && k->touches == context.touches
                 : !context.reference;
      };
      auto const at = std::ranges::find_if(c.kinds, same);
      c.classOf.push_back(size_t(at - c.kinds.begin()));
      if (at == c.kinds.end()) { c.kinds.push_back(context.reference ? &context : nullptr); }
    }
    std::vector<size_t> scope = vars[v].ancestors;
    scope.push_back(c.var);
    std::vector<double> table;
    for (size_t const k : c.classOf) {
      for (size_t i = 0; i < c.kinds.size(); ++i) { table.push_back(i == k ? 0 : FORBIDDEN); }
    }
    problem.unary.emplace_back(c.kinds.size(), 0);
    problem.factors.push_back({.scope = std::move(scope), .table = std::move(table)});
    out.companions[v] = std::move(c);
  }

  // Interactions: between two variables wherever both are active, away from their references, and on kernels the
  // model says they share there.
  std::vector<std::set<u32>> touches(n);
  for (size_t v = 0; v < n; ++v) {
    for (const BranchVariable::Context& c : vars[v].contexts) {
      if (c.reference) { touches[v].insert(c.touches); }
    }
  }
  for (size_t a = 0; a < n; ++a) {
    for (size_t b = a + 1; b < n; ++b) {
      if (!out.free[a] && !out.free[b]) { continue; }
      const Option& oa = *vars[a].axis.option;
      const Option& ob = *vars[b].axis.option;
      bool anywhere = false;
      for (u32 const ta : touches[a]) {
        for (u32 const tb : touches[b]) { anywhere = anywhere || Model::interact(oa, ta, ob, tb); }
      }
      if (!anywhere) { continue; }

      bool const drawn = Model::drawnTogether(oa, ob);
      size_t const na = vars[a].axis.values.size();
      size_t const nb = vars[b].axis.values.size();
      std::vector<double> values(na * nb);
      for (size_t pa = 0; pa < na; ++pa) {
        for (size_t pb = 0; pb < nb; ++pb) {
          values[pa * nb + pb] = model.interactionOf(drawFor(follows[a] || follows[b]), subject, vars[a].axis.label(pa),
                                                     vars[b].axis.label(pb), drawn);
        }
      }
      // A pair the draw says nothing of is no edge for the optimiser to eliminate.
      if (std::ranges::none_of(values, [](double x) { return x != 0; })) { continue; }

      std::vector<size_t> scope{a, b};
      for (size_t const v : {a, b}) {
        if (out.companions[v]) { scope.push_back(out.companions[v]->var); }
      }
      auto const where = [&](size_t v) { return size_t(std::ranges::find(scope, v) - scope.begin()); };
      size_t const ia = where(a);
      size_t const ib = where(b);

      std::vector<size_t> digits(scope.size(), 0);
      auto contextFor = [&](size_t v) -> const BranchVariable::Context* {
        const std::optional<Problem::Companion>& c = out.companions[v];
        return c ? c->kinds[digits[where(c->var)]] : steadyContext(vars[v]);
      };
      std::vector<double> table;
      for (bool more = true; more;) {
        size_t const sa = digits[ia];
        size_t const sb = digits[ib];
        double value = 0;
        if (sa != vars[a].inactive() && sb != vars[b].inactive()) {
          const BranchVariable::Context* ca = contextFor(a);
          const BranchVariable::Context* cb = contextFor(b);
          if (ca && cb && ca->reference && cb->reference && sa != *ca->reference && sb != *cb->reference &&
              Model::interact(oa, ca->touches, ob, cb->touches)) {
            value = values[sa * nb + sb];
          }
        }
        table.push_back(value);

        more = false;
        for (size_t i = scope.size(); i-- > 0;) {
          if (++digits[i] < problem.unary[scope[i]].size()) {
            more = true;
            break;
          }
          digits[i] = 0;
        }
      }

      if (scope.size() == 2) {
        problem.pairs.push_back({.a = a, .b = b, .table = std::move(table)});
      } else {
        problem.factors.push_back({.scope = std::move(scope), .table = std::move(table)});
      }
    }
  }

  problem.starts.push_back(out.assignment(space.statesOf(out.around)));
  return out;
}

}  // namespace

Optimum optimize(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                 const Subspace* subspace) {
  const Env& env = model.env();
  Problem const problem = problemOf(model, draw, subject, filter, subspace);
  const std::vector<BranchVariable>& vars = problem.space->variables();
  const UseConfig& around = problem.around;

  MinSumResult const r = minimize(problem.minsum);
  if (!(r.value < FORBIDDEN)) {
    return {.options = around, .predicted = model.predict(draw, subject, around), .exact = false};
  }
  UseConfig options = problem.space->configOf(r.assignment);
  // A held axis at a position the space has no state for -- a register limit from an earlier offer -- stays there,
  // where the space could only have read it as the reference.
  bool restored = false;
  for (size_t v = 0; v < vars.size() && subspace; ++v) {
    const BranchVariable& var = vars[v];
    if (problem.free[v] || r.assignment[v] == var.inactive()) { continue; }
    if (!var.axis.digit && !around.contains(var.axis.option->key)) { continue; }
    std::pair<int, int> const was = positionOf(env, subject.fft, around, var.axis);
    if (std::ranges::find(var.axis.values, was) != var.axis.values.end()) { continue; }
    Axis only = var.axis;
    only.values = {was};
    place(options, only, 0);
    restored = true;
  }
  if (restored) { options = fitted(env, subject.fft, options); }
  return {.options = options,
          .predicted = model.predict(draw, subject, options),
          .exact = r.exact && problem.space->unsearched().empty()};
}

double conditionalEnergy(const Model& model, const Draw& draw, const Subject& subject, const AxisFilter& filter,
                         const UseConfig& options) {
  Problem const problem = problemOf(model, draw, subject, filter, nullptr);
  return model.constantOf(draw, subject) +
    evaluate(problem.minsum, problem.assignment(problem.space->statesOf(options)));
}

}  // namespace tune
