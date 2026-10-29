// Copyright (C) Jason Lynch

#include "Branches.h"

#include "Random.h"

#include <algorithm>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>

namespace tune {

namespace {

// The tunable options, every one after those it depends on.
std::vector<const Option*> dependencyOrder() {
  std::vector<const Option*> out;
  std::set<std::string> placed;
  std::function<void(const Option&)> visit = [&](const Option& o) {
    if (placed.contains(o.key)) { return; }
    placed.insert(o.key);
    for (const std::string& key : o.dependsOn) {
      if (const Option* parent = findOption(key)) { visit(*parent); }
    }
    if (o.kind == Kind::Tunable) { out.push_back(&o); }
  };
  for (const Option& o : allOptions()) { visit(o); }
  return out;
}

// Every key `o` depends on, directly or through another.
std::set<std::string> ancestry(const Option& o) {
  std::set<std::string> out;
  std::function<void(const Option&)> visit = [&](const Option& at) {
    for (const std::string& key : at.dependsOn) {
      if (out.insert(key).second) {
        if (const Option* parent = findOption(key)) { visit(*parent); }
      }
    }
  };
  visit(o);
  return out;
}

// How many FFTs' spaces are kept; a run asks about far fewer, with a few register offers each.
constexpr size_t SPACES_KEPT = 4096;

}  // namespace

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

ConditionalSpace::ConditionalSpace(const Env& env, const FFTConfig& fft, const AxisFilter& filter) :
  env_{env}, fft_{fft} {
  for (const Option* option : dependencyOrder()) {
    std::set<std::string> const keys = ancestry(*option);
    std::vector<size_t> ancestors;
    for (size_t v = 0; v < variables_.size(); ++v) {
      if (keys.contains(variables_[v].axis.option->key)) { ancestors.push_back(v); }
    }

    // What each of the option's axes is in each context: its positions and reference, or nothing where inactive.
    struct Seen {
      Axis axis;
      std::vector<std::optional<Axis>> at;
      std::vector<u32> touches;
    };
    std::vector<Seen> seen;
    auto configAt = [&](const std::vector<size_t>& digits) {
      UseConfig config;
      for (size_t i = 0; i < ancestors.size(); ++i) {
        const BranchVariable& parent = variables_[ancestors[i]];
        if (digits[i] != parent.inactive()) { place(config, parent.axis, digits[i]); }
      }
      return canonicalConfig(env_, fft_, config);
    };

    size_t contexts = 1;
    for (size_t const a : ancestors) {
      contexts *= variables_[a].states();
      if (contexts > MAX_CONTEXTS) { break; }
    }
    if (contexts > MAX_CONTEXTS) {
      // Too many to list: whether it is unsearched or never applies here at all -- SMEM_SPILL, which depends on every
      // register limit, needs a CUDA 13 driver -- is told by probing the defaults, every ancestor moved alone, and a
      // sample of the rest.
      std::vector<size_t> const defaults = statesOf({});
      std::vector<size_t> base;
      for (size_t const a : ancestors) { base.push_back(defaults[a]); }
      std::vector<std::vector<size_t>> probes{base};
      for (size_t i = 0; i < ancestors.size(); ++i) {
        for (size_t s = 0; s < variables_[ancestors[i]].states(); ++s) {
          probes.push_back(base);
          probes.back()[i] = s;
        }
      }
      std::mt19937_64 rng{fnv1a(option->key)};
      for (size_t k = 0; k < CONTEXT_PROBES; ++k) {
        std::vector<size_t> digits;
        for (size_t const a : ancestors) { digits.push_back(rng() % variables_[a].states()); }
        probes.push_back(std::move(digits));
      }
      if (std::ranges::any_of(probes, [&](const std::vector<size_t>& digits) {
            return !axesOf(env_, fft_, configAt(digits), filter, *option).empty();
          })) {
        unsearched_.push_back(option->key);
      }
      continue;
    }

    std::vector<size_t> digits(ancestors.size(), 0);
    for (size_t context = 0; context < contexts; ++context) {
      UseConfig const canonical = configAt(digits);
      for (const Axis& axis : axesOf(env_, fft_, canonical, filter, *option)) {
        auto it = std::ranges::find(seen, axis.name, [](const Seen& s) { return s.axis.name; });
        if (it == seen.end()) {
          seen.push_back(
            {.axis = axis, .at = std::vector<std::optional<Axis>>(contexts), .touches = std::vector<u32>(contexts)});
          it = seen.end() - 1;
        }
        it->at[context] = axis;
        it->touches[context] = option->touchesFor(env_, fft_, canonical);
      }
      for (size_t i = ancestors.size(); i-- > 0;) {
        if (++digits[i] < variables_[ancestors[i]].states()) { break; }
        digits[i] = 0;
      }
    }

    for (Seen& s : seen) {
      BranchVariable v{.axis = s.axis, .canBeInactive = false, .ancestors = ancestors};
      v.axis.values.clear();
      for (const std::optional<Axis>& a : s.at) {
        if (!a) {
          v.canBeInactive = true;
          continue;
        }
        for (auto const& p : a->values) {
          if (std::ranges::find(v.axis.values, p) == v.axis.values.end()) { v.axis.values.push_back(p); }
        }
      }
      std::ranges::sort(v.axis.values);

      std::optional<std::pair<int, int>> reference;
      std::optional<u32> touches;
      for (size_t context = 0; context < s.at.size(); ++context) {
        const std::optional<Axis>& a = s.at[context];
        BranchVariable::Context c{
          .offered = std::vector<bool>(v.axis.values.size(), false), .reference = {}, .touches = 0};
        if (a) {
          for (auto const& p : a->values) {
            c.offered[size_t(std::ranges::find(v.axis.values, p) - v.axis.values.begin())] = true;
          }
          auto const ref = a->values[a->reference];
          c.reference = size_t(std::ranges::find(v.axis.values, ref) - v.axis.values.begin());
          c.touches = s.touches[context];
          if ((reference && *reference != ref) || (touches && *touches != c.touches)) { v.steady = false; }
          reference = ref;
          touches = c.touches;
        }
        auto const kind = std::ranges::find(v.kinds, c);
        v.contextKinds.push_back(u32(kind - v.kinds.begin()));
        if (kind == v.kinds.end()) { v.kinds.push_back(std::move(c)); }
      }
      variables_.push_back(std::move(v));
    }
  }
}

size_t ConditionalSpace::contextOf(size_t v, const std::vector<size_t>& states) const {
  size_t at = 0;
  for (size_t const a : variables_[v].ancestors) { at = at * variables_[a].states() + states[a]; }
  return at;
}

bool ConditionalSpace::legal(const std::vector<size_t>& states) const {
  for (size_t v = 0; v < variables_.size(); ++v) {
    const BranchVariable::Context& c = variables_[v].context(contextOf(v, states));
    bool const ok = states[v] == variables_[v].inactive() ? !c.reference : c.reference && c.offered[states[v]];
    if (!ok) { return false; }
  }
  return true;
}

UseConfig ConditionalSpace::configOf(const std::vector<size_t>& states) const {
  UseConfig config;
  for (size_t v = 0; v < variables_.size(); ++v) {
    if (states[v] != variables_[v].inactive()) { place(config, variables_[v].axis, states[v]); }
  }
  return fitted(env_, fft_, config);
}

size_t ConditionalSpace::heldAt(size_t v, size_t context, const UseConfig& canonical) const {
  const BranchVariable& var = variables_[v];
  const BranchVariable::Context& c = var.context(context);
  bool const named = var.axis.digit ? true : canonical.contains(var.axis.option->key);
  if (named) {
    auto const at = std::ranges::find(var.axis.values, positionOf(env_, fft_, canonical, var.axis));
    if (at != var.axis.values.end() && c.offered[size_t(at - var.axis.values.begin())]) {
      return size_t(at - var.axis.values.begin());
    }
  }
  return *c.reference;
}

std::vector<size_t> ConditionalSpace::statesOf(const UseConfig& config) const {
  UseConfig const canonical = canonicalConfig(env_, fft_, config);
  std::vector<size_t> states(variables_.size(), 0);
  for (size_t v = 0; v < variables_.size(); ++v) {
    size_t const context = contextOf(v, states);
    states[v] = variables_[v].context(context).reference ? heldAt(v, context, canonical) : variables_[v].inactive();
  }
  return states;
}


bool ConditionalSpace::legalContext(size_t v, size_t context) const {
  const BranchVariable& var = variables_[v];
  std::vector<size_t> states(variables_.size(), 0);
  for (size_t i = var.ancestors.size(); i-- > 0;) {
    size_t const a = var.ancestors[i];
    states[a] = context % variables_[a].states();
    context /= variables_[a].states();
  }
  for (size_t const a : var.ancestors) {
    const BranchVariable::Context& c = variables_[a].context(contextOf(a, states));
    bool const ok = states[a] == variables_[a].inactive() ? !c.reference : c.reference && c.offered[states[a]];
    if (!ok) { return false; }
  }
  return true;
}

BranchCount ConditionalSpace::legalBranches() const {
  std::call_once(branchesOnce_, [this] {
    // The structural variables and everything they depend on, in order: an assignment of these that each allows the
    // next is a legal prefix of some configuration, whose structural part is a branch.
    std::vector<bool> inScope(variables_.size(), false);
    for (size_t v = 0; v < variables_.size(); ++v) {
      if (!variables_[v].axis.option->structural) { continue; }
      inScope[v] = true;
      for (size_t const a : variables_[v].ancestors) { inScope[a] = true; }
    }
    std::vector<size_t> scope;
    for (size_t v = 0; v < variables_.size(); ++v) {
      if (inScope[v]) { scope.push_back(v); }
    }

    std::set<std::vector<size_t>> seen;
    size_t leaves = 0;
    bool complete = true;
    std::vector<size_t> states(variables_.size(), 0);
    std::function<void(size_t)> walk = [&](size_t i) {
      if (!complete) { return; }
      if (i == scope.size()) {
        if (++leaves > MAX_BRANCH_WALK) {
          complete = false;
          return;
        }
        std::vector<size_t> branch;
        for (size_t const v : scope) {
          if (variables_[v].axis.option->structural) { branch.push_back(states[v]); }
        }
        seen.insert(std::move(branch));
        return;
      }
      size_t const v = scope[i];
      const BranchVariable::Context& c = variables_[v].context(contextOf(v, states));
      for (size_t s = 0; s < variables_[v].states(); ++s) {
        bool const ok = s == variables_[v].inactive() ? !c.reference : c.reference && c.offered[s];
        if (!ok) { continue; }
        states[v] = s;
        walk(i + 1);
      }
    };
    walk(0);
    branches_ = {.legal = seen.size(), .complete = complete};
  });
  return branches_;
}

void settle(const ConditionalSpace& space, std::vector<size_t>& states, size_t from) {
  const std::vector<BranchVariable>& vars = space.variables();
  for (size_t u = from + 1; u < vars.size(); ++u) {
    const BranchVariable::Context& c = vars[u].context(space.contextOf(u, states));
    if (!c.reference) {
      states[u] = vars[u].inactive();
    } else if (states[u] == vars[u].inactive() || !c.offered[states[u]]) {
      states[u] = *c.reference;
    }
  }
}

}  // namespace tune
