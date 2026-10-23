// Copyright (C) Jason Lynch

#include "Branches.h"

#include <algorithm>
#include <functional>
#include <map>
#include <mutex>
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
    std::vector<size_t> digits(ancestors.size(), 0);
    size_t contexts = 1;
    for (size_t const a : ancestors) {
      contexts *= variables_[a].states();
      if (contexts > MAX_CONTEXTS) { break; }
    }
    if (contexts > MAX_CONTEXTS) {
      unsearched_.push_back(option->key);
      continue;
    }
    for (size_t context = 0; context < contexts; ++context) {
      UseConfig config;
      for (size_t i = 0; i < ancestors.size(); ++i) {
        const BranchVariable& parent = variables_[ancestors[i]];
        if (digits[i] != parent.inactive()) { place(config, parent.axis, digits[i]); }
      }
      UseConfig const canonical = canonicalConfig(env_, fft_, config);
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
      BranchVariable v{.axis = s.axis, .canBeInactive = false, .ancestors = ancestors, .contexts = {}, .steady = true};
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
        v.contexts.push_back(std::move(c));
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
    const BranchVariable::Context& c = variables_[v].contexts[contextOf(v, states)];
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
  const BranchVariable::Context& c = var.contexts[context];
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
    states[v] = variables_[v].contexts[context].reference ? heldAt(v, context, canonical) : variables_[v].inactive();
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
    const BranchVariable::Context& c = variables_[a].contexts[contextOf(a, states)];
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
      const BranchVariable::Context& c = variables_[v].contexts[contextOf(v, states)];
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

}  // namespace tune
