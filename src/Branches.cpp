// Copyright (C) Jason Lynch

#include "Branches.h"

#include <algorithm>
#include <functional>
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

}  // namespace

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

}  // namespace tune
