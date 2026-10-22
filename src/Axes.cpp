// Copyright (C) Jason Lynch

#include "Axes.h"

#include "Args.h"
#include "TuneDB.h"

#include <algorithm>

namespace tune {

namespace {

// What the kernels see `key` set to under `config`; nothing where they do not read it.
[[nodiscard]] std::optional<int> effectiveValue(const Env& env, const FFTConfig& fft, const UseConfig& config,
                                                const std::string& key) {
  const Option* const option = findOption(key);
  if (!option || !option->appliesTo(env, fft, config) || option->isInert(env, fft, config)) { return {}; }
  return useValue(config, key, option->defaultFor(env, fft, config));
}

}  // namespace

std::vector<std::string> Axis::keys() const {
  if (coupled) { return {"LOADS", "STORES"}; }
  return {option->key};
}

std::string Axis::label(size_t index) const {
  auto const [first, second] = values[index];
  return name + '=' + std::to_string(first) + (coupled ? '/' + std::to_string(second) : std::string{});
}

const char* toString(TailQueues queues) {
  switch (queues) {
  case TailQueues::Ntt: return "ntt";
  case TailQueues::All: return "all";
  case TailQueues::None: return "none";
  }
  return "?";
}

std::optional<TailQueues> parseTailQueues(std::string_view text) {
  for (TailQueues const q : {TailQueues::Ntt, TailQueues::All, TailQueues::None}) {
    if (text == toString(q)) { return q; }
  }
  return {};
}

std::vector<Axis> axesOf(const Env& env, const FFTConfig& fft, const UseConfig& background, const AxisFilter& filter) {
  UseConfig const from = canonicalConfig(env, fft, background);
  bool const queuesApart = filter.tailQueues == TailQueues::All ||
    (filter.tailQueues == TailQueues::Ntt && env.isNvidia && (fft.NTT_GF31 || fft.NTT_GF61));

  // A REG* key is offered the compiler's default, and once its kernel's resources are recorded the model's candidates.
  auto const offered = [&](const Option& option, int value) {
    if (option.key == "TAIL_KERNELS" && value >= 10) { return queuesApart; }
    if (!filter.registers || option.key == "NOREG" || !Args::isRegisterKey(option.key) || value == 0) { return true; }
    auto const it = filter.registers->find(option.key);
    return it != filter.registers->end() && std::ranges::find(it->second, value) != it->second.end();
  };

  std::vector<Axis> out;
  auto add = [&](Axis axis, std::pair<int, int> current, std::pair<int, int> reference) {
    for (auto const& p : {current, reference}) {
      if (std::ranges::find(axis.values, p) == axis.values.end()) { axis.values.push_back(p); }
    }
    std::ranges::sort(axis.values);
    axis.current = size_t(std::ranges::find(axis.values, current) - axis.values.begin());
    axis.reference = size_t(std::ranges::find(axis.values, reference) - axis.values.begin());
    if (axis.values.size() > 1) { out.push_back(std::move(axis)); }
  };

  for (const Option* const option : applicableOptions(env, fft, from)) {
    if (!option->compound) {
      Axis axis{.option = option, .name = option->key};
      for (int const value : option->valuesFor(env, fft, from)) {
        if (offered(*option, value)) { axis.values.emplace_back(value, 0); }
      }
      add(std::move(axis), positionOf(env, fft, from, Axis{.option = option}), {option->defaultFor(env, fft, from), 0});
      continue;
    }

    bool const isLoads = option->key == "LOADS";
    for (const AccessClass& cls : accessClasses()) {
      Axis axis{.option = option, .digit = cls.digit, .coupled = !cls.pairs.empty()};
      if (axis.coupled) {
        if (!isLoads) { continue; }
        axis.name = "LOADS+STORES[" + cls.name + "]";
        for (auto const& p : usablePairs(env, cls)) { axis.values.push_back(p); }
      } else {
        axis.name = option->key + "[" + cls.name + "]";
        for (int const mode : isLoads ? usableLoadModes(env, cls) : usableStoreModes(env, cls)) {
          axis.values.emplace_back(mode, 0);
        }
        // A class with no side in this key is not one of its axes.
        if (axis.values.empty()) { continue; }
      }
      std::pair<int, int> const current = positionOf(env, fft, from, axis);
      add(std::move(axis), current, {0, 0});
    }
  }
  return out;
}

std::pair<int, int> positionOf(const Env& env, const FFTConfig& fft, const UseConfig& config, const Axis& axis) {
  if (!axis.digit) { return {useValue(config, axis.option->key, axis.option->defaultFor(env, fft, config)), 0}; }
  u32 const digit = *axis.digit;
  if (axis.coupled) {
    return {int(getDigit(u32(useValue(config, "LOADS", 0)), digit)),
            int(getDigit(u32(useValue(config, "STORES", 0)), digit))};
  }
  return {int(getDigit(u32(useValue(config, axis.option->key, 0)), digit)), 0};
}

void place(UseConfig& config, const Axis& axis, size_t index) {
  auto const [first, second] = axis.values[index];
  if (!axis.digit) {
    config[axis.option->key] = std::to_string(first);
    return;
  }

  auto setIn = [&](const std::string& key, int mode) {
    config[key] = std::to_string(setDigit(u32(useValue(config, key, 0)), *axis.digit, u32(mode)));
  };
  if (axis.coupled) {
    setIn("LOADS", first);
    setIn("STORES", second);
  } else {
    setIn(axis.option->key, first);
  }
}

UseConfig branchOf(const Env& env, const FFTConfig& fft, const UseConfig& config) {
  UseConfig out;
  for (const Option& option : allOptions()) {
    if (option.kind != Kind::Tunable || !option.structural) { continue; }
    if (std::optional<int> const value = effectiveValue(env, fft, config, option.key)) {
      out[option.key] = std::to_string(*value);
    }
  }
  return out;
}

UseConfig fitted(const Env& env, const FFTConfig& fft, UseConfig config) {
  for (bool changed = true; changed;) {
    changed = false;
    config = canonicalConfig(env, fft, config);
    for (auto it = config.begin(); it != config.end(); ++it) {
      const Option* const option = findOption(it->first);
      if (option->compound) { continue; }
      std::vector<int> const values = option->valuesFor(env, fft, config);
      std::optional<int> const value = parseInt<int>(it->second);
      if (!value || std::ranges::find(values, *value) == values.end()) {
        config.erase(it);
        changed = true;
        break;
      }
    }
  }
  return config;
}

}  // namespace tune
