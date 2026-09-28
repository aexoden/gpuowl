// Copyright (C) Jason Lynch

#include "Probe.h"

#include "Bootstrap.h"
#include "TuneDB.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>

namespace tune {

namespace {

// What the kernels see `key` set to under `config`; nothing where they do not read it.
[[nodiscard]] std::optional<int> effectiveValue(const Env& env, const FFTConfig& fft, const UseConfig& config,
                                                const std::string& key) {
  const Option* const option = findOption(key);
  if (!option || !option->appliesTo(env, fft, config) || option->isInert(env, fft, config)) { return {}; }
  return useValue(config, key, option->defaultFor(env, fft, config));
}

// Everything `keys` depend on, directly or through one another, other than themselves.
[[nodiscard]] std::set<std::string> dependeesOf(const std::set<std::string>& keys) {
  std::set<std::string> out;
  std::vector<std::string> todo{keys.begin(), keys.end()};
  while (!todo.empty()) {
    std::string const key = std::move(todo.back());
    todo.pop_back();
    const Option* const option = findOption(key);
    if (!option) { continue; }
    for (const std::string& d : option->dependsOn) {
      if (!keys.contains(d) && out.insert(d).second) { todo.push_back(d); }
    }
  }
  return out;
}

// `config` with every key the table would not offer at its value alongside the rest returned to its default, to a
// fixpoint: a structural move can take a dependent's value out of its own list, as SHUFL_BYTES_W=16 does to WMUL at a
// wide width, and the host would only clamp it back.
[[nodiscard]] UseConfig fitted(const Env& env, const FFTConfig& fft, UseConfig config) {
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

// The keys whose value the kernels see differently under `to` than under `from`, at their values under `to`.
[[nodiscard]] std::vector<std::pair<std::string, int>> changes(const Env& env, const FFTConfig& fft,
                                                               const UseConfig& from, const UseConfig& to) {
  std::set<std::string> keys;
  for (const auto& [key, value] : from) { keys.insert(key); }
  for (const auto& [key, value] : to) { keys.insert(key); }

  std::vector<std::pair<std::string, int>> out;
  for (const std::string& key : keys) {
    std::optional<int> const before = effectiveValue(env, fft, from, key);
    std::optional<int> const after = effectiveValue(env, fft, to, key);
    if (after && before != after) { out.emplace_back(key, *after); }
  }
  return out;
}

// One answer a group -- or a cluster, taken whole -- has given in a branch: the moves from the background that put
// its axes where that answer has them, and what the answer gained against the branch's best.
struct Seed {
  std::vector<std::pair<size_t, size_t>> moves;
  double gain = 0;
};

// The best `top` distinct projections of `readings` onto the axes `unit`, the background's own first.  `readings` are
// the branch's, the background first and the rest cheapest first.  A projection this background cannot place -- a value
// that a dependent's list, as the background has it, does not offer -- is passed over.
[[nodiscard]] std::vector<Seed> seedsOf(const Env& env, const FFTConfig& fft, const std::vector<Axis>& axes,
                                        const std::vector<size_t>& unit, std::span<const Reading> readings, u32 top) {
  std::vector<Seed> out{Seed{}};
  std::set<std::vector<size_t>> seen;
  std::vector<size_t> origin;
  for (size_t const i : unit) { origin.push_back(axes[i].current); }
  seen.insert(std::move(origin));

  for (const Reading& reading : readings) {
    if (out.size() >= top) { break; }

    std::vector<size_t> at;
    for (size_t const i : unit) {
      auto const it = std::ranges::find(axes[i].values, positionOf(env, fft, reading.config, axes[i]));
      if (it == axes[i].values.end()) { break; }
      at.push_back(size_t(it - axes[i].values.begin()));
    }
    if (at.size() != unit.size() || !seen.insert(at).second) { continue; }

    // An answer cheaper than the background's own, which a race can leave behind a winner it decided by margin, counts
    // as no gain: the background's answer is first in every dimension, and the enumeration needs each to fall.
    Seed seed{.moves = {}, .gain = std::min(0.0, 1 - reading.cost / readings.front().cost)};
    for (size_t j = 0; j < unit.size(); ++j) {
      if (at[j] != axes[unit[j]].current) { seed.moves.emplace_back(unit[j], at[j]); }
    }
    out.push_back(std::move(seed));
  }
  return out;
}

// The size of a cross product, saturating rather than wrapping.
[[nodiscard]] u64 productOf(const std::vector<u64>& sizes) {
  u64 out = 1;
  for (u64 const n : sizes) {
    if (n && out > std::numeric_limits<u64>::max() / n) { return std::numeric_limits<u64>::max(); }
    out *= n;
  }
  return out;
}

class Enumerator {
public:
  Enumerator(const Env& env, const FFTConfig& fft, const UseConfig& best, u32 listed, ProbeList& out) :
    env_{env},
    fft_{fft},
    from_{canonicalConfig(env, fft, best)},
    listed_{listed},
    out_{out},
    seen_{configText(from_)} {}

  // Every point of `axes` other than where they stand now: fewest axes moved first, then the combinations of axes in
  // declaration order, then each axis's positions in ascending order.  Stops once `limit` probes have been offered.
  void enumerate(const std::vector<size_t>& axes, const std::string& stage, u32 limit) {
    ++part_;
    u32 const cap = std::min(limit, listed_);
    u32 taken = 0;
    u64 visited = 0;
    for (size_t k = 1; k <= axes.size(); ++k) {
      std::vector<size_t> pick(k);
      std::iota(pick.begin(), pick.end(), 0);
      while (true) {
        if (!positions(axes, pick, stage, cap, taken, visited)) {
          std::vector<u64> sizes;
          for (size_t const a : axes) { sizes.push_back(out_.axes[a].values.size()); }
          unlisted(stage, 1, limit, taken, productOf(sizes) - 1 - visited);
          return;
        }
        if (!nextCombination(pick, axes.size())) { break; }
      }
    }
  }

  // Every choice of one seed per dimension other than the background, highest summed gain first, ties by the seeds
  // chosen, earlier dimensions first.  Stops once `limit` probes have been offered.  Each dimension's seeds fall in
  // gain, so a choice never outranks the one with any of its seeds moved back a place, and the choices can be taken
  // from a frontier in order without counting out the cross product, which with seven dimensions of eight seeds
  // would be millions.
  void combine(const std::vector<std::vector<Seed>>& dims, const std::string& stage, u32 tier, u32 limit) {
    ++part_;
    u32 const cap = std::min(limit, listed_);
    using Point = std::pair<double, std::vector<size_t>>;
    auto const before = [](const Point& a, const Point& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    std::set<Point, decltype(before)> frontier{before};
    std::set<std::vector<size_t>> reached;

    auto push = [&](std::vector<size_t> pick) {
      if (!reached.insert(pick).second) { return; }
      double gain = 0;
      for (size_t d = 0; d < dims.size(); ++d) { gain += dims[d][pick[d]].gain; }
      frontier.emplace(gain, std::move(pick));
    };
    push(std::vector<size_t>(dims.size(), 0));

    u32 taken = 0;
    u64 visited = 0;
    while (!frontier.empty()) {
      std::vector<size_t> const pick = frontier.begin()->second;
      frontier.erase(frontier.begin());
      ++visited;
      for (size_t d = 0; d < dims.size(); ++d) {
        if (pick[d] + 1 < dims[d].size()) {
          std::vector<size_t> next = pick;
          ++next[d];
          push(std::move(next));
        }
      }

      std::vector<std::pair<size_t, size_t>> moves;
      for (size_t d = 0; d < dims.size(); ++d) {
        const std::vector<std::pair<size_t, size_t>>& seed = dims[d][pick[d]].moves;
        moves.insert(moves.end(), seed.begin(), seed.end());
      }
      // The background itself has no moves, and is not a point.
      if (!moves.empty() && offer(std::move(moves), stage, tier) && ++taken >= cap) {
        std::vector<u64> sizes;
        for (const std::vector<Seed>& seeds : dims) { sizes.push_back(seeds.size()); }
        unlisted(stage, tier, limit, taken, productOf(sizes) - visited);
        return;
      }
    }
  }

private:
  [[nodiscard]] static bool nextCombination(std::vector<size_t>& pick, size_t n) {
    size_t const k = pick.size();
    for (size_t i = k; i-- > 0;) {
      if (pick[i] < n - k + i) {
        ++pick[i];
        for (size_t j = i + 1; j < k; ++j) { pick[j] = pick[j - 1] + 1; }
        return true;
      }
    }
    return false;
  }

  // Where a stage stopped at `taken` points because they were all it was to list, and `left` it has not visited yet:
  // at most as many more as its limit allows.
  void unlisted(const std::string& stage, u32 tier, u32 limit, u32 taken, u64 left) {
    u64 const most = std::min<u64>(left, limit - taken);
    if (taken < limit && most > 0) {
      out_.unlisted.push_back({.stage = stage, .part = part_ - 1, .tier = tier, .most = most});
    }
  }

  // Every assignment of the picked axes to positions other than their current ones.  False once the limit is reached.
  bool positions(const std::vector<size_t>& axes, const std::vector<size_t>& pick, const std::string& stage, u32 limit,
                 u32& taken, u64& visited) {
    std::vector<size_t> at(pick.size(), 0);
    while (true) {
      std::vector<std::pair<size_t, size_t>> moves;
      for (size_t j = 0; j < pick.size(); ++j) {
        const Axis& axis = out_.axes[axes[pick[j]]];
        moves.emplace_back(axes[pick[j]], at[j] < axis.current ? at[j] : at[j] + 1);
      }
      ++visited;
      if (offer(std::move(moves), stage, 1) && ++taken >= limit) { return false; }

      size_t j = pick.size();
      while (j-- > 0) {
        if (++at[j] < out_.axes[axes[pick[j]]].values.size() - 1) { break; }
        at[j] = 0;
      }
      if (j == size_t(-1)) { return true; }
    }
  }

  bool offer(std::vector<std::pair<size_t, size_t>> moves, const std::string& stage, u32 tier) {
    UseConfig raw = from_;
    for (auto const& [axis, index] : moves) { place(raw, out_.axes[axis], index); }
    UseConfig const config = fitted(env_, fft_, raw);

    // Fitting took a moved axis somewhere else, so this is not the point it claims to be.
    for (auto const& [axis, index] : moves) {
      if (positionOf(env_, fft_, config, out_.axes[axis]) != out_.axes[axis].values[index]) { return false; }
    }
    if (!seen_.insert(configText(config)).second) { return false; }

    std::vector<std::pair<std::string, int>> const changed = changes(env_, fft_, from_, config);
    std::string text;
    std::set<std::string> named;
    for (const auto& [key, value] : changed) {
      text += (text.empty() ? "" : ",") + key + "=" + std::to_string(value);
      named.insert(key);
    }
    text += ldsAsideNote(fft_, from_, config, named);

    std::set<std::string> keys;
    for (auto const& [axis, index] : moves) {
      for (const std::string& key : out_.axes[axis].keys()) { keys.insert(key); }
    }
    std::set<std::string> const dependees = dependeesOf(keys);

    out_.probes.push_back({.config = config,
                           .stage = stage,
                           .tier = tier,
                           .part = part_ - 1,
                           .moves = std::move(moves),
                           .key = changed.size() == 1 ? changed.front().first : std::string{},
                           .text = std::move(text),
                           .dependees = {dependees.begin(), dependees.end()}});
    return true;
  }

  const Env& env_;
  const FFTConfig& fft_;
  UseConfig from_;
  u32 listed_;
  ProbeList& out_;
  std::set<std::string> seen_;
  u32 part_ = 0;
};

}  // namespace

std::string Strategy::text() const {
  switch (kind) {
  case Kind::Hybrid: return "hybrid";
  case Kind::Single: return "single";
  case Kind::Groups: return "groups";
  case Kind::Permute: break;
  }
  std::string out = "permute:";
  for (size_t i = 0; i < keys.size(); ++i) { out += (i ? "+" : "") + keys[i]; }
  return out;
}

Strategy parseStrategy(std::string_view text) {
  if (text == "hybrid") { return {.kind = Strategy::Kind::Hybrid}; }
  if (text == "single") { return {.kind = Strategy::Kind::Single}; }
  if (text == "groups") { return {.kind = Strategy::Kind::Groups}; }

  constexpr std::string_view PERMUTE = "permute:";
  if (!text.starts_with(PERMUTE)) {
    throw std::string{"-tune: strategy= takes hybrid, single, groups or permute:<KEY>+<KEY>..."};
  }

  Strategy out{.kind = Strategy::Kind::Permute};
  for (std::string_view rest = text.substr(PERMUTE.size());;) {
    size_t const plus = rest.find('+');
    std::string const key{rest.substr(0, plus)};
    const Option* const option = findOption(key);
    if (!option || option->kind != Kind::Tunable) {
      throw "-tune: strategy=permute: '" + key + "' is not a key the tuner searches";
    }
    if (std::ranges::find(out.keys, key) == out.keys.end()) { out.keys.push_back(key); }
    if (plus == std::string_view::npos) { break; }
    rest = rest.substr(plus + 1);
  }
  return out;
}

u64 Bin::points(const std::vector<Axis>& all) const {
  std::vector<u64> sizes;
  for (size_t const a : axes) { sizes.push_back(all[a].values.size()); }
  return productOf(sizes) - 1;
}

std::vector<Bin> binsOf(const std::vector<Axis>& axes, Group group, const Strategy& strategy, bool structuralSteps) {
  std::string const name = toString(group);
  std::vector<Bin> out;
  std::vector<size_t> rest;
  std::vector<size_t> alone;
  for (size_t i = 0; i < axes.size(); ++i) {
    const Option& option = *axes[i].option;
    if (option.group != group) { continue; }
    if (option.structural) {
      if (structuralSteps) { out.push_back({.group = group, .stage = name, .axes = {i}, .structural = true}); }
    } else {
      (option.alone && !strategy.bootstrapTree ? alone : rest).push_back(i);
    }
  }

  if (!rest.empty()) {
    size_t const width = std::min<size_t>(strategy.maxPermute, rest.size());
    size_t const bins = (rest.size() + width - 1) / width;
    for (size_t b = 0; b < bins; ++b) {
      auto const first = rest.begin() + ptrdiff_t(b * width);
      out.push_back({.group = group,
                     .stage = bins > 1 ? name + " " + std::to_string(b + 1) : name,
                     .axes = {first, first + ptrdiff_t(std::min<size_t>(width, size_t(rest.end() - first)))}});
    }
  }
  for (size_t const i : alone) { out.push_back({.group = group, .stage = name, .axes = {i}}); }
  return out;
}

std::vector<std::string> Axis::keys() const {
  if (coupled) { return {"LOADS", "STORES"}; }
  return {option->key};
}

std::vector<Axis> axesOf(const Env& env, const FFTConfig& fft, const UseConfig& background) {
  UseConfig const from = canonicalConfig(env, fft, background);

  std::vector<Axis> out;
  auto add = [&](Axis axis, std::pair<int, int> current) {
    if (std::ranges::find(axis.values, current) == axis.values.end()) {
      axis.values.push_back(current);
      std::ranges::sort(axis.values);
    }
    axis.current = size_t(std::ranges::find(axis.values, current) - axis.values.begin());
    if (axis.values.size() > 1) { out.push_back(std::move(axis)); }
  };

  for (const Option* const option : applicableOptions(env, fft, from)) {
    if (!option->compound) {
      Axis axis{.option = option, .name = option->key};
      for (int const value : option->valuesFor(env, fft, from)) { axis.values.emplace_back(value, 0); }
      add(std::move(axis), positionOf(env, fft, from, Axis{.option = option}));
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
      add(std::move(axis), current);
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

std::vector<Branch> branchesOf(const Env& env, const FFTConfig& fft, std::span<const Reading> readings) {
  std::vector<Branch> out;
  for (const Reading& reading : readings) {
    UseConfig structure = branchOf(env, fft, reading.config);
    if (std::ranges::none_of(out, [&](const Branch& b) { return b.structure == structure; })) {
      out.push_back({.structure = std::move(structure), .best = reading.config, .cost = reading.cost});
    }
  }
  if (out.size() > MAX_BRANCHES) { out.resize(MAX_BRANCHES); }
  return out;
}

namespace {

// The combo tiers above the groups of `best`'s branch.  Within a branch the structural keys are the same everywhere,
// so a group is combined by its other axes.
void combos(const Env& env, const FFTConfig& fft, const UseConfig& best, const Strategy& strategy,
            std::span<const Reading> readings, Enumerator& enumerator, ProbeList& out) {
  UseConfig const structure = branchOf(env, fft, best);
  std::vector<Reading> inBranch;
  for (const Reading& r : readings) {
    if (branchOf(env, fft, r.config) == structure) { inBranch.push_back(r); }
  }
  if (inBranch.empty()) { return; }

  auto seedsOn = [&](const std::vector<size_t>& unit) {
    return seedsOf(env, fft, out.axes, unit, inBranch, strategy.comboTop);
  };
  auto seedsIn = [&](const std::vector<Group>& groups) {
    std::vector<size_t> unit;
    for (size_t i = 0; i < out.axes.size(); ++i) {
      const Option& option = *out.axes[i].option;
      if (!option.structural && std::ranges::find(groups, option.group) != groups.end()) { unit.push_back(i); }
    }
    return seedsOn(unit);
  };

  // A group's dimensions at tier 2: one per bin, so that what two bins of one group found is tried together.
  auto dimsOf = [&](Group group) {
    std::vector<std::vector<Seed>> dims;
    if (strategy.bootstrapTree) {
      dims.push_back(seedsIn({group}));
      return dims;
    }
    for (const Bin& bin : binsOf(out.axes, group, strategy, false)) { dims.push_back(seedsOn(bin.axes)); }
    return dims;
  };

  // A dimension with no answer but the background's own adds nothing to a cross product, and a group's own bins are
  // worth combining only where two of them have answers.
  auto combine = [&](const std::vector<std::vector<Seed>>& dims, const std::string& stage, u32 tier, size_t atLeast) {
    std::vector<std::vector<Seed>> useful;
    for (const std::vector<Seed>& seeds : dims) {
      if (seeds.size() > 1) { useful.push_back(seeds); }
    }
    if (!useful.empty() && useful.size() >= atLeast) { enumerator.combine(useful, stage, tier, strategy.maxPoints); }
  };

  ClusterGraph const graph = clusterGraph(env, fft, canonicalConfig(env, fft, best));
  for (const std::vector<Group>& cluster : graph.clusters) {
    std::vector<std::vector<Seed>> dims;
    std::string stage;
    for (Group const group : cluster) {
      std::ranges::move(dimsOf(group), std::back_inserter(dims));
      stage += (stage.empty() ? "" : "+") + std::string{toString(group)};
    }
    if (cluster.size() >= 2) {
      combine(dims, stage, 2, 1);
    } else if (!strategy.bootstrapTree) {
      combine(dims, stage + " combined", 2, 2);
    }
  }
  if (!strategy.bootstrapTree) {
    for (Group const group : graph.topTier) {
      combine(dimsOf(group), std::string{toString(group)} + " combined", 2, 2);
    }
  }

  if (strategy.comboTiers < 3) { return; }
  std::vector<std::vector<Seed>> dims;
  for (Group const group : graph.topTier) { dims.push_back(seedsIn({group})); }
  for (const std::vector<Group>& cluster : graph.clusters) { dims.push_back(seedsIn(cluster)); }
  combine(dims, "all", 3, 1);
}

}  // namespace

ProbeList probesOf(const Env& env, const FFTConfig& fft, const UseConfig& best, const Strategy& strategy,
                   std::span<const Reading> readings, bool structuralSteps, u32 listed) {
  ProbeList out{.axes = axesOf(env, fft, best), .probes = {}};
  Enumerator enumerator{env, fft, best, listed, out};

  auto axesWhere = [&](auto&& pred) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < out.axes.size(); ++i) {
      if (pred(out.axes[i])) { indices.push_back(i); }
    }
    return indices;
  };

  switch (strategy.kind) {
  case Strategy::Kind::Single:
    for (size_t i = 0; i < out.axes.size(); ++i) { enumerator.enumerate({i}, "single", ~0u); }
    break;

  case Strategy::Kind::Permute: {
    std::vector<size_t> const named = axesWhere([&](const Axis& axis) {
      return std::ranges::any_of(axis.keys(), [&](const std::string& key) {
        return std::ranges::find(strategy.keys, key) != strategy.keys.end();
      });
    });
    enumerator.enumerate(named, "permute", ~0u);
    break;
  }

  case Strategy::Kind::Hybrid:
  case Strategy::Kind::Groups: {
    // Every structural step before any group's own moves: which side of a structural key is the faster one decides
    // where the rest of the entry's search is best spent, and each side is worth only what its best set costs.
    std::vector<Bin> bins;
    for (Group const group : allGroups()) {
      std::ranges::move(binsOf(out.axes, group, strategy, structuralSteps), std::back_inserter(bins));
    }
    std::ranges::stable_partition(bins, &Bin::structural);
    for (const Bin& bin : bins) {
      bool const whole = bin.structural || (out.axes[bin.axes.front()].option->alone && !strategy.bootstrapTree);
      enumerator.enumerate(bin.axes, bin.stage, whole ? ~0u : strategy.maxPoints);
    }
    for (size_t const i : axesWhere([](const Axis& a) { return a.option->group == Group::None; })) {
      enumerator.enumerate({i}, "single", ~0u);
    }
    if (strategy.combines()) { combos(env, fft, best, strategy, readings, enumerator, out); }
    break;
  }
  }
  return out;
}

u64 SearchSize::offered() const {
  u64 out = 0;
  for (const GroupSize& g : groups) {
    out += g.structural + std::accumulate(g.offered.begin(), g.offered.end(), u64{0});
  }
  return out;
}

u64 SearchSize::binned() const {
  u64 out = 0;
  for (const GroupSize& g : groups) { out += g.structural + std::accumulate(g.points.begin(), g.points.end(), u64{0}); }
  return out;
}

u64 SearchSize::whole() const {
  u64 out = 0;
  for (const GroupSize& g : groups) { out += g.structural + g.whole; }
  return out;
}

SearchSize searchSize(const Env& env, const FFTConfig& fft, const UseConfig& best, const Strategy& strategy) {
  SearchSize out;
  if (!strategy.branches()) { return out; }

  std::vector<Axis> const axes = axesOf(env, fft, best);
  Strategy lifted = strategy;
  lifted.maxPermute = NO_LIMIT;
  for (Group const group : allGroups()) {
    std::vector<Bin> const bins = binsOf(axes, group, strategy);
    if (bins.empty()) { continue; }

    SearchSize::GroupSize& g = out.groups.emplace_back();
    g.group = group;
    for (const Bin& bin : bins) {
      u64 const points = bin.points(axes);
      if (bin.structural) {
        g.structural += points;
        continue;
      }
      g.options += bin.axes.size();
      g.points.push_back(points);
      bool const alone = axes[bin.axes.front()].option->alone && !strategy.bootstrapTree;
      g.offered.push_back(alone ? points : std::min<u64>(points, strategy.maxPoints));
    }
    for (const Bin& bin : binsOf(axes, group, lifted, false)) { g.whole += bin.points(axes); }
  }
  return out;
}

bool answeredBy(const Env& env, const FFTConfig& fft, const ProbeList& list, const Probe& probe, const UseConfig& row) {
  for (auto const& [index, position] : probe.moves) {
    const Axis& axis = list.axes[index];
    if (positionOf(env, fft, row, axis) != axis.values[position]) { return false; }
  }

  return std::ranges::all_of(probe.dependees, [&](const std::string& key) {
    return effectiveValue(env, fft, row, key) == effectiveValue(env, fft, probe.config, key);
  });
}

UseConfig restartOf(const Env& env, const FFTConfig& fft, std::string_view entry, u32 k) {
  // FNV-1a then splitmix64: the same sequence on every platform and standard library, which std::hash and the
  // standard distributions do not promise.
  u64 state = 0xcbf29ce484222325;
  for (char const c : entry) { state = (state ^ u8(c)) * 0x100000001b3; }
  state ^= (u64(k) + 1) * 0x9e3779b97f4a7c15;
  auto next = [&state] {
    u64 z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  };

  UseConfig config;
  std::set<std::string> drawn;
  while (true) {
    std::vector<Axis> const axes = axesOf(env, fft, config);

    std::set<std::string> pending;
    for (const Axis& axis : axes) {
      if (drawn.contains(axis.name)) { continue; }
      for (const std::string& key : axis.keys()) { pending.insert(key); }
    }

    // A dependee drawn after its dependent would draw the dependent from a list that no longer applies; a structural
    // key changes which axes there are at all.
    auto ready = [&](const Axis& axis) {
      return !drawn.contains(axis.name) && std::ranges::none_of(axis.keys(), [&](const std::string& key) {
        return std::ranges::any_of(findOption(key)->dependsOn,
                                   [&](const std::string& d) { return d != key && pending.contains(d); });
      });
    };
    auto at = std::ranges::find_if(axes, [&](const Axis& a) { return a.option->structural && ready(a); });
    if (at == axes.end()) { at = std::ranges::find_if(axes, ready); }
    if (at == axes.end()) { break; }

    drawn.insert(at->name);
    place(config, *at, size_t(next() % at->values.size()));
    config = canonicalConfig(env, fft, config);
  }
  return fitted(env, fft, config);
}

}  // namespace tune
