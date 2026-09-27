// Copyright (C) Jason Lynch

// The per-entry option search: the configurations one step from an entry's best known option set, under the search
// strategy the user chose, and which of them the database already answers.
//
// A step moves one or more axes.  An axis is one key, or for LOADS and STORES one access class -- one digit of one of
// them, or of both where the class chooses its load and store together -- so a probe never moves the packed integer
// as a whole, and never offers a mode the compiler would quietly build as another.
//
// Pure: the probe list is a function of the option table, the best set and, for the combo tiers, the entry's readings;
// whether a probe is answered is a function of the rows.  Nothing is remembered between items, so a resumed run offers
// exactly what an uninterrupted one would.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tune {

// The defaults of how many axes one bin of a group permutes, and how many points a stage -- one bin of a group, or one
// combination of groups -- offers before it is cut short: a bin fewest axes moved first, a combination most promising
// first.
inline constexpr u32 MAX_PERMUTE = 4;
inline constexpr u32 MAX_POINTS = 64;

// Either limit, lifted.
inline constexpr u32 NO_LIMIT = ~0u;

// How many points of one stage the queue lists at a time.  A stage the limits leave larger is listed in longer windows
// as the ones before are used up, so that choosing the next item costs what has been measured rather than what the
// limits allow: with both lifted, one group of a CUDA build can have hundreds of millions of points.  Equal to
// MAX_POINTS, so that under the default limits every stage is listed whole.
inline constexpr u32 PROBE_WINDOW = MAX_POINTS;

// How many of each group's best answers the combo tiers combine, and how many tiers the search has: 1 is the groups
// alone, 2 adds the groups that share kernels combined, 3 everything combined.
inline constexpr u32 COMBO_TOP = 3;
inline constexpr u32 COMBO_TIERS = 3;

// The structural branches of one entry searched at once, cheapest first.  The rest are left to restarts.
inline constexpr u32 MAX_BRANCHES = 8;

struct Strategy {
  enum class Kind : u8 {
    Hybrid,   // groups, then the combo tiers above them
    Single,   // one axis at a time
    Groups,   // every point within each bin of each group
    Permute,  // every point of exactly the keys named
  };

  Kind kind = Kind::Hybrid;

  // Permute only, as named.
  std::vector<std::string> keys{};

  // Hybrid and groups.
  u32 maxPermute = MAX_PERMUTE;
  u32 maxPoints = MAX_POINTS;

  // Hybrid only.
  u32 comboTop = COMBO_TOP;
  u32 comboTiers = COMBO_TIERS;

  // The bootstrap's combination tree as it has always been built, so that a database whose bootstrap it completed stays
  // complete: a key searched alone is binned with the rest of its group, and a group is one dimension of every
  // combination.  Otherwise each bin is a dimension of its own at tier 2.
  bool bootstrapTree = false;

  // Whether each structural value is searched as a branch of its own: the strategies that search by group.
  [[nodiscard]] bool branches() const { return kind == Kind::Hybrid || kind == Kind::Groups; }

  // Whether there is a tier above the groups.
  [[nodiscard]] bool combines() const { return kind == Kind::Hybrid && comboTiers > 1 && comboTop > 1; }

  // As parseStrategy() reads it; the limits and the combo settings are not part of it.
  [[nodiscard]] std::string text() const;
};

// "hybrid", "single", "groups", or "permute:" followed by tunable keys joined by '+'.  Throws a message for anything
// else, since a mistyped key would otherwise search nothing.
[[nodiscard]] Strategy parseStrategy(std::string_view text);

// One dimension of the search, as it stands against one background.
struct Axis {
  const Option* option = nullptr;  // LOADS for the coupled class

  // An access class: the digit it occupies, and whether STORES moves with LOADS.
  std::optional<u32> digit{};
  bool coupled = false;

  // The positions the axis can take, ascending, the background's among them: a key's value, a class's mode, or a
  // coupled class's (load, store) pair.
  std::vector<std::pair<int, int>> values{};
  size_t current = 0;

  // "WMUL", "LOADS[trig, used once]"
  std::string name{};

  // The keys it writes.
  [[nodiscard]] std::vector<std::string> keys() const;
};

// Every axis of `background` that has somewhere to move: the applicable tunable keys in declaration order, LOADS and
// STORES taken apart into their access classes in class order, each class offering only the modes `env` can emit.
[[nodiscard]] std::vector<Axis> axesOf(const Env& env, const FFTConfig& fft, const UseConfig& background);

// Where `config` puts `axis`.
[[nodiscard]] std::pair<int, int> positionOf(const Env& env, const FFTConfig& fft, const UseConfig& config,
                                             const Axis& axis);

// Puts `axis` at its `index`th position in `config`, leaving every other digit of a packed key where it was.
void place(UseConfig& config, const Axis& axis, size_t index);

// One stage of the steps a group offers: a structural key, a key searched alone, or a bin of the group's other axes.
struct Bin {
  Group group = Group::None;

  // "Memory 2", or the group's name for a bin it is not split into, a key searched alone and a structural step.
  std::string stage;

  // Into the axes of the list, in declaration order.
  std::vector<size_t> axes;

  bool structural = false;

  // How many points the bin's axes have other than where they stand: its whole cross product, saturating.
  [[nodiscard]] u64 points(const std::vector<Axis>& all) const;
};

// The bins of `group` among `axes` under `strategy`, as probesOf() enumerates them: its structural keys one step at a
// time where `structuralSteps`, then its other axes in bins of at most `maxPermute` in declaration order, then each key
// searched alone.
[[nodiscard]] std::vector<Bin> binsOf(const std::vector<Axis>& axes, Group group, const Strategy& strategy,
                                      bool structuralSteps = true);

struct Probe {
  // Canonical.
  UseConfig config;

  // What offered it: the group and, where the group is split, its bin ("Width", "Placement 2"); "single"; "permute";
  // for a combination the groups combined ("Tail+Height"), or "all" at the top tier.
  std::string stage;

  // 1 for a step within a group, a single step, or a permutation; 2 or 3 for a combination of the tier below's answers.
  u32 tier = 1;

  // Which enumeration of the list offered it, counted from 0: a group's structural steps and its bins share a stage
  // name but not a part.
  u32 part = 0;

  // The axes it moves, into ProbeList::axes, and the position each moves to.
  std::vector<std::pair<size_t, size_t>> moves;

  // The one key it changes, which a build failure can be pinned on; empty where it changes more than one.
  std::string key;

  // The keys it changes, at their new values: "WMUL=1,LDSSWIZ_W=1".
  std::string text;

  // Everything the keys of the axes it moves depend on, other than those keys: what a row must agree with it on.
  std::vector<std::string> dependees;
};

struct ProbeList {
  std::vector<Axis> axes;
  std::vector<Probe> probes;

  // A stage whose listing stopped at `listed` points short of its own limit, and at most how many more it would offer.
  struct Unlisted {
    std::string stage;
    u32 part = 0;
    u32 tier = 1;
    u64 most = 0;
  };
  std::vector<Unlisted> unlisted{};
};

// An option set of an entry that a row could publish, canonical, and what that row says it costs.
struct Reading {
  UseConfig config;
  double cost = 0;
};

// What the kernels see each structural key set to under `config`, for those that apply: which branch it is in.
[[nodiscard]] UseConfig branchOf(const Env& env, const FFTConfig& fft, const UseConfig& config);

// One structural branch of an entry, and the cheapest reading in it.
struct Branch {
  UseConfig structure;
  UseConfig best;
  double cost = 0;
};

// The branches `readings` fall into, cheapest first, and at most MAX_BRANCHES of them.  `readings` are cheapest first,
// ties in the order given, and the first is the entry's best set.
[[nodiscard]] std::vector<Branch> branchesOf(const Env& env, const FFTConfig& fft, std::span<const Reading> readings);

// Every probe of `best` under `strategy`, most promising first, without duplicates and without `best` itself.
//
// Groups: each group in declaration order, its structural keys one step at a time -- a structural value opens a
// different set of keys, so it is a branch rather than a dimension to permute, and a step into one is offered only
// with `structuralSteps` -- and its other axes split into bins of at most `maxPermute` in declaration order, each bin's
// cross product enumerated fewest axes moved first and cut at `maxPoints`.  Single: every axis one step at a time.
// Permute: the axes of the keys named, as one cross product, in the same order and not cut short.
//
// Hybrid: what groups offers, then the combo tiers over the readings in `best`'s branch.  The seeds of a bin are its
// best `comboTop` distinct projections of those readings, the background's own first; tier 2 combines the seeds of the
// bins of each cluster of more than one group (clusterGraph()), and of each other group of more than one bin ("Memory
// combined"); tier 3 the seeds of each top-tier group and of each cluster, each taken whole.  Under `bootstrapTree` a
// group is one dimension at tier 2 as well, and a group alone has no tier-2 stage.  A stage is its cross product
// without the background, most promising first -- the highest summed gain of its seeds, a seed's gain being 1 - cost /
// the cost of `best`, and no more than 0 -- and cut at `maxPoints`.  `best` must be the first of the readings in its
// branch, and the rest must follow cheapest first.
//
// A point is dropped where some key it sets is at a value the table would not offer it alongside the rest.
//
// No stage lists more than `listed` points.  One stopped there before its own limit, with points it has not offered
// yet, is named in `unlisted`.  What is listed of a stage is always the start of what a longer `listed` would list of
// it.
[[nodiscard]] ProbeList probesOf(const Env& env, const FFTConfig& fft, const UseConfig& best, const Strategy& strategy,
                                 std::span<const Reading> readings = {}, bool structuralSteps = true,
                                 u32 listed = NO_LIMIT);

// How many steps one best set of `fft` offers, group by group, at tier 1 under `strategy`: the combinations above them
// depend on what is read, and are left out.
struct SearchSize {
  struct GroupSize {
    Group group = Group::None;
    size_t options = 0;
    size_t structural = 0;

    // Each bin's points, and how many of them the strategy offers.
    std::vector<u64> points;
    std::vector<u64> offered;

    // What the group's points come to with every bin whole: maxPermute=all.
    u64 whole = 0;
  };
  std::vector<GroupSize> groups;

  [[nodiscard]] u64 offered() const;
  [[nodiscard]] u64 binned() const;
  [[nodiscard]] u64 whole() const;
};

// Pure: counts points without listing them.  The counts are of the cross products, so a point the table would fit onto
// another is counted as well; for a strategy that does not search by group, nothing.
[[nodiscard]] SearchSize searchSize(const Env& env, const FFTConfig& fft, const UseConfig& best,
                                    const Strategy& strategy);

// Whether a row measured under `row` already answers `probe`: it has every axis the probe moves where the probe puts
// it, and agrees with the probe on every key those axes' keys depend on.  What else the row ran with does not matter,
// so a probe is offered again only once something it depends on has moved -- which is what lets the search re-offer a
// dependent key without re-measuring every key whenever any one of them wins.
[[nodiscard]] bool answeredBy(const Env& env, const FFTConfig& fft, const ProbeList& list, const Probe& probe,
                              const UseConfig& row);

// The `k`th draw of the restart sequence of the entry named `entry`: a joint assignment of every axis, each drawn
// uniformly over its positions, canonical.  Structural axes are drawn first and every other axis after the keys it
// depends on, against what has been drawn so far, so every assignment the table offers has a chance of being drawn.
// A fixed sequence per entry, so that a resumed run draws what the first one would have, and a row can be recognised
// as a restart's by drawing again.
[[nodiscard]] UseConfig restartOf(const Env& env, const FFTConfig& fft, std::string_view entry, u32 k);

}  // namespace tune
