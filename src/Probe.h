// Copyright (C) Jason Lynch

// The per-entry option search: the configurations one step from an entry's best known option set, under the search
// strategy the user chose, and which of them the database already answers.
//
// A step moves one or more axes.  An axis is one key, or for LOADS and STORES one access class -- one digit of one of
// them, or of both where the class chooses its load and store together -- so a probe never moves the packed integer
// as a whole, and never offers a mode the compiler would quietly build as another.
//
// Pure: the probe list is a function of the option table and the best set, and whether a probe is answered is a
// function of the rows.  Nothing is remembered between items, so a resumed run offers exactly what an uninterrupted
// one would.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tune {

// A stage -- one bin of a group -- whose cross product is larger than this is cut short, fewest axes moved first.
inline constexpr u32 MAX_POINTS = 64;

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

  // As parseStrategy() reads it.
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

struct Probe {
  // Canonical.
  UseConfig config;

  // What offered it: the group and, where the group is split, its bin ("Width", "Placement 2"); "single"; "permute".
  std::string stage;

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
};

// Every probe of `best` under `strategy`, most promising first, without duplicates and without `best` itself.
//
// Groups (and hybrid, until the combo tiers are built): each group in declaration order, its structural keys one step
// at a time -- a structural value opens a different set of keys, so it is a branch rather than a dimension to permute
// -- and its other axes split into bins of at most MAX_PERMUTE in declaration order, each bin's cross product
// enumerated fewest axes moved first and cut at MAX_POINTS.  Single: every axis one step at a time.  Permute: the
// axes of the keys named, as one cross product, in the same order and not cut short.
//
// A point is dropped where some key it sets is at a value the table would not offer it alongside the rest.
[[nodiscard]] ProbeList probesOf(const Env& env, const FFTConfig& fft, const UseConfig& best, const Strategy& strategy);

// Whether a row measured under `row` already answers `probe`: it has every axis the probe moves where the probe puts
// it, and agrees with the probe on every key those axes' keys depend on.  What else the row ran with does not matter,
// so a probe is offered again only once something it depends on has moved -- which is what lets the search re-offer a
// dependent key without re-measuring every key whenever any one of them wins.
[[nodiscard]] bool answeredBy(const Env& env, const FFTConfig& fft, const ProbeList& list, const Probe& probe,
                              const UseConfig& row);

}  // namespace tune
