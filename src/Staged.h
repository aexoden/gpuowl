// Copyright (C) Jason Lynch

// The staged search (-tune search=staged): what to measure next by plain rules in a fixed order, with no model.
//
// First, three stages that cover the card, each taking every item while it has one:
//
//   types      a first look at each FFT type of each kind nothing has read: its entry expected least behind
//   coverage   a first look at each shape expected within `margin` of what else serves its exponent: one entry of
//              the shape, the one serving the most weight
//   sweep      upstream's coordinate descent, per kind on the leading type's leading entry: each position of each
//              option from the leader's best set that no entry of its type has read, until none is left
//
// Then the contenders share the items: an entry whose best reading is within `margin` of what else serves its
// exponent, or one nothing has read of a shape that is.  Each item goes to a shape drawn with weight 1/rank of how far
// behind its best is, plus an equal floor for every shape, so that none is starved; it is the shape's first item of:
//
//   transfer   on its lead, the entry read least behind: its type leader's best set
//   structure  each position of each structural option the shape -- every variant and regime of it -- has not read in
//              its context, from the lead's best set: a branch is tried on every shape near the front, never pruned on
//              what the shape's defaults read
//   variants   the first look at each of its other entries, at its type leader's best set: an entry is compared tuned
//              with what it would replace, never on its defaults alone
//   transfer   its type leader's best set on each of its other entries
//   positions  each position of every other option the shape has not read, from the lead's best set: the sweep, on
//              every shape near the front
//   local      single moves from the best set of each structural branch within `margin` of each of its entries, the
//              lead's first, a branch drawn by 1/rank, until every move from its best has been read
//
// An entry's expected cost is its best reading, or before it has one, the stated prior (Prior.h) scaled by what the
// card's readings of its type and kind make of the prior.  Every stage is a function of the readings alone, so one a
// reading reopens -- a new best set, a new contender -- takes it up again at once.  Restarts, the restarts lane's
// (Explore.h), take a fixed share of the items, and reach further from the front as the run goes on (randomReach=),
// where the stages never look.  The model is never fitted.

#pragma once

#include "Acquire.h"
#include "Explore.h"

#include <array>
#include <optional>
#include <random>
#include <string>

namespace tune {

enum class Stage : u8 { Types, Coverage, Sweep, Transfer, Structure, Variants, Positions, Local };
inline constexpr size_t STAGES = 8;

[[nodiscard]] const char* toString(Stage stage);
[[nodiscard]] std::optional<Stage> parseStage(std::string_view text);

// The next item: of the first of the covering stages with one, else of a shape drawn by rank; nothing where none has
// one.  The context's model is asked only which card
// it is of.
[[nodiscard]] std::optional<Proposal> stagedStep(const AcquireContext& context, const StagedSettings& settings,
                                                 std::mt19937_64& rng);

// What each stage has left: the covering stages' items, and how many shapes' next item is of each of theirs.
using StagesLeft = std::array<size_t, STAGES>;
[[nodiscard]] StagesLeft stagesLeft(const AcquireContext& context, const StagedSettings& settings);

// "types 0, coverage 3, sweep 40, transfer 2, structure 9, variants 12, positions 4, local 1"
[[nodiscard]] std::string toString(const StagesLeft& left);

}  // namespace tune
