// Copyright (C) Jason Lynch

// The staged search (-tune search=staged): what to measure next by a fixed order of plain rules, with no model.
//
// Each item comes from the first stage that has one, and every stage is a function of the readings alone, so a stage
// a reading reopens -- a new best set, a new contender -- is taken up again at once:
//
//   types      a first look at each FFT type of each kind nothing has read: its entry the prior expects least behind
//   coverage   a first look at each shape expected within `margin` of what else serves its exponent: one entry of
//              the shape, the one serving the most weight
//   sweep      upstream's coordinate descent, per FFT type and kind on its leading entry: each position of each
//              option from the leader's best set, until the leader has read every one
//   transfer   the leader's best set on every contender of its type and kind
//   variants   a first look at every other entry -- variant, regime -- of a shape read within `margin`, at its type
//              leader's best set: an entry is compared tuned with what it would replace, never on its defaults alone
//   structure  each structural move from each contender's best set: never pruned on what its defaults read
//   local      single moves from the best set of each contender's structural branches, a branch drawn with weight
//              falling as 1/rank of how far behind it is, until every move from its best has been read
//
// A contender is an entry whose best reading is within `margin` of what else serves its exponent.  An entry's
// expected cost is its best reading, or before it has one, the stated prior (Prior.h) scaled by what the card's
// readings of its type and kind make of the prior.  Restarts, the restarts lane's (Explore.h), take a fixed share of
// the items beside the stages, and reach further from the front as the run goes on (randomReach=), where the stages
// never look.  The model is never fitted: what the stages measure is decided by the readings alone.

#pragma once

#include "Acquire.h"
#include "Explore.h"

#include <array>
#include <optional>
#include <random>
#include <string>

namespace tune {

enum class Stage : u8 { Types, Coverage, Sweep, Transfer, Variants, Structure, Local };
inline constexpr size_t STAGES = 7;

[[nodiscard]] const char* toString(Stage stage);
[[nodiscard]] std::optional<Stage> parseStage(std::string_view text);

// The next item of the first stage that has one; nothing where none has.  The context's model is asked only which card
// it is of.
[[nodiscard]] std::optional<Proposal> stagedStep(const AcquireContext& context, const StagedSettings& settings,
                                                 std::mt19937_64& rng);

// How many items each stage has left, the local stage counting the branches with a move unread.
using StagesLeft = std::array<size_t, STAGES>;
[[nodiscard]] StagesLeft stagesLeft(const AcquireContext& context, const StagedSettings& settings);

// "types 0, coverage 3, sweep 40, transfer 2, variants 12, structure 31, local 9"
[[nodiscard]] std::string toString(const StagesLeft& left);

}  // namespace tune
