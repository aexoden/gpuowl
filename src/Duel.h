// Copyright (C) Jason Lynch

// Duels.  Calls of one configuration scatter 0.1-0.5% from one to the next, more than the blocks inside a call do, and
// a card's speed wanders between sessions by amounts of its own; a set takes a line on the mean of as few as two calls,
// so among many close rivals the cheapest-reading is often the luckiest.  So before a configuration takes a line from
// the one that held it, by less than a margin, the two are timed in turns -- incumbent, challenger, challenger,
// incumbent, and again -- each pair of neighbouring calls one ratio, in which whatever drifts slowly cancels.  The
// duel goes on while the interval of the ratios straddles 1 and is wider than the epsilon either side of it, up to its
// most pairs.  Only a challenger shown cheaper takes the line: an even or undecided duel leaves it where it was
// (Readings ranks the loser behind the winner).
//
// Duels take a share of the tuning time; while they are over it, a line goes on changing hands on calls alone, and its
// duel is owed until the share has room for it.

#pragma once

#include "Lanes.h"
#include "Publish.h"
#include "Readings.h"
#include "TuneDB.h"

#include <optional>
#include <set>
#include <span>
#include <string>

namespace tune {

struct DuelSide {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  double cost = 0;  // as the readings have it now
};

// A duel to run: a line's configuration, which took it, and the cheapest read before it that would serve the line's
// heaviest exponent in its place.
struct DuelPlan {
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;  // the line's heaviest
  DuelSide incumbent;
  DuelSide challenger;
  double weight = 0;  // of the grid points the line serves

  // What tells one duel plan from another: its two configurations.
  [[nodiscard]] std::string key() const;
};

// The next duel owed: for the line of this build's frontier serving the most weight that owes one, its configuration
// against the cheapest read before it that holds its heaviest exponent -- with its accuracy settled that far -- where
// that one is no more than `margin` dearer, and the two have never duelled or both been soaked.  A configuration read
// after the line's took nothing from it, however close behind it reads.  Plans whose
// key is in `tried` are passed over.
[[nodiscard]] std::optional<DuelPlan> planDuel(const Publication& publication,
                                               const std::vector<std::vector<Candidate>>& candidates,
                                               const Readings& readings, const DuelSettings& settings,
                                               const std::set<std::string>& tried = {});

// Where a duel stands after the pairs read so far.
struct DuelVerdict {
  DuelOutcome outcome = DuelOutcome::Undecided;
  bool open = true;  // more pairs are wanted; `outcome` is not yet one
  u32 pairs = 0;
  double ratio = 1;  // the challenger's cost over the incumbent's: the geometric mean of the pairs
  double lo = 0;     // its 95% interval, by Student's t over the pairs
  double hi = 0;
};

// `logRatios`: of each pair, the log of the challenger's cost over the incumbent's.  Decided once the interval lies
// below 1 (the challenger), above it (the incumbent), or within the epsilon either side (even); undecided once there
// are `settings.pairs` pairs with none of those.
[[nodiscard]] DuelVerdict judge(std::span<const double> logRatios, const DuelSettings& settings);

// The env's duels so far: how many ended each way, and how long they ran.
struct DuelTally {
  u32 challenger = 0;
  u32 incumbent = 0;
  u32 even = 0;
  u32 undecided = 0;
  double seconds = 0;

  [[nodiscard]] u32 total() const { return challenger + incumbent + even + undecided; }
};
[[nodiscard]] DuelTally duelTally(const TuneDB& db, u32 env);

// "7 in 12.5 min: 2 won by the challenger, 1 by the incumbent, 1 even, 3 undecided"
[[nodiscard]] std::string toString(const DuelTally& tally);

// The seconds the env's duels have run.
[[nodiscard]] double duelTime(const TuneDB& db, u32 env);

// Whether a duel may begin: the time duelled so far is under the share of the tuning time.
[[nodiscard]] bool duelDue(double dueled, double elapsed, const DuelSettings& settings);

}  // namespace tune
