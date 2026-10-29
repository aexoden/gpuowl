// Copyright (C) Jason Lynch

// Soaking finalists.  A call times a configuration for a second or so; production runs one for days.  On a card at its
// power limit the chip warms over minutes, costing each iteration more energy, so even a call costed by its energy
// reads 1-2% faster than the card sustains, and by an amount that may differ between configurations.  So the few
// configurations that could serve a line's exponents best are each run for minutes, in turns, the card kept squaring
// throughout: a warm-up on the first to bring the card to the state production keeps it in, then rounds of a slice of
// each.  What each sustained becomes its cost (Readings), and how far that is from what its calls read calibrates the
// configurations not soaked.
//
// Soaks take a share of the tuning time, and none begins before that share has paid for it, so a short run spends
// nothing on them and a long one keeps its finalists soaked as they change.

#pragma once

#include "Lanes.h"
#include "Publish.h"
#include "Readings.h"
#include "TuneDB.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace tune {

struct SoakTarget {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  double cost = 0;  // as the readings have it now
};

// A soak to run: the configurations, cheapest first, at the exponent of the line they would serve.
struct SoakPlan {
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  std::vector<SoakTarget> targets;

  // What tells one soak plan from another: its exponent and its configurations.
  [[nodiscard]] std::string key() const;
};

// The next soak worth running: for the line of this build's frontier serving the most weight that has one, the line's
// configuration and the cheapest rivals of other FFTs or settings that hold its heaviest exponent, at most `top` in
// all and none dearer than the line by more than `margin`, where one of them has not been soaked yet.  Plans whose
// key is in `tried` are passed over.  Nothing where every such line's finalists have been soaked.
[[nodiscard]] std::optional<SoakPlan> planSoak(const Publication& publication,
                                               const std::vector<std::vector<Candidate>>& candidates,
                                               const Readings& readings, const SoakSettings& settings,
                                               const std::set<std::string>& tried = {});

// The seconds the env's soaks have run, warm-ups included.
[[nodiscard]] double soakTime(const TuneDB& db, u32 env);

// Whether a soak of `plan` fits the share: the time soaked with it would be at most `share` of the tuning time.
[[nodiscard]] bool soakDue(double soaked, double elapsed, const SoakPlan& plan, const SoakSettings& settings);

}  // namespace tune
