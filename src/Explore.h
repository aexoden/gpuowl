// Copyright (C) Jason Lynch

// What the tuner measures besides what the model proposes, and how the run's time is shared out.
//
// Lanes share the elapsed tuning time:
//
//   model         the model's proposals (Acquire.h)
//   coverage      every entry's first look -- one call at its defaults -- and then more, a round at a time: the entries
//                 the model expects cheapest relative to what else serves their exponents first, but a quarter drawn
//                 at random, so that none waits on the model's opinion of it for good
//   restarts      random starting points, each followed by a short local search from its own best, whatever the
//                 model thinks of it
//   interactions  reserved
//
// The shares follow a timetable of elapsed time, never of results: a lane whose items find nothing keeps its
// entitlement, which is what protects it from a model that is confidently wrong.  Each item is charged to the lane it
// served, and the lane furthest behind its entitlement that has work goes next; a lane with none lends its turn to the
// next, without giving up what it is owed.

#pragma once

#include "Acquire.h"
#include "Lanes.h"
#include "TuneDB.h"

#include <optional>
#include <random>
#include <string>

namespace tune {

// A restart's local search: RESTART_PACKET steps, and more while its best is among its last two, up to RESTART_MOST.
inline constexpr u32 RESTART_PACKET = 8;
inline constexpr u32 RESTART_MOST = 32;

// The share of the coverage lane's picks drawn at random rather than by rank.
inline constexpr double RANDOM_SHARE = 0.25;


struct Proposal {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  Lane lane = Lane::Model;
  std::string what{};  // a word, for the item record
  u32 packet = 0;
  u32 ordinal = 0;
  double predictedUs = 0;  // under the posterior mean
  std::string why{};
  std::string record{};  // the item record's lane word, where it is not the lane's own: "confirm", "audit"
};

// The next item of the coverage lane: a first look at an entry nothing has read, the next item of the least tried
// entry's round, or the next set of a contender's least developed branch -- its defaults, then single moves from its
// fastest set to positions the branch has not read, those the branch opens first; nothing where none is left.
// `elapsed` is the tuning time so far, which sets how far a random pick may reach (Reach).
[[nodiscard]] std::optional<Proposal> coverageStep(const AcquireContext& context, const ExploreSettings& settings,
                                                   double elapsed, std::mt19937_64& rng);

// The next item of the interactions lane: on a contender, from its fastest set, two options that interact -- one
// depending on the other, of one group, or sharing a kernel -- moved together to positions never read together there.
// What two moves do together is what single moves, and a model of their sum, cannot show.
[[nodiscard]] std::optional<Proposal> interactionStep(const AcquireContext& context, const ExploreSettings& settings,
                                                      double elapsed, std::mt19937_64& rng);

// The next step of the audit's trials: restarts as that lane's, but each exactly RESTART_PACKET steps and drawn
// without reading other trials (no recombination), so that the trials are independent; items recorded as "audit".
[[nodiscard]] std::optional<Proposal> auditTrialStep(const AcquireContext& context, const ExploreSettings& settings,
                                                     double elapsed, const TuneDB& db, u32 env, std::mt19937_64& rng);

// What a restart's call is expected to cost on average, in microseconds per iteration: the entries a start may draw
// (within reach), each as likely as the draw makes it, at what it costs (measured, else as the model expects).  How
// long an audit's trials take depends on it, since the reach lets them draw entries far dearer than the frontier.
[[nodiscard]] double expectedStartCost(const AcquireContext& context, const ExploreSettings& settings, double elapsed);

// The next item of the restarts lane: the open restart's next local step from its own best, or a new start.
[[nodiscard]] std::optional<Proposal> restartStep(const AcquireContext& context, const ExploreSettings& settings,
                                                  double elapsed, const TuneDB& db, u32 env, std::mt19937_64& rng);

}  // namespace tune
