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

// The most calls an entry's first packet and a restart's local search take, and the share of first looks and packet
// items drawn at random rather than by rank.
inline constexpr u32 FIRST_PACKET = 4;
inline constexpr u32 RESTART_PACKET = 8;
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
};

// The next item of the coverage lane: a first look at an entry nothing has read, or the next item of the least
// developed entry's packet; nothing where every entry is blocked.
[[nodiscard]] std::optional<Proposal> coverageStep(const AcquireContext& context, std::mt19937_64& rng);

// The next item of the restarts lane: the open restart's next local step from its own best, or a new start.
[[nodiscard]] std::optional<Proposal> restartStep(const AcquireContext& context, const TuneDB& db, u32 env,
                                                  std::mt19937_64& rng);

}  // namespace tune
