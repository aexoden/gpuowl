// Copyright (C) Jason Lynch

// The lanes a tuning run's time is shared between (Explore.h), the timetable of their shares, and the time each has
// had, as the database records it.

#pragma once

#include "TuneDB.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace tune {

enum class Lane : u8 { Model, Coverage, Restarts, Interactions };
inline constexpr size_t LANES = 4;

[[nodiscard]] const char* toString(Lane lane);
[[nodiscard]] std::optional<Lane> parseLane(std::string_view text);

// The lane an item record names for work no lane chose: a gate reading, a configuration's second call.
inline constexpr const char* DUTY = "duty";

// How far behind what else serves its timing exponent an entry may be and still be a contender, whose structural
// branches the coverage lane develops: half its items go to them, the other half to first looks and rounds over every
// entry.  Infinite for every entry read.
inline constexpr double CONTENDERS = 0.15;

// How far from the frontier a random pick -- a first look, a restart's entry -- may reach: to entries that cost at most
// `start` times what else serves their timing exponent (measured, else as the model expects), the factor doubling every
// `doubling` seconds of tuning.  A first half hour has no time for an FFT fourteen times too slow; a long run reaches
// every entry.  `start` infinite for every entry from the first.
struct Reach {
  double start = 1.5;
  double doubling = 1800;

  [[nodiscard]] double at(double elapsed) const;
  [[nodiscard]] bool operator==(const Reach&) const = default;
};

// "1.5x@30m", "2x@1h", or "all".
[[nodiscard]] std::optional<Reach> parseReach(std::string_view text);
[[nodiscard]] std::string toString(const Reach& reach);

// A span of time: "48h", "90m", "2h30m", "30s"; nothing for text that is not one.
[[nodiscard]] std::optional<double> parseDuration(std::string_view text);
// "2h30m", "45m", "1h".
[[nodiscard]] std::string durationText(double seconds);

// The share of a budget kept for its end: fresh calls of what is published, then the audit.
inline constexpr double RESERVE = 0.25;

// The improvement of the workload's cost, as a fraction, that counts as the audit finding something.
inline constexpr double TARGET = 0.005;

// The least a position read on other shapes, or other variants, but not this one counts as new in the model's
// exploration (Acquire.cpp): as much as the axis's effect was learnt to vary between them, but no less.  An option
// that cost 5% on every shape read so far is learnt not to vary, which would otherwise leave the shape where it saves
// 10% unread for good.
inline constexpr double NEW_FLOOR = 0.25;

// The width of the bands of the coverage lane's sweep of the near front, 0 for none: every single move from the best
// set of each shape within the band of what else serves its exponent, the nearest first, then the next band, on to the
// contenders.  Off by default: on simulated cards it reached the shapes it should without springing more traps in two
// hours than the coverage lane's branches and breadth did, and it took a third of their time.
inline constexpr double SWEEP_BAND = 0;

// How far behind what else serves its exponent a restart's best may be for the restart to go on past its start.  One
// that far behind makes no line, and on a slow FFT each of its steps costs as much as the FFT is slow: on two cards a
// day into a run, restarts were most of the time spent on FFTs over twice the cost of the best.
inline constexpr double RESTART_CUT = 0.5;

// A call predicted to cost this many times what is published at its exponent times SHORT_BLOCKS blocks rather than
// BLOCKS_PER_CALL: its reading need only show the FFT far behind, and each block costs as much as the FFT is slow.
inline constexpr double SHORT_CALLS = 2;
inline constexpr u32 SHORT_BLOCKS = 2;

// The soaks of finalists (Soak.h): the share of the tuning time they may take, how many configurations a soak takes
// and how close behind a line one must be, and how long a soak's warm-up and each turn of a configuration last.
inline constexpr double SOAK_SHARE = 0.1;
inline constexpr u32 SOAK_TOP = 3;
inline constexpr double SOAK_MARGIN = 0.03;
inline constexpr double SOAK_WARM = 120;
inline constexpr double SOAK_SLICE = 60;
inline constexpr u32 SOAK_ROUNDS = 3;

// -tune soak=<share>%|0, soakTop=, soakMargin=, soakWarm=, soakSlice=, soakRounds=.
struct SoakSettings {
  double share = SOAK_SHARE;  // 0 for no soaks
  u32 top = SOAK_TOP;
  double margin = SOAK_MARGIN;
  double warm = SOAK_WARM;
  double slice = SOAK_SLICE;
  u32 rounds = SOAK_ROUNDS;

  // How long a soak of `n` configurations runs.
  [[nodiscard]] double length(size_t n) const { return warm + slice * double(rounds) * double(n); }

  [[nodiscard]] bool operator==(const SoakSettings&) const = default;
};

// What the lanes beyond the model's are allowed.

struct ExploreSettings {
  double contenders = CONTENDERS;
  Reach reach{};
  double sweepBand = SWEEP_BAND;
  double restartCut = RESTART_CUT;
};

using LaneShares = std::array<double, LANES>;

// Each lane's share of the elapsed tuning time: `start` at first, moving linearly to `end` over `rampSeconds`.
struct Timetable {
  LaneShares start{0.7, 0.2, 0.1, 0};
  LaneShares end{0.3, 0.3, 0.2, 0.2};
  double rampSeconds = 4 * 3600;

  [[nodiscard]] LaneShares at(double elapsed) const;

  // What each lane is owed after `elapsed` seconds: the integral of its share.
  [[nodiscard]] LaneShares owed(double elapsed) const;

  // Whether `lane` has a share at any time.
  [[nodiscard]] bool serves(Lane lane) const { return start[size_t(lane)] > 0 || end[size_t(lane)] > 0; }

  [[nodiscard]] bool operator==(const Timetable&) const = default;
};

// "70/20/10/0~30/30/20/20@4h" (percentages, model/coverage/restarts/interactions, at the start ~ at the end of the
// ramp), "70/20/10/0" for one split throughout, or "model" for the model alone.  Each split must add up to 100.
[[nodiscard]] std::optional<Timetable> parseTimetable(std::string_view text);
[[nodiscard]] std::string toString(const Timetable& timetable);

// The time the env's items took, per lane: from each item record to the next of its session, or to the session's last
// record, and at least a second.  A session with no item records, from before lanes, counts its calls to the model.
struct LaneTime {
  LaneShares seconds{};
  double duty = 0;

  [[nodiscard]] double total() const;
};
[[nodiscard]] LaneTime laneTime(const TuneDB& db, u32 env);

// "model 21.0 min (70%), coverage 6.2 min (21%), restarts 2.8 min (9%), interactions 0.0 min (0%); duty 1.5 min"
[[nodiscard]] std::string toString(const LaneTime& time);

// The lanes the furthest behind what they are owed first, the model breaking ties.
[[nodiscard]] std::array<Lane, LANES> laneOrder(const Timetable& timetable, const LaneTime& time);

}  // namespace tune
