// Copyright (C) Jason Lynch

// The drift anchor: one designated configuration, re-timed through a session, whose readings say how much the device
// itself has moved since the row a database is about to be compared with was taken.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// How often the anchor is re-timed.  The only item scheduled by time rather than by value.
inline constexpr double ANCHOR_EVERY_SEC = 300;

// Iterations per anchor block, fixed here rather than taken from the block size in force.  The anchor is a reference
// clock and not an estimate of anything production does, so it gains nothing by following `-block`, and a reading
// taken over a different amount of work is not comparable with the baseline it divides -- the per-iteration cost of a
// block carries its boundary drain and its Gerbicz check.  Moving this constant invalidates every stored baseline, in
// the way any host-side change to how a call is timed does.
inline constexpr u32 ANCHOR_BLOCK_SIZE = 1000;

// Movement of the anchor, as a fraction of its first reading, that is reported and that is treated as more than the
// correction can be trusted to absorb.
inline constexpr double DRIFT_WARN = 0.02;
inline constexpr double DRIFT_ALARM = 0.10;

// How long to leave the device alone before asking the anchor again, once it has read past the alarm.
inline constexpr double ALARM_COOLDOWN_SEC = 5;

enum class DriftLevel : u8 { Steady, Warn, Alarm };

[[nodiscard]] const char* toString(DriftLevel level);

// Where `ratio` falls, in either direction: a device that has sped up has moved just as far as one that has slowed.
[[nodiscard]] DriftLevel driftLevelOf(double ratio);

// The configuration a session's drift is measured against, and where it is timed.
struct AnchorSpec {
  std::string fft;
  u64 exponent = 0;

  [[nodiscard]] bool valid() const { return !fft.empty() && exponent > 0; }

  // "<spec>@<exponent>", as a session row carries it.
  [[nodiscard]] std::string text() const;

  [[nodiscard]] bool operator==(const AnchorSpec&) const = default;
};

[[nodiscard]] std::optional<AnchorSpec> parseAnchorSpec(std::string_view text);

// What an env's anchor is chosen among at `exponent`: for each FFT type production chooses among, the smallest shape
// whose default variant is eligible there, at that variant and the automatic carry, leaving out prime-factor middles.
// Smallest-that-fits is the cheapest thing in a family to re-time.  In type order, FP64 first; empty for an exponent
// nothing can hold.
[[nodiscard]] std::vector<AnchorSpec> anchorCandidates(u64 exponent);

// One candidate's reading, in microseconds per iteration.
struct AnchorReading {
  AnchorSpec anchor;
  double us = 0;
};

// The anchor a race of the candidates gives: the cheapest reading, the earlier candidate on a tie.  A card is anchored
// on what it is good at -- one whose FP64 runs at a thirty-second of its FP32 has no use for an FP64 anchor that costs
// a sixth of the run to re-time.  Nothing where no candidate gave a reading.
[[nodiscard]] std::optional<AnchorSpec> raceWinner(const std::vector<AnchorReading>& readings);

// The anchor's readings, against the first one taken for this env.
struct AnchorState {
  // The first reading of the first session of the env, which every later session divides by.  Zero until one exists,
  // and then the next reading establishes it.
  double baseline = 0;

  double latest = 0;
  double ratio = 1;
  u32 readings = 0;

  // The device stayed past the alarm across a re-timing, so the correction is doing more work than it should be
  // trusted with.
  bool alarmed = false;

  // Folds a reading in and says where the result sits.  A reading of zero or less is not a reading.
  DriftLevel observe(double mean);
};

}  // namespace tune
