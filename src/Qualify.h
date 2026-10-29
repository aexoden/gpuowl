// Copyright (C) Jason Lynch

// -measure: whether what the tuner reads on this card is what production runs at.  The tuner times a configuration in
// calls of a few seconds, between which the card idles; production squares for days.  A card whose clocks follow its
// temperature or its power budget runs a short call faster than it sustains, and by an amount that differs between
// configurations, so readings that are each repeatable can still rank configurations as production would not.
//
// One configuration is measured five ways, with the card's own readings (Telemetry.h) beside each:
//
//   bursts      calls as the tuner times them, each after `idle` seconds of the card idling
//   soak        squaring without a break until three minutes read like the three before, at most `soak` long
//   sustained   `minutes` more minutes of it: what production runs at, and how far one minute strays from another
//   heated      calls with the card kept squaring on the last Gpu while the next is built, so that it never cools
//               between them, and `heat` seconds of the call's own load before its timed blocks: a way the tuner
//               could time, on a card whose clocks it cannot lock
//   hot bursts  the bursts again, straight after, with no idle before them but the build
//
// and the verdict says whether the bursts read faster than the card sustains, whether the heated calls do, whether
// the sustained minutes wander, and what to do about it.

#pragma once

#include "common.h"
#include "GpuCommon.h"
#include "Telemetry.h"
#include "TestKind.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// What -measure was asked for: an FFT spec, then comma-separated settings.
struct MeasureArgs {
  std::string fft;
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;      // 0: the largest prime the FFT holds
  u32 bursts = 6;        // calls in each round of bursts
  double idle = 5;       // seconds the card idles before each of the first bursts: the gap the tuner leaves
  double soak = 900;     // the most seconds the soak runs
  double settle = 0.001; // three minutes within this of the three before end the soak
  u32 minutes = 5;       // sustained minutes
  double heat = 3;       // seconds of a heated call's own load before its timed blocks
  double gap = 0.01;     // bursts faster than sustained by more than this read optimistically
  double spread = 0.005; // sustained minutes further apart than this wander
};

// "<fft>[,kind=prp|ll][,exp=<E>][,bursts=<n>][,idle=<t>][,soak=<t>][,settle=<p>%][,minutes=<n>][,heat=<t>]
// [,gap=<p>%][,spread=<p>%]".  Throws a message for anything it cannot read.
[[nodiscard]] MeasureArgs parseMeasureArgs(std::string_view text);

// A stretch of uninterrupted squaring: when it ended (seconds on the sampler's clock), its mean and its blocks.
struct Chunk {
  double end = 0;
  double us = 0;
  u32 blocks = 0;
};

// The mean of each whole minute since `from`, by the chunks that ended in it, weighted by their blocks.  A minute is
// whole once a chunk has ended after it; a whole minute in which no chunk ended (one longer than a minute) is skipped.
[[nodiscard]] std::vector<double> minuteMeans(std::span<const Chunk> chunks, double from);

// The minutes compared at a time to say whether a soak has settled.
inline constexpr size_t SETTLE_MINUTES = 3;

// The last minute of the first SETTLE_MINUTES whose mean lies within `settle` of the mean of the SETTLE_MINUTES before
// them; nothing where none does yet.  Minutes, not a minute: a card at its power limit swings 0.5-1% from one minute
// to the next as the limit moves its clocks (a 4070 SUPER read 204.6-206.5 for a quarter of an hour), so no single
// pair settles it, and one pair can agree by chance while the card is still warming.
[[nodiscard]] std::optional<size_t> settledAt(std::span<const double> minutes, double settle);

// What each call of a round drew over its timed blocks: as taken (WindowPower::watts), by the energy counter and by
// the power readings; NaN where the card did not say.  The two disagree where the card's readings lag behind a call.
struct CallWatts {
  std::vector<double> taken;
  std::vector<double> counter;
  std::vector<double> readings;
};

// What the four ways read, and what the card said meanwhile.
struct MeasureResult {
  std::vector<double> bursts;     // microseconds per iteration, one per call
  std::vector<double> soak;       // per minute
  std::optional<size_t> settled;  // the soak minute it settled at; nothing where it ran out of time first
  std::vector<double> sustained;  // per minute
  std::vector<double> hot;        // per call
  std::vector<double> heated;     // per call
  CardSummary burstCard;          // over the bursts' timed blocks alone
  CardSummary sustainedCard;
  CardSummary heatedCard;         // over the heated calls' timed blocks
  CallWatts burstWatts;            // per call, over its timed blocks
  CallWatts hotWatts;
  CallWatts heatedWatts;
  double sustainedWatts = NAN;     // over the sustained minutes
};

[[nodiscard]] double median(std::vector<double> v);

struct MeasureVerdict {
  double burst = 0;      // median of the bursts
  double hot = 0;        // median of the hot bursts
  double heated = 0;     // median of the heated calls
  double sustained = 0;  // median of the sustained minutes
  double burstGap = 0;   // sustained / burst - 1: how much faster a burst reads than the card sustains
  double hotGap = 0;     // the same for the hot bursts
  double heatedGap = 0;  // and for the heated calls
  bool heatedHolds = false;  // the heated calls read within `gap` of sustained
  double spread = 0;     // (max - min) / median of the sustained minutes, or of their 3-minute means where 5 or more

  // What the card drew, where it says: watts and millijoules per iteration in the bursts and sustained, its power
  // limit, and the share of that limit it held for long.
  double burstWatts = NAN;
  double burstMJ = NAN;
  double sustainedWatts = NAN;
  double sustainedMJ = NAN;
  double capW = NAN;
  double held = NAN;
  // What the tuner reckons a call here costs (Readings: its timing, or its energy over HOLD_SHARE of the limit,
  // whichever is more), and how far what the card sustains is from it.
  double estimate = 0;
  double estimateGap = 0;

  bool optimistic = false;
  bool wandering = false;
  std::vector<std::string> lines;  // what it means, and what to do
};

// `vendor` "nvidia", "amd" or empty; `bdf` the card's PCI address, for the commands it suggests.
[[nodiscard]] MeasureVerdict judge(const MeasureResult& result, const MeasureArgs& args, std::string_view vendor,
                                   const std::string& bdf);

// measure-report.json: the readings and the verdict, for a script or a later comparison.
[[nodiscard]] std::string measureJson(const MeasureArgs& args, const std::string& options, u64 exponent,
                                      const std::string& telemetry, const MeasureResult& result,
                                      const MeasureVerdict& verdict);

inline constexpr const char* MEASURE_REPORT = "measure-report.json";

// Runs the measurement on the open device and logs it; false where it could not be completed.
[[nodiscard]] bool runMeasure(GpuCommon shared, const MeasureArgs& args);

}  // namespace tune
