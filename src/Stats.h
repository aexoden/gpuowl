// Copyright (C) Jason Lynch
//
// Measurement statistics for GPU timing.

#pragma once

#include "common.h"

#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace tune {

// Blocks timed per call, after one warm-up block.
inline constexpr u32 BLOCKS_PER_CALL = 4;

// No row is concluded from fewer calls than this.
inline constexpr u32 MIN_CALLS = 2;

// A block above the median by more than this factor is considered an outlier.
inline constexpr double SPIKE_CUT = 1.02;

// The greatest fraction of blocks that can be considered outliers.
inline constexpr double SPIKE_MAX_FRACTION = 0.25;

// Standard errors added to a mean before it is ranked.
inline constexpr double PESSIMISM_SIGMA = 2.0;

// Calls a session spends warming the device before the first one it records.  The first Gpu of a process reads slow on
// every vendor tried -- by 0.7-1.5% on gfx906 and by a few tenths on nVidia -- which is more than the differences a
// race resolves, and without this it lands on whichever candidate happens to go first.
inline constexpr u32 SESSION_WARM_CALLS = 1;

// Bands for the ratio of how far repeated measurements of one configuration actually move to how far the spread they
// declare says they should.  Judgement rather than a hypothesis test: over a handful of readings the ratio is itself
// noisy, so the bands say which way to lean.
//
// A row's bar is built from the row's own blocks, so a device whose readings jump also declares a wide bar and the
// row-level ratio stays near 1 by construction -- calibration, not sensitivity.  What the blocks inside one call
// predict is fixed independently of the jumping, which is why the sensitive band is there and is scaled by
// sqrt(blocks per call): a row's bar can exceed the blocks' prediction by at most that much (Sec 7.2).
inline constexpr double NOISE_BLOCK_SIGMAS = 3.0;  // x sqrt(blocks per call): above this the block spread says nothing
inline constexpr double NOISE_DISTURBED = 3.0;     // row readings moving this much further than the bar they declared
inline constexpr double NOISE_DRIFTING = 2.0;      // a trend between rows, which is the anchor's job and not the bar's

// A well-behaved card's row-level ratio is expected between 1/sqrt(blocks per call) and 1, not at 1: Sec 7.2's bound
// on between-call scatter is exact where that scatter carries everything and conservative by up to
// sqrt(blocks per call) where it does not.  So a bar is "wider than it needs to be" only below that floor by a
// further margin -- at 1 the band would fire on a perfectly calibrated device.
inline constexpr double NOISE_CONSERVATIVE = 1.5;  // x below the 1/sqrt(blocks per call) floor

// Whether what moved the readings was a trend rather than noise, from the shape of the series alone: over n readings a
// straight ramp puts `observed` at sqrt(n(n+1)/6) times `neighbour` -- 1.8 over four, 3.5 over eight -- while
// stationary noise puts the two together.  A trend is the anchor's problem (Sec 7.3), so it is named as one instead of
// being reported as an error bar that lies.
inline constexpr double NOISE_TREND_SHAPE = 1.8;

// The result of a call.
enum class Status : u8 { Ok, Err, NoCompile, Unsupported, Lost };

[[nodiscard]] const char* toString(Status status);
[[nodiscard]] std::optional<Status> parseStatus(std::string_view text);

// Mean, sample standard deviation and standard error of the mean over a set of samples.
struct Stats {
  double mean = 0;
  double sd = 0;
  u32 n = 0;

  [[nodiscard]] double sem() const;
  [[nodiscard]] double relSem() const;
};

[[nodiscard]] Stats statsOf(std::span<const double> samples);

struct CoreStats {
  Stats stats;

  u32 dropped = 0;

  // Too many samples would have been dropped as outliers.
  bool declined = false;
};

[[nodiscard]] CoreStats coreStats(std::span<const double> samples, double spikeCut = SPIKE_CUT,
                                  double maxFraction = SPIKE_MAX_FRACTION);

// One timing in microseconds per iteration.
struct Measurement {
  double mean = 0;
  double stddev = 0;
  u32 blocks = 0;
  u32 calls = 0;
  double drift = 1;

  Status status = Status::Ok;
  u64 ts = 0;

  [[nodiscard]] double cost() const;
  [[nodiscard]] double costStddev() const;

  [[nodiscard]] bool ok() const { return status == Status::Ok && blocks > 0; }
};

[[nodiscard]] Measurement measurementOf(const CoreStats& core, double drift = 1, u64 ts = 0);

[[nodiscard]] double standardError(const Measurement& m);
[[nodiscard]] double relStandardError(const Measurement& m);

[[nodiscard]] double pessimisticCost(const Measurement& m);

[[nodiscard]] bool concluded(const Measurement& m);

// One call, as it describes itself.
struct CallSummary {
  double mean = 0;
  double sd = 0;
  u32 blocks = 0;

  // The anchor ratio in force when the call was taken; 1 leaves the reading as measured.
  double drift = 1;
};

// How far a set of readings moved, against how far the spread they declare says they should have.
struct Spread {
  u32 n = 0;
  double mean = 0;

  double predicted = 0;  // the declared standard error of one reading, pooled over the readings
  double observed = 0;   // the standard deviation of the readings themselves
  double neighbour = 0;  // the same, from successive differences alone, so a smooth trend does not enter it
  double trend = 0;      // last reading minus first

  [[nodiscard]] double ratio() const { return against(observed); }
  [[nodiscard]] double detrended() const { return against(neighbour); }

private:
  // Readings that declare no spread at all and then move anyway are unbounded, not zero: reporting zero would print a
  // figure contradicting the verdict beside it, and would hide the one case where the declared bar is most wrong.
  [[nodiscard]] double against(double what) const {
    if (predicted > 0) { return what / predicted; }
    return what > 0 ? std::numeric_limits<double>::infinity() : 0;
  }
};

enum class NoiseVerdict : u8 { TooFew, Matches, Conservative, Drifting, Disturbed };

[[nodiscard]] const char* toString(NoiseVerdict verdict);

struct NoiseReport {
  // Between calls, against what the blocks inside one call predict.  Blocks share a warm-up, a compile and one moment
  // of the clock curve, so this is expected above 1 and says how much (Sec 7.2); what it is good for is telling a
  // trend apart from correlation, which move `observed` the same way and want opposite fixes.
  Spread call;

  // Between rows of `callsPerRow` calls, against the error bar such a row declares through standardError().  This is
  // the bar the search acts on, so this is the ratio that has to read near 1.
  Spread row;

  u32 callsPerRow = MIN_CALLS;
  NoiseVerdict verdict = NoiseVerdict::TooFew;

  // Whether measurements taken here can be raced against each other at all.
  [[nodiscard]] bool trustworthy() const {
    return verdict == NoiseVerdict::Matches || verdict == NoiseVerdict::Conservative;
  }
};

// Pure.  Calls are in the order they were taken: rows are consecutive runs of `callsPerRow`, and a trailing partial row
// is left out, since a bar built from fewer calls is not the bar the others declared.
[[nodiscard]] NoiseReport noiseOf(std::span<const CallSummary> calls, u32 callsPerRow = MIN_CALLS);

void mergeInto(Measurement& into, const Measurement& add);
[[nodiscard]] Measurement merged(Measurement a, const Measurement& b);

}  // namespace tune
