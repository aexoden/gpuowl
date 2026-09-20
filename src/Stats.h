// Copyright (C) Jason Lynch
//
// Measurement statistics for GPU timing.

#pragma once

#include "common.h"

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

void mergeInto(Measurement& into, const Measurement& add);
[[nodiscard]] Measurement merged(Measurement a, const Measurement& b);

}  // namespace tune
