// Copyright (C) Jason Lynch

// Where an option set stops being accurate enough, derived from rounding-error readings rather than inherited from the
// fitted bpw table.  One reading's z is extrapolated along the local slope of bits per word against z to where the set
// would read the standard it is held to, and that reach is then read to confirm it.
//
// A pure function of the readings: the reach emission publishes and the reading the queue takes next are one replay of
// the same derivation, so a later process resumes it from the database alone and nothing derived can go stale against
// the evidence it came from.
//
// This is the one place where a modelling error produces a wrong residue rather than a slow run, which is why every
// extrapolation is bounded and nothing is concluded from an extrapolation alone: a reach is only ever where a reading
// was taken.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <functional>
#include <optional>
#include <string>

namespace tune {

// How far from the standard a reading may be and still be extrapolated to a reach.  Beyond it the slope is not trusted
// over the distance, so the derivation reads again at its estimate and extrapolates from there.
inline constexpr double EXTRAPOLATION_CAP_Z = 4;

// How far below a reach that failed its confirmation the next one is proposed, in bits per word, and how many times
// before the set is rejected.
inline constexpr double REACH_GUARD_BPW = 0.05;
inline constexpr u32 REACH_TRIES = 2;

// Readings a derivation may take before it proposes a reach.  z falls smoothly with bits per word, so a secant along it
// settles within the cap in two or three; one that has not after this many is not converging on anything.
inline constexpr u32 MAX_DERIVE_READINGS = 8;

// Bits per word per unit of z, as upstream's -ztune clamps it.  Measured on a Tesla P100: 0.014-0.018 near z 28 on
// 512:15:512, 0.013-0.024 near it on 2:512:8:512.
inline constexpr double MIN_SLOPE = 0.005;
inline constexpr double MAX_SLOPE = 0.025;

// The slope before a set has two readings to fit one from: -ztune's 0.015 at 4M words falling to 0.012 at 7.5M,
// linear in log size and continued past both ends, clamped.
[[nodiscard]] double startSlope(u64 words);

struct ZReading {
  u64 exponent = 0;
  double z = 0;

  // Rounding errors large enough to be recorded; with two or fewer there is nothing to fit z to, which only happens
  // when the errors are far too small to matter.
  u32 n = 0;
  bool checkOk = true;
};

// The slope between two readings, clamped: bits per word fall as z rises, so it is positive.  The starting guess
// where the two cannot give one -- the same exponent, or one of them with too few errors to have a z.
[[nodiscard]] double fitSlope(const ZReading& a, const ZReading& b, u64 words);

// What a reach is derived towards, and what a reading taken there has to show to confirm it.  The two differ only by
// the room a different rounding path needs for its sampling, so that a set whose arithmetic is its reference's derives
// exactly its reference's reach.
struct Standard {
  double aim = 0;
  double bar = 0;
};

struct ReachProblem {
  u64 words = 0;

  // Where a reach may be proposed: the interval the set would be published for, no higher than the carry can run.
  u64 lo = 0;
  u64 hi = 0;

  Standard standard{};
};

enum class ReachState : u8 { Owed, Confirmed, Rejected };

struct ReachOutcome {
  ReachState state = ReachState::Owed;

  // Where the next reading is to be taken while owed, and the reach once confirmed: always a prime a reading was taken
  // at, never an extrapolation.
  u64 exponent = 0;

  // A rejection's reason, for the log.
  std::string why{};
};

// The latest reading of the set at exactly `exponent`, if one was taken.
using ReadingAt = std::function<std::optional<ZReading>(u64 exponent)>;

// Derives a reach from `first` and whatever readings `at` holds at the exponents the derivation asks for, in either
// direction: down from a reading that falls short of the standard, up from one that clears it.  Pure.
[[nodiscard]] ReachOutcome deriveReach(const ReachProblem& problem, const ZReading& first, const ReadingAt& at);

// The largest exponent the carry `fft` runs with can take at all: a pinned 32-bit carry stops where carry32BPW() says,
// and in any case below 19 bits per word, where the CARRY32 code breaks.  No limit otherwise.
[[nodiscard]] u64 carryCeiling(const FFTConfig& fft);

}  // namespace tune
