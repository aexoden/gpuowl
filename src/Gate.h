// Copyright (C) Jason Lynch

// Whether a configuration's rounding stays safe as far as it would be published for, and if not, how far it does.
//
// A configuration is published for an FFT up to the FFT's table maximum.  It is read there first, where its rounding
// errors are largest.  A reading that is safe confirms that reach; one that is not starts a bisection downward, which
// settles once a safe reading and an unsafe one above it are within REACH_RESOLUTION of each other, and the reach is
// then the safe one.  A configuration whose reach settles below the workload serves none of it.  Pure: a function of
// the readings.

#pragma once

#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Readings.h"

#include <span>
#include <vector>

namespace tune {

// How close a lowered reach's safe and unsafe readings must be for it to be settled.
inline constexpr double REACH_RESOLUTION = 0.01;

// How far below its accuracy reference's reading a set that changes the rounding may read.  Two sets whose arithmetic is
// the same read exactly the same z, so this is room for sampling a different rounding path, and no more: tune measured
// MM2_CHAIN=1 costing 2.1 on a Tesla P100 at 512:8:512.
inline constexpr double ACCURACY_SLACK_Z = 0.5;

// The Gumbel z of the largest rounding error below which a reading is unsafe: the table's own reach reads about 24 on
// FP64 FFTs and well above 6 on the FP32 hybrids.
[[nodiscard]] double minSafeZ(enum FFT_TYPES type);

enum class Accuracy : u8 {
  Exact,      // no rounding: GF31 and GF61 arithmetic is exact
  Confirmed,  // safe at the table maximum
  Lowered,    // safe only up to a reach below it, settled
  Owed,       // a reading is owed, at `next`
  Rejected,   // unsafe throughout the workload
};

[[nodiscard]] const char* toString(Accuracy accuracy);

struct Verdict {
  Accuracy state = Accuracy::Owed;
  u64 reach = 0;  // how far it may be published: the table maximum, or the lowered reach
  u64 next = 0;   // where the next reading is owed
  bool owesReference = false;  // the reading owed at `next` is of the accuracy reference, not of the set
};

// Whether `options` holds a key at a value that changes the rounding on `fft` (Option::changesRounding): such a set spends
// accuracy the fitted table's reach does not allow for, and is held to its accuracy reference.  A value that does not
// parse counts, which is the safe error.
[[nodiscard]] bool movesAccuracy(const Env& env, const FFTConfig& fft, const UseConfig& options);

// `options` with every key that changes the rounding at a value that rounds as its default does: the set one that moves
// such a key is held to, read at the same exponents.  Canonical (canonicalConfig), as the readings key it.
[[nodiscard]] UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& options);

// Whether a reading is safe for an FFT of `type`.  A reading with too few rounding errors to say anything is taken as
// safe: the arithmetic hardly rounds there.  `reference`, for a set that changes the rounding: its accuracy reference's
// reading at the same exponent, which the set must read within ACCURACY_SLACK_Z of; it is never safe where that reading
// is not, nor rounding visibly where the reference hardly rounds.
[[nodiscard]] bool safe(enum FFT_TYPES type, const RoeReading& reading, const RoeReading* reference = nullptr);

// `floor`: the lowest exponent the configuration would serve in the workload; a reach that settles below it rejects it.
// `reference`, for a set that changes the rounding: its accuracy reference's readings.  A reading of the set counts once
// the reference has been read at its exponent, which is owed until then, the highest first.
[[nodiscard]] Verdict verdictOf(const FFTConfig& fft, std::span<const RoeReading> readings, u64 floor,
                                const std::vector<RoeReading>* reference = nullptr);

// The verdict on `options` on `fft` from what `readings` hold, held to its accuracy reference where it changes the
// rounding: what publishing, the duels and the contender table all judge a set by.
[[nodiscard]] Verdict verdictFor(const Readings& readings, const FFTConfig& fft, const UseConfig& options, u64 floor);

// The set whose reading `verdict` owes: `options`, or its accuracy reference where that is the one owed.
[[nodiscard]] UseConfig owedOptions(const Readings& readings, const FFTConfig& fft, const UseConfig& options,
                                    const Verdict& verdict);

}  // namespace tune
