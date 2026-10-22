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
#include "Readings.h"

#include <span>

namespace tune {

// How close a lowered reach's safe and unsafe readings must be for it to be settled.
inline constexpr double REACH_RESOLUTION = 0.01;

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
};

// Whether a reading is safe for an FFT of `type`.  A reading with too few rounding errors to say anything is taken as
// safe: the arithmetic hardly rounds there.
[[nodiscard]] bool safe(enum FFT_TYPES type, const RoeReading& reading);

// `floor`: the lowest exponent the configuration would serve in the workload; a reach that settles below it rejects it.
[[nodiscard]] Verdict verdictOf(const FFTConfig& fft, std::span<const RoeReading> readings, u64 floor);

}  // namespace tune
