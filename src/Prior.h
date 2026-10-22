// Copyright (C) Jason Lynch

// What an FFT is expected to cost before anything on this card has been measured: its work, N log2 N, times a constant
// per FFT type.  Which type is cheapest depends on the card by up to a factor of eight, so these only order a cold
// start; readings replace them as soon as there are any.

#pragma once

#include "FFTConfig.h"

#include <cmath>

namespace tune {

[[nodiscard]] inline double priorWork(u32 size) { return double(size) * std::log2(double(size)); }

// Picoseconds per unit of work: the lower of two cards' readings at 118063003, rounded down -- an RTX A4000 (fast FP32
// and integer, FP64 at 1/32) and a Tesla P100 (FP64 at 1/2), on the smallest shape of each type holding that exponent.
// The last three types are not ones production chooses among, and borrow from their nearest relative.
[[nodiscard]] constexpr double statedPriorK(enum FFT_TYPES type) {
  switch (type) {
  case FFT64: return 6.25;
  case FFT3161: return 17.7;
  case FFT3261: return 17.0;
  case FFT61: return 11.6;
  case FFT323161: return 27.5;
  case FFT6431: return 13.9;
  case FFT3231: return 17.0;
  case FFT31: return 11.6;
  case FFT32: return 17.0;
  }
  return 6.25;
}

// Microseconds per iteration the stated constant gives `shape`.
[[nodiscard]] inline double statedPrior(const FFTShape& shape) {
  return statedPriorK(shape.fft_type) * priorWork(shape.size()) * 1e-6;
}

}  // namespace tune
