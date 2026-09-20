// Copyright (C) Jason Lynch

// Turning a (FFT, test kind, exponent, -use options) into a timing with an error bar, and into a verdict on the
// rounding error the configuration spends to get it.

#pragma once

#include "FFTConfig.h"
#include "GpuCommon.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "UseResolve.h"

struct IterSamples;  // Gpu.h

namespace tune {

// The Gumbel z a configuration's largest rounding error has to clear to be usable at all.
[[nodiscard]] double minSafeZ(enum FFT_TYPES type);

// One call to the GPU.
struct Call {
  Measurement measurement;

  // Every timed block, spikes included, in microseconds per iteration.
  std::vector<double> usPerIt;

  u32 dropped = 0;
  bool declined = false;

  u64 res64 = 0;
  bool checkOk = true;

  // The options the kernels were built with.
  UseConfig ran;
};

// Spike rejection and the statistics over what survives.  Pure; a failed Gerbicz check makes the row an error.
[[nodiscard]] Call summarize(const IterSamples& samples);

// Builds a Gpu for the configuration and times it.  Exceptions from the build or the run propagate.
[[nodiscard]] Call timeCall(GpuCommon shared, const FFTConfig& fft, TestKind kind, u64 exponent,
                            const UseConfig& options, u32 nBlocks = BLOCKS_PER_CALL, u32 blockSize = 1000);

// Results from a rounding error check.
struct RoeCheck {
  // False for a pure NTT: GF31 and GF61 arithmetic is exact, so there is no rounding error to measure.
  bool applicable = false;

  double z = 0;  // the Gumbel z of the largest rounding error against 0.5
  u32 n = 0;     // rounding errors behind z
  double maxRoe = 0;
  double minZ = 0;
  u64 exponent = 0;  // where it was measured
  bool checkOk = true;

  [[nodiscard]] bool conclusive() const { return applicable && n > 2; }

  [[nodiscard]] bool passed() const { return !applicable || (checkOk && (n <= 2 || z >= minZ)); }
};

// Measures the rounding error of one configuration at `exponent`.
//
// Always on the PRP kernel set: only the PRP carry kernels have a ROE variant, so an LL Gpu collects no rounding errors
// at all.
[[nodiscard]] RoeCheck roeCheck(GpuCommon shared, const FFTConfig& fft, const UseConfig& options, u64 exponent);

// The -measure subcommand: times one configuration, prints its blocks, and reports its rounding error.  Returns false
// on a failure worth a non-zero exit status.
[[nodiscard]] bool runMeasure(GpuCommon shared, const std::string& fftSpec, u64 exponent);

}  // namespace tune
