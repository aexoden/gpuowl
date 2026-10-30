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

// How far below its accuracy reference's reading at the table top a set that changes the rounding may read, as a
// fraction of that z: the set serves up to where it is that accurate, which is the accuracy the default rounding is
// trusted with at the table's reach.  Two sets whose arithmetic is the same read exactly the same z; two that round
// otherwise read apart by sampling noise as well as by what the arithmetic costs, and the noise grows with z.
// `-tune accuracy,repeat=5` on a Tesla P100 (CUDA and OpenCL) and an RTX A4000, 2026-10-30, read one gap's sd at 0.1 to
// 2.1 Z, up to about 4.5% of z, and TAIL_KERNELS, TAIL_TRIGS, TAIL_TRIGS32 and MM2_CHAIN costing nothing readings tell
// apart from it: none of about 150 such gaps read further below than 6.2%.  A real cost is a little reach: z falls by
// some 8 (FP64 at 18 bpw) to 20 (the FP32 hybrids at 34 bpw) per 1% of exponent near the top, so 256RADIX8_H=1
// (-10 to -16%) or TABMUL_CHAIN32=1 (-30 to -43%) is as accurate as the default at the top within about half a percent.
inline constexpr double ACCURACY_SLACK = 0.07;

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
// such a key is held to, read at the table top.  Canonical (canonicalConfig), as the readings key it.

// The keys of `options` held at a value that changes the rounding on `fft`, each spelled as the value whose arithmetic
// it compiles (Option::roundingValue), and nothing else.  Two sets of one FFT with the same answer round alike, bit for
// bit, so a rounding reading of one is a reading of the other.  That holds while every rounding key's default depends
// on the FFT alone, as canonicalConfig() then drops the same defaults from both.
[[nodiscard]] UseConfig roundingOf(const Env& env, const FFTConfig& fft, const UseConfig& options);

// Whether a card's rounding readings serve every set that rounds alike (Readings::roes): where `-tune accuracy` has
// shown the option table's accuracy flags to hold, on nVidia under CUDA and OpenCL (a Tesla P100 and an RTX A4000,
// 2026-10-30).  Elsewhere every set is read on its own until a sweep there shows the same.
[[nodiscard]] bool sharesRounding(const Env& env);
[[nodiscard]] UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& options);

// Whether a reading is safe for an FFT of `type`.  A reading with too few rounding errors to say anything is taken as
// safe: the arithmetic hardly rounds there.  `reference`, for a set that changes the rounding: its accuracy reference's
// reading at the table top, which the set must read within ACCURACY_SLACK of, and never below the floor.  Where that
// reading failed its check or is below the floor itself, the floor alone holds; a set that rounds visibly where the
// reference hardly rounds is never safe.
[[nodiscard]] bool safe(enum FFT_TYPES type, const RoeReading& reading, const RoeReading* reference = nullptr);

// `floor`: the lowest exponent the configuration would serve in the workload; a reach that settles below it rejects it.
// `reference`, for a set that changes the rounding: its accuracy reference's readings, of which the one at the table
// top counts.  The set's own readings count once that one has been taken, which is owed after the set's first.
[[nodiscard]] Verdict verdictOf(const FFTConfig& fft, std::span<const RoeReading> readings, u64 floor,
                                const std::vector<RoeReading>* reference = nullptr);

// The verdict on `options` on `fft` from what `readings` hold, held to its accuracy reference where it changes the
// rounding: what publishing, the duels and the contender table all judge a set by.
[[nodiscard]] Verdict verdictFor(const Readings& readings, const FFTConfig& fft, const UseConfig& options, u64 floor);

// The set whose reading `verdict` owes: `options`, or its accuracy reference where that is the one owed.
[[nodiscard]] UseConfig owedOptions(const Readings& readings, const FFTConfig& fft, const UseConfig& options,
                                    const Verdict& verdict);

}  // namespace tune
