// Copyright (C) Jason Lynch

// What stands in for the Gerbicz check when an LL configuration is timed.
//
// LL has no self-check: the Gerbicz identity closes over squarings, not over x -> x^2 - 2.  What it has instead is that
// the sequence starts from a fixed seed and every step is determined by the exponent, so the residue after a given
// number of iterations is the same for every correct FFT, variant and option set.  A timing call's residue is checked
// against a reference for its (exponent, iterations).
//
// No single reading may decide that reference, nor condemn a configuration: a card without ECC flips a bit now and
// then, and nothing else would catch it.  So the reference is a vote of built-in-default readings on different FFTs,
// and a reading that disagrees with it is taken again before anything is concluded.  Only built-in defaults vote: an
// option that breaks the LL carry computes the same wrong sequence on every FFT, so two readings under it would agree.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "TuneDB.h"

#include <functional>
#include <optional>
#include <vector>

namespace tune {

// How many FFTs' built-in defaults are read at one (exponent, iterations) before the vote is given up there.  Two agree
// unless one of them was hit by a fault or computes LL wrongly, so four leaves room for one of each.
inline constexpr u32 LL_MAX_WITNESSES = 4;

// Whether `ran` sets no tunable key: the configuration upstream would run.
[[nodiscard]] bool atBuiltInDefaults(const UseConfig& ran);

// The env's readings of built-in defaults at (exponent, iters), in the order they were taken.
[[nodiscard]] std::vector<RefRow> referenceReadings(const TuneDB& db, u32 env, u64 exponent, u64 iters);

// The residue the first two readings on different FFTs agreed on, in the order they were taken; empty while none have.
// A later agreement never revises an earlier one.
[[nodiscard]] std::optional<u64> agreedResidue(const std::vector<RefRow>& readings);

// The FFTs whose built-in defaults are read to settle a reference at `exponent`: `own` first, since it holds the
// exponent by construction, then as unlike it as the device allows -- no arithmetic in common, then another type, then
// another shape of its own type -- the cheapest first within each.  Only what `env` can compile.
[[nodiscard]] std::vector<FFTConfig> witnessOrder(const Env& env, const FFTConfig& own, u64 exponent);

// Reads the residue the built-in defaults of `fft` give; empty where no reading was taken.
using ReadWitness = std::function<std::optional<u64>(const FFTConfig& fft)>;

// The reference at (exponent, iters) for session `sess`'s env, reading the witnesses in `order` that have no reading
// there yet until two agree or LL_MAX_WITNESSES FFTs have been read.  Each reading is recorded as a `ref` row.  Empty
// where no two agree.
[[nodiscard]] std::optional<u64> settleReference(TuneDB& db, u32 sess, u64 exponent, u64 iters,
                                                 const std::vector<FFTConfig>& order, const ReadWitness& read);

}  // namespace tune
