// Copyright (C) Jason Lynch

// The accuracy gate: whether a configuration's rounding error, at the top of the interval it would be published for, is
// one production can live with.  A pure function of the roe rows, so emission refuses what the gate has not passed
// without any device work, and the queue can ask which readings are still owed.
//
// Every configuration has to clear the floor production itself warns at.  One that moves a key which changes the
// rounding has, besides, to read no worse than the same configuration with those keys at their defaults: the fitted
// table's reach describes the defaults, so a set that spends accuracy has no claim on it.

#pragma once

#include "Eligibility.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "TuneDB.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tune {

// The Gumbel z below which production warns that a run is in danger (Gpu::doBigLog).
[[nodiscard]] double minSafeZ(enum FFT_TYPES type);

// The z fftbpw.h fitted FP64's reach at.  A reading at the published reach that clears it confirms that reach.
inline constexpr double TARGET_Z = 28;

// How far below its default-accuracy reading a set that moves an accuracy key may read.  Two sets whose arithmetic is
// the same read exactly the same z, so this is room for the sampling of a different rounding path, and no more:
// measured on a Tesla P100 at 512:8:512, MM2_CHAIN=1 costs 2.1.
inline constexpr double ACCURACY_SLACK_Z = 0.5;

// The iterations Gpu::measureROE runs, its warm-up among them.
inline constexpr u32 ROE_ITERATIONS = 2000;

// Whether `opts` holds a key that changes the rounding at a value other than its default on `fft`.  Suspected keys
// count: taking one for inert when it is not is the error that costs a residue.
[[nodiscard]] bool movesAccuracy(const Env& env, const FFTConfig& fft, const UseConfig& opts);

// `opts` with every key that changes the rounding at its default, stated rather than dropped so that nothing layered
// underneath can set it otherwise: the set whose reading a set that moves one is held to.
[[nodiscard]] UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& opts);

// The prime the gate reads `span` at: the largest it holds, where the rounding errors are largest.  0 where it holds
// none.
[[nodiscard]] u64 gateExponent(const Interval& span);

enum class GateState : u8 { Owed, Passed, Rejected };

struct GateVerdict {
  GateState state = GateState::Owed;

  // Published beside a passed set.
  Evidence evidence = Evidence::Unvalidated;

  // An owed set's: whether the reading still missing is of the set's accuracy reference rather than of the set, and
  // where the reference is to be read -- where the set itself was, which is the only place the two compare.
  bool owesReference = false;
  u64 referenceAt = 0;

  // A rejected set's reason, for the log.
  std::string why{};
};

// The verdict on one set of `type` from its own reading and, for a set that moves an accuracy key, its reference's; a
// missing reading is owed.  Pure.
[[nodiscard]] GateVerdict judge(enum FFT_TYPES type, const std::optional<RoeRow>& own, bool movesAccuracy,
                                const std::optional<RoeRow>& reference);

// Judges configurations against the roe rows one env has.
class Gates {
public:
  Gates(const TuneDB& db, u32 env, const Env& device);

  // The verdict on `opts` published up to an interval whose gate exponent is `exponent`: a reading of the set counts if
  // it was taken in the regime the exponent runs in, at or above it, and a reading of its accuracy reference only at
  // the exponent the set's own was taken at, since every set reads worse further up.  Exact arithmetic passes without
  // one.
  //
  // A reading is of the option set the kernels were built with: its tunable keys as canonicalConfig() has them, and
  // every other key as written, since a key the table does not search can still change what is built.
  [[nodiscard]] GateVerdict operator()(const FFTConfig& fft, u64 exponent, const UseConfig& opts) const;

private:
  // The latest reading of `opts` at `exponent`, or with `above` at or above it in its regime.
  [[nodiscard]] std::optional<RoeRow> reading(const FFTConfig& fft, u64 exponent, const UseConfig& opts,
                                              bool above) const;

  Env device_;

  // By spec, then by the text of the gate's identity for the set the kernels were built with.
  std::map<std::pair<std::string, std::string>, std::vector<RoeRow>> rows_;
};

}  // namespace tune
