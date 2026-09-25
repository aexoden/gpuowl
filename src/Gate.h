// Copyright (C) Jason Lynch

// The accuracy gate: whether a configuration's rounding error, at the top of the interval it would be published for, is
// one production can live with, and where it is not, how far down it is.  A pure function of the roe rows, so emission
// refuses what the gate has not passed without any device work, and the queue can ask which readings are still owed.
//
// Every configuration has to clear the floor production itself warns at.  One that moves a key which changes the
// rounding has, besides, to read no worse than the same configuration with those keys at their defaults: the fitted
// table's reach describes the defaults, so a set that spends accuracy has no claim on it.  A set that falls short at
// the top of its interval is not refused for it: a reach is derived for it below (Reach.h), and it is published up to
// there.

#pragma once

#include "Eligibility.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Reach.h"
#include "TuneDB.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tune {

// The Gumbel z below which production warns that a run is in danger (Gpu::doBigLog).
[[nodiscard]] double minSafeZ(enum FFT_TYPES type);

// The z fftbpw.h fitted FP64's reach at.  A reading at the published reach that clears it confirms that reach, and a
// set whose defaults fall short of the floor at the top of the table's reach is derived a reach where it reads this.
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

  // An owed set's: the reading still missing -- of the set's accuracy reference rather than of the set, and where.
  // A reference is read where the set was, the only place the two compare; a set, at the top of its interval, or
  // where the derivation of its reach asks next.
  bool owesReference = false;
  u64 owedAt = 0;

  // A passed set's: the largest exponent it is published to, and whether that was derived from its readings rather
  // than inherited from the fitted table.
  u64 reach = 0;
  bool derived = false;

  // A rejected set's: whether it was refused for reading too inaccurately where it was read, which a reach lower down
  // may cure, rather than for something no exponent changes.
  bool derivable = false;

  // A rejected set's reason, for the log.
  std::string why{};
};

// What a set is held to, given whether it moves an accuracy key and its reference's reading where the set was read:
// no worse than that reading where the reference clears the floor there, and otherwise, as for a set at default
// accuracy, the fitted standard.  Pure.
[[nodiscard]] Standard standardFor(enum FFT_TYPES type, bool movesAccuracy, const std::optional<RoeRow>& reference);

// The verdict on one set of `type` at the top of its interval, from its own reading and, for a set that moves an
// accuracy key, its reference's; a missing reading is owed.  Pure.
[[nodiscard]] GateVerdict judge(enum FFT_TYPES type, const std::optional<RoeRow>& own, bool movesAccuracy,
                                const std::optional<RoeRow>& reference);

// Judges configurations against the roe rows one env has.
class Gates {
public:
  Gates(const TuneDB& db, u32 env, const Env& device);

  // The verdict on `opts` over `span`, the interval the fitted table gives it in one regime: at the top of it, a
  // reading of the set counts if it was taken in that regime at or above the gate exponent, and a reading of its
  // accuracy reference only at the exponent the set's own was taken at, since every set reads worse further up.  A set
  // that falls short there has a reach derived for it inside `span`, from readings at exactly the exponents the
  // derivation asks for.  Exact arithmetic passes without one.
  //
  // A reading is of the option set the kernels were built with: its tunable keys as canonicalConfig() has them, and
  // every other key as written, since a key the table does not search can still change what is built.
  [[nodiscard]] GateVerdict operator()(const FFTConfig& fft, const Interval& span, const UseConfig& opts) const;

  // The latest reading of `opts` at `exponent`, or with `above` at or above it in its regime.
  [[nodiscard]] std::optional<RoeRow> reading(const FFTConfig& fft, u64 exponent, const UseConfig& opts,
                                              bool above) const;

private:
  Env device_;

  // By spec, then by the text of the gate's identity for the set the kernels were built with.
  std::map<std::pair<std::string, std::string>, std::vector<RoeRow>> rows_;
};

}  // namespace tune
