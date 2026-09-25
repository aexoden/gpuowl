// Copyright (C) Jason Lynch
// Which -use keys change the rounding, measured: `-tune accuracy` reads every value of every key against the option set
// it moved from, on each family's bootstrap configuration at one exponent.  Kernels that do the same arithmetic read
// the same rounding errors exactly, so a key is shown not to spend accuracy only where every value it offers reads
// bit-identical to the set it moved from; any difference at all makes it a key the gate has to hold to its reference.
#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "UseResolve.h"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

class GpuCommon;

namespace tune {

enum class MeasureOutcome : u8;
struct TuneCommand;

// The key a move of a coupled access class is read under: it moves a digit of LOADS and of STORES together.
inline constexpr const char* COUPLED = "LOADS+STORES";

// One reading the sweep takes.
struct SweepPoint {
  FFTConfig fft;
  u64 exponent = 0;

  // Canonical.
  UseConfig config;

  // The key moved, or COUPLED, and the move ("MM2_CHAIN=1"); both empty for a variant's defaults.
  std::string key;
  std::string text;

  // The set this one is compared with, at the same FFT and exponent; none for a variant's defaults.
  std::optional<UseConfig> background;
};

// A reading as the database holds it.  z and maxRoe are rounded there and only summarise the error samples, so two
// readings are told apart by the samples' fingerprint alone.
struct SweepReading {
  double z = 0;
  u32 n = 0;
  double maxRoe = 0;
  bool checkOk = true;
  u64 fingerprint = 0;
};

using SweepLookup = std::function<std::optional<SweepReading>(const FFTConfig&, u64 exponent, const UseConfig&)>;

// Where `fft` is read: the top of its range, where it has the least accuracy to spare and where the gate reads it.
// 0 where it can run nothing.
[[nodiscard]] u64 sweepExponent(const FFTConfig& fft);

// Pure.  What the sweep reads on one family, given what it has read so far, all at one exponent -- the lowest
// sweepExponent() of the shape's runnable variants, so that the variants compare: each variant at its defaults, the
// family's own first, and then every move within `groups` from a set that has read without
// failing its check -- the defaults, and every structural move, whose dependents are only offered from it.  A move
// ("MM2_CHAIN=1") is planned once per variant, at the first set it is offered from, and no later variant offers it
// again once it has been read -- a structural one once it has read with a passing check -- so a key whose values
// depend on the variant (MM_CHAIN offers only its default at M=1) is read at the variant that offers them, and a move
// that would not build, or a structural move or set whose reading fails, is offered again by the next.  Call it again
// after every reading, until it plans nothing unread.
[[nodiscard]] std::vector<SweepPoint> sweepPlan(const Env& env, const FFTConfig& family,
                                                const std::vector<Group>& groups, const SweepLookup& readingOf);

enum class Change : u8 {
  Same,     // the same error samples as its background, bit for bit: the same arithmetic
  Differs,  // a different rounding
  Fails,    // its check failed where its background's passed
  Unread,   // either reading is missing or has no fingerprint, or the background's own check failed
};

[[nodiscard]] const char* toString(Change change);

[[nodiscard]] Change compare(const std::optional<SweepReading>& point, const std::optional<SweepReading>& background);

// Everything the readings say about one key, over every family it was read on.
struct KeyVerdict {
  std::string key;
  u32 same = 0;
  u32 differs = 0;
  u32 fails = 0;
  u32 unread = 0;

  // The furthest a value read below its background, and which move on which FFT that was.
  double worstDz = 0;
  std::string worstAt;

  // By move ("TAIL_KERNELS=0"), every one read against a background that passed: whether any reading of it changed
  // the rounding.
  std::map<std::string, bool> changedBy;

  // Yes where any value changed the rounding, None where every value read was identical, nothing where none was read.
  [[nodiscard]] std::optional<AccuracyImpact> measured() const;
};

// Pure.  One verdict per key the points move, in key order.
[[nodiscard]] std::vector<KeyVerdict> keyVerdicts(const std::vector<SweepPoint>& points, const SweepLookup& readingOf);

// `-tune accuracy`: sweeps each family's bootstrap FFT at the scope's probe, or the one FFT the command names,
// recording every reading as the gate's own, and reports what each key did against what the option table says of it.
[[nodiscard]] MeasureOutcome runAccuracy(const GpuCommon& shared, const TuneCommand& command);

}  // namespace tune
