// Copyright (C) Jason Lynch

// Which -use keys change the rounding, measured: `-tune accuracy` reads every value of every key against the option set
// it moved from, on one FFT of each type at one exponent.  Kernels that do the same arithmetic read the same rounding
// errors exactly, so a key is shown not to spend accuracy only where every value it offers reads bit-identical to the
// set it moved from; any difference at all makes it a key the gate has to hold to its reference (Gate.h).  The option
// table's accuracy flags are what the gate goes by, and this is how they are kept true as the kernels change.
//
// Ported from the tune branch's Accuracy.{h,cpp}.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "TuneDB.h"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

class GpuCommon;

namespace tune {

enum class RunOutcome : u8;
struct TuneCommand;

// The key a move of a coupled access class is read under: it moves a digit of LOADS and of STORES together.
inline constexpr const char* COUPLED = "LOADS+STORES";

// One move of one axis within a group, from a background set.
struct Move {
  UseConfig config;  // canonical
  std::string key;   // the key moved, or empty for a coupled access class
  std::string text;  // "MM2_CHAIN=1", "LOADS[carry]=4,STORES[carry]=2"
};

// Every single move of an axis of `group` from `background` on `fft`, each to a set the background is not.
[[nodiscard]] std::vector<Move> movesWithin(const Env& env, const FFTConfig& fft, const UseConfig& background,
                                            Group group);

// One reading the sweep takes.
struct SweepPoint {
  FFTConfig fft;
  u64 exponent = 0;
  UseConfig config;  // canonical

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

// The latest reading `db` holds in `env` of each (FFT, exponent, canonical set) that has a fingerprint: what the sweep
// has read so far, so that one stopped part way resumes where it was.
[[nodiscard]] SweepLookup sweepLookup(const TuneDB& db, u32 env, const Env& device);

// Where `fft` is read: the top of its range, where it has the least accuracy to spare and where the gate reads it.
// 0 where it can run nothing.
[[nodiscard]] u64 sweepExponent(const FFTConfig& fft);

// Pure.  What the sweep reads on one family, given what it has read so far, all at one exponent -- the lowest
// sweepExponent() of the shape's runnable variants, so that the variants compare: each variant at its defaults, the
// family's own first, and then every move within `groups` from a set that has read without failing its check -- the
// defaults, and every structural move, whose dependents are only offered from it.  A move is planned once per variant,
// at the first set it is offered from, and no later variant offers it again once it has been read -- a structural one
// once it has read with a passing check -- so a key whose values depend on the variant is read at the variant that
// offers them.  Call it again after every reading, until it plans nothing unread.
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

// Whether the option table has a move ("TAIL_KERNELS=0") changing the rounding; a COUPLED move it never marks.
[[nodiscard]] bool tableChanges(const std::string& key, const std::string& move);

// `-tune accuracy`: sweeps the FFT the command names, or else one FFT of each type that rounds -- the cheapest read at
// the probe, else the cheapest by its prior -- recording every reading as the gate's own, and reports what each key
// did against what the option table says of it.
[[nodiscard]] RunOutcome runAccuracy(GpuCommon shared, const TuneCommand& command);

}  // namespace tune
