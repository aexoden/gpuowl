// Copyright (C) Jason Lynch

// Which configuration a production run uses for one exponent: the published entry that covers it, the complete option
// set that entry was measured under, and what happens when the user's own settings override part of it.
//
// The user's files outrank the published entry, so production can resolve options nothing was ever measured with. A
// wrong cost is a slow run; a wrong reach is a wrong residue. So an entry production would not reproduce keeps only
// the reach every configuration of its shape inherits, and says which keys took the rest away.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Selection.h"
#include "UseResolve.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class Args;

namespace tune {

// The FFT a task runs and the complete option set it runs under, ready to hand to Gpu.
struct Choice {
  FFTConfig fft;
  UseConfig options;

  // The entry this came from, absent where no published entry covers the exponent and the shape scan answered instead.
  std::optional<SelectionEntry> entry;

  // The keys production resolves differently from the way the entry was measured; empty unless there is an entry.
  std::vector<std::string> shadowed;

  // The largest exponent this choice is validated for, before -fftOverdrive is applied to it.
  u64 reach = 0;
};

// The FFT an -fft argument names, resolved as production's own shape scan resolves it -- so a bare size, a shape with
// no variant, and a full spec all name here what they would name there. Nothing where there is no -fft argument, or
// where it is not a spec at all, which the shape scan refuses in its own words.
[[nodiscard]] std::optional<FFTConfig> pinnedFft(const Args& args);

// How this run would actually run `fft` at `E`, which is not always how the exponent alone says it would: -carry long
// forces the long-carry kernels at any bits per word.
[[nodiscard]] Regime runRegime(const Args& args, const FFTConfig& fft, u64 E);

// The reach published for exactly this configuration -- same FFT, kind, regime and option set -- or the FFT's own
// inherited limit where nothing publishes one. A measurement may reach further than the fitted table or fall short of
// it, and a configuration that fell short is one production must not run above, by whichever path it arrives at it.
//
// Keyed by the regime the exponent alone gives, since that is how entries are keyed; a -carry long run that lands on
// the same options is held to the same limit, which is the conservative reading in both directions.
[[nodiscard]] u64 publishedReach(const SelectionFile& file, const FFTConfig& fft, TestKind kind,
                                 const UseConfig& options, u64 E);

// The keys whose resolved value is not the one `entry` was measured under, in key order. A key that changes nothing
// here -- one the option table says does not apply to this FFT under these settings -- is not one of them; a key the
// table does not know is, since nothing here can call it inert.
[[nodiscard]] std::vector<std::string> shadowedKeys(const Args& args, const Env& env, const SelectionFile& file,
                                                    const SelectionEntry& entry, const FFTConfig& fft);

// The cheapest published entry that covers `E` once shadowing has been accounted for, or nothing where none does.
// Entries are compared by cost rather than walked in file order, so a hand-edited file that is out of order still
// answers with its cheapest eligible entry rather than with whichever one was typed first.
[[nodiscard]] std::optional<Choice> chooseFrom(const SelectionFile& file, const Args& args, const Env& env, u64 E,
                                               TestKind kind);

// Where the selection file is looked for: beside the work, else in the pool directory, as tune.txt is.
[[nodiscard]] fs::path selectionPath(const Args& args);

// What to run, warning once about anything a published entry lost to the user's own settings. Falls back to the shape
// scan where there is no file, no entry for the exponent, or an -fft spec nothing was published for -- and asks it
// again for something larger where its answer is a configuration published as reaching less far than this exponent.
[[nodiscard]] Choice choose(const Args& args, const Env& env, u64 E, TestKind kind);

// Says once, at startup, which config-file keys would shadow what the selection file publishes. Silent where the file
// is absent or nothing in it is shadowed, so an untuned or cleanly tuned machine gains no noise.
void reportShadowing(const Args& args, const Env& env);

}  // namespace tune
