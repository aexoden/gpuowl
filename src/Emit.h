// Copyright (C) Jason Lynch

// Builds the selection file from the measurement database: a pure function of TuneDB plus one env, with no device
// work and nothing to interrupt.
//
// Every published entry is a transcript of one measured row -- its cost, its interval, and the complete option set the
// row was built under, copied rather than inferred by comparing rows.  So an entry says what was run and how fast it
// was, and a configuration nobody measured cannot reach the file by being assembled out of parts of several that were.
//
// The grammar itself lives in Selection.{h,cpp}, which production also reads.

#pragma once

#include "Gate.h"
#include "Selection.h"
#include "TuneDB.h"
#include "TuneEntry.h"
#include "UseResolve.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tune {

// What a configuration with no entry of its own runs at: the options every family agreed on, and one line per family
// that disagreed.
struct Defaults {
  UseConfig global;
  std::vector<UseLine> family;
};

// The provenance comment, which says which database this file was derived from and what it was derived for.  Kept
// across a read so a rewrite cannot silently reattribute the file.
struct Provenance {
  u64 ts = 0;
  std::string db;
  u32 env = 0;

  // The objective and the workload it was taken over; omitted from the line when no workload is given, which is what
  // a database-only emission has.
  double T = 0;
  u64 workloadLo = 0;
  u64 workloadHi = 0;
};

[[nodiscard]] std::string provenanceOf(const Provenance& from);

// Which option sets count: nothing is published that the accuracy gate has not passed, but the search works from --
// and values its items against -- every set the gate has not rejected, since a reading it still owes is taken next.
enum class Gating : u8 { Required, Assumed };

// Every entry `env`'s own measurements support, cheapest first.  Rows of another env -- another card, or the same card
// under other kernels -- are invisible here, as they are to every other comparison.
//
// `defaults` is what the entries will be published beside: an entry outranks those lines, so one whose recorded option
// set does not name a key they set would run differently from the way it was measured, and is not published.
[[nodiscard]] std::vector<SelectionEntry> entriesFor(const TuneDB& db, u32 env, const Defaults& defaults = {},
                                                     Gating gating = Gating::Required);

// A publishable option set, and the row it would be a transcript of.
struct OptionSet {
  SelectionEntry entry;
  Measurement m;

  // Where the row was taken, which is where another call pools with it.
  u64 exponent = 0;

  // What the accuracy gate has made of the set so far, over the interval the fitted table gives it -- which is the
  // entry's own until the gate derives a reach below it or raises one above it.
  GateVerdict gate{};
};

// Every option set of every identity that could be published beside `defaults`, one per option set, before any is
// dropped for being dominated: the one a slightly cheaper set of the same identity keeps out is still what production
// would run if that reading were the unlucky one.  A set the accuracy gate rejected is not one of them, whatever it
// costs; one it still owes a reading is.
[[nodiscard]] std::vector<OptionSet> optionSetsFor(const TuneDB& db, u32 env, const Defaults& defaults = {});

// The option sets optionSetsFor() leaves out because the accuracy gate rejected them.
[[nodiscard]] std::vector<OptionSet> rejectedSets(const TuneDB& db, u32 env, const Defaults& defaults = {});

// What entriesFor() chooses the table from: every option set of every identity that no other of the same identity
// dominates, including those that another identity's entry would keep out of the table.
[[nodiscard]] std::vector<SelectionEntry> candidatesFor(const TuneDB& db, u32 env, const Defaults& defaults = {},
                                                        Gating gating = Gating::Required);

// The option sets the table would publish if every reading the gate owes passed, that are waiting on one: the readings
// that stand between what the search has found and what production runs.
[[nodiscard]] std::vector<OptionSet> gatesOwed(const TuneDB& db, u32 env, const Defaults& defaults = {});

// Whether `defaults` would change what a row measured under `opts` builds on `fft`, so that emission would not publish
// it beside them.
[[nodiscard]] bool shadowedBy(const Defaults& defaults, const Env& env, const FFTConfig& fft, TestKind kind,
                              const UseConfig& opts);

// The file as it would be published, with an exclusion for every configuration the env measured computing a wrong
// answer.  Empty of entries where the database holds none for the env, which is a fact about the database and is
// published as such.
[[nodiscard]] std::optional<SelectionFile> emit(const TuneDB& db, const Defaults& defaults, const Provenance& from);

// Upstream's tune.txt for a binary that reads no selection file: one line per FFT the file publishes at the fitted
// table's own reach, at default rounding, or with exact arithmetic, and no other -- nor any FFT whose default rounding
// was held short of the end of a band in any regime, since a line says nothing of regimes, nor any whose defaults are
// excluded in any regime. Such a binary runs an FFT up to the table's reach under options of its own, so a line is
// safe only where the table's reach holds at default rounding; its cost is the cheapest entry of that FFT, and the
// lines form the cost/reach frontier its reader keeps.
[[nodiscard]] std::vector<TuneEntry> compatibilityView(const SelectionFile& file, const Env& env);

// The view in upstream's own format.
[[nodiscard]] std::string compatibilityText(const std::vector<TuneEntry>& view);

// Writes the view of `file` to `path` through a sibling temporary and a rename, returning how many lines it holds.
size_t writeCompatibility(const fs::path& path, const SelectionFile& file, const Env& env);

// Writes it through a sibling temporary and a rename, so a reader sees either the whole file or the old one; and the
// compatibility view beside it to `compat`, the same way, where one is asked for.
[[nodiscard]] bool publish(const fs::path& path, const TuneDB& db, const Defaults& defaults, const Provenance& from,
                           const std::optional<fs::path>& compat = {});

}  // namespace tune
