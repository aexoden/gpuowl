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

#include "Selection.h"
#include "TuneDB.h"
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

// Every entry `env`'s own measurements support, cheapest first.  Rows of another env -- another card, or the same card
// under other kernels -- are invisible here, as they are to every other comparison.
//
// `defaults` is what the entries will be published beside: an entry outranks those lines, so one whose recorded option
// set does not name a key they set would run differently from the way it was measured, and is not published.
[[nodiscard]] std::vector<SelectionEntry> entriesFor(const TuneDB& db, u32 env, const Defaults& defaults = {});

// Whether `defaults` would change what a row measured under `opts` builds on `fft`, so that emission would not publish
// it beside them.
[[nodiscard]] bool shadowedBy(const Defaults& defaults, const Env& env, const FFTConfig& fft, TestKind kind,
                              const UseConfig& opts);

// The file as it would be published.  Empty of entries where the database holds none for the env, which is a fact
// about the database and is published as such.
[[nodiscard]] std::optional<SelectionFile> emit(const TuneDB& db, const Defaults& defaults, const Provenance& from);

// Writes it through a sibling temporary and a rename, so a reader sees either the whole file or the old one.
[[nodiscard]] bool publish(const fs::path& path, const TuneDB& db, const Defaults& defaults, const Provenance& from);

}  // namespace tune
