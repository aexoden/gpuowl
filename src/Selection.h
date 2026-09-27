// Copyright (C) Jason Lynch

// Selection files store tuning results used to choose an FFT in production.
//
// Costs and option sets share one file so an atomic replacement keeps each validated reach paired with the options it
// was measured with. An entry stores the complete effective options for every key the tuner varied, defaults included.
//
// Invalid files are rejected as a whole to avoid using unvalidated configurations or reach limits. Unknown record kinds
// and comments are preserved for compatibility.

#pragma once

#include "common.h"
#include "Eligibility.h"
#include "OptionSpace.h"
#include "TuneDB.h"
#include "UseResolve.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

struct SelectionEntry {
  // A stable hash of (spec, kind, regime, opts), printed in hex.
  std::string id;

  double cost = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;

  u64 emin = 0;
  u64 reach = 0;
  Regime regime{};

  Evidence evidence = Evidence::Unvalidated;
  UseConfig opts;
};

// A configuration measured computing a wrong answer somewhere in a regime.  It has no entry, and production must not
// arrive at it by another path either: the shape scan, tune.txt, or an entry the user's own settings resolve into it.
struct Exclusion {
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};

  // The complete option set the kernels were built with, as the measurement recorded it.
  UseConfig opts;

  [[nodiscard]] bool operator==(const Exclusion&) const = default;
};

// How far every option set of an FFT that rounds as `rounding` does was measured to hold in one regime, where no entry
// the file publishes says so already: production holds whatever it runs under that arithmetic to `reach`, whichever
// path it arrives by, but never walks to a limit as it does to an entry.
struct Limit {
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  u64 reach = 0;

  // Only the keys held at a value that changes the rounding, as roundingOf() gives them; empty for the defaults'.
  UseConfig rounding;

  [[nodiscard]] bool operator==(const Limit&) const = default;
};

struct SelectionFile {
  std::string provenance;

  std::vector<std::pair<std::string, std::string>> global;
  std::vector<UseLine> family;
  std::vector<SelectionEntry> entries;
  std::vector<Limit> limits;
  std::vector<Exclusion> excluded;
  std::vector<std::string> unknown;

  // The layers this file contributes to production's option precedence, weakest first.
  [[nodiscard]] SelectionLayers layersFor(const SelectionEntry& entry) const;
};

[[nodiscard]] std::string entryId(const std::string& fft, TestKind kind, Regime regime, const UseConfig& opts);

// Fills in `id` and `regime`, canonicalizes each spec, and sorts entries by ascending cost, and limits and exclusions
// by what they name, dropping repeats. Logs and returns false on failure.
[[nodiscard]] bool finalize(SelectionFile& file);

[[nodiscard]] std::optional<SelectionFile> parseSelection(std::string_view text, std::string_view name);
[[nodiscard]] std::optional<SelectionFile> readSelection(const fs::path& path);

[[nodiscard]] std::string text(const SelectionFile& file);

void writeSelection(const fs::path& path, const SelectionFile& file);

}  // namespace tune
