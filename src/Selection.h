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

struct SelectionFile {
  std::string provenance;

  std::vector<std::pair<std::string, std::string>> global;
  std::vector<UseLine> family;
  std::vector<SelectionEntry> entries;
  std::vector<std::string> unknown;

  // The layers this file contributes to production's option precedence, weakest first.
  [[nodiscard]] SelectionLayers layersFor(const SelectionEntry& entry) const;
};

[[nodiscard]] std::string entryId(const std::string& fft, TestKind kind, Regime regime, const UseConfig& opts);

// Fills in `id` and `regime`, canonicalizes each spec, and sorts by ascending cost. Logs and returns false on failure.
[[nodiscard]] bool finalize(SelectionFile& file);

[[nodiscard]] std::optional<SelectionFile> parseSelection(std::string_view text, std::string_view name);
[[nodiscard]] std::optional<SelectionFile> readSelection(const fs::path& path);

[[nodiscard]] std::string text(const SelectionFile& file);

void writeSelection(const fs::path& path, const SelectionFile& file);

}  // namespace tune
