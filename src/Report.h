// Copyright (C) Jason Lynch

// tune-report.json: where a run stands, for a script or another program to read without parsing the log.  Rewritten
// with each regret estimate and when the run ends.

#pragma once

#include "Audit.h"
#include "Contenders.h"
#include "Coverage.h"
#include "Duel.h"
#include "Explore.h"
#include "Feedback.h"
#include "Lanes.h"
#include "Publish.h"

#include <filesystem>
#include <optional>
#include <string>

namespace tune {

inline constexpr const char* REPORT_FILE = "tune-report.json";

struct RunReport {
  std::string version{};
  std::string settings{};
  std::string state{};  // "running", "budget spent", "stopped", "converged", "device lost"
  double elapsed = 0;   // the campaign's tuning seconds
  double budget = 0;    // 0 for none
  double T = 0;
  double regret = 0;  // the model's search-gap estimate: no evidence of coverage
  double calibration = 1;  // Readings::calibration()
  LaneTime lanes{};
  std::optional<CoverageSummary> coverage{};
  std::optional<Outstanding> outstanding{};  // as of the last regret estimate
  const Publication* publication = nullptr;
  std::optional<AuditSummary> audit{};
  std::optional<DuelTally> duels{};
  const std::map<ConfigKey, Sustained>* production = nullptr;  // sustainedOf() the feedback beside the tune file
  const std::vector<KindContest>* contenders = nullptr;          // of `publication`
};

// What production sustained running a published line's configuration; nullptr where it has not run it.
[[nodiscard]] const Sustained* sustainedOn(const Line& line, const std::map<ConfigKey, Sustained>& production);

// Over the lines of `publication` production has run: how many, and the median of what it sustained on each over the
// cost the line was published at.  NaN for none.
struct ProductionGap {
  u32 lines = 0;
  double ratio = NAN;
};

[[nodiscard]] ProductionGap productionGap(const Publication& publication,
                                          const std::map<ConfigKey, Sustained>& production);

[[nodiscard]] std::string reportJson(const RunReport& report);

// Writes `json` to REPORT_FILE in `dir`, replacing it whole; nothing where `dir` is empty.
void writeReport(const std::filesystem::path& dir, const std::string& json);

}  // namespace tune
