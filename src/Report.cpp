// Copyright (C) Jason Lynch

#include "Report.h"

#include "CycleFile.h"
#include "TuneDB.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>

namespace tune {

namespace {

std::string jsonString(std::string_view text) {
  std::string out = "\"";
  for (char const c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  return out + '"';
}

// A finite number, or null.
std::string number(double x) {
  if (!std::isfinite(x)) { return "null"; }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", x);
  return buf;
}

}  // namespace

std::string reportJson(const RunReport& r) {
  std::string out = "{\n";
  out += "  \"version\": " + jsonString(r.version) + ",\n";
  out += "  \"settings\": " + jsonString(r.settings) + ",\n";
  out += "  \"state\": " + jsonString(r.state) + ",\n";
  out += "  \"elapsedSeconds\": " + number(r.elapsed) + ",\n";
  out += "  \"budgetSeconds\": " + (r.budget > 0 ? number(r.budget) : std::string{"null"}) + ",\n";
  out += "  \"T\": " + number(r.T) + ",\n";
  out += "  \"modelGapEstimate\": " + number(r.regret) + ",\n";

  out += "  \"laneSeconds\": {";
  for (size_t i = 0; i < LANES; ++i) {
    out += std::string{i ? ", " : ""} + jsonString(toString(Lane(i))) + ": " + number(r.lanes.seconds[i]);
  }
  out += ", \"duty\": " + number(r.lanes.duty) + "},\n";

  if (r.coverage) {
    const CoverageSummary& c = *r.coverage;
    out += "  \"coverage\": {\"entries\": " + std::to_string(c.entries) +
      ", \"entriesRead\": " + std::to_string(c.entriesRead) +
      ", \"branchesVisited\": " + std::to_string(c.branchesVisited) +
      ", \"branchesLegal\": " + std::to_string(c.branches.legal) +
      ", \"branchesLegalIsExact\": " + (c.branches.complete ? "true" : "false") +
      ", \"positionsCovered\": " + std::to_string(c.positions.covered) +
      ", \"positionsTotal\": " + std::to_string(c.positions.total) + "},\n";
  }

  out += "  \"published\": [";
  bool first = true;
  if (r.publication) {
    for (const KindPublication& kind : r.publication->kinds) {
      for (const Line& line : kind.lines) {
        char reach[32];
        std::snprintf(reach, sizeof(reach), "%" PRIu64, line.reach());
        out += std::string{first ? "\n" : ",\n"} + "    {\"kind\": " + jsonString(toString(kind.kind)) +
          ", \"fft\": " + jsonString(line.candidate->subject.fft.spec()) +
          ", \"options\": " + jsonString(configText(line.reading->options)) + ", \"usPerIt\": " + number(line.cost()) +
          ", \"maxExponent\": " + reach + "}";
        first = false;
      }
    }
  }
  out += first ? "],\n" : "\n  ],\n";

  if (r.audit && r.audit->declared) {
    const AuditSummary& a = *r.audit;
    out += "  \"audit\": {\"state\": " + jsonString(toString(a.state)) +
      ", \"trialsDeclared\": " + std::to_string(a.trials) + ", \"trialsDone\": " + std::to_string(a.done) +
      ", \"startsUnread\": " + std::to_string(a.failedStarts) + ", \"baselineT\": " + number(a.baseline) +
      ", \"target\": " + number(a.target) + ", \"gain\": " + number(a.gain) +
      ", \"discoveryBound95\": " + (a.state == AuditSummary::State::Complete ? number(a.bound) : std::string{"null"}) +
      ", \"found\": [";
    for (size_t i = 0; i < a.found.size(); ++i) {
      out += std::string{i ? ", " : ""} + "{\"fft\": " + jsonString(a.found[i]->candidate->subject.fft.spec()) +
        ", \"options\": " + jsonString(configText(a.found[i]->reading->options)) + "}";
    }
    out += "]}\n";
  } else {
    out += "  \"audit\": null\n";
  }
  return out + "}\n";
}

void writeReport(const std::filesystem::path& dir, const std::string& json) {
  if (dir.empty()) { return; }
  CycleFile file{dir / REPORT_FILE};
  file->write(json);
}

}  // namespace tune
