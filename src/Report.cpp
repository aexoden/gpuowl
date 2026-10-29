// Copyright (C) Jason Lynch

#include "Report.h"

#include "CycleFile.h"
#include "TuneDB.h"

#include <algorithm>
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

std::string productionJson(const Line& line, const std::map<ConfigKey, Sustained>* production) {
  const Sustained* const s = production ? sustainedOn(line, *production) : nullptr;
  if (!s) { return "null"; }
  return "{\"usPerIt\": " + number(s->usPerIt) + ", \"ratio\": " + number(s->usPerIt / line.cost()) +
    ", \"stretches\": " + std::to_string(s->stretches) + ", \"iterations\": " + std::to_string(s->iters) + "}";
}

}  // namespace

const Sustained* sustainedOn(const Line& line, const std::map<ConfigKey, Sustained>& production) {
  auto const it = production.find(Readings::keyOf(line.subject, line.reading->options));
  return it == production.end() ? nullptr : &it->second;
}

ProductionGap productionGap(const Publication& publication, const std::map<ConfigKey, Sustained>& production) {
  std::vector<double> ratios;
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.written) {
      if (const Sustained* const s = sustainedOn(line, production)) { ratios.push_back(s->usPerIt / line.cost()); }
    }
  }
  if (ratios.empty()) { return {}; }
  std::ranges::sort(ratios);
  size_t const n = ratios.size();
  return {.lines = u32(n), .ratio = n % 2 ? ratios[n / 2] : (ratios[n / 2 - 1] + ratios[n / 2]) / 2};
}

std::string reportJson(const RunReport& r) {
  std::string out = "{\n";
  out += "  \"version\": " + jsonString(r.version) + ",\n";
  out += "  \"settings\": " + jsonString(r.settings) + ",\n";
  out += "  \"state\": " + jsonString(r.state) + ",\n";
  out += "  \"elapsedSeconds\": " + number(r.elapsed) + ",\n";
  out += "  \"budgetSeconds\": " + (r.budget > 0 ? number(r.budget) : std::string{"null"}) + ",\n";
  out += "  \"T\": " + number(r.T) + ",\n";
  out += "  \"calibration\": " + number(r.calibration) + ",\n";
  out += "  \"modelGapEstimate\": " + number(r.regret) + ",\n";
  if (r.production && r.publication) {
    ProductionGap const gap = productionGap(*r.publication, *r.production);
    out += "  \"production\": {\"lines\": " + std::to_string(gap.lines) + ", \"ratio\": " + number(gap.ratio) + "},\n";
  } else {
    out += "  \"production\": null,\n";
  }

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

  if (r.outstanding) {
    const Outstanding& o = *r.outstanding;
    out += "  \"outstanding\": {\"entriesUnread\": " + std::to_string(o.unread) +
      ", \"entriesUnreadWithinReach\": " + std::to_string(o.unreadWithinReach) +
      ", \"contenders\": " + std::to_string(o.contenders) +
      ", \"contenderBranches\": " + std::to_string(o.contenderBranches) +
      ", \"leastDevelopedBranchSets\": " + std::to_string(o.leastDeveloped) + "},\n";
  }

  out += "  \"published\": [";
  bool first = true;
  if (r.publication) {
    for (const KindPublication& kind : r.publication->kinds) {
      for (const Line& line : kind.written) {
        char range[64];
        std::snprintf(range, sizeof(range), "\"minExponent\": %" PRIu64 ", \"maxExponent\": %" PRIu64, line.lo(),
                      line.reach());
        out += std::string{first ? "\n" : ",\n"} + "    {\"kind\": " + jsonString(toString(kind.kind)) +
          ", \"fft\": " + jsonString(line.subject.fft.spec()) +
          ", \"regime\": " + jsonString(line.subject.regime.label()) +
          ", \"options\": " + jsonString(configText(line.reading->options)) + ", \"usPerIt\": " + number(line.cost()) +
          ", " + range + ", \"earlierBuild\": " + (line.carried ? "true" : "false") +
          ", \"soaked\": " + (line.reading->isSoaked() ? "true" : "false") + ", \"timedUsPerIt\": " +
          number(line.reading->timed) + ", \"production\": " + productionJson(line, r.production) + "}";
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
      out += std::string{i ? ", " : ""} + "{\"fft\": " + jsonString(a.found[i]->subject.fft.spec()) +
        ", \"options\": " + jsonString(configText(a.found[i]->reading->options)) + "}";
    }
    out += "], \"confirmed\": [";
    bool firstConfirmed = true;
    for (const AuditSummary::Confirmed& c : a.confirmed) {
      if (c.fresh.calls == 0) { continue; }
      out += std::string{firstConfirmed ? "" : ", "} +
        "{\"fft\": " + jsonString(c.line->subject.fft.spec()) +
        ", \"options\": " + jsonString(configText(c.line->reading->options)) +
        ", \"pooledUsPerIt\": " + number(c.line->reading->mean) + ", \"freshUsPerIt\": " + number(c.fresh.mean) +
        ", \"freshCalls\": " + std::to_string(c.fresh.calls) + "}";
      firstConfirmed = false;
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
