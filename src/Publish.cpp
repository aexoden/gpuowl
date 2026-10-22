// Copyright (C) Jason Lynch

#include "Publish.h"

#include "CycleFile.h"

#include <algorithm>
#include <cinttypes>
#include <map>

namespace tune {

namespace {

// Each FFT's candidate carrying the most weight: the regime its line is chosen in.
std::map<std::string, const Candidate*> mainCandidates(const std::vector<Candidate>& candidates) {
  std::map<std::string, const Candidate*> out;
  for (const Candidate& c : candidates) {
    auto [it, fresh] = out.try_emplace(c.subject.fft.spec(), &c);
    if (!fresh && c.weight() > it->second->weight()) { it->second = &c; }
  }
  return out;
}

}  // namespace

bool Line::holds(u64 E) const { return E >= minExp(candidate->subject.fft) && E <= reach(); }

const Line* KindPublication::at(u64 E) const {
  for (const Line& line : lines) {
    if (line.holds(E)) { return &line; }
  }
  return nullptr;
}

const KindPublication* Publication::of(TestKind kind) const {
  auto const it = std::ranges::find(kinds, kind, &KindPublication::kind);
  return it == kinds.end() ? nullptr : &*it;
}

double Publication::T() const {
  double sum = 0;
  for (const KindPublication& k : kinds) { sum += k.T; }
  return sum;
}

double Publication::coverage() const {
  if (kinds.empty()) { return 0; }
  double sum = 0;
  for (const KindPublication& k : kinds) { sum += k.covered; }
  return sum / double(kinds.size());
}

Publication publish(const RunScope& scope, const std::vector<std::vector<Candidate>>& candidates,
                    const Readings& readings) {
  Publication out;

  for (size_t ki = 0; ki < scope.grids.size() && ki < candidates.size(); ++ki) {
    const Grid& grid = scope.grids[ki];
    KindPublication pub{.kind = grid.kind, .lines = {}};
    std::vector<Line> lines;
    std::vector<std::pair<Line, double>> owed;            // a line whose gate is owed, and its cost
    std::vector<std::pair<ConcludeItem, double>> short_;  // a reading short of its calls, and its mean

    for (auto const& [spec, c] : mainCandidates(candidates[ki])) {
      std::vector<const Reading*> rs = readings.of(c->subject);
      std::erase_if(rs, [](const Reading* r) { return r->failed || r->calls == 0; });
      std::ranges::sort(rs, {}, &Reading::pessimistic);

      u64 const floor = c->points.front().exponent;
      bool gateOwed = false;
      for (const Reading* r : rs) {
        if (r->calls < MIN_CALLS) { continue; }
        Verdict const v = verdictOf(c->subject.fft, readings.roes(c->subject.fft, r->options), floor);
        if (v.state == Accuracy::Rejected) { continue; }
        if (v.state == Accuracy::Owed) {
          if (!gateOwed) { owed.emplace_back(Line{.candidate = c, .reading = r, .accuracy = v}, r->pessimistic()); }
          gateOwed = true;
          continue;
        }
        lines.push_back({.candidate = c, .reading = r, .accuracy = v});
        break;
      }

      // The cheapest reading by its mean, where it is short of its calls: a second call may make it a line.
      const Reading* best = nullptr;
      for (const Reading* r : rs) {
        if (!best || r->mean < best->mean) { best = r; }
      }
      if (best && best->calls < MIN_CALLS) {
        short_.emplace_back(ConcludeItem{.candidate = c, .options = best->options}, best->mean);
      }
    }

    // The frontier as production reads it (TuneEntry::update): cheapest first, a line dropped where a cheaper one
    // reaches at least as far.
    std::ranges::sort(lines, {}, &Line::cost);
    for (const Line& line : lines) {
      if (pub.lines.empty() || line.reach() > pub.lines.back().reach()) { pub.lines.push_back(line); }
    }

    for (const GridPoint& p : grid.points) {
      if (const Line* line = pub.at(p.exponent)) {
        pub.T += p.weight * line->cost();
        pub.covered += p.weight;
      }
    }

    // A gate reading or a call is owed only where the line it would make would be production's somewhere.
    auto wouldServe = [&](const Candidate* c, double cost) {
      double weight = 0;
      for (const GridPoint& p : c->points) {
        const Line* const now = pub.at(p.exponent);
        if (!now || cost < now->cost()) { weight += p.weight; }
      }
      return weight;
    };
    for (auto const& [line, cost] : owed) {
      if (double const w = wouldServe(line.candidate, cost); w > 0) {
        out.owedGates.push_back({.fft = line.candidate->subject.fft,
                                 .options = line.reading->options,
                                 .exponent = line.accuracy.next,
                                 .weight = w});
      }
    }
    for (auto const& [item, mean] : short_) {
      if (wouldServe(item.candidate, mean) > 0) { out.owedCalls.push_back(item); }
    }
    out.kinds.push_back(std::move(pub));
  }

  std::ranges::sort(out.owedGates, std::greater{}, &GateItem::weight);
  std::ranges::stable_sort(out.owedCalls, std::greater{}, [](const ConcludeItem& i) { return i.candidate->weight(); });
  return out;
}

const char* tuneFileName(TestKind kind) { return kind == TestKind::LL ? "tune-ll.txt" : "tune.txt"; }

std::string tuneFileText(const KindPublication& kind, const std::string& header) {
  std::string out;
  for (size_t at = 0; at < header.size();) {
    size_t const eol = std::min(header.find('\n', at), header.size());
    out += "# " + header.substr(at, eol - at) + '\n';
    at = eol + 1;
  }
  for (const Line& line : kind.lines) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%6.1f %14s # %" PRIu64, line.cost(), line.candidate->subject.fft.spec().c_str(),
             line.reach());
    out += buf;
    for (const auto& [key, value] : line.reading->options) { out += ", " + key + '=' + value; }
    out += '\n';
  }
  return out;
}

void writeTuneFile(const fs::path& dir, const KindPublication& kind, const std::string& header) {
  CycleFile file{dir / tuneFileName(kind.kind)};
  file->write(tuneFileText(kind, header));
}

}  // namespace tune
