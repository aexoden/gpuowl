// Copyright (C) Jason Lynch

#include "Publish.h"

#include "CycleFile.h"

#include <algorithm>
#include <cinttypes>
#include <iterator>
#include <map>
#include <optional>
#include <set>

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

// Each FFT read for `kind` that the run does not hold, in the regime it has the most calls in: where an earlier run
// spent its time on it.
std::map<std::string, Subject> unheldSubjects(const Readings& readings, TestKind kind,
                                              const std::map<std::string, const Candidate*>& held) {
  std::map<std::pair<std::string, std::string>, std::pair<Subject, u32>> calls;
  for (auto const& [key, r] : readings.all()) {
    if (key.kind != kind || r.failed || r.calls == 0 || held.contains(key.fft)) { continue; }
    auto [it, fresh] =
      calls.try_emplace({key.fft, key.regime}, Subject{.fft = r.fft, .kind = kind, .regime = r.regime}, 0);
    it->second.second += r.calls;
  }
  std::map<std::string, std::pair<Subject, u32>> most;
  for (auto const& [at, read] : calls) {
    auto [it, fresh] = most.try_emplace(at.first, read);
    if (!fresh && read.second > it->second.second) { it->second = read; }
  }
  std::map<std::string, Subject> out;
  for (auto const& [spec, read] : most) { out.emplace(spec, read.first); }
  return out;
}

struct FftFront {
  std::vector<Line> lines;   // cheapest first
  std::optional<Line> owed;  // the cheapest set waiting for its gate, where the run holds the FFT
};

// One FFT's frontier: each set, cheapest first, that reaches further than every cheaper one.  Where the run holds the
// FFT (`c`), a set whose gate the run owes waits for it; elsewhere nothing will settle that gate, so the set reaches as
// far as its highest safe reading.
FftFront frontOf(const Subject& subject, const Candidate* c, const Readings& readings) {
  std::vector<const Reading*> rs = readings.of(subject);
  std::erase_if(rs, [](const Reading* r) { return r->failed || r->calls < MIN_CALLS; });
  std::ranges::sort(rs, {}, &Reading::pessimistic);

  u64 const lowest = minExp(subject.fft);
  u64 const table = maxExp(subject.fft);
  FftFront out;
  u64 reached = 0;
  for (const Reading* r : rs) {
    if (reached >= table) { break; }
    std::vector<RoeReading> const roes = readings.roes(subject.fft, r->options);
    if (c) {
      Verdict const v = verdictOf(subject.fft, roes, c->points.front().exponent);
      if (v.state == Accuracy::Owed) {
        if (!out.owed) { out.owed = Line{.subject = subject, .candidate = c, .reading = r, .accuracy = v}; }
        continue;
      }
    }
    Verdict const v = verdictOf(subject.fft, roes, lowest);
    if (v.state == Accuracy::Rejected || v.reach < lowest || v.reach <= reached) { continue; }
    out.lines.push_back({.subject = subject, .candidate = c, .reading = r, .accuracy = v});
    reached = v.reach;
  }
  return out;
}

// Every FFT's front, the run's candidates in their main regime.
std::vector<FftFront> frontsOf(const Readings& readings, TestKind kind,
                               const std::map<std::string, const Candidate*>& held) {
  std::vector<FftFront> out;
  for (auto const& [spec, c] : held) { out.push_back(frontOf(c->subject, c, readings)); }
  for (auto const& [spec, subject] : unheldSubjects(readings, kind, held)) {
    out.push_back(frontOf(subject, nullptr, readings));
  }
  return out;
}

// The frontier as production reads it (TuneEntry::update): cheapest first, a line dropped where a cheaper one reaches
// at least as far.
std::vector<Line> frontier(std::vector<Line> lines) {
  std::ranges::stable_sort(lines, {}, &Line::cost);
  std::vector<Line> out;
  for (const Line& line : lines) {
    if (out.empty() || line.reach() > out.back().reach()) { out.push_back(line); }
  }
  return out;
}

}  // namespace

bool Line::holds(u64 E) const { return E >= minExp(subject.fft) && E <= reach(); }

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

size_t Carried::size() const {
  size_t n = 0;
  for (auto const& [kind, ls] : lines) { n += ls.size(); }
  return n;
}

Carried carriedLines(const TuneDB& db, u32 env, std::span<const Grid> grids) {
  Carried out;
  const DbEnv* const now = db.findEnv(env);
  if (!now) { return out; }
  std::vector<const DbEnv*> earlier;
  for (const DbEnv& e : db.envs()) {
    if (e.id != env && e.sameCard(*now)) { earlier.push_back(&e); }
  }
  std::ranges::sort(earlier, std::greater{}, &DbEnv::id);

  std::map<TestKind, std::set<std::string>> taken;
  for (const DbEnv* e : earlier) {
    const Readings& readings = out.readings.emplace_back(db, e->id, e->toEnv());
    for (const Grid& grid : grids) {
      std::vector<Line>& lines = out.lines[grid.kind];
      std::set<std::string>& specs = taken[grid.kind];
      std::set<std::string> fresh;
      for (FftFront& f : frontsOf(readings, grid.kind, {})) {
        if (f.lines.empty() || specs.contains(f.lines.front().subject.fft.spec())) { continue; }
        fresh.insert(f.lines.front().subject.fft.spec());
        std::ranges::move(f.lines, std::back_inserter(lines));
      }
      specs.merge(fresh);
    }
  }
  return out;
}

Publication publish(const RunScope& scope, const std::vector<std::vector<Candidate>>& candidates,
                    const Readings& readings, const Carried* carried) {
  Publication out;

  for (size_t ki = 0; ki < scope.grids.size() && ki < candidates.size(); ++ki) {
    const Grid& grid = scope.grids[ki];
    KindPublication pub{.kind = grid.kind, .lines = {}, .written = {}};
    std::map<std::string, const Candidate*> const held = mainCandidates(candidates[ki]);
    std::vector<Line> lines;
    std::vector<Line> owed;
    for (FftFront& f : frontsOf(readings, grid.kind, held)) {
      std::ranges::move(f.lines, std::back_inserter(lines));
      if (f.owed) { owed.push_back(*f.owed); }
    }

    // The cheapest reading by its mean, where it is short of its calls: a second call may make it a line.
    std::vector<std::pair<ConcludeItem, double>> short_;
    for (auto const& [spec, c] : held) {
      const Reading* best = nullptr;
      for (const Reading* r : readings.of(c->subject)) {
        if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
      }
      if (best && best->calls < MIN_CALLS) {
        short_.emplace_back(ConcludeItem{.candidate = c, .options = best->options}, best->mean);
      }
    }

    pub.lines = frontier(lines);
    if (carried) {
      if (auto const it = carried->lines.find(grid.kind); it != carried->lines.end()) {
        std::set<std::string> specs;
        for (const Line& line : lines) { specs.insert(line.subject.fft.spec()); }
        for (const Line& line : it->second) {
          if (!specs.contains(line.subject.fft.spec())) { lines.push_back(line); }
        }
      }
    }
    pub.written = frontier(std::move(lines));

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
    for (const Line& line : owed) {
      if (double const w = wouldServe(line.candidate, line.cost()); w > 0) {
        out.owedGates.push_back(
          {.fft = line.subject.fft, .options = line.reading->options, .exponent = line.accuracy.next, .weight = w});
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
  for (const Line& line : kind.written) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%6.1f %14s # %" PRIu64, line.cost(), line.subject.fft.spec().c_str(), line.reach());
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
