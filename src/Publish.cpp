// Copyright (C) Jason Lynch

#include "Publish.h"

#include "BuildId.h"
#include "CycleFile.h"

#include <algorithm>
#include <cinttypes>
#include <iterator>
#include <map>
#include <optional>
#include <set>

namespace tune {

namespace {

constexpr TestKind KINDS[] = {TestKind::PRP, TestKind::LL};

// A subject's FFT and regime, which is what one line of the file can stand for.
std::string placeOf(const Subject& subject) { return subject.fft.spec() + ' ' + subject.regime.label(); }

// The run's candidate of each FFT and regime: the heaviest, should there be two.
std::map<std::string, const Candidate*> heldCandidates(const std::vector<Candidate>& candidates) {
  std::map<std::string, const Candidate*> out;
  for (const Candidate& c : candidates) {
    auto [it, fresh] = out.try_emplace(placeOf(c.subject), &c);
    if (!fresh && c.weight() > it->second->weight()) { it->second = &c; }
  }
  return out;
}

// Each FFT and regime read for `kind` that the run does not hold.
std::vector<Subject> unheldSubjects(const Readings& readings, TestKind kind,
                                    const std::map<std::string, const Candidate*>& held) {
  std::map<std::string, Subject> out;
  for (auto const& [key, r] : readings.all()) {
    if (key.kind != kind || r.failed || r.calls == 0) { continue; }
    Subject const subject{.fft = r.fft, .kind = kind, .regime = r.regime};
    if (!held.contains(placeOf(subject))) { out.try_emplace(placeOf(subject), subject); }
  }
  std::vector<Subject> subjects;
  for (auto& [place, subject] : out) { subjects.push_back(std::move(subject)); }
  return subjects;
}

// The exponents the subject's FFT runs in the subject's regime; nothing where it never does.
std::optional<Interval> spanOf(const Subject& subject) {
  for (const Interval& i : intervals(subject.fft, minExp(subject.fft), maxExp(subject.fft))) {
    if (i.regime == subject.regime) { return i; }
  }
  return {};
}

struct FftFront {
  std::vector<Line> lines;   // cheapest first
  std::optional<Line> owed;  // the cheapest set waiting for its gate, where the run holds the subject
};

// One FFT's frontier in one regime: each set, cheapest first, that reaches further than every cheaper one.  Where the
// run holds the subject (`c`), a set whose gate the run owes waits for it; elsewhere nothing will settle that gate, so
// the set reaches as far as its highest safe reading.  Where `why` is given, it is told what became of every reading,
// the lines' as Published.
FftFront frontOf(const Subject& subject, const Candidate* c, const Readings& readings,
                 std::vector<StandingOf>* why = nullptr) {
  FftFront out;
  std::vector<const Reading*> rs = readings.of(subject);
  std::ranges::sort(rs, {}, &Reading::pessimistic);
  std::optional<Interval> const span = spanOf(subject);
  if (why) {
    for (const Reading* r : rs) {
      if (!span) {
        why->push_back({.reading = r, .standing = Standing::NoRegime});
      } else if (r->failed) {
        why->push_back({.reading = r, .standing = Standing::Failed});
      } else if (r->calls < MIN_CALLS) {
        why->push_back({.reading = r, .standing = Standing::ShortOfCalls});
      }
    }
  }
  if (!span) { return out; }
  std::erase_if(rs, [](const Reading* r) { return r->failed || r->calls < MIN_CALLS; });

  u64 reached = 0;
  for (const Reading* r : rs) {
    if (reached >= span->hi) {
      if (!why) { break; }
      why->push_back({.reading = r, .standing = Standing::Outreached, .by = out.lines.back().reading});
      continue;
    }
    std::vector<RoeReading> const roes = readings.roes(subject.fft, r->options);
    if (c) {
      Verdict const v = verdictOf(subject.fft, roes, c->points.front().exponent);
      if (v.state == Accuracy::Owed) {
        if (!out.owed) {
          out.owed = Line{.subject = subject, .candidate = c, .reading = r, .accuracy = v, .interval = *span};
        }
        if (why) { why->push_back({.reading = r, .standing = Standing::GateOwed, .at = v.next}); }
        continue;
      }
    }
    Verdict const v = verdictOf(subject.fft, roes, span->lo);
    u64 const hi = std::min(v.reach, span->hi);
    if (v.state == Accuracy::Rejected || hi < span->lo || hi <= reached) {
      if (why) {
        Standing const s = v.state == Accuracy::Rejected ? Standing::Rejected
          : hi < span->lo                                ? Standing::ShortReach
                                                         : Standing::Outreached;
        why->push_back({.reading = r,
                        .standing = s,
                        .at = s == Standing::ShortReach ? v.reach : 0,
                        .reach = hi,
                        .by = s == Standing::Outreached ? out.lines.back().reading : nullptr});
      }
      continue;
    }
    out.lines.push_back({.subject = subject,
                         .candidate = c,
                         .reading = r,
                         .accuracy = v,
                         .interval = {.lo = span->lo, .hi = hi, .regime = subject.regime}});
    if (why) { why->push_back({.reading = r, .standing = Standing::Published, .reach = hi}); }
    reached = hi;
  }
  return out;
}

// Every FFT's front in every regime read, the run's candidates first.
std::vector<FftFront> frontsOf(const Readings& readings, TestKind kind,
                               const std::map<std::string, const Candidate*>& held) {
  std::vector<FftFront> out;
  for (auto const& [place, c] : held) { out.push_back(frontOf(c->subject, c, readings)); }
  for (const Subject& subject : unheldSubjects(readings, kind, held)) {
    out.push_back(frontOf(subject, nullptr, readings));
  }
  return out;
}

// Whether the disjoint, ordered spans `covered` hold every exponent of [lo, hi].
bool coveredThroughout(const std::vector<std::pair<u64, u64>>& covered, u64 lo, u64 hi) {
  for (auto const& [a, b] : covered) {
    if (a > lo) { return false; }
    if (b >= lo) {
      if (b >= hi) { return true; }
      lo = b + 1;
    }
  }
  return false;
}

void cover(std::vector<std::pair<u64, u64>>& covered, u64 lo, u64 hi) {
  covered.emplace_back(lo, hi);
  std::ranges::sort(covered);
  std::vector<std::pair<u64, u64>> merged;
  for (auto const& [a, b] : covered) {
    if (!merged.empty() && a <= merged.back().second + 1) {
      merged.back().second = std::max(merged.back().second, b);
    } else {
      merged.emplace_back(a, b);
    }
  }
  covered = std::move(merged);
}

// The frontier as production reads it: cheapest first, a line kept where it is the cheapest at some exponent it
// holds, and dropped where cheaper lines hold every one of them.
std::vector<Line> frontier(std::vector<Line> lines) {
  std::ranges::stable_sort(lines, {}, &Line::cost);
  std::vector<Line> out;
  std::vector<std::pair<u64, u64>> covered;
  for (Line& line : lines) {
    if (coveredThroughout(covered, line.lo(), line.reach())) { continue; }
    cover(covered, line.lo(), line.reach());
    out.push_back(std::move(line));
  }
  return out;
}

// Whether this build has settled the configuration a carried line stands for: read it MIN_CALLS times, or seen it fail --
// a process dying while holding it among the failures, whether or not it read a cost first.
bool settled(const Readings& readings, const Line& line) {
  if (readings.died(line.subject.fft, line.reading->options)) { return true; }
  const Reading* const r = readings.find(line.subject, line.reading->options);
  return r && (r->failed || r->calls >= MIN_CALLS);
}

}  // namespace

const char* toString(Standing standing) {
  switch (standing) {
  case Standing::Published: return "published";
  case Standing::Failed: return "failed";
  case Standing::ShortOfCalls: return "short of calls";
  case Standing::GateOwed: return "accuracy owed";
  case Standing::Rejected: return "rejected";
  case Standing::ShortReach: return "short reach";
  case Standing::Outreached: return "outreached";
  case Standing::Undercut: return "undercut";
  case Standing::NoRegime: return "no such regime";
  }
  return "?";
}

const Candidate* heldCandidate(const std::vector<Candidate>& candidates, const Subject& subject) {
  const Candidate* out = nullptr;
  for (const Candidate& c : candidates) {
    if (placeOf(c.subject) == placeOf(subject) && c.subject.kind == subject.kind &&
        (!out || c.weight() > out->weight())) {
      out = &c;
    }
  }
  return out;
}

std::vector<StandingOf> standingsOf(const KindPublication& pub, const Subject& subject, const Candidate* held,
                                    const Readings& readings) {
  std::vector<StandingOf> out;
  (void)frontOf(subject, held, readings, &out);
  for (StandingOf& s : out) {
    if (s.standing != Standing::Published) { continue; }
    auto const own = std::ranges::find(pub.written, s.reading, &Line::reading);
    if (own != pub.written.end()) {
      s.line = &*own;
      continue;
    }
    // Cheaper lines hold every exponent it reaches: say which one runs where the run would weigh it most, or at its
    // reach where the run holds none of it.
    s.standing = Standing::Undercut;
    s.at = s.reach;
    if (held) {
      double heaviest = -1;
      for (const GridPoint& p : held->points) {
        if (p.exponent <= s.reach && p.weight > heaviest) {
          heaviest = p.weight;
          s.at = p.exponent;
        }
      }
    }
    auto const runs = std::ranges::find_if(pub.written, [&](const Line& line) { return line.holds(s.at); });
    s.line = runs == pub.written.end() ? nullptr : &*runs;
  }
  std::ranges::stable_sort(out, {}, [](const StandingOf& s) { return s.reading->pessimistic(); });
  return out;
}

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
  double sum = 0;
  u32 n = 0;
  for (const KindPublication& k : kinds) {
    if (!k.tuned) { continue; }
    sum += k.covered;
    ++n;
  }
  return n ? sum / n : 0;
}

size_t Carried::size() const {
  size_t n = 0;
  for (auto const& [kind, ls] : lines) { n += ls.size(); }
  return n;
}

Carried carriedLines(const TuneDB& db, u32 env) {
  Carried out;
  const DbEnv* const now = db.findEnv(env);
  if (!now) { return out; }
  Env const card = now->toEnv();
  std::vector<const DbEnv*> earlier;
  for (const DbEnv& e : db.envs()) {
    if (e.id != env && e.sameCard(*now)) { earlier.push_back(&e); }
  }
  std::ranges::sort(earlier, std::greater{}, &DbEnv::id);

  std::set<ConfigKey> taken;
  for (const DbEnv* e : earlier) {
    const Readings& readings = out.readings.emplace_back(db, e->id, e->toEnv());
    for (TestKind const kind : KINDS) {
      std::vector<Line>& lines = out.lines[kind];
      std::set<ConfigKey> fresh;
      for (FftFront& f : frontsOf(readings, kind, {})) {
        for (Line& line : f.lines) {
          // Options this build's table no longer names as they were would not run as they were measured.
          if (canonicalConfig(card, line.subject.fft, line.reading->options) != line.reading->options) { continue; }
          ConfigKey const key = Readings::keyOf(line.subject, line.reading->options);
          if (taken.contains(key)) { continue; }
          fresh.insert(key);
          line.carried = true;
          lines.push_back(std::move(line));
        }
      }
      taken.merge(fresh);
    }
  }
  return out;
}

Publication publish(const RunScope& scope, const std::vector<std::vector<Candidate>>& candidates,
                    const Readings& readings, const Carried* carried) {
  Publication out;

  // This build's lines and the carried ones it has not settled, of one kind.
  auto linesOf = [&](TestKind kind, const std::map<std::string, const Candidate*>& held, std::vector<Line>& owed) {
    std::vector<Line> lines;
    for (FftFront& f : frontsOf(readings, kind, held)) {
      std::ranges::move(f.lines, std::back_inserter(lines));
      if (f.owed) { owed.push_back(*f.owed); }
    }
    std::vector<Line> withCarried = lines;
    if (carried) {
      if (auto const it = carried->lines.find(kind); it != carried->lines.end()) {
        for (const Line& line : it->second) {
          if (settled(readings, line)) { continue; }
          Line copy = line;
          auto const c = held.find(placeOf(line.subject));
          copy.candidate = c == held.end() ? nullptr : c->second;
          withCarried.push_back(std::move(copy));
        }
      }
    }
    return std::pair{std::move(lines), std::move(withCarried)};
  };

  for (size_t ki = 0; ki < scope.grids.size() && ki < candidates.size(); ++ki) {
    const Grid& grid = scope.grids[ki];
    KindPublication pub{.kind = grid.kind, .tuned = true, .lines = {}, .written = {}};
    std::map<std::string, const Candidate*> const held = heldCandidates(candidates[ki]);
    std::vector<Line> owed;
    auto [lines, withCarried] = linesOf(grid.kind, held, owed);

    // The cheapest reading by its mean, where it is short of its calls: a second call may make it a line.
    std::vector<std::pair<ConcludeItem, double>> short_;
    for (auto const& [place, c] : held) {
      const Reading* best = nullptr;
      for (const Reading* r : readings.of(c->subject)) {
        if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
      }
      if (best && best->calls < MIN_CALLS) {
        short_.emplace_back(ConcludeItem{.candidate = c, .options = best->options}, best->mean);
      }
    }

    pub.lines = frontier(std::move(lines));
    pub.written = frontier(std::move(withCarried));

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
    // A carried line the tune file holds is read by this build before anything else replaces it, wherever the run
    // holds its FFT and regime.
    for (const Line& line : pub.written) {
      if (line.carried && line.candidate) {
        out.owedCalls.push_back({.candidate = line.candidate, .options = line.reading->options, .carried = true});
      }
    }
    for (auto const& [item, mean] : short_) {
      if (wouldServe(item.candidate, mean) > 0) { out.owedCalls.push_back(item); }
    }
    out.kinds.push_back(std::move(pub));
  }

  // The kinds the run does not tune keep their lines too: the file holds every kind.
  for (TestKind const kind : KINDS) {
    if (out.of(kind)) { continue; }
    std::vector<Line> owed;
    auto [lines, withCarried] = linesOf(kind, {}, owed);
    if (withCarried.empty()) { continue; }
    out.kinds.push_back(
      {.kind = kind, .tuned = false, .lines = frontier(std::move(lines)), .written = frontier(std::move(withCarried))});
  }

  std::ranges::sort(out.owedGates, std::greater{}, &GateItem::weight);
  std::ranges::stable_sort(out.owedCalls, std::greater{}, [](const ConcludeItem& i) { return i.candidate->weight(); });
  return out;
}

std::string tuneFileText(const Publication& publication, const std::string& header) {
  char buf[160];
  snprintf(buf, sizeof(buf), "# %s, build %016" PRIx64 "\n", TUNE_FILE_MAGIC, buildFingerprint());
  std::string out = buf;
  for (size_t at = 0; at < header.size();) {
    size_t const eol = std::min(header.find('\n', at), header.size());
    out += "# " + header.substr(at, eol - at) + '\n';
    at = eol + 1;
  }
  for (TestKind const kind : KINDS) {
    const KindPublication* const pub = publication.of(kind);
    if (!pub) { continue; }
    for (const Line& line : pub->written) {
      snprintf(buf, sizeof(buf), "%6.1f %14s %s %s %" PRIu64 "-%" PRIu64, line.cost(),
               line.subject.fft.spec().c_str(), toString(kind), line.subject.regime.label().c_str(), line.lo(),
               line.reach());
      out += buf;
      std::string sep = " # ";
      for (const auto& [key, value] : line.reading->options) {
        out += sep + key + '=' + value;
        sep = ", ";
      }
      out += '\n';
    }
  }
  return out;
}

void writeTuneFile(const fs::path& dir, const Publication& publication, const std::string& header) {
  CycleFile file{dir / TUNE_FILE};
  file->write(tuneFileText(publication, header));
}

}  // namespace tune
