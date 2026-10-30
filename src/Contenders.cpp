// Copyright (C) Jason Lynch

#include "Contenders.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <map>
#include <set>

namespace tune {

namespace {

// The ratio's interval: two standard errors either side, near enough 95%.
constexpr double Z95 = 1.96;

std::string optionsText(const UseConfig& options) { return options.empty() ? "the defaults" : configText(options); }

// "+1.2%", "-0.4%"
std::string signedPercent(double ratio) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%+.1f%%", (ratio - 1) * 100);
  return buf;
}

std::string usText(double us) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.1f", us);
  return buf;
}

// "512:7:512:212 short32 -use ZEROHACK_W=0"
std::string configName(const Subject& subject, const UseConfig& options) {
  return subject.fft.spec() + ' ' + subject.regime.label() + " -use " + optionsText(options);
}

struct Ratio {
  double ratio = 1;
  double lo = 1;
  double hi = 1;
};

// The ratio of `a`'s mean to `b`'s, with its interval from their relative standard errors.
Ratio ratioOf(const Reading& a, const Reading& b) {
  double const ratio = a.mean / b.mean;
  double const spread = Z95 * std::hypot(a.se / a.mean, b.se / b.mean);
  return {.ratio = ratio, .lo = ratio * std::exp(-spread), .hi = ratio * std::exp(spread)};
}

// The rows a stretch can take a rival from: a set that might yet take a line.
bool couldTakeALine(Standing s) {
  return s == Standing::Undercut || s == Standing::ShortOfCalls || s == Standing::GateOwed;
}

SpaceRead spaceRead(const Env& env, const Candidate& c, const Readings& readings,
                    const std::function<AxisFilter(const Candidate&)>& filter) {
  std::vector<const Reading*> const of = readings.of(c.subject);
  std::shared_ptr<const ConditionalSpace> const space = spaceOf(env, c.subject.fft, filter(c));
  return {.branches = archiveOf(env, of).size(),
          .legal = space->legalBranches(),
          .positions = positionCoverage(*space, of)};
}

// "branches 4 of 48, positions 140 of 920"
std::string readText(const SpaceRead& read) {
  char buf[128];
  snprintf(buf, sizeof(buf), "branches %zu of %s%zu, positions %zu of %zu", read.branches,
           read.legal.complete ? "" : "over ", read.legal.legal, read.positions.covered, read.positions.total);
  return buf;
}

// Whether the run owes `standing` a call or an accuracy reading.
bool owes(const Publication& publication, const Subject& subject, const StandingOf& standing) {
  const UseConfig& options = standing.reading->options;
  if (standing.standing == Standing::GateOwed) {
    return std::ranges::any_of(publication.owedGates, [&](const GateItem& g) {
      return g.fft.spec() == subject.fft.spec() && g.of == options;
    });
  }
  if (standing.standing == Standing::ShortOfCalls) {
    return std::ranges::any_of(publication.owedCalls, [&](const ConcludeItem& i) {
      return i.candidate->subject.fft.spec() == subject.fft.spec() && i.candidate->subject.kind == subject.kind &&
        i.candidate->subject.regime == subject.regime && i.options == options;
    });
  }
  return false;
}

}  // namespace

std::vector<KindContest> contendersOf(const Env& env, const RunScope& scope,
                                      const std::vector<std::vector<Candidate>>& candidates,
                                      const Readings& readings, const Publication& publication,
                                      const std::function<AxisFilter(const Candidate&)>& filter,
                                      const ContestSettings& settings) {
  std::vector<KindContest> out;
  for (size_t ki = 0; ki < scope.grids.size() && ki < candidates.size(); ++ki) {
    const Grid& grid = scope.grids[ki];
    const KindPublication* const pub = publication.of(grid.kind);
    if (!pub) { continue; }
    KindContest kind{.kind = grid.kind, .contests = {}};

    // The stretches, by the line production runs at each point in exponent order.
    std::vector<GridPoint> points = grid.points;
    std::ranges::sort(points, {}, &GridPoint::exponent);
    double total = 0;
    for (const GridPoint& p : points) { total += p.weight; }
    for (const GridPoint& p : points) {
      auto const runs = std::ranges::find_if(pub->written, [&](const Line& line) { return line.holds(p.exponent); });
      const Line* const line = runs == pub->written.end() ? nullptr : &*runs;
      if (kind.contests.empty() || kind.contests.back().line != line) {
        kind.contests.push_back({.line = line, .lo = p.exponent, .hi = p.exponent, .rivals = {}});
      }
      kind.contests.back().hi = p.exponent;
      kind.contests.back().weight += total > 0 ? p.weight / total : 0;
    }

    // Each held FFT and regime's rows that might yet take a line, the cheapest first: read once, used by every
    // stretch the candidate holds points of.
    std::map<std::string, const Candidate*> held;
    for (const Candidate& c : candidates[ki]) {
      auto [it, fresh] = held.try_emplace(c.subject.fft.spec() + ' ' + c.subject.regime.label(), &c);
      if (!fresh && c.weight() > it->second->weight()) { it->second = &c; }
    }
    std::map<const Candidate*, std::vector<StandingOf>> rows;
    for (auto const& [place, c] : held) {
      if (readings.of(c->subject).empty()) { continue; }
      std::vector<StandingOf> standings = standingsOf(*pub, c->subject, c, readings);
      std::erase_if(standings, [](const StandingOf& s) { return !couldTakeALine(s.standing); });
      if (!standings.empty()) { rows.emplace(c, std::move(standings)); }
    }

    for (Contest& contest : kind.contests) {
      if (contest.line) {
        if (const Candidate* held = heldCandidate(candidates[ki], contest.line->subject)) {
          contest.lineRead = spaceRead(env, *held, readings, filter);
        }
      }
      std::vector<Rival> rivals;
      for (auto const& [c, standings] : rows) {
        bool const overlaps = std::ranges::any_of(
          c->points, [&](const GridPoint& p) { return p.exponent >= contest.lo && p.exponent <= contest.hi; });
        if (!overlaps) { continue; }
        // Its cheapest row that holds some point of the stretch: a gate or a call settles the others' reach.
        for (const StandingOf& s : standings) {
          bool const reaches = s.reach == 0 || s.reach >= contest.lo;
          if (!reaches || (contest.line && s.reading == contest.line->reading)) { continue; }
          Ratio const ratio = contest.line ? ratioOf(*s.reading, *contest.line->reading) : Ratio{};
          rivals.push_back({.subject = c->subject, .standing = s, .ratio = ratio.ratio, .lo = ratio.lo, .hi = ratio.hi});
          break;
        }
      }
      if (!contest.line && !rivals.empty()) {
        // No line: each rival against the cheapest of them.
        auto const cheapest = std::ranges::min_element(rivals, {}, [](const Rival& r) { return r.standing.reading->mean; });
        const Reading* const base = cheapest->standing.reading;
        for (Rival& r : rivals) {
          Ratio const ratio = ratioOf(*r.standing.reading, *base);
          r.ratio = ratio.ratio;
          r.lo = ratio.lo;
          r.hi = ratio.hi;
        }
      }
      std::ranges::stable_sort(rivals, {}, &Rival::ratio);
      std::erase_if(rivals, [&](const Rival& r) { return r.ratio > 1 + settings.within; });
      if (rivals.size() > settings.top) { rivals.erase(rivals.begin() + ptrdiff_t(settings.top), rivals.end()); }
      for (Rival& r : rivals) {
        r.owed = owes(publication, r.subject, r.standing);
        if (const Candidate* held = heldCandidate(candidates[ki], r.subject)) {
          r.read = spaceRead(env, *held, readings, filter);
        }
      }
      contest.rivals = std::move(rivals);
    }
    out.push_back(std::move(kind));
  }
  return out;
}

std::string whyText(const StandingOf& s, bool owed) {
  char buf[256];
  switch (s.standing) {
  case Standing::Published:
    if (s.line) {
      snprintf(buf, sizeof(buf), "published, %" PRIu64 "-%" PRIu64, s.line->lo(), s.line->reach());
      return buf;
    }
    return "published";
  case Standing::Failed: return "failed: a call computed wrongly, failed to build, or the device refused it";
  case Standing::ShortOfCalls:
    snprintf(buf, sizeof(buf), "short of calls: %u of %u%s", s.reading->calls, MIN_CALLS,
             owed ? ", the next is owed" : ", none owed: it would serve no exponent of the workload");
    return buf;
  case Standing::GateOwed:
    snprintf(buf, sizeof(buf), "accuracy owed at %" PRIu64 "%s", s.at,
             owed ? "" : ", not asked for: it would serve no exponent of the workload");
    return buf;
  case Standing::Rejected: return "rejected: unsafe throughout its regime";
  case Standing::ShortReach:
    snprintf(buf, sizeof(buf), "short reach: safe only to %" PRIu64 ", below its regime", s.at);
    return buf;
  case Standing::Outreached:
    if (s.by) {
      return "outreached: " + optionsText(s.by->options) + " is cheaper, at " + usText(s.by->pessimistic()) +
        ", and reaches as far";
    }
    return "outreached";
  case Standing::Undercut:
    if (s.line) {
      snprintf(buf, sizeof(buf), "undercut: at %" PRIu64 " production runs %s at %.1f, %.1f%% cheaper", s.at,
               s.line->subject.fft.spec().c_str(), s.line->cost(),
               (1 - s.line->cost() / s.reading->pessimistic()) * 100);
      return std::string{buf} + (s.line->carried ? ", an earlier build's line" : "");
    }
    return "undercut";
  case Standing::NoRegime: return "its FFT never runs in this regime";
  }
  return "?";
}

std::vector<std::string> contestLines(const std::vector<KindContest>& table) {
  std::vector<std::string> out;
  char buf[256];
  for (const KindContest& kind : table) {
    for (const Contest& contest : kind.contests) {
      snprintf(buf, sizeof(buf), "%s %" PRIu64 "-%" PRIu64 ", %.0f%% of the weight: ", toString(kind.kind), contest.lo,
               contest.hi, contest.weight * 100);
      std::string line = buf;
      if (contest.line) {
        line += usText(contest.line->cost()) + ' ' + configName(contest.line->subject, contest.line->reading->options) +
          "; " + readText(contest.lineRead) + (contest.line->carried ? "; an earlier build's line" : "");
      } else {
        line += "no line";
      }
      out.push_back(line);
      for (const Rival& r : contest.rivals) {
        snprintf(buf, sizeof(buf), "    %s [%s, %s] %.1f ", signedPercent(r.ratio).c_str(), signedPercent(r.lo).c_str(),
                 signedPercent(r.hi).c_str(), r.standing.reading->mean);
        out.push_back(buf + configName(r.subject, r.standing.reading->options) + ": " +
                      toString(r.standing.standing) + ", " + std::to_string(r.standing.reading->calls) + " call" +
                      (r.standing.reading->calls == 1 ? "" : "s") +
                      (r.standing.reading->beaten > 0 ? ", lost a duel" : "") + "; " + readText(r.read));
      }
      if (contest.rivals.empty()) { out.emplace_back("    no rival within reach"); }
    }
  }
  return out;
}

bool matchesFft(const ExplainQuery& query, const std::string& spec) {
  return spec == query.fft || spec.starts_with(query.fft + ':');
}

std::vector<std::string> explainLines(const Env& env, const ExplainQuery& query, const RunScope& scope,
                                      const std::vector<std::vector<Candidate>>& candidates,
                                      const Readings& readings, const Publication& publication) {
  // Every subject of the FFT read or held, per kind.
  std::map<ConfigKey, Subject> subjects;
  for (auto const& [key, r] : readings.all()) {
    if (matchesFft(query, key.fft)) {
      subjects.try_emplace(Readings::keyOf({.fft = r.fft, .kind = r.kind, .regime = r.regime}, {}),
                           Subject{.fft = r.fft, .kind = r.kind, .regime = r.regime});
    }
  }
  for (const std::vector<Candidate>& kind : candidates) {
    for (const Candidate& c : kind) {
      if (matchesFft(query, c.subject.fft.spec())) { subjects.try_emplace(Readings::keyOf(c.subject, {}), c.subject); }
    }
  }

  std::vector<std::string> out;
  if (subjects.empty()) {
    out.push_back(query.fft + ": nothing read, and no candidate of this workload; " +
                  "an FFT outside the workload, or one this card cannot run, is never measured");
    return out;
  }

  char buf[256];
  // The workload's candidates nothing has read, which would each say only that: one line per kind and regime.
  std::map<std::string, std::vector<std::string>> unread;
  for (auto const& [key, subject] : subjects) {
    const KindPublication* const pub = publication.of(subject.kind);
    const Candidate* held = nullptr;
    for (size_t ki = 0; ki < scope.grids.size() && ki < candidates.size(); ++ki) {
      if (scope.grids[ki].kind == subject.kind) { held = heldCandidate(candidates[ki], subject); }
    }
    if (held && readings.of(subject).empty()) {
      unread[std::string{toString(subject.kind)} + ' ' + subject.regime.label()].push_back(subject.fft.spec());
      continue;
    }
    std::string const name = subject.fft.spec() + ' ' + toString(subject.kind) + ' ' + subject.regime.label();
    if (held) {
      snprintf(buf, sizeof(buf), "%s: a candidate over %" PRIu64 "-%" PRIu64 ", %zu grid points", name.c_str(),
               held->interval.lo, held->interval.hi, held->points.size());
      out.emplace_back(buf);
    } else {
      out.push_back(name + ": outside this workload; its lines are kept, but nothing more is measured");
    }
    if (!pub) {
      out.emplace_back("    nothing published of this kind");
      continue;
    }

    std::vector<StandingOf> const standings = standingsOf(*pub, subject, held, readings);
    std::vector<StandingOf> shown = standings;
    if (query.options) {
      UseConfig const wanted = canonicalConfig(env, subject.fft, *query.options);
      std::erase_if(shown, [&](const StandingOf& s) { return s.reading->options != wanted; });
      if (shown.empty()) {
        out.push_back("    -use " + optionsText(wanted) + " was never read on it" +
                      (standings.empty() ? std::string{}: "; the nearest sets that were:"));
        std::vector<StandingOf> near = standings;
        std::ranges::stable_sort(near, {}, [&](const StandingOf& s) { return axesApart(s.reading->options, wanted); });
        if (near.size() > 3) { near.erase(near.begin() + 3, near.end()); }
        shown = std::move(near);
      }
    }
    if (shown.empty()) { out.emplace_back("    nothing read"); }
    for (const StandingOf& s : shown) {
      snprintf(buf, sizeof(buf), "    %8.1f ± %.1f, %u call%s%s -use ", s.reading->mean, s.reading->se,
               s.reading->calls, s.reading->calls == 1 ? "" : "s", s.reading->isSoaked() ? ", soaked" : "");
      out.push_back(buf + optionsText(s.reading->options) + ": " + whyText(s, owes(publication, subject, s)) +
                    (s.reading->beaten > 0 ? "; ranked behind the winner of a duel it lost" : ""));
    }
  }
  for (auto const& [place, specs] : unread) {
    std::string line = place + ", candidates of this workload never read:";
    for (const std::string& spec : specs) { line += ' ' + spec; }
    out.push_back(line);
  }
  return out;
}

}  // namespace tune
