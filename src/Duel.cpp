// Copyright (C) Jason Lynch

#include "Duel.h"

#include "Gate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tune {

namespace {

// No duel's ratio is known better than this, relative, whatever its pairs say: two pairs can agree by chance far
// better than calls do.
constexpr double DUEL_SE_FLOOR = 0.0005;

// Student's t at 97.5% for 1 to 30 degrees of freedom; the normal's beyond.
constexpr double T975[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                           2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                           2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};

double t975(size_t df) { return df >= 1 && df <= std::size(T975) ? T975[df - 1] : 1.96; }

std::string sideKey(const DuelSide& side) {
  return side.candidate->subject.fft.spec() + ' ' + side.candidate->subject.regime.label() + ':' +
    configText(side.options);
}

// Whether the configuration's accuracy is settled as far as `exponent`, from the lowest exponent the candidate serves.
bool reaches(const Candidate& c, const Readings& readings, const UseConfig& options, u64 exponent) {
  Verdict const v = verdictOf(c.subject.fft, readings.roes(c.subject.fft, options), c.points.front().exponent);
  return v.state != Accuracy::Owed && v.state != Accuracy::Rejected && v.reach >= exponent;
}

}  // namespace

std::string DuelPlan::key() const { return sideKey(incumbent) + " vs " + sideKey(challenger); }

std::optional<DuelPlan> planDuel(const Publication& publication, const std::vector<std::vector<Candidate>>& candidates,
                                 const Readings& readings, const DuelSettings& settings,
                                 const std::set<std::string>& tried) {
  if (settings.share <= 0) { return {}; }

  std::optional<DuelPlan> out;
  for (size_t ki = 0; ki < publication.kinds.size() && ki < candidates.size(); ++ki) {
    const KindPublication& pub = publication.kinds[ki];
    if (!pub.tuned) { continue; }
    for (const Line& line : pub.lines) {
      if (!line.candidate || line.carried) { continue; }
      // The weight the line serves, and its heaviest exponent.
      double weight = 0;
      const GridPoint* heaviest = nullptr;
      for (const GridPoint& p : line.candidate->points) {
        if (pub.at(p.exponent) != &line) { continue; }
        weight += p.weight;
        if (!heaviest || p.weight > heaviest->weight) { heaviest = &p; }
      }
      if (!heaviest || (out && weight <= out->weight)) { continue; }

      // The cheapest configuration read before the line's that would serve that exponent: the one it took the line
      // from.  One read after it, however cheap, took nothing from it.
      const Candidate* rivalCandidate = nullptr;
      const Reading* rival = nullptr;
      for (const Candidate& c : candidates[ki]) {
        if (!c.interval.contains(heaviest->exponent)) { continue; }
        for (const Reading* r : readings.of(c.subject)) {
          if (r == line.reading || r->failed || r->calls < MIN_CALLS || r->first > line.reading->first) { continue; }
          if (rival && r->pessimistic() >= rival->pessimistic()) { continue; }
          if (!reaches(c, readings, r->options, heaviest->exponent)) { continue; }
          rivalCandidate = &c;
          rival = r;
        }
      }
      if (!rival) { continue; }
      if (rival->pessimistic() > line.cost() * (1 + settings.margin)) { continue; }
      if (rival->isSoaked() && line.reading->isSoaked()) { continue; }
      if (readings.dueled(Readings::keyOf(line.subject, line.reading->options),
                          Readings::keyOf(rivalCandidate->subject, rival->options))) {
        continue;
      }
      DuelPlan plan{
        .kind = pub.kind,
        .exponent = heaviest->exponent,
        .incumbent = {.candidate = rivalCandidate, .options = rival->options, .cost = rival->pessimistic()},
        .challenger = {.candidate = line.candidate, .options = line.reading->options, .cost = line.cost()},
        .weight = weight};
      if (tried.contains(plan.key())) { continue; }
      out = std::move(plan);
    }
  }
  return out;
}

DuelVerdict judge(std::span<const double> logRatios, const DuelSettings& settings) {
  DuelVerdict out;
  size_t const n = logRatios.size();
  out.pairs = u32(n);
  if (n == 0) { return out; }
  double sum = 0;
  for (double const x : logRatios) { sum += x; }
  double const mean = sum / double(n);
  out.ratio = std::exp(mean);
  if (n < 2) {
    out.lo = out.hi = out.ratio;
    out.open = n < settings.pairs;
    return out;
  }
  double ss = 0;
  for (double const x : logRatios) { ss += (x - mean) * (x - mean); }
  double const se = std::max(std::sqrt(ss / double(n - 1) / double(n)), DUEL_SE_FLOOR);
  double const half = t975(n - 1) * se;
  out.lo = std::exp(mean - half);
  out.hi = std::exp(mean + half);

  out.open = false;
  if (out.hi < 1) {
    out.outcome = DuelOutcome::Challenger;
  } else if (out.lo > 1) {
    out.outcome = DuelOutcome::Incumbent;
  } else if (out.lo >= 1 - settings.epsilon && out.hi <= 1 + settings.epsilon) {
    out.outcome = DuelOutcome::Even;
  } else {
    out.open = n < settings.pairs;
  }
  return out;
}

DuelTally duelTally(const TuneDB& db, u32 env) {
  DuelTally out;
  for (const DuelRow& r : db.duels()) {
    if (db.envOf(r.sess) != env) { continue; }
    out.seconds += r.sec;
    switch (r.outcome) {
    case DuelOutcome::Challenger: ++out.challenger; break;
    case DuelOutcome::Incumbent: ++out.incumbent; break;
    case DuelOutcome::Even: ++out.even; break;
    case DuelOutcome::Undecided: ++out.undecided; break;
    }
  }
  return out;
}

std::string toString(const DuelTally& tally) {
  char buf[160];
  snprintf(buf, sizeof(buf), "%u in %.1f min: %u won by the challenger, %u by the incumbent, %u even, %u undecided",
           tally.total(), tally.seconds / 60, tally.challenger, tally.incumbent, tally.even, tally.undecided);
  return buf;
}

double duelTime(const TuneDB& db, u32 env) {
  double out = 0;
  for (const DuelRow& r : db.duels()) {
    if (db.envOf(r.sess) == env) { out += r.sec; }
  }
  return out;
}

bool duelDue(double dueled, double elapsed, const DuelSettings& settings) {
  if (settings.share <= 0) { return false; }
  return settings.share >= 1 || dueled <= settings.share * elapsed;
}

}  // namespace tune
