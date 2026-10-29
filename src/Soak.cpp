// Copyright (C) Jason Lynch

#include "Soak.h"

#include <algorithm>
#include <map>

namespace tune {

std::string SoakPlan::key() const {
  std::string out = std::to_string(exponent);
  for (const SoakTarget& t : targets) { out += ' ' + t.candidate->subject.fft.spec() + ':' + configText(t.options); }
  return out;
}

std::optional<SoakPlan> planSoak(const Publication& publication, const std::vector<std::vector<Candidate>>& candidates,
                                 const Readings& readings, const SoakSettings& settings,
                                 const std::set<std::string>& tried) {
  if (settings.share <= 0 || settings.top == 0) { return {}; }

  struct Option {
    double weight = 0;
    SoakPlan plan;
  };
  std::vector<Option> options;
  for (size_t ki = 0; ki < publication.kinds.size() && ki < candidates.size(); ++ki) {
    const KindPublication& pub = publication.kinds[ki];
    if (!pub.tuned) { continue; }
    for (const Line& line : pub.lines) {
      if (!line.candidate) { continue; }
      // The weight the line serves, and its heaviest exponent.
      double weight = 0;
      const GridPoint* heaviest = nullptr;
      for (const GridPoint& p : line.candidate->points) {
        if (pub.at(p.exponent) != &line) { continue; }
        weight += p.weight;
        if (!heaviest || p.weight > heaviest->weight) { heaviest = &p; }
      }
      if (!heaviest) { continue; }

      // Every concluded configuration that holds that exponent, within the margin of the line.
      std::vector<SoakTarget> rivals;
      for (const Candidate& c : candidates[ki]) {
        if (!c.interval.contains(heaviest->exponent)) { continue; }
        for (const Reading* r : readings.of(c.subject)) {
          if (r->failed || r->calls < MIN_CALLS) { continue; }
          if (r->pessimistic() > line.cost() * (1 + settings.margin)) { continue; }
          rivals.push_back({.candidate = &c, .options = r->options, .cost = r->pessimistic()});
        }
      }
      std::ranges::stable_sort(rivals, {}, &SoakTarget::cost);
      if (rivals.size() > settings.top) { rivals.resize(settings.top); }

      bool fresh = false;
      for (const SoakTarget& t : rivals) {
        const Reading* const r = readings.find(t.candidate->subject, t.options);
        fresh = fresh || !r || !r->isSoaked();
      }
      if (!fresh) { continue; }
      SoakPlan plan{.kind = pub.kind, .exponent = heaviest->exponent, .targets = std::move(rivals)};
      if (tried.contains(plan.key())) { continue; }
      options.push_back({.weight = weight, .plan = std::move(plan)});
    }
  }
  if (options.empty()) { return {}; }
  return std::ranges::max(options, {}, &Option::weight).plan;
}

double soakTime(const TuneDB& db, u32 env) {
  double out = 0;
  for (const SoakRow& r : db.soaks()) {
    if (db.envOf(r.sess) == env) { out += r.sec; }
  }
  return out;
}

bool soakDue(double soaked, double elapsed, const SoakPlan& plan, const SoakSettings& settings) {
  if (settings.share <= 0 || plan.targets.empty()) { return false; }
  double const length = settings.length(plan.targets.size());
  return soaked + length <= settings.share * (elapsed + length);
}

}  // namespace tune
