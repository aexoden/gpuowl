// Copyright (C) Jason Lynch

#include "Audit.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace tune {

namespace {

constexpr const char* CONFIRM = "confirm";
constexpr const char* AUDIT = "audit";

// The reading of `options` on `fft` for `kind`, whatever its regime; nullptr where there is none.
const Reading* readingOf(const Readings& readings, const std::string& fft, TestKind kind, const UseConfig& options) {
  std::string const text = configText(options);
  for (const auto& [key, r] : readings.all()) {
    if (key.fft == fft && key.kind == kind && key.options == text) { return &r; }
  }
  return nullptr;
}

std::string usText(double us) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", us);
  return buf;
}

std::string percentText(double fraction) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f%%", fraction * 100);
  return buf;
}

}  // namespace

AuditState auditState(const TuneDB& db, u32 env, const Readings& readings) {
  AuditState out;
  for (const AuditRow& a : db.audits()) {
    if (db.envOf(a.sess) == env) { out.declared = a; }
  }
  if (!out.declared) { return out; }

  std::map<u32, std::vector<const ItemRow*>> trials;
  for (const ItemRow& r : db.items()) {
    if (db.envOf(r.sess) == env && r.lane == AUDIT && r.ts >= out.declared->ts && r.packet) {
      trials[r.packet].push_back(&r);
    }
  }
  out.started = u32(trials.size());
  for (auto it = trials.begin(); it != trials.end(); ++it) {
    const ItemRow& start = *it->second.front();
    const UseConfig* const options = db.findCfg(start.cfg);
    const Reading* const r = options ? readingOf(readings, start.fft, start.kind, *options) : nullptr;
    bool const failed = r && (r->failed || r->calls == 0);
    out.failedStarts += failed;
    // A trial is done with all its steps, with a start that could not be read, or once a later one has begun.
    bool const last = std::next(it) == trials.end();
    out.done += !last || failed || it->second.size() >= RESTART_PACKET;
  }
  return out;
}

std::optional<Proposal> endStep(const AcquireContext& context, const Publication& publication, TuneDB& db, u32 env,
                                u32 sess, u64 now, const ExploreSettings& settings, double elapsed, double budget,
                                double target, std::mt19937_64& rng) {
  // Fresh calls of what is published, the least confirmed line first.
  const Line* least = nullptr;
  u32 leastCalls = CONFIRM_CALLS;
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.lines) {
      std::string const text = configText(line.reading->options);
      u32 calls = 0;
      for (const ItemRow& r : db.items()) {
        const UseConfig* const options = db.findCfg(r.cfg);
        calls += db.envOf(r.sess) == env && r.lane == CONFIRM && r.fft == line.candidate->subject.fft.spec() &&
          r.kind == kind.kind && options && configText(*options) == text;
      }
      if (calls < leastCalls) {
        least = &line;
        leastCalls = calls;
      }
    }
  }
  if (least) {
    const Candidate& c = *least->candidate;
    double const mean = std::exp(context.model.predict(context.model.mean(), c.subject, least->reading->options));
    return Proposal{.candidate = &c,
                    .options = least->reading->options,
                    .lane = Lane::Model,
                    .what = "line",
                    .packet = 0,
                    .ordinal = leastCalls + 1,
                    .predictedUs = mean,
                    .why = "confirming a published line, fresh call " + std::to_string(leastCalls + 1) + " of " +
                      std::to_string(CONFIRM_CALLS),
                    .record = CONFIRM};
  }

  AuditState const state = auditState(db, env, context.readings);
  if (!state.declared) {
    // As many trials as the time left holds at the campaign's pace so far, declared now and not changed after.
    size_t items = 0;
    for (const ItemRow& r : db.items()) { items += db.envOf(r.sess) == env; }
    double const perItem = laneTime(db, env).total() / double(std::max<size_t>(1, items));
    u32 const trials = u32(std::max(0.0, std::floor((budget - elapsed) / (RESTART_PACKET * perItem))));
    (void)db.add(AuditRow{.sess = sess, .trials = trials, .baseline = publication.T(), .target = target, .ts = now});
    log("tune: audit: %u trials of %u calls declared for the %.0f minutes left, against T %s us/it\n", trials,
        RESTART_PACKET, (budget - elapsed) / 60, usText(publication.T()).c_str());
    if (trials == 0) { return {}; }
  } else if (state.done >= state.declared->trials) {
    return {};
  }

  std::optional<Proposal> step = auditTrialStep(context, settings, elapsed, db, env, rng);
  // No trial beyond those declared.
  AuditState const now_ = auditState(db, env, context.readings);
  if (step && step->ordinal == 1 && now_.declared && now_.started >= now_.declared->trials) { return {}; }
  return step;
}

std::vector<std::string> auditReport(const TuneDB& db, u32 env, const Readings& readings,
                                     const Publication& publication, bool budgetSpent) {
  AuditState const state = auditState(db, env, readings);
  if (!state.declared) {
    return {std::string{"audit: none; "} +
            (budgetSpent ? "the budget ended before what is published was confirmed" : "not yet begun")};
  }
  const AuditRow& a = *state.declared;
  double const now = publication.T();
  double const gain = a.baseline > 0 ? 1 - now / a.baseline : 0;
  std::vector<std::string> out;
  out.push_back("audit: " + std::to_string(a.trials) + " trials declared against T " + usText(a.baseline) + "; " +
                std::to_string(state.done) + " done" +
                (state.failedStarts ? " (" + std::to_string(state.failedStarts) + " starts could not be read)" : ""));
  out.push_back("audit: T now " + usText(now) + ", " + percentText(gain) + " below the baseline; the target is " +
                percentText(a.target));

  // The published lines an audit trial read first.
  std::map<std::pair<std::string, std::string>, std::string> firstLane;
  for (const ItemRow& r : db.items()) {
    const UseConfig* const options = db.findCfg(r.cfg);
    if (db.envOf(r.sess) == env && options && r.lane != DUTY) {
      firstLane.try_emplace({r.fft, configText(*options)}, r.lane);
    }
  }
  u32 found = 0;
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.lines) {
      auto const it = firstLane.find({line.candidate->subject.fft.spec(), configText(line.reading->options)});
      if (it != firstLane.end() && it->second == AUDIT) {
        ++found;
        out.push_back("audit: found " + line.candidate->subject.fft.spec() + " -use " +
                      configText(line.reading->options) + " at " + usText(line.cost()) + " us/it");
      }
    }
  }

  bool const complete = state.done >= a.trials;
  if (found > 0 && gain > a.target) {
    out.push_back(std::string{"audit: found gains"} + (complete ? "" : ", the budget spent before every trial ran") +
                  ": the search had not converged");
  } else if (!complete) {
    out.push_back("audit: budget exhausted, " + std::to_string(state.done) + " of " + std::to_string(a.trials) +
                  " trials run");
  } else if (a.trials == 0) {
    out.push_back("audit: no time was left for a trial");
  } else {
    double const bound = 1 - std::pow(0.05, 1.0 / a.trials);
    out.push_back("audit: complete: no trial improved the workload by more than the target; with 95% confidence one "
                  "trial of this generator does so with probability below " +
                  percentText(bound) +
                  ".  Not a claim of an optimum: a narrow enough basin escapes any number of trials");
  }
  return out;
}

}  // namespace tune
