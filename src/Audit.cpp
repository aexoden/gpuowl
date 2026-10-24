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
    // A campaign has one audit, so every audit item is its: no clock decides it, since sessions' clocks need not agree.
    if (db.envOf(r.sess) == env && r.lane == AUDIT && r.packet) { trials[r.packet].push_back(&r); }
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
    // A call's time goes with its cost: the trials' draws are scaled from what the campaign's calls cost so far.
    size_t items = 0;
    for (const ItemRow& r : db.items()) { items += db.envOf(r.sess) == env; }
    double const perItem = laneTime(db, env).total() / double(std::max<size_t>(1, items));
    double paid = 0;
    double calls = 0;
    for (const auto& [key, r] : context.readings.all()) {
      if (!r.failed && r.calls > 0) {
        paid += r.mean * r.calls;
        calls += r.calls;
      }
    }
    double const drawn = expectedStartCost(context, settings, elapsed);
    double const scale = calls > 0 && drawn > 0 ? std::max(1.0, drawn / (paid / calls)) : 1;
    u32 const trials = u32(std::max(0.0, std::floor((budget - elapsed) / (RESTART_PACKET * perItem * scale))));
    (void)db.add(AuditRow{.sess = sess, .trials = trials, .baseline = publication.T(), .target = target, .ts = now});
    log("tune: audit: %u trials of %u calls declared for the %.0f minutes left (a trial's call expected to take %.1fx "
        "the campaign's so far), against T %s us/it\n",
        trials, RESTART_PACKET, (budget - elapsed) / 60, scale, usText(publication.T()).c_str());
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

const char* toString(AuditSummary::State state) {
  switch (state) {
  case AuditSummary::State::None: return "none";
  case AuditSummary::State::Running: return "running";
  case AuditSummary::State::FoundGains: return "found gains";
  case AuditSummary::State::Exhausted: return "budget exhausted";
  case AuditSummary::State::NoTime: return "no time";
  case AuditSummary::State::Complete: return "complete";
  }
  return "?";
}

AuditSummary auditSummary(const TuneDB& db, u32 env, const Readings& readings, const Publication& publication,
                          bool budgetSpent) {
  AuditState const state = auditState(db, env, readings);
  AuditSummary out{.now = publication.T()};
  if (!state.declared) { return out; }
  const AuditRow& a = *state.declared;
  out.declared = true;
  out.trials = a.trials;
  out.done = state.done;
  out.failedStarts = state.failedStarts;
  out.baseline = a.baseline;
  out.target = a.target;
  out.gain = a.baseline > 0 ? 1 - out.now / a.baseline : 0;

  // The published lines an audit trial read first.
  std::map<std::pair<std::string, std::string>, std::string> firstLane;
  for (const ItemRow& r : db.items()) {
    const UseConfig* const options = db.findCfg(r.cfg);
    if (db.envOf(r.sess) == env && options && r.lane != DUTY) {
      firstLane.try_emplace({r.fft, configText(*options)}, r.lane);
    }
  }
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.lines) {
      auto const it = firstLane.find({line.candidate->subject.fft.spec(), configText(line.reading->options)});
      if (it != firstLane.end() && it->second == AUDIT) { out.found.push_back(&line); }
    }
  }

  bool const complete = state.done >= a.trials;
  if (!out.found.empty() && out.gain > a.target) {
    out.state = AuditSummary::State::FoundGains;
  } else if (!complete) {
    out.state = budgetSpent ? AuditSummary::State::Exhausted : AuditSummary::State::Running;
  } else if (a.trials == 0) {
    out.state = AuditSummary::State::NoTime;
  } else {
    out.state = AuditSummary::State::Complete;
    out.bound = 1 - std::pow(0.05, 1.0 / a.trials);
  }
  return out;
}

std::vector<std::string> auditReport(const AuditSummary& s, bool budgetSpent) {
  if (!s.declared) {
    return {std::string{"audit: none; "} +
            (budgetSpent ? "the budget ended before what is published was confirmed" : "not yet begun")};
  }
  std::vector<std::string> out;
  out.push_back("audit: " + std::to_string(s.trials) + " trials declared against T " + usText(s.baseline) + "; " +
                std::to_string(s.done) + " done" +
                (s.failedStarts ? " (" + std::to_string(s.failedStarts) + " starts could not be read)" : ""));
  for (const Line* line : s.found) {
    out.push_back("audit: found " + line->candidate->subject.fft.spec() + " -use " +
                  configText(line->reading->options) + " at " + usText(line->cost()) + " us/it");
  }
  // What moved T since the baseline: the trials' finds, or the search that went on around and after them.
  out.push_back("audit: T now " + usText(s.now) + ", " + percentText(s.gain) + " below the baseline" +
                (s.gain > 0 && s.found.empty() ? ", none of it found by a trial" : "") + "; the target is " +
                percentText(s.target));
  switch (s.state) {
  case AuditSummary::State::FoundGains:
    out.push_back(std::string{"audit: found gains"} +
                  (s.done < s.trials ? ", the budget spent before every trial ran" : "") +
                  ": the search had not converged");
    break;
  case AuditSummary::State::Running:
    out.push_back("audit: running, " + std::to_string(s.done) + " of " + std::to_string(s.trials) + " trials run");
    break;
  case AuditSummary::State::Exhausted:
    out.push_back("audit: budget exhausted, " + std::to_string(s.done) + " of " + std::to_string(s.trials) +
                  " trials run");
    break;
  case AuditSummary::State::NoTime: out.push_back("audit: no time was left for a trial"); break;
  case AuditSummary::State::Complete:
    out.push_back("audit: complete: no trial improved the workload by more than the target; with 95% confidence one "
                  "trial of this generator does so with probability below " +
                  percentText(s.bound) +
                  ".  Not a claim of an optimum: a narrow enough basin escapes any number of trials");
    break;
  case AuditSummary::State::None: break;
  }
  return out;
}

}  // namespace tune
