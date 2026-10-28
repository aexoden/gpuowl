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

std::vector<Finalist> finalists(const AcquireContext& context, const Publication& publication, double target) {
  std::vector<Finalist> out;
  auto const add = [&](const Candidate* c, const UseConfig& options, bool rival) {
    bool const known = std::ranges::any_of(out, [&](const Finalist& f) {
      return f.candidate->subject.fft.spec() == c->subject.fft.spec() && f.candidate->subject.kind == c->subject.kind &&
        f.options == options;
    });
    if (!known) { out.push_back({.candidate = c, .options = options, .rival = rival}); }
  };
  for (size_t k = 0; k < context.scope.grids.size() && k < context.candidates.size(); ++k) {
    const KindPublication* const kind = publication.of(context.scope.grids[k].kind);
    if (!kind) { continue; }
    // The candidates read at all: a few hundred of thousands.
    std::vector<std::pair<const Candidate*, std::vector<const Reading*>>> read;
    for (const Candidate& c : context.candidates[k]) {
      std::vector<const Reading*> of = context.readings.of(c.subject);
      std::erase_if(of, [](const Reading* r) { return r->failed || r->calls == 0; });
      if (!of.empty()) { read.emplace_back(&c, std::move(of)); }
    }
    for (const Line& line : kind->lines) {
      // A line of an FFT the run does not hold cannot be called by it.
      if (!line.candidate) { continue; }
      add(line.candidate, line.reading->options, false);
      // Its closest rival over the points it holds.
      const Candidate* rival = nullptr;
      const Reading* rivalReading = nullptr;
      double closest = 1 + 2 * target;
      for (const GridPoint& p : context.scope.grids[k].points) {
        if (kind->at(p.exponent) != &line) { continue; }
        for (const auto& [c, of] : read) {
          if (!c->interval.contains(p.exponent)) { continue; }
          for (const Reading* r : of) {
            if (r == line.reading) { continue; }
            if (double const ratio = r->mean / line.reading->mean; ratio < closest) {
              closest = ratio;
              rival = c;
              rivalReading = r;
            }
          }
        }
      }
      if (rival) { add(rival, rivalReading->options, true); }
    }
  }
  // The lines first: a rival already a line is confirmed as one.
  std::ranges::stable_partition(out, [](const Finalist& f) { return !f.rival; });
  return out;
}

FreshCalls freshCalls(const TuneDB& db, u32 env) {
  // A confirmation item's call is the next call of its session of the same set.
  std::map<u32, std::vector<const CallRow*>> bySession;
  for (const CallRow& call : db.calls()) {
    if (db.envOf(call.sess) == env) { bySession[call.sess].push_back(&call); }
  }
  FreshCalls out;
  std::map<std::tuple<std::string, TestKind, u32>, double> sums;
  for (const ItemRow& item : db.items()) {
    if (db.envOf(item.sess) != env || item.lane != CONFIRM) { continue; }
    const std::vector<const CallRow*>& calls = bySession[item.sess];
    auto it = std::ranges::lower_bound(calls, item.ts, {}, &CallRow::ts);
    for (; it != calls.end(); ++it) {
      if ((*it)->fft == item.fft && (*it)->kind == item.kind && (*it)->cfg == item.cfg) { break; }
    }
    if (it == calls.end()) { continue; }
    auto const key = std::tuple{item.fft, item.kind, item.cfg};
    ++out[key].calls;
    if ((*it)->ok()) { sums[key] += (*it)->cost(); }
  }
  for (auto& [key, fresh] : out) { fresh.mean = sums[key] / fresh.calls; }
  return out;
}

Fresh freshOf(const FreshCalls& fresh, const TuneDB& db, const std::string& fft, TestKind kind,
              const UseConfig& options) {
  auto const it = fresh.find({fft, kind, db.findCfgId(options)});
  return it == fresh.end() ? Fresh{} : it->second;
}

std::optional<Proposal> endStep(const AcquireContext& context, const Publication& publication, TuneDB& db, u32 env,
                                u32 sess, u64 now, const ExploreSettings& settings, double elapsed, double budget,
                                double target, std::mt19937_64& rng) {
  // Fresh calls of what is published, and of each line's closest rival: the least confirmed first.
  std::optional<Finalist> least;
  u32 leastShort = 0;  // calls owed
  u32 leastCalls = 0;
  FreshCalls const fresh = freshCalls(db, env);
  for (const Finalist& f : finalists(context, publication, target)) {
    u32 const owed = f.rival ? RIVAL_CALLS : CONFIRM_CALLS;
    u32 const calls = freshOf(fresh, db, f.candidate->subject.fft.spec(), f.candidate->subject.kind, f.options).calls;
    if (calls < owed && (!least || calls < leastCalls)) {
      least = f;
      leastShort = owed;
      leastCalls = calls;
    }
  }
  if (least) {
    const Candidate& c = *least->candidate;
    double const mean = std::exp(context.model.predict(context.model.mean(), c.subject, least->options));
    return Proposal{.candidate = &c,
                    .options = least->options,
                    .lane = Lane::Model,
                    .what = least->rival ? "rival" : "line",
                    .packet = 0,
                    .ordinal = leastCalls + 1,
                    .predictedUs = mean,
                    .why = std::string{"confirming "} +
                      (least->rival ? "a published line's closest rival" : "a published line") + ", fresh call " +
                      std::to_string(leastCalls + 1) + " of " + std::to_string(leastShort),
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
      auto const it = firstLane.find({line.subject.fft.spec(), configText(line.reading->options)});
      if (it != firstLane.end() && it->second == AUDIT) { out.found.push_back(&line); }
    }
  }

  FreshCalls const fresh = freshCalls(db, env);
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.lines) {
      out.confirmed.push_back(
        {.line = &line,
         .fresh = freshOf(fresh, db, line.subject.fft.spec(), kind.kind, line.reading->options)});
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
  for (const AuditSummary::Confirmed& c : s.confirmed) {
    if (c.fresh.calls == 0) { continue; }
    out.push_back("confirmed: " + c.line->subject.fft.spec() + " " + usText(c.line->reading->mean) +
                  " us/it over all its calls, " + usText(c.fresh.mean) + " over " + std::to_string(c.fresh.calls) +
                  " fresh ones (" + percentText(c.fresh.mean / c.line->reading->mean - 1) + ")");
  }
  for (const Line* line : s.found) {
    out.push_back("audit: found " + line->subject.fft.spec() + " -use " +
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
