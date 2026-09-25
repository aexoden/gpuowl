// Copyright (C) Jason Lynch

#include "Accuracy.h"

#include "Args.h"
#include "Bootstrap.h"
#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"
#include "GpuCommon.h"
#include "log.h"
#include "Measure.h"
#include "Primes.h"
#include "Scheduler.h"
#include "TuneDB.h"
#include "Tuner.h"

#include <algorithm>
#include <cinttypes>
#include <deque>
#include <set>

namespace tune {

namespace {

[[nodiscard]] std::vector<FFTConfig> variantsOf(const Env& env, const FFTConfig& family) {
  std::vector<FFTConfig> out{family};
  for (u32 const v : runnableVariants(env, family.shape)) {
    if (v == family.variant) { continue; }
    out.emplace_back(family.shape, v, canonicalCarry(family.shape, v, family.carry));
  }
  return out;
}

[[nodiscard]] std::string pointName(const SweepPoint& p) {
  return p.fft.spec() + " " + (p.key.empty() ? "defaults" : p.text);
}

[[nodiscard]] std::string readingText(const SweepReading& r) {
  char buf[96];
  snprintf(buf, sizeof(buf), "z %6.2f n %5u max %.4f %s", r.z, r.n, r.maxRoe, r.checkOk ? "OK" : "EE");
  return buf;
}

[[nodiscard]] std::string changeText(const SweepPoint& p, const SweepLookup& readingOf) {
  if (!p.background) { return {}; }
  std::optional<SweepReading> const own = readingOf(p.fft, p.exponent, p.config);
  std::optional<SweepReading> const bg = readingOf(p.fft, p.exponent, *p.background);
  Change const c = compare(own, bg);
  if (c != Change::Differs) { return toString(c); }
  char buf[48];
  snprintf(buf, sizeof(buf), "%s %+.2f", toString(c), own->z - bg->z);
  return buf;
}

// Whether the table has a move ("TAIL_KERNELS=0") changing the rounding; COUPLED moves keys it never marks.
[[nodiscard]] bool tableChanges(const std::string& key, const std::string& move) {
  const Option* const option = findOption(key);
  if (!option) { return false; }
  std::optional<int> const value = parseInt<int>(std::string_view{move}.substr(move.find('=') + 1));
  return option->accuracyImpact != AccuracyImpact::None && (!value || option->changesRounding(*value));
}

[[nodiscard]] const char* impactText(std::optional<AccuracyImpact> impact) {
  if (!impact) { return "unread"; }
  switch (*impact) {
  case AccuracyImpact::None: return "none";
  case AccuracyImpact::Suspected: return "?";
  case AccuracyImpact::Yes: return "yes";
  }
  return "?";
}

}  // namespace

u64 sweepExponent(const FFTConfig& fft) {
  static const Primes primes;
  u64 const top = primes.prevPrime(maxExp(fft) + 1);
  return isEligible(fft, top) ? top : 0;
}

std::vector<SweepPoint> sweepPlan(const Env& env, const FFTConfig& family, const std::vector<Group>& groups,
                                  const SweepLookup& readingOf) {
  std::vector<SweepPoint> out;

  // Moves no later variant needs to offer: one read, or for a structural move one read with a passing check, since
  // what it opens is only read against it.
  std::set<std::string> settled;

  std::vector<FFTConfig> const variants = variantsOf(env, family);
  u64 exponent = 0;
  for (const FFTConfig& fft : variants) {
    if (u64 const e = sweepExponent(fft); e && (!exponent || e < exponent)) { exponent = e; }
  }
  if (!exponent) { return out; }

  for (const FFTConfig& fft : variants) {
    if (!isEligible(fft, exponent)) { continue; }
    out.push_back({.fft = fft, .exponent = exponent, .config = {}, .key = {}, .text = {}, .background = {}});

    std::set<std::string> planned = settled;
    std::deque<UseConfig> backgrounds{UseConfig{}};
    while (!backgrounds.empty()) {
      UseConfig const background = std::move(backgrounds.front());
      backgrounds.pop_front();

      std::optional<SweepReading> const read = readingOf(fft, exponent, background);
      if (!read || !read->checkOk) { continue; }

      for (Group const group : groups) {
        for (Move& move : movesWithin(env, fft, background, group)) {
          if (!planned.insert(move.text).second) { continue; }
          const Option* const option = findOption(move.key);
          bool const structural = option && option->structural;
          if (structural) { backgrounds.push_back(move.config); }
          if (std::optional<SweepReading> const own = readingOf(fft, exponent, move.config);
              own && (own->checkOk || !structural)) {
            settled.insert(move.text);
          }
          out.push_back({.fft = fft,
                         .exponent = exponent,
                         .config = std::move(move.config),
                         .key = option ? std::move(move.key) : COUPLED,
                         .text = std::move(move.text),
                         .background = background});
        }
      }
    }
  }
  return out;
}

const char* toString(Change change) {
  switch (change) {
  case Change::Same: return "same";
  case Change::Differs: return "differs";
  case Change::Fails: return "fails its check";
  case Change::Unread: return "unread";
  }
  return "?";
}

Change compare(const std::optional<SweepReading>& point, const std::optional<SweepReading>& background) {
  if (!point || !background || !background->checkOk) { return Change::Unread; }
  if (!point->checkOk) { return Change::Fails; }
  if (!point->fingerprint || !background->fingerprint) { return Change::Unread; }
  return point->fingerprint == background->fingerprint ? Change::Same : Change::Differs;
}

std::optional<AccuracyImpact> KeyVerdict::measured() const {
  if (differs || fails) { return AccuracyImpact::Yes; }
  if (same) { return AccuracyImpact::None; }
  return {};
}

std::vector<KeyVerdict> keyVerdicts(const std::vector<SweepPoint>& points, const SweepLookup& readingOf) {
  std::map<std::string, KeyVerdict> byKey;
  for (const SweepPoint& p : points) {
    if (!p.background) { continue; }
    KeyVerdict& v = byKey[p.key];
    v.key = p.key;

    std::optional<SweepReading> const own = readingOf(p.fft, p.exponent, p.config);
    std::optional<SweepReading> const bg = readingOf(p.fft, p.exponent, *p.background);
    Change const change = compare(own, bg);
    if (change != Change::Unread) { v.changedBy[p.text] |= change != Change::Same; }
    switch (change) {
    case Change::Same: ++v.same; break;
    case Change::Unread: ++v.unread; break;
    case Change::Fails: ++v.fails; break;
    case Change::Differs:
      ++v.differs;
      if (double const dz = own->z - bg->z; v.worstAt.empty() || dz < v.worstDz) {
        v.worstDz = dz;
        v.worstAt = pointName(p);
      }
      break;
    }
  }

  std::vector<KeyVerdict> out;
  for (auto& [key, v] : byKey) { out.push_back(std::move(v)); }
  return out;
}

MeasureOutcome runAccuracy(const GpuCommon& shared, const TuneCommand& command) {
  Args& args = *shared.args;
  fs::path const dir = fs::current_path();

  Takeover const takeover = takeOverConfig(args);
  log("accuracy: %s\n", describe(takeover).c_str());

  RunScope const scope = makeScope(command.scope, scanWorktodo(worktodoFiles(args, dir)));
  Env const env = detectEnv(*shared.context, args);

  std::vector<FFTConfig> families;
  if (!command.fft.empty()) {
    families.emplace_back(command.fft);
  } else {
    std::vector<FFTConfig> inScope;
    for (const Baseline& b : baselines(env, scope)) { inScope.push_back(b.fft); }
    for (const Family& f : bootstrapFamilies(env, scope.probe, inScope)) { families.push_back(f.fft); }
  }
  std::vector<Group> const groups = command.groups.empty() ? allGroups() : command.groups;

  fs::path const dbPath = dir / TuneDB::DEFAULT_NAME;
  TuneDB db;
  if (!db.lockForWriting(dbPath) || !db.load(dbPath)) { return MeasureOutcome::Failed; }
  db.attach(dbPath);

  Session session{shared, db, env};
  if (!session.begin(0)) {
    log("accuracy: '%s' would not take a session\n", dbPath.string().c_str());
    return MeasureOutcome::Failed;
  }

  // What the database holds, as the gate matches it: a reading already taken is not taken again, so a sweep stopped
  // part way resumes, and a fresh one is compared as it was written.  One with no fingerprint cannot be compared
  // exactly, and is taken again.
  Gates gates{db, session.envId(), env};
  SweepLookup const readingOf = [&](const FFTConfig& fft, u64 exponent,
                                    const UseConfig& config) -> std::optional<SweepReading> {
    std::optional<RoeRow> const row = gates.reading(fft, exponent, config, false);
    if (!row || !row->fp) { return {}; }
    return SweepReading{
      .z = row->z, .n = row->n, .maxRoe = row->maxRoe, .checkOk = row->checkOk, .fingerprint = row->fp};
  };

  log("accuracy: over the FFT types in scope at the probe %" PRIu64 " (%s), each read at the top of its range\n",
      scope.probe, scope.probeSource.c_str());

  std::vector<SweepPoint> all;
  for (const FFTConfig& family : families) {
    if (exactArithmetic(family)) {
      log("accuracy: %s: exact arithmetic, nothing rounds\n", family.spec().c_str());
      continue;
    }
    log("accuracy: %s %s\n", typeName(family.shape.fft_type), family.spec().c_str());

    // Whatever would not give a reading -- a build failure, a hold -- is asked for once.
    std::set<std::string> tried;
    std::vector<SweepPoint> plan;
    for (;;) {
      plan = sweepPlan(env, family, groups, readingOf);
      auto const next = std::ranges::find_if(plan, [&](const SweepPoint& p) {
        return !readingOf(p.fft, p.exponent, p.config) && !tried.contains(p.fft.spec() + " " + configText(p.config));
      });
      if (next == plan.end()) { break; }
      tried.insert(next->fft.spec() + " " + configText(next->config));

      session.varying(next->key.empty() ? std::vector<std::string>{} : std::vector<std::string>{next->key});
      RoeCheck const r = session.checkRoe(next->fft, next->config, next->exponent);
      if (session.stopped()) { break; }
      gates = Gates{db, session.envId(), env};

      std::optional<SweepReading> const got = readingOf(next->fft, next->exponent, next->config);
      log("accuracy: [%zu/%zu] %-36s at %" PRIu64 ": %s %s\n", size_t(next - plan.begin()) + 1, plan.size(),
          pointName(*next).c_str(), next->exponent, got ? readingText(*got).c_str() : toString(r.status),
          changeText(*next, readingOf).c_str());
    }
    all.insert(all.end(), plan.begin(), plan.end());
    if (session.stopped()) { break; }
  }

  log("accuracy: each variant at its defaults\n");
  for (const SweepPoint& p : all) {
    if (p.background) { continue; }
    std::optional<SweepReading> const r = readingOf(p.fft, p.exponent, p.config);
    log("accuracy:   %-24s at %" PRIu64 " (%.2f bpw): %s\n", p.fft.spec().c_str(), p.exponent,
        double(p.exponent) / double(p.fft.size()), r ? readingText(*r).c_str() : "unread");
  }

  log("accuracy: each key, against the set it moved from (same = bit-identical rounding)\n");
  u32 disagree = 0;
  for (const KeyVerdict& v : keyVerdicts(all, readingOf)) {
    std::string changed;
    std::string otherwise;
    for (const auto& [move, did] : v.changedBy) {
      if (did) { changed += (changed.empty() ? "" : " ") + move; }
      if (did != tableChanges(v.key, move)) { otherwise += (otherwise.empty() ? "" : " ") + move; }
    }
    if (!otherwise.empty()) { ++disagree; }

    char worst[128] = "";
    if (v.differs) { snprintf(worst, sizeof(worst), ", worst %+.2f at %s", v.worstDz, v.worstAt.c_str()); }
    const Option* const option = findOption(v.key);
    log("accuracy:   %-22s %-6s (table: %-4s) same %u, differs %u, fails %u, unread %u%s\n", v.key.c_str(),
        impactText(v.measured()), impactText(option ? option->accuracyImpact : AccuracyImpact::None), v.same, v.differs,
        v.fails, v.unread, worst);
    if (!changed.empty()) { log("accuracy:     changes the rounding: %s\n", changed.c_str()); }
    if (!otherwise.empty()) { log("accuracy:     the table says otherwise of: %s\n", otherwise.c_str()); }
  }
  log("accuracy: %u key(s) read otherwise than the table has them, value by value\n", disagree);

  if (session.deviceLost()) { return MeasureOutcome::DeviceLost; }
  session.end();
  return session.cannotRecord() ? MeasureOutcome::Failed : MeasureOutcome::Ok;
}

}  // namespace tune
