// Copyright (C) Jason Lynch

#include "Accuracy.h"

#include "Args.h"
#include "Axes.h"
#include "Candidates.h"
#include "Eligibility.h"
#include "FFTVariants.h"
#include "GpuCommon.h"
#include "log.h"
#include "Measure.h"
#include "Primes.h"
#include "Prior.h"
#include "Readings.h"
#include "TuneCommand.h"
#include "Tuner.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <tuple>

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

// How a move read against its background, at its own exponent.
[[nodiscard]] Change changeOf(const SweepPoint& p, const SweepLookup& readingOf) {
  return compare(readingOf(p.fft, p.exponent, p.config), readingOf(p.fft, p.exponent, *p.background));
}

// How many distinct values a reading's samples took and how many were the largest, where it was just taken.
[[nodiscard]] std::string levelsText(const RoeCheck& r) {
  if (r.status != Status::Ok || !r.applicable || !r.n) { return {}; }
  char buf[64];
  snprintf(buf, sizeof(buf), " (%u levels, %u at max)", r.levels, r.atMax);
  return buf;
}

// " +1.20 -0.35 ...: mean -0.40, sd 0.90" of a run of z or gaps; no sd under two values.
[[nodiscard]] std::string spreadText(const std::vector<double>& xs, const char* format) {
  std::string out;
  char buf[48];
  double sum = 0;
  for (double const x : xs) {
    snprintf(buf, sizeof(buf), format, x);
    out += buf;
    sum += x;
  }
  if (xs.empty()) { return out + " unread"; }
  double const mean = sum / double(xs.size());
  double squares = 0;
  for (double const x : xs) { squares += (x - mean) * (x - mean); }
  if (xs.size() < 2) {
    snprintf(buf, sizeof(buf), ": mean %+.2f, sd -", mean);
  } else {
    snprintf(buf, sizeof(buf), ": mean %+.2f, sd %.2f", mean, std::sqrt(squares / double(xs.size() - 1)));
  }
  return out + buf;
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

// One FFT of each type in scope that rounds: the cheapest read at the probe where the type has a reading there, else
// the cheapest by its prior -- what the search would tune first.
[[nodiscard]] std::vector<FFTConfig> familiesIn(const Env& env, const RunScope& scope, const Readings& readings) {
  std::map<enum FFT_TYPES, std::pair<double, FFTConfig>> measured, prior;
  for (const Candidate& c : candidatesOf(env, scope, TestKind::PRP)) {
    if (!c.interval.contains(scope.probe) || exactArithmetic(c.subject.fft)) { continue; }
    enum FFT_TYPES const type = c.subject.fft.shape.fft_type;
    for (const Reading* r : readings.of(c.subject)) {
      if (r->failed || r->calls == 0) { continue; }
      auto const [at, fresh] = measured.try_emplace(type, r->mean, c.subject.fft);
      if (!fresh && r->mean < at->second.first) { at->second = {r->mean, c.subject.fft}; }
    }
    double const cost = statedPrior(c.subject.fft.shape);
    auto const [at, fresh] = prior.try_emplace(type, cost, c.subject.fft);
    if (!fresh && cost < at->second.first) { at->second = {cost, c.subject.fft}; }
  }
  std::vector<FFTConfig> out;
  for (const auto& [type, p] : prior) {
    auto const m = measured.find(type);
    out.push_back(m != measured.end() ? m->second.second : p.second);
  }
  return out;
}

}  // namespace

std::vector<Move> movesWithin(const Env& env, const FFTConfig& fft, const UseConfig& background, Group group) {
  UseConfig const from = canonicalConfig(env, fft, background);

  std::vector<Move> out;
  std::set<std::string> seen{configText(from)};
  for (const Axis& axis : axesOf(env, fft, from)) {
    // A register limit caps what a kernel may use, not what it computes: every one of its values would be a reading.
    if (axis.option->group != group || Args::isRegisterKey(axis.option->key)) { continue; }

    for (size_t i = 0; i < axis.values.size(); ++i) {
      if (i == axis.current) { continue; }
      UseConfig moved = from;
      place(moved, axis, i);

      std::string text;
      for (const std::string& key : axis.keys()) { text += (text.empty() ? "" : ",") + key + "=" + moved[key]; }

      // A coupled class moves two keys, so a change there is not pinned on either.
      UseConfig config = canonicalConfig(env, fft, moved);
      if (seen.insert(configText(config)).second) {
        out.push_back(
          {.config = std::move(config), .key = axis.coupled ? std::string{} : axis.option->key, .text = std::move(text)});
      }
    }
  }
  return out;
}

SweepLookup sweepLookup(const TuneDB& db, u32 env, const Env& device) {
  using Key = std::tuple<std::string, u64, std::string>;
  auto latest = std::make_shared<std::map<Key, SweepReading>>();
  for (const RoeRow& row : db.roes()) {
    if (db.envOf(row.sess) != env || !row.fp) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const recorded = db.findCfg(row.cfg);
    if (!fft || !recorded) { continue; }
    (*latest)[{fft->spec(), row.exponent, configText(canonicalConfig(device, *fft, *recorded))}] =
      SweepReading{.z = row.z, .n = row.n, .maxRoe = row.maxRoe, .checkOk = row.checkOk, .fingerprint = row.fp};
  }
  return [latest, device](const FFTConfig& fft, u64 exponent, const UseConfig& config) -> std::optional<SweepReading> {
    auto const it = latest->find({fft.spec(), exponent, configText(canonicalConfig(device, fft, config))});
    if (it == latest->end()) { return {}; }
    return it->second;
  };
}

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

std::vector<u64> repeatExponents(const FFTConfig& fft, u64 exponent, u32 k) {
  static const Primes primes;
  std::vector<u64> out{exponent};
  for (u64 e = primes.prevPrime(exponent); out.size() < k && e && isEligible(fft, e); e = primes.prevPrime(e)) {
    out.push_back(e);
  }
  return out;
}

std::vector<SweepPoint> repeatPlan(const std::vector<SweepPoint>& points, const SweepLookup& readingOf, u32 k) {
  std::vector<SweepPoint> out;
  for (const SweepPoint& p : points) {
    if (!p.background || changeOf(p, readingOf) != Change::Differs) { continue; }
    for (u64 const e : repeatExponents(p.fft, p.exponent, k)) {
      if (e == p.exponent) { continue; }
      SweepPoint again = p;
      again.exponent = e;
      out.push_back(std::move(again));
    }
  }
  return out;
}

std::vector<double> gapsOf(const SweepPoint& point, const std::vector<u64>& exponents, const SweepLookup& readingOf) {
  std::vector<double> out;
  if (!point.background) { return out; }
  for (u64 const e : exponents) {
    std::optional<SweepReading> const own = readingOf(point.fft, e, point.config);
    std::optional<SweepReading> const bg = readingOf(point.fft, e, *point.background);
    if (own && bg && own->checkOk && bg->checkOk) { out.push_back(own->z - bg->z); }
  }
  return out;
}

bool tableChanges(const std::string& key, const std::string& move) {
  const Option* const option = findOption(key);
  if (!option) { return false; }
  std::optional<int> const value = parseInt<int>(std::string_view{move}.substr(move.find('=') + 1));
  return option->accuracyImpact != AccuracyImpact::None && (!value || option->changesRounding(*value));
}

RunOutcome runAccuracy(GpuCommon shared, const TuneCommand& command) {
  const Args& args = *shared.args;
  std::optional<RunScope> const scope = scopeFor(command, args);
  if (!scope) { return RunOutcome::Failed; }
  reportScope(*scope);

  fs::path const path = fs::current_path() / TuneDB::DEFAULT_NAME;
  TuneDB db;
  if (!db.lockForWriting(path) || !db.load(path)) { return RunOutcome::Failed; }
  db.attach(path);

  Env const card = detectEnv(*shared.context, args);
  Bench bench{shared, db, card};
  if (!bench.begin(scope->probe, "accuracy", {}, command.tag)) {
    log("accuracy: could not open a session in %s\n", path.string().c_str());
    return RunOutcome::Failed;
  }

  std::vector<FFTConfig> families;
  if (!command.accuracyFft.empty()) {
    families.push_back(*parseFft(command.accuracyFft));
  } else {
    families = familiesIn(card, *scope, Readings{db, bench.envId(), card});
  }
  std::vector<Group> const groups = command.accuracyGroups.empty() ? allGroups() : command.accuracyGroups;

  // What the database holds: a reading already taken is not taken again, so a sweep stopped part way resumes.
  SweepLookup readingOf = sweepLookup(db, bench.envId(), card);

  std::vector<SweepPoint> all;
  for (const FFTConfig& family : families) {
    if (exactArithmetic(family)) {
      log("accuracy: %s: exact arithmetic, nothing rounds\n", family.spec().c_str());
      continue;
    }
    log("accuracy: %s\n", family.spec().c_str());

    // Whatever would not give a reading -- a build failure, a hold -- is asked for once.
    std::set<std::string> tried;
    std::vector<SweepPoint> plan;
    for (;;) {
      plan = sweepPlan(card, family, groups, readingOf);
      auto const next = std::ranges::find_if(plan, [&](const SweepPoint& p) {
        return !readingOf(p.fft, p.exponent, p.config) && !tried.contains(p.fft.spec() + " " + configText(p.config));
      });
      if (next == plan.end()) { break; }
      tried.insert(next->fft.spec() + " " + configText(next->config));

      RoeCheck const r = bench.roe(next->fft, next->config, next->exponent);
      if (bench.stopped()) { break; }
      readingOf = sweepLookup(db, bench.envId(), card);

      std::optional<SweepReading> const got = readingOf(next->fft, next->exponent, next->config);
      log("accuracy: [%zu/%zu] %-36s at %" PRIu64 ": %s%s %s\n", size_t(next - plan.begin()) + 1, plan.size(),
          pointName(*next).c_str(), next->exponent, got ? readingText(*got).c_str() : toString(r.status),
          levelsText(r).c_str(), changeText(*next, readingOf).c_str());
    }
    all.insert(all.end(), plan.begin(), plan.end());
    if (bench.stopped()) { break; }
  }

  // Each move that changed the rounding, read again beside its background a little further down: two roundings of the
  // same data read apart by their samples as well as by their arithmetic, and only the spread tells which.
  u32 const repeat = command.accuracyRepeat;
  if (repeat > 1 && !bench.stopped()) {
    std::vector<SweepPoint> const again = repeatPlan(all, readingOf, repeat);
    std::set<std::string> tried;
    for (size_t i = 0; i < again.size() && !bench.stopped(); ++i) {
      const SweepPoint& p = again[i];
      for (bool const own : {true, false}) {
        const UseConfig& config = own ? p.config : *p.background;
        std::string const id = p.fft.spec() + " " + std::to_string(p.exponent) + " " + configText(config);
        if (readingOf(p.fft, p.exponent, config) || !tried.insert(id).second) { continue; }
        RoeCheck const r = bench.roe(p.fft, config, p.exponent);
        if (bench.stopped()) { break; }
        readingOf = sweepLookup(db, bench.envId(), card);

        std::optional<SweepReading> const got = readingOf(p.fft, p.exponent, config);
        std::string const name = p.fft.spec() + " " + (own ? p.text : "beside " + p.text);
        log("accuracy: again [%zu/%zu] %-36s at %" PRIu64 ": %s%s\n", i + 1, again.size(), name.c_str(), p.exponent,
            got ? readingText(*got).c_str() : toString(r.status), levelsText(r).c_str());
      }
    }
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

    char worst[160] = "";
    if (v.differs) { snprintf(worst, sizeof(worst), ", worst %+.2f at %s", v.worstDz, v.worstAt.c_str()); }
    const Option* const option = findOption(v.key);
    log("accuracy:   %-22s %-6s (table: %-4s) same %u, differs %u, fails %u, unread %u%s\n", v.key.c_str(),
        impactText(v.measured()), impactText(option ? option->accuracyImpact : AccuracyImpact::None), v.same, v.differs,
        v.fails, v.unread, worst);
    if (!changed.empty()) { log("accuracy:     changes the rounding: %s\n", changed.c_str()); }
    if (!otherwise.empty()) { log("accuracy:     the table says otherwise of: %s\n", otherwise.c_str()); }
  }
  log("accuracy: %u key(s) read otherwise than the table has them, value by value\n", disagree);

  if (repeat > 1) {
    auto differs = [&](const SweepPoint& p) { return p.background && changeOf(p, readingOf) == Change::Differs; };
    log("accuracy: each move that changes the rounding, at up to %u exponents (its z less its background's)\n", repeat);
    for (const SweepPoint& p : all) {
      if (!differs(p)) { continue; }
      std::vector<double> const gaps = gapsOf(p, repeatExponents(p.fft, p.exponent, repeat), readingOf);
      log("accuracy:   %-40s%s\n", pointName(p).c_str(), spreadText(gaps, " %+.2f").c_str());
    }
    log("accuracy: each background's own z at the same exponents: its spread is one reading's sampling noise\n");
    std::set<std::string> shown;
    for (const SweepPoint& p : all) {
      if (!differs(p)) { continue; }
      std::string const name = p.fft.spec() + " " + (p.background->empty() ? "defaults" : configText(*p.background));
      if (!shown.insert(name).second) { continue; }
      std::vector<double> zs;
      for (u64 const e : repeatExponents(p.fft, p.exponent, repeat)) {
        if (std::optional<SweepReading> const r = readingOf(p.fft, e, *p.background); r && r->checkOk) {
          zs.push_back(r->z);
        }
      }
      log("accuracy:   %-40s%s\n", name.c_str(), spreadText(zs, " %.2f").c_str());
    }
  }

  if (bench.deviceLost()) { return RunOutcome::DeviceLost; }
  bench.end();
  return RunOutcome::Ok;
}

}  // namespace tune
