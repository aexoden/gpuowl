// Copyright (C) Jason Lynch

#include "Scheduler.h"

#include "Args.h"
#include "FFTVariants.h"
#include "log.h"
#include "Primes.h"
#include "Progress.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>

namespace tune {

namespace {

[[nodiscard]] bool carriesWeight(const Grid& grid, const Interval& band) {
  return std::ranges::any_of(grid.points,
                             [&](const GridPoint& p) { return p.weight > 0 && band.contains(p.exponent); });
}

// Section 7.1: at the probe where the band holds it, since that is the exponent that matters most; otherwise at the
// largest prime the band reaches.  0 where the band holds no prime.
[[nodiscard]] u64 timingExponent(const Primes& primes, const Interval& band, u64 probe) {
  if (band.contains(probe)) { return probe; }

  u64 const top = primes.isPrime(band.hi) ? band.hi : primes.prevPrime(band.hi);
  return top >= band.lo ? top : 0;
}

// The row a call of `ran` at `exponent` landed on, as every call to it so far merges.
[[nodiscard]] std::optional<Measurement> rowOf(const TuneDB& db, u32 env, const FFTConfig& fft, TestKind kind,
                                               u64 exponent, const UseConfig& ran) {
  std::string const spec = fft.spec();
  for (const RunRow& row : db.mergedRuns()) {
    if (row.exponent != exponent || row.kind != kind || row.fft != spec || db.envOf(row.sess) != env) { continue; }
    if (const UseConfig* const opts = db.findCfg(row.cfg); opts && *opts == ran && row.m.ok()) { return row.m; }
  }
  return std::nullopt;
}

// The cheapest option set of each identity emission could publish -- whether or not another identity keeps it out of
// the table, since tuning it is how it might get in.
[[nodiscard]] std::map<EntryKey, const SelectionEntry*> bestEntries(const std::vector<SelectionEntry>& candidates) {
  std::map<EntryKey, const SelectionEntry*> out;
  for (const SelectionEntry& e : candidates) {
    auto const [at, fresh] = out.try_emplace({e.fft, e.kind, e.regime.label()}, &e);
    if (!fresh && std::tuple{e.cost, e.id} < std::tuple{at->second->cost, at->second->id}) { at->second = &e; }
  }
  return out;
}

// Every option set of each entry that emission could publish, canonical, cheapest first: by pessimistic cost, then as
// emission breaks a tie, so that the first is the entry's best set.
[[nodiscard]] std::map<EntryKey, std::vector<Reading>> readingsOf(const std::vector<OptionSet>& sets, const Env& env) {
  std::vector<const OptionSet*> order;
  for (const OptionSet& s : sets) { order.push_back(&s); }
  std::ranges::stable_sort(order, [](const OptionSet* a, const OptionSet* b) {
    return std::tuple{a->entry.cost, a->entry.id} < std::tuple{b->entry.cost, b->entry.id};
  });

  std::map<EntryKey, std::vector<Reading>> out;
  for (const OptionSet* s : order) {
    const SelectionEntry& e = s->entry;
    auto const fft = parseFft(e.fft);
    if (!fft) { continue; }
    out[{e.fft, e.kind, e.regime.label()}].push_back(
      {.config = canonicalConfig(env, *fft, e.opts), .cost = e.cost, .error = standardError(s->m)});
  }
  return out;
}

}  // namespace

double CallClock::iterSeconds(double usPerIt) const {
  return double(WARMUP_BLOCKS + BLOCKS_PER_CALL) * blockSize_ * usPerIt * 1e-6;
}

double CallClock::gateSeconds(double usPerIt) const { return ROE_ITERATIONS * usPerIt * 1e-6 + overhead() + compile(); }

double CallClock::overhead() const { return overheadN_ ? overheadSum_ / overheadN_ : CALL_OVERHEAD_SEC; }

double CallClock::compile() const { return compileN_ ? compileSum_ / compileN_ : COMPILE_ESTIMATE_SEC; }

double CallClock::seconds(double usPerIt, bool fresh) const {
  return iterSeconds(usPerIt) + overhead() + (fresh ? compile() : 0);
}

void CallClock::observe(double seconds, double usPerIt, bool fresh) {
  double const extra = seconds - iterSeconds(usPerIt);
  if (fresh) {
    compileSum_ += std::max(0.0, extra - overhead());
    ++compileN_;
  } else {
    overheadSum_ += std::max(0.0, extra);
    ++overheadN_;
  }
}

std::vector<Baseline> baselines(const Env& env, const RunScope& scope, const std::vector<FFTShape>& shapes) {
  Primes const primes;
  std::vector<Baseline> out;

  // At the automatic carry only.  It already runs each regime of the shape, and splits its bands where the regime
  // changes, so a pinned carry could only be the same kernels over less of the range (a 32-bit carry, capped where the
  // automatic one switches) or a 64-bit carry where the automatic one needs only 32 bits -- slower over a band it
  // already serves.  Neither can be the cheapest thing at any exponent.
  for (const FFTShape& shape : shapes) {
    if (!scope.admits(shape)) { continue; }
    for (u32 const variant : runnableVariants(env, shape)) {
      FFTConfig const fft{shape, variant, CARRY_AUTO};
      for (const Interval& band : intervals(fft, minExp(fft), maxExp(fft))) {
        for (const Grid& grid : scope.grids) {
          if (!carriesWeight(grid, band)) { continue; }
          if (u64 const exponent = timingExponent(primes, band, scope.probe)) {
            out.push_back({.fft = fft, .kind = grid.kind, .band = band, .exponent = exponent});
          }
        }
      }
    }
  }

  return out;
}

const char* toString(ItemKind kind) {
  switch (kind) {
  case ItemKind::Anchor: return "anchor";
  case ItemKind::Baseline: return "baseline";
  case ItemKind::Probe: return "probe";
  case ItemKind::Combo: return "combo";
  case ItemKind::Refine: return "refine";
  case ItemKind::Restart: return "restart";
  case ItemKind::Gate: return "gate";
  case ItemKind::Reach: return "reach";
  }
  return "?";
}

Scheduler::Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize, Bootstrap bootstrap,
                     std::optional<Strategy> strategy, bool restarts, bool gate, Halving halving,
                     Exploration exploration) :
  scope_{std::move(scope)},
  baselines_{std::move(baselines)},
  clock_{blockSize},
  bootstrap_{std::move(bootstrap)},
  strategy_{std::move(strategy)},
  restarts_{restarts},
  gate_{gate},
  halving_{halving},
  exploration_{exploration} {
  for (const Baseline& b : baselines_) { searches_.emplace_back(b); }
}

std::string Scheduler::builtKey(const FFTConfig& fft, const UseConfig& options) const {
  return fft.spec() + " " + configText(canonicalConfig(bootstrap_.env(), fft, options));
}

Item Scheduler::itemOf(size_t index, Candidate candidate) const {
  auto const kind = [](Offer offer) {
    switch (offer) {
    case Offer::Lines:
    case Offer::Probe: return ItemKind::Probe;
    case Offer::Combo: return ItemKind::Combo;
    case Offer::Restart: return ItemKind::Restart;
    }
    return ItemKind::Probe;
  };

  bool const fresh = !built_.contains(builtKey(baselines_[index].fft, candidate.options));
  return {.kind = kind(candidate.kind),
          .index = index,
          .options = std::move(candidate.options),
          .moved = std::move(candidate.moved),
          .what = std::move(candidate.what),
          .exponent = candidate.exponent,
          .value = candidate.value,
          .cost = candidate.cost,
          .seconds = clock_.seconds(candidate.observed > 0 ? candidate.observed : candidate.cost, fresh),
          .fresh = fresh,
          .calls = candidate.calls,
          .draw = candidate.draw,
          .tier = candidate.tier,
          .unlisted = candidate.unlisted};
}

std::string Scheduler::keyOf(const Item& item) const {
  switch (item.kind) {
  case ItemKind::Anchor: return "anchor";
  case ItemKind::Probe:
  case ItemKind::Combo:
  case ItemKind::Refine:
  case ItemKind::Restart:
    return baselines_[item.index].label() + " " +
      configText(canonicalConfig(bootstrap_.env(), baselines_[item.index].fft, item.options));
  case ItemKind::Reach:
    return item.fft->spec() + " " + item.span.regime.label() + " reach " +
      configText(canonicalConfig(bootstrap_.env(), *item.fft, item.options)) + "@" + std::to_string(item.exponent);
  case ItemKind::Gate:
    return baselines_[item.index].label() + " " + toString(item.kind) + " " +
      configText(canonicalConfig(bootstrap_.env(), baselines_[item.index].fft, item.options)) + "@" +
      std::to_string(item.exponent);
  case ItemKind::Baseline: break;
  }
  return baselines_[item.index].label() + "@" + std::to_string(item.exponent);
}

std::optional<size_t> Scheduler::entryOf(const Family& family, TestKind kind) const {
  std::string const spec = family.fft.spec();
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    if (b.kind == kind && b.band.contains(bootstrap_.probe()) && b.fft.spec() == spec) { return i; }
  }
  return {};
}

BootstrapState Scheduler::bootstrapState(const TuneDB& db, u32 env) const {
  BootstrapState out = bootstrap_.state(db, env, u64(BOOTSTRAP_ROUNDS) * halving_.roundCalls);
  out.complete = true;
  for (FamilyState& f : out.families) {
    bool const owed = f.phase == FamilyPhase::Unread || f.phase == FamilyPhase::Owed;
    std::optional<size_t> const entry = entryOf(f.family, out.kind);
    if (owed && (!strategy_ || !entry)) {
      f.phase = FamilyPhase::Skipped;
    } else if (f.phase == FamilyPhase::Unread) {
      auto const tried = attempts_.find(*entry);
      if (tried != attempts_.end() && tried->second >= MAX_ATTEMPTS) { f.phase = FamilyPhase::Held; }
    }
    out.complete = out.complete && f.phase != FamilyPhase::Unread && f.phase != FamilyPhase::Owed;
  }
  return out;
}

Defaults Scheduler::lines(const TuneDB& db, u32 env) const {
  TestKind const kind = scope_.grid(TestKind::PRP) || !scope_.grid(TestKind::LL) ? TestKind::PRP : TestKind::LL;
  return publishedLines(bootstrap_.env(), scope_.probe, kind, candidatesFor(db, env));
}

std::vector<Item> Scheduler::bootstrapReads(const BootstrapState& state, const TuneDB& db, u32 env,
                                            const Progress& progress, const Objective& objective) const {
  std::vector<Item> out;
  for (const FamilyState& f : state.families) {
    if (f.phase != FamilyPhase::Unread) { continue; }
    size_t const i = *entryOf(f.family, state.kind);
    const Baseline& b = baselines_[i];

    Partial p{};
    if (auto const at = progress.partial.find({b.key(), configText({})}); at != progress.partial.end()) {
      p = at->second;
    }
    u64 const exponent = p.calls && b.band.contains(p.exponent) ? p.exponent : b.exponent;
    if (u32 const cfg = db.findCfgId({}); cfg && db.diedOn(env, cfg, b.kind, b.fft.spec(), exponent)) { continue; }

    double const estimate = objective.priorModel().cost(b.fft.shape);
    bool const fresh = !built_.contains(builtKey(b.fft, {}));
    out.push_back({.kind = ItemKind::Baseline,
                   .index = i,
                   .options = {},
                   .moved = {},
                   .what = "at the built-in defaults",
                   .exponent = exponent,
                   .value = 0,
                   .cost = estimate,
                   .seconds = clock_.seconds(estimate / PRIOR_OPTIMISM, fresh),
                   .fresh = fresh,
                   .calls = p.calls,
                   .bootstrap = true});
  }

  // Which families are worth searching is a comparison between them, so every one is read before any is searched,
  // the cheapest first.
  std::ranges::stable_sort(out, {}, &Item::seconds);
  return out;
}

std::map<EntryKey, size_t> Scheduler::entryIndex() const {
  std::map<EntryKey, size_t> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    out.try_emplace({baselines_[i].fft.spec(), baselines_[i].kind, baselines_[i].band.regime.label()}, i);
  }
  return out;
}

std::vector<Item> Scheduler::gateItems(const TuneDB& db, u32 env) const {
  if (!gate_) { return {}; }
  const Env& device = bootstrap_.env();
  std::map<EntryKey, size_t> const indexOf = entryIndex();

  std::vector<Item> out;
  std::set<std::string> offered;
  for (const OptionSet& s : gatesOwed(db, env)) {
    const SelectionEntry& e = s.entry;

    // An entry of a run over another workload waits for a run whose workload weighs it.
    auto const at = indexOf.find({e.fft, e.kind, e.regime.label()});
    if (at == indexOf.end() || !s.gate.owedAt) { continue; }
    const Baseline& b = baselines_[at->second];

    UseConfig const canonical = canonicalConfig(device, b.fft, e.opts);
    std::string const text = canonical.empty() ? "the built-in defaults" : configText(canonical);
    Interval const span{.lo = e.emin, .hi = e.reach, .regime = e.regime};
    bool const deriving = !s.gate.owesReference && s.gate.owedAt != gateExponent(span);
    Item item{.kind = ItemKind::Gate,
              .index = at->second,
              .options = s.gate.owesReference ? accuracyReference(device, b.fft, e.opts) : e.opts,
              .subject = e.opts,
              .span = span,
              .moved = {},
              .what = s.gate.owesReference ? "the default accuracy of " + text
                : deriving                 ? text + ", deriving its reach"
                                           : text,
              .exponent = s.gate.owedAt,
              .value = 1,
              .seconds = clock_.gateSeconds(s.m.cost()),
              .fresh = true,
              .calls = 0};

    // Two kinds of one set, or two sets of one reference, owe the same reading.
    std::string const key = keyOf(item);
    if (!offered.insert(key).second) { continue; }
    if (auto const n = gateAttempts_.find(key); n != gateAttempts_.end() && n->second >= MAX_ATTEMPTS) { continue; }
    if (u32 const cfg = db.findCfgId(item.options); cfg && db.diedOn(env, cfg, TestKind::PRP, e.fft, item.exponent)) {
      continue;
    }
    out.push_back(std::move(item));
  }

  std::ranges::stable_sort(out, {}, &Item::seconds);
  return out;
}

std::vector<Item> Scheduler::coverItems(std::span<const Item> baselines, const Objective& objective) const {
  if (!gate_) { return {}; }

  std::vector<Item> out;
  for (const Item& item : baselines) {
    const Baseline& b = baselines_[item.index];
    std::optional<u64> lo;
    u64 hi = 0;
    for (const ObjectivePoint& p : objective.points()) {
      if (p.kind != b.kind || p.weight <= 0 || !p.cost || p.cost->measured() || !b.band.contains(p.exponent)) {
        continue;
      }
      lo = std::min(lo.value_or(p.exponent), p.exponent);
      hi = std::max(hi, p.exponent);
    }
    if (!lo) { continue; }

    Item& cover = out.emplace_back(item);
    cover.cover = true;
    cover.what = "covering " + std::to_string(*lo) + (*lo == hi ? "" : "-" + std::to_string(hi));
  }

  rankByShape(out);
  return out;
}

std::vector<Item> Scheduler::reachItems(const TuneDB& db, u32 env, std::span<const OptionSet> sets,
                                        const Objective& objective) const {
  if (!gate_) { return {}; }
  const Env& device = bootstrap_.env();

  std::vector<Item> out;
  std::map<std::string, size_t> offered;
  for (const OptionSet& s : sets) {
    const SelectionEntry& e = s.entry;
    if (s.gate.state != GateState::Passed || !s.gate.raiseAt) { continue; }
    auto const fft = parseFft(e.fft);
    if (!fft) { continue; }

    // Valued as if the reading about to be taken confirms the reach it is taken at, which is the derivation's own
    // estimate of it; a backoff lowers the estimate, and the next item is worth less.
    double const value =
      saving(objective.points(), e.kind, {.lo = e.reach + 1, .hi = s.gate.raiseAt, .regime = e.regime}, e.cost);
    if (value <= 0) { continue; }

    UseConfig const canonical = canonicalConfig(device, *fft, e.opts);
    Item item{.kind = ItemKind::Reach,
              .index = 0,
              .fft = fft,
              .options = e.opts,
              .subject = e.opts,
              .span = {.lo = e.emin, .hi = e.reach, .regime = e.regime},
              .moved = {},
              .what = e.fft + " " + e.regime.label() + " " +
                (canonical.empty() ? "the built-in defaults" : configText(canonical)) + ", raising its reach",
              .exponent = s.gate.raiseAt,
              .value = value,
              .cost = e.cost,
              .seconds = clock_.gateSeconds(s.m.cost()),
              .fresh = true,
              .calls = 0};

    // The set's other kinds are raised by the same reading, which is worth what it saves in each; the key names the
    // reading, not a kind.
    std::string const key = keyOf(item);
    if (auto const seen = offered.find(key); seen != offered.end()) {
      out[seen->second].value += value;
      continue;
    }
    if (auto const n = gateAttempts_.find(key); n != gateAttempts_.end() && n->second >= MAX_ATTEMPTS) { continue; }
    if (u32 const cfg = db.findCfgId(item.options); cfg && db.diedOn(env, cfg, TestKind::PRP, e.fft, item.exponent)) {
      continue;
    }
    offered.emplace(key, out.size());
    out.push_back(std::move(item));
  }
  return out;
}

std::vector<Item> Scheduler::baselineItems(const TuneDB& db, u32 env, const Objective& objective) const {
  Progress const progress = progressOf(db, env, bootstrap_.env());
  GainModel const gains = gainsOf(db, env);
  std::vector<Item> out = baselineItems(db, env, progress, gains, objective);
  if (strategy_) {
    std::vector<OptionSet> const sets = optionSetsFor(db, env);
    std::ranges::move(sweepItems(db, env, progress, readingsOf(sets, bootstrap_.env()), gains, objective),
                      std::back_inserter(out));
  }
  return out;
}

std::vector<Item> Scheduler::baselineItems(const TuneDB& db, u32 env, const Progress& progress, const GainModel& gains,
                                           const Objective& objective) const {
  GainDist const unmeasured = gains.global();

  std::vector<Item> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    std::string const spec = b.fft.spec();
    EntryKey const key{spec, b.kind, b.band.regime.label()};

    if (progress.settled.contains(key)) { continue; }
    if (auto const at = attempts_.find(i); at != attempts_.end() && at->second >= MAX_ATTEMPTS) { continue; }

    // At the built-in defaults: a baseline is what the entry costs untuned, which is what everything it gains is a
    // gain over, and the lines are one step of its search like any other.
    UseConfig options{};
    if (progress.failed.contains({key, configText(options)})) { continue; }

    Partial p{};
    if (auto const at = progress.partial.find({key, configText(options)}); at != progress.partial.end()) {
      p = at->second;
    }

    // Resumed where it was started, so that the calls already made pool with the ones still to make.
    u64 const exponent = p.calls && b.band.contains(p.exponent) ? p.exponent : b.exponent;

    // Held back as a measurement would hold it back, which is by what was asked for rather than by what ran.
    if (u32 const cfg = db.findCfgId(options); cfg && db.diedOn(env, cfg, b.kind, spec, exponent)) { continue; }

    double const estimate = objective.priorModel().cost(b.fft.shape);
    double const value = expectedSaving(objective.points(), b.kind, b.band, estimate, unmeasured);

    bool const fresh = !built_.contains(builtKey(b.fft, options));
    out.push_back({.kind = ItemKind::Baseline,
                   .index = i,
                   .options = std::move(options),
                   .moved = {},
                   .what = {},
                   .exponent = exponent,
                   .value = value,
                   .cost = estimate,
                   .seconds = clock_.seconds(estimate / PRIOR_OPTIMISM, fresh),
                   .fresh = fresh,
                   .calls = p.calls});
  }

  return out;
}

std::vector<Item> Scheduler::sweepItems(const TuneDB& db, u32 env, const Progress& progress,
                                        const std::map<EntryKey, std::vector<Reading>>& readings,
                                        const GainModel& gains, const Objective& objective) const {
  GainDist const unmeasured = gains.global();

  // Where each entry stands at the built-in defaults.  An entry is read once a reading there has concluded or failed;
  // one under any other options does not say what the entry costs untuned.
  struct Owed {
    EntryKey key;
    bool measured = false;
    bool read = false;
    bool runnable = false;
    Partial partial{};
    u64 exponent = 0;

    // Its cheapest reading under any options where it has one, else its prior without the optimism the value model
    // gives it, so that the margin means the same measured or not.
    double estimate = 0;
  };
  std::vector<Owed> owed;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    std::string const spec = b.fft.spec();
    Owed& o = owed.emplace_back(Owed{.key = {spec, b.kind, b.band.regime.label()}});
    EntrySet const defaults{o.key, configText({})};

    auto const measured = readings.find(o.key);
    o.measured = measured != readings.end();
    o.estimate = o.measured ? measured->second.front().cost : objective.priorModel().cost(b.fft.shape) / PRIOR_OPTIMISM;

    auto const at = progress.concluded.find(o.key);
    o.read = progress.failed.contains(defaults) ||
      (at != progress.concluded.end() && std::ranges::any_of(at->second, &UseConfig::empty));

    if (auto const p = progress.partial.find(defaults); p != progress.partial.end()) { o.partial = p->second; }
    o.exponent = o.partial.calls && b.band.contains(o.partial.exponent) ? o.partial.exponent : b.exponent;

    // Held back as a measurement would hold it back.
    auto const tried = attempts_.find(i);
    u32 const cfg = db.findCfgId({});
    o.runnable = !o.read && (tried == attempts_.end() || tried->second < MAX_ATTEMPTS) &&
      !(cfg && db.diedOn(env, cfg, b.kind, spec, o.exponent));
  }

  // What each point is held to: what production is measured to run there, or the prior of an entry not read yet that
  // serves it, if cheaper -- the cheapest are read first, and each reading replaces its prior.  Not the prior of one
  // that cannot be read, which nothing would ever replace, nor a reading production does not run there.  What a prior
  // holds back waits on an entry the sweep still owes, and would be ranked after it anyway: the sweep never reads more,
  // or finishes later, than against measured costs alone, and a prior that proves right saves what it held back.
  std::vector<std::optional<double>> fastest = measuredAt(objective);
  for (size_t p = 0; p < objective.points().size(); ++p) {
    const ObjectivePoint& point = objective.points()[p];
    if (point.weight <= 0) { continue; }
    for (size_t i = 0; i < baselines_.size(); ++i) {
      const Baseline& b = baselines_[i];
      if (owed[i].measured || !owed[i].runnable || b.kind != point.kind || !b.band.contains(point.exponent)) {
        continue;
      }
      fastest[p] = std::min(fastest[p].value_or(owed[i].estimate), owed[i].estimate);
    }
  }

  std::vector<Item> out;
  sweepWithin_ = 0;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    const Owed& o = owed[i];

    bool contends = false;
    for (size_t p = 0; p < objective.points().size() && !contends; ++p) {
      const ObjectivePoint& point = objective.points()[p];
      contends = fastest[p] && point.kind == b.kind && b.band.contains(point.exponent) &&
        o.estimate <= (1 + exploration_.sweepMargin) * *fastest[p];
    }

    // One read is counted as read whether or not it still contends, so that what the sweep says it has read only
    // grows as estimates give way to readings.
    sweepWithin_ += contends || o.read;
    if (!contends || !o.runnable) { continue; }

    bool const fresh = !built_.contains(builtKey(b.fft, {}));
    out.push_back({.kind = ItemKind::Baseline,
                   .index = i,
                   .options = {},
                   .moved = {},
                   .what = "at the built-in defaults",
                   .exponent = o.exponent,
                   .value = expectedSaving(objective.points(), b.kind, b.band, o.estimate, unmeasured),
                   .cost = o.estimate,
                   .seconds = clock_.seconds(o.estimate, fresh),
                   .fresh = fresh,
                   .calls = o.partial.calls,
                   .sweep = true});
  }
  rankByShape(out);
  sweepOwed_ = 0;
  std::set<size_t> indices;
  for (const Item& item : out) { sweepOwed_ += indices.insert(item.index).second; }
  return out;
}

std::vector<std::optional<double>> Scheduler::measuredAt(const Objective& objective) const {
  std::vector<std::optional<double>> out(objective.points().size());
  for (size_t p = 0; p < objective.points().size(); ++p) {
    const ObjectivePoint& point = objective.points()[p];
    if (point.weight > 0 && point.cost && point.cost->measured()) { out[p] = point.cost->us; }
  }
  return out;
}

bool Scheduler::Standing::within(size_t i, double margin) const {
  return gap[i] && (*gap[i] <= margin || (atDefaults[i] && *atDefaults[i] <= margin));
}

bool Scheduler::Standing::ahead(size_t a, size_t b) const {
  return std::tuple{!gap[a], gap[a].value_or(0), -weight[a], a} <
    std::tuple{!gap[b], gap[b].value_or(0), -weight[b], b};
}

Scheduler::Standing Scheduler::standingOf(const std::map<EntryKey, std::vector<Reading>>& readings,
                                          const Objective& objective) const {
  std::vector<std::optional<double>> const fastest = measuredAt(objective);
  std::span<const ObjectivePoint> const points = objective.points();
  auto const serves = [&](const Baseline& b, const ObjectivePoint& point) {
    return point.weight > 0 && point.kind == b.kind && b.band.contains(point.exponent);
  };

  Standing out{.gap = std::vector<std::optional<double>>(baselines_.size()),
               .atDefaults = std::vector<std::optional<double>>(baselines_.size()),
               .weight = std::vector<double>(baselines_.size())};
  std::vector<const std::vector<Reading>*> measured(baselines_.size());
  std::vector<std::optional<double>> defaults(baselines_.size());
  std::vector<std::optional<double>> cheapestDefaults(points.size());
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    auto const at = readings.find({b.fft.spec(), b.kind, b.band.regime.label()});
    if (at == readings.end()) { continue; }
    measured[i] = &at->second;
    if (auto const d = std::ranges::find_if(at->second, [](const Reading& r) { return r.config.empty(); });
        d != at->second.end()) {
      defaults[i] = d->cost;
    }
    for (size_t p = 0; p < points.size(); ++p) {
      if (!serves(b, points[p])) { continue; }
      if (defaults[i]) { cheapestDefaults[p] = std::min(cheapestDefaults[p].value_or(*defaults[i]), *defaults[i]); }
      if (fastest[p] && points[p].cost->fft == b.fft.spec()) { out.weight[i] += points[p].weight; }
    }
  }

  // Only against what is measured: an estimate says what to read next, and a prior that is wrong, or can never be
  // read, must not decide which of the entries already read is searched.
  for (size_t i = 0; i < baselines_.size(); ++i) {
    if (!measured[i]) { continue; }
    for (size_t p = 0; p < points.size(); ++p) {
      if (!fastest[p] || !serves(baselines_[i], points[p])) { continue; }
      double const gap = measured[i]->front().cost / *fastest[p] - 1;
      out.gap[i] = std::min(out.gap[i].value_or(gap), gap);
      if (defaults[i] && cheapestDefaults[p]) {
        double const behind = *defaults[i] / *cheapestDefaults[p] - 1;
        out.atDefaults[i] = std::min(out.atDefaults[i].value_or(behind), behind);
      }
    }
  }
  return out;
}

std::set<enum FFT_TYPES> Scheduler::acceptedTypes(const Standing& standing) const {
  std::set<enum FFT_TYPES> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    if (standing.within(i, exploration_.typeMargin)) { out.insert(baselines_[i].fft.shape.fft_type); }
  }
  return out;
}

std::vector<size_t> Scheduler::explorable(const Standing& standing, const std::set<enum FFT_TYPES>& accepted,
                                          const Objective& objective) const {
  std::vector<size_t> read;
  std::vector<std::pair<double, size_t>> unread;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    if (!accepted.contains(baselines_[i].fft.shape.fft_type)) { continue; }
    if (standing.gap[i]) {
      read.push_back(i);
    } else {
      unread.emplace_back(objective.priorModel().cost(baselines_[i].fft.shape), i);
    }
  }
  std::ranges::stable_sort(read, [&](size_t a, size_t b) { return standing.ahead(a, b); });
  std::ranges::stable_sort(unread, {}, &std::pair<double, size_t>::first);
  for (const auto& [prior, i] : unread) { read.push_back(i); }
  return read;
}

std::vector<Item> Scheduler::linesSweepItems(const TuneDB& db, u32 env, const SearchContext& context,
                                             const std::map<EntryKey, std::vector<Reading>>& readings,
                                             const Standing& standing, const std::vector<size_t>& entries,
                                             const Objective& objective) const {
  linesOwed_ = 0;
  linesWithin_ = 0;
  const LinesRow* latest = nullptr;
  for (const LinesRow& row : db.lines()) {
    if (db.envOf(row.sess) == env && (!latest || row.n > latest->n)) { latest = &row; }
  }
  if (!exploration_.linesSweep || !latest) { return {}; }

  Defaults const swept = recordedLines(db, *latest);
  const Env& device = bootstrap_.env();
  std::vector<std::optional<double>> const fastest = measuredAt(objective);
  double const margin = exploration_.linesMarginOf(latest->n);
  std::vector<Item> out;
  for (size_t const i : entries) {
    if (!nearFastest(i, standing, fastest, objective, margin)) { continue; }
    const Baseline& b = baselines_[i];
    EntryKey const key = b.key();
    auto const at = readings.find(key);
    const Reading* const best = at == readings.end() ? nullptr : &at->second.front();

    bool owed = false;
    bool within = false;
    auto const read = [&](UseConfig options, const std::string& what) {
      std::string const text = configText(options);
      if (options.empty()) { return; }
      within = true;
      if (context.progress.answered.contains({key, text})) { return; }
      Candidate c{.kind = Offer::Lines, .options = std::move(options), .what = what + text, .exponent = b.exponent};
      if (auto const p = context.progress.partial.find({key, text});
          p != context.progress.partial.end() && b.band.contains(p->second.exponent)) {
        c.exponent = p->second.exponent;
        c.calls = p->second.calls;
        c.observed = p->second.mean;
      }
      if (!searches_[i].runnable(context, c.options, c.exponent)) { return; }
      // One not read yet is read under the lines alone, which is all the sweep needs of it: a reading at the built-in
      // defaults would say less of what it costs tuned.
      c.cost = best ? best->cost : objective.priorModel().cost(b.fft.shape) / PRIOR_OPTIMISM;
      c.value = std::numeric_limits<double>::min();
      Item& item = out.emplace_back(itemOf(i, std::move(c)));
      item.linesSweep = true;
      owed = true;
    };
    // The entry's own findings under the lines first, as its search takes them.
    UseConfig const lines = underDefaults(device, b.fft, b.kind, swept);
    if (best && !best->config.empty()) {
      UseConfig const over = underDefaults(device, b.fft, b.kind, swept, best->config);
      if (over != lines) { read(over, "its best set under the default lines "); }
    }
    read(lines, "the default lines ");
    linesOwed_ += owed;
    linesWithin_ += within;
  }
  return out;
}

std::optional<Defaults> Scheduler::linesDue(const TuneDB& db, u32 env, const BootstrapState& state, const Defaults& now,
                                            bool ended) const {
  if (!exploration_.linesSweep || !strategy_ || !state.complete || (now.global.empty() && now.family.empty())) {
    return {};
  }

  const LinesRow* latest = nullptr;
  for (const LinesRow& row : db.lines()) {
    if (db.envOf(row.sess) == env && (!latest || row.n > latest->n)) { latest = &row; }
  }
  if (!latest) { return now; }
  if (linesText(recordedLines(db, *latest)) == linesText(now)) { return {}; }

  // A round of one entry ends its halving.
  ended = ended || std::ranges::any_of(db.rounds(), [&](const RoundRow& r) {
            return db.envOf(r.sess) == env && r.members.size() == 1 && r.n > latest->after;
          });
  return ended ? std::optional{now} : std::nullopt;
}

std::vector<u64> Scheduler::searchCalls(const TuneDB& db, u32 env) const {
  const Env& device = bootstrap_.env();
  std::map<EntryKey, u64> searched;
  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || row.m.status == Status::Lost) { continue; }
    auto const fft = parseFft(row.fft);
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (!fft || !opts || canonicalConfig(device, *fft, *opts).empty()) { continue; }
    searched[{row.fft, row.kind, row.regime.label()}] += std::max<u32>(row.m.calls, 1);
  }

  std::vector<u64> out;
  for (const Baseline& b : baselines_) {
    auto const at = searched.find({b.fft.spec(), b.kind, b.band.regime.label()});
    out.push_back(at == searched.end() ? 0 : at->second);
  }
  return out;
}

u64 Scheduler::searchCallsSince(const TuneDB& db, u32 env, u64 ts) const {
  const Env& device = bootstrap_.env();
  std::map<EntryKey, size_t> const indexOf = entryIndex();
  // A configuration is canonicalised once, however many rows name it.
  std::map<std::pair<std::string, u32>, bool> searching;
  u64 out = 0;
  for (const RunRow& row : db.runs()) {
    if (row.m.ts <= ts || db.envOf(row.sess) != env || row.m.status == Status::Lost) { continue; }
    if (!indexOf.contains({row.fft, row.kind, row.regime.label()})) { continue; }
    auto const [at, fresh] = searching.try_emplace({row.fft, row.cfg}, false);
    if (fresh) {
      auto const fft = parseFft(row.fft);
      const UseConfig* const opts = db.findCfg(row.cfg);
      at->second = fft && opts && !canonicalConfig(device, *fft, *opts).empty();
    }
    if (at->second) { out += std::max<u32>(row.m.calls, 1); }
  }
  return out;
}

bool Scheduler::nearFastest(size_t i, const Standing& standing, const std::vector<std::optional<double>>& fastest,
                            const Objective& objective, double margin) const {
  if (standing.gap[i]) { return *standing.gap[i] <= margin; }

  // Without the optimism the value model gives a prior, so that the margin means the same measured or not.
  const Baseline& b = baselines_[i];
  double const estimate = objective.priorModel().cost(b.fft.shape) / PRIOR_OPTIMISM;
  for (size_t p = 0; p < objective.points().size(); ++p) {
    const ObjectivePoint& point = objective.points()[p];
    if (fastest[p] && point.weight > 0 && point.kind == b.kind && b.band.contains(point.exponent) &&
        estimate <= (1 + margin) * *fastest[p]) {
      return true;
    }
  }
  return false;
}

std::vector<size_t> Scheduler::poolOf(const Standing& standing, const std::function<bool(size_t)>& eligible,
                                      u32 limit) const {
  // Each shape's variants ranked, and the pool filled rank by rank.
  std::map<std::tuple<std::string, TestKind>, std::vector<size_t>> byShape;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    if (standing.gap[i] && eligible(i)) { byShape[{baselines_[i].fft.shape.spec(), baselines_[i].kind}].push_back(i); }
  }
  std::vector<std::pair<size_t, size_t>> ranked;
  for (auto& [shape, members] : byShape) {
    std::ranges::stable_sort(members, [&](size_t a, size_t b) { return standing.ahead(a, b); });
    for (size_t r = 0; r < members.size(); ++r) { ranked.emplace_back(r, members[r]); }
  }
  std::ranges::sort(ranked, [&](const auto& a, const auto& b) {
    return a.first != b.first ? a.first < b.first : standing.ahead(a.second, b.second);
  });

  std::vector<size_t> out;
  for (const auto& [rank, i] : ranked) {
    if (out.size() < limit) { out.push_back(i); }
  }
  std::ranges::sort(out, [&](size_t a, size_t b) { return standing.ahead(a, b); });
  return out;
}

RoundMember Scheduler::memberOf(size_t index, u64 from) const {
  const Baseline& b = baselines_[index];
  return {.fft = b.fft.spec(), .kind = b.kind, .regime = b.band.regime, .from = from};
}

std::vector<RoundRow> Scheduler::adoption(const Standing& standing, const std::vector<u64>& calls) const {
  std::vector<RoundRow> const fresh{{.sess = 0, .n = 1, .round = 0, .calls = 0, .ts = 0, .members = {}}};

  // As that rule chose its contenders: within the margin now.
  std::vector<size_t> const first =
    poolOf(standing, [&](size_t i) { return *standing.gap[i] <= CONTEND_MARGIN; }, halving_.contenders);
  if (first.size() < 2) { return fresh; }

  // Who took part, so that none of them is taken for a newcomer.
  RoundRow took{.sess = 0, .n = 1, .round = 1, .calls = halving_.roundCalls, .ts = 0, .members = {}};
  for (size_t const i : first) { took.members.push_back(memberOf(i, 0)); }

  // That rule's round r was over once every entry in it had roundCalls * (2^(r+1) - 1) calls in all.
  std::vector<size_t> pool = first;
  u32 round = 0;
  for (; pool.size() > 1 && round < 40; ++round) {
    u64 const upTo = u64(halving_.roundCalls) * ((u64{2} << round) - 1);
    if (std::ranges::none_of(pool, [&](size_t i) { return calls[i] < upTo; })) {
      pool.resize((pool.size() + 1) / 2);
      continue;
    }
    if (round == 0) { return fresh; }

    u64 const before = u64(halving_.roundCalls) * ((u64{1} << round) - 1);
    RoundRow now{
      .sess = 0, .n = 2, .round = round + 1, .calls = u64(halving_.roundCalls) << round, .ts = 0, .members = {}};
    for (size_t const i : pool) { now.members.push_back(memberOf(i, std::min(calls[i], before))); }
    return {took, now};
  }

  RoundRow over{.sess = 0, .n = 2, .round = round + 1, .calls = 0, .ts = 0, .members = {memberOf(pool.front(), 0)}};
  return {took, over};
}

HalvingState Scheduler::halvingState(const TuneDB& db, u32 env,
                                     const std::map<EntryKey, std::vector<Reading>>& readings,
                                     const Objective& objective, const Explored& explored, bool held) const {
  HalvingState out;
  if (!halving_.on() || !strategy_) { return out; }

  Standing const standing = standingOf(readings, objective);
  std::vector<u64> const calls = searchCalls(db, env);
  std::map<EntryKey, size_t> const indexOf = entryIndex();

  // Only the rounds of entries this workload weighs: a halving over other kinds or bands says nothing of how far these
  // have been explored, and a workload that returns to those takes their halving up where it stood.
  std::vector<RoundRow> rounds;
  u32 n = 0;
  bool recorded = false;
  for (const RoundRow& r : db.rounds()) {
    if (db.envOf(r.sess) != env) { continue; }
    recorded = true;
    n = std::max(n, r.n);
    if (r.members.empty() || std::ranges::any_of(r.members, [&](const RoundMember& m) {
          return indexOf.contains({m.fft, m.kind, m.regime.label()});
        })) {
      rounds.push_back(r);
    }
  }
  std::ranges::stable_sort(rounds, {}, &RoundRow::n);
  if (!recorded) {
    out.unrecorded = adoption(standing, calls);
    rounds = out.unrecorded;
    for (const RoundRow& r : rounds) { n = std::max(n, r.n); }
  }

  // A round is begun only where one of its entries has a step to take: one that none has would be over as soon as it
  // began, and the next begun after it, each with twice the calls.  For the same reason a round is never of no calls.
  auto const due = [&](const std::vector<size_t>& members) { return std::ranges::any_of(members, explored.offers); };
  auto const twice = [&](u64 calls) {
    u64 constexpr most = std::numeric_limits<u64>::max();
    return std::max<u64>(calls > most / 2 ? most : 2 * calls, halving_.roundCalls);
  };
  auto begin = [&](u32 round, u64 budget, const std::vector<size_t>& members) {
    RoundRow row{.sess = 0, .n = ++n, .round = round, .calls = budget, .ts = 0, .members = {}};
    for (size_t const i : members) { row.members.push_back(memberOf(i, calls[i])); }
    out.unrecorded.push_back(row);
    rounds.push_back(std::move(row));
  };

  // The first round of the halving `last` is a round of, and what that halving's rounds gave out up to `last`.
  using Rows = std::vector<RoundRow>::reverse_iterator;
  auto const firstOf = [&](Rows last) {
    return std::find_if(last, rounds.rend(), [](const RoundRow& r) { return r.round == 1 && r.members.size() > 1; });
  };
  auto const spent = [&](Rows last) {
    u64 out = 0;
    for (Rows r = last; r != rounds.rend(); ++r) {
      if (r->members.size() > 1) { out += r->members.size() * r->calls; }
      if (r->round == 1 && r->members.size() > 1) { break; }
    }
    return out;
  };
  auto const halvings = [&](Rows first) {
    return u32(
      std::count_if(first, rounds.rend(), [](const RoundRow& r) { return r.round == 1 && r.members.size() > 1; }));
  };
  auto const contends = [&](size_t i) { return standing.contends(i); };

  // The latest round before `last` still owing calls to entries no round since has taken up: entries the workload did
  // not weigh when the round was ended.  A round is ended wherever its entries in the workload have had theirs, or a
  // workload that has narrowed for good would never halve again; those it could not see are served once a workload
  // that weighs them runs, before anything else is decided.
  struct Debt {
    Rows row;
    std::vector<size_t> owed;
    std::vector<u64> had;
  };
  auto const debtBefore = [&](Rows last) -> std::optional<Debt> {
    std::set<size_t> later;
    for (Rows r = rounds.rbegin(); r != rounds.rend(); ++r) {
      Debt debt{.row = r, .owed = {}, .had = {}};
      for (const RoundMember& m : r->members) {
        auto const at = indexOf.find({m.fft, m.kind, m.regime.label()});
        if (at == indexOf.end() || !later.insert(at->second).second) { continue; }
        size_t const i = at->second;
        u64 const had = calls[i] > m.from ? calls[i] - m.from : 0;
        if (r != last && r->members.size() > 1 && had < r->calls && explored.offers(i)) {
          debt.owed.push_back(i);
          debt.had.push_back(had);
        }
      }
      if (!debt.owed.empty()) { return debt; }
    }
    return std::nullopt;
  };

  // Each pass either finds where the halving stands, or records a round that has just ended or begun, so that a round
  // whose entries have nothing to take is passed over at once.
  for (u32 pass = 0; pass < 2 * baselines_.size() + 64; ++pass) {
    auto const last =
      std::ranges::find_if(rounds.rbegin(), rounds.rend(), [](const RoundRow& r) { return !r.members.empty(); });

    if (last == rounds.rend()) {
      std::vector<size_t> const pool = poolOf(standing, contends, halving_.contenders);
      if (pool.size() < 2 || !due(pool)) { return out; }
      if (held) {
        out.waiting = true;
        return out;
      }
      begin(1, halving_.roundCalls, pool);
      continue;
    }

    // The round's entries this workload weighs, and the calls each has had in it.
    std::vector<size_t> pool;
    std::vector<u64> had;
    bool owed = false;
    for (const RoundMember& m : last->members) {
      auto const at = indexOf.find({m.fft, m.kind, m.regime.label()});
      if (at == indexOf.end()) { continue; }
      size_t const i = at->second;
      pool.push_back(i);
      had.push_back(calls[i] > m.from ? calls[i] - m.from : 0);
      owed = owed || (had.back() < last->calls && explored.offers(i)) || explored.resumes(i);
    }
    auto const first = firstOf(last);
    out.halving = first == rounds.rend() ? 0 : halvings(first);
    out.contenders = u32(first == rounds.rend() ? last->members.size() : first->members.size());

    auto const serve = [&](Rows row, std::vector<size_t> members, std::vector<u64> served) {
      auto const head = firstOf(row);
      out.halving = head == rounds.rend() ? 0 : halvings(head);
      out.contenders = u32(head == rounds.rend() ? row->members.size() : head->members.size());
      out.active = true;
      out.round = row->round - 1;
      out.n = row->n;
      out.budget = row->calls;
      out.pool = std::move(members);
      out.calls = std::move(served);
      return out;
    };
    if (last->members.size() > 1 && owed) { return serve(last, std::move(pool), std::move(had)); }
    if (auto debt = debtBefore(last)) { return serve(debt->row, std::move(debt->owed), std::move(debt->had)); }

    if (last->members.size() > 1) {
      // Over: its better half go on, and where one is left, it leads until the next halving.  A half with nothing left
      // to take has nothing left to show, and the next halving begins instead.
      std::ranges::sort(pool, [&](size_t a, size_t b) { return standing.ahead(a, b); });
      pool.resize((pool.size() + 1) / 2);
      if (pool.size() == 1 || (pool.size() > 1 && due(pool))) {
        begin(last->round + 1, pool.size() > 1 ? twice(last->calls) : spent(last), pool);
        continue;
      }
    } else if (!pool.empty() && explored.worth(pool.front())) {
      // The search ranked by value has its share, whichever entries it goes to: the one left need not be what it
      // prices highest, and counting that one's calls alone would never end a turn whose steps go elsewhere.  It ends
      // early, as before, once the one left has nothing worth a call, so that a contender's step worth one is still
      // there to begin the next halving.  A round just begun has had nothing yet.
      u64 const since = last->sess ? searchCallsSince(db, env, last->ts) : 0;
      if (since < spent(last)) {
        out.leading = true;
        out.budget = spent(last);
        out.pool = std::move(pool);
        out.calls = {since};
        return out;
      }
    }

    // The next halving, over every contender as they stand now, its first round twice the last one's.
    std::vector<size_t> const next = poolOf(standing, contends, halving_.contenders);
    bool const byRule = (first == rounds.rend() ? 0 : halvings(first)) < halving_.halvings;
    if (next.size() < 2 || !due(next) || (!byRule && std::ranges::none_of(next, explored.worth))) {
      out.pool = std::move(pool);
      return out;
    }
    if (held) {
      out.waiting = true;
      out.pool = std::move(pool);
      return out;
    }
    begin(1, twice(first == rounds.rend() ? u64{halving_.roundCalls} : first->calls), next);
  }
  return out;
}

void Scheduler::rankByShape(std::vector<Item>& items) const {
  auto const shapeOf = [&](const Item& i) {
    const Baseline& b = baselines_[i.index];
    return std::tuple{b.fft.shape.spec(), b.kind, b.band.regime.label()};
  };
  std::map<std::tuple<std::string, TestKind, std::string>, double> best;
  for (const Item& i : items) {
    double& rate = best[shapeOf(i)];
    rate = std::max(rate, i.rate());
  }
  auto const rank = [&](const Item& i) {
    const FFTConfig& fft = baselines_[i.index].fft;
    return std::tuple{-best.at(shapeOf(i)), shapeOf(i), fft.variant != defaultVariant(fft.shape), -i.rate()};
  };
  std::ranges::stable_sort(items, [&](const Item& a, const Item& b) { return rank(a) < rank(b); });
}

bool Scheduler::swept(const TuneDB& db, u32 env, const Objective& objective) const {
  const Env& device = bootstrap_.env();
  Progress const progress = progressOf(db, env, device);
  GainModel const gains = gainsOf(db, env);
  if (!coverItems(baselineItems(db, env, progress, gains, objective), objective).empty()) { return false; }
  if (!strategy_) { return true; }
  return std::ranges::none_of(
    sweepItems(db, env, progress, readingsOf(optionSetsFor(db, env), device), gains, objective),
    [&](const Item& item) { return baselines_[item.index].band.contains(scope_.probe); });
}

namespace {

bool isSearch(const Item& item) {
  return item.kind == ItemKind::Probe || item.kind == ItemKind::Combo || item.kind == ItemKind::Restart;
}

// Each entry's search items in the places the ranking gave them, but in the order its search takes them: the ranking
// decides how often an entry is served, and its search what it is served with, which a price per kind of step would
// otherwise decide -- a combination or another branch's step waiting on every step it is priced below.
void inSearchOrder(std::vector<Item>& items) {
  std::map<size_t, std::vector<size_t>> places;
  for (size_t k = 0; k < items.size(); ++k) {
    if (isSearch(items[k])) { places[items[k].index].push_back(k); }
  }
  for (const auto& [index, at] : places) {
    std::vector<Item> mine;
    for (size_t const k : at) { mine.push_back(std::move(items[k])); }
    std::ranges::stable_sort(mine, {}, &Item::order);
    for (size_t j = 0; j < at.size(); ++j) { items[at[j]] = std::move(mine[j]); }
  }
}

}  // namespace

std::vector<Item> Scheduler::admissible(const TuneDB& db, u32 env, const Objective& objective, double floor) const {
  BootstrapState const state = bootstrapState(db, env);
  const Env& device = bootstrap_.env();
  Progress const progress = progressOf(db, env, device);
  GainModel const gains = gainsOf(db, env);

  std::vector<Item> out = baselineItems(db, env, progress, gains, objective);
  std::vector<Item> const cover = coverItems(out, objective);
  std::map<size_t, Item> unread;
  for (const Item& item : out) { unread.emplace(item.index, item); }

  // The option sets of each entry, which also say how much each option set costs where it was measured.
  std::vector<OptionSet> const sets = strategy_ || gate_ ? optionSetsFor(db, env) : std::vector<OptionSet>{};
  std::map<EntryKey, std::vector<Reading>> const readings = readingsOf(sets, device);
  std::vector<Item> const sweep =
    strategy_ ? sweepItems(db, env, progress, readings, gains, objective) : std::vector<Item>{};

  // Where no round is recorded, where an earlier build's halving stood is recorded before anything else is measured:
  // calls made from here on, the bootstrap's among them, are not calls of any round of it.  Otherwise the halving
  // stands where it was last found while readings taken by rule go first.
  lastHalving_.unrecorded.clear();
  if (halving_.on() && strategy_ &&
      std::ranges::none_of(db.rounds(), [&](const RoundRow& r) { return db.envOf(r.sess) == env; })) {
    lastHalving_.unrecorded = adoption(standingOf(readings, objective), searchCalls(db, env));
  }

  if (std::vector<Item> gates = gateItems(db, env); !gates.empty()) { return gates; }
  if (!cover.empty()) { return cover; }

  // The sweep at the probe first, since each family is searched on its type's cheapest reading there.
  std::vector<Item> swept = sweep;
  std::ranges::stable_partition(swept,
                                [&](const Item& item) { return baselines_[item.index].band.contains(scope_.probe); });

  // Every family is read before any is searched, since which ones are worth searching is a comparison between them;
  // and none is read or searched until the configuration each is on is recorded.
  bool const chosen = bootstrap_.chosen(db, env);
  std::vector<Item> bootstrap = chosen ? bootstrapReads(state, db, env, progress, objective) : std::vector<Item>{};

  std::ranges::move(reachItems(db, env, sets, objective), std::back_inserter(out));

  // What each entry's search offers priced by rule, which a round of the halving serves whatever the value model makes
  // of it; asked of an entry only once the halving needs to know.
  std::function<std::vector<Item>(size_t)> byRuleOf;
  std::optional<size_t> searched;
  std::optional<Defaults> defaults;

  // The entries worth exploring, and what the lines sweep owes them.
  std::optional<Standing> standing;
  std::vector<size_t> explore;
  std::vector<Item> relines;
  explorable_ = 0;
  linesOwed_ = 0;
  lookOwed_ = 0;

  if (strategy_) {
    defaults = lines(db, env);

    SearchContext const context{.device = device,
                                .strategy = *strategy_,
                                .db = db,
                                .env = env,
                                .progress = progress,
                                .lines = *defaults,
                                .restarts = restarts_};
    auto const offer = [&, context](size_t i, bool rule) {
      const Baseline& b = baselines_[i];
      EntryKey const key = b.key();
      auto const at = readings.find(key);
      if (at == readings.end()) { return std::vector<Item>{}; }

      GainDist const moves = gains.forEntry(key);
      GainDist const combinations = gains.comboForEntry(key);
      std::optional<GainDist> jumps;
      Worth const price = [&](Offer offer, double cost) {
        const GainDist* dist = &moves;
        if (offer == Offer::Combo) { dist = &combinations; }
        if (offer == Offer::Restart) { dist = &(jumps ? *jumps : jumps.emplace(gains.restartForEntry(key))); }
        double const value = expectedSaving(objective.points(), b.kind, b.band, cost, *dist);
        // By rule, nothing is priced out.
        return rule ? std::max(value, std::numeric_limits<double>::min()) : value;
      };
      std::vector<Item> items;
      for (Candidate& c : searches_[i].offers(context, at->second, price)) {
        // A measurement begun is a configuration a call has already read, not a step whose gain on the entry's best
        // set is unknown: it is worth at least what it would save were that reading to hold.
        if (c.calls > 0 && c.observed > 0) {
          c.value = std::max(c.value, saving(objective.points(), b.kind, b.band, c.observed));
        }
        items.push_back(itemOf(i, std::move(c)));
        items.back().order = u32(items.size() - 1);
      }
      return items;
    };
    byRuleOf = [offer](size_t i) { return offer(i, true); };

    if (exploration_.lookCalls > 0 || exploration_.linesSweep) {
      standing = standingOf(readings, objective);
      explore = explorable(*standing, acceptedTypes(*standing), objective);
      relines = linesSweepItems(db, env, context, readings, *standing, explore, objective);
    }

    // The cheapest family still owed its calls, whose search has something to offer.
    if (chosen && bootstrap.empty()) {
      std::vector<const FamilyState*> owed;
      for (const FamilyState& f : state.families) {
        if (f.phase == FamilyPhase::Owed) { owed.push_back(&f); }
      }
      std::ranges::stable_sort(owed, {}, &FamilyState::best);
      for (const FamilyState* f : owed) {
        size_t const i = *entryOf(f->family, state.kind);
        bootstrap = offer(i, true);
        if (!bootstrap.empty()) {
          for (Item& item : bootstrap) { item.bootstrap = true; }
          searched = i;
          break;
        }
      }
    }

    for (size_t i = 0; i < baselines_.size(); ++i) {
      if (i != searched) { std::ranges::move(offer(i, false), std::back_inserter(out)); }
    }

    std::map<EntryKey, size_t> const indexOf = entryIndex();

    // At the exponent and under the options its row was recorded at, so that the call pools with it.
    std::vector<double> const worth = refineValues(sets, objective.points());
    for (size_t s = 0; s < sets.size(); ++s) {
      if (worth[s] <= 0) { continue; }
      const SelectionEntry& e = sets[s].entry;
      auto const at = indexOf.find({e.fft, e.kind, e.regime.label()});
      if (at == indexOf.end()) { continue; }
      const Baseline& b = baselines_[at->second];

      UseConfig const canonical = canonicalConfig(device, b.fft, e.opts);
      Item item{.kind = ItemKind::Refine,
                .index = at->second,
                .options = e.opts,
                .moved = {},
                .what = canonical.empty() ? "the built-in defaults" : configText(canonical),
                .exponent = sets[s].exponent,
                .value = worth[s],
                .seconds = 0,
                .fresh = true,
                .calls = sets[s].m.calls};
      if (auto const n = unrecordedRefines_.find(keyOf(item));
          n != unrecordedRefines_.end() && n->second >= MAX_ATTEMPTS) {
        continue;
      }
      if (u32 const cfg = db.findCfgId(item.options); cfg && db.diedOn(env, cfg, e.kind, e.fft, item.exponent)) {
        continue;
      }

      item.fresh = !built_.contains(builtKey(b.fft, item.options));
      item.seconds = clock_.seconds(sets[s].m.cost(), item.fresh);
      out.push_back(std::move(item));
    }
  }

  std::ranges::stable_sort(out, [](const Item& a, const Item& b) { return a.rate() > b.rate(); });
  inSearchOrder(out);

  // While the halving is on, its round is all the search there is: the steps of the contenders still short of their
  // calls, priced by rule, since a round is there to find what the value model cannot see coming.  A contender whose
  // search the bootstrap has is served there, and its calls count for both.
  std::set<size_t> worth;
  for (const std::vector<Item>* items : {&out, &bootstrap}) {
    for (const Item& item : *items) {
      if (isSearch(item) && item.value > 0 && item.value >= floor) { worth.insert(item.index); }
    }
  }
  std::map<size_t, std::vector<Item>> explored;
  auto const exploring = [&](size_t i) -> const std::vector<Item>& {
    auto const [at, fresh] = explored.try_emplace(i);
    if (fresh) {
      if (i == searched) {
        at->second = bootstrap;
      } else if (byRuleOf) {
        at->second = byRuleOf(i);
        std::erase_if(at->second, [](const Item& item) { return !isSearch(item); });
      }
    }
    return at->second;
  };
  Explored const asked{
    .offers = [&](size_t i) { return !exploring(i).empty(); },
    .resumes =
      [&](size_t i) { return std::ranges::any_of(exploring(i), [](const Item& item) { return item.calls > 0; }); },
    .worth = [&](size_t i) { return worth.contains(i); }};

  // A first look is taken an entry at a time, nearest the fastest first: one with no reading is read at the built-in
  // defaults, and one with a reading has what its search offers, in its own order and whatever it is priced at, until
  // it has had its calls.  Not the entry the bootstrap is searching, which has its calls there.
  std::vector<Item> looks;
  if (exploration_.lookCalls > 0 && standing) {
    std::vector<u64> const calls = searchCalls(db, env);
    std::vector<std::optional<double>> const fastest = measuredAt(objective);
    for (size_t const i : explore) {
      if (!nearFastest(i, *standing, fastest, objective, exploration_.lookMargin)) { continue; }
      ++explorable_;
      if (calls[i] >= exploration_.lookCalls) { continue; }
      ++lookOwed_;
      if (!looks.empty() || i == searched) { continue; }
      if (readings.contains(baselines_[i].key())) {
        looks = exploring(i);
      } else if (auto const at = unread.find(i); at != unread.end()) {
        looks = {at->second};
        looks.front().what = "at the built-in defaults";
      }
      for (Item& item : looks) { item.look = true; }
    }
  }

  // No halving begins before the work ahead of it is done, since each part of that work can change who contends.
  bool const relining = !relines.empty() || (defaults && linesDue(db, env, state, *defaults, true));
  bool const held = !swept.empty() || !state.complete || relining || !looks.empty();
  lastHalving_ = halvingState(db, env, readings, objective, asked, held);
  lastHalving_.waiting = lastHalving_.waiting && swept.empty() && state.complete && relines.empty() && looks.empty();

  // A round once begun runs to its end.
  if (lastHalving_.active) {
    std::map<size_t, u64> owed;
    std::vector<Item> round;
    for (size_t k = 0; k < lastHalving_.pool.size(); ++k) {
      size_t const i = lastHalving_.pool[k];
      owed.emplace(i, lastHalving_.calls[k]);
      if (i == searched) { continue; }
      // An entry that has had its calls still finishes a step it began, so that the calls it made count for something.
      for (const Item& item : exploring(i)) {
        if (lastHalving_.calls[k] >= lastHalving_.budget && !item.calls) { continue; }
        round.push_back(item);
        round.back().halving = true;
      }
    }
    // The contender furthest from its calls first, so that a round is spread across its contenders as it goes; each
    // contender's steps in its search's order.
    std::ranges::stable_sort(round, [&](const Item& a, const Item& b) { return owed.at(a.index) < owed.at(b.index); });
    inSearchOrder(round);
    if (!round.empty()) { return round; }
  }

  if (!swept.empty()) { return swept; }
  if (!bootstrap.empty()) { return bootstrap; }
  if (!relines.empty()) { return relines; }
  if (!looks.empty()) { return looks; }
  return out;
}

Phase Scheduler::phase(const BootstrapState& state, const std::vector<Item>& ranked, const Objective& objective,
                       double floor) const {
  auto const any = [&](auto&& pred) { return std::ranges::any_of(ranked, pred); };
  char buf[256];

  if (any([](const Item& i) { return i.kind == ItemKind::Gate; })) {
    auto const n = std::ranges::count_if(ranked, [](const Item& i) { return i.kind == ItemKind::Gate; });
    snprintf(buf, sizeof(buf), "accuracy gate: %u %s owed", u32(n), n == 1 ? "reading" : "readings");
    return {buf, "gate " + std::to_string(n)};
  }

  if (any([](const Item& i) { return i.cover; })) {
    snprintf(buf, sizeof(buf), "covering the workload: %.1f%% of its weight measured", 100 * objective.measured());
    std::string const text = buf;
    snprintf(buf, sizeof(buf), "cover %.0f%%", 100 * objective.measured());
    return {text, buf};
  }

  // What takes turns, in the order the turns go round.
  std::vector<Phase> parts;
  if (any([](const Item& i) { return i.sweep; })) {
    u32 const read = sweepWithin_ - std::min(sweepOwed_, sweepWithin_);
    snprintf(buf, sizeof(buf), "defaults sweep: %u of %u FFTs read at the built-in defaults", read, sweepWithin_);
    parts.push_back({buf, "sweep " + std::to_string(read) + "/" + std::to_string(sweepWithin_)});
  }
  if (any([](const Item& i) { return i.linesSweep; })) {
    u32 const read = linesWithin_ - std::min(linesOwed_, linesWithin_);
    snprintf(buf, sizeof(buf), "lines sweep: %u of %u FFTs read under the default lines", read, linesWithin_);
    parts.push_back({buf, "lines " + std::to_string(read) + "/" + std::to_string(linesWithin_)});
  }
  if (any([](const Item& i) { return i.look; })) {
    u32 const done = explorable_ - std::min(lookOwed_, explorable_);
    snprintf(buf, sizeof(buf), "first looks: %u of %u FFTs have had their %u calls of search", done, explorable_,
             exploration_.lookCalls);
    parts.push_back({buf, "look " + std::to_string(done) + "/" + std::to_string(explorable_)});
  }

  if (auto const first = std::ranges::find_if(ranked, &Item::bootstrap); first != ranked.end()) {
    u32 types = 0;
    u32 typesDone = 0;
    const FamilyState* searched = nullptr;
    for (const FamilyState& f : state.families) {
      if (f.phase != FamilyPhase::Owed && f.phase != FamilyPhase::Served) { continue; }
      ++types;
      typesDone += f.phase == FamilyPhase::Served;
      if (entryOf(f.family, state.kind) == first->index) { searched = &f; }
    }
    if (first->kind == ItemKind::Baseline || !searched) {
      parts.push_back({"bootstrap: reading each FFT type at its defaults", "bootstrap"});
    } else {
      std::string const name = std::string{typeName(searched->family.type)} + " " + baselines_[first->index].label();
      u64 const calls = std::min(searched->calls, state.budget);
      snprintf(buf, sizeof(buf), "bootstrap: %s, %" PRIu64 " of %" PRIu64 " calls of search; type %u of %u",
               name.c_str(), calls, state.budget, typesDone + 1, types);
      std::string const text = buf;
      snprintf(buf, sizeof(buf), "bootstrap %" PRIu64 "/%" PRIu64, calls, state.budget);
      parts.push_back({text, buf});
    }
  }

  if (lastHalving_.active && any([](const Item& i) { return i.halving; })) {
    const HalvingState& h = lastHalving_;
    u32 rounds = h.round;
    for (size_t n = h.pool.size(); n > 1; n = (n + 1) / 2) { ++rounds; }
    u64 done = 0;
    for (u64 const calls : h.calls) { done += std::min(h.budget, calls); }
    std::string const which = h.halving > 1 ? "halving " + std::to_string(h.halving) : "halving";
    snprintf(buf, sizeof(buf), "%s: round %u of %u, %zu contenders, %" PRIu64 " of %" PRIu64 " calls", which.c_str(),
             h.round + 1, std::max(rounds, h.round + 1), h.pool.size(), done, h.budget * h.pool.size());
    std::string const text = buf;
    snprintf(buf, sizeof(buf), "halving %u/%u", h.round + 1, std::max(rounds, h.round + 1));
    parts.push_back({text, buf});
  } else if (any([floor](const Item& i) {
               return !i.sweep && !i.bootstrap && !i.linesSweep && !i.look && worthRunning(i, floor);
             })) {
    const HalvingState& h = lastHalving_;
    if (h.leading && !h.calls.empty()) {
      snprintf(buf, sizeof(buf), "searching by expected gain, %" PRIu64 " of %" PRIu64 " calls before the next halving",
               std::min(h.calls.front(), h.budget), h.budget);
      parts.push_back({buf, "search"});
    } else {
      parts.push_back({"searching by expected gain", "search"});
    }
  }

  if (parts.empty()) { return {"nothing left worth running", "done"}; }
  Phase out = parts.front();
  for (size_t k = 1; k < parts.size(); ++k) {
    out.text += " + " + parts[k].text;
    out.brief += " + " + parts[k].brief;
  }
  return out;
}

bool byRule(const Item& item) {
  return item.bootstrap || item.kind == ItemKind::Gate || item.cover || item.sweep || item.look || item.linesSweep ||
    item.halving || (isSearch(item) && item.calls > 0);
}

bool worthRunning(const Item& item, double floor) {
  if (byRule(item)) { return true; }
  return item.value > 0 && item.value >= floor;
}

std::optional<Item> Scheduler::pick(const std::vector<Item>& ranked, double floor) const {
  auto const worth = [floor](const Item& i) { return worthRunning(i, floor); };
  auto const top = std::ranges::find_if(ranked, worth);
  if (top == ranked.end()) { return {}; }
  if (keyOf(*top) != last_) { return *top; }

  for (auto it = std::next(top); it != ranked.end(); ++it) {
    if (worth(*it) && it->rate() >= (1 - INTERLEAVE_EPS) * top->rate()) { return *it; }
  }
  return *top;
}

void Scheduler::ran(const Item& item, double seconds, double usPerIt, bool recorded) {
  last_ = keyOf(item);
  if (item.kind == ItemKind::Anchor) { return; }

  if (usPerIt > 0) { clock_.observe(seconds, usPerIt, item.fresh); }

  if (item.kind == ItemKind::Gate || item.kind == ItemKind::Reach) {
    ++gateAttempts_[last_];
    return;
  }

  built_.insert(builtKey(baselines_[item.index].fft, item.options));
  switch (item.kind) {
  case ItemKind::Probe:
  case ItemKind::Combo:
  case ItemKind::Restart: searches_[item.index].tried(bootstrap_.env(), item.options); return;
  case ItemKind::Refine:
    if (!recorded || usPerIt <= 0) { ++unrecordedRefines_[last_]; }
    return;
  case ItemKind::Baseline: ++attempts_[item.index]; return;
  case ItemKind::Anchor:
  case ItemKind::Gate:
  case ItemKind::Reach: return;
  }
}

Defaults recordedLines(const TuneDB& db, const LinesRow& row) {
  Defaults out;
  if (const UseConfig* const global = db.findCfg(row.global)) { out.global = *global; }
  for (const auto& [type, cfg] : row.family) {
    const UseConfig* const uses = db.findCfg(cfg);
    if (!uses) { continue; }
    FFTSelector selector;
    selector.type = type;
    out.family.push_back({.selector = selector, .uses = {uses->begin(), uses->end()}});
  }
  return out;
}

std::string linesText(const Defaults& lines) {
  std::string out = configText(lines.global);
  for (const UseLine& line : lines.family) {
    UseConfig const uses{line.uses.begin(), line.uses.end()};
    out += "; ! " + line.selector.spec() + " " + configText(uses);
  }
  return out;
}

namespace {

// Says once what the bootstrap has come to: each family it will not search and why, each family as its search begins
// and as it has had its calls, and the lines once every family is settled.  Nothing while the configuration each family
// is on is still to be chosen, since until then it is only the cheapest read so far.
class BootstrapLog {
public:
  void report(const BootstrapState& state, bool enabled, bool chosen, const std::string& lines) {
    if (enabled && !chosen) { return; }
    for (const FamilyState& f : state.families) {
      std::string const name = std::string{typeName(f.family.type)} + " " + f.family.fft.spec() +
        (state.kind == TestKind::PRP ? "" : std::string{" "} + toString(state.kind));
      if (!said_.insert(name + " " + toString(f.phase)).second) { continue; }

      switch (f.phase) {
      case FamilyPhase::Skipped:
        if (!enabled) {
          log("tune: bootstrap: %s is not searched first, bootstrapping being turned off\n", name.c_str());
        } else if (f.reading) {
          log("tune: bootstrap: %s is not searched first: at %.3f us/it it would take more than a %.0f%% gain to bring "
              "it level with the cheapest type\n",
              name.c_str(), f.best, 100 * BOOTSTRAP_GAIN);
        }
        break;
      case FamilyPhase::Held:
        log("tune: bootstrap: %s cannot be measured at the built-in defaults, so it is not searched first\n",
            name.c_str());
        break;
      case FamilyPhase::Owed:
        log("tune: bootstrap: %s is searched first, for %" PRIu64 " calls, from %.3f us/it at the built-in defaults\n",
            name.c_str(), state.budget, f.reading);
        break;
      case FamilyPhase::Served:
        log("tune: bootstrap: %s has had its %" PRIu64 " calls of search, and is at %.3f us/it\n", name.c_str(),
            state.budget, f.best);
        break;
      case FamilyPhase::Unread: break;
      }
    }

    if (enabled && state.complete && !complete_) {
      complete_ = true;
      log("tune: bootstrap complete; the default lines are %s\n", lines.c_str());
    }
  }

private:
  std::set<std::string> said_;
  bool complete_ = false;
};

// Whether a combo row already declares this configuration: a stop can cut the first call short after its declaration.
[[nodiscard]] bool declaredCombo(const TuneDB& db, u32 envId, const Env& env, const FFTConfig& fft, TestKind kind,
                                 u64 exponent, const UseConfig& options) {
  std::string const spec = fft.spec();
  std::string const regime = regimeOf(fft, exponent).label();
  UseConfig const canonical = canonicalConfig(env, fft, options);
  return std::ranges::any_of(db.combos(), [&](const ComboRow& row) {
    const UseConfig* const opts = db.findCfg(row.cfg);
    return opts && db.envOf(row.sess) == envId && row.fft == spec && row.kind == kind && row.regime.label() == regime &&
      canonicalConfig(env, fft, *opts) == canonical;
  });
}

// What the gate made of `item`'s subject once its reading is in, for the log.
[[nodiscard]] std::string gateOutcome(const TuneDB& db, u32 envId, const Env& env, const FFTConfig& fft,
                                      const Item& item) {
  GateVerdict const verdict = Gates{db, envId, env}(fft, item.span, item.subject);
  switch (verdict.state) {
  case GateState::Passed:
    return std::string{"passed, "} + toString(verdict.evidence) +
      (verdict.derived ? " up to " + std::to_string(verdict.reach) + ", the reach derived for it" : "");
  case GateState::Rejected: return "rejected: " + verdict.why + ", so it is never published";
  case GateState::Owed:
    return verdict.owesReference ? "owes the reading of its default accuracy"
      : verdict.owedAt != item.exponent
      ? "short of its standard; its reach is read next at " + std::to_string(verdict.owedAt)
      : "still owed";
  }
  return "?";
}

// What a reach reading made of its set's reach, for the log.
[[nodiscard]] std::string reachOutcome(const TuneDB& db, u32 envId, const Env& env, const FFTConfig& fft,
                                       const Item& item) {
  GateVerdict const verdict = Gates{db, envId, env}(fft, item.span, item.subject);
  if (verdict.state == GateState::Passed && verdict.reach > item.span.hi) {
    return "raised to " + std::to_string(verdict.reach) + ", above the table's " + std::to_string(item.span.hi);
  }
  if (verdict.state == GateState::Passed && verdict.raiseAt) {
    return "not yet confirmed; read next at " + std::to_string(verdict.raiseAt);
  }
  return "no raise confirmed, so it stays at the table's " + std::to_string(item.span.hi);
}

// The best option set of the entry `b`, canonical, and its cost; nothing where no row of it could be published.
[[nodiscard]] std::optional<std::string> bestOf(const TuneDB& db, u32 envId, const Env& env, const Baseline& b) {
  std::vector<SelectionEntry> const candidates = candidatesFor(db, envId, Gating::Assumed);
  std::map<EntryKey, const SelectionEntry*> const best = bestEntries(candidates);
  auto const at = best.find({b.fft.spec(), b.kind, b.band.regime.label()});
  if (at == best.end()) { return {}; }

  char cost[32];
  snprintf(cost, sizeof(cost), "%.3f us/it", at->second->cost);
  std::string const opts = configText(canonicalConfig(env, b.fft, at->second->opts));
  return (opts.empty() ? std::string{"the built-in defaults"} : opts) + ", " + cost;
}

}  // namespace

QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench, const Publisher& publish, double stop,
                     Watch* watch) {
  QueueReport out;
  BootstrapLog bootstrapLog;
  bool const bootstrapping = scheduler.bootstrap().enabled();

  // What is published, and what the items are valued against: the same but for the sets the gate still owes, whose
  // readings are taken before anything is valued.
  BootstrapState state = scheduler.bootstrapState(db, env);
  Objective objective{db, env, scheduler.scope()};
  Objective valuing{db, env, scheduler.scope(), Gating::Assumed};
  Defaults lines = scheduler.lines(db, env);
  std::string said = linesText(lines);
  out.startT = objective.T();
  publish(objective, lines);

  // The halving's round as last said, so that each is said once as it begins.
  std::optional<u32> saidRound;

  auto rescore = [&] {
    state = scheduler.bootstrapState(db, env);
    objective = Objective{db, env, scheduler.scope()};
    valuing = Objective{db, env, scheduler.scope(), Gating::Assumed};
    lines = scheduler.lines(db, env);
    std::string text = linesText(lines);
    if (text != said) { log("tune: the default lines are now %s, from the best sets published\n", text.c_str()); }
    bootstrapLog.report(state, bootstrapping, scheduler.bootstrap().chosen(db, env), text);
    said = std::move(text);
  };

  // A lines sweep is recorded as it becomes due: once the bootstrap is done, once a halving has ended, and where the
  // lines have changed since the last, as the next halving would begin.
  auto const relines = [&](bool waiting) {
    std::optional<Defaults> const due = scheduler.linesDue(db, env, state, lines, waiting);
    if (!due) { return false; }
    LinesRow row{.sess = 0, .n = 1, .after = 0, .ts = 0, .global = db.internCfg(due->global), .family = {}};
    for (const LinesRow& r : db.lines()) {
      if (db.envOf(r.sess) == env) { row.n = std::max(row.n, r.n + 1); }
    }
    for (const RoundRow& r : db.rounds()) {
      if (db.envOf(r.sess) == env) { row.after = std::max(row.after, r.n); }
    }
    for (const UseLine& line : due->family) {
      if (line.selector.type) {
        row.family.emplace_back(*line.selector.type, db.internCfg({line.uses.begin(), line.uses.end()}));
      }
    }
    bench.declareLines(row);
    double const margin = scheduler.exploration().linesMarginOf(row.n);
    char within[64] = "";
    if (!std::isinf(margin)) { snprintf(within, sizeof(within), " within %g%% of the fastest", 100 * margin); }
    log("tune: lines sweep %u: every FFT worth exploring%s is read under the default lines %s\n", row.n, within,
        linesText(*due).c_str());
    return true;
  };

  while (!bench.stopped()) {
    // Once the defaults sweep is done, which configuration each family is searched on is recorded, so that it stays put
    // as its search adds readings.
    if (bootstrapping && !bench.anchorDue() && !scheduler.bootstrap().chosen(db, env) &&
        scheduler.swept(db, env, valuing)) {
      std::string names;
      for (const Family& f : scheduler.bootstrap().unrecorded(db, env)) {
        bench.declareBootstrap(f.fft, scheduler.bootstrap().probe());
        names += (names.empty() ? "" : ", ") + std::string{typeName(f.type)} + " " + f.fft.spec();
      }
      log("tune: bootstrap at %" PRIu64 " over %s\n", scheduler.bootstrap().probe(), names.c_str());
      rescore();
    }

    (void)relines(false);
    std::vector<Item> ranked = scheduler.admissible(db, env, valuing, stop * valuing.T());
    if (scheduler.lastHalving().waiting && relines(true)) {
      ranked = scheduler.admissible(db, env, valuing, stop * valuing.T());
    }
    const HalvingState& h = scheduler.lastHalving();
    for (const RoundRow& round : h.unrecorded) { bench.declareRound(round); }
    if (h.active && saidRound != h.n) {
      std::string names;
      for (size_t const i : h.pool) { names += (names.empty() ? "" : ", ") + scheduler.baselines()[i].fft.spec(); }
      std::string const which = h.halving > 1 ? "halving " + std::to_string(h.halving) : "halving";
      log("tune: %s: round %u, %zu of the %u contenders, %" PRIu64 " calls of search each in this round: %s\n",
          which.c_str(), h.round + 1, h.pool.size(), h.contenders, h.budget, names.c_str());
      saidRound = h.n;
    } else if (!h.active && saidRound) {
      if (h.leading) {
        log("tune: halving done; %s is left, and the search is ranked by what it is expected to gain for %" PRIu64
            " calls of search, when the next halving begins\n",
            scheduler.baselines()[h.pool.front()].label().c_str(), h.budget);
      } else {
        log("tune: halving done; %s is left, and the search is ranked by what it is expected to gain from here\n",
            h.pool.empty() ? "nothing" : scheduler.baselines()[h.pool.front()].label().c_str());
      }
      saidRound.reset();
    }
    // From the ranking the pick is made from, so that the figures are the ones the stopping rule is about to use.
    if (watch) {
      RunProgress p =
        progressOf(out, objective.T(), objective.measured(), ranked, stop, stop * valuing.T(), state.complete);
      p.phase = scheduler.phase(state, ranked, objective, stop * valuing.T());
      watch->progress(p);
      watch->state({.scheduler = scheduler,
                    .db = db,
                    .env = env,
                    .objective = objective,
                    .ranked = ranked,
                    .floor = stop * valuing.T()});
    }
    std::optional<Item> const item = scheduler.pick(ranked, stop * valuing.T());
    if (!item) {
      out.end =
        std::ranges::any_of(ranked, [](const Item& i) { return i.value > 0; }) ? QueueEnd::BelowStop : QueueEnd::Dry;
      break;
    }

    // Scheduled by the clock rather than by value, and ahead of everything else when it is due, but only while there
    // is something to divide by it: the first reading is what every row of the session is divided by.
    if (bench.anchorDue()) {
      if (watch) { watch->measuring("the drift anchor"); }
      bench.timeAnchor();
      scheduler.ran({.kind = ItemKind::Anchor}, 0, 0);
      ++out.anchors;

      // The first one may have been a race, whose readings are rows like any other.
      rescore();
      continue;
    }

    const Baseline* const baseline = item->kind != ItemKind::Reach ? &scheduler.baselines()[item->index] : nullptr;
    const FFTConfig& fft = item->fft ? *item->fft : baseline->fft;
    TestKind const kind = baseline ? baseline->kind : TestKind::PRP;
    bool const reads = item->kind == ItemKind::Gate || item->kind == ItemKind::Reach;
    bool const probing = baseline && item->kind != ItemKind::Baseline && !reads;
    std::string const label = (!baseline              ? item->what
                                 : item->what.empty() ? baseline->label()
                                                      : baseline->label() + " " + item->what) +
      (item->look ? ", its first look" : "");
    std::optional<std::string> const bestBefore =
      probing ? bestOf(db, env, scheduler.bootstrap().env(), *baseline) : std::nullopt;

    // Once, before its first call; a resumed one was declared by the process that started it.
    if (item->kind == ItemKind::Restart && item->calls == 0) {
      bench.declareRestart(fft, kind, item->exponent, item->options, item->draw);
    }
    if (item->tier > 1 && item->calls == 0 &&
        !declaredCombo(db, env, scheduler.bootstrap().env(), fft, kind, item->exponent, item->options)) {
      bench.declareCombo(fft, kind, item->exponent, item->options, item->tier);
    }
    // A refine is the next call of a row that is already concluded, not a resumption of one that was interrupted.
    bool const counted = item->kind == ItemKind::Refine;
    std::string const call = counted ? " (call " + std::to_string(item->calls + 1) + ")"
      : item->calls                  ? " (resumed at call " + std::to_string(item->calls + 1) + ")"
                                     : "";
    if (watch) {
      watch->measuring(std::to_string(out.items + 1) + ". " + toString(item->kind) + " " + label + " at " +
                       std::to_string(item->exponent) + (reads ? "" : call));
    }

    Bench::Result result;
    Bench::Reading reading;
    if (reads) {
      reading = bench.gate(fft, item->exponent, item->options);
      result = {.completed = reading.completed, .seconds = reading.seconds, .usPerIt = 0, .ran = reading.ran};
    } else {
      result = bench.run(fft, kind, item->exponent, item->options, item->moved);
    }

    // A call cut short by a stop recorded nothing, so there is nothing to account for.
    if (!result.completed && bench.stopped()) { break; }

    // A call other than a baseline's counts for its item only if its kernels were built as asked.  Where the host sets
    // a value aside the row lands on another configuration, and without this the same one would be asked for for ever.
    bool recorded = result.completed;
    if (item->kind != ItemKind::Baseline && recorded) {
      const Env& device = scheduler.bootstrap().env();
      UseConfig const built = canonicalConfig(device, fft, result.ran);
      if (built != canonicalConfig(device, fft, item->options)) {
        recorded = false;
        log("tune: %s was built as %s, so its calls cannot count for it\n", label.c_str(), configText(built).c_str());
      }
    }

    scheduler.ran(*item, result.seconds, result.completed ? result.usPerIt : 0, recorded);
    ++out.items;
    ++out.spent[item->kind].items;
    out.spent[item->kind].seconds += result.seconds;

    double const before = objective.T();
    rescore();
    publish(objective, lines);

    std::string outcome;
    std::optional<Measurement> row;
    if (reads && result.completed) {
      const Env& device = scheduler.bootstrap().env();
      outcome = item->kind == ItemKind::Gate ? gateOutcome(db, env, device, fft, *item)
                                             : reachOutcome(db, env, device, fft, *item);
      log("tune: %u. %s %s at %" PRIu64 ": z %.2f over %u rounding errors, check %s, %.1f s -- %s; T %.3f -> %.3f "
          "us/it\n",
          out.items, toString(item->kind), label.c_str(), item->exponent, reading.z, reading.n,
          reading.checkOk ? "OK" : "failed", result.seconds, outcome.c_str(), before, objective.T());
    } else if (result.completed) {
      row = rowOf(db, env, fft, kind, item->exponent, result.ran);
      std::string ranked;
      if (row) {
        char buf[64];
        snprintf(buf, sizeof(buf), ", ranked at %.3f over %u call%s", pessimisticCost(*row), row->calls,
                 row->calls == 1 ? "" : "s");
        ranked = buf;
      }
      log("tune: %u. %s %s at %" PRIu64 "%s: %.3f us/it%s, %.1f s; T %.3f -> %.3f us/it\n", out.items,
          toString(item->kind), label.c_str(), item->exponent, call.c_str(), result.usPerIt, ranked.c_str(),
          result.seconds, before, objective.T());
    } else if (result.status == Status::Err) {
      log("tune: %u. %s %s at %" PRIu64 "%s computed wrongly twice, and is held out as an error\n", out.items,
          toString(item->kind), label.c_str(), item->exponent, call.c_str());
    } else {
      log("tune: %u. %s %s at %" PRIu64 "%s gave no reading\n", out.items, toString(item->kind), label.c_str(),
          item->exponent, call.c_str());
    }

    std::string best;
    if (probing) {
      std::optional<std::string> const bestAfter = bestOf(db, env, scheduler.bootstrap().env(), *baseline);
      if (bestAfter && bestAfter != bestBefore) {
        log("tune: %s is now best at %s\n", baseline->label().c_str(), bestAfter->c_str());
        best = *bestAfter;
      }
    }
    if (watch) {
      watch->finished({.n = out.items,
                       .kind = item->kind,
                       .label = label,
                       .bootstrap = item->bootstrap,
                       .exponent = item->exponent,
                       .call = reads ? "" : call,
                       .completed = result.completed,
                       .seconds = result.seconds,
                       .usPerIt = result.usPerIt,
                       .ranked = row ? pessimisticCost(*row) : 0,
                       .calls = row ? row->calls : 0,
                       .reads = reads,
                       .z = reading.z,
                       .checkOk = reading.checkOk,
                       .outcome = outcome,
                       .before = before,
                       .after = objective.T(),
                       .best = std::move(best)});
    }
  }

  out.stopped = bench.stopped();
  if (out.stopped) { out.end = QueueEnd::Stopped; }
  out.endT = objective.T();

  out.left = scheduler.admissible(db, env, valuing, stop * valuing.T());
  out.valuedT = valuing.T();
  out.floor = stop * valuing.T();
  return out;
}

}  // namespace tune
