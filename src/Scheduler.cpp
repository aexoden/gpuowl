// Copyright (C) Jason Lynch

#include "Scheduler.h"

#include "Args.h"
#include "FFTVariants.h"
#include "log.h"
#include "Primes.h"
#include "Progress.h"

#include <algorithm>
#include <cinttypes>
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
  std::vector<const SelectionEntry*> order;
  for (const OptionSet& s : sets) { order.push_back(&s.entry); }
  std::ranges::stable_sort(order, [](const SelectionEntry* a, const SelectionEntry* b) {
    return std::tuple{a->cost, a->id} < std::tuple{b->cost, b->id};
  });

  std::map<EntryKey, std::vector<Reading>> out;
  for (const SelectionEntry* e : order) {
    auto const fft = parseFft(e->fft);
    if (!fft) { continue; }
    out[{e->fft, e->kind, e->regime.label()}].push_back(
      {.config = canonicalConfig(env, *fft, e->opts), .cost = e->cost});
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
                     std::optional<Strategy> strategy, bool restarts, bool gate, Halving halving) :
  scope_{std::move(scope)},
  baselines_{std::move(baselines)},
  clock_{blockSize},
  bootstrap_{std::move(bootstrap)},
  strategy_{std::move(strategy)},
  restarts_{restarts},
  gate_{gate},
  halving_{halving} {
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
          .seconds = clock_.seconds(candidate.cost, fresh),
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
    if (db.isNogo(env, e.fft, item.options)) { continue; }
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
    if (db.isNogo(env, e.fft, item.options)) { continue; }
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
    if (db.isNogo(env, spec, options)) { continue; }
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
    o.runnable = !o.read && (tried == attempts_.end() || tried->second < MAX_ATTEMPTS) && !db.isNogo(env, spec, {}) &&
      !(cfg && db.diedOn(env, cfg, b.kind, spec, o.exponent));
  }

  // What each point is held to: what production is measured to run there, or the prior of an entry not read yet that
  // serves it, if cheaper -- the cheapest are read first, and each reading replaces its prior.  Not the prior of one
  // that cannot be read, which nothing would ever replace, nor a reading production does not run there.
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
        o.estimate <= (1 + CONTEND_MARGIN) * *fastest[p];
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

std::vector<std::optional<double>> Scheduler::gapsOf(const std::map<EntryKey, std::vector<Reading>>& readings,
                                                     const Objective& objective) const {
  std::vector<std::optional<double>> const fastest = measuredAt(objective);

  // Only against what is measured: an estimate says what to read next, and a prior that is wrong, or can never be
  // read, must not decide which of the entries already read is searched.
  std::vector<std::optional<double>> out(baselines_.size());
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    auto const measured = readings.find({b.fft.spec(), b.kind, b.band.regime.label()});
    if (measured == readings.end()) { continue; }
    for (size_t p = 0; p < objective.points().size(); ++p) {
      const ObjectivePoint& point = objective.points()[p];
      if (!fastest[p] || point.kind != b.kind || !b.band.contains(point.exponent)) { continue; }
      double const gap = measured->second.front().cost / *fastest[p] - 1;
      out[i] = std::min(out[i].value_or(gap), gap);
    }
  }
  return out;
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

std::vector<size_t> Scheduler::poolOf(const std::vector<std::optional<double>>& gaps,
                                      const std::function<bool(size_t)>& eligible, u32 limit) const {
  // Each shape's variants ranked by gap, and the pool filled rank by rank.
  std::map<std::tuple<std::string, TestKind>, std::vector<size_t>> byShape;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    if (gaps[i] && *gaps[i] <= CONTEND_MARGIN && eligible(i)) {
      byShape[{baselines_[i].fft.shape.spec(), baselines_[i].kind}].push_back(i);
    }
  }
  std::vector<std::tuple<size_t, double, size_t>> ranked;
  for (auto& [shape, members] : byShape) {
    std::ranges::stable_sort(members, [&](size_t a, size_t b) { return *gaps[a] < *gaps[b]; });
    for (size_t r = 0; r < members.size(); ++r) { ranked.emplace_back(r, *gaps[members[r]], members[r]); }
  }
  std::ranges::sort(ranked);

  std::vector<size_t> out;
  for (const auto& [rank, gap, i] : ranked) {
    if (out.size() < limit) { out.push_back(i); }
  }
  std::ranges::sort(out, [&](size_t a, size_t b) { return std::tuple{*gaps[a], a} < std::tuple{*gaps[b], b}; });
  return out;
}

RoundMember Scheduler::memberOf(size_t index, u64 from) const {
  const Baseline& b = baselines_[index];
  return {.fft = b.fft.spec(), .kind = b.kind, .regime = b.band.regime, .from = from};
}

std::vector<RoundRow> Scheduler::adoption(const std::vector<std::optional<double>>& gaps,
                                          const std::vector<u64>& calls) const {
  std::vector<RoundRow> const fresh{{.sess = 0, .n = 1, .round = 0, .calls = 0, .ts = 0, .members = {}}};
  std::vector<size_t> const first = poolOf(gaps, [](size_t) { return true; }, halving_.contenders);
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
                                     const Objective& objective, const std::set<size_t>& offering,
                                     const std::set<size_t>& resuming) const {
  HalvingState out;
  if (!halving_.on() || !strategy_) { return out; }

  std::vector<std::optional<double>> const gaps = gapsOf(readings, objective);
  std::vector<u64> const calls = searchCalls(db, env);
  std::map<EntryKey, size_t> const indexOf = entryIndex();

  std::vector<RoundRow> rounds;
  for (const RoundRow& r : db.rounds()) {
    if (db.envOf(r.sess) == env) { rounds.push_back(r); }
  }
  std::ranges::stable_sort(rounds, {}, &RoundRow::n);
  if (rounds.empty()) {
    out.unrecorded = adoption(gaps, calls);
    rounds = out.unrecorded;
  }

  auto begin = [&](u32 round, u64 budget, const std::vector<size_t>& members) {
    RoundRow row{.sess = 0, .n = rounds.back().n + 1, .round = round, .calls = budget, .ts = 0, .members = {}};
    for (size_t const i : members) { row.members.push_back(memberOf(i, calls[i])); }
    out.unrecorded.push_back(row);
    rounds.push_back(std::move(row));
  };
  auto byGap = [&](size_t a, size_t b) {
    return std::tuple{!gaps[a], gaps[a].value_or(0), a} < std::tuple{!gaps[b], gaps[b].value_or(0), b};
  };

  // Each pass either finds the round under way, or records one that has just ended or begun, so that a round whose
  // entries have nothing to take is passed over at once.  A halving can begin only for entries never in a round, and
  // each round of one is smaller than the last.
  for (u32 pass = 0; pass < 2 * baselines_.size() + 64; ++pass) {
    auto const last =
      std::ranges::find_if(rounds.rbegin(), rounds.rend(), [](const RoundRow& r) { return !r.members.empty(); });
    std::optional<size_t> left;

    if (last != rounds.rend()) {
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
        owed = owed || (had.back() < last->calls && offering.contains(i)) || resuming.contains(i);
      }

      if (last->members.size() > 1) {
        // How many the halving's first round took.
        auto const first = std::find_if(last, rounds.rend(), [](const RoundRow& r) { return r.round == 1; });
        out.contenders = u32(first == rounds.rend() ? last->members.size() : first->members.size());

        if (owed) {
          out.active = true;
          out.round = last->round - 1;
          out.n = last->n;
          out.budget = last->calls;
          out.pool = std::move(pool);
          out.calls = std::move(had);
          return out;
        }

        // Over: its better half go on.
        std::ranges::sort(pool, byGap);
        pool.resize((pool.size() + 1) / 2);
        if (pool.size() > 1) {
          begin(last->round + 1, 2 * last->calls, pool);
          continue;
        }
        if (!pool.empty()) {
          begin(last->round + 1, 0, pool);
          left = pool.front();
        }
      } else if (!pool.empty()) {
        left = pool.front();
      }
    }

    // Those within the margin never in a round, beside the one the last halving left.
    std::set<EntryKey> took;
    for (const RoundRow& r : rounds) {
      for (const RoundMember& m : r.members) { took.insert({m.fft, m.kind, m.regime.label()}); }
    }
    auto const newcomer = [&](size_t i) {
      const Baseline& b = baselines_[i];
      return !took.contains({b.fft.spec(), b.kind, b.band.regime.label()});
    };
    std::vector<size_t> pool = poolOf(gaps, newcomer, halving_.contenders - (left ? 1 : 0));
    if (!pool.empty() && left) { pool.push_back(*left); }
    if (pool.size() > 1) {
      std::ranges::sort(pool, byGap);
      begin(1, halving_.roundCalls, pool);
      continue;
    }

    if (left) { out.pool = {*left}; }
    return out;
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

std::vector<Item> Scheduler::admissible(const TuneDB& db, u32 env, const Objective& objective) const {
  BootstrapState const state = bootstrapState(db, env);
  const Env& device = bootstrap_.env();
  Progress const progress = progressOf(db, env, device);
  GainModel const gains = gainsOf(db, env);

  std::vector<Item> out = baselineItems(db, env, progress, gains, objective);
  std::vector<Item> const cover = coverItems(out, objective);

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
    lastHalving_.unrecorded = adoption(gapsOf(readings, objective), searchCalls(db, env));
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

  if (strategy_) {
    Defaults const defaults = lines(db, env);

    SearchContext const context{.device = device,
                                .strategy = *strategy_,
                                .db = db,
                                .env = env,
                                .progress = progress,
                                .lines = defaults,
                                .restarts = restarts_};
    auto const offer = [&](size_t i, bool rule) {
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
        items.push_back(itemOf(i, std::move(c)));
        items.back().bootstrap = rule;
      }
      return items;
    };

    // The cheapest family still owed its calls, whose search has something to offer.
    std::optional<size_t> searched;
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
      if (db.isNogo(env, e.fft, item.options)) { continue; }
      if (u32 const cfg = db.findCfgId(item.options); cfg && db.diedOn(env, cfg, e.kind, e.fft, item.exponent)) {
        continue;
      }

      item.fresh = !built_.contains(builtKey(b.fft, item.options));
      item.seconds = clock_.seconds(sets[s].m.cost(), item.fresh);
      out.push_back(std::move(item));
    }
  }

  std::ranges::stable_sort(out, [](const Item& a, const Item& b) { return a.rate() > b.rate(); });

  // While the halving is on, its round is all the search there is: the steps of the contenders still short of their
  // calls.  A contender whose search the bootstrap has is served there, and its calls count for both.
  auto const search = [](const Item& i) {
    return i.kind == ItemKind::Probe || i.kind == ItemKind::Combo || i.kind == ItemKind::Restart;
  };
  std::set<size_t> offering;
  std::set<size_t> resuming;
  for (const std::vector<Item>* items : {&out, &bootstrap}) {
    for (const Item& item : *items) {
      if (!search(item)) { continue; }
      offering.insert(item.index);
      if (item.calls) { resuming.insert(item.index); }
    }
  }
  lastHalving_ = halvingState(db, env, readings, objective, offering, resuming);
  if (lastHalving_.active) {
    // An entry that has had its calls still finishes a step it began, so that the calls it made count for something.
    std::map<size_t, u64> owed;
    for (size_t k = 0; k < lastHalving_.pool.size(); ++k) { owed.emplace(lastHalving_.pool[k], lastHalving_.calls[k]); }
    std::vector<Item> round;
    for (Item& item : out) {
      auto const at = owed.find(item.index);
      if (!search(item) || at == owed.end() || (at->second >= lastHalving_.budget && !item.calls)) { continue; }
      item.halving = true;
      round.push_back(std::move(item));
    }
    // The contender furthest from its calls first, so that a round is spread across its contenders as it goes.
    std::ranges::stable_sort(round, [&](const Item& a, const Item& b) { return owed.at(a.index) < owed.at(b.index); });
    out = std::move(round);
  }
  return inTurn({std::move(swept), std::move(bootstrap), std::move(out)});
}

std::vector<Item> Scheduler::inTurn(std::array<std::vector<Item>, TURNS> turns) const {
  size_t const first = lastTurn_ ? (size_t(*lastTurn_) + 1) % TURNS : 0;
  std::vector<Item> out;
  for (size_t at = 0; std::ranges::any_of(turns, [&](const auto& items) { return at < items.size(); }); ++at) {
    for (size_t k = 0; k < TURNS; ++k) {
      std::vector<Item>& items = turns[(first + k) % TURNS];
      if (at < items.size()) { out.push_back(std::move(items[at])); }
    }
  }
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
    snprintf(buf, sizeof(buf), "halving: round %u of %u, %zu contenders, %" PRIu64 " of %" PRIu64 " calls", h.round + 1,
             std::max(rounds, h.round + 1), h.pool.size(), done, h.budget * h.pool.size());
    std::string const text = buf;
    snprintf(buf, sizeof(buf), "halving %u/%u", h.round + 1, std::max(rounds, h.round + 1));
    parts.push_back({text, buf});
  } else if (any([floor](const Item& i) { return !i.sweep && !i.bootstrap && worthRunning(i, floor); })) {
    parts.push_back({"searching by expected gain", "search"});
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
  return item.bootstrap || item.kind == ItemKind::Gate || item.cover || item.sweep || item.halving;
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

  if (item.sweep) {
    lastTurn_ = Turn::Sweep;
  } else if (item.bootstrap) {
    lastTurn_ = Turn::Bootstrap;
  } else if (item.kind != ItemKind::Gate && !item.cover) {
    lastTurn_ = Turn::Search;
  }

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

    std::vector<Item> const ranked = scheduler.admissible(db, env, valuing);
    const HalvingState& h = scheduler.lastHalving();
    for (const RoundRow& round : h.unrecorded) { bench.declareRound(round); }
    if (h.active && saidRound != h.n) {
      std::string names;
      for (size_t const i : h.pool) { names += (names.empty() ? "" : ", ") + scheduler.baselines()[i].fft.spec(); }
      log("tune: halving: round %u, %zu of the %u contenders, %" PRIu64 " calls of search each in this round: %s\n",
          h.round + 1, h.pool.size(), h.contenders, h.budget, names.c_str());
      saidRound = h.n;
    } else if (!h.active && saidRound) {
      log("tune: halving done; %s is left, and the search is ranked by what it is expected to gain from here\n",
          h.pool.empty() ? "nothing" : scheduler.baselines()[h.pool.front()].label().c_str());
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
    std::string const label = !baseline ? item->what
      : item->what.empty()              ? baseline->label()
                                        : baseline->label() + " " + item->what;
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
    if (reads && result.completed) {
      const Env& device = scheduler.bootstrap().env();
      outcome = item->kind == ItemKind::Gate ? gateOutcome(db, env, device, fft, *item)
                                             : reachOutcome(db, env, device, fft, *item);
      log("tune: %u. %s %s at %" PRIu64 ": z %.2f over %u rounding errors, check %s, %.1f s -- %s; T %.3f -> %.3f "
          "us/it\n",
          out.items, toString(item->kind), label.c_str(), item->exponent, reading.z, reading.n,
          reading.checkOk ? "OK" : "failed", result.seconds, outcome.c_str(), before, objective.T());
    } else if (result.completed) {
      log("tune: %u. %s %s at %" PRIu64 "%s: %.3f us/it, %.1f s; T %.3f -> %.3f us/it\n", out.items,
          toString(item->kind), label.c_str(), item->exponent, call.c_str(), result.usPerIt, result.seconds, before,
          objective.T());
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

  out.left = scheduler.admissible(db, env, valuing);
  out.valuedT = valuing.T();
  out.floor = stop * valuing.T();
  return out;
}

}  // namespace tune
