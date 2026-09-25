// Copyright (C) Jason Lynch

#include "Scheduler.h"

#include "Args.h"
#include "FFTVariants.h"
#include "log.h"
#include "Primes.h"

#include <algorithm>
#include <cinttypes>
#include <iterator>
#include <tuple>
#include <utility>

namespace tune {

namespace {

// Enough calls for any entry to conclude twice over.  An entry tried this often without concluding is failing in a way
// that records nothing, and trying it again would only repeat that.
constexpr u32 MAX_ATTEMPTS = 2 * MIN_CALLS;

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

// The partial row a baseline resumes from: the exponent it was started at, and the calls it has.
struct Partial {
  u64 exponent = 0;
  u32 calls = 0;
};

}  // namespace

struct Progress {
  // Settled by a concluded row emission would publish beside the current lines -- the baseline is the first
  // measurement, and whatever measured it second is not one -- but not by one the lines have since shadowed, whose
  // options are no longer what the configuration runs at.
  std::set<EntryKey> settled;

  // A failure is a verdict on the options it was taken under, which a repeat would only repeat; by the canonical set.
  std::set<std::pair<EntryKey, std::string>> failed;

  // By the canonical option set as well: calls under other options do not pool with the ones a baseline will make.
  std::map<std::pair<EntryKey, std::string>, Partial> partial;

  // Every option set a row has concluded, canonical, shadowed or not: what a probe asks may already be answered by a
  // row the lines have since withdrawn from publication.
  std::map<EntryKey, std::vector<UseConfig>> concluded;

  // The option sets a row has concluded or failed, and how many sets each entry has rows of at all.
  std::set<EntrySet> answered;
  std::map<EntryKey, std::set<std::string>> sets;
};

namespace {

// Where each entry stands, from the rows `env` has.  A row the device lost under is not recorded, so it says nothing
// either way.
[[nodiscard]] Progress progressOf(const TuneDB& db, u32 env, const Env& device, const Defaults& defaults) {
  Progress out;

  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || row.m.status == Status::Lost) { continue; }

    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (!opts || !fft) { continue; }

    EntryKey const key{row.fft, row.kind, row.regime.label()};
    UseConfig const canonical = canonicalConfig(device, *fft, *opts);
    std::string const text = configText(canonical);
    out.sets[key].insert(text);
    if (row.m.status != Status::Ok) {
      out.failed.insert({key, text});
      out.answered.insert({key, text});
      continue;
    }
    if (concluded(row.m)) {
      if (!shadowedBy(defaults, device, *fft, row.kind, *opts)) { out.settled.insert(key); }
      out.concluded[key].push_back(canonical);
      out.answered.insert({key, text});
      continue;
    }

    Partial& p = out.partial[{key, configText(canonical)}];
    if (row.m.calls > p.calls || (row.m.calls == p.calls && row.exponent < p.exponent)) {
      p.exponent = row.exponent;
      p.calls = row.m.calls;
    }
  }

  return out;
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

UseConfig besideLines(const Env& env, const FFTConfig& fft, TestKind kind, const Defaults& defaults, UseConfig config) {
  if (defaults.global.empty() && defaults.family.empty()) { return config; }
  UseConfig const canonical = config;

  // Fitted against the set as it stands, as emission fits them: whether a line's value is one the table offers can
  // turn on the set's own keys -- a probe back to MULTI_Q=0 makes an L2_STRIPING line legal again -- and each key named
  // here can do the same to another.
  for (bool changed = true; changed;) {
    changed = false;
    SelectionLayers const layers = fittedTo({.global = {defaults.global.begin(), defaults.global.end()},
                                             .family = defaults.family,
                                             .entry = {config.begin(), config.end()}},
                                            env, fft, kind);

    for (const auto& [key, value] : resolveConfig(Args{true}, fft, kind, layers)) {
      const Option* const option = findOption(key);
      if (config.contains(key) || !option || option->kind != Kind::Tunable || !option->appliesTo(env, fft, canonical) ||
          option->isInert(env, fft, canonical)) {
        continue;
      }
      int const own = option->defaultFor(env, fft, canonical);
      if (parseInt<int>(value) != own) {
        config[key] = std::to_string(own);
        changed = true;
      }
    }
  }
  return config;
}

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

std::string Baseline::label() const { return fft.spec() + " " + toString(kind) + " " + band.regime.label(); }

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
  case ItemKind::Bootstrap: return "bootstrap";
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
                     std::optional<Strategy> strategy, bool restarts, bool gate) :
  scope_{std::move(scope)},
  baselines_{std::move(baselines)},
  clock_{blockSize},
  bootstrap_{std::move(bootstrap)},
  strategy_{std::move(strategy)},
  restarts_{restarts},
  gate_{gate} {}

std::string Scheduler::builtKey(const FFTConfig& fft, const UseConfig& options) const {
  return fft.spec() + " " + configText(canonicalConfig(bootstrap_.env(), fft, options));
}

Scheduler::ListMemo& Scheduler::probeList(const Baseline& entry, const UseConfig& best,
                                          std::span<const Reading> readings, bool structuralSteps,
                                          std::string from) const {
  const Env& env = bootstrap_.env();
  const FFTConfig& fft = entry.fft;
  UseConfig const canonical = canonicalConfig(env, fft, best);
  auto const [at, fresh] =
    probeLists_.try_emplace(entry.label() + " " + configText(canonical) + (structuralSteps ? "" : " within"));
  ListMemo& memo = at->second;
  if (!fresh && memo.from == from) { return memo; }

  // Against the same background, so over the same axes.
  auto identity = [](const Probe& p) {
    std::string out = configText(p.config);
    for (auto const& [axis, position] : p.moves) { out += " " + std::to_string(axis) + ":" + std::to_string(position); }
    return out;
  };
  std::set<std::string> answered;
  for (size_t p = 0; p < memo.list.probes.size(); ++p) {
    if (memo.answered[p]) { answered.insert(identity(memo.list.probes[p])); }
  }

  memo.from = std::move(from);
  memo.list = probesOf(env, fft, canonical, *strategy_, readings, structuralSteps);
  memo.answered.assign(memo.list.probes.size(), false);
  for (size_t p = 0; p < memo.list.probes.size(); ++p) {
    memo.answered[p] = answered.contains(identity(memo.list.probes[p]));
  }
  memo.checked.clear();
  return memo;
}

const UseConfig& Scheduler::draw(size_t index, u32 k) const {
  auto const [at, fresh] = draws_.try_emplace({index, k});
  if (fresh) {
    const Baseline& b = baselines_[index];
    at->second = canonicalConfig(bootstrap_.env(), b.fft, restartOf(bootstrap_.env(), b.fft, b.label(), k));
  }
  return at->second;
}

std::optional<Item> Scheduler::nextRestart(const TuneDB& db, u32 env, const Progress& progress,
                                           const Defaults& defaults, size_t index) const {
  const Env& device = bootstrap_.env();
  const Baseline& b = baselines_[index];
  std::string const spec = b.fft.spec();
  std::string const regime = b.band.regime.label();
  EntryKey const key{spec, b.kind, regime};

  // From the last draw declared, which a stop may have left partly measured, so that it is resumed rather than skipped.
  RestartScan& scan = scans_[index];
  for (const JumpRow& row : db.jumps()) {
    if (db.envOf(row.sess) == env && row.fft == spec && row.kind == b.kind && row.regime.label() == regime) {
      scan.next = std::max(scan.next, row.k);
    }
  }
  if (auto const sets = progress.sets.find(key); sets != progress.sets.end()) {
    scan.seen.insert(sets->second.begin(), sets->second.end());
  }

  auto itemOf = [&](u32 k) {
    const UseConfig& drawn = draw(index, k);
    Item item{.kind = ItemKind::Restart,
              .index = index,
              .options = besideLines(device, b.fft, b.kind, defaults, drawn),
              .moved = {},
              .what = "#" + std::to_string(k + 1) + " " + (drawn.empty() ? "the built-in defaults" : configText(drawn)),
              .exponent = b.exponent,
              .value = 0,
              .seconds = 0,
              .fresh = true,
              .calls = 0,
              .draw = k};
    if (auto const p = progress.partial.find({key, configText(drawn)});
        p != progress.partial.end() && b.band.contains(p->second.exponent)) {
      item.exponent = p->second.exponent;
      item.calls = p->second.calls;
    }
    return item;
  };

  auto runnable = [&](u32 k) {
    if (progress.answered.contains({key, configText(draw(index, k))})) { return false; }
    Item const item = itemOf(k);
    u32 const cfg = db.findCfgId(item.options);
    auto const attempts = probeAttempts_.find(keyOf(item));
    return !shadowedBy(defaults, device, b.fft, b.kind, item.options) &&
      (attempts == probeAttempts_.end() || attempts->second < MAX_ATTEMPTS) && !db.isNogo(env, spec, item.options) &&
      !(cfg && db.diedOn(env, cfg, b.kind, spec, item.exponent));
  };

  std::optional<u32> const k = nextRunnable(scan, [&](u32 k) { return configText(draw(index, k)); }, runnable);
  if (!k) { return {}; }
  return itemOf(*k);
}

std::optional<u32> nextRunnable(RestartScan& scan, const std::function<std::string(u32)>& text,
                                const std::function<bool(u32)>& runnable) {
  while (!scan.exhausted) {
    if (runnable(scan.next)) { return scan.next; }
    scan.repeats = scan.seen.insert(text(scan.next)).second ? 0 : scan.repeats + 1;
    scan.exhausted = scan.repeats >= RESTART_REPEATS;
    ++scan.next;
  }
  return {};
}

std::string Scheduler::keyOf(const Item& item) const {
  switch (item.kind) {
  case ItemKind::Anchor: return "anchor";
  case ItemKind::Bootstrap: return bootstrap_.families()[item.index].fft.spec() + " " + configText(item.options);
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

BootstrapState Scheduler::bootstrapState(const TuneDB& db, u32 env) const {
  std::set<std::string> excluded;
  for (const auto& [key, n] : unrecorded_) {
    if (n >= MAX_ATTEMPTS) { excluded.insert(key); }
  }
  return bootstrap_.state(db, env, excluded);
}

std::vector<Item> Scheduler::bootstrapItems(const BootstrapState& state, const Objective& objective) const {
  std::vector<Item> out;
  for (const Turn& turn : state.turns) {
    const Family& family = bootstrap_.families()[turn.family];
    double const estimate = objective.priorModel().cost(family.fft.shape);
    Item item{.kind = ItemKind::Bootstrap,
              .index = turn.family,
              .options = turn.config,
              .moved = turn.key,
              .what = std::string{typeName(family.type)} + " " + family.fft.spec() + " " + turn.text,
              .exponent = bootstrap_.probe(),
              .value = 1,
              .seconds = 0,
              .fresh = true,
              .calls = turn.calls,
              .draw = 0,
              .tier = turn.tier};
    item.fresh = !built_.contains(keyOf(item));
    item.seconds = clock_.seconds(estimate / PRIOR_OPTIMISM, item.fresh);
    out.push_back(std::move(item));
  }

  // Before any family has been read, the cheapest first: which families are worth tuning waits on all of them.
  bool const reading = std::ranges::none_of(state.families, [](const FamilyState& f) {
    return f.phase == FamilyPhase::Racing || f.phase == FamilyPhase::Done;
  });
  if (reading) {
    std::ranges::stable_sort(out, [](const Item& a, const Item& b) { return a.seconds < b.seconds; });
  }
  return out;
}

std::map<EntryKey, size_t> Scheduler::entryIndex() const {
  std::map<EntryKey, size_t> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    out.try_emplace({baselines_[i].fft.spec(), baselines_[i].kind, baselines_[i].band.regime.label()}, i);
  }
  return out;
}

std::vector<Item> Scheduler::gateItems(const TuneDB& db, u32 env, const Defaults& defaults) const {
  if (!gate_) { return {}; }
  const Env& device = bootstrap_.env();
  std::map<EntryKey, size_t> const indexOf = entryIndex();

  std::vector<Item> out;
  std::set<std::string> offered;
  for (const OptionSet& s : gatesOwed(db, env, defaults)) {
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
  BootstrapState const state = bootstrapState(db, env);
  Progress const progress = progressOf(db, env, bootstrap_.env(), state.defaults);
  return baselineItems(db, env, state, progress, gainsOf(db, env), objective);
}

std::vector<Item> Scheduler::baselineItems(const TuneDB& db, u32 env, const BootstrapState& state,
                                           const Progress& progress, const GainModel& gains,
                                           const Objective& objective) const {
  const Env& device = bootstrap_.env();
  GainDist const unmeasured = gains.global();

  std::vector<Item> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    std::string const spec = b.fft.spec();
    EntryKey const key{spec, b.kind, b.band.regime.label()};

    if (progress.settled.contains(key)) { continue; }
    if (auto const at = attempts_.find(i); at != attempts_.end() && at->second >= MAX_ATTEMPTS) { continue; }

    UseConfig options = underDefaults(device, b.fft, b.kind, state.defaults);
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

std::vector<Item> Scheduler::admissible(const TuneDB& db, u32 env, const Objective& objective) const {
  BootstrapState const state = bootstrapState(db, env);
  if (!state.turns.empty()) { return bootstrapItems(state, objective); }
  if (std::vector<Item> gates = gateItems(db, env, state.defaults); !gates.empty()) { return gates; }

  const Env& device = bootstrap_.env();
  Progress const progress = progressOf(db, env, device, state.defaults);
  GainModel const gains = gainsOf(db, env);

  std::vector<Item> out = baselineItems(db, env, state, progress, gains, objective);

  // The option sets of each entry, which also say how much each option set costs where it was measured.
  std::vector<OptionSet> const sets =
    strategy_ || gate_ ? optionSetsFor(db, env, state.defaults) : std::vector<OptionSet>{};
  std::ranges::move(reachItems(db, env, sets, objective), std::back_inserter(out));

  if (strategy_) {
    std::map<EntryKey, std::vector<Reading>> const readings = readingsOf(sets, device);

    for (size_t i = 0; i < baselines_.size(); ++i) {
      const Baseline& b = baselines_[i];
      EntryKey const key{b.fft.spec(), b.kind, b.band.regime.label()};
      auto const at = readings.find(key);
      if (at == readings.end()) { continue; }
      const std::vector<Reading>& ofEntry = at->second;

      // Each structural branch is searched from its own best set, and only the entry's best set steps into others.
      std::vector<Branch> branches;
      if (strategy_->branches()) {
        branches = branchesOf(device, b.fft, ofEntry);
      } else {
        branches.push_back({.structure = {}, .best = ofEntry.front().config, .cost = ofEntry.front().cost});
      }

      std::vector<UseConfig> const none;
      auto const rows = progress.concluded.find(key);
      const std::vector<UseConfig>& concluded = rows != progress.concluded.end() ? rows->second : none;
      std::vector<std::string> texts;
      for (const UseConfig& row : concluded) { texts.push_back(configText(row)); }

      // What each branch's readings say, which is all its combo tiers read.
      std::vector<std::string> from(branches.size());
      if (strategy_->combines()) {
        for (const Reading& r : ofEntry) {
          UseConfig const structure = branchOf(device, b.fft, r.config);
          auto const in = std::ranges::find_if(branches, [&](const Branch& br) { return br.structure == structure; });
          if (in != branches.end()) {
            from[size_t(in - branches.begin())] += configText(r.config) + " " + std::to_string(r.cost) + ";";
          }
        }
      }

      GainDist const entryGains = gains.forEntry(key);
      GainDist const comboGains = gains.comboForEntry(key);
      size_t const before = out.size();
      std::set<std::string> offered;
      for (size_t branch = 0; branch < branches.size(); ++branch) {
        // Every probe of a branch is worth the same: what the gains this entry and the device have shown expect a move
        // from the branch's best set to save.  Every combo likewise, by the gains combinations have shown.
        double const value = expectedSaving(objective.points(), b.kind, b.band, branches[branch].cost, entryGains);
        double const comboValue = expectedSaving(objective.points(), b.kind, b.band, branches[branch].cost, comboGains);
        if (value <= 0 && comboValue <= 0) { continue; }

        ListMemo& memo = probeList(b, branches[branch].best, ofEntry, branch == 0, std::move(from[branch]));
        const ProbeList& list = memo.list;
        for (size_t r = 0; r < concluded.size(); ++r) {
          if (!memo.checked.insert(texts[r]).second) { continue; }
          for (size_t p = 0; p < list.probes.size(); ++p) {
            if (!memo.answered[p]) { memo.answered[p] = answeredBy(device, b.fft, list, list.probes[p], concluded[r]); }
          }
        }

        // The list is in tier order, and a combination waits for the tiers below it to be answered.
        std::optional<u32> lowest;
        for (size_t p = 0; p < list.probes.size(); ++p) {
          if (memo.answered[p]) { continue; }
          const Probe& probe = list.probes[p];
          if (lowest && probe.tier > *lowest) { break; }
          double const worth = probe.tier > 1 ? comboValue : value;
          if (worth <= 0) { continue; }
          std::string const text = configText(probe.config);
          if (progress.failed.contains({key, text})) { continue; }

          UseConfig options = besideLines(device, b.fft, b.kind, state.defaults, probe.config);
          if (shadowedBy(state.defaults, device, b.fft, b.kind, options)) { continue; }

          Item item{.kind = probe.tier > 1 ? ItemKind::Combo : ItemKind::Probe,
                    .index = i,
                    .options = std::move(options),
                    .moved = probe.key,
                    .what = probe.stage + " " + probe.text,
                    .exponent = b.exponent,
                    .value = worth,
                    .cost = branches[branch].cost,
                    .seconds = 0,
                    .fresh = true,
                    .calls = 0,
                    .draw = 0,
                    .tier = probe.tier};
          if (auto const n = probeAttempts_.find(keyOf(item)); n != probeAttempts_.end() && n->second >= MAX_ATTEMPTS) {
            continue;
          }
          if (auto const p = progress.partial.find({key, text});
              p != progress.partial.end() && b.band.contains(p->second.exponent)) {
            item.exponent = p->second.exponent;
            item.calls = p->second.calls;
          }

          if (db.isNogo(env, b.fft.spec(), item.options)) { continue; }
          if (u32 const cfg = db.findCfgId(item.options);
              cfg && db.diedOn(env, cfg, b.kind, b.fft.spec(), item.exponent)) {
            continue;
          }
          if (!offered.insert(text).second) { continue; }
          lowest = std::min(lowest.value_or(probe.tier), probe.tier);

          item.fresh = !built_.contains(builtKey(b.fft, item.options));
          item.seconds = clock_.seconds(branches[branch].cost, item.fresh);
          out.push_back(std::move(item));
        }
      }

      // At a local optimum of the declared moves, and only there, a jump: worth what a move from the entry's best set
      // is, by what the entry's earlier jumps found, and offered only once no probe is left rather than left to a score
      // to rank below them.
      double const value =
        expectedSaving(objective.points(), b.kind, b.band, ofEntry.front().cost, gains.restartForEntry(key));
      std::optional<Item> next = restarts_ && value > 0 && out.size() == before
        ? nextRestart(db, env, progress, state.defaults, i)
        : std::nullopt;
      if (next) {
        Item item = std::move(*next);
        item.value = value;
        item.cost = ofEntry.front().cost;
        item.fresh = !built_.contains(builtKey(b.fft, item.options));
        item.seconds = clock_.seconds(ofEntry.front().cost, item.fresh);
        out.push_back(std::move(item));
      }
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
  return out;
}

bool worthRunning(const Item& item, double floor) {
  if (item.kind == ItemKind::Bootstrap || item.kind == ItemKind::Gate) { return true; }
  return item.value > 0 && item.value >= floor;
}

std::optional<Item> Scheduler::pick(const std::vector<Item>& ranked, double floor) const {
  auto const worth = [floor](const Item& i) { return worthRunning(i, floor); };
  auto const top = std::ranges::find_if(ranked, worth);
  if (top == ranked.end()) { return {}; }
  if (keyOf(*top) != last_) { return *top; }

  if (top->kind == ItemKind::Bootstrap) {
    auto const other = std::ranges::find_if(ranked, [&](const Item& i) { return keyOf(i) != last_; });
    return other != ranked.end() ? *other : *top;
  }

  for (auto it = std::next(top); it != ranked.end(); ++it) {
    if (worth(*it) && it->rate() >= (1 - INTERLEAVE_EPS) * top->rate()) { return *it; }
  }
  return *top;
}

void Scheduler::ran(const Item& item, double seconds, double usPerIt, bool recorded) {
  last_ = keyOf(item);
  if (item.kind == ItemKind::Anchor) { return; }

  if (usPerIt > 0) { clock_.observe(seconds, usPerIt, item.fresh); }

  if (item.kind == ItemKind::Bootstrap) {
    built_.insert(last_);
    if (!recorded || usPerIt <= 0) { ++unrecorded_[last_]; }
    return;
  }

  if (item.kind == ItemKind::Gate || item.kind == ItemKind::Reach) {
    ++gateAttempts_[last_];
    return;
  }

  built_.insert(builtKey(baselines_[item.index].fft, item.options));
  switch (item.kind) {
  case ItemKind::Probe:
  case ItemKind::Combo:
  case ItemKind::Restart: ++probeAttempts_[last_]; return;
  case ItemKind::Refine:
    if (!recorded || usPerIt <= 0) { ++unrecordedRefines_[last_]; }
    return;
  case ItemKind::Baseline: ++attempts_[item.index]; return;
  case ItemKind::Anchor:
  case ItemKind::Bootstrap:
  case ItemKind::Gate:
  case ItemKind::Reach: return;
  }
}

namespace {

// Says once what the bootstrap has come to: each family it will not tune and why, each race as it is decided, and the
// lines once every family is settled.
class BootstrapLog {
public:
  void report(const BootstrapState& state, bool enabled) {
    for (const FamilyState& f : state.families) {
      std::string const name = std::string{typeName(f.family.type)} + " " + f.family.fft.spec();

      if ((f.phase == FamilyPhase::Skipped || f.phase == FamilyPhase::Held) && said_.insert(name).second) {
        if (!enabled) {
          log("tune: bootstrap: %s is not tuned, bootstrapping being turned off\n", name.c_str());
        } else if (f.phase == FamilyPhase::Held) {
          log("tune: bootstrap: %s cannot be measured at the built-in defaults, so it is not tuned\n", name.c_str());
        } else {
          log("tune: bootstrap: %s is not tuned: at %.3f us/it it would take more than a %.0f%% gain to bring it level"
              " with the cheapest family\n",
              name.c_str(), f.reading, 100 * RACE_GAIN);
        }
      }

      for (const Decision& d : f.decisions) {
        std::string const key = name + " " + d.stage + " " + std::to_string(d.round);
        if (!said_.insert(key).second) { continue; }
        std::string const round = d.round ? " (round " + std::to_string(d.round + 1) + ")" : "";
        log("tune: bootstrap: %s %s%s decided %s among %u: %s, %.3f +- %.3f us/it\n", name.c_str(), d.stage.c_str(),
            round.c_str(), toString(d.how), d.candidates, d.winner.c_str(), d.cost, d.se);
      }
    }

    if (enabled && state.complete && !complete_) {
      complete_ = true;
      std::string lines = configText(state.defaults.global);
      for (const UseLine& line : state.defaults.family) {
        UseConfig const uses{line.uses.begin(), line.uses.end()};
        lines += "; ! " + line.selector.spec() + " " + configText(uses);
      }
      log("tune: bootstrap complete; the defaults are %s\n", lines.c_str());
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
[[nodiscard]] std::optional<std::string> bestOf(const TuneDB& db, u32 envId, const Env& env, const Defaults& defaults,
                                                const Baseline& b) {
  std::vector<SelectionEntry> const candidates = candidatesFor(db, envId, defaults, Gating::Assumed);
  std::map<EntryKey, const SelectionEntry*> const best = bestEntries(candidates);
  auto const at = best.find({b.fft.spec(), b.kind, b.band.regime.label()});
  if (at == best.end()) { return {}; }

  char cost[32];
  snprintf(cost, sizeof(cost), "%.3f us/it", at->second->cost);
  std::string const opts = configText(canonicalConfig(env, b.fft, at->second->opts));
  return (opts.empty() ? std::string{"the built-in defaults"} : opts) + ", " + cost;
}

}  // namespace

QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench, const Publisher& publish, double stop) {
  QueueReport out;
  BootstrapLog bootstrapLog;
  bool const bootstrapping = scheduler.bootstrap().enabled();

  // What is published, and what the items are valued against: the same but for the sets the gate still owes, whose
  // readings are taken before anything is valued.
  BootstrapState state = scheduler.bootstrapState(db, env);
  Objective objective{db, env, scheduler.scope(), state.defaults};
  Objective valuing{db, env, scheduler.scope(), state.defaults, Gating::Assumed};
  out.startT = objective.T();
  publish(objective, state.defaults);

  auto rescore = [&] {
    state = scheduler.bootstrapState(db, env);
    objective = Objective{db, env, scheduler.scope(), state.defaults};
    valuing = Objective{db, env, scheduler.scope(), state.defaults, Gating::Assumed};
    bootstrapLog.report(state, bootstrapping);
  };

  while (!bench.stopped()) {
    std::vector<Item> const ranked = scheduler.admissible(db, env, valuing);
    std::optional<Item> const item = scheduler.pick(ranked, stop * valuing.T());
    if (!item) {
      out.end =
        std::ranges::any_of(ranked, [](const Item& i) { return i.value > 0; }) ? QueueEnd::BelowStop : QueueEnd::Dry;
      break;
    }

    // Scheduled by the clock rather than by value, and ahead of everything else when it is due, but only while there
    // is something to divide by it: the first reading is what every row of the session is divided by.
    if (bench.anchorDue()) {
      bench.timeAnchor();
      scheduler.ran({.kind = ItemKind::Anchor}, 0, 0);
      ++out.anchors;

      // The first one may have been a race, whose readings are rows like any other.
      rescore();
      continue;
    }

    const Baseline* const baseline = item->kind != ItemKind::Bootstrap && item->kind != ItemKind::Reach
      ? &scheduler.baselines()[item->index]
      : nullptr;
    const FFTConfig& fft = item->fft ? *item->fft
      : baseline                     ? baseline->fft
                                     : scheduler.bootstrap().families()[item->index].fft;
    TestKind const kind = baseline ? baseline->kind : TestKind::PRP;
    bool const reads = item->kind == ItemKind::Gate || item->kind == ItemKind::Reach;
    bool const probing = baseline && item->kind != ItemKind::Baseline && !reads;
    std::string const label = !baseline ? item->what
      : probing || reads                ? baseline->label() + " " + item->what
                                        : baseline->label();
    std::optional<std::string> const bestBefore =
      probing ? bestOf(db, env, scheduler.bootstrap().env(), state.defaults, *baseline) : std::nullopt;

    // Once, before its first call; a resumed one was declared by the process that started it.
    if (item->kind == ItemKind::Restart && item->calls == 0) {
      bench.declareRestart(fft, kind, item->exponent, item->options, item->draw);
    }
    if (item->tier > 1 && item->calls == 0 &&
        !declaredCombo(db, env, scheduler.bootstrap().env(), fft, kind, item->exponent, item->options)) {
      bench.declareCombo(fft, kind, item->exponent, item->options, item->tier);
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
    publish(objective, state.defaults);

    // A refine is the next call of a row that is already concluded, not a resumption of one that was interrupted.
    bool const counted = item->kind == ItemKind::Bootstrap || item->kind == ItemKind::Refine;
    std::string const call = counted ? " (call " + std::to_string(item->calls + 1) + ")"
      : item->calls                  ? " (resumed at call " + std::to_string(item->calls + 1) + ")"
                                     : "";
    if (reads && result.completed) {
      const Env& device = scheduler.bootstrap().env();
      std::string const outcome = item->kind == ItemKind::Gate ? gateOutcome(db, env, device, fft, *item)
                                                               : reachOutcome(db, env, device, fft, *item);
      log("tune: %u. %s %s at %" PRIu64 ": z %.2f over %u rounding errors, check %s, %.1f s -- %s; T %.3f -> %.3f "
          "us/it\n",
          out.items, toString(item->kind), label.c_str(), item->exponent, reading.z, reading.n,
          reading.checkOk ? "OK" : "failed", result.seconds, outcome.c_str(), before, objective.T());
    } else if (result.completed) {
      log("tune: %u. %s %s at %" PRIu64 "%s: %.3f us/it, %.1f s; T %.3f -> %.3f us/it\n", out.items,
          toString(item->kind), label.c_str(), item->exponent, call.c_str(), result.usPerIt, result.seconds, before,
          objective.T());
    } else {
      log("tune: %u. %s %s at %" PRIu64 "%s gave no reading\n", out.items, toString(item->kind), label.c_str(),
          item->exponent, call.c_str());
    }

    if (probing) {
      std::optional<std::string> const bestAfter =
        bestOf(db, env, scheduler.bootstrap().env(), state.defaults, *baseline);
      if (bestAfter && bestAfter != bestBefore) {
        log("tune: %s is now best at %s\n", baseline->label().c_str(), bestAfter->c_str());
      }
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
