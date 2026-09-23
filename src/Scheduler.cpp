// Copyright (C) Jason Lynch

#include "Scheduler.h"

#include "FFTVariants.h"
#include "log.h"
#include "Primes.h"

#include <algorithm>
#include <cinttypes>
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

using KernelKey = std::tuple<std::string, TestKind, std::string>;

// The partial row a baseline resumes from: the exponent it was started at, and the calls it has.
struct Partial {
  u64 exponent = 0;
  u32 calls = 0;
};

struct Progress {
  // Settled by a concluded row emission would publish beside the current lines -- the baseline is the first
  // measurement, and whatever measured it second is not one -- but not by one the lines have since shadowed, whose
  // options are no longer what the configuration runs at.
  std::set<KernelKey> settled;

  // A failure is a verdict on the options it was taken under, which a repeat would only repeat; by the canonical set.
  std::set<std::pair<KernelKey, std::string>> failed;

  // By the canonical option set as well: calls under other options do not pool with the ones a baseline will make.
  std::map<std::pair<KernelKey, std::string>, Partial> partial;
};

// Where each entry stands, from the rows `env` has.  A row the device lost under is not recorded, so it says nothing
// either way.
[[nodiscard]] Progress progressOf(const TuneDB& db, u32 env, const Env& device, const Defaults& defaults) {
  Progress out;

  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || row.m.status == Status::Lost) { continue; }

    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (!opts || !fft) { continue; }

    KernelKey const key{row.fft, row.kind, row.regime.label()};
    if (row.m.status != Status::Ok) {
      out.failed.insert({key, configText(canonicalConfig(device, *fft, *opts))});
      continue;
    }
    if (concluded(row.m)) {
      if (!shadowedBy(defaults, device, *fft, row.kind, *opts)) { out.settled.insert(key); }
      continue;
    }

    Partial& p = out.partial[{key, configText(canonicalConfig(device, *fft, *opts))}];
    if (row.m.calls > p.calls || (row.m.calls == p.calls && row.exponent < p.exponent)) {
      p.exponent = row.exponent;
      p.calls = row.m.calls;
    }
  }

  return out;
}

}  // namespace

double expectedSaving(double best, double estimate) {
  double out = 0;
  for (const GainBin& bin : GAIN_PRIOR) { out += bin.p * std::max(0.0, best - estimate * (1 - bin.gain)); }
  return out;
}

double CallClock::iterSeconds(double usPerIt) const {
  return double(WARMUP_BLOCKS + BLOCKS_PER_CALL) * blockSize_ * usPerIt * 1e-6;
}

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
  }
  return "?";
}

Scheduler::Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize, Bootstrap bootstrap) :
  scope_{std::move(scope)}, baselines_{std::move(baselines)}, clock_{blockSize}, bootstrap_{std::move(bootstrap)} {}

std::string Scheduler::keyOf(const Item& item) const {
  switch (item.kind) {
  case ItemKind::Anchor: return "anchor";
  case ItemKind::Bootstrap: return bootstrap_.families()[item.index].fft.spec() + " " + configText(item.options);
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
              .calls = turn.calls};
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

std::vector<Item> Scheduler::admissible(const TuneDB& db, u32 env, const Objective& objective) const {
  BootstrapState const state = bootstrapState(db, env);
  if (!state.turns.empty()) { return bootstrapItems(state, objective); }

  const Env& device = bootstrap_.env();
  Progress const progress = progressOf(db, env, device, state.defaults);

  std::vector<Item> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    std::string const spec = b.fft.spec();
    KernelKey const key{spec, b.kind, b.band.regime.label()};

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
    double value = 0;
    for (const ObjectivePoint& point : objective.points()) {
      if (point.kind != b.kind || !point.cost || !b.band.contains(point.exponent)) { continue; }
      value += point.weight * expectedSaving(point.cost->us, estimate);
    }

    bool const fresh = !built_.contains(spec + " " + configText(options));
    out.push_back({.kind = ItemKind::Baseline,
                   .index = i,
                   .options = std::move(options),
                   .moved = {},
                   .what = {},
                   .exponent = exponent,
                   .value = value,
                   .seconds = clock_.seconds(estimate / PRIOR_OPTIMISM, fresh),
                   .fresh = fresh,
                   .calls = p.calls});
  }

  std::ranges::stable_sort(out, [](const Item& a, const Item& b) { return a.rate() > b.rate(); });
  return out;
}

std::optional<Item> Scheduler::pick(const std::vector<Item>& ranked) const {
  if (ranked.empty() || ranked.front().value <= 0) { return {}; }

  const Item& top = ranked.front();
  if (keyOf(top) != last_) { return top; }

  if (top.kind == ItemKind::Bootstrap) {
    auto const other = std::ranges::find_if(ranked, [&](const Item& i) { return keyOf(i) != last_; });
    return other != ranked.end() ? *other : top;
  }

  for (auto it = std::next(ranked.begin()); it != ranked.end() && it->value > 0; ++it) {
    if (it->rate() >= (1 - INTERLEAVE_EPS) * top.rate()) { return *it; }
  }
  return top;
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

  built_.insert(baselines_[item.index].fft.spec() + " " + configText(item.options));
  ++attempts_[item.index];
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
          log("tune: bootstrap: %s is not tuned: at %.3f us/it no gain the prior allows would bring it level with the"
              " cheapest family\n",
              name.c_str(), f.reading);
        }
      }

      for (const Decision& d : f.decisions) {
        std::string const key = name + " " + toString(d.group) + " " + std::to_string(d.round);
        if (!said_.insert(key).second) { continue; }
        std::string const round = d.round ? " (round " + std::to_string(d.round + 1) + ")" : "";
        log("tune: bootstrap: %s %s%s decided %s among %u: %s, %.3f +- %.3f us/it\n", name.c_str(), toString(d.group),
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

}  // namespace

QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench, const Publisher& publish) {
  QueueReport out;
  BootstrapLog bootstrapLog;
  bool const bootstrapping = scheduler.bootstrap().enabled();

  BootstrapState state = scheduler.bootstrapState(db, env);
  Objective objective{db, env, scheduler.scope(), state.defaults};
  out.startT = objective.T();
  publish(objective, state.defaults);

  auto rescore = [&] {
    state = scheduler.bootstrapState(db, env);
    objective = Objective{db, env, scheduler.scope(), state.defaults};
    bootstrapLog.report(state, bootstrapping);
  };

  while (!bench.stopped()) {
    // Scheduled by the clock rather than by value, and ahead of everything else when it is due: the first reading is
    // what every row of the session is divided by.
    if (bench.anchorDue()) {
      bench.timeAnchor();
      scheduler.ran({.kind = ItemKind::Anchor}, 0, 0);
      ++out.anchors;

      // The first one may have been a race, whose readings are rows like any other.
      rescore();
      continue;
    }

    std::vector<Item> const ranked = scheduler.admissible(db, env, objective);
    std::optional<Item> const item = scheduler.pick(ranked);
    if (!item) {
      log("tune: nothing left is worth measuring (%zu %s still unmeasured)\n", ranked.size(),
          ranked.size() == 1 ? "entry" : "entries");
      break;
    }

    const Baseline* const baseline = item->kind == ItemKind::Baseline ? &scheduler.baselines()[item->index] : nullptr;
    const FFTConfig& fft = baseline ? baseline->fft : scheduler.bootstrap().families()[item->index].fft;
    TestKind const kind = baseline ? baseline->kind : TestKind::PRP;
    std::string const label = baseline ? baseline->label() : item->what;

    Bench::Result const result = bench.run(fft, kind, item->exponent, item->options, item->moved);

    // A call cut short by a stop recorded nothing, so there is nothing to account for.
    if (!result.completed && bench.stopped()) { break; }

    // A race candidate's calls count for it only if its kernels were built as asked.  Where the host sets a value aside
    // the row lands on another candidate, and without this the race would ask for the same one for ever.
    bool recorded = result.completed;
    if (item->kind == ItemKind::Bootstrap && recorded) {
      UseConfig const built = canonicalConfig(scheduler.bootstrap().env(), fft, result.ran);
      if (built != item->options) {
        recorded = false;
        log("tune: %s was built as %s, so its calls cannot count for it\n", label.c_str(), configText(built).c_str());
      }
    }

    scheduler.ran(*item, result.seconds, result.completed ? result.usPerIt : 0, recorded);
    ++out.items;

    double const before = objective.T();
    rescore();
    publish(objective, state.defaults);

    std::string const call = item->kind == ItemKind::Bootstrap ? " (call " + std::to_string(item->calls + 1) + ")"
      : item->calls ? " (resumed at call " + std::to_string(item->calls + 1) + ")"
                    : "";
    if (result.completed) {
      log("tune: %u. %s %s at %" PRIu64 "%s: %.3f us/it, %.1f s; T %.3f -> %.3f us/it\n", out.items,
          toString(item->kind), label.c_str(), item->exponent, call.c_str(), result.usPerIt, result.seconds, before,
          objective.T());
    } else {
      log("tune: %u. %s %s at %" PRIu64 "%s gave no reading\n", out.items, toString(item->kind), label.c_str(),
          item->exponent, call.c_str());
    }
  }

  out.stopped = bench.stopped();
  out.endT = objective.T();
  return out;
}

}  // namespace tune
