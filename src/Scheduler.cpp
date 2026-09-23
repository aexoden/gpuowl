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

struct Progress {
  bool settled = false;

  // The partial row the entry resumes from: the exponent it was started at, and the calls it has.
  u64 exponent = 0;
  u32 calls = 0;
};

// Where each entry stands, from the rows `env` has: settled by any concluded row -- the baseline is the first
// measurement, and whatever measured it second is not one -- or by a failure, which a repeat would only repeat.  A row
// the device lost under is not recorded, so it says nothing either way.
[[nodiscard]] std::map<KernelKey, Progress> progressOf(const TuneDB& db, u32 env) {
  std::map<KernelKey, Progress> out;

  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env || row.m.status == Status::Lost) { continue; }

    Progress& p = out[{row.fft, row.kind, row.regime.label()}];
    if (row.m.status != Status::Ok || concluded(row.m)) {
      p.settled = true;
    } else if (row.m.calls > p.calls || (row.m.calls == p.calls && row.exponent < p.exponent)) {
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
  case ItemKind::Baseline: return "baseline";
  }
  return "?";
}

Scheduler::Scheduler(RunScope scope, std::vector<Baseline> baselines, u32 blockSize) :
  scope_{std::move(scope)}, baselines_{std::move(baselines)}, clock_{blockSize} {}

std::string Scheduler::keyOf(const Item& item) const {
  if (item.kind == ItemKind::Anchor) { return "anchor"; }
  return baselines_[item.index].label() + "@" + std::to_string(item.exponent);
}

std::vector<Item> Scheduler::admissible(const TuneDB& db, u32 env, const Objective& objective) const {
  std::map<KernelKey, Progress> const progress = progressOf(db, env);
  u32 const defaultsCfg = db.findCfgId({});

  std::vector<Item> out;
  for (size_t i = 0; i < baselines_.size(); ++i) {
    const Baseline& b = baselines_[i];
    std::string const spec = b.fft.spec();

    Progress p{};
    if (auto const at = progress.find({spec, b.kind, b.band.regime.label()}); at != progress.end()) { p = at->second; }
    if (p.settled) { continue; }
    if (auto const at = attempts_.find(i); at != attempts_.end() && at->second >= MAX_ATTEMPTS) { continue; }

    // Resumed where it was started, so that the calls already made pool with the ones still to make.
    u64 const exponent = p.calls && b.band.contains(p.exponent) ? p.exponent : b.exponent;

    // Held back as a measurement would hold it back, which is by what was asked for rather than by what ran.
    if (db.isNogo(env, spec, {})) { continue; }
    if (defaultsCfg && db.diedOn(env, defaultsCfg, b.kind, spec, exponent)) { continue; }

    double const estimate = objective.priorModel().cost(b.fft.shape);
    double value = 0;
    for (const ObjectivePoint& point : objective.points()) {
      if (point.kind != b.kind || !point.cost || !b.band.contains(point.exponent)) { continue; }
      value += point.weight * expectedSaving(point.cost->us, estimate);
    }

    bool const fresh = !built_.contains(spec);
    out.push_back({.kind = ItemKind::Baseline,
                   .index = i,
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

  for (auto it = std::next(ranked.begin()); it != ranked.end() && it->value > 0; ++it) {
    if (it->rate() >= (1 - INTERLEAVE_EPS) * top.rate()) { return *it; }
  }
  return top;
}

void Scheduler::ran(const Item& item, double seconds, double usPerIt) {
  last_ = keyOf(item);
  if (item.kind != ItemKind::Baseline) { return; }

  if (usPerIt > 0) { clock_.observe(seconds, usPerIt, item.fresh); }
  built_.insert(baselines_[item.index].fft.spec());
  ++attempts_[item.index];
}

QueueReport runQueue(Scheduler& scheduler, TuneDB& db, u32 env, Bench& bench,
                     const std::function<void(const Objective&)>& publish) {
  QueueReport out;

  Objective objective{db, env, scheduler.scope()};
  out.startT = objective.T();
  publish(objective);

  while (!bench.stopped()) {
    // Scheduled by the clock rather than by value, and ahead of everything else when it is due: the first reading is
    // what every row of the session is divided by.
    if (bench.anchorDue()) {
      bench.timeAnchor();
      scheduler.ran({.kind = ItemKind::Anchor}, 0, 0);
      ++out.anchors;

      // The first one may have been a race, whose readings are rows like any other.
      objective = Objective{db, env, scheduler.scope()};
      continue;
    }

    std::vector<Item> const ranked = scheduler.admissible(db, env, objective);
    std::optional<Item> const item = scheduler.pick(ranked);
    if (!item) {
      log("tune: nothing left is worth measuring (%zu %s still unmeasured)\n", ranked.size(),
          ranked.size() == 1 ? "entry" : "entries");
      break;
    }

    const Baseline& baseline = scheduler.baselines()[item->index];
    Bench::Result const result = bench.run(baseline.fft, baseline.kind, item->exponent);

    // A call cut short by a stop recorded nothing, so there is nothing to account for.
    if (!result.completed && bench.stopped()) { break; }

    scheduler.ran(*item, result.seconds, result.completed ? result.usPerIt : 0);
    ++out.items;

    double const before = objective.T();
    objective = Objective{db, env, scheduler.scope()};
    publish(objective);

    std::string const resumed = item->calls ? " (resumed at call " + std::to_string(item->calls + 1) + ")" : "";
    if (result.completed) {
      log("tune: %u. %s %s at %" PRIu64 "%s: %.3f us/it, %.1f s; T %.3f -> %.3f us/it\n", out.items,
          toString(item->kind), baseline.label().c_str(), item->exponent, resumed.c_str(), result.usPerIt,
          result.seconds, before, objective.T());
    } else {
      log("tune: %u. %s %s at %" PRIu64 "%s gave no reading\n", out.items, toString(item->kind),
          baseline.label().c_str(), item->exponent, resumed.c_str());
    }
  }

  out.stopped = bench.stopped();
  out.endT = objective.T();
  return out;
}

}  // namespace tune
