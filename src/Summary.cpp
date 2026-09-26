// Copyright (C) Jason Lynch

#include "Summary.h"

#include "Bootstrap.h"
#include "Emit.h"
#include "log.h"
#include "Objective.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <set>

namespace tune {

namespace {

[[nodiscard]] EntryKey keyOf(const Baseline& b) { return {b.fft.spec(), b.kind, b.band.regime.label()}; }

[[nodiscard]] RunSummary::Drift driftOf(const TuneDB& db, u32 sess) {
  RunSummary::Drift out;
  if (const SessRow* const row = db.findSession(sess)) { out.alarmed = row->alarmed; }

  double furthest = 0;
  for (const AnchorRow& row : db.anchors()) {
    if (row.sess != sess || !(row.ratio > 0)) { continue; }
    if (!out.readings) {
      out.anchor = row.fft + '@' + std::to_string(row.exponent);
      out.first = out.lo = out.hi = row.ratio;
    }
    ++out.readings;
    out.last = row.ratio;
    out.lo = std::min(out.lo, row.ratio);
    out.hi = std::max(out.hi, row.ratio);
    if (std::abs(row.ratio - 1) >= furthest) {
      furthest = std::abs(row.ratio - 1);
      out.level = driftLevelOf(row.ratio);
    }
  }
  return out;
}

[[nodiscard]] std::string percent(double fraction) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.3g%%", fraction * 100);
  return buf;
}

}  // namespace

std::string itemLabel(const Scheduler& scheduler, const Item& item) {
  if (item.kind == ItemKind::Bootstrap || item.kind == ItemKind::Reach) { return item.what; }
  std::string const entry = scheduler.baselines()[item.index].label();
  return item.what.empty() ? entry : entry + " " + item.what;
}

std::string familyCounts(const RunSummary::Family& f, const std::string& heldBy) {
  u32 const out = f.entries - f.measured - f.unmeasured - f.waiting;
  std::string counts = std::to_string(f.measured) + " of " + std::to_string(f.entries) + " entries measured";
  if (f.unmeasured) { counts += ", " + std::to_string(f.unmeasured) + " not"; }
  if (f.waiting) { counts += ", " + std::to_string(f.waiting) + " waiting on " + heldBy; }
  if (out) { counts += ", " + std::to_string(out) + " ruled out or given up on"; }
  return counts;
}

RunSummary summarize(const Scheduler& scheduler, const TuneDB& db, u32 env, u32 sess, const QueueReport& report,
                     double stop) {
  RunSummary out;
  out.end = report.end;
  out.T = report.valuedT;
  out.stop = stop;
  out.floor = report.floor;
  out.items = report.items;
  out.anchors = report.anchors;
  out.spent = report.spent;

  BootstrapState const state = scheduler.bootstrapState(db, env);
  Objective const valuing{db, env, scheduler.scope(), state.defaults, Gating::Assumed};
  Objective const published{db, env, scheduler.scope(), state.defaults};
  GainModel const gains = gainsOf(db, env);
  GainDist const unmeasured = gains.global();

  std::set<EntryKey> measured;
  for (const OptionSet& s : optionSetsFor(db, env, state.defaults)) {
    measured.insert({s.entry.fft, s.entry.kind, s.entry.regime.label()});
  }

  std::map<enum FFT_TYPES, RunSummary::Family> families;
  std::map<EntryKey, enum FFT_TYPES> typeOf;
  for (const Baseline& b : scheduler.baselines()) {
    enum FFT_TYPES const type = b.fft.shape.fft_type;
    if (!typeOf.try_emplace(keyOf(b), type).second) { continue; }
    families[type].type = type;
    ++families[type].entries;
  }

  // Least gain first; where no gain could justify either, the one worth more.
  std::map<enum FFT_TYPES, std::pair<double, double>> closest;
  std::set<EntryKey> offered;
  std::map<ItemKind, RunSummary::Remaining> remaining;
  for (const Item& item : report.left) {
    RunSummary::Remaining& r = remaining[item.kind];
    r.kind = item.kind;
    ++r.count;
    if (r.count == 1 || item.value > r.value) {
      r.value = item.value;
      r.best = itemLabel(scheduler, item);
    }

    if (item.kind != ItemKind::Baseline) { continue; }
    const Baseline& b = scheduler.baselines()[item.index];
    if (!offered.insert(keyOf(b)).second) { continue; }
    RunSummary::Family& f = families[b.fft.shape.fft_type];
    ++f.unmeasured;

    std::optional<double> const gain = requiredGain(valuing.points(), b.kind, b.band, item.cost, report.floor);
    std::pair<double, double> const rank{gain.value_or(std::numeric_limits<double>::infinity()), -item.value};
    auto const [at, first] = closest.try_emplace(f.type, rank);
    if (!first && rank >= at->second) { continue; }
    at->second = rank;
    f.closest = itemLabel(scheduler, item);
    f.closestValue = item.value;
    f.gain = gain;
    f.chance = gain ? unmeasured.chanceOfAtLeast(*gain) : 0;
  }

  // Held back by rule, the first measurements are not in what the queue left; they are the ones it would offer.
  std::set<EntryKey> waiting;
  for (const Item& item : scheduler.baselineItems(db, env, valuing)) {
    const Baseline& b = scheduler.baselines()[item.index];
    if (offered.contains(keyOf(b)) || !waiting.insert(keyOf(b)).second) { continue; }
    ++families[b.fft.shape.fft_type].waiting;
  }
  if (!waiting.empty()) {
    bool const bootstrapping =
      std::ranges::any_of(report.left, [](const Item& i) { return i.kind == ItemKind::Bootstrap; });
    out.heldBy = bootstrapping ? "the bootstrap" : "the accuracy gate";
  }

  for (const auto& [key, type] : typeOf) {
    if (measured.contains(key) && !offered.contains(key) && !waiting.contains(key)) { ++families[type].measured; }
  }
  for (auto& [type, f] : families) { out.families.push_back(std::move(f)); }
  for (auto& [kind, r] : remaining) { out.remaining.push_back(std::move(r)); }

  std::map<TestKind, RunSummary::Uncovered> uncovered;
  for (const ObjectivePoint& point : published.points()) {
    if (!point.cost || point.cost->measured() || point.weight <= 0) { continue; }
    RunSummary::Uncovered& u =
      uncovered.try_emplace(point.kind, RunSummary::Uncovered{point.kind, 0, point.exponent, point.exponent})
        .first->second;
    u.weight += point.weight;
    u.lo = std::min(u.lo, point.exponent);
    u.hi = std::max(u.hi, point.exponent);
  }
  for (auto& [kind, u] : uncovered) { out.uncovered.push_back(u); }
  out.unservable = published.unservable();

  out.moves = gains.all().n();
  out.moveGains = gains.global();
  out.combos = gains.combos().n();
  out.comboGains = gains.globalCombo();

  out.drift = driftOf(db, sess);
  return out;
}

std::string spentText(const std::map<ItemKind, QueueReport::Spent>& spent) {
  std::string out;
  for (const auto& [kind, s] : spent) {
    char buf[96];
    snprintf(buf, sizeof(buf), "%s%u %s (%.1f min)", out.empty() ? "" : ", ", s.items, toString(kind), s.seconds / 60);
    out += buf;
  }
  return out;
}

void logSummary(const RunSummary& s) {
  std::string const ran = spentText(s.spent);
  log("tune: summary: %u %s and %u anchor %s%s%s\n", s.items, s.items == 1 ? "item" : "items", s.anchors,
      s.anchors == 1 ? "reading" : "readings", ran.empty() ? "" : ": ", ran.c_str());

  std::string const threshold = s.stop > 0 ? "stop=" + percent(s.stop) + " of T" : "anything";
  switch (s.end) {
  case QueueEnd::Stopped:
    log("tune: summary: stopped before the queue was done; T %.3f us/it, and an item is worth running at %.4f us/it "
        "(stop=%s)\n",
        s.T, s.floor, percent(s.stop).c_str());
    break;
  case QueueEnd::BelowStop:
    log("tune: summary: nothing left is worth %s (%.4f us/it of T %.3f us/it)\n", threshold.c_str(), s.floor, s.T);
    break;
  case QueueEnd::Dry: log("tune: summary: nothing left is worth anything; T %.3f us/it\n", s.T); break;
  }

  std::string const worth = s.stop > 0 ? "to be worth " + threshold : "to save anything";
  for (const RunSummary::Family& f : s.families) {
    std::string const counts = familyCounts(f, s.heldBy);

    if (!f.unmeasured) {
      log("tune: summary: %s: %s\n", typeName(f.type), counts.c_str());
    } else if (!f.gain) {
      log("tune: summary: %s: %s; no gain could make one worth %s\n", typeName(f.type), counts.c_str(),
          threshold.c_str());
    } else {
      log("tune: summary: %s: %s; the closest, %s, needed a gain of %s %s (the gains learnt give that a %s chance), "
          "and was worth %.4f us/it\n",
          typeName(f.type), counts.c_str(), f.closest.c_str(), percent(*f.gain).c_str(), worth.c_str(),
          percent(f.chance).c_str(), f.closestValue);
    }
  }

  for (const RunSummary::Remaining& r : s.remaining) {
    if (r.kind == ItemKind::Bootstrap || r.kind == ItemKind::Gate) {
      log("tune: summary: left: %u %s, which run by rule ahead of anything valued; next: %s\n", r.count,
          toString(r.kind), r.best.c_str());
      continue;
    }
    log("tune: summary: left: %u %s, the best worth %.4f us/it (%s of T): %s\n", r.count, toString(r.kind), r.value,
        percent(s.T > 0 ? r.value / s.T : 0).c_str(), r.best.c_str());
  }

  for (const RunSummary::Uncovered& u : s.uncovered) {
    log("tune: summary: %s of the %s weight, over %" PRIu64 "-%" PRIu64 ", has no entry and is priced by the prior\n",
        percent(u.weight).c_str(), toString(u.kind), u.lo, u.hi);
  }
  if (s.unservable > 0) {
    log("tune: summary: %s of the weight is on exponents no FFT can run\n", percent(s.unservable).c_str());
  }

  auto bins = [](const GainDist& g) {
    std::string out;
    for (size_t i = 0; i < GAIN_BINS; ++i) {
      out += (out.empty() ? "" : " ") + percent(GAIN_AT[i]) + ":" + percent(g.p[i]);
    }
    return out;
  };
  log("tune: summary: %.0f %s observed; one is expected to gain %s (the prior says %s), and 16%% or more %s of the "
      "time; by bin %s\n",
      s.moves, s.moves == 1 ? "move" : "moves", percent(s.moveGains.mean()).c_str(), percent(GAIN_PRIOR.mean()).c_str(),
      percent(s.moveGains.chanceOfAtLeast(0.16)).c_str(), bins(s.moveGains).c_str());
  log("tune: summary: %.0f %s observed; one is expected to gain %s (the prior says %s); by bin %s\n", s.combos,
      s.combos == 1 ? "combination" : "combinations", percent(s.comboGains.mean()).c_str(),
      percent(GAIN_COMBO_PRIOR.mean()).c_str(), bins(s.comboGains).c_str());

  const RunSummary::Drift& d = s.drift;
  if (!d.readings) {
    log("tune: summary: drift: the anchor was not read in this session, so its rows are as measured\n");
  } else {
    log("tune: summary: drift: %u %s of %s, ratio %.4f -> %.4f (%.4f to %.4f), %s%s\n", d.readings,
        d.readings == 1 ? "reading" : "readings", d.anchor.c_str(), d.first, d.last, d.lo, d.hi,
        d.level == DriftLevel::Steady   ? "within the warning"
          : d.level == DriftLevel::Warn ? "past the warning"
                                        : "past the alarm",
        d.alarmed ? "; the session is flagged, the correction doing more than it should be trusted with" : "");
  }
}

}  // namespace tune
