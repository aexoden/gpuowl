// Copyright (C) Jason Lynch

#include "Bootstrap.h"

#include "Anchor.h"
#include "Args.h"
#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"
#include "Probe.h"
#include "Scheduler.h"
#include "UseResolve.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <utility>

namespace tune {

const char* typeName(enum FFT_TYPES type) {
  switch (type) {
  case FFT64: return "FFT64";
  case FFT3161: return "FFT3161";
  case FFT3261: return "FFT3261";
  case FFT61: return "FFT61";
  case FFT323161: return "FFT323161";
  case FFT3231: return "FFT3231";
  case FFT6431: return "FFT6431";
  case FFT31: return "FFT31";
  case FFT32: return "FFT32";
  }
  return "?";
}

const char* toString(FamilyPhase phase) {
  switch (phase) {
  case FamilyPhase::Unread: return "unread";
  case FamilyPhase::Held: return "held";
  case FamilyPhase::Skipped: return "skipped";
  case FamilyPhase::Owed: return "owed";
  case FamilyPhase::Served: return "served";
  }
  return "?";
}

UseConfig canonicalConfig(const Env& env, const FFTConfig& fft, const UseConfig& config) {
  UseConfig out;
  for (const auto& [key, value] : config) {
    const Option* const option = findOption(key);
    if (option && option->kind == Kind::Tunable) { out.emplace(key, value); }
  }

  // One key at a time, since dropping one can change whether another applies or what its default is.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto it = out.begin(); it != out.end(); ++it) {
      const Option& option = *findOption(it->first);
      std::optional<int> const value = parseInt<int>(it->second);
      if (!option.appliesTo(env, fft, out) || option.isInert(env, fft, out) ||
          (value && *value == option.defaultFor(env, fft, out))) {
        out.erase(it);
        changed = true;
        break;
      }
    }
  }
  return out;
}

UseConfig builtAs(const Env& env, const FFTConfig& fft, const UseConfig& config) {
  UseConfig out = canonicalConfig(env, fft, config);
  for (const auto& [key, value] : config) {
    const Option* const option = findOption(key);
    if (!option || option->kind != Kind::Tunable) { out.emplace(key, value); }
  }
  return out;
}

std::vector<Move> movesWithin(const Env& env, const FFTConfig& fft, const UseConfig& background, Group group) {
  UseConfig const from = canonicalConfig(env, fft, background);

  std::vector<Move> out;
  std::set<std::string> seen{configText(from)};
  for (const Axis& axis : axesOf(env, fft, from)) {
    if (axis.option->group != group) { continue; }

    for (size_t i = 0; i < axis.values.size(); ++i) {
      if (i == axis.current) { continue; }
      UseConfig moved = from;
      place(moved, axis, i);

      std::string text;
      for (const std::string& key : axis.keys()) { text += (text.empty() ? "" : ",") + key + "=" + moved[key]; }

      // A coupled class moves two keys, so a build failure there is not pinned on either.
      UseConfig config = canonicalConfig(env, fft, moved);
      if (seen.insert(configText(config)).second) {
        out.push_back({.config = std::move(config),
                       .key = axis.coupled ? std::string{} : axis.option->key,
                       .text = std::move(text)});
      }
    }
  }
  return out;
}

Bootstrap::Bootstrap(Env env, u64 probe, std::vector<Family> families, bool enabled, TestKind kind) :
  env_{std::move(env)}, probe_{probe}, families_{std::move(families)}, enabled_{enabled}, kind_{kind} {}

std::vector<Family> Bootstrap::familiesIn(const TuneDB& db, u32 env) const {
  std::vector<Family> out = families_;
  std::set<enum FFT_TYPES> recorded;
  for (const BootRow& row : db.boots()) {
    if (row.probe != probe_ || db.envOf(row.sess) != env) { continue; }
    auto const fft = parseFft(row.fft);
    if (!fft || !recorded.insert(fft->shape.fft_type).second) { continue; }
    auto const at = std::ranges::find(out, fft->shape.fft_type, &Family::type);
    if (at != out.end()) { at->fft = *fft; }
  }
  if (!enabled_ || recorded.size() == out.size()) { return out; }

  // The cheapest reading of the type at the built-in defaults at the probe: what the defaults sweep found fastest
  // there, and so what the search would tune first.
  std::map<enum FFT_TYPES, std::pair<double, FFTConfig>> cheapest;
  for (const RunRow& row : db.mergedRuns()) {
    if (row.kind != kind_ || row.exponent != probe_ || !row.m.ok() || !concluded(row.m) || db.envOf(row.sess) != env) {
      continue;
    }
    auto const fft = parseFft(row.fft);
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (!fft || !opts || !canonicalConfig(env_, *fft, *opts).empty()) { continue; }
    std::vector<u32> const variants = runnableVariants(env_, fft->shape);
    if (std::ranges::find(variants, fft->variant) == variants.end()) { continue; }
    auto const [at, fresh] = cheapest.try_emplace(fft->shape.fft_type, row.m.cost(), *fft);
    if (!fresh && row.m.cost() < at->second.first) { at->second = {row.m.cost(), *fft}; }
  }

  for (Family& f : out) {
    if (recorded.contains(f.type)) { continue; }
    if (auto const at = cheapest.find(f.type); at != cheapest.end()) { f.fft = at->second.second; }
  }
  return out;
}

namespace {

[[nodiscard]] std::set<enum FFT_TYPES> recordedTypes(const TuneDB& db, u32 env, u64 probe) {
  std::set<enum FFT_TYPES> out;
  for (const BootRow& row : db.boots()) {
    if (row.probe != probe || db.envOf(row.sess) != env) { continue; }
    if (auto const fft = parseFft(row.fft)) { out.insert(fft->shape.fft_type); }
  }
  return out;
}

}  // namespace

bool Bootstrap::chosen(const TuneDB& db, u32 env) const {
  std::set<enum FFT_TYPES> const types = recordedTypes(db, env, probe_);
  return std::ranges::all_of(families_, [&](const Family& f) { return types.contains(f.type); });
}

std::vector<Family> Bootstrap::unrecorded(const TuneDB& db, u32 env) const {
  std::set<enum FFT_TYPES> const types = recordedTypes(db, env, probe_);
  std::vector<Family> out = familiesIn(db, env);
  std::erase_if(out, [&](const Family& f) { return types.contains(f.type); });
  return out;
}

BootstrapState Bootstrap::state(const TuneDB& db, u32 env, u64 budget) const {
  BootstrapState out{.families = {}, .kind = kind_, .budget = budget, .complete = true};
  for (const Family& family : familiesIn(db, env)) { out.families.push_back({.family = family}); }
  if (!enabled_) {
    for (FamilyState& f : out.families) { f.phase = FamilyPhase::Skipped; }
    return out;
  }

  // Each family's entry: its configuration, in the kind the bootstrap is in, in the regime band that holds the probe.
  std::map<EntryKey, size_t> entries;
  for (size_t f = 0; f < out.families.size(); ++f) {
    const FFTConfig& fft = out.families[f].family.fft;
    entries.emplace(EntryKey{fft.spec(), kind_, regimeOf(fft, probe_).label()}, f);
  }

  std::vector<bool> failed(out.families.size(), false);
  for (const RunRow& row : db.mergedRuns()) {
    if (row.m.status == Status::Lost || db.envOf(row.sess) != env) { continue; }
    auto const at = entries.find({row.fft, row.kind, row.regime.label()});
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (at == entries.end() || !opts) { continue; }
    FamilyState& s = out.families[at->second];

    // By what the kernels saw rather than by how the set was spelled, so that a reading counts whatever wrote it.
    bool const defaults = canonicalConfig(env_, s.family.fft, *opts).empty();
    if (!defaults) { s.calls += std::max<u32>(row.m.calls, 1); }
    if (row.exponent != probe_) { continue; }
    if (row.m.status != Status::Ok) {
      failed[at->second] = failed[at->second] || defaults;
      continue;
    }
    if (!row.m.ok() || !concluded(row.m)) { continue; }
    if (defaults) { s.reading = s.reading ? std::min(s.reading, row.m.cost()) : row.m.cost(); }
    s.best = s.best ? std::min(s.best, row.m.cost()) : row.m.cost();
  }

  double cheapest = 0;
  for (const FamilyState& s : out.families) {
    if (s.reading && (!cheapest || s.best < cheapest)) { cheapest = s.best; }
  }

  for (size_t f = 0; f < out.families.size(); ++f) {
    FamilyState& s = out.families[f];
    std::string const spec = s.family.fft.spec();
    u32 const cfg = db.findCfgId({});
    if (failed[f] || (cfg && db.diedOn(env, cfg, kind_, spec, probe_))) {
      s.phase = FamilyPhase::Held;
    } else if (!s.reading) {
      s.phase = FamilyPhase::Unread;
    } else if (s.calls >= budget) {
      s.phase = FamilyPhase::Served;
    } else if (s.best * (1 - BOOTSTRAP_GAIN) >= cheapest) {
      // Against the cheapest family as tuned so far, so that a family which only came close to the defaults is not
      // searched once the cheapest has pulled away.
      s.phase = FamilyPhase::Skipped;
    } else {
      s.phase = FamilyPhase::Owed;
    }
    out.complete = out.complete && s.phase != FamilyPhase::Unread && s.phase != FamilyPhase::Owed;
  }
  return out;
}

std::vector<Family> bootstrapFamilies(const Env& env, u64 probe, const std::vector<FFTConfig>& inScope) {
  std::set<enum FFT_TYPES> types;
  for (const FFTConfig& fft : inScope) { types.insert(fft.shape.fft_type); }

  std::vector<Family> out;
  for (const AnchorSpec& candidate : anchorCandidates(probe)) {
    FFTConfig const fft{candidate.fft};
    if (!types.contains(fft.shape.fft_type)) { continue; }
    std::vector<u32> const variants = runnableVariants(env, fft.shape);
    if (std::ranges::find(variants, fft.variant) == variants.end()) { continue; }
    out.push_back({.type = fft.shape.fft_type, .fft = fft});
  }
  return out;
}

Defaults defaultLines(const Env& env, const std::vector<std::pair<Family, UseConfig>>& decided) {
  std::set<std::string> keys;
  for (const auto& [family, config] : decided) {
    for (const auto& [key, value] : config) { keys.insert(key); }
  }

  Defaults out;
  std::map<enum FFT_TYPES, std::vector<std::pair<std::string, std::string>>> lines;

  for (const std::string& key : keys) {
    const Option* const option = findOption(key);
    if (!option) { continue; }

    // What each family the key reaches runs it at: its own winner, or the built-in value it kept.
    std::set<std::string> values;
    for (const auto& [family, config] : decided) {
      if (!option->appliesTo(env, family.fft, config) || option->isInert(env, family.fft, config)) { continue; }
      auto const at = config.find(key);
      values.insert(at != config.end() ? at->second : std::to_string(option->defaultFor(env, family.fft, config)));
    }

    if (values.size() == 1) {
      out.global[key] = *values.begin();
      continue;
    }
    for (const auto& [family, config] : decided) {
      if (auto const at = config.find(key); at != config.end()) { lines[family.type].emplace_back(key, at->second); }
    }
  }

  for (auto& [type, uses] : lines) {
    FFTSelector selector;
    selector.type = type;
    out.family.push_back({.selector = selector, .uses = std::move(uses)});
  }
  return out;
}

UseConfig underDefaults(const Env& env, const FFTConfig& fft, TestKind kind, const Defaults& defaults,
                        const UseConfig& over) {
  SelectionLayers const layers{
    .global = {defaults.global.begin(), defaults.global.end()}, .family = defaults.family, .entry = {}};
  UseConfig const lines = resolveConfig(Args{true}, fft, kind, fittedTo(layers, env, fft, kind));
  if (over.empty()) { return canonicalConfig(env, fft, lines); }

  // What `over` holds of a key the lines leave alone is what a value the lines set has to fit against.
  std::vector<std::pair<std::string, std::string>> kept;
  for (const auto& [key, value] : over) {
    if (!lines.contains(key)) { kept.emplace_back(key, value); }
  }
  SelectionLayers const stacked{.global = {lines.begin(), lines.end()}, .family = {}, .entry = std::move(kept)};
  return canonicalConfig(env, fft, resolveConfig(Args{true}, fft, kind, fittedTo(stacked, env, fft, kind)));
}

Defaults publishedLines(const Env& env, u64 probe, TestKind kind, const std::vector<SelectionEntry>& published) {
  struct Evidence {
    std::tuple<u64, double, std::string> rank;
    Family family;
    UseConfig config;
  };
  std::map<enum FFT_TYPES, Evidence> best;

  for (const SelectionEntry& e : published) {
    if (e.kind != kind) { continue; }
    auto const fft = parseFft(e.fft);
    // Baselines are taken at the built-in defaults, so an entry still published there has not been searched and says
    // nothing about its type's options: taken as evidence it would only return the lines to the defaults, and pull a
    // key every searched type agrees on off the global line.
    if (!fft || canonicalConfig(env, *fft, e.opts).empty()) { continue; }

    u64 const distance = probe < e.emin ? e.emin - probe : probe > e.reach ? probe - e.reach : 0;
    Evidence candidate{.rank = {distance, e.cost, e.id}, .family = {fft->shape.fft_type, *fft}, .config = e.opts};
    auto const [at, fresh] = best.try_emplace(candidate.family.type, candidate);
    if (!fresh && candidate.rank < at->second.rank) { at->second = std::move(candidate); }
  }

  std::vector<std::pair<Family, UseConfig>> sets;
  for (const auto& [type, evidence] : best) {
    const FFTConfig& fft = evidence.family.fft;
    UseConfig config = canonicalConfig(env, fft, evidence.config);
    for (const auto& [key, value] : roundingOf(env, fft, config)) { config.erase(key); }
    sets.emplace_back(evidence.family, canonicalConfig(env, fft, config));
  }
  return defaultLines(env, sets);
}

}  // namespace tune
