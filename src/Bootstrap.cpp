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

namespace {

// What the rows of one family's bootstrap configuration at the probe say about each option set measured there.
struct Readings {
  std::map<std::string, Measurement> ok;
  std::set<std::string> failed;
};

[[nodiscard]] Readings readingsOf(const Env& env, const Family& family, u64 probe, const std::vector<RunRow>& runs,
                                  const TuneDB& db, u32 envId) {
  Readings out;
  std::string const spec = family.fft.spec();

  for (const RunRow& row : runs) {
    if (row.fft != spec || row.kind != TestKind::PRP || row.exponent != probe || row.m.status == Status::Lost ||
        db.envOf(row.sess) != envId) {
      continue;
    }
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (!opts) { continue; }

    // By what the kernels saw rather than by how the set was spelled, so that a candidate finds its readings whatever
    // wrote them.
    std::string const key = configText(canonicalConfig(env, family.fft, *opts));
    if (row.m.status != Status::Ok) {
      out.failed.insert(key);
    } else if (row.m.ok()) {
      auto const [at, fresh] = out.ok.try_emplace(key, row.m);
      if (!fresh && row.m.calls > at->second.calls) { at->second = row.m; }
    }
  }
  return out;
}

[[nodiscard]] double seOf(const RaceEntry& e) { return e.m ? standardError(*e.m) : 0; }

[[nodiscard]] u32 callsOf(const RaceEntry& e) { return e.m ? e.m->calls : 0; }

// Fewest calls first, then the order the candidates were offered in, so that a race is called turn about.  Where one
// candidate is all that needs a call, the others still in the race follow it, so that its calls are taken between
// theirs rather than back to back: two calls in a row of one configuration compare two moments as much as two
// configurations.
[[nodiscard]] std::vector<size_t> byCalls(const std::vector<RaceEntry>& entries, std::vector<size_t> which,
                                          const std::vector<size_t>& live) {
  auto fewest = [&](size_t a, size_t b) { return callsOf(entries[a]) < callsOf(entries[b]); };
  std::ranges::stable_sort(which, fewest);
  if (which.size() == 1) {
    std::vector<size_t> others;
    for (size_t const i : live) {
      if (i != which.front()) { others.push_back(i); }
    }
    std::ranges::stable_sort(others, fewest);
    which.insert(which.end(), others.begin(), others.end());
  }
  return which;
}

}  // namespace

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

const char* toString(RaceHow how) {
  switch (how) {
  case RaceHow::Pending: return "pending";
  case RaceHow::Separated: return "by separation";
  case RaceHow::Margin: return "by margin";
  case RaceHow::Alone: return "unopposed";
  }
  return "?";
}

const char* toString(FamilyPhase phase) {
  switch (phase) {
  case FamilyPhase::Unread: return "unread";
  case FamilyPhase::Held: return "held";
  case FamilyPhase::Skipped: return "skipped";
  case FamilyPhase::Waiting: return "waiting";
  case FamilyPhase::Racing: return "racing";
  case FamilyPhase::Done: return "done";
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

RaceResult decideRace(const std::vector<RaceEntry>& entries) {
  std::vector<size_t> live;
  for (size_t i = 0; i < entries.size(); ++i) {
    if (!entries[i].out) { live.push_back(i); }
  }

  if (live.empty()) { return {.how = RaceHow::Alone, .winner = {}, .next = {}}; }
  if (live.size() == 1) { return {.how = RaceHow::Alone, .winner = live.front(), .next = {}}; }

  std::vector<size_t> unsampled;
  for (size_t const i : live) {
    if (callsOf(entries[i]) < MIN_CALLS) { unsampled.push_back(i); }
  }
  if (!unsampled.empty()) { return {.how = RaceHow::Pending, .winner = {}, .next = byCalls(entries, unsampled, live)}; }

  size_t leader = live.front();
  for (size_t const i : live) {
    if (entries[i].m->cost() < entries[leader].m->cost()) { leader = i; }
  }
  const RaceEntry& lead = entries[leader];
  double const leadCost = lead.m->cost();

  std::vector<size_t> tied{leader};
  std::vector<size_t> unsettled;
  for (size_t const i : live) {
    if (i == leader) { continue; }
    const RaceEntry& rival = entries[i];
    double const cost = rival.m->cost();

    bool const separated = leadCost + RACE_CONFIDENCE * seOf(lead) < cost - RACE_CONFIDENCE * seOf(rival);
    if (separated) { continue; }

    if (cost - leadCost <= RACE_MARGIN * leadCost || rival.m->calls >= RACE_MAX_CALLS) {
      tied.push_back(i);
    } else {
      unsettled.push_back(i);
    }
  }

  if (!unsettled.empty()) {
    if (lead.m->calls < RACE_MAX_CALLS) { unsettled.push_back(leader); }
    return {.how = RaceHow::Pending, .winner = {}, .next = byCalls(entries, unsettled, live)};
  }

  size_t const winner = *std::ranges::min_element(tied, [&](size_t a, size_t b) {
    return std::tuple{entries[a].config.size(), -i64(callsOf(entries[a])), entries[a].m->cost(), a} <
      std::tuple{entries[b].config.size(), -i64(callsOf(entries[b])), entries[b].m->cost(), b};
  });
  return {.how = tied.size() > 1 ? RaceHow::Margin : RaceHow::Separated, .winner = winner, .next = {}};
}

Bootstrap::Bootstrap(Env env, u64 probe, std::vector<Family> families, bool enabled, u32 comboTiers) :
  env_{std::move(env)}, probe_{probe}, families_{std::move(families)}, enabled_{enabled}, comboTiers_{comboTiers} {}

BootstrapState Bootstrap::state(const TuneDB& db, u32 env, const std::set<std::string>& excluded) const {
  BootstrapState out;
  for (const Family& family : families_) { out.families.push_back({.family = family}); }

  if (!enabled_) {
    for (FamilyState& f : out.families) { f.phase = FamilyPhase::Skipped; }
    out.complete = true;
    return out;
  }

  std::vector<RunRow> const runs = db.mergedRuns();
  std::vector<Readings> readings;
  for (const Family& family : families_) { readings.push_back(readingsOf(env_, family, probe_, runs, db, env)); }

  auto entryOf = [&](size_t f, const UseConfig& config, std::string text) {
    std::string const key = configText(config);
    std::string const spec = families_[f].fft.spec();
    RaceEntry e{.config = config, .text = std::move(text), .m = {}, .out = false};
    if (auto const at = readings[f].ok.find(key); at != readings[f].ok.end()) { e.m = at->second; }

    u32 const cfg = db.findCfgId(config);
    e.out = readings[f].failed.contains(key) || excluded.contains(spec + " " + key) || db.isNogo(env, spec, config) ||
      (cfg && db.diedOn(env, cfg, TestKind::PRP, spec, probe_));
    return e;
  };

  // A reading of every family at the defaults comes first: which of them are worth tuning is a comparison between them.
  bool anyUnread = false;
  for (size_t f = 0; f < families_.size(); ++f) {
    FamilyState& s = out.families[f];
    RaceEntry const defaults = entryOf(f, {}, "defaults");
    if (defaults.out) {
      s.phase = FamilyPhase::Held;
    } else if (!defaults.m) {
      anyUnread = true;
      out.turns.push_back({.family = f, .config = {}, .key = {}, .text = "defaults", .calls = 0});
    } else {
      s.reading = defaults.m->cost();
    }
  }
  if (anyUnread) { return out; }

  // Worth tuning while a gain of RACE_GAIN would let it overtake the cheapest family -- at its defaults
  // until that family has been tuned, and then as tuned, so that a family which only came close to the defaults is not
  // raced once the cheapest has pulled away.
  double best = 0;
  for (const FamilyState& s : out.families) {
    if (s.reading > 0 && (!best || s.reading < best)) { best = s.reading; }
  }

  std::vector<size_t> order;
  for (size_t f = 0; f < families_.size(); ++f) {
    if (out.families[f].phase != FamilyPhase::Held) { order.push_back(f); }
  }
  std::ranges::stable_sort(order,
                           [&](size_t a, size_t b) { return out.families[a].reading < out.families[b].reading; });

  bool racing = false;
  for (size_t const f : order) {
    FamilyState& s = out.families[f];
    if (s.reading * (1 - RACE_GAIN) >= best) {
      s.phase = FamilyPhase::Skipped;
      continue;
    }
    s.phase = FamilyPhase::Done;

    // What the decided races read, which the combinations are built from.  Only these: a later search of the same
    // configuration as an entry measures other option sets, which must not reopen what the bootstrap decided.
    std::vector<Reading> answers;
    std::set<std::string> answered;
    auto keep = [&](const std::vector<RaceEntry>& entries) {
      for (const RaceEntry& e : entries) {
        if (e.m && !e.out && answered.insert(configText(e.config)).second) {
          answers.push_back({.config = e.config, .cost = e.m->cost()});
        }
      }
    };

    // Races `entries`, the incumbent first, as `stage`: the winner's place among them, 0 where every candidate is out,
    // and nothing while the race is still being called, having said what to call.
    bool pending = false;
    auto race = [&](std::vector<RaceEntry> entries, const std::vector<std::string>& keys, const std::string& stage,
                    u32 tier, u32 round) -> std::optional<size_t> {
      RaceResult const result = decideRace(entries);
      if (result.how == RaceHow::Pending) {
        pending = true;
        s.phase = racing ? FamilyPhase::Waiting : FamilyPhase::Racing;
        s.stage = stage;
        if (!racing) {
          for (size_t const i : result.next) {
            out.turns.push_back({.family = f,
                                 .config = entries[i].config,
                                 .key = keys[i],
                                 .text = stage + " " + entries[i].text,
                                 .calls = callsOf(entries[i]),
                                 .tier = tier});
          }
        }
        s.entries = std::move(entries);
        s.race = result;
        return {};
      }

      keep(entries);
      if (!result.winner) { return 0; }
      const RaceEntry& won = entries[*result.winner];
      s.decisions.push_back({.stage = stage,
                             .round = round,
                             .how = result.how,
                             .winner = won.text,
                             .cost = won.m ? won.m->cost() : 0,
                             .se = seOf(won),
                             .candidates = u32(entries.size())});
      s.decided = won.config;
      return *result.winner;
    };

    for (Group const group : allGroups()) {
      // Again from each winner that moved: a structural key's dependents are offered only against the background as it
      // stands, so the keys INPLACE=0 opens are raced in the round after it wins.
      for (u32 round = 0; round < GROUP_ROUNDS && !pending; ++round) {
        // The lines hold every key that changes the rounding at its default.  Production applies them to every shape
        // nothing was published for, where no gate ever read them.
        std::vector<Move> moves = movesWithin(env_, families_[f].fft, s.decided, group);
        std::erase_if(moves, [&](const Move& m) { return movesAccuracy(env_, families_[f].fft, m.config); });
        if (moves.empty()) { break; }

        std::vector<RaceEntry> entries{entryOf(f, s.decided, "the incumbent")};
        std::vector<std::string> keys{""};
        for (const Move& move : moves) {
          entries.push_back(entryOf(f, move.config, move.text));
          keys.push_back(move.key);
        }
        if (race(std::move(entries), keys, toString(group), 1, round).value_or(0) == 0) { break; }
      }
      if (pending) { break; }
    }

    // Then each stage of the combination tree once, from the background as it stands: the groups that share kernels
    // combined, then everything.  A stage combines what the tiers below it read, so the answers of a tier join the
    // seeds only once the tier is done, and a stage's candidates do not move while it is being raced.
    std::set<std::string> combined;
    for (u32 tier = 2; tier <= comboTiers_ && !pending; ++tier) {
      std::vector<Reading> fromTier;
      while (!pending) {
        RaceEntry const incumbent = entryOf(f, s.decided, "the incumbent");
        if (!incumbent.m) { break; }

        std::vector<Reading> readings{{.config = s.decided, .cost = incumbent.m->cost()}};
        for (const Reading& r : answers) {
          if (r.config != s.decided) { readings.push_back(r); }
        }
        std::ranges::stable_sort(readings.begin() + 1, readings.end(), {}, &Reading::cost);

        std::string asked;
        for (const Reading& r : readings) { asked += configText(r.config) + "@" + std::to_string(r.cost) + " "; }
        auto& [was, list] = stageLists_[{f, tier}];
        if (was != asked) {
          Strategy const tree{
            .kind = Strategy::Kind::Hybrid, .comboTop = COMBO_TOP, .comboTiers = tier, .bootstrapTree = true};
          list = probesOf(env_, families_[f].fft, s.decided, tree, readings);
          was = std::move(asked);
        }
        auto const next = std::ranges::find_if(
          list.probes, [&](const Probe& p) { return p.tier == tier && !combined.contains(p.stage); });
        if (next == list.probes.end()) { break; }
        std::string const stage = next->stage;
        combined.insert(stage);

        std::vector<RaceEntry> entries{incumbent};
        std::vector<std::string> keys{""};
        for (const Probe& p : list.probes) {
          if (p.stage != stage || movesAccuracy(env_, families_[f].fft, p.config)) { continue; }
          entries.push_back(entryOf(f, p.config, p.text));
          keys.push_back(p.key);
        }
        size_t const before = answers.size();
        (void)race(std::move(entries), keys, stage, tier, 0);
        fromTier.insert(fromTier.end(), answers.begin() + ptrdiff_t(before), answers.end());
        answers.resize(before);
      }
      answers.insert(answers.end(), fromTier.begin(), fromTier.end());
    }
    racing = racing || pending;

    if (!pending) {
      RaceEntry const tuned = entryOf(f, s.decided, "");
      if (tuned.m && tuned.m->cost() < best) { best = tuned.m->cost(); }
    }
  }

  std::vector<std::pair<Family, UseConfig>> decided;
  for (const FamilyState& s : out.families) {
    if (s.phase == FamilyPhase::Done) { decided.emplace_back(s.family, s.decided); }
  }
  out.defaults = defaultLines(env_, decided);
  out.complete = !racing;
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

UseConfig underDefaults(const Env& env, const FFTConfig& fft, TestKind kind, const Defaults& defaults) {
  SelectionLayers const layers{
    .global = {defaults.global.begin(), defaults.global.end()}, .family = defaults.family, .entry = {}};
  return canonicalConfig(env, fft, resolveConfig(Args{true}, fft, kind, fittedTo(layers, env, fft, kind)));
}

Defaults publishedLines(const Env& env, u64 probe, TestKind kind, const std::vector<SelectionEntry>& published,
                        const BootstrapState& bootstrap) {
  struct Evidence {
    std::tuple<u64, double, std::string> rank;
    Family family;
    UseConfig config;
  };
  std::map<enum FFT_TYPES, Evidence> best;

  for (const SelectionEntry& e : published) {
    if (e.kind != kind) { continue; }
    auto const fft = parseFft(e.fft);
    if (!fft) { continue; }

    u64 const distance = probe < e.emin ? e.emin - probe : probe > e.reach ? probe - e.reach : 0;
    Evidence candidate{.rank = {distance, e.cost, e.id}, .family = {fft->shape.fft_type, *fft}, .config = e.opts};
    auto const [at, fresh] = best.try_emplace(candidate.family.type, candidate);
    if (!fresh && candidate.rank < at->second.rank) { at->second = std::move(candidate); }
  }

  if (best.empty()) { return bootstrap.defaults; }

  for (const FamilyState& f : bootstrap.families) {
    if (f.phase == FamilyPhase::Done) {
      best.try_emplace(f.family.type, Evidence{.rank = {}, .family = f.family, .config = f.decided});
    }
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
