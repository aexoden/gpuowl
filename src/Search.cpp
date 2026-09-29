// Copyright (C) Jason Lynch

#include "Search.h"

#include "Bootstrap.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace tune {

namespace {

// How often one re-score widens the listing of one branch whose partly listed stage offers nothing.  More than one, so
// that a window used up is replaced at once; bounded, so that a long stretch of points that cannot run is read over
// several re-scores rather than in one.
constexpr u32 MAX_WIDENINGS = 4;

}  // namespace

std::string Baseline::label() const { return fft.spec() + " " + toString(kind) + " " + band.regime.label(); }

EntryKey Baseline::key() const { return {fft.spec(), kind, band.regime.label()}; }

Progress progressOf(const TuneDB& db, u32 env, const Env& device) {
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
      out.settled.insert(key);
      out.concluded[key].push_back(canonical);
      out.answered.insert({key, text});
      continue;
    }

    Partial& p = out.partial[{key, text}];
    if (row.m.calls > p.calls || (row.m.calls == p.calls && row.exponent < p.exponent)) {
      p.exponent = row.exponent;
      p.calls = row.m.calls;
    }
  }

  return out;
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

EntrySearch::EntrySearch(Baseline entry) : entry_{std::move(entry)}, key_{entry_.key()} {}

void EntrySearch::tried(const Env& device, const UseConfig& options) {
  ++attempts_[configText(canonicalConfig(device, entry_.fft, options))];
}

bool EntrySearch::runnable(const SearchContext& context, const UseConfig& options, u64 exponent) const {
  if (!attempts_.empty()) {
    auto const n = attempts_.find(configText(canonicalConfig(context.device, entry_.fft, options)));
    if (n != attempts_.end() && n->second >= MAX_ATTEMPTS) { return false; }
  }

  // Held back as a measurement would hold it back, which is by what was asked for rather than by what ran.
  const std::string& spec = std::get<0>(key_);
  if (context.db.isNogo(context.env, spec, options)) { return false; }
  u32 const cfg = context.db.findCfgId(options);
  return !(cfg && context.db.diedOn(context.env, cfg, entry_.kind, spec, exponent));
}

void EntrySearch::resume(const SearchContext& context, const std::string& canonical, Candidate& candidate) const {
  // Where it was started, so that the calls already made pool with the ones still to make.
  if (auto const p = context.progress.partial.find({key_, canonical});
      p != context.progress.partial.end() && entry_.band.contains(p->second.exponent)) {
    candidate.exponent = p->second.exponent;
    candidate.calls = p->second.calls;
  }
}

EntrySearch::ListMemo& EntrySearch::probeList(const SearchContext& context, const UseConfig& best,
                                              std::span<const Reading> readings, bool structuralSteps, std::string from,
                                              bool widen) {
  UseConfig const canonical = canonicalConfig(context.device, entry_.fft, best);
  auto const [at, fresh] = lists_.try_emplace(configText(canonical) + (structuralSteps ? "" : " within"));
  ListMemo& memo = at->second;
  if (!fresh && memo.from == from && !widen) { return memo; }
  if (widen) { memo.listed = memo.listed > NO_LIMIT / 2 ? NO_LIMIT : 2 * memo.listed; }

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
  memo.list = probesOf(context.device, entry_.fft, canonical, context.strategy, readings, structuralSteps, memo.listed);
  memo.answered.assign(memo.list.probes.size(), false);
  for (size_t p = 0; p < memo.list.probes.size(); ++p) {
    memo.answered[p] = answered.contains(identity(memo.list.probes[p]));
  }
  memo.checked.clear();
  return memo;
}

const UseConfig& EntrySearch::draw(const Env& device, u32 k) {
  auto const [at, fresh] = draws_.try_emplace(k);
  if (fresh) { at->second = canonicalConfig(device, entry_.fft, restartOf(device, entry_.fft, entry_.label(), k)); }
  return at->second;
}

std::optional<Candidate> EntrySearch::nextRestart(const SearchContext& context) {
  const std::string& spec = std::get<0>(key_);
  const std::string& regime = std::get<2>(key_);

  // From the last draw declared, which a stop may have left partly measured, so that it is resumed rather than skipped.
  for (const JumpRow& row : context.db.jumps()) {
    if (context.db.envOf(row.sess) == context.env && row.fft == spec && row.kind == entry_.kind &&
        row.regime.label() == regime) {
      scan_.next = std::max(scan_.next, row.k);
    }
  }
  if (auto const sets = context.progress.sets.find(key_); sets != context.progress.sets.end()) {
    scan_.seen.insert(sets->second.begin(), sets->second.end());
  }

  auto candidateOf = [&](u32 k) {
    const UseConfig& drawn = draw(context.device, k);
    std::string const text = configText(drawn);
    Candidate out{.kind = Offer::Restart,
                  .options = drawn,
                  .what = "#" + std::to_string(k + 1) + " " + (drawn.empty() ? "the built-in defaults" : text),
                  .exponent = entry_.exponent,
                  .draw = k};
    resume(context, text, out);
    return out;
  };

  auto ok = [&](u32 k) {
    if (context.progress.answered.contains({key_, configText(draw(context.device, k))})) { return false; }
    Candidate const c = candidateOf(k);
    return runnable(context, c.options, c.exponent);
  };

  std::optional<u32> const k = nextRunnable(scan_, [&](u32 k) { return configText(draw(context.device, k)); }, ok);
  if (!k) { return {}; }
  return candidateOf(*k);
}

std::vector<Candidate> EntrySearch::offers(const SearchContext& context, std::span<const Reading> readings,
                                           const Worth& worth) {
  assert(!readings.empty());
  const Env& device = context.device;
  const Progress& progress = context.progress;
  const Reading& best = readings.front();

  std::vector<Branch> branches;
  if (context.strategy.branches()) {
    branches = branchesOf(device, entry_.fft, readings);
  } else {
    branches.push_back({.structure = {}, .best = best.config, .cost = best.cost});
  }

  std::vector<UseConfig> const none;
  auto const rows = progress.concluded.find(key_);
  const std::vector<UseConfig>& concluded = rows != progress.concluded.end() ? rows->second : none;
  std::vector<std::string> texts;
  for (const UseConfig& row : concluded) { texts.push_back(configText(row)); }

  // What each branch's readings say, which is all its combo tiers read.
  std::vector<std::string> from(branches.size());
  if (context.strategy.combines()) {
    for (const Reading& r : readings) {
      UseConfig const structure = branchOf(device, entry_.fft, r.config);
      auto const in = std::ranges::find_if(branches, [&](const Branch& br) { return br.structure == structure; });
      if (in != branches.end()) {
        from[size_t(in - branches.begin())] += configText(r.config) + " " + std::to_string(r.cost) + ";";
      }
    }
  }

  std::vector<Candidate> out;
  std::set<std::string> offered;

  // The lines move as other entries are searched, so an entry is offered them as they stand whatever it has found
  // itself, and where it has found something, what it found with the lines laid over it.
  auto offerLines = [&](UseConfig jump, const std::string& what) {
    std::string const text = configText(jump);
    if (jump.empty() || offered.contains(text)) { return; }
    Candidate c{.kind = Offer::Lines,
                .options = std::move(jump),
                .what = what + text,
                .exponent = entry_.exponent,
                .cost = best.cost,
                .value = worth(Offer::Lines, best.cost)};
    resume(context, text, c);
    if (c.value > 0 && !progress.answered.contains({key_, text}) && runnable(context, c.options, c.exponent)) {
      offered.insert(text);
      out.push_back(std::move(c));
    }
  };
  offerLines(underDefaults(device, entry_.fft, entry_.kind, context.lines), "the default lines ");
  if (!best.config.empty()) {
    offerLines(underDefaults(device, entry_.fft, entry_.kind, context.lines, best.config),
               "its best set under the default lines ");
  }

  for (size_t branch = 0; branch < branches.size(); ++branch) {
    double const cost = branches[branch].cost;
    double const value = worth(Offer::Probe, cost);
    double const comboValue = worth(Offer::Combo, cost);
    if (value <= 0 && comboValue <= 0) { continue; }

    // Offers what the listing in `memo` has not answered.  Unless it is the `last` listing, false where a stage listed
    // in part offered nothing, having then offered nothing at all.
    auto offerListed = [&](ListMemo& memo, bool last) {
      const ProbeList& list = memo.list;
      for (size_t r = 0; r < concluded.size(); ++r) {
        if (!memo.checked.insert(texts[r]).second) { continue; }
        for (size_t p = 0; p < list.probes.size(); ++p) {
          if (!memo.answered[p]) {
            memo.answered[p] = answeredBy(device, entry_.fft, list, list.probes[p], concluded[r]);
          }
        }
      }

      // The list is in tier order, and a combination waits for the tiers below it to be answered, the points of a stage
      // not listed yet among them.
      std::optional<u32> lowest;
      for (const ProbeList::Unlisted& u : list.unlisted) { lowest = std::min(lowest.value_or(u.tier), u.tier); }
      size_t const first = out.size();
      std::vector<std::string> taken;
      std::set<u32> fed;
      for (size_t p = 0; p < list.probes.size(); ++p) {
        if (memo.answered[p]) { continue; }
        const Probe& probe = list.probes[p];
        if (lowest && probe.tier > *lowest) { break; }
        double const probeWorth = probe.tier > 1 ? comboValue : value;
        if (probeWorth <= 0) { continue; }
        std::string const text = configText(probe.config);
        if (progress.failed.contains({key_, text})) { continue; }

        Candidate c{.kind = probe.tier > 1 ? Offer::Combo : Offer::Probe,
                    .options = probe.config,
                    .moved = probe.key,
                    .what = probe.stage + " " + probe.text,
                    .exponent = entry_.exponent,
                    .tier = probe.tier,
                    .cost = cost,
                    .value = probeWorth};
        resume(context, text, c);
        if (!runnable(context, c.options, c.exponent)) { continue; }
        if (!offered.insert(text).second) { continue; }
        taken.push_back(text);
        lowest = std::min(lowest.value_or(probe.tier), probe.tier);

        if (fed.insert(probe.part).second) {
          auto const u = std::ranges::find(list.unlisted, probe.part, &ProbeList::Unlisted::part);
          if (u != list.unlisted.end()) { c.unlisted = u->most; }
        }
        out.push_back(std::move(c));
      }

      // A part of a tier above the one offered is waiting for it, not starved.
      bool const starved = std::ranges::any_of(
        list.unlisted, [&](const ProbeList::Unlisted& u) { return u.tier <= *lowest && !fed.contains(u.part); });
      if (!starved || last) { return true; }
      out.erase(out.begin() + ptrdiff_t(first), out.end());
      for (const std::string& text : taken) { offered.erase(text); }
      return false;
    };

    // A stage listed in part whose listed points offer nothing lists more of them, so that its next points are offered
    // as they would be were it listed whole.
    ListMemo* memo = &probeList(context, branches[branch].best, readings, branch == 0, std::move(from[branch]));
    for (u32 widened = 0; !offerListed(*memo, widened == MAX_WIDENINGS); ++widened) {
      memo = &probeList(context, branches[branch].best, readings, branch == 0, memo->from, true);
    }
  }

  // A jump is offered only once no step is left, rather than left to a price to rank below them.
  if (context.restarts && out.empty()) {
    if (double const value = worth(Offer::Restart, best.cost); value > 0) {
      if (std::optional<Candidate> next = nextRestart(context)) {
        next->cost = best.cost;
        next->value = value;
        out.push_back(std::move(*next));
      }
    }
  }
  return out;
}

}  // namespace tune
