// Copyright (C) Jason Lynch

#include "Search.h"

#include "Bootstrap.h"

#include <algorithm>
#include <cassert>
#include <tuple>
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
    u64& last = out.sets[key][text];
    last = std::max(last, row.m.ts);
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
      p.options = canonical;
      p.mean = row.m.mean;
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
  u32 const cfg = context.db.findCfgId(options);
  return !(cfg && context.db.diedOn(context.env, cfg, entry_.kind, spec, exponent));
}

void EntrySearch::resume(const SearchContext& context, const std::string& canonical, Candidate& candidate) const {
  // Where it was started, so that the calls already made pool with the ones still to make.
  if (auto const p = context.progress.partial.find({key_, canonical});
      p != context.progress.partial.end() && entry_.band.contains(p->second.exponent)) {
    candidate.exponent = p->second.exponent;
    candidate.calls = p->second.calls;
    candidate.observed = p->second.mean;
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
  std::set<std::string> seen;
  for (size_t p = 0; p < memo.list.probes.size(); ++p) {
    if (memo.answered[p]) { answered.insert(identity(memo.list.probes[p])); }
    if (memo.seen[p]) { seen.insert(identity(memo.list.probes[p])); }
  }

  memo.from = std::move(from);
  memo.list = probesOf(context.device, entry_.fft, canonical, context.strategy, readings, structuralSteps, memo.listed);
  size_t const n = memo.list.probes.size();
  memo.texts.resize(n);
  memo.answered.assign(n, false);
  memo.seen.assign(n, false);
  for (size_t p = 0; p < n; ++p) {
    const Probe& probe = memo.list.probes[p];
    memo.texts[p] = configText(probe.config);
    memo.answered[p] = answered.contains(identity(probe));
    memo.seen[p] = seen.contains(identity(probe));
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
    for (const auto& [text, ts] : sets->second) { scan_.seen.insert(text); }
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

bool EntrySearch::restartDue(const SearchContext& context) const {
  auto const sets = context.progress.sets.find(key_);
  if (sets == context.progress.sets.end()) { return false; }

  const JumpRow* declared = nullptr;
  for (const JumpRow& row : context.db.jumps()) {
    if (context.db.envOf(row.sess) == context.env && row.fft == std::get<0>(key_) && row.kind == entry_.kind &&
        row.regime.label() == std::get<2>(key_) && (!declared || row.ts >= declared->ts)) {
      declared = &row;
    }
  }

  // A draw begun is finished first: until it concludes, its calls count for nothing.
  if (declared) {
    const UseConfig* const drawn = context.db.findCfg(declared->cfg);
    if (drawn &&
        context.progress.partial.contains({key_, configText(canonicalConfig(context.device, entry_.fft, *drawn))})) {
      return true;
    }
  }
  auto const since =
    std::ranges::count_if(sets->second, [&](const auto& set) { return !declared || set.second > declared->ts; });
  return u64(since) >= RESTART_PERIOD;
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

  // A restart draw is a sample of the space, not a step from a best set: it says nothing of what one of its keys does.
  std::map<std::string, u32> draws;
  for (const JumpRow& row : context.db.jumps()) {
    if (context.db.envOf(row.sess) != context.env || row.fft != std::get<0>(key_) || row.kind != entry_.kind ||
        row.regime.label() != std::get<2>(key_)) {
      continue;
    }
    if (const UseConfig* const drawn = context.db.findCfg(row.cfg)) {
      draws.emplace(configText(canonicalConfig(device, entry_.fft, *drawn)), row.k);
    }
  }

  // What each branch's readings say, which is all its combo tiers read.
  std::vector<std::string> from(branches.size());
  if (context.strategy.combines()) {
    for (const Reading& r : readings) {
      UseConfig const structure = branchOf(device, entry_.fft, r.config);
      auto const in = std::ranges::find_if(branches, [&](const Branch& br) { return br.structure == structure; });
      if (in != branches.end()) {
        from[size_t(in - branches.begin())] +=
          configText(r.config) + " " + std::to_string(r.cost) + " " + std::to_string(r.error) + ";";
      }
    }
  }

  std::vector<Candidate> out;
  std::set<std::string> offered;

  auto const pinned = [&](const UseConfig& options) {
    return context.db.failedWith(context.env, std::get<0>(key_), options);
  };

  bool restarted = false;
  if (context.restarts && restartDue(context)) {
    if (double const value = worth(Offer::Probe, best.cost); value > 0) {
      if (std::optional<Candidate> next = nextRestart(context)) {
        next->cost = best.cost;
        next->value = value;
        offered.insert(configText(next->options));
        out.push_back(std::move(*next));
        restarted = true;
      }
    }
  }

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

  // What the branches offer, before it is put in the order the entry takes it: steps never taken, then those taken
  // from another best set, then those that set a value a build failed with.
  struct Listed {
    Candidate candidate;
    int pass = 0;
    bool structural = false;
    size_t branch = 0;

    // A step never taken: how many steps of its stage go before it, the ones the rows have taken among them.
    size_t turn = 0;
  };
  std::vector<Listed> listed;
  bool const rotate = context.strategy.branches();

  for (size_t branch = 0; branch < branches.size(); ++branch) {
    double const cost = branches[branch].cost;
    double const value = worth(Offer::Probe, cost);
    double const comboValue = worth(Offer::Combo, cost);
    if (value <= 0 && comboValue <= 0) { continue; }

    // A step taken from another best set is priced as a jump is, by what the entry's jumps have found: that it did
    // nothing there is evidence against it here, though not a verdict.
    std::optional<double> againValue;
    auto const again = [&] { return againValue ? *againValue : *(againValue = worth(Offer::Restart, cost)); };

    // Lists what the listing in `memo` has not answered.  Unless it is the `last` listing, false where a stage listed
    // in part offered nothing, having then listed nothing at all.
    auto offerListed = [&](ListMemo& memo, bool last) {
      const ProbeList& list = memo.list;
      for (size_t r = 0; r < concluded.size(); ++r) {
        if (!memo.checked.insert(texts[r]).second) { continue; }
        bool const drawn = draws.contains(texts[r]);
        for (size_t p = 0; p < list.probes.size(); ++p) {
          if (memo.answered[p]) { continue; }
          memo.answered[p] = memo.texts[p] == texts[r];
          if (!memo.answered[p] && !memo.seen[p] && !drawn) {
            memo.seen[p] = sameStep(device, entry_.fft, list, list.probes[p], concluded[r]);
          }
        }
      }

      // What each stage has had: its steps a row has taken, from this best set or another.
      std::map<u32, size_t> had;
      for (size_t p = 0; p < list.probes.size(); ++p) {
        if (memo.answered[p] || memo.seen[p]) { ++had[list.probes[p].part]; }
      }

      // A combination waits for the steps of the groups it combines, the points of a stage of theirs not listed yet
      // among them, since it combines what those found; steps of other groups do not hold it back.  A step taken from
      // another best set is no answer here, but it waits for the steps that were never taken, a combination included:
      // what it did there is some evidence of what it will do here.  A step that sets a value a build failed with
      // waits for both.
      std::optional<u32> lowest;
      std::map<u32, std::set<Group>> waiting;
      for (const ProbeList::Unlisted& u : list.unlisted) {
        lowest = std::min(lowest.value_or(u.tier), u.tier);
        waiting[u.tier].insert(u.groups.begin(), u.groups.end());
      }
      auto const held = [&](const Probe& probe) {
        for (auto const& [tier, groups] : waiting) {
          if (tier >= probe.tier) { break; }
          if (std::ranges::any_of(probe.groups, [&](Group g) { return groups.contains(g); })) { return true; }
        }
        return false;
      };

      size_t const first = listed.size();
      std::vector<std::string> taken;
      std::set<u32> fed;
      std::map<u32, size_t> turns;
      auto const rank = [&](size_t p) { return pinned(list.probes[p].config) ? 2 : memo.seen[p] ? 1 : 0; };
      for (int const pass : {0, 1, 2}) {
        for (size_t p = 0; p < list.probes.size(); ++p) {
          if (memo.answered[p] || rank(p) != pass) { continue; }
          const Probe& probe = list.probes[p];
          if (held(probe)) { continue; }
          double const probeWorth = pass == 1 ? again() : probe.tier > 1 ? comboValue : value;
          if (probeWorth <= 0) { continue; }
          const std::string& text = memo.texts[p];
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
          waiting[probe.tier].insert(probe.groups.begin(), probe.groups.end());

          if (fed.insert(probe.part).second) {
            auto const u = std::ranges::find(list.unlisted, probe.part, &ProbeList::Unlisted::part);
            if (u != list.unlisted.end()) { c.unlisted = u->most; }
          }
          listed.push_back({.candidate = std::move(c),
                            .pass = pass,
                            .structural = probe.structural,
                            .branch = branch,
                            .turn = pass == 0 && rotate ? had[probe.part] + turns[probe.part]++ : 0});
        }
      }

      // A part of a tier above the one offered is waiting for it, not starved.
      bool const starved = std::ranges::any_of(
        list.unlisted, [&](const ProbeList::Unlisted& u) { return u.tier <= *lowest && !fed.contains(u.part); });
      if (!starved || last) { return true; }
      listed.erase(listed.begin() + ptrdiff_t(first), listed.end());
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

  // Structural steps first, since which side of a structural key is the faster one decides where the rest of the
  // search is best spent.  Then, where the strategy searches by group, the stages take turns a step at a time, a step
  // going after as many steps of every other stage as its own has had, so that a large stage does not keep the rest
  // waiting until it is done, and a resumed run takes them in the order an uninterrupted one would.  A stage of the
  // best branch takes a turn for every turn the same stage takes in all the other branches together.  One key at a
  // time, no stage is large, and each key's values are taken together, as coordinate descent takes them.
  size_t const others = std::max<size_t>(1, branches.size() - 1);
  std::ranges::stable_sort(listed, {}, [&](const Listed& l) {
    return std::tuple{l.pass, !l.structural, l.branch ? l.turn * others : l.turn, l.branch};
  });
  for (Listed& l : listed) { out.push_back(std::move(l.candidate)); }

  // A measurement begun is finished, even where nothing above lists it any more or a step from the best set is worth
  // nothing: the best set it was a step from, the lines it was laid under or the stage it was listed in have moved on
  // since, and its call counts for nothing until it concludes.  The built-in defaults are the sweep's.
  for (auto p = progress.partial.lower_bound({key_, {}}); p != progress.partial.end() && p->first.first == key_; ++p) {
    const std::string& text = p->first.second;
    const Partial& partial = p->second;
    if (partial.options.empty() || offered.contains(text) || progress.answered.contains(p->first) ||
        !entry_.band.contains(partial.exponent) || !runnable(context, partial.options, partial.exponent)) {
      continue;
    }
    auto const drawn = draws.find(text);
    Candidate c{.kind = drawn != draws.end() ? Offer::Restart : Offer::Probe,
                .options = partial.options,
                .what = (drawn != draws.end() ? "#" + std::to_string(drawn->second + 1) + " " : "unfinished ") + text,
                .exponent = partial.exponent,
                .calls = partial.calls,
                .observed = partial.mean,
                .draw = drawn != draws.end() ? drawn->second : 0,
                .cost = best.cost,
                .value = std::max(0.0, worth(Offer::Probe, best.cost))};
    offered.insert(text);
    out.push_back(std::move(c));
  }

  // Otherwise a jump is offered only once no step is left, rather than left to a price to rank below them.
  if (context.restarts && !restarted && out.empty()) {
    if (double const value = worth(Offer::Restart, best.cost); value > 0) {
      if (std::optional<Candidate> next = nextRestart(context)) {
        next->cost = best.cost;
        next->value = value;
        out.push_back(std::move(*next));
      }
    }
  }

  // A measurement begun first, since its calls count for nothing until it concludes.
  (void)std::ranges::stable_partition(out, [](const Candidate& c) { return c.calls > 0; });
  (void)std::ranges::stable_partition(out, [&](const Candidate& c) { return !pinned(c.options); });
  return out;
}

}  // namespace tune
