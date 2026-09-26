// Copyright (C) Jason Lynch

#include "Emit.h"

#include "Args.h"
#include "Bootstrap.h"
#include "CycleFile.h"
#include "Gate.h"
#include "log.h"
#include "version.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace tune {

namespace {

// An entry under construction, beside the row it is a transcript of.  The row is kept only for the tie-breaks: two
// candidates that cost the same and reach the same distance are separated by how well each is supported, so that
// re-emitting an unchanged database produces an unchanged file.
using Candidate = OptionSet;

// Lower is better.
bool better(const Candidate& a, const Candidate& b) {
  return std::tuple{a.entry.cost, b.m.calls, b.m.ts} < std::tuple{b.entry.cost, a.m.calls, a.m.ts};
}

// Whether `a` is at least as good as `b` everywhere and better somewhere: no more expensive, starting no higher, and
// reaching no lower.  Two candidates equal on all three leave each other standing, since neither displaces the other
// and dropping one would be a coin toss written into the file.
bool dominates(const SelectionEntry& a, const SelectionEntry& b) {
  bool const asGood = a.cost <= b.cost && a.emin <= b.emin && a.reach >= b.reach;
  return asGood && (a.cost < b.cost || a.emin < b.emin || a.reach > b.reach);
}

// The candidates no other candidate of the same `group` dominates.  Quadratic in the size of a group, which is the
// number of option sets measured for one FFT and not a number that grows with the run.
template<typename Group> std::vector<Candidate> frontier(const std::vector<Candidate>& in, Group group) {
  std::vector<Candidate> out;

  for (const Candidate& c : in) {
    bool const beaten = std::ranges::any_of(
      in, [&](const Candidate& other) { return group(other) == group(c) && dominates(other.entry, c.entry); });

    if (!beaten) { out.push_back(c); }
  }

  return out;
}

// A configuration in a regime: what an entry is published for, and what a wrong answer is evidence about.
using Configuration = std::tuple<std::string, TestKind, std::string, std::string>;

Configuration configurationOf(const RunRow& row, const UseConfig& opts) {
  return {row.fft, row.kind, row.regime.label(), configText(opts)};
}

// The configurations that computed a wrong answer somewhere in a regime.  The kernels are what a wrong residue or a
// failed Gerbicz check is evidence about, and the regime is the span over which they are constant -- which is the same
// reason an entry covers one -- so such a row withdraws the configuration from the whole of it rather than from the
// exponent that happened to catch it.  A build that would not compile or a run the backend refused says nothing about
// the answers the configuration computes, and a lost device is attributed to the fault and not to what was running.
std::map<Configuration, Exclusion> condemned(const TuneDB& db, u32 env, const std::vector<RunRow>& runs) {
  std::map<Configuration, Exclusion> out;

  for (const RunRow& row : runs) {
    if (db.envOf(row.sess) != env || row.m.status != Status::Err) { continue; }
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (opts) {
      out.try_emplace(configurationOf(row, *opts),
                      Exclusion{.fft = row.fft, .kind = row.kind, .regime = row.regime, .opts = *opts});
    }
  }

  return out;
}

// The keys this file's own default lines would add to an option set that does not name them.  An entry outranks those
// lines, so a set that is complete -- as a recorded one is -- leaves nothing for them to add, and where something is
// left the entry's cost is a reading of settings other than the ones production would resolve.
std::vector<std::string> shadowedKeys(const Defaults& defaults, const Env& env, const FFTConfig& fft,
                                      const SelectionEntry& entry) {
  SelectionLayers const layers = fittedTo({.global = {defaults.global.begin(), defaults.global.end()},
                                           .family = defaults.family,
                                           .entry = {entry.opts.begin(), entry.opts.end()}},
                                          env, fft, entry.kind);

  std::vector<std::string> out;
  for (const auto& [key, value] : resolveConfig(Args{true}, fft, entry.kind, layers)) {
    if (entry.opts.contains(key)) { continue; }

    // A key the table does not know is one nothing here can call inert, so it counts.  One the lines only set to the
    // value the entry ran at anyway, its default, changes nothing the kernels see.
    const Option* const option = findOption(key);
    if (!option) {
      out.push_back(key);
      continue;
    }
    if (!option->appliesTo(env, fft, entry.opts) || option->isInert(env, fft, entry.opts)) { continue; }
    if (parseInt<int>(value) != option->defaultFor(env, fft, entry.opts)) { out.push_back(key); }
  }

  return out;
}

using Identity = std::tuple<std::string, TestKind, std::string>;

Identity identityOf(const SelectionEntry& e) { return {e.fft, e.kind, e.regime.label()}; }

}  // namespace

std::string provenanceOf(const Provenance& from) {
  char buf[256];
  snprintf(buf, sizeof(buf), "written %llu by %s from %s env %u", (unsigned long long)from.ts, VERSION,
           from.db.empty() ? "-" : from.db.c_str(), from.env);

  std::string out = buf;
  if (from.workloadHi >= from.workloadLo && from.workloadHi > 0) {
    snprintf(buf, sizeof(buf), "; T=%.1f over workload %llu-%llu", from.T, (unsigned long long)from.workloadLo,
             (unsigned long long)from.workloadHi);
    out += buf;
  }

  return out;
}

bool shadowedBy(const Defaults& defaults, const Env& env, const FFTConfig& fft, TestKind kind, const UseConfig& opts) {
  SelectionEntry entry;
  entry.kind = kind;
  entry.opts = opts;
  return !shadowedKeys(defaults, env, fft, entry).empty();
}

namespace {

std::vector<Candidate> gatedSets(const TuneDB& db, u32 env, const Defaults& defaults, Gating gating) {
  std::vector<Candidate> sets = optionSetsFor(db, env, defaults);
  if (gating == Gating::Required) {
    std::erase_if(sets, [](const Candidate& c) { return c.gate.state != GateState::Passed; });
  }
  return sets;
}

// A configuration that costs more but reaches further is kept, because nothing else may reach that far cheaply, and
// picking one option set per identity before the table is built would lose it.
std::vector<Candidate> identityFrontier(const std::vector<Candidate>& sets) {
  return frontier(sets, [](const Candidate& c) { return identityOf(c.entry); });
}

void sortByCost(std::vector<Candidate>& sets) {
  std::ranges::sort(sets, [](const Candidate& a, const Candidate& b) {
    return std::tuple{a.entry.cost, a.entry.id} < std::tuple{b.entry.cost, b.entry.id};
  });
}

// Then across the table, per kind, since an entry no exponent would ever choose is one production would only walk
// past.  Cheapest first.
std::vector<Candidate> tableOf(const std::vector<Candidate>& sets) {
  std::vector<Candidate> out = frontier(identityFrontier(sets), [](const Candidate& c) { return c.entry.kind; });
  sortByCost(out);
  return out;
}

std::vector<Candidate> tableFor(const TuneDB& db, u32 env, const Defaults& defaults, Gating gating) {
  return tableOf(gatedSets(db, env, defaults, gating));
}

// Whether the reach derived for a passed set stops short of the one the fitted table gives its FFT.  Production has to
// hold that arithmetic to it on paths that never walk to the set's own entry -- a shape scan landing on the same FFT,
// or another entry's options overridden into it -- and the file is the only place it can learn it from.
bool reduced(const Candidate& c) {
  if (c.gate.state != GateState::Passed || !c.gate.derived) { return false; }
  auto const fft = parseFft(c.entry.fft);
  return fft && c.entry.reach < interval(*fft, c.exponent).hi;
}

// The table, and every set whose reach was reduced that the table's frontier dropped as covered more cheaply.
std::vector<Candidate> publishedFor(const TuneDB& db, u32 env, const Defaults& defaults) {
  std::vector<Candidate> const sets = gatedSets(db, env, defaults, Gating::Required);
  std::vector<Candidate> out = tableOf(sets);

  std::set<std::string> ids;
  for (const Candidate& c : out) { ids.insert(c.entry.id); }

  for (const Candidate& c : sets) {
    if (reduced(c) && !ids.contains(c.entry.id)) { out.push_back(c); }
  }

  sortByCost(out);
  return out;
}

// Every option set of every identity, each with the gate's verdict on it, the rejected ones included.
[[nodiscard]] std::vector<Candidate> judgedSets(const TuneDB& db, u32 env, const Defaults& defaults) {
  const DbEnv* const row = db.findEnv(env);
  if (!row) { return {}; }

  Env const built = row->toEnv();
  Gates const gates{db, env, built};

  std::vector<RunRow> const runs = db.mergedRuns();
  std::map<Configuration, Exclusion> const failed = condemned(db, env, runs);

  // One candidate per option set of an identity.  A configuration measured at two exponents of one regime is two rows
  // of one thing -- the cost is per iteration, and the regime is what decides which kernels ran -- so the better
  // supported of the two is what gets published, rather than both under one id.
  std::map<std::string, Candidate> byId;

  for (const RunRow& row : runs) {
    if (db.envOf(row.sess) != env || !concluded(row.m)) { continue; }

    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (!opts || !fft) { continue; }

    if (!isWriteableConfig(*opts)) {
      log("emit: skipping %s, measured under an option this format cannot write back\n", row.fft.c_str());
      continue;
    }

    if (failed.contains(configurationOf(row, *opts))) { continue; }

    Interval const span = interval(*fft, row.exponent);
    if (span.empty()) { continue; }

    // A row whose regime is not the one its own exponent runs in was written by something that disagrees with this
    // build about what the kernels do, and nothing downstream can tell which of the two is right.
    if (span.regime != row.regime) {
      log("emit: skipping %s at %llu, recorded as %s where it runs as %s\n", row.fft.c_str(),
          (unsigned long long)row.exponent, row.regime.label().c_str(), span.regime.label().c_str());
      continue;
    }

    Candidate candidate{.entry = {.id = {},
                                  .cost = pessimisticCost(row.m),
                                  .fft = row.fft,
                                  .kind = row.kind,
                                  .emin = span.lo,
                                  .reach = span.hi,
                                  .regime = span.regime,
                                  .evidence = Evidence::Unvalidated,
                                  .opts = *opts},
                        .m = row.m,
                        .exponent = row.exponent,
                        .gate = {}};

    candidate.entry.id = entryId(candidate.entry.fft, candidate.entry.kind, candidate.entry.regime, *opts);

    // Not reported: every row a bootstrap race took before its later races moved the background is one of these, so it
    // is how the database ordinarily looks rather than something wrong with it.
    if (!shadowedKeys(defaults, built, *fft, candidate.entry).empty()) { continue; }

    auto const [at, fresh] = byId.emplace(candidate.entry.id, candidate);
    if (!fresh && better(candidate, at->second)) { at->second = candidate; }
  }

  // Judged once per set rather than once per row: the rows of one set in one regime share its interval.  The cost is
  // per iteration and the same anywhere in the regime, so a set timed above the reach the gate derived for it is still
  // published below it, and one whose reach was raised above the table is published above it.
  std::vector<Candidate> candidates;
  for (auto& [id, candidate] : byId) {
    SelectionEntry& e = candidate.entry;
    candidate.gate = gates(*parseFft(e.fft), Interval{.lo = e.emin, .hi = e.reach, .regime = e.regime}, e.opts);
    if (candidate.gate.state == GateState::Passed) {
      e.reach = candidate.gate.reach;
      e.evidence = candidate.gate.evidence;
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

}  // namespace

std::vector<OptionSet> optionSetsFor(const TuneDB& db, u32 env, const Defaults& defaults) {
  std::vector<OptionSet> out = judgedSets(db, env, defaults);
  std::erase_if(out, [](const OptionSet& s) { return s.gate.state == GateState::Rejected; });
  return out;
}

std::vector<OptionSet> rejectedSets(const TuneDB& db, u32 env, const Defaults& defaults) {
  std::vector<OptionSet> out = judgedSets(db, env, defaults);
  std::erase_if(out, [](const OptionSet& s) { return s.gate.state != GateState::Rejected; });
  return out;
}

std::vector<SelectionEntry> candidatesFor(const TuneDB& db, u32 env, const Defaults& defaults, Gating gating) {
  std::vector<SelectionEntry> out;
  for (Candidate& c : identityFrontier(gatedSets(db, env, defaults, gating))) { out.push_back(std::move(c.entry)); }
  return out;
}

std::vector<SelectionEntry> entriesFor(const TuneDB& db, u32 env, const Defaults& defaults, Gating gating) {
  std::vector<SelectionEntry> out;
  for (Candidate& c : tableFor(db, env, defaults, gating)) { out.push_back(std::move(c.entry)); }
  return out;
}

std::vector<OptionSet> gatesOwed(const TuneDB& db, u32 env, const Defaults& defaults) {
  std::vector<OptionSet> out = tableFor(db, env, defaults, Gating::Assumed);
  std::erase_if(out, [](const OptionSet& s) { return s.gate.state != GateState::Owed; });
  return out;
}

std::optional<SelectionFile> emit(const TuneDB& db, const Defaults& defaults, const Provenance& from) {
  SelectionFile file{.provenance = provenanceOf(from),
                     .global = {defaults.global.begin(), defaults.global.end()},
                     .family = defaults.family,
                     .entries = {},
                     .excluded = {},
                     .unknown = {}};

  for (Candidate& c : publishedFor(db, from.env, defaults)) { file.entries.push_back(std::move(c.entry)); }

  // Leaving a configuration out of the entries only keeps the walk from it; production reaches configurations by other
  // paths, and needs to be told which ones it must not arrive at.
  for (auto& [configuration, exclusion] : condemned(db, from.env, db.mergedRuns())) {
    if (isWriteableConfig(exclusion.opts)) { file.excluded.push_back(std::move(exclusion)); }
  }

  if (!finalize(file)) { return {}; }
  return file;
}

std::vector<TuneEntry> compatibilityView(const SelectionFile& file, const Env& env) {
  std::map<std::string, TuneEntry> cheapest;
  std::set<std::string> shortOfTable;

  // An older binary runs a listed FFT at its defaults, and reads no exclusion that would stop it.
  std::set<std::string> wrongAtDefaults;
  for (const Exclusion& x : file.excluded) {
    auto const fft = parseFft(x.fft);
    if (fft && canonicalConfig(env, *fft, x.opts).empty()) { wrongAtDefaults.insert(fft->spec()); }
  }

  for (const SelectionEntry& e : file.entries) {
    auto const fft = parseFft(e.fft);
    if (!fft) { continue; }

    if (e.evidence != Evidence::NotApplicable) {
      // Only an entry at default rounding speaks for what an older binary runs. One held short of the end of its band,
      // in any regime, is a limit tune.txt cannot express, and the older binary would run the FFT past it.
      if (movesAccuracy(env, *fft, e.opts)) { continue; }
      if (e.reach < interval(*fft, e.emin).hi) {
        shortOfTable.insert(fft->spec());
        continue;
      }

      // And only one in the regime the table's reach ends in says that reach holds.
      if (e.regime != regimeOf(*fft, maxExp(*fft))) { continue; }
    }

    auto const [at, fresh] = cheapest.try_emplace(fft->spec(), TuneEntry{e.cost, *fft});
    if (!fresh) { at->second.cost = std::min(at->second.cost, e.cost); }
  }

  std::vector<TuneEntry> lines;
  for (const auto& [spec, line] : cheapest) {
    if (!shortOfTable.contains(spec) && !wrongAtDefaults.contains(spec)) { lines.push_back(line); }
  }
  std::ranges::sort(lines, [](const TuneEntry& a, const TuneEntry& b) { return a.cost < b.cost; });

  // Upstream's reader keeps the cost/reach frontier and drops the rest, so the file holds exactly that.
  std::vector<TuneEntry> out;
  for (const TuneEntry& line : lines) { (void)line.update(out); }
  return out;
}

std::string compatibilityText(const std::vector<TuneEntry>& view) {
  std::string out;
  for (const TuneEntry& line : view) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%6.1f %14s # %llu\n", line.cost, line.fft.spec().c_str(),
             (unsigned long long)line.fft.maxExp());
    out += buf;
  }
  return out;
}

size_t writeCompatibility(const fs::path& path, const SelectionFile& file, const Env& env) {
  std::vector<TuneEntry> const view = compatibilityView(file, env);
  CycleFile out{path};
  out->write(compatibilityText(view));
  return view.size();
}

bool publish(const fs::path& path, const TuneDB& db, const Defaults& defaults, const Provenance& from,
             const std::optional<fs::path>& compat) {
  auto const file = emit(db, defaults, from);
  if (!file) { return false; }

  writeSelection(path, *file);

  if (compat) {
    const DbEnv* const env = db.findEnv(from.env);
    writeCompatibility(*compat, *file, env ? env->toEnv() : Env{});
  }
  return true;
}

}  // namespace tune
