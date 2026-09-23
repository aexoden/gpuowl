// Copyright (C) Jason Lynch

#include "Emit.h"

#include "Args.h"
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
struct Candidate {
  SelectionEntry entry;
  u32 calls = 0;
  u64 ts = 0;
};

// Lower is better.
bool better(const Candidate& a, const Candidate& b) {
  return std::tuple{a.entry.cost, b.calls, b.ts} < std::tuple{b.entry.cost, a.calls, a.ts};
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
std::set<Configuration> condemned(const TuneDB& db, u32 env, const std::vector<RunRow>& runs) {
  std::set<Configuration> out;

  for (const RunRow& row : runs) {
    if (db.envOf(row.sess) != env || row.m.status != Status::Err) { continue; }
    const UseConfig* const opts = db.findCfg(row.cfg);
    if (opts) { out.insert(configurationOf(row, *opts)); }
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

// The reach and evidence state recorded for a configuration, keyed as the reach rows are keyed but by the option set
// itself rather than by the id naming it, so that a database declaring one set twice does not hide half its evidence.
class Reaches {
public:
  Reaches(const TuneDB& db, u32 env) {
    for (const ReachRow& row : db.latestReaches()) {
      if (db.envOf(row.sess) != env) { continue; }
      const UseConfig* const opts = db.findCfg(row.cfg);
      if (!opts) { continue; }

      // The database's own fold groups these by the id naming the option set, and hands them back in the order the
      // ids first appeared rather than in the order they were written.  Two ids spelling one set therefore arrive in
      // an order that says nothing about which reading is the later one, so it is asked for here.
      auto const [at, fresh] = rows_.try_emplace(Key{row.fft, row.kind, row.regime.label(), configText(*opts)}, row);
      if (!fresh && row.ts >= at->second.ts) { at->second = row; }
    }
  }

  // Inherited where nothing was measured: the fitted table's own limit, marked as the state that says so.
  [[nodiscard]] std::pair<u64, Evidence> operator()(const FFTConfig& fft, const RunRow& row,
                                                    const UseConfig& opts) const {
    auto const it = rows_.find(Key{row.fft, row.kind, row.regime.label(), configText(opts)});
    if (it != rows_.end()) { return {it->second.reach, it->second.evidence}; }

    return {maxExp(fft), exactArithmetic(fft) ? Evidence::NotApplicable : Evidence::Unvalidated};
  }

private:
  using Key = std::tuple<std::string, TestKind, std::string, std::string>;

  std::map<Key, ReachRow> rows_;
};

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

std::vector<Candidate> identityFrontier(const TuneDB& db, u32 env, const Defaults& defaults) {
  const DbEnv* const row = db.findEnv(env);
  if (!row) { return {}; }

  Env const built = row->toEnv();
  Reaches const reachOf{db, env};

  std::vector<RunRow> const runs = db.mergedRuns();
  std::set<Configuration> const failed = condemned(db, env, runs);

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

    auto const [reach, evidence] = reachOf(*fft, row, *opts);
    Interval const span = interval(*fft, row.exponent, reach);
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
                                  .evidence = evidence,
                                  .opts = *opts},
                        .calls = row.m.calls,
                        .ts = row.m.ts};

    candidate.entry.id = entryId(candidate.entry.fft, candidate.entry.kind, candidate.entry.regime, *opts);

    // Not reported: every row a bootstrap race took before its later races moved the background is one of these, so it
    // is how the database ordinarily looks rather than something wrong with it.
    if (!shadowedKeys(defaults, built, *fft, candidate.entry).empty()) { continue; }

    auto const [at, fresh] = byId.emplace(candidate.entry.id, candidate);
    if (!fresh && better(candidate, at->second)) { at->second = candidate; }
  }

  std::vector<Candidate> candidates;
  for (auto& [id, candidate] : byId) { candidates.push_back(std::move(candidate)); }

  // A configuration that costs more but reaches further is kept, because nothing else may reach that far cheaply, and
  // picking one option set per identity before the table is built would lose it.
  return frontier(candidates, [](const Candidate& c) { return identityOf(c.entry); });
}

}  // namespace

std::vector<SelectionEntry> candidatesFor(const TuneDB& db, u32 env, const Defaults& defaults) {
  std::vector<SelectionEntry> out;
  for (Candidate& c : identityFrontier(db, env, defaults)) { out.push_back(std::move(c.entry)); }
  return out;
}

std::vector<SelectionEntry> entriesFor(const TuneDB& db, u32 env, const Defaults& defaults) {
  // Then across the table, per kind, since an entry no exponent would ever choose is one production would only walk
  // past.
  std::vector<Candidate> candidates =
    frontier(identityFrontier(db, env, defaults), [](const Candidate& c) { return c.entry.kind; });

  std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) {
    return std::tuple{a.entry.cost, a.entry.id} < std::tuple{b.entry.cost, b.entry.id};
  });

  std::vector<SelectionEntry> out;
  out.reserve(candidates.size());
  for (Candidate& c : candidates) { out.push_back(std::move(c.entry)); }

  return out;
}

std::optional<SelectionFile> emit(const TuneDB& db, const Defaults& defaults, const Provenance& from) {
  SelectionFile file{.provenance = provenanceOf(from),
                     .global = {defaults.global.begin(), defaults.global.end()},
                     .family = defaults.family,
                     .entries = entriesFor(db, from.env, defaults),
                     .unknown = {}};

  if (!finalize(file)) { return {}; }
  return file;
}

bool publish(const fs::path& path, const TuneDB& db, const Defaults& defaults, const Provenance& from) {
  auto const file = emit(db, defaults, from);
  if (!file) { return false; }

  writeSelection(path, *file);
  return true;
}

}  // namespace tune
