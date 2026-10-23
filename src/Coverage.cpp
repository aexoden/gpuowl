// Copyright (C) Jason Lynch

#include "Coverage.h"

#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace tune {

namespace {

bool succeeded(const Reading& r) { return !r.failed && r.calls > 0; }

// The seed least like those already chosen, the faster of equals; nothing where every one left is a seed's twin.
const Reading* mostUnlike(const std::vector<const Reading*>& sorted, const std::vector<const Reading*>& seeds) {
  const Reading* best = nullptr;
  size_t bestDistance = 0;
  for (const Reading* r : sorted) {
    if (std::ranges::find(seeds, r) != seeds.end()) { continue; }
    size_t nearest = SIZE_MAX;
    for (const Reading* s : seeds) { nearest = std::min(nearest, axesApart(r->options, s->options)); }
    if (nearest > bestDistance) {
      best = r;
      bestDistance = nearest;
    }
  }
  return best;
}

}  // namespace

std::vector<BranchArchive> archiveOf(const Env& env, const std::vector<const Reading*>& readings) {
  std::map<UseConfig, std::vector<const Reading*>> byBranch;
  for (const Reading* r : readings) {
    if (succeeded(*r)) { byBranch[branchOf(env, r->fft, r->options)].push_back(r); }
  }

  std::vector<BranchArchive> out;
  for (auto& [branch, members] : byBranch) {
    std::ranges::sort(members, {}, &Reading::mean);
    BranchArchive a{.branch = branch, .seeds = {members.front()}, .configurations = members.size()};
    const Reading& fastest = *members.front();
    for (size_t i = 1; i < members.size(); ++i) {
      if (members[i]->mean - fastest.mean <= WITHIN_ERROR * std::hypot(members[i]->se, fastest.se)) {
        a.seeds.push_back(members[i]);
        break;
      }
    }
    while (a.seeds.size() < SEEDS_PER_BRANCH) {
      const Reading* const next = mostUnlike(members, a.seeds);
      if (!next) { break; }
      a.seeds.push_back(next);
    }
    out.push_back(std::move(a));
  }
  std::ranges::sort(out, {}, [](const BranchArchive& a) { return a.seeds.front()->mean; });
  return out;
}

size_t axesApart(const UseConfig& a, const UseConfig& b) {
  std::set<std::string> keys;
  for (const auto& [k, v] : a) { keys.insert(k); }
  for (const auto& [k, v] : b) { keys.insert(k); }

  size_t out = 0;
  for (const std::string& key : keys) {
    const Option* const option = findOption(key);
    if (option && option->compound) {
      u32 const pa = u32(useValue(a, key, 0));
      u32 const pb = u32(useValue(b, key, 0));
      for (const AccessClass& cls : accessClasses()) { out += getDigit(pa, cls.digit) != getDigit(pb, cls.digit); }
      continue;
    }
    auto const ia = a.find(key);
    auto const ib = b.find(key);
    out += ia == a.end() || ib == b.end() || ia->second != ib->second;
  }
  return out;
}

PositionCoverage positionCoverage(const ConditionalSpace& space, const std::vector<const Reading*>& readings) {
  const std::vector<BranchVariable>& vars = space.variables();
  std::set<std::tuple<size_t, size_t, size_t>> covered;
  for (const Reading* r : readings) {
    if (!succeeded(*r) || r->fft.spec() != space.fft().spec()) { continue; }
    std::vector<size_t> const states = space.statesOf(r->options);
    if (space.configOf(states) != r->options) { continue; }
    for (size_t v = 0; v < vars.size(); ++v) {
      if (states[v] != vars[v].inactive()) { covered.emplace(v, space.contextOf(v, states), states[v]); }
    }
  }

  PositionCoverage out{.covered = covered.size(), .total = 0};
  for (size_t v = 0; v < vars.size(); ++v) {
    for (size_t context = 0; context < vars[v].contexts(); ++context) {
      const BranchVariable::Context& c = vars[v].context(context);
      if (!c.reference || !space.legalContext(v, context)) { continue; }
      out.total += size_t(std::ranges::count(c.offered, true));
    }
  }
  return out;
}

CoverageSummary summarize(const Env& env, const std::vector<std::vector<Candidate>>& candidates,
                          const Readings& readings, const std::function<AxisFilter(const Candidate&)>& filter) {
  // A subject split into several candidates (one per interval) is one entry.
  std::vector<const Candidate*> entries;
  std::set<ConfigKey> seen;
  for (const std::vector<Candidate>& kind : candidates) {
    for (const Candidate& c : kind) {
      if (seen.insert(Readings::keyOf(c.subject, {})).second) { entries.push_back(&c); }
    }
  }

  // Building an entry's space is most of the cost, once per run: thousands of entries, across the machine's threads.
  struct Entry {
    size_t visited = 0;
    BranchCount branches{};
    PositionCoverage positions{};
  };
  std::vector<Entry> each(entries.size());
  parallelFor(entries.size(), [&](size_t i) {
    const Candidate& c = *entries[i];
    std::vector<const Reading*> const of = readings.of(c.subject);
    std::shared_ptr<const ConditionalSpace> const space = spaceOf(env, c.subject.fft, filter(c));
    each[i] = {.visited = archiveOf(env, of).size(),
               .branches = space->legalBranches(),
               .positions = positionCoverage(*space, of)};
  });

  CoverageSummary out{.entries = entries.size()};
  for (const Entry& e : each) {
    out.entriesRead += e.visited > 0;
    out.branchesVisited += e.visited;
    out.branches.legal += e.branches.legal;
    out.branches.complete = out.branches.complete && e.branches.complete;
    out.positions.covered += e.positions.covered;
    out.positions.total += e.positions.total;
  }
  return out;
}

std::string toString(const CoverageSummary& s) {
  double const share = s.positions.total ? 100.0 * double(s.positions.covered) / double(s.positions.total) : 0;
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), " (%.1f%%)", share);
  return "entries " + std::to_string(s.entriesRead) + " of " + std::to_string(s.entries) + " read; branches " +
    std::to_string(s.branchesVisited) + " of " + (s.branches.complete ? "" : "at least ") +
    std::to_string(s.branches.legal) + "; positions " + std::to_string(s.positions.covered) + " of " +
    std::to_string(s.positions.total) + buffer;
}

}  // namespace tune
