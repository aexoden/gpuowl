// Copyright (C) Jason Lynch

// What the readings cover of each entry's legal configurations, and the measured configurations a search may start
// from.  A function of the readings alone, so it is the same after a restart.
//
//   - An entry's branches: every structural branch (branchOf()) it has a successful reading in, each with a few seeds
//     -- the fastest set, one within error of it, and those most unlike them -- however far behind the entry's best
//     it is, since a branch whose first readings lose may still win once its own options are searched.
//   - An entry's positions: each axis position in each context of what its option depends on (Branches.h), out of all
//     the legal ones.  What was read on another FFT is no reading of this one, however much the model learns from it.

#pragma once

#include "Branches.h"
#include "Candidates.h"
#include "Readings.h"

#include <functional>
#include <vector>

namespace tune {

// The most seeds a branch keeps, and how many standard errors apart two costs may be and still be within error.
inline constexpr size_t SEEDS_PER_BRANCH = 4;
inline constexpr double WITHIN_ERROR = 2;

struct BranchArchive {
  UseConfig branch{};                   // branchOf()
  std::vector<const Reading*> seeds{};  // the fastest first
  size_t configurations = 0;            // distinct sets read successfully in it
};

// The branches of one subject's readings, the fastest first.
[[nodiscard]] std::vector<BranchArchive> archiveOf(const Env& env, const std::vector<const Reading*>& readings);

// How many axes two canonical sets put at different positions, an access class of LOADS or STORES counting as one.
[[nodiscard]] size_t axesApart(const UseConfig& a, const UseConfig& b);

// Of the legal (axis, context, position) cells of one entry, how many its readings have been at.  A reading of another
// FFT, or of a set the space cannot state -- a register limit outside the present offer -- covers nothing.
struct PositionCoverage {
  size_t covered = 0;
  size_t total = 0;
};
[[nodiscard]] PositionCoverage positionCoverage(const ConditionalSpace& space,
                                                const std::vector<const Reading*>& readings);

// The same summed over a run's candidates, with their entries and branches.
struct CoverageSummary {
  size_t entries = 0;
  size_t entriesRead = 0;  // with a successful reading
  size_t branchesVisited = 0;
  BranchCount branches{};  // legal, summed; incomplete where any entry's count is
  PositionCoverage positions{};
};
[[nodiscard]] CoverageSummary summarize(const Env& env, const std::vector<std::vector<Candidate>>& candidates,
                                        const Readings& readings,
                                        const std::function<AxisFilter(const Candidate&)>& filter);

// "entries 23 of 140 read; branches 41 of 5321; positions 812 of 51234 (1.6%)"
[[nodiscard]] std::string toString(const CoverageSummary& summary);

}  // namespace tune
