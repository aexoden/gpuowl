// Copyright (C) Jason Lynch

// The contender table: for each line production would run, the configurations that could take it and why none has.
//
// The workload's grid is cut, per kind, into the stretches one line serves -- and the stretches no line holds -- in
// exponent order.  A stretch's rivals are the other FFTs and regimes the run holds over any of its points, each by its
// cheapest unpublished set: one undercut by the lines, short of its calls or owing its accuracy reading.  A set that
// failed, is unsafe there or that a cheaper set of its own FFT outreaches never could take a line, so it is no rival.
// A rival is shown while its cost is within `within` of the line's, the nearest first, at most `top` of them, each
// with the ratio of its mean to the line's and that ratio's 95% interval from the two readings' standard errors, and
// how much of its FFT's space has been read: of its structural branches, and of its option positions.  The line's own
// FFT is shown the same way.  A function of the readings and the publication, so a status shows what a run would.
//
// `-tune explain,fft=<spec>[,KEY=VALUE...]` answers the narrower question, "why is this not used": every set read on
// the FFT (or the one named) in each regime, with what the publisher made of it.

#pragma once

#include "Coverage.h"
#include "Publish.h"

#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tune {

// How far dearer than a line a rival may read and still be shown, and how many are shown per line.
inline constexpr double WITHIN = 0.05;
inline constexpr size_t TOP = 5;

// -tune within=<p>%|all, top=<n>|all.
struct ContestSettings {
  double within = WITHIN;  // infinite for every rival
  size_t top = TOP;        // SIZE_MAX for every rival

  [[nodiscard]] bool operator==(const ContestSettings&) const = default;
};

// What has been read of one FFT's space: its structural branches with a successful reading out of the legal ones,
// and its option positions.
struct SpaceRead {
  size_t branches = 0;
  BranchCount legal{};
  PositionCoverage positions{};
};

struct Rival {
  Subject subject;
  StandingOf standing;
  double ratio = 1;  // its mean over the line's (over the cheapest rival's, where no line holds the stretch)
  double lo = 1;     // the ratio's 95% interval
  double hi = 1;
  bool owed = false;  // the run owes it the call or the accuracy reading it is short of
  SpaceRead read{};
};

struct Contest {
  const Line* line = nullptr;  // nullptr for a stretch no line holds
  u64 lo = 0;                  // the stretch's first and last grid points
  u64 hi = 0;
  double weight = 0;     // its share of the kind's grid weight
  SpaceRead lineRead{};  // of the line's FFT
  std::vector<Rival> rivals;
};

struct KindContest {
  TestKind kind = TestKind::PRP;
  std::vector<Contest> contests;  // in exponent order
};

// The table over the run's tuned kinds; `candidates` per kind, in the order of scope.grids, as `publication` was
// published from them.  `filter` limits each FFT's space as the run's searches do.
[[nodiscard]] std::vector<KindContest> contendersOf(const Env& env, const RunScope& scope,
                                                    const std::vector<std::vector<Candidate>>& candidates,
                                                    const Readings& readings, const Publication& publication,
                                                    const std::function<AxisFilter(const Candidate&)>& filter,
                                                    const ContestSettings& settings);

// The table as log lines, one per line and one per rival:
//   "LL 41851678-68141184, 62% of the weight: 510.6 512:7:512:212 short32 -use ...; branches 4 of 48, positions ..."
//   "    +0.4% [-0.3%, +1.1%] 512.8 512:14:256:212 short32 -use ...: undercut, 2 calls; branches 1 of 48, ..."
[[nodiscard]] std::vector<std::string> contestLines(const std::vector<KindContest>& table);

// Why a reading stands as it does, in a phrase: "undercut: at 67000000 production runs 512:7:512:212 at 510.6, 2.1%
// cheaper".  `owed`: a call or reading the run owes it.
[[nodiscard]] std::string whyText(const StandingOf& standing, bool owed);

// What to explain: an FFT, by its full spec or its shape alone (every variant of it), and optionally one option set.
struct ExplainQuery {
  std::string fft{};
  std::optional<UseConfig> options{};

  [[nodiscard]] bool operator==(const ExplainQuery&) const = default;
};

// Whether `spec` is the query's FFT: the same spec, or a variant of the shape it names.
[[nodiscard]] bool matchesFft(const ExplainQuery& query, const std::string& spec);

// The explanation as log lines: for each kind and regime the FFT was read or is held in, every set read (or the one
// named, canonicalised for `env`, and where it was never read, the nearest sets that were), the cheapest first, with
// its cost, calls and standing.
[[nodiscard]] std::vector<std::string> explainLines(const Env& env, const ExplainQuery& query, const RunScope& scope,
                                                    const std::vector<std::vector<Candidate>>& candidates,
                                                    const Readings& readings, const Publication& publication);

}  // namespace tune
