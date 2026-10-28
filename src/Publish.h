// Copyright (C) Jason Lynch

// What the measurements support publishing, and what production would then run.
//
// The tune files hold everything known, whatever range a run tunes: an FFT outside it keeps the lines an earlier run
// found.  An FFT's lines are its option sets read at least MIN_CALLS times with no failure, in one carry regime, whose
// accuracy is known, each reaching further than every cheaper one.  Production takes, at each exponent, the cheapest
// line that holds it, so that is what the objective T sums: per test kind, over the run's grid, the weight of each
// point times the cost of the line production would run there.

#pragma once

#include "Candidates.h"
#include "Gate.h"
#include "Readings.h"

#include <deque>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace tune {

struct Line {
  Subject subject;
  const Candidate* candidate = nullptr;  // where the run holds the FFT
  const Reading* reading = nullptr;
  Verdict accuracy{};  // its reach is the line's: outside the run, an unsettled gate's highest safe reading

  [[nodiscard]] double cost() const { return reading->pessimistic(); }
  [[nodiscard]] u64 reach() const { return accuracy.reach; }
  [[nodiscard]] bool holds(u64 E) const;
};

// A reading the gate owes before a line can be published: of `options` on `fft`, at `exponent`.
struct GateItem {
  FFTConfig fft;
  UseConfig options{};
  u64 exponent = 0;
  double weight = 0;  // of the grid points the line would serve, which orders the items
};

// A call owed before a line can be published: a reading that would be one, short of its calls.
struct ConcludeItem {
  const Candidate* candidate = nullptr;
  UseConfig options{};
};

struct KindPublication {
  TestKind kind = TestKind::PRP;
  std::vector<Line> lines;    // this build's frontier, cheapest first
  std::vector<Line> written;  // the frontier the tune file holds: those and the lines carried from earlier builds
  double T = 0;               // the weighted cost production would run, over the points some line holds
  double covered = 0;         // the weight of those points

  // The line production would run at `E`; nullptr where none holds it.
  [[nodiscard]] const Line* at(u64 E) const;
};

struct Publication {
  std::vector<KindPublication> kinds;
  std::vector<GateItem> owedGates;      // heaviest first
  std::vector<ConcludeItem> owedCalls;  // heaviest first

  [[nodiscard]] const KindPublication* of(TestKind kind) const;

  // T summed over the kinds, and the share of their weight it covers (1 when every point has a line).
  [[nodiscard]] double T() const;
  [[nodiscard]] double coverage() const;
};

// The lines of earlier builds of the same card, for the FFTs this build has none of yet: each FFT's from the newest
// build with a line for it.  They are written, but kept out of T, so an old build's costs never stand in the way of
// measuring this one's.  The lines refer to `readings`.
struct Carried {
  std::deque<Readings> readings;
  std::map<TestKind, std::vector<Line>> lines;

  [[nodiscard]] size_t size() const;
};

[[nodiscard]] Carried carriedLines(const TuneDB& db, u32 env, std::span<const Grid> grids);

// `candidates` per kind, in the order of scope.grids.
[[nodiscard]] Publication publish(const RunScope& scope, const std::vector<std::vector<Candidate>>& candidates,
                                  const Readings& readings, const Carried* carried = nullptr);

// The tune file production reads for `kind`: tune.txt for PRP, tune-ll.txt for LL.
[[nodiscard]] const char* tuneFileName(TestKind kind);

// Upstream's tune.txt lines: "<cost> <spec> # <reach>, KEY=VAL, ...", the written frontier cheapest first.
[[nodiscard]] std::string tuneFileText(const KindPublication& kind, const std::string& header);

// Writes it to `dir`, through a temporary and a rename.
void writeTuneFile(const fs::path& dir, const KindPublication& kind, const std::string& header);

}  // namespace tune
