// Copyright (C) Jason Lynch

// What the measurements support publishing, and what production would then run.
//
// The tune file holds everything known, whatever range a run tunes: an FFT outside it keeps the lines an earlier run
// found.  A line is one configuration -- FFT, test kind, carry regime and option set -- over the exponents it runs in
// that regime: no line ever stands for a configuration it was not measured as.  An FFT's lines in one regime are its
// option sets read at least MIN_CALLS times with no failure, whose accuracy is known, each reaching further than every
// cheaper one.  Production takes, at each exponent, the cheapest line that holds it, so that is what the objective T
// sums: per test kind, over the run's grid, the weight of each point times the cost of the line production would run
// there.

#pragma once

#include "Candidates.h"
#include "Gate.h"
#include "Readings.h"
#include "TuneEntry.h"

#include <deque>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tune {

struct Line {
  Subject subject;
  const Candidate* candidate = nullptr;  // where the run holds the subject
  const Reading* reading = nullptr;
  Verdict accuracy{};  // outside the run, an unsettled gate's highest safe reading
  Interval interval{};  // the exponents the line holds: one regime of the FFT, up to where the accuracy holds
  bool carried = false;  // an earlier build's reading, which this build has not settled yet

  [[nodiscard]] double cost() const { return reading->pessimistic(); }
  [[nodiscard]] u64 lo() const { return interval.lo; }
  [[nodiscard]] u64 reach() const { return interval.hi; }
  [[nodiscard]] bool holds(u64 E) const { return interval.contains(E); }
};

// A reading the gate owes before a line can be published: of `options` on `fft`, at `exponent`.
struct GateItem {
  FFTConfig fft;
  UseConfig options{};  // the set to read: `of`, or its accuracy reference (Gate.h) where that reading is owed
  UseConfig of{};       // the set whose gate the reading settles
  u64 exponent = 0;
  double weight = 0;  // of the grid points the line would serve, which orders the items
};

// A call owed before a line can be published: a reading that would be one, short of its calls, or a line an earlier
// build published that this build has to read for itself before anything replaces it.
struct ConcludeItem {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  bool carried = false;
};

struct KindPublication {
  TestKind kind = TestKind::PRP;
  bool tuned = false;         // the run tunes this kind: T and the owed items are of it
  std::vector<Line> lines;    // this build's frontier, cheapest first
  std::vector<Line> written;  // the frontier the tune file holds: those and the lines carried from earlier builds
  double T = 0;               // the weighted cost production would run, over the points some line holds
  double covered = 0;         // the weight of those points

  // The line production would run at `E`; nullptr where none holds it.
  [[nodiscard]] const Line* at(u64 E) const;
};

struct Publication {
  std::vector<KindPublication> kinds;  // the run's kinds, in the order of its grids, then any other kind read
  std::vector<GateItem> owedGates;      // heaviest first
  std::vector<ConcludeItem> owedCalls;  // heaviest first

  [[nodiscard]] const KindPublication* of(TestKind kind) const;

  // T summed over the kinds, and the share of the tuned kinds' weight it covers (1 when every point has a line).
  [[nodiscard]] double T() const;
  [[nodiscard]] double coverage() const;
};

// Why a configuration holds a line, or does not, as the publisher sees it.
enum class Standing : u8 {
  Published,     // it holds a line of the tune file
  Failed,        // a call computed wrongly, failed to build, or the device refused it
  ShortOfCalls,  // read fewer than MIN_CALLS times
  GateOwed,      // its accuracy reading is owed, at `at`
  Rejected,      // unsafe throughout the regime
  ShortReach,    // its reach, `at`, falls short of the regime's lowest exponent
  Outreached,    // `by`, a cheaper set of the same FFT, reaches as far
  Undercut,      // cheaper lines hold every exponent it reaches: at `at`, production runs `line`
  NoRegime,      // the FFT never runs in the regime it was read in
};

[[nodiscard]] const char* toString(Standing standing);

struct StandingOf {
  const Reading* reading = nullptr;
  Standing standing = Standing::Published;
  u64 at = 0;
  u64 reach = 0;                 // how far it would be published, where the gate has settled that; else 0
  const Reading* by = nullptr;   // Outreached
  const Line* line = nullptr;    // Published: its own line; Undercut: the line production runs at `at`
};

// Why each reading of `subject` holds a line of `pub`, or does not, the cheapest first.  `held`: the run's candidate
// of the subject, nullptr where the run does not hold it; `readings`: those `pub` was published from.
[[nodiscard]] std::vector<StandingOf> standingsOf(const KindPublication& pub, const Subject& subject,
                                                  const Candidate* held, const Readings& readings);

// The run's candidate of the subject's FFT and regime in `candidates`, the heaviest should there be two; nullptr where
// there is none.
[[nodiscard]] const Candidate* heldCandidate(const std::vector<Candidate>& candidates, const Subject& subject);

// The lines of earlier builds of the same card, under the same tag: each configuration's from the newest build that
// published it, as long as this build's option table still names its options as they were.  They are written until
// this build has read the configuration itself, but kept out of T, so an old build's costs never stand in the way of
// measuring this one's.  The lines refer to `readings`.
struct Carried {
  std::deque<Readings> readings;
  std::map<TestKind, std::vector<Line>> lines;

  [[nodiscard]] size_t size() const;
};

[[nodiscard]] Carried carriedLines(const TuneDB& db, u32 env);

// `candidates` per kind, in the order of scope.grids.
[[nodiscard]] Publication publish(const RunScope& scope, const std::vector<std::vector<Candidate>>& candidates,
                                  const Readings& readings, const Carried* carried = nullptr);

// The file production reads.  It opens with "# TUNE_FILE_MAGIC, build <fingerprint>" (TuneEntry.h): production knows
// the format by it, and warns where the build named is not its own.
inline constexpr const char* TUNE_FILE = "tune.txt";

// One line per configuration, every kind, the cheapest of each kind first:
//   "<cost> <spec> <kind> <regime> <lo>-<hi> # KEY=VAL, ..."
// under "# prpll tune v2, build <fingerprint>" and `header`, a comment line per line of it.
[[nodiscard]] std::string tuneFileText(const Publication& publication, const std::string& header);

// Writes it to `dir`, through a temporary and a rename.
void writeTuneFile(const fs::path& dir, const Publication& publication, const std::string& header);

}  // namespace tune
