// Copyright (C) Jason Lynch

// The end of a budgeted campaign: what is published is confirmed by fresh calls, then an audit runs independent
// trials -- restarts drawn from the run's generator, each exactly RESTART_PACKET calls, none reading another -- whose
// number is declared once from the time left and never changed.  What it can establish is narrow: with no trial
// improving the workload by more than the target, the chance that one trial of this generator finds such an
// improvement is below 1 - 0.05^(1/n) (95%).  That is no bound on what the search has missed, nor a claim of an
// optimum: a narrow enough basin is missed by any number of trials.

#pragma once

#include "Explore.h"
#include "Publish.h"
#include "TuneDB.h"

#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace tune {

// How many fresh calls each published line has in the end phase before the audit is declared, and each line's closest
// rival: a set of another candidate, holding one of the line's points, measured within twice the target of it.
inline constexpr u32 CONFIRM_CALLS = 4;
inline constexpr u32 RIVAL_CALLS = 2;

// What the end phase reads afresh: every published line, and each line's closest rival.
struct Finalist {
  const Candidate* candidate = nullptr;
  UseConfig options{};
  bool rival = false;
};
[[nodiscard]] std::vector<Finalist> finalists(const AcquireContext& context, const Publication& publication,
                                              double target);

// A finalist's fresh calls: what the end phase measured of it, apart from the exploration's own calls.
struct Fresh {
  u32 calls = 0;
  double mean = 0;  // each call costed as `readings` cost calls; as timed where no readings were given
};
using FreshCalls = std::map<std::tuple<std::string, TestKind, u32>, Fresh>;  // by FFT, kind and cfg
// With `readings`, a fresh call costs what they would make of it -- the power floor, the soaks' calibration -- so it
// compares with what they pool; without, only the counts are meant.
[[nodiscard]] FreshCalls freshCalls(const TuneDB& db, u32 env, const Readings* readings = nullptr);
[[nodiscard]] Fresh freshOf(const FreshCalls& fresh, const TuneDB& db, const std::string& fft, TestKind kind,
                            const UseConfig& options);

// Where the env's end phase stands, from its records.
struct AuditState {
  std::optional<AuditRow> declared{};  // the latest audit
  u32 started = 0;                     // its trials begun
  u32 done = 0;                        // its trials finished: all their steps, or a start that could not be read
  u32 failedStarts = 0;
};
[[nodiscard]] AuditState auditState(const TuneDB& db, u32 env, const Readings& readings);

// The next item of the end phase: a fresh call of the least confirmed published line, or once each has CONFIRM_CALLS
// the audit's next trial step, declaring the audit first (its record written to `db`).  Nothing once the audit is
// done.  `budget` and `elapsed` are the campaign's tuning seconds.
[[nodiscard]] std::optional<Proposal> endStep(const AcquireContext& context, const Publication& publication, TuneDB& db,
                                              u32 env, u32 sess, u64 now, const ExploreSettings& settings,
                                              double elapsed, double budget, double target, std::mt19937_64& rng);

// What the campaign can say of its audit.
struct AuditSummary {
  enum class State : u8 { None, Running, FoundGains, Exhausted, NoTime, Complete };
  State state = State::None;
  bool declared = false;
  u32 trials = 0;
  u32 done = 0;
  u32 failedStarts = 0;
  double baseline = 0;  // T when declared
  double target = 0;
  double now = 0;                    // T now
  double gain = 0;                   // 1 - now / baseline
  double bound = 0;                  // where complete: the chance one trial finds the target, at 95%
  std::vector<const Line*> found{};  // published lines an audit trial read first

  // Each published line's fresh calls beside its pooled mean, to show what choosing the fastest of many cost.
  struct Confirmed {
    const Line* line = nullptr;
    Fresh fresh{};
  };
  std::vector<Confirmed> confirmed{};
};
[[nodiscard]] const char* toString(AuditSummary::State state);
[[nodiscard]] AuditSummary auditSummary(const TuneDB& db, u32 env, const Readings& readings,
                                        const Publication& publication, bool budgetSpent);

// The same as the log gives it, a line each.
[[nodiscard]] std::vector<std::string> auditReport(const AuditSummary& summary, bool budgetSpent);

}  // namespace tune
