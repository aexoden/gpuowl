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

#include <optional>
#include <random>
#include <string>
#include <vector>

namespace tune {

// How many fresh calls each published line has in the end phase before the audit is declared.
inline constexpr u32 CONFIRM_CALLS = 4;

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

// What the campaign can say at its end, a line each: the audit's trials, what they found, and the state it ends in.
[[nodiscard]] std::vector<std::string> auditReport(const TuneDB& db, u32 env, const Readings& readings,
                                                   const Publication& publication, bool budgetSpent);

}  // namespace tune
