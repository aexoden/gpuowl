// Copyright (C) Jason Lynch

// What -tune was asked to do: one comma-separated word, as -measure's settings are, since the argument parser gives an
// option a single token.  A leading verb picks a subcommand; anything else, or nothing, is a tuning run.

#pragma once

#include "Axes.h"
#include "Lanes.h"
#include "Scope.h"

#include <string>
#include <string_view>

namespace tune {

// A run stops once the posterior expects the published choice to be within this fraction of the best the card could
// do over the workload.  Measured against what is left to gain, not against a clock.
inline constexpr double STOP = 0.001;

// The share of items given to a step new to the type, the shape or the FFT (untriedStep()), whatever the draws say.
inline constexpr double SWEEP = 0.2;

enum class TuneVerb : u8 {
  Run,     // tune on the device
  Status,  // where the database stands, with no device
  Emit,    // publish the tune files the database supports, with no device
  Scope,   // the range, the probe and the grid a run would work within, with no device
};

[[nodiscard]] const char* toString(TuneVerb verb);

struct TuneCommand {
  TuneVerb verb = TuneVerb::Run;
  ScopeArgs scope{};
  TailQueues tailQueues = TailQueues::Ntt;
  double stop = STOP;  // 0 to run until stopped
  double sweep = SWEEP;
  Timetable lanes{};
  bool dashboard = false;  // display only: not part of the settings a run records
  u64 seed = 0;            // the draws' seed; 0 for one from the clock
};

// Throws a message for anything it cannot read, naming -oldtune for an option word of upstream's own tuner.
[[nodiscard]] TuneCommand parseTuneCommand(std::string_view text);

// A run's settings as one word a run would take, its scope resolved, so that a later status values things as the run
// did whatever the worktodo says by then.
[[nodiscard]] std::string settingsWord(const RunScope& scope, const TuneCommand& command);

}  // namespace tune
