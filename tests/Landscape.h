// Copyright (C) Jason Lynch

// A pretend device whose cost landscape is known, for judging how the tuner spends its time: every configuration costs
// what the landscape says, a call takes what a call of that cost and a first build would, and a run is cut off at a
// budget of pretend seconds.  Since the best configuration of each entry is planted, what the run published can be
// compared with the best it could have published, at every moment of the run.

#pragma once

#include "Scheduler.h"
#include "TuneDB.h"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace landscape {

struct Landscape {
  // What `options` costs on `fft`, in microseconds per iteration.
  std::function<double(const FFTConfig&, tune::TestKind, const tune::UseConfig&)> cost;

  // Whether a build of `options` on `fft` fails; none does by default.
  std::function<bool(const FFTConfig&, const tune::UseConfig&)> fails = [](const FFTConfig&, const tune::UseConfig&) {
    return false;
  };

  // The least any entry of `fft` in `kind` can cost: what the landscape was built to have as its optimum.
  std::function<double(const FFTConfig&, tune::TestKind)> best;
};

// The tuner's settings: as a run takes them by default, but for what a test names.
struct Policy {
  tune::Strategy strategy{};
  tune::Halving halving{.contenders = tune::CONTENDERS, .roundCalls = tune::ROUND_CALLS};
  bool bootstrap = true;
  bool restarts = true;
  bool gate = true;
};

struct Scenario {
  Landscape landscape;
  tune::RunScope scope;

  // The FFTs the tuner may choose between, as specs; each is an entry in every band of it the workload weighs.
  std::vector<std::string> ffts;

  // Pretend seconds the run is given.
  double budget = 3600;
};

// How far what was published is from the best the landscape allows, at one moment of a run.
struct Moment {
  double seconds = 0;
  u32 items = 0;

  // The weighted mean, over the workload, of how much dearer what is published is than the best it could be: 0 once
  // the best is published everywhere.  Nothing while any weighted point has nothing measured published for it.
  std::optional<double> regret;
};

struct Outcome {
  std::vector<Moment> trace;
  double seconds = 0;
  u32 items = 0;

  // Calls by entry label, and the configuration each entry publishes at the end.
  std::map<std::string, u32> calls;
  std::map<std::string, std::string> published;

  // Each round of the halving the run recorded, as "<round> <calls>: <spec> <spec> ...", and the FFT each bootstrap
  // searched.
  std::vector<std::string> rounds;
  std::vector<std::string> bootstrapped;

  [[nodiscard]] std::optional<double> regret() const { return trace.empty() ? std::nullopt : trace.back().regret; }

  // The pretend seconds at which the regret fell to `within` and stayed there, if it did.  What is published is
  // priced pessimistically, a little above what it was measured at, so no run's regret reaches 0; every planted gain
  // is several times the default.
  [[nodiscard]] std::optional<double> settled(double within = 0.005) const;
};

// Runs `scenario` under `policy` on a fresh database, or on `db` where one is given, until the budget is spent or
// nothing is left to do.
[[nodiscard]] Outcome simulate(const Scenario& scenario, const Policy& policy = {}, tune::TuneDB* db = nullptr);

// The env every scenario is run on: an nVidia card, which is what the option table offers the most on.
[[nodiscard]] tune::Env device();

}  // namespace landscape
