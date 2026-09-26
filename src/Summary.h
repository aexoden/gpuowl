// Copyright (C) Jason Lynch

// What a tuning run did and did not do, in the queue's own numbers: what it ran, what it left and what that was worth,
// what gain would have justified the entries it never measured, what the gains it saw teach, and how far the device
// drifted under it.  "Not explored" is then a number rather than a silence.
//
// A pure function of the database, the queue and the report the queue left, so that every figure is one the scheduler
// itself computed, or recomputes from the same inputs.

#pragma once

#include "Anchor.h"
#include "common.h"
#include "FFTConfig.h"
#include "Scheduler.h"
#include "TuneDB.h"
#include "Value.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tune {

struct RunSummary {
  QueueEnd end = QueueEnd::Dry;

  // The objective items were valued against at the end, the stop fraction, and so what an item had to be worth to run.
  double T = 0;
  double stop = 0;
  double floor = 0;

  u32 items = 0;
  u32 anchors = 0;
  std::map<ItemKind, QueueReport::Spent> spent;

  // One FFT type's entries: how many there are, how many have been measured, how many the queue still offers a first
  // measurement of, and how many it would offer one of but for what runs ahead of them by rule (`heldBy`) -- the rest
  // were ruled out, or given up on.  Of those it offers, the one the least gain would have justified, that gain (none
  // where no gain could), and how likely the gains learnt say a gain that large is.  Those waiting are not scored: the
  // queue has not ranked them, and what they would be worth waits on the lines the bootstrap is still deciding.
  struct Family {
    enum FFT_TYPES type = FFT64;
    u32 entries = 0;
    u32 measured = 0;
    u32 unmeasured = 0;
    u32 waiting = 0;

    std::string closest;
    double closestValue = 0;
    std::optional<double> gain;
    double chance = 0;
  };
  std::vector<Family> families;

  // What the waiting entries wait on: "the bootstrap" or "the accuracy gate"; empty where none wait.
  std::string heldBy;

  // What the queue still offers of one kind, and the one of them it values most.
  struct Remaining {
    ItemKind kind = ItemKind::Baseline;
    bool byRule = false;
    u32 count = 0;
    std::string best;
    double value = 0;

    // Past those, at most how many more points the stages listed in part have.
    u64 unlisted = 0;
  };
  std::vector<Remaining> remaining;

  // The share of a kind's weight on exponents no measured entry is published for, where the prior still stands in,
  // and the span of those exponents.
  struct Uncovered {
    TestKind kind = TestKind::PRP;
    double weight = 0;
    u64 lo = 0;
    u64 hi = 0;
  };
  std::vector<Uncovered> uncovered;
  double unservable = 0;

  double moves = 0;
  GainDist moveGains{};
  double combos = 0;
  GainDist comboGains{};

  // The session's anchor readings, as ratios of the env's first: the first and last of them, and the furthest either
  // way.
  struct Drift {
    std::string anchor;
    u32 readings = 0;
    double first = 1;
    double last = 1;
    double lo = 1;
    double hi = 1;
    DriftLevel level = DriftLevel::Steady;
    bool alarmed = false;
  };
  Drift drift;
};

// What an item is, for the log: its entry and what it does there.
[[nodiscard]] std::string itemLabel(const Scheduler& scheduler, const Item& item);

// "<n> of <m> entries measured", and those not, those waiting on `heldBy` and those ruled out, where there are any.
[[nodiscard]] std::string familyCounts(const RunSummary::Family& family, const std::string& heldBy);

// "<n> <kind> (<m> min)" for each kind of item run, comma-separated.
[[nodiscard]] std::string spentText(const std::map<ItemKind, QueueReport::Spent>& spent);

// Where `scheduler` left `env` in `report`, with `sess` the session the run was.
[[nodiscard]] RunSummary summarize(const Scheduler& scheduler, const TuneDB& db, u32 env, u32 sess,
                                   const QueueReport& report, double stop);

void logSummary(const RunSummary& summary);

}  // namespace tune
