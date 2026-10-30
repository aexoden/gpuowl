// Copyright (C) Jason Lynch

// The tuning run: measure what the model says is worth measuring until the published choice is expected to be within
// `stop` of the best the card can do over the workload, publishing the tune files as they improve.  Under search=staged
// the staged search (Staged.h) chooses instead, with no model, until a budget or a stop ends the run.

#pragma once

#include "Acquire.h"
#include "Lanes.h"
#include "Measure.h"
#include "Soak.h"
#include "TuneCommand.h"
#include "Watch.h"

#include <ctime>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>

class GpuCommon;

namespace tune {

// What the loop measures through: the device, or for a test a simulation.  Both record what they measure in the
// database the loop reads.
class Device {
public:
  virtual ~Device() = default;

  // One recorded call of `options` on `candidate`, at its timing exponent, timing `blocks` blocks.
  virtual Call call(const Candidate& candidate, const UseConfig& options, u32 blocks) = 0;

  // One recorded rounding-error reading.
  virtual RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) = 0;

  // A recorded soak of the plan's configurations (Soak.h): what each sustained, NaN where it could not be read.  A
  // device that cannot soak reads nothing.
  virtual std::vector<double> soak(const SoakPlan& plan, const SoakSettings& settings) {
    (void)settings;
    return std::vector<double>(plan.targets.size(), std::numeric_limits<double>::quiet_NaN());
  }

  // Why the configuration is never built here; empty where it may be.
  [[nodiscard]] virtual std::string held(const Candidate& candidate, const UseConfig& options) const = 0;

  // Times the drift anchor where it is due.
  virtual void keepAnchor() {}

  // The device can measure nothing more, or a stop was asked for.
  [[nodiscard]] virtual bool stopped() const = 0;

  // The session its records are written under.
  [[nodiscard]] virtual u32 session() const = 0;

  // The time its records are stamped with, in seconds: what the lanes' time is measured by.
  [[nodiscard]] virtual u64 now() const { return u64(std::time(nullptr)); }
};

struct LoopSettings {
  double stop = STOP;
  double sweep = SWEEP;
  Timetable lanes{};
  ExploreSettings explore{};
  double budget = 0;  // tuning seconds for the campaign, every session counted (laneTime()); 0 for none
  double reserve = RESERVE;
  double target = TARGET;
  double newFloor = NEW_FLOOR;
  double shortCalls = SHORT_CALLS;  // 0 for none
  double powerHold = POWER_HOLD;    // Readings; 0 leaves every cost as timed
  u32 callBlocks = BLOCKS_PER_CALL;  // timed blocks in a call (the device's calls are shaped to match)
  SoakSettings soak{};               // soaks of the finalists (Soak.h)
  DuelSettings duel{};               // duels of a line's taker and its holder (Duel.h)
  Search search = Search::Model;     // what chooses the items no duty claims
  StagedSettings staged{};           // the staged search's (Staged.h)
  ContestSettings contest{};         // the report's contender table (Contenders.h)
  u64 seed = 1;
  TailQueues tailQueues = TailQueues::Ntt;
  u32 regretEvery = 32;  // calls between estimates of the regret
  u32 regretDraws = 5;
  u32 maxItems = 0;        // 0 for no limit; for tests
  fs::path publishTo{};    // where the tune files go; empty to write none
  std::string header{};    // the comment the tune files open with
  std::string settings{};  // the run's settings word, for the report beside the tune files
  Watch* watch = nullptr;
};

struct LoopResult {
  u32 calls = 0;
  u32 gateReadings = 0;
  double regret = std::numeric_limits<double>::infinity();
  double T = 0;
  bool converged = false;    // ended because the regret fell below `stop`
  bool budgetSpent = false;  // ended because the campaign's tuning time reached its budget
  double elapsed = 0;        // the campaign's tuning time at the end
};

// What the occupancy model offers each candidate's register limits.  A kernel's registers move with the options -- a
// structural branch, the access modes, hoisting -- so each key is offered the plan read nearest the candidate's best
// set (`regs` and `kres` records): in its branch where there is one, else the fewest axes from it.
[[nodiscard]] std::map<const Candidate*, RegisterOffer>
registerOffers(const TuneDB& db, u32 env, const Env& card, const std::vector<std::vector<Candidate>>& candidates,
               const Readings& readings);

// Runs the loop on `db`, whose env `env` the device measures.  The database is read again after every item.
LoopResult runLoop(Device& device, TuneDB& db, u32 env, const Env& card, const RunScope& scope,
                   const LoopSettings& settings);

enum class RunOutcome : u8 { Ok, Failed, DeviceLost };

// -tune: the run on the open device, in the current directory.
[[nodiscard]] RunOutcome runTune(GpuCommon shared, const TuneCommand& command);

// -tune status / emit / scope: no device.
[[nodiscard]] bool runOffline(const TuneCommand& command, const Args& args);

}  // namespace tune
