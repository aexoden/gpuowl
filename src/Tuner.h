// Copyright (C) Jason Lynch

// The tuning run: measure what the model says is worth measuring until the published choice is expected to be within
// `stop` of the best the card can do over the workload, publishing the tune files as they improve.

#pragma once

#include "Acquire.h"
#include "Lanes.h"
#include "Measure.h"
#include "TuneCommand.h"
#include "Watch.h"

#include <ctime>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

class GpuCommon;

namespace tune {

// What the loop measures through: the device, or for a test a simulation.  Both record what they measure in the
// database the loop reads.
class Device {
public:
  virtual ~Device() = default;

  // One recorded call of `options` on `candidate`, at its timing exponent.
  virtual Call call(const Candidate& candidate, const UseConfig& options) = 0;

  // One recorded rounding-error reading.
  virtual RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) = 0;

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
  double contenders = CONTENDERS;
  u64 seed = 1;
  TailQueues tailQueues = TailQueues::Ntt;
  u32 regretEvery = 32;  // calls between estimates of the regret
  u32 regretDraws = 5;
  u32 maxItems = 0;      // 0 for no limit; for tests
  fs::path publishTo{};  // where the tune files go; empty to write none
  std::string header{};  // the comment the tune files open with
  Watch* watch = nullptr;
};

struct LoopResult {
  u32 calls = 0;
  u32 gateReadings = 0;
  double regret = std::numeric_limits<double>::infinity();
  double T = 0;
  bool converged = false;  // ended because the regret fell below `stop`
};

// Runs the loop on `db`, whose env `env` the device measures.  The database is read again after every item.
LoopResult runLoop(Device& device, TuneDB& db, u32 env, const Env& card, const RunScope& scope,
                   const LoopSettings& settings);

enum class RunOutcome : u8 { Ok, Failed, DeviceLost };

// -tune: the run on the open device, in the current directory.
[[nodiscard]] RunOutcome runTune(GpuCommon shared, const TuneCommand& command);

// -tune status / emit / scope: no device.
[[nodiscard]] bool runOffline(const TuneCommand& command, const Args& args);

}  // namespace tune
