// Copyright (C) Jason Lynch

// Classic mode (-tune mode=classic): upstream's own tuning procedure (Tune::tune in tune.cpp) -- one option at a time
// over a few representative FFTs, then every favoured shape over the workload, then the register limits of the entries
// that made the cost/reach frontier -- with this tuner's measurement, test kinds, faults, database, validation and
// emission. The stages, candidates and order are upstream's; whether a candidate is kept is decided by error-barred
// readings, with upstream's thresholds as settings.
//
// The procedure is a function of the database: every decision is made from readings, and a reading is taken only where
// the database lacks it, so a run that stops, faults or restarts replays what it has and carries on, and status and
// emit replay it without the device.  Each test kind runs the whole procedure on its own readings.

#pragma once

#include "ClassicSettings.h"
#include "common.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Probe.h"
#include "Stats.h"
#include "TuneDB.h"
#include "Tuner.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// What a classic run measures with: the device's side, or a replay that answers from the rows alone and stops at the
// first reading the database lacks.
class ClassicBench {
public:
  virtual ~ClassicBench() = default;

  // The reading of `options` on `fft` at `exponent` with at least `calls` calls, taking those it lacks; empty where
  // none can be had: the build failed or computed wrongly, the host built something else, or the run stopped.
  [[nodiscard]] virtual std::optional<Measurement> read(const FFTConfig& fft, TestKind kind, u64 exponent,
                                                        const UseConfig& options, u32 calls) = 0;

  // One accuracy reading; false where none was taken.
  [[nodiscard]] virtual bool gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) = 0;

  // Records that the option sweeps were taken on `fft` at `probe`, as a bootstrap family.
  virtual void declareBootstrap(const FFTConfig& fft, u64 probe) = 0;

  // On request, because the device can do nothing more, or, replaying, at the first reading the rows lack.
  [[nodiscard]] virtual bool stopped() const = 0;

  // Says what the procedure is doing, for the log; a replay says nothing.
  virtual void say(const std::string& text) = 0;
};

// Where a classic run stands.
struct ClassicOutcome {
  // The lines each kind's option sweeps settled on, as published beside the entries: the first kind's as the global
  // line and every other kind's as a line of its own.
  Defaults lines{};

  // Whether every stage of every kind has been run to its end.
  bool complete = false;

  // The stage the run is in or stopped in, as the log names it.
  std::string stage{};
};

struct ClassicContext {
  const Env& device;
  const TuneDB& db;
  u32 env = 0;
  const RunScope& scope;
  std::vector<TestKind> kinds;
  ClassicSettings settings{};
  TailQueues tailQueues = TailQueues::Ntt;
};

// Runs, or replays, the classic procedure for every kind.
[[nodiscard]] ClassicOutcome runClassic(const ClassicContext& context, ClassicBench& bench);

class ReadingIndex;

// A replay over the rows alone: what `runClassic` would do next is where it stops.
class ReplayBench final : public ClassicBench {
public:
  ReplayBench(const TuneDB& db, u32 env, const Env& device);
  ~ReplayBench() override;

  [[nodiscard]] std::optional<Measurement> read(const FFTConfig& fft, TestKind kind, u64 exponent,
                                                const UseConfig& options, u32 calls) override;
  [[nodiscard]] bool gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) override;
  void declareBootstrap(const FFTConfig&, u64) override {}
  [[nodiscard]] bool stopped() const override { return !next_.empty(); }
  void say(const std::string&) override {}

  // The first reading the rows lacked, as the log would name it; empty where none was.
  [[nodiscard]] const std::string& next() const { return next_; }

private:
  const Env& device_;
  std::unique_ptr<ReadingIndex> index_;
  std::string next_;
};

// The device's side: each reading a call at a time on `bench`, which records its rows, the drift anchor kept up as
// the run's queue keeps it.
class Bench;
class SessionClassicBench final : public ClassicBench {
public:
  SessionClassicBench(Bench& bench, const TuneDB& db, u32 env, const Env& device);
  ~SessionClassicBench() override;

  [[nodiscard]] std::optional<Measurement> read(const FFTConfig& fft, TestKind kind, u64 exponent,
                                                const UseConfig& options, u32 calls) override;
  [[nodiscard]] bool gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) override;
  void declareBootstrap(const FFTConfig& fft, u64 probe) override;
  [[nodiscard]] bool stopped() const override;
  void say(const std::string& text) override;

  // Calls made, and the seconds they took.
  [[nodiscard]] u32 calls() const { return calls_; }
  [[nodiscard]] double seconds() const { return seconds_; }

private:
  Bench& bench_;
  const TuneDB& db_;
  u32 env_;
  const Env& device_;
  std::unique_ptr<ReadingIndex> index_;
  std::set<std::string> failed_;
  u32 calls_ = 0;
  double seconds_ = 0;
};

// The reading of `options` on `fft` at `exponent` in `env`, whatever the configuration was spelled as when it was
// recorded; empty where there is none.
[[nodiscard]] std::optional<Measurement> classicReading(const TuneDB& db, u32 env, const Env& device,
                                                        const FFTConfig& fft, TestKind kind, u64 exponent,
                                                        const UseConfig& options);

}  // namespace tune
