// Copyright (C) Jason Lynch

// Turning a configuration -- (FFT, test kind, exponent, -use options) -- into a recorded reading, and into a rounding
// error reading, on the open device; and keeping a measuring session honest about drift, faults and wrong answers.

#pragma once

#include "Anchor.h"
#include "Args.h"
#include "Eligibility.h"
#include "FFTConfig.h"
#include "GpuCommon.h"
#include "KernelCompiler.h"
#include "Occupancy.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "Telemetry.h"
#include "TestKind.h"
#include "timeutil.h"
#include "TuneDB.h"

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct IterSamples;  // Gpu.h
class Gpu;           // Gpu.h

namespace tune {

class Heater;  // Measure.cpp

// A kernel a REG* key limits, as one call built it, with what the occupancy model offers its key on this device.
struct KernelUse {
  std::string key{};
  KernelResources resources{};
  RegisterPlan plan{};
};

// One call to the GPU.
struct Call {
  Status status = Status::Ok;
  CoreStats core{};             // over the blocks kept
  std::vector<double> usPerIt;  // every timed block, spikes included
  u64 res64 = 0;
  bool checkOk = true;
  u64 iters = 0;                     // behind res64, the untimed warm-up included
  UseConfig ran{};                   // the options the kernels were built with
  std::vector<KernelUse> kernels{};  // REG-limited kernels; empty where the device's limits are not modelled
  double buildSec = 0;               // building the Gpu, a compile included on a cold kernel cache
  std::vector<std::string> loaded{};  // the Gpu's kernels the call ran, by name (Gpu::loadedKernels)
  double timedSec = 0;
  std::chrono::steady_clock::time_point timedFrom{};  // the timed blocks (IterSamples)
  std::chrono::steady_clock::time_point timedTo{};
  std::optional<WindowPower> power{};  // what the card drew over them, where it says

  [[nodiscard]] bool ok() const { return status == Status::Ok && core.stats.n > 0; }
  [[nodiscard]] double mean() const { return core.stats.mean; }
};

// Spike rejection and the statistics over what survives.  A failed Gerbicz check makes the call an error.
[[nodiscard]] Call summarize(const IterSamples& samples);

// Untimed blocks a call runs before its timed ones.
inline constexpr u32 CALL_WARMUP_BLOCKS = 1;

// How often a session reads the card (Telemetry.h), and how many seconds of readings it keeps: more than a soak's
// longest slice.
inline constexpr double SAMPLE_PERIOD = 0.1;
inline constexpr double SAMPLE_KEEP = 300;

// One configuration of a soak (Soak.h).
struct SoakEntry {
  FFTConfig fft;
  TestKind kind = TestKind::PRP;
  UseConfig options{};
};

[[nodiscard]] constexpr u64 callIterations(u32 nBlocks, u32 blockSize) {
  return u64(CALL_WARMUP_BLOCKS + nBlocks) * blockSize;
}

// The tuner's copy of the shared args: a tuning reading is of the configuration named and nothing else, so every
// tunable -use key is set aside wherever it came from (config.txt, a "! <fft>" line, the command line), as is a
// pinned carry.  Keys that are never searched (NO_ASM, DEBUG, ...) stay in force.
struct Takeover {
  Args args;
  std::set<std::string> setAside{};
  bool carry = false;  // a pinned carry was set aside
};
[[nodiscard]] Takeover takeOver(const Args& shared);

// Builds a Gpu for the configuration and times it.  Exceptions from the build or the run propagate.  `keep`: where the
// Gpu goes once timed, rather than being let go.
[[nodiscard]] Call timeCall(GpuCommon shared, const Env& env, const FFTConfig& fft, TestKind kind, u64 exponent,
                            const UseConfig& options, u32 nBlocks, u32 blockSize, double heatSec = CALL_HEAT_SEC,
                            std::unique_ptr<Gpu>* keep = nullptr);

// The call to record for an LL reading checked against `reference`.  A fault can flip a bit of any one reading, so one
// that disagrees is read again: the second is recorded, and as an error only where it disagrees too.
[[nodiscard]] Call checkedAgainst(u64 reference, Call first, const std::function<Call()>& again);

// The same for a PRP call whose Gerbicz check failed.
[[nodiscard]] Call checkedTwice(Call first, const std::function<Call()>& again);

// A rounding-error reading, always on the PRP kernels: only they have a ROE build, and the LL carry rounds alike.
struct RoeCheck {
  bool applicable = false;  // false for exact arithmetic
  bool checkOk = true;
  double z = 0;  // the Gumbel z of the largest rounding error against 0.5
  u32 n = 0;     // rounding errors behind z
  double maxRoe = 0;
  u64 fingerprint = 0;  // of the rounding errors themselves (RoeInfo): equal only where the arithmetic is
  u32 levels = 0;       // distinct values among them, and how many are the largest (RoeInfo)
  u32 atMax = 0;
  u64 exponent = 0;
  UseConfig ran{};
  Status status = Status::Ok;  // why there is no reading, where there is none
};

[[nodiscard]] RoeCheck roeCheck(GpuCommon shared, const Env& env, const FFTConfig& fft, const UseConfig& options,
                                u64 exponent);

struct Failure {
  Status status = Status::Ok;
  bool stop = false;   // a stop on request or a lost device
  bool fatal = false;  // the device is gone
  std::string what{};
};

[[nodiscard]] Failure classify(std::string_view message);

// One measuring session on the open device.  Every call it makes is recorded with the attempt declared first, so that a
// process the device takes down leaves behind what it was holding, and nothing it held is built again.
class Bench {
public:
  Bench(GpuCommon shared, TuneDB& db, const Env& env);

  // Opens the session.  `probe` is the exponent the session measures around: where the env has no drift anchor pinned
  // yet, the candidates are raced there when it is first due, and the winner is pinned for the env's life.  `tag` is
  // part of the env (DbEnv::tag).
  [[nodiscard]] bool begin(u64 probe, const std::string& settings, std::string machine = {}, std::string tag = {});

  // Closes an attempt left open, unless the device was lost: then it is what the next process must not build again.
  void end();
  ~Bench();

  Bench(const Bench&) = delete;
  Bench& operator=(const Bench&) = delete;

  // What a reading of this session is divided by to compare it with one of another session.
  [[nodiscard]] double drift() const { return anchorState_.ratio; }

  // Times the anchor where it is due.  A recorded call does this itself; calling it chooses when.
  void keepAnchor();

  // One recorded call of the configuration, at the production block size: `nBlocks` timed blocks, or for 0 the
  // session's (shapeCalls()).
  Call call(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks = 0);

  // The seconds of heat before a call's timed blocks and how many there are, for every call() after; a run's heat= and
  // blocks= settings.  The anchor's calls keep CALL_HEAT_SEC and BLOCKS_PER_CALL, so they compare across sessions.
  // `heater`: whether the last call's Gpu goes on squaring while the session is idle (heater=; see Heater).
  // `ahead`: whether a call's kernels compile before its Gpu is built (compileAhead=; see compileAhead).
  void shapeCalls(double heatSec, u32 blocks, bool heater = true, bool ahead = true) {
    heatSec_ = heatSec;
    blocks_ = blocks;
    heaterOn_ = heater;
    aheadOn_ = ahead;
  }

  // One recorded rounding-error reading.
  RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent);

  // A soak of `entries` at `exponent`, every slice recorded (SoakRow): all of them built and held at once, `warm`
  // seconds of the first that builds to bring the card to its sustained state, then `rounds` rounds of `slice` seconds
  // of each in turn, so the card never idles between them.  What each sustained over its rounds; NaN where it could
  // not be built or read.
  std::vector<double> soak(const std::vector<SoakEntry>& entries, u64 exponent, double warm, double slice,
                           u32 rounds);

  // Why the configuration is never built again here; empty where it may be.
  [[nodiscard]] std::string held(const FFTConfig& fft, const UseConfig& options) const;

  [[nodiscard]] u32 envId() const { return envId_; }
  [[nodiscard]] u32 session() const { return session_; }
  [[nodiscard]] const Env& env() const { return env_; }
  [[nodiscard]] u32 blockSize() const { return blockSize_; }

  [[nodiscard]] bool stopped() const { return stopped_; }
  [[nodiscard]] bool deviceLost() const { return deviceLost_; }
  [[nodiscard]] bool cannotRecord() const { return cannotRecord_; }

private:
  [[nodiscard]] Call runCall(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                             u32 blockSize, bool record, double heatSec = CALL_HEAT_SEC);

  // One call under a declared attempt, recording nothing else.
  [[nodiscard]] Call attempt(GpuCommon shared, const FFTConfig& fft, TestKind kind, u64 exponent,
                             const UseConfig& options, u32 nBlocks, u32 blockSize, double heatSec = CALL_HEAT_SEC);

  void raceAnchor();
  [[nodiscard]] bool anchorDue() const;
  void unanchor() { anchor_ = {}; }

  // The residue an LL call must reproduce, settled by a vote of other FFTs' built-in defaults where the env has none
  // agreed yet; empty where none can be agreed.
  [[nodiscard]] std::optional<u64> llReference(const FFTConfig& fft, u64 exponent, u32 nBlocks, u32 blockSize);

  void recordRegisters(const FFTConfig& fft, TestKind kind, const Call& call, u32 cfg);

  // Lets the card cool: stops the heater and lets its Gpu go, before anything else is built.  A heater that failed for
  // any reason but a stop leaves the device asked whether it is still usable, so a loss is not pinned on what is built
  // next.
  void coolDown();

  // Compiles the call's kernels ahead (Gpu::compileAhead) before the heater stops, so the card goes on squaring through the
  // compile: those that calls of its kind and FFT type have run so far this session.  A kind's first call compiles on its run.
  void compileAhead(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options);

  [[nodiscard]] bool deviceUsable();
  void lost(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, const std::string& what,
            const char* during);
  [[nodiscard]] Status failed(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                              const std::string& message, const char* during);
  [[nodiscard]] Status caught(const std::function<void()>& work, const FFTConfig& fft, TestKind kind, u64 exponent,
                              const UseConfig& options, const char* during);
  void cannotDeclare(const FFTConfig& fft, const UseConfig& options);

  GpuCommon shared_;
  Args args_;  // the take-over of the shared args, which every Gpu here is built from
  TuneDB& db_;
  Env env_;
  u32 blockSize_ = 1000;
  double heatSec_ = CALL_HEAT_SEC;
  u32 blocks_ = BLOCKS_PER_CALL;
  bool heaterOn_ = true;
  std::unique_ptr<Heater> heater_;
  bool aheadOn_ = true;
  // The kernels calls have run, by kind and FFT type (kernelsRunBy), for compileAhead
  std::map<std::string, std::set<std::string>> kernelsRun_;
  u32 aheadCalls_ = 0;   // calls compileAhead compiled for, and the kernels it compiled for them
  u32 aheadKernels_ = 0;
  KernelCompiler::Tally tallyAtBegin_{};

  // The card's own readings, for what each call drew; none where the card cannot be read.
  std::unique_ptr<Telemetry> telemetry_;
  std::unique_ptr<Sampler> sampler_;

  AnchorSpec anchor_{};
  AnchorState anchorState_{};
  std::vector<AnchorSpec> race_;
  u32 baselineCfg_ = 0;  // the options the env's anchor baseline was taken under; 0 where there is none yet
  Timer sinceAnchor_{};
  bool inAnchor_ = false;

  u32 envId_ = 0;
  u32 session_ = 0;
  bool warmed_ = false;
  bool stopped_ = false;
  bool deviceLost_ = false;
  std::string lastFailure_;  // what the latest failure said (failed()), for the failure log
  [[nodiscard]] std::string deviceName() const;  // the card, as its env names it
  bool cannotRecord_ = false;

  std::set<std::pair<u64, u64>> unreferenced_;  // (exponent, iterations) where no LL reference could be agreed
};

}  // namespace tune
