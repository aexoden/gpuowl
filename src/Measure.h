// Copyright (C) Jason Lynch

// Turning a (FFT, test kind, exponent, -use options) into a timing with an error bar, and into a verdict on the
// rounding error the configuration spends to get it.

#pragma once

#include "Anchor.h"
#include "FFTConfig.h"
#include "GpuCommon.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "timeutil.h"
#include "TuneDB.h"
#include "UseResolve.h"

#include <functional>
#include <string>
#include <vector>

struct IterSamples;  // Gpu.h

namespace tune {

// The Gumbel z a configuration's largest rounding error has to clear to be usable at all.
[[nodiscard]] double minSafeZ(enum FFT_TYPES type);

// One call to the GPU.
struct Call {
  Measurement measurement;

  // Every timed block, spikes included, in microseconds per iteration.
  std::vector<double> usPerIt;

  u32 dropped = 0;
  bool declined = false;

  u64 res64 = 0;
  bool checkOk = true;

  // The options the kernels were built with.
  UseConfig ran;

  // Wall-clock seconds, so that what an item costs can be told from what it measures: building the Gpu (a compile
  // included, on a cold kernel cache) against the timed blocks themselves.
  double buildSec = 0;
  double timedSec = 0;

  [[nodiscard]] CallSummary summary(double drift = 1) const {
    return {.mean = measurement.mean, .sd = measurement.stddev, .blocks = measurement.blocks, .drift = drift};
  }
};

// Spike rejection and the statistics over what survives.  Pure; a failed Gerbicz check makes the row an error.
[[nodiscard]] Call summarize(const IterSamples& samples);

// Builds a Gpu for the configuration and times it.  Exceptions from the build or the run propagate.
[[nodiscard]] Call timeCall(GpuCommon shared, const FFTConfig& fft, TestKind kind, u64 exponent,
                            const UseConfig& options, u32 nBlocks = BLOCKS_PER_CALL, u32 blockSize = 1000);

// Results from a rounding error check.
struct RoeCheck {
  // False for a pure NTT: GF31 and GF61 arithmetic is exact, so there is no rounding error to measure.
  bool applicable = false;

  double z = 0;  // the Gumbel z of the largest rounding error against 0.5
  u32 n = 0;     // rounding errors behind z
  double maxRoe = 0;
  double minZ = 0;
  u64 exponent = 0;  // where it was measured
  bool checkOk = true;

  // Records the reason for no reading when there is none.
  Status status = Status::Ok;

  [[nodiscard]] bool conclusive() const { return applicable && n > 2; }

  [[nodiscard]] bool passed() const {
    return status != Status::Ok || !applicable || (checkOk && (n <= 2 || z >= minZ));
  }
};

// Measures the rounding error of one configuration at `exponent`.
//
// Always on the PRP kernel set: only the PRP carry kernels have a ROE variant, so an LL Gpu collects no rounding errors
// at all.
[[nodiscard]] RoeCheck roeCheck(GpuCommon shared, const FFTConfig& fft, const UseConfig& options, u64 exponent);

struct Failure {
  Status status = Status::Ok;
  bool stop = false;   // a stop on request or a lost device
  bool fatal = false;  // stopped because the device is gone
  std::string what;
};

[[nodiscard]] Failure classify(std::string_view message);

class Session {
public:
  Session(GpuCommon shared, TuneDB& db, const Env& env);

  // `probe` is the exponent the session is measuring around. Where the env has no anchor pinned yet, the candidates
  // for one are raced there when the anchor is first due, and the winner is pinned for the env's life. Zero leaves the
  // session unanchored: its rows are recorded as measured.
  [[nodiscard]] bool begin(u64 probe = 0);
  void end();

  // What every row of this session is divided by to compare it with a row of another one.
  [[nodiscard]] double drift() const { return anchorState_.ratio; }

  // Whether the session has an anchor -- or a race for one -- that has not been timed yet, or not for
  // ANCHOR_EVERY_SEC.
  [[nodiscard]] bool anchorDue() const;

  // Settles a pending race: one recorded call of each candidate the env has no reading of yet, at the production block
  // size, and the cheapest reading becomes the anchor.  Those calls are baselines like any other, so the race costs
  // nothing a tuning run would not spend anyway.
  void raceAnchor();

  // Times the anchor if it is due, and updates the drift the rows carry.  A recorded call does this itself, so calling
  // it is only a way of choosing when.
  void keepAnchor();

  [[nodiscard]] const AnchorSpec& anchor() const { return anchor_; }

  [[nodiscard]] u32 id() const { return session_; }
  [[nodiscard]] u32 envId() const { return envId_; }

  // Returns the reason the configuration is never built again here.
  [[nodiscard]] std::string held(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options) const;

  // Runs one configuration and discards the reading.  The first Gpu of a process is slow everywhere it has been
  // measured, and a session that does not absorb that charges it to whichever configuration it happens to run first,
  // so the first call a session makes is preceded by SESSION_WARM_CALLS of these unless one was asked for already.
  Call warmUp(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
              u32 nBlocks = BLOCKS_PER_CALL, u32 blockSize = 1000);

  // Runs one configuration, returning its call result.
  [[nodiscard]] Call run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                         u32 nBlocks = BLOCKS_PER_CALL, u32 blockSize = 1000);

  // Checks the rounding error of one configuration at `exponent`.
  [[nodiscard]] RoeCheck checkRoe(const FFTConfig& fft, const UseConfig& options, u64 exponent);

  // Runs one configuration under an attempt, reporting whether it was attempted and completed.
  [[nodiscard]] bool underAttempt(const FFTConfig& fft, TestKind kind, u64 epxonent, const UseConfig& options,
                                  const char* during, const std::function<void()>& work);

  // Sets the keys that vary for this session.
  void varying(std::vector<std::string> keys) { varying_ = std::move(keys); }

  [[nodiscard]] bool stopped() const { return stopped_; }

  [[nodiscard]] bool cannotRecord() const { return cannotRecord_; }

  [[nodiscard]] bool deviceLost() const { return deviceLost_; }

private:
  [[nodiscard]] Call runCall(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                             u32 blockSize, bool record);

  // Whether the context can still do anything at all.
  [[nodiscard]] bool deviceUsable();

  // Stops the session on a lost device and says which configuration took it down.
  void lost(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, const std::string& what,
            const char* during);

  // Records `options` value for the known bad configuration.
  void noteNogo(const FFTConfig& fft, const UseConfig& options);

  // Stops anchoring this session, keeping whatever ratio the anchor last gave: a reading already taken against the
  // env's baseline is still the best thing its rows have.
  void unanchor() { anchor_ = {}; }

  // Handles an exception from a build or a run.
  [[nodiscard]] Status failed(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                              const std::string& message, const char* during);

  // Runs `work`, converting exceptions into a single result.
  [[nodiscard]] Status caught(const std::function<void()>& work, const FFTConfig& fft, TestKind kind, u64 exponent,
                              const UseConfig& options, const char* during);

  // Reports that the attempt could not be declared.
  void cannotDeclare(const FFTConfig& fft, const UseConfig& options);

  GpuCommon shared_;
  TuneDB& db_;
  Env env_;

  AnchorSpec anchor_{};
  AnchorState anchorState_{};

  // The candidates still to be raced for the env's anchor; empty once one is pinned.
  std::vector<AnchorSpec> race_;

  // The options the env's baseline reading was taken under; 0 when there is no baseline yet.
  u32 baselineCfg_ = 0;

  Timer sinceAnchor_{};
  bool inAnchor_ = false;

  u32 envId_ = 0;
  u32 session_ = 0;
  bool warmed_ = false;
  bool stopped_ = false;
  bool deviceLost_ = false;
  bool cannotRecord_ = false;
  std::vector<std::string> varying_;
};

enum class MeasureOutcome : u8 { Ok, Failed, DeviceLost };

// What -measure was asked for: an FFT spec and then comma-separated settings.
struct MeasureArgs {
  std::string fft;

  // A second, different configuration timed alternately with the first, whose readings stand in for the drift anchor.
  // Empty for no anchor, which leaves the readings as measured.
  std::string anchorFft;

  u64 exponent = 0;  // 0: the top of the FFT's range

  u32 calls = 8;
  u32 blocks = BLOCKS_PER_CALL;
  u32 blockSize = 0;  // 0: the production block size

  bool roe = true;
  bool drain = false;

  // Time the drift anchor and record the ratio on every row. Off when `anchorFft` names one instead: that anchor is
  // timed alternately with the configuration under test, and two corrections applied at once describe neither.
  bool drift = true;
};

// Pure.  Throws a message for anything it does not understand, so a mistyped setting is a usage error.
[[nodiscard]] MeasureArgs parseMeasureArgs(std::string_view text);

// The -measure subcommand: times one configuration repeatedly and reports whether the error bar a row declares
// describes how far independent readings of it actually move.  Also prints what one call spends on construction, what
// its rounding error is, and records every reading in the tuning database alongside the attempt it declared first.
[[nodiscard]] MeasureOutcome runMeasure(GpuCommon shared, const MeasureArgs& want);

}  // namespace tune
