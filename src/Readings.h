// Copyright (C) Jason Lynch

// What one env's database says, as the tuner reads it: every configuration that was called, its readings pooled, and
// every rounding-error reading.  Pure: a function of the database, rebuilt whenever it has grown.

#pragma once

#include "Eligibility.h"
#include "Model.h"
#include "Telemetry.h"
#include "TuneDB.h"

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tune {

// Which configuration a reading is of: canonical FFT spec, kind, carry regime and option set.
struct ConfigKey {
  std::string fft;
  TestKind kind = TestKind::PRP;
  std::string regime;
  std::string options;  // configText() of the canonical set

  [[nodiscard]] auto operator<=>(const ConfigKey&) const = default;
};

// One configuration's calls.
struct Reading {
  FFTConfig fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  UseConfig options{};  // canonical
  u64 exponent = 0;     // where it was last called

  u32 calls = 0;        // that read a cost
  double mean = 0;      // of the calls' costs (drift-corrected, and as the power limit holds them), us/it
  double se = 0;        // standard error of that mean, between calls; a floor where there is one call
  bool failed = false;  // a call computed wrongly, failed to build, or the device refused it

  // What the calls drew, where the card says (PowerRow): the mean energy of an iteration in millijoules and power in
  // watts, and how many calls' costs the power limit set rather than their timing.
  double mjPerIt = NAN;
  double watts = NAN;
  u32 powerBound = 0;

  // What the calls alone say it costs, as above: before the soaks set `mean`, or the calibration scaled it.
  double timed = 0;
  // Its soaks (Soak.h), where it has had them: the mean of its slices after their warm-ups, and how many there were.
  // A soaked configuration costs what it sustained.
  double soaked = NAN;
  u32 soakSlices = 0;

  [[nodiscard]] bool isSoaked() const { return soakSlices > 0; }

  // What a published line costs: the mean, pessimistically by its error bar.
  [[nodiscard]] double pessimistic() const;
};

// One rounding-error reading of a configuration.
struct RoeReading {
  u64 exponent = 0;
  bool checkOk = false;
  double z = 0;
  u32 n = 0;
};

class Readings {
public:
  // `powerHold`: POWER_HOLD, as a run's powerHold= setting has it.
  Readings(const TuneDB& db, u32 env, const Env& device, double powerHold = POWER_HOLD);

  [[nodiscard]] const std::map<ConfigKey, Reading>& all() const { return readings_; }

  // The readings of one subject, every option set it was called with.
  [[nodiscard]] std::vector<const Reading*> of(const Subject& subject) const;

  [[nodiscard]] const Reading* find(const Subject& subject, const UseConfig& options) const;

  // Every successful call, as the model learns from it.
  [[nodiscard]] const std::vector<Observation>& observations() const { return observations_; }

  // The rounding-error readings of one FFT under one option set (kind-free: they are taken on the PRP kernels).
  [[nodiscard]] std::vector<RoeReading> roes(const FFTConfig& fft, const UseConfig& options) const;

  [[nodiscard]] static ConfigKey keyOf(const Subject& subject, const UseConfig& options);

  // What a soak sustains against what calls read of the same configuration, the median over every soaked
  // configuration of the env; 1 where none has been soaked.  The cost of a configuration not soaked yet is its calls'
  // times this, so soaked and unsoaked compare on one footing: calls on a card at its power limit read 1-2% fast even
  // with the power floor, as the chip warms over minutes no call lasts.
  [[nodiscard]] double calibration() const { return calibration_; }

private:
  std::map<ConfigKey, Reading> readings_;
  std::vector<Observation> observations_;
  double calibration_ = 1;
  std::map<std::pair<std::string, std::string>, std::vector<RoeReading>> roes_;
};

}  // namespace tune
