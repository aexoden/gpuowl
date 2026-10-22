// Copyright (C) Jason Lynch

// What one env's database says, as the tuner reads it: every configuration that was called, its readings pooled, and
// every rounding-error reading.  Pure: a function of the database, rebuilt whenever it has grown.

#pragma once

#include "Eligibility.h"
#include "Model.h"
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
  double mean = 0;      // of the calls' drift-corrected costs, microseconds per iteration
  double se = 0;        // standard error of that mean, between calls; a floor where there is one call
  bool failed = false;  // a call computed wrongly, failed to build, or the device refused it

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
  Readings(const TuneDB& db, u32 env, const Env& device);

  [[nodiscard]] const std::map<ConfigKey, Reading>& all() const { return readings_; }

  // The readings of one subject, every option set it was called with.
  [[nodiscard]] std::vector<const Reading*> of(const Subject& subject) const;

  [[nodiscard]] const Reading* find(const Subject& subject, const UseConfig& options) const;

  // Every successful call, as the model learns from it.
  [[nodiscard]] const std::vector<Observation>& observations() const { return observations_; }

  // The rounding-error readings of one FFT under one option set (kind-free: they are taken on the PRP kernels).
  [[nodiscard]] std::vector<RoeReading> roes(const FFTConfig& fft, const UseConfig& options) const;

  [[nodiscard]] static ConfigKey keyOf(const Subject& subject, const UseConfig& options);

private:
  std::map<ConfigKey, Reading> readings_;
  std::vector<Observation> observations_;
  std::map<std::pair<std::string, std::string>, std::vector<RoeReading>> roes_;
};

}  // namespace tune
