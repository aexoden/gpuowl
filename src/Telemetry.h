// Copyright (C) Jason Lynch

// What a card says about itself while it works: its clocks, its temperature, the power it draws against its limit, and
// what is holding its clocks back.  NVIDIA cards through NVML, loaded at run time so that a machine without it still
// runs; AMD cards through the amdgpu driver's sysfs files, which say nothing of why the clocks are held back.  Read on
// a thread of its own every fraction of a second, so that what a timing saw can be set beside it.

#pragma once

#include "common.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace tune {

// One reading of the card.  NaN where the card does not say.
struct CardSample {
  double t = 0;  // seconds since the sampler began
  double smMHz = NAN;
  double memMHz = NAN;
  double tempC = NAN;
  double powerW = NAN;
  double capW = NAN;  // the power limit in force
  u64 held = 0;       // NVML's clock event reasons (throttle reasons); 0 where none, or none are reported
};

class Telemetry {
public:
  virtual ~Telemetry() = default;

  // A reading now; its time is left to the sampler.  Nothing where the card did not answer.
  [[nodiscard]] virtual std::optional<CardSample> read() = 0;

  // Where the readings come from: "NVML, 0000:03:00.0", "amdgpu sysfs, card1".
  [[nodiscard]] virtual std::string source() const = 0;

  // Whether the source says why clocks are held back (CardSample::held).
  [[nodiscard]] virtual bool saysWhy() const { return false; }
};

// The card at PCI address `bdf` ("03:00.0"), by its vendor's interface.  Nothing where none can be read, with `why`
// saying what was missing.  An empty `bdf` is taken for the only card of its vendor, where there is only one.
[[nodiscard]] std::unique_ptr<Telemetry> openTelemetry(const std::string& bdf, bool nvidia, bool amd, std::string& why);

// Reads a Telemetry every `period` seconds on a thread of its own, from construction until destruction.  Without one
// it reads nothing, and every window is empty.
class Sampler {
public:
  Sampler(Telemetry* telemetry, double period);
  ~Sampler();

  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // Seconds since construction, on the clock the samples' times are on.
  [[nodiscard]] double now() const;

  // The samples taken in [from, to].
  [[nodiscard]] std::vector<CardSample> between(double from, double to) const;

private:
  Telemetry* telemetry_;
  std::chrono::steady_clock::time_point start_;
  mutable std::mutex mutex_;
  std::vector<CardSample> samples_;
  std::jthread thread_;
};

// The MHz of the line amdgpu marks current in a pp_dpm_* file ("1: 1500Mhz *"); nothing where no line is marked.
[[nodiscard]] std::optional<double> currentDpmMHz(std::string_view ppDpm);

// The reasons NVML names by these clock event bits, most telling first: "power cap, thermal".  The idle bit is left
// out: a card that is working is not idle, and one that is reads nothing worth naming.
[[nodiscard]] std::string heldText(u64 bits);

// What a window of samples says, for a report.
struct CardSummary {
  u32 n = 0;
  double smMin = NAN;
  double smMedian = NAN;
  double smMax = NAN;
  double memMedian = NAN;
  double tempMedian = NAN;
  double tempMax = NAN;
  double powerMedian = NAN;
  double capW = NAN;
  // Each reason clocks were held back for, with the share of the samples it held them in.
  std::vector<std::pair<std::string, double>> held;
};

[[nodiscard]] CardSummary summarizeCard(std::span<const CardSample> samples);

// "SM 1328 MHz (1300-1328), mem 715 MHz, 74 C (max 76), 210/250 W; held back by power cap in 12% of samples"; empty
// for a summary of no samples.
[[nodiscard]] std::string toString(const CardSummary& summary, bool saysWhy);

}  // namespace tune
