// Copyright (C) Jason Lynch

// What a card says about itself while it works: its clocks, its temperature, the power it draws against its limit, and
// what is holding its clocks back.  NVIDIA cards through NVML, loaded at run time so that a machine without it still
// runs; AMD cards through the amdgpu driver's sysfs files, which say nothing of why the clocks are held back.  Read on
// a thread of its own every fraction of a second, so that what a timing saw can be set beside it.

#pragma once

#include "common.h"
#include "clwrap.h"

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
  double capW = NAN;     // the power limit in force
  double energyJ = NAN;  // the card's energy counter, joules since some moment of its own
  u64 held = 0;          // NVML's clock event reasons (throttle reasons); 0 where none, or none are reported
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

  // Whether its power readings are of the moment, rather than an average over the last second or more.
  [[nodiscard]] virtual bool instantPower() const { return false; }
};

// The card at PCI address `bdf` ("03:00.0"), by its vendor's interface.  Nothing where none can be read, with `why`
// saying what was missing.  An empty `bdf` is taken for the only card of its vendor, where there is only one.
[[nodiscard]] std::unique_ptr<Telemetry> openTelemetry(const std::string& bdf, bool nvidia, bool amd, std::string& why);

// The same for an OpenCL device, by its PCI address and vendor.
[[nodiscard]] std::unique_ptr<Telemetry> openTelemetryFor(cl_device_id id, std::string& why);

// The device's PCI address ("03:00.0"): AMD's topology query (which the CUDA backend answers too), else NVIDIA's
// OpenCL extension; empty where neither says.
[[nodiscard]] std::string pciAddressOf(cl_device_id id);

// Reads a Telemetry every `period` seconds on a thread of its own, from construction until destruction, keeping the
// last `keep` seconds of samples (all of them for 0).  Without a Telemetry it reads nothing, and every window is empty.
class Sampler {
public:
  Sampler(Telemetry* telemetry, double period, double keep = 0);
  ~Sampler();

  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // Seconds since construction, on the clock the samples' times are on.
  [[nodiscard]] double now() const;
  [[nodiscard]] double at(std::chrono::steady_clock::time_point t) const;

  // The samples taken in [from, to].
  [[nodiscard]] std::vector<CardSample> between(double from, double to) const;

  // Waits until a sample later than `t` has been taken, at most `most` seconds: what a window ending at `t` needs
  // around its end.  False where none came (or nothing is read).
  bool waitPast(double t, double most) const;

  [[nodiscard]] bool instantPower() const { return telemetry_ && telemetry_->instantPower(); }

private:
  Telemetry* telemetry_;
  double keep_;
  std::chrono::steady_clock::time_point start_;
  mutable std::mutex mutex_;
  std::vector<CardSample> samples_;
  std::jthread thread_;
};

// The share of its power limit a card that runs into it holds for long: a 4070 SUPER with a 220 W limit held 208 W
// under sustained load (-measure, 2026-10-28).  A tuner's call times a second or so, too short for the limit to catch
// up with it, so it reads faster than the card can sustain; there, an iteration could be costed at least the energy
// it took over this share of the limit (Readings, -tune powerHold=), which -measure checks against what the card
// sustains.
inline constexpr double HOLD_SHARE = 0.95;

// The tuner's powerHold= unless set: off.  On the 4070 SUPER the energy NVML gives for a call's second of timing read
// 15% below an iteration's sustained energy, at higher clocks, and swung 16% between identical calls: a slow sensor
// still carrying the idle before the call.  A cost built on it would charge one call for the call before.  The power
// is recorded with every call regardless, so the floor can be tried on the readings later.
inline constexpr double POWER_HOLD = 0;

// What a card drew over a window: from its energy counter where it has one, else from its power readings.
struct WindowPower {
  double joules = NAN;
  double watts = NAN;  // mean, by whichever of the two below is taken
  double capW = NAN;   // the power limit in force
  double smMHz = NAN;  // median
  bool counted = false;  // taken from the energy counter

  double counterWatts = NAN;  // by the energy counter, where it brackets the window
  double readingWatts = NAN;  // by the power readings
  u32 readings = 0;           // power readings inside the window
};

// The fewest instantaneous power readings inside a window that are taken over the energy counter.
inline constexpr u32 INSTANT_READINGS = 3;

// Over [from, to], from `samples` ordered by time and reaching a little past either end; nothing where they say
// nothing of it.  The energy counter is interpolated at either end between the samples around it; power readings
// give their median over those inside the window, or the nearest one where the window falls between two.  The
// counter is taken where it brackets the window, unless the readings are `instant` and INSTANT_READINGS of them fall
// inside it.
[[nodiscard]] std::optional<WindowPower> powerOver(std::span<const CardSample> samples, double from, double to,
                                                   bool instant = false);

// The same from a sampler, for a window of the steady clock.
[[nodiscard]] std::optional<WindowPower> powerOver(const Sampler& sampler, std::chrono::steady_clock::time_point from,
                                                   std::chrono::steady_clock::time_point to);

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
