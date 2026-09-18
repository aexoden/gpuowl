// Copyright (C) Jason Lynch

// Where a configuration may run, and what it runs as while it is there: the smallest exponent whose bits per word the
// Gpu constructor accepts, the largest the fitted bpw table holds it accurate to, and the carry regimes in between.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// Exponent-dependent switches in Gpu that change what runs.
struct Regime {
  // The carry is expanded into carryA/carryB rather than fused into the width pass. Forced below 10 bits per word.
  bool longCarry = false;

  // The kernels are compiled with -DCARRY64, a 64-bit carry. Used either if the exponent needs one or if the spec pins
  // one.
  bool carry64 = false;

  [[nodiscard]] bool operator==(const Regime&) const = default;

  // "short32", "short64", "long32", "long64"
  [[nodiscard]] std::string label() const;
};

[[nodiscard]] std::optional<Regime> parseRegime(std::string_view label);

// Inclusive exponent range over which one configuration runs one way.
struct Interval {
  u64 lo = 1;
  u64 hi = 0;
  Regime regime{};

  [[nodiscard]] bool operator==(const Interval&) const = default;

  [[nodiscard]] bool empty() const { return hi < lo; }
  [[nodiscard]] bool contains(u64 E) const { return lo <= E && E <= hi; }
};

// Bits per word as the Gpu constructor computes it, including its conversion of the exponent to float.
[[nodiscard]] float bitsPerWord(const FFTConfig& fft, u64 E);

// How Gpu would run `fft` at `E`. Says nothing about whether it is able as an ineligible exponent still has a regime.
[[nodiscard]] Regime regimeOf(const FFTConfig& fft, u64 E);

// The smallest exponent the Gpu constructor accepts for this FFT.
[[nodiscard]] u64 minExp(const FFTConfig& fft);

// The largest exponent the fitted bpw table holds this configuration accurate to, at the default options.
[[nodiscard]] u64 maxExp(const FFTConfig& fft);

[[nodiscard]] bool isEligible(const FFTConfig& fft, u64 E);

// The interval an entry measured at `E` covers, clamped to [minExp(fft), reach].
[[nodiscard]] Interval interval(const FFTConfig& fft, u64 E, u64 reach);
[[nodiscard]] Interval interval(const FFTConfig& fft, u64 E);

// [lo, hi] raised to minExp(fft) and cut at every regime change, in ascending order.
[[nodiscard]] std::vector<Interval> intervals(const FFTConfig& fft, u64 lo, u64 hi);

}  // namespace tune
