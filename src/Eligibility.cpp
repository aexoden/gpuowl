// Copyright (C) Jason Lynch

#include "Eligibility.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace tune {

namespace {

// The smallest E in [lo, hi] satisfying `pred`, or hi + 1 if none does. `pred` must be monotone over the range: false
// below its boundary, true at and above it.
template<typename Pred> u64 lowerBoundary(u64 lo, u64 hi, Pred pred) {
  if (lo > hi || !pred(hi)) { return hi + 1; }

  while (lo < hi) {
    u64 const mid = lo + (hi - lo) / 2;
    if (pred(mid)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  return lo;
}

// The wider carry (64 bits, or 96 for FFT323161) because the exponent asks for one, rather than because the spec pins
// one.
bool needsWideCarry(const FFTConfig& fft, u64 E) { return fft.carry == CARRY_AUTO && fft.shape.needsLargeCarry(E); }

}  // namespace

std::string Regime::label() const {
  return string{longCarry ? "long" : "short"} + (carry96 ? "96" : carry64 ? "64" : "32");
}

std::optional<Regime> parseRegime(std::string_view label) {
  for (bool longCarry : {false, true}) {
    for (Regime const regime : {Regime{longCarry, false}, Regime{longCarry, true}, Regime{longCarry, false, true}}) {
      if (regime.label() == label) { return regime; }
    }
  }

  return {};
}

bool exactArithmetic(const FFTConfig& fft) { return !fft.FFT_FP64 && !fft.FFT_FP32; }

float bitsPerWord(const FFTConfig& fft, u64 E) { return E / float(fft.size()); }

Regime regimeOf(const FFTConfig& fft, u64 E) {
  // As Gpu's constructor decides it: carryFused leaves part of each carry unnormalized, which only settles when the
  // words are well above sqrt(N).
  double const longBelow = std::max(10.0, 0.5 * std::log2(double(fft.size())) + 0.5);
  bool const longCarry = bitsPerWord(fft, E) < longBelow;
  bool const wide = needsWideCarry(fft, E) || fft.carry == CARRY_64;
  if (fft.shape.fft_type == FFT323161) { return {.longCarry = longCarry, .carry64 = !wide, .carry96 = wide}; }
  return {.longCarry = longCarry, .carry64 = wide};
}

u64 minExp(const FFTConfig& fft) {
  u64 const estimate = u64(std::ceil(double(fft.minBpw()) * fft.size()));
  u64 const slack = 64;
  return lowerBoundary(estimate < slack ? 0 : estimate - slack, estimate + slack,
                       [&](u64 E) { return bitsPerWord(fft, E) >= fft.minBpw(); });
}

u64 maxExp(const FFTConfig& fft) { return fft.maxExp(); }

bool isEligible(const FFTConfig& fft, u64 E) { return bitsPerWord(fft, E) >= fft.minBpw(); }

std::vector<Interval> intervals(const FFTConfig& fft, u64 lo, u64 hi) {
  lo = std::max(lo, minExp(fft));
  if (lo > hi) { return {}; }

  u64 const shortCarryFrom = lowerBoundary(lo, hi, [&](u64 E) { return !regimeOf(fft, E).longCarry; });
  u64 const wideCarryFrom = lowerBoundary(lo, hi, [&](u64 E) { return needsWideCarry(fft, E); });

  std::vector<u64> starts{lo};
  for (u64 cut : {shortCarryFrom, wideCarryFrom}) {
    if (cut > lo && cut <= hi) { starts.push_back(cut); }
  }

  std::ranges::sort(starts);
  starts.erase(std::unique(starts.begin(), starts.end()), starts.end());

  std::vector<Interval> out;
  for (size_t i = 0; i < starts.size(); ++i) {
    u64 const end = i + 1 < starts.size() ? starts[i + 1] - 1 : hi;
    Regime const regime = regimeOf(fft, starts[i]);
    assert(regimeOf(fft, end) == regime);
    out.push_back({starts[i], end, regime});
  }

  return out;
}

Interval interval(const FFTConfig& fft, u64 E, u64 reach) {
  if (E > reach || !isEligible(fft, E)) { return {}; }

  for (const Interval& candidate : intervals(fft, minExp(fft), reach)) {
    if (candidate.contains(E)) { return candidate; }
  }

  return {};
}

Interval interval(const FFTConfig& fft, u64 E) { return interval(fft, E, maxExp(fft)); }

}  // namespace tune
