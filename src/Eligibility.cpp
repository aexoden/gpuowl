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

// A 64-bit carry because the exponent asks for one, rather than because the spec pins one.
bool needsCarry64(const FFTConfig& fft, u64 E) { return fft.carry == CARRY_AUTO && fft.shape.needsLargeCarry(E); }

}  // namespace

std::string Regime::label() const { return string{longCarry ? "long" : "short"} + (carry64 ? "64" : "32"); }

std::optional<Regime> parseRegime(std::string_view label) {
  for (bool longCarry : {false, true}) {
    for (bool carry64 : {false, true}) {
      Regime const regime{longCarry, carry64};
      if (regime.label() == label) { return regime; }
    }
  }

  return {};
}

float bitsPerWord(const FFTConfig& fft, u64 E) { return E / float(fft.size()); }

Regime regimeOf(const FFTConfig& fft, u64 E) {
  return {.longCarry = bitsPerWord(fft, E) < 10.0f, .carry64 = needsCarry64(fft, E) || fft.carry == CARRY_64};
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
  u64 const carry64From = lowerBoundary(lo, hi, [&](u64 E) { return needsCarry64(fft, E); });

  std::vector<u64> starts{lo};
  for (u64 cut : {shortCarryFrom, carry64From}) {
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
