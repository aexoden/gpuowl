// Copyright (C) Jason Lynch

#include "Reach.h"

#include "Eligibility.h"
#include "Primes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace tune {

namespace {

[[nodiscard]] std::string format(const char* fmt, double a, double b = 0) {
  char buf[160];
  snprintf(buf, sizeof(buf), fmt, a, b);
  return buf;
}

const Primes& primes() {
  static const Primes out;
  return out;
}

// The largest prime at or below `E` that is still at least `lo`, or 0.
[[nodiscard]] u64 primeAtOrBelow(u64 E, u64 lo) {
  if (E < 2) { return 0; }
  u64 const p = primes().isPrime(E) ? E : primes().prevPrime(E);
  return p >= lo ? p : 0;
}

// The smallest prime at or above `lo` that is still at most `hi`, or 0.
[[nodiscard]] u64 primeAtOrAbove(u64 lo, u64 hi) {
  u64 const p = primes().isPrime(lo) ? lo : primes().nextPrime(lo);
  return p <= hi ? p : 0;
}

}  // namespace

double startSlope(u64 words) {
  double const at4M = std::log2(4.0 * 1024 * 1024);
  double const at7M5 = std::log2(7.5 * 1024 * 1024);
  double const s = 0.015 + (std::log2(double(words)) - at4M) / (at7M5 - at4M) * (0.012 - 0.015);
  return std::clamp(s, MIN_SLOPE, MAX_SLOPE);
}

double fitSlope(const ZReading& a, const ZReading& b, u64 words) {
  if (a.exponent == b.exponent || a.z == b.z || a.n <= 2 || b.n <= 2) { return startSlope(words); }
  double const dBpw = (double(a.exponent) - double(b.exponent)) / double(words);
  return std::clamp(dBpw / (b.z - a.z), MIN_SLOPE, MAX_SLOPE);
}

ReachOutcome deriveReach(const ReachProblem& p, const ZReading& first, const ReadingAt& at) {
  auto const bpw = [&](u64 E) { return double(E) / double(p.words); };

  // The prime a reach at `b` bits per word is proposed at, held inside [lo, hi]; 0 where that holds none.
  auto const proposalAt = [&](double b) {
    double const E = b * double(p.words);
    if (E <= double(p.lo)) { return primeAtOrAbove(p.lo, p.hi); }
    return primeAtOrBelow(u64(std::min(E, double(p.hi))), p.lo);
  };

  auto const owed = [](u64 E) { return ReachOutcome{.state = ReachState::Owed, .exponent = E, .why = {}}; };
  auto const confirmed = [](u64 E) { return ReachOutcome{.state = ReachState::Confirmed, .exponent = E, .why = {}}; };
  auto const rejected = [](std::string why) {
    return ReachOutcome{.state = ReachState::Rejected, .exponent = 0, .why = std::move(why)};
  };

  // Errors too small to fit z to are more accurate than any z a standard can state.
  auto const passes = [&](const ZReading& r) { return r.checkOk && (r.n <= 2 || r.z >= p.standard.bar); };

  // Every reading the replay has used, since one taken on the way may already confirm a band the backoff comes to.
  std::vector<ZReading> used{first};

  ZReading current = first;
  std::optional<ZReading> previous;
  u64 proposal = 0;
  for (u32 count = 1;; ++count) {
    if (current.n <= 2 && current.checkOk && current.exponent <= p.hi) { return confirmed(current.exponent); }

    double const slope = previous ? fitSlope(*previous, current, p.words) : startSlope(p.words);
    proposal = proposalAt(bpw(current.exponent) + (current.z - p.standard.aim) * slope);
    if (!proposal) { return rejected("its interval holds no prime to propose a reach at"); }

    bool const near = std::abs(current.z - p.standard.aim) <= EXTRAPOLATION_CAP_Z;
    if (near || proposal == current.exponent) { break; }

    if (count >= MAX_DERIVE_READINGS) {
      return rejected(format("its z did not come within reach of %.2f in %.0f readings", p.standard.aim,
                             double(MAX_DERIVE_READINGS)));
    }

    std::optional<ZReading> const next = at(proposal);
    if (!next) { return owed(proposal); }
    previous = current;
    current = *next;
    used.push_back(current);
  }

  // Each guard band is taken from the proposal, and a band that would leave the interval ends the backoff there.  A
  // reading already taken between one band and the next that clears the bar is a reach confirmed higher up than the
  // band would be: z scatters by about a unit from one prime to the next, so the proposal can fail just above a
  // reading that passed.
  u64 tried = proposal;
  for (u32 t = 0; t <= REACH_TRIES; ++t) {
    double const b = bpw(proposal) - t * REACH_GUARD_BPW;
    if (b * double(p.words) < double(p.lo)) { break; }

    u64 const E = t ? primeAtOrBelow(u64(b * double(p.words)), p.lo) : proposal;
    if (!E) { break; }

    if (t) {
      std::optional<u64> best;
      for (const ZReading& r : used) {
        if (r.exponent >= E && r.exponent < tried && passes(r) && (!best || r.exponent > *best)) { best = r.exponent; }
      }
      if (best) { return confirmed(*best); }
    }

    std::optional<ZReading> const reading = E == current.exponent ? std::optional{current} : at(E);
    if (!reading) { return owed(E); }
    if (passes(*reading)) { return confirmed(E); }
    used.push_back(*reading);
    tried = E;
  }

  return rejected(format("it read below %.2f at its proposed reach and at each guard band under it", p.standard.bar));
}

u64 carryCeiling(const FFTConfig& fft) {
  // As FFTConfig::maxBpw() has it: the single-arithmetic FFTs take their carry type from CARRY64, and FFT3231's
  // carries are always 32-bit.
  enum FFT_TYPES const type = fft.shape.fft_type;
  bool const pinned32 =
    fft.carry == CARRY_32 && (type == FFT64 || type == FFT3231 || type == FFT61 || type == FFT31 || type == FFT32);
  if (!pinned32) { return std::numeric_limits<u64>::max(); }

  u64 const words = fft.size();
  u64 const below19 = 19 * words - 1;
  return std::min(u64(double(fft.shape.carry32BPW()) * double(words)), below19);
}

u64 raiseCeiling(const FFTConfig& fft) {
  u64 const top = maxExp(fft);
  u64 const cap = std::min(u64((double(fft.maxBpw()) + MAX_RAISE_BPW) * double(fft.size())), carryCeiling(fft));
  if (cap <= top) { return std::min(cap, top); }

  std::vector<Interval> const above = intervals(fft, top, cap);
  return above.empty() ? top : above.front().hi;
}

}  // namespace tune
