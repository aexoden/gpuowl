// Copyright (C) Jason Lynch

#include "Gate.h"

#include "Eligibility.h"
#include "Primes.h"

#include <algorithm>

namespace tune {

namespace {

const Primes& primes() {
  static Primes const p;
  return p;
}

u64 primeAtOrBelow(u64 E) { return primes().isPrime(E) ? E : primes().prevPrime(E); }

// A value that does not parse is taken to change the rounding, which is the safe error.
bool changesRounding(const Option& option, const std::string& value) {
  std::optional<int> const v = parseInt<int>(value);
  return option.accuracyImpact != AccuracyImpact::None && (!v || option.changesRounding(*v));
}

}  // namespace

bool movesAccuracy(const Env& env, const FFTConfig& fft, const UseConfig& options) {
  for (const auto& [key, value] : canonicalConfig(env, fft, options)) {
    if (const Option* const o = findOption(key); o && changesRounding(*o, value)) { return true; }
  }
  return false;
}

UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& options) {
  UseConfig out = options;
  // Until nothing moves, since one key's default can turn another on.
  for (bool changed = true; changed;) {
    changed = false;
    for (const Option& option : allOptions()) {
      if (option.kind != Kind::Tunable || option.accuracyImpact == AccuracyImpact::None ||
          !option.appliesTo(env, fft, out)) {
        continue;
      }
      std::string const own = std::to_string(option.defaultFor(env, fft, out));
      auto const at = out.find(option.key);
      if (at != out.end() && at->second != own && !changesRounding(option, at->second)) { continue; }
      if (at == out.end() || at->second != own) {
        out[option.key] = own;
        changed = true;
      }
    }
  }
  return canonicalConfig(env, fft, out);
}

double minSafeZ(enum FFT_TYPES type) { return type == FFT64 ? 20 : 6; }

const char* toString(Accuracy accuracy) {
  switch (accuracy) {
  case Accuracy::Exact: return "exact";
  case Accuracy::Confirmed: return "confirmed";
  case Accuracy::Lowered: return "lowered";
  case Accuracy::Owed: return "owed";
  case Accuracy::Rejected: return "rejected";
  }
  return "?";
}

bool safe(enum FFT_TYPES type, const RoeReading& reading, const RoeReading* reference) {
  if (!reading.checkOk) { return false; }
  if (reading.n <= 2) { return true; }
  double const floor = minSafeZ(type);
  if (!reference) { return reading.z >= floor; }
  if (!reference->checkOk || reference->n <= 2 || reference->z < floor) { return false; }
  return reading.z >= std::max(reference->z * (1 - ACCURACY_SLACK), floor);
}

Verdict verdictOf(const FFTConfig& fft, std::span<const RoeReading> readings, u64 floor,
                  const std::vector<RoeReading>* reference) {
  u64 const table = maxExp(fft);
  if (exactArithmetic(fft)) { return {.state = Accuracy::Exact, .reach = table}; }

  // The reference's latest reading at an exponent, for a set held to one.
  auto referenceAt = [&](u64 E) -> const RoeReading* {
    const RoeReading* out = nullptr;
    for (const RoeReading& r : *reference) {
      if (r.exponent == E) { out = &r; }
    }
    return out;
  };
  // A reading of a set held to its reference counts once the reference has been read beside it.
  auto judged = [&](const RoeReading& r) { return !reference || referenceAt(r.exponent); };
  auto isSafe = [&](const RoeReading& r) {
    return safe(fft.shape.fft_type, r, reference ? referenceAt(r.exponent) : nullptr);
  };

  u64 const top = primeAtOrBelow(table);
  u64 pass = 0;
  u64 unjudged = 0;
  for (const RoeReading& r : readings) {
    if (!judged(r)) {
      unjudged = std::max(unjudged, r.exponent);
    } else if (isSafe(r)) {
      pass = std::max(pass, r.exponent);
    }
  }
  if (unjudged) { return {.state = Accuracy::Owed, .reach = pass, .next = unjudged, .owesReference = true}; }
  if (pass >= top) { return {.state = Accuracy::Confirmed, .reach = table}; }

  // The lowest unsafe reading above the highest safe one: an unsafe reading below a safe one is noise in a reading
  // near the threshold, and the safe one above it stands.
  u64 fail = 0;
  for (const RoeReading& r : readings) {
    if (!isSafe(r) && r.exponent > pass && (!fail || r.exponent < fail)) { fail = r.exponent; }
  }
  if (!fail) { return {.state = Accuracy::Owed, .reach = pass, .next = top}; }

  u64 const low = std::max(pass, floor);
  if (fail <= floor || (pass < floor && double(fail - floor) <= REACH_RESOLUTION * double(floor))) {
    return {.state = Accuracy::Rejected};
  }
  if (pass && double(fail - pass) <= REACH_RESOLUTION * double(pass)) {
    return pass < floor ? Verdict{.state = Accuracy::Rejected} : Verdict{.state = Accuracy::Lowered, .reach = pass};
  }

  u64 const next = primeAtOrBelow(low + (fail - low) / 2);
  // Nothing left between them to read: what passed is the reach.
  if (next <= low) {
    return pass >= floor ? Verdict{.state = Accuracy::Lowered, .reach = pass} : Verdict{.state = Accuracy::Rejected};
  }
  return {.state = Accuracy::Owed, .reach = pass, .next = next};
}

Verdict verdictFor(const Readings& readings, const FFTConfig& fft, const UseConfig& options, u64 floor) {
  std::vector<RoeReading> const own = readings.roes(fft, options);
  if (!movesAccuracy(readings.device(), fft, options)) { return verdictOf(fft, own, floor); }
  std::vector<RoeReading> const reference = readings.roes(fft, accuracyReference(readings.device(), fft, options));
  return verdictOf(fft, own, floor, &reference);
}

UseConfig owedOptions(const Readings& readings, const FFTConfig& fft, const UseConfig& options, const Verdict& verdict) {
  return verdict.owesReference ? accuracyReference(readings.device(), fft, options) : options;
}

}  // namespace tune
