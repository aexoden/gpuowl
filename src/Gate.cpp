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

}  // namespace

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

bool safe(enum FFT_TYPES type, const RoeReading& reading) {
  return reading.checkOk && (reading.n <= 2 || reading.z >= minSafeZ(type));
}

Verdict verdictOf(const FFTConfig& fft, std::span<const RoeReading> readings, u64 floor) {
  u64 const table = maxExp(fft);
  if (exactArithmetic(fft)) { return {.state = Accuracy::Exact, .reach = table}; }

  u64 const top = primeAtOrBelow(table);
  u64 pass = 0;
  for (const RoeReading& r : readings) {
    if (safe(fft.shape.fft_type, r)) { pass = std::max(pass, r.exponent); }
  }
  if (pass >= top) { return {.state = Accuracy::Confirmed, .reach = table}; }

  // The lowest unsafe reading above the highest safe one: an unsafe reading below a safe one is noise in a reading
  // near the threshold, and the safe one above it stands.
  u64 fail = 0;
  for (const RoeReading& r : readings) {
    if (!safe(fft.shape.fft_type, r) && r.exponent > pass && (!fail || r.exponent < fail)) { fail = r.exponent; }
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

}  // namespace tune
