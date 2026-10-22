// Copyright (C) Jason Lynch

#include "Candidates.h"

#include "FFTVariants.h"
#include "Primes.h"

#include <algorithm>

namespace tune {

double Candidate::weight() const {
  double sum = 0;
  for (const GridPoint& p : points) { sum += p.weight; }
  return sum;
}

std::vector<Candidate> candidatesOf(const Env& env, const RunScope& scope, TestKind kind) {
  std::vector<Candidate> out;
  const Grid* const grid = scope.grid(kind);
  if (!grid) { return out; }

  static Primes const primes;
  for (const FFTShape& shape : FFTShape::allShapes()) {
    if (!scope.admits(shape)) { continue; }
    for (u32 const variant : runnableVariants(env, shape)) {
      FFTConfig const fft{shape, variant, CARRY_AUTO};
      u64 const hi = std::min(scope.hi, maxExp(fft));
      if (hi < scope.lo) { continue; }

      for (const Interval& interval : intervals(fft, scope.lo, hi)) {
        Candidate c{
          .subject = {.fft = fft, .kind = kind, .regime = interval.regime}, .interval = interval, .points = {}};
        for (const GridPoint& p : grid->points) {
          if (p.weight > 0 && interval.contains(p.exponent)) { c.points.push_back(p); }
        }
        if (c.points.empty()) { continue; }

        u64 const want = interval.contains(scope.probe) ? scope.probe : c.points.back().exponent;
        u64 const timing = primes.isPrime(want) ? want : primes.prevPrime(want);
        c.timing = timing >= interval.lo ? timing : primes.nextPrime(interval.lo - 1);
        if (!interval.contains(c.timing)) { continue; }
        out.push_back(std::move(c));
      }
    }
  }
  return out;
}

}  // namespace tune
