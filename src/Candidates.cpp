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

        // Few distinct timing exponents, since each LL one needs a reference residue settled first: the probe, else
        // the workload's ends, else the candidate's own top point.
        u64 timing = 0;
        for (u64 const E :
             {scope.probe, grid->points.front().exponent, grid->points.back().exponent, c.points.back().exponent}) {
          u64 const prime = primes.isPrime(E) ? E : primes.prevPrime(E);
          if (interval.contains(prime)) {
            timing = prime;
            break;
          }
        }
        if (!timing) { continue; }
        c.timing = timing;
        out.push_back(std::move(c));
      }
    }
  }
  return out;
}

}  // namespace tune
