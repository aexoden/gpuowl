#include "common.h"

#include <vector>

class Primes {
  u32 limit;
  std::vector<bool> primeMap;
  std::vector<u32> primes;

public:
  struct Range {
    typedef std::vector<u32>::const_iterator T;
    T b, e;
    T begin() { return b; }
    T end() { return e; }
  };

  explicit Primes(u32 limit);

  bool isPrime(u32 x) {
    return (x == 2) || ((x & 1) && (x <= limit) && primeMap[(x - 1) >> 1]);
  }

  Range from(u32 p) {
    auto it = primes.cbegin(), end = primes.cend();
    while (it < end && *it < p) { ++it; }
    return {it, end};
  }

  std::vector<std::pair<u32, u32>> factors(u32 x);

  std::vector<u32> divisors(u32 x);

  // Multiplicative order of 2 modulo p. Equivalent PARI-GP: z(p) = znorder(Mod(2, p)).
  u32 zn2(u32 p);
};
