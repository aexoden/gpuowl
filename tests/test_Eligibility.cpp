// Copyright (C) Jason Lynch

// GPU-free tests of eligibility and carry regimes (src/Eligibility.cpp). The boundaries are checked against the
// expressions Gpu's constructor and FFTShape enforce, and against hand-computed exponents for one shape, so that a
// change to either is a test failure rather than an entry published for exponents it cannot run.

#include "test.h"

#include "Eligibility.h"
#include "FFTVariants.h"

#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

// 256K words: minBpw 3.0 puts the floor at 786432, the long-carry switch at 10 bpw is 2621440, and carry32BPW() caps
// at 19.0 there, so needsLargeCarry() turns on at 19 * 262144. A shape reads the fitted bpw table as it is built, so
// build it inside a test rather than at namespace scope, where it would race that table's initialization.
FFTShape small() { return FFTShape{FFT64, 256, 2, 256}; }
constexpr u64 SMALL_SIZE = 262'144;

FFTConfig cfg(const FFTShape& shape, CARRY_KIND carry = CARRY_AUTO) {
  return FFTConfig{shape, defaultVariant(shape), carry};
}

} // namespace

TEST(regime_labels_round_trip) {
  std::set<std::string> labels;
  for (bool longCarry : {false, true}) {
    for (bool carry64 : {false, true}) {
      Regime const regime{longCarry, carry64};
      labels.insert(regime.label());
      CHECK(parseRegime(regime.label()) == regime);
    }
  }
  CHECK_EQ(labels.size(), size_t(4));
  CHECK(!parseRegime("short"));
  CHECK(!parseRegime(""));
}

TEST(min_exp_fixtures) {
  CHECK_EQ(minExp(cfg(small())), 3 * SMALL_SIZE);
  CHECK_EQ(minExp(FFTConfig{FFTShape{FFT32, 256, 2, 256}, 2, CARRY_AUTO}), SMALL_SIZE);
}

TEST(min_exp_is_the_constructor_boundary) {
  for (const FFTShape& shape : FFTShape::allShapes()) {
    FFTConfig const fft = cfg(shape);
    u64 const floorExp = minExp(fft);
    CHECK(isEligible(fft, floorExp));
    CHECK(!isEligible(fft, floorExp - 1));
    CHECK(bitsPerWord(fft, floorExp) >= fft.minBpw());
    CHECK(bitsPerWord(fft, floorExp - 1) < fft.minBpw());
  }
}

TEST(regime_boundary_fixtures) {
  FFTConfig const fft = cfg(small());

  CHECK(regimeOf(fft, 10 * SMALL_SIZE - 1).longCarry);
  CHECK(!regimeOf(fft, 10 * SMALL_SIZE).longCarry);

  CHECK(!regimeOf(fft, 19 * SMALL_SIZE - 1).carry64);
  CHECK(regimeOf(fft, 19 * SMALL_SIZE).carry64);
}

// The done-when: an identity spanning a regime boundary yields two entries, not one.
TEST(spanning_a_boundary_yields_two_entries) {
  FFTConfig const fft = cfg(small());
  std::vector<Interval> const split = intervals(fft, 1'000'000, 4'000'000);

  CHECK_EQ(split.size(), size_t(2));
  CHECK(split[0] == (Interval{1'000'000, 10 * SMALL_SIZE - 1, {.longCarry = true}}));
  CHECK(split[1] == (Interval{10 * SMALL_SIZE, 4'000'000, {}}));
}

TEST(spanning_both_boundaries_yields_three_entries) {
  FFTConfig const fft = cfg(small());
  std::vector<Interval> const split = intervals(fft, 1, 6'000'000);

  CHECK_EQ(split.size(), size_t(3));
  CHECK(split[0] == (Interval{3 * SMALL_SIZE, 10 * SMALL_SIZE - 1, {.longCarry = true}}));
  CHECK(split[1] == (Interval{10 * SMALL_SIZE, 19 * SMALL_SIZE - 1, {}}));
  CHECK(split[2] == (Interval{19 * SMALL_SIZE, 6'000'000, {.carry64 = true}}));
}

// The done-when: the bottom of the range is excluded where it must be.
TEST(the_bottom_of_the_range_is_excluded) {
  FFTConfig const fft = cfg(small());

  CHECK_EQ(intervals(fft, 1, 3 * SMALL_SIZE - 1).size(), size_t(0));

  std::vector<Interval> const atTheFloor = intervals(fft, 1, 3 * SMALL_SIZE);
  CHECK_EQ(atTheFloor.size(), size_t(1));
  CHECK(atTheFloor.front() == (Interval{3 * SMALL_SIZE, 3 * SMALL_SIZE, {.longCarry = true}}));
  CHECK(interval(fft, 3 * SMALL_SIZE - 1).empty());

  // A shape whose floor is above the whole workload covers none of it.
  FFTConfig const big = cfg(FFTShape{FFT64, 4096, 16, 1024});
  CHECK(minExp(big) > 400'000'000);
  CHECK_EQ(intervals(big, 100'000'000, 400'000'000).size(), size_t(0));
}

TEST(a_pinned_carry_removes_the_carry_boundary) {
  FFTShape const shape{FFT64, 1024, 13, 256};

  FFTConfig const pinned64 = cfg(shape, CARRY_64);
  CHECK_EQ(pinned64.carry, CARRY_64);
  for (const Interval& part : intervals(pinned64, minExp(pinned64), maxExp(pinned64))) { CHECK(part.regime.carry64); }

  FFTConfig const pinned32 = cfg(shape, CARRY_32);
  CHECK_EQ(pinned32.carry, CARRY_32);
  for (const Interval& part : intervals(pinned32, minExp(pinned32), maxExp(pinned32))) { CHECK(!part.regime.carry64); }

  // The pinned 32-bit carry costs bits per word, so its entries stop below where the automatic carry's do.
  CHECK(maxExp(pinned32) < maxExp(cfg(shape)));
}

TEST(interval_is_the_run_containing_the_exponent) {
  FFTConfig const fft = cfg(small());
  u64 const reach = maxExp(fft);

  for (const Interval& part : intervals(fft, minExp(fft), reach)) {
    for (u64 E : {part.lo, part.lo + (part.hi - part.lo) / 2, part.hi}) {
      CHECK(interval(fft, E, reach) == part);
      CHECK_EQ(interval(fft, E, reach).regime.label(), regimeOf(fft, E).label());
    }
  }

  CHECK(interval(fft, reach + 1, reach).empty());
}

// Over every shape: the split covers its range exactly once, every cut is a real regime change, and no run holds two
// regimes.
TEST(the_split_partitions_every_shape) {
  for (const FFTShape& shape : FFTShape::allShapes()) {
    FFTConfig const fft = cfg(shape);
    u64 const lo = minExp(fft);
    u64 const hi = maxExp(fft);
    if (hi < lo) { continue; }

    std::vector<Interval> const split = intervals(fft, 1, hi);
    CHECK(!split.empty());
    CHECK_EQ(split.front().lo, lo);
    CHECK_EQ(split.back().hi, hi);

    for (size_t i = 0; i < split.size(); ++i) {
      CHECK(!split[i].empty());
      CHECK(regimeOf(fft, split[i].lo) == split[i].regime);
      CHECK(regimeOf(fft, split[i].hi) == split[i].regime);
      if (i > 0) {
        CHECK_EQ(split[i].lo, split[i - 1].hi + 1);
        CHECK(!(split[i].regime == split[i - 1].regime));
      }
    }
  }
}
