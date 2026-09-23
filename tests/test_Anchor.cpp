// Copyright (C) Jason Lynch

// GPU-free tests of the drift anchor (src/Anchor.cpp).

#include "test.h"

#include "Anchor.h"
#include "Eligibility.h"
#include "FFTVariants.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace tune;

TEST(drift_levels_are_symmetric) {
  CHECK(driftLevelOf(1) == DriftLevel::Steady);
  CHECK(driftLevelOf(1.01) == DriftLevel::Steady);
  CHECK(driftLevelOf(0.99) == DriftLevel::Steady);

  CHECK(driftLevelOf(1 + DRIFT_WARN) == DriftLevel::Warn);
  CHECK(driftLevelOf(1 - DRIFT_WARN) == DriftLevel::Warn);
  CHECK(driftLevelOf(1.05) == DriftLevel::Warn);

  CHECK(driftLevelOf(1 + DRIFT_ALARM) == DriftLevel::Alarm);
  CHECK(driftLevelOf(1 - DRIFT_ALARM - 1e-9) == DriftLevel::Alarm);
  CHECK(driftLevelOf(2) == DriftLevel::Alarm);

  // No reading is no drift, not an infinite one.
  CHECK(driftLevelOf(0) == DriftLevel::Steady);
  CHECK(driftLevelOf(-1) == DriftLevel::Steady);
}

TEST(anchor_spec_round_trips) {
  auto const parsed = parseAnchorSpec("1:512:8:512:202@118063003");
  CHECK(parsed.has_value());
  CHECK_EQ(parsed->exponent, u64(118'063'003));
  CHECK_EQ(parsed->text(), std::string{"1:512:8:512:202@118063003"});

  // The spec half is canonicalized, so two spellings of one configuration pin one anchor.
  auto const spelled = parseAnchorSpec("1:512:8:512@118063003");
  CHECK(spelled.has_value());
  CHECK_EQ(spelled->fft, FFTConfig{"1:512:8:512"}.spec());
  CHECK_EQ(parseAnchorSpec(spelled->text())->text(), spelled->text());
}

TEST(anchor_spec_refuses_what_is_not_one) {
  CHECK(!parseAnchorSpec(""));
  CHECK(!parseAnchorSpec("-"));
  CHECK(!parseAnchorSpec("512:8:512"));    // no exponent
  CHECK(!parseAnchorSpec("@118063003"));   // no configuration
  CHECK(!parseAnchorSpec("512:8:512@"));   // an empty exponent
  CHECK(!parseAnchorSpec("512:8:512@0"));  // and one that is not an exponent
  CHECK(!parseAnchorSpec("512:8:512@12x"));
  CHECK(!parseAnchorSpec("not-an-fft@118063003"));
}

TEST(anchor_candidates_are_the_smallest_eligible_shape_of_each_type) {
  u64 const E = 118'063'003;
  std::vector<AnchorSpec> const candidates = anchorCandidates(E);

  // Every type production chooses among holds 118M somewhere, so each is raced, FP64 first.
  CHECK_EQ(candidates.size(), size_t(6));
  CHECK(FFTConfig{candidates.front().fft}.shape.fft_type == FFT64);

  std::set<int> types;
  for (const AnchorSpec& c : candidates) {
    CHECK_EQ(c.exponent, E);
    FFTConfig const fft{c.fft};
    CHECK(types.insert(fft.shape.fft_type).second);
    CHECK(!interval(fft, E).empty());
    CHECK_EQ(fft.variant, defaultVariant(fft.shape));
    CHECK(fft.carry == CARRY_AUTO);

    // Nothing of the same type and eligible there is smaller.
    for (const FFTShape& shape : FFTShape::allShapes()) {
      if (shape.fft_type != fft.shape.fft_type || shape.size() >= fft.size()) { continue; }
      CHECK(interval(FFTConfig{shape, defaultVariant(shape), CARRY_AUTO}, E).empty());
    }
  }
}

TEST(anchor_candidates_grow_with_the_exponent) {
  // A larger probe needs at least as large a shape of each type, and the choice is deterministic.
  u64 const small = 10'000'019;
  u64 const large = 400'000'009;
  std::vector<AnchorSpec> const a = anchorCandidates(small);
  std::vector<AnchorSpec> const b = anchorCandidates(large);
  CHECK(!a.empty());
  CHECK(!b.empty());
  CHECK(FFTConfig{a.front().fft}.size() <= FFTConfig{b.front().fft}.size());
  CHECK(anchorCandidates(small) == a);

  // An exponent nothing can hold -- too large for the largest shape, or below every bits-per-word floor -- has no
  // candidates, and neither has no exponent at all.
  CHECK(anchorCandidates(0).empty());
  CHECK(anchorCandidates(1).empty());
  CHECK(anchorCandidates(u64(1) << 62).empty());
}

TEST(the_race_anchors_on_the_cheapest_reading) {
  AnchorSpec const fp64{.fft = "1K:13:256:212", .exponent = 118'063'003};
  AnchorSpec const hybrid{.fft = "51:1K:8:256:202", .exponent = 118'063'003};
  AnchorSpec const ntt{.fft = "3:1K:8:512:202", .exponent = 118'063'003};

  // A card slow at FP64 is anchored on something it is good at, and one good at FP64 on FP64.
  CHECK(raceWinner({{fp64, 8088}, {hybrid, 5173}, {ntt, 2250}}) == ntt);
  CHECK(raceWinner({{fp64, 957}, {hybrid, 1284}, {ntt, 3450}}) == fp64);

  // A tie goes to the earlier candidate, and a candidate with no reading is not in the race.
  CHECK(raceWinner({{fp64, 1000}, {hybrid, 1000}}) == fp64);
  CHECK(raceWinner({{fp64, 0}, {hybrid, 1200}}) == hybrid);
  CHECK(!raceWinner({{fp64, 0}}).has_value());
  CHECK(!raceWinner({}).has_value());
}

TEST(anchor_state_establishes_its_own_baseline) {
  AnchorState state;
  CHECK(state.observe(100) == DriftLevel::Steady);
  CHECK_EQ(state.baseline, 100.0);
  CHECK_EQ(state.ratio, 1.0);
  CHECK_EQ(state.readings, 1u);

  CHECK(state.observe(101) == DriftLevel::Steady);
  CHECK_EQ(state.baseline, 100.0);
  CHECK_EQ(state.latest, 101.0);
  CHECK_EQ(state.ratio, 1.01);
  CHECK_EQ(state.readings, 2u);
}

TEST(anchor_state_divides_by_an_inherited_baseline) {
  // What a second session does: the env's first reading is already known, so the very first reading of this session
  // carries the movement between the two.
  AnchorState state{.baseline = 100};
  CHECK(state.observe(103) == DriftLevel::Warn);
  CHECK_EQ(state.ratio, 1.03);
  CHECK(state.observe(115) == DriftLevel::Alarm);
  CHECK(std::fabs(state.ratio - 1.15) < 1e-12);
  CHECK_EQ(state.baseline, 100.0);
}

TEST(anchor_state_ignores_a_reading_that_is_not_one) {
  AnchorState state{.baseline = 100};
  CHECK(state.observe(0) == DriftLevel::Steady);
  CHECK(state.observe(-5) == DriftLevel::Steady);
  CHECK_EQ(state.readings, 0u);
  CHECK_EQ(state.ratio, 1.0);

  // A failed reading after a good one leaves the good one standing, which is what the rows then carry.
  CHECK(state.observe(112) == DriftLevel::Alarm);
  CHECK(state.observe(0) == DriftLevel::Alarm);
  CHECK_EQ(state.latest, 112.0);
  CHECK_EQ(state.readings, 1u);
}
