// Copyright (C) Jason Lynch

// GPU-free tests of the drift anchor (src/Anchor.cpp).

#include "test.h"

#include "Anchor.h"
#include "Eligibility.h"
#include "FFTVariants.h"

#include <cmath>
#include <string>

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

TEST(anchor_is_the_smallest_eligible_fp64_shape) {
  u64 const E = 118'063'003;
  auto const chosen = chooseAnchor(E);
  CHECK(chosen.has_value());
  CHECK_EQ(chosen->exponent, E);

  FFTConfig const fft{chosen->fft};
  CHECK(fft.shape.fft_type == FFT64);
  CHECK(!interval(fft, E).empty());
  CHECK_EQ(fft.variant, defaultVariant(fft.shape));
  CHECK(fft.carry == CARRY_AUTO);

  // Nothing FP64 and eligible there is smaller.
  for (const FFTShape& shape : FFTShape::allShapes()) {
    if (shape.fft_type != FFT64 || shape.size() >= fft.size()) { continue; }
    CHECK(interval(FFTConfig{shape, defaultVariant(shape), CARRY_AUTO}, E).empty());
  }
}

TEST(anchor_grows_with_the_exponent) {
  // A larger probe needs at least as large an anchor, and the choice is deterministic.
  u64 const small = 10'000'019;
  u64 const large = 400'000'009;
  auto const a = chooseAnchor(small);
  auto const b = chooseAnchor(large);
  CHECK(a.has_value());
  CHECK(b.has_value());
  CHECK(FFTConfig{a->fft}.size() <= FFTConfig{b->fft}.size());
  CHECK_EQ(chooseAnchor(small)->text(), a->text());

  // An exponent no FP64 shape can hold -- too large for the largest, or below the bits-per-word floor of the smallest
  // -- has no anchor, and neither has no exponent at all.
  CHECK(!chooseAnchor(0));
  CHECK(!chooseAnchor(1));
  CHECK(!chooseAnchor(u64(1) << 62));
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
