// Copyright (C) Jason Lynch

// Tests the reach derivation as a pure function of its readings: the starting slope and the fitted one, the cap on
// extrapolation, confirmation in either direction, the guard-band backoff, and each state a derivation ends in.

#include "Reach.h"

#include "Eligibility.h"
#include "Primes.h"

#include "test.h"

#include <cmath>
#include <limits>
#include <map>
#include <optional>

using namespace tune;

namespace {

// 2:512:8:512 at its table reach when the readings below were taken, 132791664 (31.66 bits per word).
constexpr u64 WORDS = 4 * 1024 * 1024;
constexpr u64 TOP = 132'791'657;
constexpr u64 LO = 12'582'912;

double bpwOf(u64 E) { return double(E) / double(WORDS); }

// A set whose z rises by one for every `slope` bits per word below TOP, where it reads `z`.
struct Model {
  double z = 0;
  double slope = 0.015;
  bool checkOk = true;

  [[nodiscard]] double at(u64 E) const { return z + (bpwOf(TOP) - bpwOf(E)) / slope; }
};

// The readings a derivation has asked for so far, as the database would hold them.
struct Readings {
  std::map<u64, ZReading> held;

  [[nodiscard]] ReadingAt lookup() const {
    return [this](u64 E) -> std::optional<ZReading> {
      auto const it = held.find(E);
      return it != held.end() ? std::optional{it->second} : std::nullopt;
    };
  }

  void take(u64 E, double z, bool checkOk = true, u32 n = 2000) {
    held[E] = {.exponent = E, .z = z, .n = n, .checkOk = checkOk};
  }
};

ReachProblem problem(Standard standard, u64 hi = TOP) {
  return {.words = WORDS, .lo = LO, .hi = hi, .standard = standard};
}

constexpr Standard FITTED{.aim = 28, .bar = 28};

// Runs a derivation to its end against `model`, taking every reading it owes; the outcome and the readings taken.
std::pair<ReachOutcome, Readings> settle(const ReachProblem& p, const Model& model, u64 from = TOP) {
  Readings r;
  r.take(from, model.at(from), model.checkOk);
  ReachOutcome out = deriveReach(p, r.held.at(from), r.lookup());
  for (u32 n = 0; out.state == ReachState::Owed && n < 30; ++n) {
    CHECK(!r.held.contains(out.exponent));
    r.take(out.exponent, model.at(out.exponent), model.checkOk);
    out = deriveReach(p, r.held.at(from), r.lookup());
  }
  return {out, r};
}

bool isPrime(u64 E) {
  static const Primes primes;
  return primes.isPrime(E);
}

}  // namespace

TEST(the_starting_slope_is_ztunes_guess_and_stays_in_its_clamp) {
  CHECK(std::abs(startSlope(4 * 1024 * 1024) - 0.015) < 1e-12);
  CHECK(std::abs(startSlope(7.5 * 1024 * 1024) - 0.012) < 1e-12);
  CHECK(startSlope(6 * 1024 * 1024) < 0.015 && startSlope(6 * 1024 * 1024) > 0.012);

  // Continued past both ends of what it was measured over, and clamped.
  CHECK_EQ(startSlope(256 * 1024), MAX_SLOPE);
  CHECK_EQ(startSlope(u64(1) << 40), MIN_SLOPE);
}

TEST(a_slope_is_fitted_from_two_readings_and_clamped) {
  ZReading const a{.exponent = TOP, .z = 20, .n = 2000, .checkOk = true};
  ZReading const b{.exponent = TOP - WORDS / 10, .z = 26, .n = 2000, .checkOk = true};
  CHECK(std::abs(fitSlope(a, b, WORDS) - 0.1 / 6) < 1e-6);
  CHECK_EQ(fitSlope(a, b, WORDS), fitSlope(b, a, WORDS));

  // A z that rises with the bits per word is noise, and is clamped rather than followed.
  CHECK_EQ(fitSlope(a, {.exponent = b.exponent, .z = 19, .n = 2000, .checkOk = true}, WORDS), MIN_SLOPE);
  CHECK_EQ(fitSlope(a, {.exponent = b.exponent, .z = 1000, .n = 2000, .checkOk = true}, WORDS), MIN_SLOPE);
  CHECK_EQ(fitSlope(a, {.exponent = b.exponent, .z = 20.5, .n = 2000, .checkOk = true}, WORDS), MAX_SLOPE);

  // Nothing to fit: one exponent, one z, or a reading with no z.
  CHECK_EQ(fitSlope(a, a, WORDS), startSlope(WORDS));
  CHECK_EQ(fitSlope(a, {.exponent = b.exponent, .z = 20, .n = 2000, .checkOk = true}, WORDS), startSlope(WORDS));
  CHECK_EQ(fitSlope(a, {.exponent = b.exponent, .z = 0, .n = 2, .checkOk = true}, WORDS), startSlope(WORDS));
}

TEST(a_reading_within_the_cap_is_extrapolated_and_confirmed_where_it_points) {
  Readings r;
  r.take(TOP, 25);
  ReachOutcome const owed = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(owed.state == ReachState::Owed);

  // Three units short at -ztune's slope for 4M words: 0.045 bits per word down, at the prime there.
  u64 const expected = u64((bpwOf(TOP) - 3 * 0.015) * WORDS);
  CHECK(owed.exponent <= expected && owed.exponent > expected - 1000);
  CHECK(isPrime(owed.exponent));

  // The proposal is confirmed by a reading there, not by the extrapolation.
  r.take(owed.exponent, 28.1);
  ReachOutcome const done = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(done.state == ReachState::Confirmed);
  CHECK_EQ(done.exponent, owed.exponent);
}

TEST(a_reading_beyond_the_cap_is_read_again_nearer_before_anything_is_proposed) {
  Readings r;
  r.take(TOP, 6.1, false);
  ReachOutcome const first = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(first.state == ReachState::Owed);

  // Read where the extrapolation points, and short of the aim by more than the cap again: that reading is not a
  // confirmation to back off from but the next point to extrapolate from, along the slope the two now give.
  r.take(first.exponent, 20);
  ReachOutcome const second = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(second.state == ReachState::Owed);
  double const fitted = (bpwOf(TOP) - bpwOf(first.exponent)) / (20 - 6.1);
  u64 const expected = u64((bpwOf(first.exponent) - 8 * fitted) * WORDS);
  CHECK(second.exponent <= expected && second.exponent > expected - 1000);

  // Nor is a reading that clears the bar taken as the reach while it is beyond the cap: there is further to go.
  Readings high;
  high.take(TOP, 33);
  ReachOutcome const up = deriveReach(problem(FITTED, TOP + WORDS), high.held.at(TOP), high.lookup());
  CHECK(up.state == ReachState::Owed);
  CHECK(up.exponent > TOP);
  high.take(up.exponent, 33);
  ReachOutcome const further = deriveReach(problem(FITTED, TOP + WORDS), high.held.at(TOP), high.lookup());
  CHECK(further.state == ReachState::Owed);
  CHECK(further.exponent > up.exponent);
}

TEST(a_derivation_down_settles_where_the_set_reads_its_standard) {
  for (double slope : {0.008, 0.015, 0.021}) {
    Model const model{.z = 6.1, .slope = slope, .checkOk = true};
    auto const [out, readings] = settle(problem(FITTED), model);
    CHECK(out.state == ReachState::Confirmed);
    CHECK(readings.held.contains(out.exponent));
    CHECK(isPrime(out.exponent));
    CHECK(model.at(out.exponent) >= 28);

    // No more than a guard band under where the model reads exactly 28.
    double const at28 = bpwOf(TOP) - (28 - 6.1) * slope;
    CHECK(bpwOf(out.exponent) <= at28 + 1e-9);
    CHECK(bpwOf(out.exponent) > at28 - REACH_GUARD_BPW);
    CHECK(readings.held.size() <= 5);
  }
}

TEST(a_derivation_up_is_confirmed_as_one_down_is) {
  // Increases are confirmed exactly as reductions are: nothing is inherited in either direction.
  u64 const hi = TOP + WORDS / 2;
  Model const model{.z = 31, .slope = 0.015, .checkOk = true};
  auto const [out, readings] = settle(problem(FITTED, hi), model);
  CHECK(out.state == ReachState::Confirmed);
  CHECK(out.exponent > TOP);
  CHECK(readings.held.contains(out.exponent));
  CHECK(model.at(out.exponent) >= 28);
  CHECK(bpwOf(out.exponent) > bpwOf(TOP) + 3 * 0.015 - REACH_GUARD_BPW);

  // Never past where the carry can go, however accurate the set is there.
  Model const accurate{.z = 70, .slope = 0.015, .checkOk = true};
  auto const [capped, _] = settle(problem(FITTED, hi), accurate);
  CHECK(capped.state == ReachState::Confirmed);
  CHECK(capped.exponent <= hi);
  CHECK(capped.exponent > hi - 1000);
}

TEST(a_reach_that_fails_its_confirmation_backs_off_a_guard_band_at_a_time) {
  Readings r;
  r.take(TOP, 25);
  ReachOutcome const proposed = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(proposed.state == ReachState::Owed);

  // Read short of the bar at the proposal, then at the first band under it; confirmed at the second.
  r.take(proposed.exponent, 27.9);
  ReachOutcome const band1 = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(band1.state == ReachState::Owed);
  CHECK(std::abs(bpwOf(proposed.exponent) - REACH_GUARD_BPW - bpwOf(band1.exponent)) < 1e-4);

  r.take(band1.exponent, 27.95);
  ReachOutcome const band2 = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(band2.state == ReachState::Owed);
  CHECK(std::abs(bpwOf(proposed.exponent) - 2 * REACH_GUARD_BPW - bpwOf(band2.exponent)) < 1e-4);

  Readings confirmed = r;
  confirmed.take(band2.exponent, 30);
  ReachOutcome const done = deriveReach(problem(FITTED), confirmed.held.at(TOP), confirmed.lookup());
  CHECK(done.state == ReachState::Confirmed);
  CHECK_EQ(done.exponent, band2.exponent);

  // Short there too, after REACH_TRIES bands, the set is rejected; and a failed check is short whatever its z.
  for (auto [z, ok] : {std::pair{27.99, true}, std::pair{40.0, false}}) {
    Readings rejected = r;
    rejected.take(band2.exponent, z, ok);
    ReachOutcome const out = deriveReach(problem(FITTED), rejected.held.at(TOP), rejected.lookup());
    CHECK(out.state == ReachState::Rejected);
    CHECK_EQ(out.why, std::string{"it read below 28.00 at its proposed reach and at each guard band under it"});
  }
}

TEST(a_derivation_that_leaves_its_interval_is_rejected) {
  // Every exponent of the interval reads short: the backoff stops at the bottom rather than proposing under it.
  u64 const lo = TOP - WORDS / 20;
  ReachProblem const narrow{.words = WORDS, .lo = lo, .hi = TOP, .standard = FITTED};
  auto const [out, readings] = settle(narrow, Model{.z = 10, .slope = 0.015, .checkOk = true});
  CHECK(out.state == ReachState::Rejected);
  for (const auto& [E, reading] : readings.held) { CHECK(E >= lo && E <= TOP); }
}

TEST(a_derivation_that_does_not_converge_is_rejected) {
  // z that does not move with the bits per word is not something a reach can be derived from.
  auto const [out, readings] = settle(problem(FITTED), Model{.z = 10, .slope = 1e9, .checkOk = true});
  CHECK(out.state == ReachState::Rejected);
  CHECK_EQ(readings.held.size(), size_t{MAX_DERIVE_READINGS});
  CHECK(out.why.starts_with("its z did not come within reach of 28.00"));
}

TEST(too_few_rounding_errors_on_the_way_down_confirm_the_reach_there) {
  Readings r;
  r.take(TOP, 10);
  ReachOutcome const owed = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(owed.state == ReachState::Owed);

  r.take(owed.exponent, 0, true, 1);
  ReachOutcome const done = deriveReach(problem(FITTED), r.held.at(TOP), r.lookup());
  CHECK(done.state == ReachState::Confirmed);
  CHECK_EQ(done.exponent, owed.exponent);
}

TEST(a_relative_standard_aims_at_its_reference_and_confirms_at_the_bar) {
  // A set 2.1 z under its reference's 24.44 aims at 24.44 and is confirmed anywhere it reads 23.94.
  Standard const relative{.aim = 24.44, .bar = 23.94};
  Readings r;
  r.take(TOP, 22.34);
  ReachOutcome const owed = deriveReach(problem(relative), r.held.at(TOP), r.lookup());
  CHECK(owed.state == ReachState::Owed);
  CHECK(bpwOf(owed.exponent) < bpwOf(TOP) - 2.1 * 0.015 + 1e-6);

  r.take(owed.exponent, 24.0);
  CHECK(deriveReach(problem(relative), r.held.at(TOP), r.lookup()).state == ReachState::Confirmed);
}

TEST(a_reading_above_the_interval_derives_a_reach_inside_it) {
  // The gate counts a reading above the top of the interval; a derivation from it proposes nothing above the top.
  Readings r;
  u64 const above = TOP + WORDS / 100;
  r.take(above, 27.5);
  ReachOutcome const owed = deriveReach(problem(FITTED), r.held.at(above), r.lookup());
  CHECK(owed.state == ReachState::Owed);
  CHECK(owed.exponent <= TOP);
}

TEST(a_pinned_32_bit_carry_is_the_only_ceiling_on_a_reach) {
  CHECK_EQ(carryCeiling(FFTConfig{"512:15:512:212"}), std::numeric_limits<u64>::max());
  CHECK_EQ(carryCeiling(FFTConfig{"512:15:512:212:1"}), std::numeric_limits<u64>::max());

  // A pin the carry folds away where it changes nothing (canonicalCarry()), so a shape where it does.
  std::optional<FFTConfig> found;
  for (const char* spec : {"512:15:512:212:0", "1K:8:1K:202:0", "1K:16:1K:202:0", "4K:16:1K:202:0"}) {
    if (FFTConfig const c{spec}; !found && c.carry == CARRY_32) { found = c; }
  }
  CHECK(found.has_value());
  if (!found) { return; }
  FFTConfig const& pinned = *found;
  u64 const ceiling = carryCeiling(pinned);
  CHECK_EQ(ceiling, u64(double(pinned.shape.carry32BPW()) * double(pinned.size())));
  CHECK(ceiling < 19 * pinned.size());

  // Where carry32BPW() reaches its hard limit, the ceiling stays under 19 bits per word.
  FFTConfig const small{"256:2:256:202:0"};
  CHECK_EQ(small.shape.carry32BPW(), 19.0f);
  CHECK_EQ(carryCeiling(small), 19 * small.size() - 1);

  // A hybrid's carry is not the kernel CARRY_32 pins.
  CHECK_EQ(carryCeiling(FFTConfig{"2:512:8:512:202:0"}), std::numeric_limits<u64>::max());
}

TEST(a_reading_taken_on_the_way_that_passes_is_preferred_to_a_band_under_it) {
  // Read on a Tesla P100, OpenCL, 2:512:8:512:212 at default options, in the order the derivation asked for them.
  Readings r;
  r.take(132'791'657, 6.136, false);
  auto next = [&] { return deriveReach(problem(FITTED), r.held.at(TOP), r.lookup()); };

  CHECK_EQ(next().exponent, u64(131'416'079));
  r.take(131'416'079, 20.83);
  CHECK_EQ(next().exponent, u64(130'744'829));
  r.take(130'744'829, 28.94);

  // Clear of 28 by less than the cap, so extrapolated a little further up, where z scattered a unit low.
  CHECK_EQ(next().exponent, u64(130'822'607));
  r.take(130'822'607, 27.61);

  // The reading under it that passed is the reach, rather than a guard band under that.
  ReachOutcome const done = next();
  CHECK(done.state == ReachState::Confirmed);
  CHECK_EQ(done.exponent, u64(130'744'829));
}

TEST(a_reach_is_raised_no_further_than_the_table_allows_it) {
  // 512:15:512:212's automatic carry changes to 64 bits a little past its table reach, and the raise stops there.
  FFTConfig const shortTop{"512:15:512:212"};
  u64 const ceiling = raiseCeiling(shortTop);
  CHECK(ceiling > maxExp(shortTop));
  CHECK(regimeOf(shortTop, ceiling) == regimeOf(shortTop, maxExp(shortTop)));
  CHECK(regimeOf(shortTop, ceiling + 1) != regimeOf(shortTop, maxExp(shortTop)));

  // Where the regime runs on, MAX_RAISE_BPW past the table.
  FFTConfig const wideTop{"1K:8:1K:202"};
  CHECK(regimeOf(wideTop, maxExp(wideTop)).carry64);
  CHECK_EQ(raiseCeiling(wideTop), u64((double(wideTop.maxBpw()) + MAX_RAISE_BPW) * double(wideTop.size())));

  // A pinned 32-bit carry has no more room than the carry gives it.
  for (const char* spec : {"512:15:512:212:0", "1K:8:1K:202:0", "1K:16:1K:202:0", "4K:16:1K:202:0"}) {
    if (FFTConfig const pinned{spec}; pinned.carry == CARRY_32) {
      CHECK(raiseCeiling(pinned) <= std::max(carryCeiling(pinned), maxExp(pinned)));
    }
  }
}
