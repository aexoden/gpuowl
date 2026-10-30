// Copyright (C) Jason Lynch

#include "test.h"

#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"

#include <vector>

using namespace tune;

namespace {

RoeReading at(u64 E, double z, bool ok = true) { return {.exponent = E, .checkOk = ok, .z = z, .n = 9000}; }

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

}  // namespace

TEST(exact_arithmetic_needs_no_reading) {
  FFTConfig const ntt = canonical(FFTConfig{"3:1K:8:256:202"});
  Verdict const v = verdictOf(ntt, {}, 60'000'000);
  CHECK(v.state == Accuracy::Exact);
  CHECK_EQ(v.reach, maxExp(ntt));
}

TEST(a_set_is_read_at_the_table_top_first_and_confirmed_there) {
  FFTConfig const fft = canonical(FFTConfig{"2:512:4:512:212"});
  Verdict const owed = verdictOf(fft, {}, 60'000'000);
  CHECK(owed.state == Accuracy::Owed);
  CHECK(owed.next <= maxExp(fft) && owed.next > maxExp(fft) - 1000);

  std::vector<RoeReading> const readings{at(owed.next, 9.5)};
  Verdict const v = verdictOf(fft, readings, 60'000'000);
  CHECK(v.state == Accuracy::Confirmed);
  CHECK_EQ(v.reach, maxExp(fft));
}

TEST(an_unsafe_top_bisects_down_to_a_lowered_reach) {
  FFTConfig const fft = canonical(FFTConfig{"2:512:4:512:212"});
  u64 const floor = 60'000'000;
  std::vector<RoeReading> readings{at(verdictOf(fft, {}, floor).next, 4.0)};

  // A device whose rounding turns unsafe at 70M: each owed reading reads safe below it and unsafe above.
  for (int step = 0; step < 12; ++step) {
    Verdict const v = verdictOf(fft, readings, floor);
    if (v.state != Accuracy::Owed) { break; }
    CHECK(v.next > floor && v.next < readings.front().exponent);
    readings.push_back(at(v.next, v.next < 70'000'000 ? 8.0 : 4.0));
  }
  Verdict const v = verdictOf(fft, readings, floor);
  CHECK(v.state == Accuracy::Lowered);
  CHECK(v.reach < 70'000'000 && v.reach > 69'000'000);
  CHECK(readings.size() <= 6);
}

TEST(a_reach_below_the_workload_rejects_the_set) {
  FFTConfig const fft = canonical(FFTConfig{"2:512:4:512:212"});
  std::vector<RoeReading> const readings{at(maxExp(fft) - 100, 4.0), at(59'000'000, 4.0)};
  CHECK(verdictOf(fft, readings, 60'000'000).state == Accuracy::Rejected);
}

TEST(a_failed_check_is_unsafe_whatever_z_says) {
  CHECK(!safe(FFT64, at(1, 30, false)));
  CHECK(safe(FFT64, at(1, 20)));
  CHECK(!safe(FFT64, at(1, 19.9)));
  CHECK(safe(FFT3261, at(1, 6)));
}

TEST(a_set_that_changes_the_rounding_is_held_to_the_same_set_at_default_rounding) {
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  UseConfig const moved{{"MM2_CHAIN", "1"}, {"WMUL", "1"}};
  CHECK(movesAccuracy(nvidia(), fft, moved));
  CHECK(!movesAccuracy(nvidia(), fft, {{"WMUL", "1"}}));
  CHECK(accuracyReference(nvidia(), fft, moved) == canonicalConfig(nvidia(), fft, {{"WMUL", "1"}}));
  // At middle variant 1, MM2_CHAIN=2 is the default's rounding, so it moves nothing.
  CHECK(!movesAccuracy(nvidia(), canonical(FFTConfig{"1K:13:256:212"}), {{"MM2_CHAIN", "2"}}));
}

TEST(a_set_held_to_its_reference_is_judged_against_its_reading_at_the_table_top) {
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  u64 const floor = 60'000'000;
  std::vector<RoeReading> reference;

  // Its own reading first, at the table top, then its reference's there.
  Verdict const first = verdictOf(fft, {}, floor, &reference);
  CHECK(first.state == Accuracy::Owed && !first.owesReference);
  u64 const top = first.next;
  std::vector<RoeReading> own{at(top, 23.6)};
  Verdict const second = verdictOf(fft, own, floor, &reference);
  CHECK(second.state == Accuracy::Owed && second.owesReference);
  CHECK_EQ(second.next, top);

  // Within ACCURACY_SLACK of the reference: the table's reach.
  reference.push_back(at(top, 24.0));
  CHECK(verdictOf(fft, own, floor, &reference).state == Accuracy::Confirmed);

  // Further below it, though well above the floor: the reach is bisected down, its own readings owed.
  own = {at(top, 22.2)};
  Verdict const cut = verdictOf(fft, own, floor, &reference);
  CHECK(cut.state == Accuracy::Owed && !cut.owesReference);
  CHECK(cut.next > floor && cut.next < top);
  CHECK(verdictOf(fft, own, floor).state == Accuracy::Confirmed);  // a set held to the floor alone passes there

  // Lower down, as accurate as the reference is at the top: it serves there, with no reading of the reference owed
  // beside it, and a reference reading taken there changes nothing.
  own.push_back(at(cut.next, 23.0));
  Verdict const lower = verdictOf(fft, own, floor, &reference);
  CHECK(!lower.owesReference && lower.reach == cut.next);
  CHECK(lower.state == Accuracy::Owed || lower.state == Accuracy::Lowered);
  reference.push_back(at(cut.next, 40.0));
  Verdict const again = verdictOf(fft, own, floor, &reference);
  CHECK(again.state == lower.state && again.reach == lower.reach && again.next == lower.next);
}

TEST(a_set_is_held_within_the_slack_of_its_reference_and_never_below_the_floor) {
  RoeReading const own = at(1, 25);
  RoeReading const ref = at(1, 24);
  CHECK(safe(FFT64, own, &ref));
  // 7% below 24 is 22.32.
  CHECK(safe(FFT64, at(1, 22.4), &ref));
  CHECK(!safe(FFT64, at(1, 22.2), &ref));
  // The slack is a share of z: as wide in Z again at twice the z.
  RoeReading const high = at(1, 48);
  CHECK(safe(FFT64, at(1, 44.7), &high));
  CHECK(!safe(FFT64, at(1, 44.6), &high));
  // Where the reference reads below the floor, or failed its check, the floor alone holds.
  RoeReading const unsafeRef = at(1, 19);
  CHECK(safe(FFT64, own, &unsafeRef));
  CHECK(!safe(FFT64, at(1, 19.9), &unsafeRef));
  RoeReading const failedRef = at(1, 24, false);
  CHECK(safe(FFT64, at(1, 20.1), &failedRef));
  CHECK(!safe(FFT64, at(1, 19.9), &failedRef));
  // Never where the reference hardly rounds while the set rounds visibly.
  RoeReading quietRef = at(1, 24);
  quietRef.n = 2;
  CHECK(!safe(FFT64, own, &quietRef));
  // The floor still stands below a reference that barely clears it.
  RoeReading const nearFloor = at(1, 20.2);
  CHECK(!safe(FFT64, at(1, 19.9), &nearFloor));
  // A set that hardly rounds is safe beside anything.
  RoeReading quiet = at(1, 0);
  quiet.n = 1;
  CHECK(safe(FFT64, quiet, &unsafeRef));
}

TEST(sets_that_round_alike_have_one_rounding) {
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  // Keys that do not change the rounding drop out, and a value is spelled as the one whose arithmetic it compiles.
  CHECK(roundingOf(nvidia(), fft, {{"ZEROHACK_H", "0"}, {"WMUL", "1"}}).empty());
  CHECK((roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "11"}, {"WMUL", "1"}}) == UseConfig{{"TAIL_KERNELS", "1"}}));
  CHECK(roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "1"}}) == roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "11"}}));
  CHECK(roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "0"}}) != roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "1"}}));
  // 13 compiles as 3, whose arithmetic is the default's.
  CHECK(roundingOf(nvidia(), fft, {{"TAIL_KERNELS", "13"}}).empty());

  // Shared where the flags were measured: nVidia, under CUDA and OpenCL alike.
  CHECK(sharesRounding(nvidia()));
  CHECK(sharesRounding(Env{.isNvidia = true}));
  CHECK(!sharesRounding(Env{.isAmd = true}));
  CHECK(!sharesRounding(Env{}));
}
