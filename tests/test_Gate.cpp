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

TEST(a_set_held_to_its_reference_is_judged_beside_its_reading) {
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  u64 const floor = 60'000'000;
  std::vector<RoeReading> reference;

  // Its own reading first, at the table top, then its reference's at the same exponent.
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

  // Further below it, though well above the floor: the reach is bisected down, its own reading owed first.
  own = {at(top, 22.2)};
  Verdict const cut = verdictOf(fft, own, floor, &reference);
  CHECK(cut.state == Accuracy::Owed && !cut.owesReference);
  CHECK(cut.next > floor && cut.next < top);
  CHECK(verdictOf(fft, own, floor).state == Accuracy::Confirmed);  // a set held to the floor alone passes there
}

TEST(a_set_is_never_safer_than_its_reference) {
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
  // Not where the reference itself is unsafe, failed its check, or hardly rounds while the set rounds visibly.
  RoeReading const unsafeRef = at(1, 19);
  CHECK(!safe(FFT64, own, &unsafeRef));
  RoeReading const failedRef = at(1, 24, false);
  CHECK(!safe(FFT64, own, &failedRef));
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
