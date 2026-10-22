// Copyright (C) Jason Lynch

#include "test.h"

#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"

#include <vector>

using namespace tune;

namespace {

RoeReading at(u64 E, double z, bool ok = true) { return {.exponent = E, .checkOk = ok, .z = z, .n = 9000}; }

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
