// Copyright (C) Jason Lynch

// GPU-free tests of the parts of src/Measure.cpp that do not touch a device: folding a call's
// blocks into a row, the accuracy floor, and the verdict on a rounding-error reading.

#include "test.h"

#include "Gpu.h"
#include "Measure.h"

#include <cmath>
#include <vector>

using namespace tune;

namespace {

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

IterSamples samplesOf(std::vector<double> usPerIt, bool checkOk = true) {
  return {.usPerIt = std::move(usPerIt), .checkOk = checkOk, .res64 = 0x5685d572fcaebcae, .iters = 5000};
}

}  // namespace

TEST(summarize_pools_the_blocks_of_one_call) {
  Call const c = summarize(samplesOf({1000, 1002, 998, 1000}));

  CHECK_EQ(c.measurement.blocks, 4u);
  CHECK_EQ(c.measurement.calls, 1u);
  CHECK_EQ(c.dropped, 0u);
  CHECK(!c.declined);
  CHECK(near(c.measurement.mean, 1000));
  CHECK(c.measurement.ok());
  CHECK_EQ(c.res64, u64{0x5685d572fcaebcae});
  CHECK_EQ(c.usPerIt.size(), size_t{4});
}

TEST(summarize_drops_an_upward_spike) {
  // One block 10% above the others: the spike sets the whole spread if it is kept.
  Call const c = summarize(samplesOf({1000, 1001, 1100, 999, 1000, 1000, 1001, 999}));

  CHECK_EQ(c.dropped, 1u);
  CHECK_EQ(c.measurement.blocks, 7u);
  CHECK(!c.declined);
  CHECK(c.measurement.mean < 1001);

  // Every block is reported, spikes included: the rejection is a statistic, not a deletion.
  CHECK_EQ(c.usPerIt.size(), size_t{8});
}

TEST(summarize_declines_when_the_core_is_not_a_core) {
  // Two of five blocks above the cut: "core plus spikes" does not describe this, so nothing is
  // dropped and the row keeps the spread it really has.
  Call const c = summarize(samplesOf({1000, 1000, 1000, 1100, 1100}));

  CHECK(c.declined);
  CHECK_EQ(c.dropped, 0u);
  CHECK_EQ(c.measurement.blocks, 5u);
  CHECK(near(c.measurement.mean, 1040));
}

TEST(summarize_fails_the_row_on_a_failed_gerbicz_check) {
  Call const c = summarize(samplesOf({1000, 1000, 1000, 1000}, false));

  CHECK(!c.checkOk);
  CHECK(c.measurement.status == Status::Err);
  CHECK(!c.measurement.ok());

  // The samples are still plausible, which is the point: only the check says they are worthless.
  CHECK(near(c.measurement.mean, 1000));
}

TEST(summarize_of_no_blocks_is_not_a_row) {
  Call const c = summarize(samplesOf({}));

  CHECK_EQ(c.measurement.blocks, 0u);
  CHECK_EQ(c.measurement.calls, 0u);
  CHECK(!c.measurement.ok());
}

TEST(min_safe_z_is_stricter_for_fp64) {
  CHECK(near(minSafeZ(FFT64), 20));
  for (enum FFT_TYPES type : {FFT32, FFT31, FFT61, FFT3161, FFT3261, FFT323161}) { CHECK(near(minSafeZ(type), 6)); }
}

TEST(roe_verdict_needs_evidence_to_reject) {
  RoeCheck const notApplicable{.applicable = false, .z = 0, .n = 0, .minZ = 6};
  CHECK(notApplicable.passed());
  CHECK(!notApplicable.conclusive());

  RoeCheck const good{.applicable = true, .z = 31.5, .n = 400, .minZ = 20};
  CHECK(good.passed());
  CHECK(good.conclusive());

  RoeCheck const tooLow{.applicable = true, .z = 12.0, .n = 400, .minZ = 20};
  CHECK(!tooLow.passed());
  CHECK(tooLow.conclusive());

  // Two samples cannot condemn a configuration, however low the z they give.
  RoeCheck const thin{.applicable = true, .z = 1.0, .n = 2, .minZ = 20};
  CHECK(thin.passed());
  CHECK(!thin.conclusive());

  // A failed Gerbicz check does, whatever the z.
  RoeCheck const broken{.applicable = true, .z = 40.0, .n = 400, .minZ = 20, .checkOk = false};
  CHECK(!broken.passed());
}

TEST(ll_timing_is_refused_rather_than_answered_with_prp) {
  // The kind reaches only the option resolution, so an LL request would come back as a PRP timing
  // with a PRP Gerbicz verdict.  The refusal happens before anything touches the device.
  GpuCommon const shared{};
  FFTConfig const fft{"512:15:512:202"};

  bool threw = false;
  try {
    (void)timeCall(shared, fft, TestKind::LL, 142'438'559, {});
  } catch (const char* mes) { threw = true; }

  CHECK(threw);
}
