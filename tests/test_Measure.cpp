// Copyright (C) Jason Lynch

// GPU-free tests of the parts of src/Measure.cpp that do not touch a device: folding a call's
// blocks into a row, the accuracy floor, and the verdict on a rounding-error reading.

#include "test.h"

#include "BuildId.h"
#include "Gpu.h"
#include "Measure.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
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

// What a thrown message means. A verdict here goes into the database and is read as final, so the cost of reading a
// lost device as a fact about the configuration is every configuration measured after it.
TEST(a_stop_is_not_a_failure_of_anything) {
  Failure const f = classify("stop requested");
  CHECK(f.stop);
  CHECK(!f.fatal);
  CHECK(f.status == Status::Ok);
}

TEST(a_lost_device_is_fatal_and_is_not_recorded_against_the_configuration) {
  for (const char* message :
       {"DEVICE_NOT_AVAILABLE (-2) clFinish(q) at src/clwrap.cpp:326 finish", "DEVICE_NOT_FOUND (-1) clGetDeviceIDs"}) {
    Failure const f = classify(message);
    CHECK(f.stop);
    CHECK(f.fatal);
  }
}

TEST(a_build_failure_is_permanent_and_a_refusal_is_not) {
  CHECK(classify("Can't compile fftmiddlein.cl").status == Status::NoCompile);
  CHECK(classify("Can't find kernel tailMulZero").status == Status::NoCompile);

  // Everything else is a fact about this device: most often a launch asking for more than it has, which is the
  // tuner's most ordinary answer and must not read as a lost device.
  Failure const refused = classify("OUT_OF_RESOURCES (-5) clEnqueueNDRangeKernel at src/clwrap.cpp:1 run");
  CHECK(refused.status == Status::Unsupported);
  CHECK(!refused.stop && !refused.fatal);
}

TEST(the_message_survives_the_classification) {
  CHECK_EQ(classify("Can't compile shufl.cl").what, std::string{"Can't compile shufl.cl"});
}

// A check that could not be taken is not a check that found nothing to measure. Before this distinction existed, a
// failed accuracy check reported "not applicable (exact arithmetic)" on an FP64 configuration and exited 0.
TEST(an_accuracy_check_that_could_not_be_taken_is_not_a_pass) {
  RoeCheck const notRun{.applicable = true, .minZ = 20, .status = Status::Unsupported};
  CHECK(notRun.status != Status::Ok);
  CHECK(!notRun.conclusive());

  // It is not a *rejection* either: a configuration that would not run produced no timing to adopt, so the gate has
  // nothing to reject. Whoever needs to know that no reading exists asks the status, which is why it is on the result.
  CHECK(notRun.passed());

  RoeCheck const exact{.applicable = false, .minZ = 20};
  CHECK(exact.status == Status::Ok);
  CHECK(exact.passed());

  // And a reading that was taken and failed still fails.
  RoeCheck const tooNoisy{.applicable = true, .z = 3, .n = 100, .minZ = 20};
  CHECK(tooNoisy.status == Status::Ok);
  CHECK(!tooNoisy.passed());
}

// --- what -measure was asked for -------------------------------------------------------------------------------

namespace {

// parseMeasureArgs reports a usage error by throwing; the message itself is what reaches the user.
bool rejected(std::string_view text) {
  try {
    (void)parseMeasureArgs(text);
    return false;
  } catch (const std::string&) { return true; } catch (const char*) {
    return true;
  }
}

}  // namespace

TEST(a_bare_spec_measures_with_every_default) {
  MeasureArgs const a = parseMeasureArgs("512:15:512:202");

  CHECK_EQ(a.fft, std::string{"512:15:512:202"});
  CHECK_EQ(a.calls, 8u);
  CHECK_EQ(a.blocks, BLOCKS_PER_CALL);
  CHECK_EQ(a.blockSize, 0u);  // the production block size, which only Args knows
  CHECK_EQ(a.exponent, 0u);   // the top of the FFT's range
  CHECK(a.anchorFft.empty());
  CHECK(a.roe);
  CHECK(!a.drain);
  CHECK(a.drift);
}

TEST(every_setting_is_read_off_the_spec) {
  MeasureArgs const a =
    parseMeasureArgs("4:1K:8:256:202,n=12,blocks=6,block=500,exp=118063003,anchor=256:2:256,roe=0,drain=1");

  CHECK_EQ(a.fft, std::string{"4:1K:8:256:202"});
  CHECK_EQ(a.anchorFft, std::string{"256:2:256"});
  CHECK_EQ(a.exponent, 118'063'003u);
  CHECK_EQ(a.calls, 12u);
  CHECK_EQ(a.blocks, 6u);
  CHECK_EQ(a.blockSize, 500u);
  CHECK(!a.roe);
  CHECK(a.drain);

  // The scheduled anchor and the alternating one correct for the same thing; naming the second turns the first off.
  CHECK(!a.drift);
  CHECK(!parseMeasureArgs("512:15:512:202,drift=0").drift);
  CHECK(parseMeasureArgs("512:15:512:202,drift=1").drift);
  CHECK(rejected("512:15:512:202,drift=yes"));
}

TEST(the_spec_can_also_be_named_by_key) {
  CHECK_EQ(parseMeasureArgs("fft=512:15:512,n=4").fft, std::string{"512:15:512"});
}

TEST(a_setting_that_would_conclude_nothing_is_a_usage_error) {
  // A verdict compares two rows of MIN_CALLS, so anything under 2 * MIN_CALLS could time the configuration but never
  // judge its error bar, which is what the command is for.  A single-iteration block times a warm-up, not an iteration.
  CHECK(rejected("512:15:512,n=1"));
  CHECK(rejected("512:15:512,n=2"));
  CHECK(rejected("512:15:512,n=3"));
  CHECK_EQ(parseMeasureArgs("512:15:512,n=4").calls, 2 * MIN_CALLS);
  CHECK(rejected("512:15:512,blocks=1"));
  CHECK(rejected("512:15:512,block=1"));
}

TEST(a_mistyped_setting_is_refused_rather_than_ignored) {
  CHECK(rejected("512:15:512,calls=8"));  // n=, not calls=
  CHECK(rejected("512:15:512,n=eight"));
  CHECK(rejected("512:15:512,512:15:512"));     // a second bare token is a typo, not a second FFT
  CHECK(rejected("n=8"));                       // no FFT at all
  CHECK(rejected("512:15:512,"));               // a trailing comma leaves an empty token
  CHECK(rejected("nonsense:spec"));             // not an FFT the parser accepts
  CHECK(rejected("512:15:512,anchor=nope:x"));  // nor is the anchor
}

namespace {

// A session needs a database and an Env to open, and neither needs a device: what is exercised below is which anchor
// the session adopts and what it divides by, all of which happens before anything is built.
Env const NVIDIA{.isNvidia = true, .computeCapability = 806, .deviceName = "a card", .driverVersion = "1"};

TuneDB dbOf(const std::string& rows) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n' + rows, "fixture"));
  return db;
}

}  // namespace

TEST(a_session_picks_an_anchor_at_the_exponent_it_probes) {
  TuneDB db;
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));

  auto const want = chooseAnchor(118'063'003);
  CHECK(want.has_value());
  CHECK_EQ(session.anchor().text(), want->text());
  CHECK_EQ(db.sessions().at(0).anchor, want->text());
  CHECK_EQ(session.drift(), 1.0);
}

TEST(a_session_with_no_probe_is_unanchored) {
  TuneDB db;
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin());

  CHECK(!session.anchor().valid());
  CHECK_EQ(db.sessions().at(0).anchor, std::string{});
  CHECK_EQ(session.drift(), 1.0);
}

TEST(a_later_session_takes_the_anchor_its_env_is_pinned_to) {
  // The first session of the env chose 512:15:512:212 at an exponent this one is not probing; it still divides by the
  // movement of that, since a ratio against anything else says nothing about the rows already stored.
  TuneDB db =
    dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0 machine=- build=" +
         [] {
           char b[32];
           snprintf(b, sizeof(b), "%016llx", (unsigned long long)buildFingerprint());
           return std::string{b};
         }() +
         "\n"
         "cfg   1 -\n"
         "sess  1 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073\n"
         "anchor 1 512:15:512:212 143400073 1 1774.230 1.0000 1753471410\n");

  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));

  CHECK_EQ(session.anchor().text(), std::string{"512:15:512:212@143400073"});
  CHECK_EQ(db.sessions().at(1).anchor, std::string{"512:15:512:212@143400073"});
  CHECK_EQ(db.sessions().at(1).env, 1u);
}

TEST(an_env_of_other_kernels_is_not_this_ones_anchor) {
  // Same card, another build: a new env, so the anchor and the baseline are chosen afresh rather than inherited from
  // measurements taken against kernels this binary no longer has.
  TuneDB db =
    dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0 machine=- build=dead\n"
         "cfg   1 -\n"
         "sess  1 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073\n"
         "anchor 1 512:15:512:212 143400073 1 1774.230 1.0000 1753471410\n");

  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));

  CHECK_EQ(db.envs().size(), size_t{2});
  CHECK_EQ(session.envId(), 2u);
  CHECK_EQ(session.anchor().text(), chooseAnchor(118'063'003)->text());
}

TEST(an_anchor_a_generation_died_on_is_not_built_again) {
  // The restart unit re-execs the same command, so a configuration that took the device down is named by the env and
  // re-timed by every generation. held() is what stops that, and the anchor is subject to it like anything else.
  std::string const build = [] {
    char b[32];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long)buildFingerprint());
    return std::string{b};
  }();
  auto const anchor = chooseAnchor(118'063'003);
  CHECK(anchor.has_value());

  TuneDB db = dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0 machine=-"
                   " build=" +
                   build +
                   "\n"
                   "cfg   1 -\n"
                   "sess  1 env=1 start=1753471200 gen=0 anchor=" +
                   anchor->text() +
                   "\n"
                   "try   1 " +
                   anchor->fft + " prp " + to_string(anchor->exponent) + " 1 1753471250\n");

  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));
  CHECK(!session.held(FFTConfig{anchor->fft}, TestKind::PRP, anchor->exponent, {}).empty());
}
