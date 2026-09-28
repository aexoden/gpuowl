// Copyright (C) Jason Lynch

// GPU-free tests of the parts of src/Measure.cpp that do not touch a device: folding a call's
// blocks into a row, the accuracy floor, and the verdict on a rounding-error reading.

#include "test.h"

#include "Args.h"
#include "BuildId.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Measure.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <optional>
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

TEST(a_call_runs_its_warm_up_before_the_blocks_it_times) {
  CHECK_EQ(callIterations(BLOCKS_PER_CALL, 1000), u64(5000));
  CHECK_EQ(summarize(samplesOf({100.0, 100.1})).iters, u64(5000));
}

namespace {

constexpr u64 REFERENCE = 0x3f45bf9bea7213ea;

Call llCall(u64 res64, Status status = Status::Ok) {
  Call c = summarize(samplesOf({100.0, 100.1, 100.0, 100.1}));
  c.res64 = res64;
  c.measurement.status = status;
  return c;
}

// Reads `second` if asked, and counts the asking.
struct Again {
  Call second;
  u32 asked = 0;

  [[nodiscard]] std::function<Call()> reader() {
    return [this] {
      ++asked;
      return second;
    };
  }
};

}  // namespace

TEST(an_ll_reading_that_agrees_with_the_reference_is_taken_as_it_is) {
  Again again{.second = llCall(REFERENCE)};
  Call const c = checkedAgainst(REFERENCE, llCall(REFERENCE), again.reader());
  CHECK(c.measurement.ok());
  CHECK_EQ(again.asked, 0u);
}

TEST(an_ll_reading_that_disagrees_is_read_again_before_anything_is_concluded) {
  Again again{.second = llCall(REFERENCE)};
  Call const c = checkedAgainst(REFERENCE, llCall(0x1234), again.reader());
  CHECK_EQ(again.asked, 1u);
  CHECK(c.measurement.ok());
  CHECK_EQ(c.res64, REFERENCE);
}

TEST(an_ll_reading_that_disagrees_twice_is_an_error) {
  Again again{.second = llCall(0x1234)};
  Call const c = checkedAgainst(REFERENCE, llCall(0x1234), again.reader());
  CHECK_EQ(again.asked, 1u);
  CHECK(c.measurement.status == Status::Err);

  // Two different wrong residues are no better.
  Again other{.second = llCall(0x5678)};
  CHECK(checkedAgainst(REFERENCE, llCall(0x1234), other.reader()).measurement.status == Status::Err);
}

TEST(an_ll_call_that_gave_no_reading_says_nothing_about_the_residue) {
  // A failed first call is recorded as its failure, and is not read again for a residue it never had.
  Again again{.second = llCall(REFERENCE)};
  CHECK(checkedAgainst(REFERENCE, llCall(0, Status::NoCompile), again.reader()).measurement.status ==
        Status::NoCompile);
  CHECK_EQ(again.asked, 0u);

  // Nor is a second call cut short an error of the configuration.
  Again stopped{.second = llCall(0, Status::Lost)};
  CHECK(checkedAgainst(REFERENCE, llCall(0x1234), stopped.reader()).measurement.status == Status::Lost);
}

TEST(a_call_that_fails_its_own_check_is_read_again_before_anything_is_concluded) {
  Again passes{.second = llCall(0)};
  Call const c = checkedTwice(llCall(0, Status::Err), passes.reader());
  CHECK_EQ(passes.asked, 1u);
  CHECK(c.measurement.ok());

  Again fails{.second = llCall(0, Status::Err)};
  CHECK(checkedTwice(llCall(0, Status::Err), fails.reader()).measurement.status == Status::Err);
  CHECK_EQ(fails.asked, 1u);

  // A second call cut short is not a second failure.
  Again stopped{.second = llCall(0, Status::Lost)};
  CHECK(checkedTwice(llCall(0, Status::Err), stopped.reader()).measurement.status == Status::Lost);
}

TEST(a_call_that_passed_its_check_or_gave_no_reading_is_not_read_again) {
  for (Status const status : {Status::Ok, Status::NoCompile, Status::Unsupported, Status::Lost}) {
    Again again{.second = llCall(0, Status::Err)};
    CHECK(checkedTwice(llCall(0, status), again.reader()).measurement.status == status);
    CHECK_EQ(again.asked, 0u);
  }
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
  CHECK(a.kind == TestKind::PRP);
}

TEST(an_ll_configuration_is_measured_by_kind) {
  CHECK(parseMeasureArgs("512:15:512:202,kind=ll").kind == TestKind::LL);
  CHECK(parseMeasureArgs("512:15:512:202,kind=prp").kind == TestKind::PRP);
  CHECK(rejected("512:15:512:202,kind=cert"));

  // The drain control compares the PRP loop's block boundaries.
  CHECK(rejected("512:15:512:202,kind=ll,drain=1"));
  CHECK(parseMeasureArgs("512:15:512:202,kind=ll,drain=0").kind == TestKind::LL);
}

// Refused before anything is opened or built, so it needs no device and leaves no database behind.
TEST(a_forced_carry_is_not_measured) {
  Args args{true};
  args.parse("-carry long");
  CHECK(runMeasure(GpuCommon{.context = nullptr, .args = &args, .bufCache = nullptr, .background = nullptr},
                   parseMeasureArgs("256:2:256:202")) == MeasureOutcome::Failed);
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

TEST(a_fresh_env_races_for_its_anchor) {
  // Nothing is known about the card yet, so nothing is pinned: the candidates are raced when the anchor is first due,
  // and the session row, written before that, names none.
  TuneDB db;
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));

  CHECK(!session.anchor().valid());
  CHECK(session.anchorDue());
  CHECK_EQ(db.sessions().at(0).anchor, std::string{});
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
  TuneDB db = dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                   " fp64=1 builtins=1 machine=- build=" +
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
  TuneDB db = dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                   " fp64=1 builtins=1 machine=- build=dead\n"
                   "cfg   1 -\n"
                   "sess  1 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073\n"
                   "anchor 1 512:15:512:212 143400073 1 1774.230 1.0000 1753471410\n");

  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(118'063'003));

  CHECK_EQ(db.envs().size(), size_t{2});
  CHECK_EQ(session.envId(), 2u);
  CHECK(!session.anchor().valid());
  CHECK(session.anchorDue());
}

TEST(an_anchor_a_generation_died_on_is_not_built_again) {
  // The restart unit re-execs the same command, so a configuration that took the device down is named by the env and
  // re-timed by every generation. held() is what stops that, and the anchor is subject to it like anything else.
  std::string const build = [] {
    char b[32];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long)buildFingerprint());
    return std::string{b};
  }();
  std::vector<AnchorSpec> const candidates = anchorCandidates(118'063'003);
  CHECK(!candidates.empty());
  std::optional<AnchorSpec> const anchor = candidates.front();

  TuneDB db = dbOf("env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                   " fp64=1 builtins=1 machine=-"
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

  // Nor is it built to warm the device: a warm-up is a build like any other.  This session has no device, so building
  // anything at all would fail here, and no attempt is declared for it.
  size_t const tries = db.tries().size();
  Call const warm = session.warmUp(FFTConfig{anchor->fft}, TestKind::PRP, anchor->exponent, {});
  CHECK(!warm.measurement.ok());
  CHECK_EQ(db.tries().size(), tries);
}

TEST(a_race_the_env_has_readings_for_builds_nothing_and_pins_the_cheapest) {
  // Every candidate already read at the built-in defaults, by a session whose race a stop cut short before it pinned
  // anything: the race is settled from the record, so this session, with no device, builds nothing.
  std::string const build = [] {
    char b[32];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long)buildFingerprint());
    return std::string{b};
  }();
  u64 const E = 118'063'003;
  std::vector<AnchorSpec> const candidates = anchorCandidates(E);
  CHECK(candidates.size() >= 3);

  // Dearest first, so the last is the cheapest; the second-last is dearer only by a little.
  std::string rows = "env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                     " fp64=1 builtins=1 machine=-"
                     " build=" +
    build +
    "\n"
    "cfg   1 -\n"
    "cfg   2 TAIL_KERNELS=0\n"
    "sess  1 env=1 start=1753471200 gen=0 anchor=-\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    double const us = 9000 - 1000.0 * double(i);
    FFTConfig const fft{candidates[i].fft};
    rows += "run   1 " + candidates[i].fft + " prp " + to_string(E) + " " + regimeOf(fft, E).label() + " 1 " +
      std::to_string(us) + " 1.0 4 1 1.0000 ok 1753471300\n";
  }
  // A reading of the dearest under a searched option is not a reading of the configuration that would be anchored,
  // however cheap.
  FFTConfig const dearest{candidates.front().fft};
  rows += "run   1 " + candidates.front().fft + " prp " + to_string(E) + " " + regimeOf(dearest, E).label() +
    " 2 10.0 1.0 4 1 1.0000 ok 1753471310\n";

  TuneDB db = dbOf(rows);
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(E));
  CHECK(session.anchorDue());

  session.raceAnchor();
  CHECK(session.anchor() == candidates.back());
  CHECK_EQ(db.tries().size(), size_t{0});

  // The race is over: a second call has nothing to do.
  session.raceAnchor();
  CHECK(session.anchor() == candidates.back());
}

TEST(a_race_passes_over_a_candidate_an_earlier_generation_died_on) {
  std::string const build = [] {
    char b[32];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long)buildFingerprint());
    return std::string{b};
  }();
  u64 const E = 118'063'003;
  std::vector<AnchorSpec> const candidates = anchorCandidates(E);
  CHECK(candidates.size() >= 2);

  // The cheapest reading's configuration later took the device down; the next cheapest is the anchor.
  std::string rows = "env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                     " fp64=1 builtins=1 machine=-"
                     " build=" +
    build +
    "\n"
    "cfg   1 -\n"
    "sess  1 env=1 start=1753471200 gen=0 anchor=-\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    FFTConfig const fft{candidates[i].fft};
    rows += "run   1 " + candidates[i].fft + " prp " + to_string(E) + " " + regimeOf(fft, E).label() + " 1 " +
      std::to_string(9000 - 1000.0 * double(i)) + " 1.0 4 1 1.0000 ok 1753471300\n";
  }
  rows += "try   1 " + candidates.back().fft + " prp " + to_string(E) + " 1 1753471400\n";

  TuneDB db = dbOf(rows);
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin(E));
  session.raceAnchor();
  CHECK(session.anchor() == candidates[candidates.size() - 2]);
}

TEST(an_ll_call_with_no_agreed_reference_records_nothing_against_the_configuration) {
  TuneDB db = dbOf("");
  Session session{GpuCommon{}, db, NVIDIA};
  CHECK(session.begin());

  // Four FFTs' built-in defaults read at this exponent and length, no two alike: the witnesses are spent.
  u64 constexpr E = 118'063'003;
  u64 const iters = callIterations(BLOCKS_PER_CALL, 1000);
  u64 residue = 1;
  for (const char* spec : {"1K:8:1K:202", "2:1K:8:512:202", "3:1K:8:512", "51:1K:8:512"}) {
    CHECK(db.add(RefRow{.sess = session.id(),
                        .fft = FFTConfig{spec}.spec(),
                        .exponent = E,
                        .iters = iters,
                        .res64 = residue++,
                        .ts = 1}));
  }

  // Nothing is built, and no row stands against the configuration: the missing reference is scoped to (E, iters),
  // while a failure row would hold the set out of its whole regime.
  for (int attempt = 0; attempt < 2; ++attempt) {
    Call const c = session.run(FFTConfig{"1K:8:1K:202"}, TestKind::LL, E, {}, BLOCKS_PER_CALL, 1000);
    CHECK(c.measurement.status == Status::Unsupported);
  }
  CHECK(db.runs().empty());
  CHECK(db.tries().empty());
  CHECK(!session.stopped());
}
