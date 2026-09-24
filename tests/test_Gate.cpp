// Copyright (C) Jason Lynch

// Tests the accuracy gate as a pure function of the readings: the floor every set has to clear, the default-accuracy
// reading a set that spends accuracy is held to, which evidence state a reading earns, and which of a database's
// readings count for a set published up to a given exponent.

#include "Gate.h"

#include "Anchor.h"
#include "Bootstrap.h"

#include "test.h"

#include <optional>
#include <string>

using namespace tune;

namespace {

Env nvidia() {
  Env env;
  env.isNvidia = true;
  env.computeCapability = 806;
  return env;
}

RoeRow reading(double z, u32 n = 2000, bool checkOk = true) {
  return {.sess = 1, .fft = {}, .exponent = 0, .cfg = 0, .z = z, .n = n, .maxRoe = 0.3, .checkOk = checkOk, .ts = 0};
}

// An FFT with an FP32 part, where TAIL_TRIGS32 applies and defaults to 2.
FFTConfig hybrid() { return FFTConfig{"2:512:8:512:202"}; }
FFTConfig fp64() { return FFTConfig{"512:15:512:212"}; }

struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  u64 ts = 1'753'471'300;

  Fixture() {
    env = db.internEnv(dbEnvOf(nvidia()));
    sess = db.beginSession(env, "512:15:512:212@118063003", 0, 1'753'471'200);
    CHECK(env && sess);
  }

  void read(const FFTConfig& fft, u64 exponent, const UseConfig& opts, double z, bool checkOk = true) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = fft.spec(),
                        .exponent = exponent,
                        .cfg = db.internCfg(opts),
                        .z = z,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = checkOk,
                        .ts = ++ts}));
  }

  [[nodiscard]] GateVerdict verdict(const FFTConfig& fft, u64 exponent, const UseConfig& opts) const {
    return Gates{db, env, nvidia()}(fft, exponent, opts);
  }
};

}  // namespace

TEST(a_set_is_read_at_the_largest_prime_its_interval_holds) {
  CHECK_EQ(gateExponent({.lo = 78'643'196, .hi = 143'413'744, .regime = {}}), u64(143'413'741));
  CHECK_EQ(gateExponent({.lo = 78'643'196, .hi = 143'413'741, .regime = {}}), u64(143'413'741));

  // 24 to 28 holds no prime, and an empty interval holds nothing.
  CHECK_EQ(gateExponent({.lo = 24, .hi = 28, .regime = {}}), u64(0));
  CHECK_EQ(gateExponent({.lo = 100, .hi = 99, .regime = {}}), u64(0));
}

TEST(a_set_is_owed_until_it_has_a_reading) {
  GateVerdict const v = judge(FFT64, std::nullopt, false, std::nullopt);
  CHECK(v.state == GateState::Owed);
  CHECK(!v.owesReference);

  // A set that spends accuracy owes its own reading before its reference's.
  CHECK(!judge(FFT64, std::nullopt, true, std::nullopt).owesReference);
}

TEST(every_set_has_to_clear_the_floor_production_warns_at) {
  CHECK(judge(FFT64, reading(19.99), false, std::nullopt).state == GateState::Rejected);
  CHECK(judge(FFT64, reading(20), false, std::nullopt).state == GateState::Passed);

  // Everything with an FP32 or NTT part is held to 6.
  CHECK(judge(FFT3261, reading(5.9), false, std::nullopt).state == GateState::Rejected);
  CHECK(judge(FFT3261, reading(6.1), false, std::nullopt).state == GateState::Passed);

  // A failed Gerbicz check is a wrong answer, whatever the errors looked like on the way.
  GateVerdict const failed = judge(FFT64, reading(30, 2000, false), false, std::nullopt);
  CHECK(failed.state == GateState::Rejected);
  CHECK_EQ(failed.why, std::string{"its Gerbicz check failed"});

  CHECK_EQ(judge(FFT64, reading(17.25), false, std::nullopt).why, std::string{"z 17.25 is below the floor of 20"});
}

TEST(a_reading_at_the_fitted_standard_confirms_the_inherited_reach) {
  CHECK(judge(FFT64, reading(24.44), false, std::nullopt).evidence == Evidence::Unvalidated);
  CHECK(judge(FFT64, reading(28), false, std::nullopt).evidence == Evidence::Confirmed);
  CHECK(judge(FFT3261, reading(28.5), false, std::nullopt).evidence == Evidence::Confirmed);
}

TEST(too_few_rounding_errors_pass_only_a_set_at_default_accuracy) {
  GateVerdict const sparse = judge(FFT64, reading(0, 2), false, std::nullopt);
  CHECK(sparse.state == GateState::Passed);
  CHECK(sparse.evidence == Evidence::Unavailable);

  CHECK(judge(FFT64, reading(0, 2), true, reading(24)).state == GateState::Rejected);
  CHECK(judge(FFT64, reading(0, 3), false, std::nullopt).state == GateState::Rejected);
}

TEST(a_set_that_spends_accuracy_reads_no_worse_than_its_defaults) {
  // Measured on a Tesla P100 at 512:8:512: the defaults read 24.44, and MM2_CHAIN=1 22.34.
  GateVerdict const owed = judge(FFT64, reading(22.34), true, std::nullopt);
  CHECK(owed.state == GateState::Owed);
  CHECK(owed.owesReference);

  GateVerdict const worse = judge(FFT64, reading(22.34), true, reading(24.44));
  CHECK(worse.state == GateState::Rejected);
  CHECK_EQ(worse.why, std::string{"z 22.34 is below the 24.44 its defaults read"});

  CHECK(judge(FFT64, reading(23.95), true, reading(24.44)).state == GateState::Passed);
  CHECK(judge(FFT64, reading(23.93), true, reading(24.44)).state == GateState::Rejected);
  CHECK(judge(FFT64, reading(29), true, reading(28.5)).evidence == Evidence::Confirmed);

  // Still the floor first, however badly the defaults read.
  CHECK(judge(FFT64, reading(19), true, reading(18)).state == GateState::Rejected);

  // Defaults that fail their own check cannot be read worse than; defaults too accurate to fit cannot be compared with.
  CHECK(judge(FFT3261, reading(7), true, reading(3, 2000, false)).state == GateState::Passed);
  CHECK(judge(FFT3261, reading(7), true, reading(0, 1)).state == GateState::Rejected);
}

TEST(only_keys_that_change_the_rounding_spend_accuracy) {
  Env const env = nvidia();
  CHECK(!movesAccuracy(env, hybrid(), {}));
  CHECK(!movesAccuracy(env, hybrid(), {{"TAIL_TRIGS32", "2"}}));
  CHECK(movesAccuracy(env, hybrid(), {{"TAIL_TRIGS32", "0"}}));
  CHECK(movesAccuracy(env, hybrid(), {{"TAIL_TRIGS32", "1"}, {"UNROLL_W", "1"}}));
  CHECK(!movesAccuracy(env, hybrid(), {{"UNROLL_W", "1"}}));

  // A key that does not apply to the FFT changes nothing on it.
  CHECK(!movesAccuracy(env, fp64(), {{"TAIL_TRIGS32", "0"}}));

  // Suspected keys count.
  CHECK(movesAccuracy(env, fp64(), {{"MM2_CHAIN", "1"}}) ==
        (findOption("MM2_CHAIN")->defaultFor(env, fp64(), {}) != 1));
}

TEST(a_reference_is_the_set_with_its_accuracy_keys_at_their_defaults) {
  Env const env = nvidia();
  UseConfig const moved{{"TAIL_TRIGS32", "0"}, {"UNROLL_W", "1"}};
  UseConfig const reference = accuracyReference(env, hybrid(), moved);

  CHECK_EQ(reference.at("TAIL_TRIGS32"), std::string{"2"});
  CHECK_EQ(reference.at("UNROLL_W"), std::string{"1"});
  CHECK(!movesAccuracy(env, hybrid(), reference));
  CHECK(canonicalConfig(env, hybrid(), reference) == canonicalConfig(env, hybrid(), UseConfig{{"UNROLL_W", "1"}}));

  // Stated, so that nothing layered under it can set it otherwise.
  CHECK(accuracyReference(env, hybrid(), {}).contains("TAIL_TRIGS32"));
}

TEST(a_reading_counts_in_its_own_regime_at_or_above_the_gate) {
  Fixture f;
  UseConfig const opts{{"TAIL_KERNELS", "3"}};
  CHECK(!canonicalConfig(nvidia(), fp64(), opts).empty());

  // 512:15:512 runs short32 up to 143413744.
  CHECK(f.verdict(fp64(), 143'413'741, opts).state == GateState::Owed);

  f.read(fp64(), 120'000'007, opts, 30);
  CHECK(f.verdict(fp64(), 143'413'741, opts).state == GateState::Owed);

  // Past the regime's end the kernels are others.
  f.read(fp64(), 150'000'001, opts, 30);
  CHECK(f.verdict(fp64(), 143'413'741, opts).state == GateState::Owed);

  f.read(fp64(), 143'413'741, opts, 17);
  CHECK(f.verdict(fp64(), 143'413'741, opts).state == GateState::Rejected);

  // The later reading stands, as a later roe row always does.
  f.read(fp64(), 143'413'741, opts, 25);
  CHECK(f.verdict(fp64(), 143'413'741, opts).state == GateState::Passed);

  // One spelling of the set is every spelling of it.
  CHECK(f.verdict(fp64(), 143'413'741, {{"TAIL_KERNELS", "3"}, {"TAIL_TRIGS32", "0"}}).state == GateState::Passed);
  CHECK(f.verdict(fp64(), 143'413'741, {}).state == GateState::Owed);
}

TEST(a_set_that_spends_accuracy_waits_for_its_references_reading) {
  Fixture f;
  u64 const top = 131'952'797;
  UseConfig const moved{{"TAIL_TRIGS32", "0"}};

  f.read(hybrid(), top, moved, 12);
  GateVerdict const owed = f.verdict(hybrid(), top, moved);
  CHECK(owed.state == GateState::Owed);
  CHECK(owed.owesReference);

  f.read(hybrid(), top, accuracyReference(nvidia(), hybrid(), moved), 12.3);
  CHECK(f.verdict(hybrid(), top, moved).state == GateState::Passed);

  f.read(hybrid(), top, {}, 13);
  CHECK(f.verdict(hybrid(), top, moved).state == GateState::Rejected);
}

TEST(exact_arithmetic_passes_without_a_reading) {
  Fixture f;
  GateVerdict const v = f.verdict(FFTConfig{"3:1K:8:512:202"}, 152'674'507, {});
  CHECK(v.state == GateState::Passed);
  CHECK(v.evidence == Evidence::NotApplicable);
}

TEST(another_envs_readings_do_not_count) {
  Fixture f;
  Env other = nvidia();
  other.computeCapability = 600;
  u32 const otherEnv = f.db.internEnv(dbEnvOf(other));
  u32 const otherSess = f.db.beginSession(otherEnv, "512:15:512:212@118063003", 0, 1'753'471'200);
  CHECK(f.db.add(RoeRow{.sess = otherSess,
                        .fft = fp64().spec(),
                        .exponent = 143'413'741,
                        .cfg = f.db.internCfg({}),
                        .z = 25,
                        .n = 2000,
                        .maxRoe = 0.3,
                        .checkOk = true,
                        .ts = 1}));
  CHECK(f.verdict(fp64(), 143'413'741, {}).state == GateState::Owed);
}

TEST(a_set_and_its_reference_are_compared_at_one_exponent) {
  // 512:15:512 runs short32 up to 143498475, past its table reach, so a reading above the gate exponent still counts.
  Fixture f;
  FFTConfig const fft{"512:15:512:202"};
  UseConfig const moved{{"MM2_CHAIN", "1"}};
  CHECK(movesAccuracy(nvidia(), fft, moved));
  UseConfig const reference = accuracyReference(nvidia(), fft, moved);
  u64 const gate = 143'413'741;
  u64 const above = 143'498'461;

  f.read(fft, gate, moved, 22);
  f.read(fft, gate, reference, 25);
  CHECK(f.verdict(fft, gate, moved).state == GateState::Rejected);

  // A reference read higher up, where every set reads worse, says nothing about the set read at the gate.
  f.read(fft, above, reference, 21);
  CHECK(f.verdict(fft, gate, moved).state == GateState::Rejected);

  // Read where its reference was, the set is compared there, and owes nothing it does not need.
  f.read(fft, above, moved, 20.8);
  CHECK(f.verdict(fft, gate, moved).state == GateState::Passed);

  // A set read only higher up owes its reference there, not at the gate exponent, where it would never compare.
  Fixture g;
  g.read(fft, above, moved, 20.8);
  g.read(fft, gate, reference, 25);
  GateVerdict const owed = g.verdict(fft, gate, moved);
  CHECK(owed.state == GateState::Owed);
  CHECK(owed.owesReference);
  CHECK_EQ(owed.referenceAt, above);
}

TEST(a_set_the_table_does_not_search_is_its_own_configuration_to_the_gate) {
  // LDSMUL_W is never searched, but it changes the kernels: a reading of the defaults says nothing about it.
  Fixture f;
  f.read(fp64(), 143'413'741, {}, 25);
  CHECK(f.verdict(fp64(), 143'413'741, {}).state == GateState::Passed);
  CHECK(f.verdict(fp64(), 143'413'741, {{"LDSMUL_W", "2"}}).state == GateState::Owed);
  CHECK(f.verdict(fp64(), 143'413'741, {{"NOT_A_KEY", "1"}}).state == GateState::Owed);

  f.read(fp64(), 143'413'741, {{"LDSMUL_W", "2"}}, 23);
  CHECK(f.verdict(fp64(), 143'413'741, {{"LDSMUL_W", "2"}}).state == GateState::Passed);
}
