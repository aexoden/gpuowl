// Copyright (C) Jason Lynch

// Tests the accuracy gate as a pure function of the readings: the floor every set has to clear, the default-accuracy
// reading a set that spends accuracy is held to, which evidence state a reading earns, which of a database's readings
// count for a set published up to a given exponent, and the reach derived for a set that falls short there.

#include "Gate.h"

#include "Anchor.h"
#include "Bootstrap.h"

#include "test.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>

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

  // Over the interval the fitted table gives `fft` in the regime `exponent` runs in, whose top it is.
  [[nodiscard]] GateVerdict verdict(const FFTConfig& fft, u64 exponent, const UseConfig& opts) const {
    Interval const span = interval(fft, exponent);
    CHECK_EQ(gateExponent(span), exponent);
    return Gates{db, env, nvidia()}(fft, span, opts);
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
  CHECK(judge(FFT64, reading(19.99), false, std::nullopt).derivable);
  CHECK(judge(FFT64, reading(20), false, std::nullopt).state == GateState::Passed);

  // Everything with an FP32 or NTT part is held to 6.
  CHECK(judge(FFT3261, reading(5.9), false, std::nullopt).state == GateState::Rejected);
  CHECK(judge(FFT3261, reading(6.1), false, std::nullopt).state == GateState::Passed);

  // A failed Gerbicz check is a wrong answer, whatever the errors looked like on the way.
  GateVerdict const failed = judge(FFT64, reading(30, 2000, false), false, std::nullopt);
  CHECK(failed.state == GateState::Rejected);
  CHECK(failed.derivable);
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

  GateVerdict const incomparable = judge(FFT64, reading(0, 2), true, reading(24));
  CHECK(incomparable.state == GateState::Rejected);
  CHECK(!incomparable.derivable);
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

  // Defaults that fall short here are held to the fitted standard, and so is the set; defaults too accurate to fit
  // cannot be compared with, at any exponent.
  for (const RoeRow& failing : {reading(3, 2000, false), reading(5.5)}) {
    GateVerdict const heldTo28 = judge(FFT3261, reading(7), true, failing);
    CHECK(heldTo28.state == GateState::Rejected);
    CHECK(heldTo28.derivable);
    CHECK_EQ(heldTo28.why, std::string{"z 7.00 is below the 28 its defaults are held to, which fall short here"});
  }
  CHECK(judge(FFT3261, reading(28), true, reading(3, 2000, false)).state == GateState::Passed);
  CHECK(judge(FFT3261, reading(7), true, reading(0, 1)).state == GateState::Rejected);
  CHECK(!judge(FFT3261, reading(7), true, reading(0, 1)).derivable);

  // A set that fails its own check still needs its reference's reading: that is what its reach is derived towards.
  CHECK(judge(FFT3261, reading(6.1, 2000, false), true, std::nullopt).owesReference);
}

TEST(a_set_is_held_to_its_reference_where_the_reference_clears_the_floor_and_to_28_otherwise) {
  Standard const relative = standardFor(FFT64, true, reading(24.44));
  CHECK_EQ(relative.aim, 24.44);
  CHECK_EQ(relative.bar, 24.44 - ACCURACY_SLACK_Z);

  // Never below the floor, however close to it the reference reads.
  CHECK_EQ(standardFor(FFT64, true, reading(20.2)).bar, 20.0);

  for (Standard const s : {standardFor(FFT64, false, reading(24.44)), standardFor(FFT64, true, reading(19)),
                           standardFor(FFT64, true, reading(30, 2000, false)), standardFor(FFT64, true, std::nullopt),
                           standardFor(FFT3261, true, reading(5.9))}) {
    CHECK_EQ(s.aim, TARGET_Z);
    CHECK_EQ(s.bar, TARGET_Z);
  }
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

  CHECK(movesAccuracy(env, fp64(), {{"MM2_CHAIN", "1"}}) ==
        (findOption("MM2_CHAIN")->defaultFor(env, fp64(), {}) != 1));

  // A value measured to round as the default does spends nothing, though its key's other values do.
  CHECK(movesAccuracy(env, fp64(), {{"TAIL_KERNELS", "0"}}));
  CHECK(!movesAccuracy(env, fp64(), {{"TAIL_KERNELS", "3"}}));
  CHECK(movesAccuracy(env, fp64(), {{"TAIL_KERNELS", "3"}, {"TAIL_TRIGS", "0"}}));

  // A value that does not parse is taken to spend it.
  CHECK(movesAccuracy(env, fp64(), {{"TAIL_KERNELS", "x"}}));
}

TEST(what_decides_the_rounding_is_the_keys_held_at_a_value_that_changes_it) {
  Env const env = nvidia();
  CHECK(roundingOf(env, hybrid(), {}).empty());
  CHECK(roundingOf(env, hybrid(), {{"UNROLL_W", "1"}, {"TAIL_TRIGS32", "2"}}).empty());
  UseConfig const trigs{{"TAIL_TRIGS32", "0"}};
  CHECK(roundingOf(env, hybrid(), {{"TAIL_TRIGS32", "0"}, {"UNROLL_W", "1"}}) == trigs);

  // A value measured to round as the default does is the default's rounding, and so are the keys around it.
  CHECK(roundingOf(env, fp64(), {{"TAIL_KERNELS", "3"}, {"UNROLL_W", "1"}}) == roundingOf(env, fp64(), {}));
  UseConfig const single{{"TAIL_KERNELS", "1"}};
  CHECK(roundingOf(env, fp64(), single) == single);
  CHECK(roundingOf(env, fp64(), {{"TAIL_KERNELS", "1"}}) != roundingOf(env, fp64(), {{"TAIL_KERNELS", "0"}}));

  // A key that does not apply to the FFT changes nothing on it.
  CHECK(roundingOf(env, fp64(), {{"TAIL_TRIGS32", "0"}}).empty());
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

  // A value that rounds as the default is kept, being part of what the reference is the reference of.
  UseConfig const tail = accuracyReference(env, fp64(), {{"TAIL_KERNELS", "3"}, {"TAIL_TRIGS", "0"}});
  CHECK_EQ(tail.at("TAIL_KERNELS"), std::string{"3"});
  CHECK_EQ(tail.at("TAIL_TRIGS"), std::string{"2"});
  UseConfig const single = accuracyReference(env, fp64(), {{"TAIL_KERNELS", "1"}});
  CHECK_EQ(single.at("TAIL_KERNELS"), std::string{"2"});
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

  // Short of the floor, it owes a reading further down, where its reach is derived.
  f.read(fp64(), 143'413'741, opts, 17);
  GateVerdict const deriving = f.verdict(fp64(), 143'413'741, opts);
  CHECK(deriving.state == GateState::Owed);
  CHECK(deriving.owedAt < 143'413'741);

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

  // Read again better, the defaults show what the set spent, and its reach is derived further down.
  f.read(hybrid(), top, {}, 13);
  GateVerdict const deriving = f.verdict(hybrid(), top, moved);
  CHECK(deriving.state == GateState::Owed);
  CHECK(!deriving.owesReference);
  CHECK(deriving.owedAt < top);
}

TEST(exact_arithmetic_passes_without_a_reading) {
  Fixture f;
  FFTConfig const ntt{"3:1K:8:512:202"};
  Interval const span = interval(ntt, 152'674'507);
  GateVerdict const v = Gates{f.db, f.env, nvidia()}(ntt, span, {});
  CHECK(v.state == GateState::Passed);
  CHECK_EQ(v.reach, span.hi);
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
  FFTConfig const fft{"512:15:512:212"};
  UseConfig const moved{{"MM2_CHAIN", "1"}};
  CHECK(movesAccuracy(nvidia(), fft, moved));
  UseConfig const reference = accuracyReference(nvidia(), fft, moved);
  u64 const gate = 143'413'741;
  u64 const above = 143'498'461;

  // Below its reference, a reach is derived for it under the gate.
  f.read(fft, gate, moved, 22);
  f.read(fft, gate, reference, 25);
  GateVerdict const deriving = f.verdict(fft, gate, moved);
  CHECK(deriving.state == GateState::Owed);
  CHECK(!deriving.owesReference);
  CHECK(deriving.owedAt < gate);

  // A reference read higher up, where every set reads worse, says nothing about the set read at the gate.
  f.read(fft, above, reference, 21);
  CHECK(f.verdict(fft, gate, moved).state == GateState::Owed);
  CHECK_EQ(f.verdict(fft, gate, moved).owedAt, deriving.owedAt);

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
  CHECK_EQ(owed.owedAt, above);
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

namespace {

// Answers every reading the gate owes `opts` from `zAt` -- the set's own, or its reference's -- until it concludes; the
// verdict and the readings taken.
std::pair<GateVerdict, u32> settle(
  Fixture& f, const FFTConfig& fft, u64 top, const UseConfig& opts,
  const std::function<double(bool reference, u64 E)>& zAt,
  const std::function<bool(u64 E)>& checkOk = [](u64) { return true; }) {
  u32 taken = 0;
  GateVerdict v = f.verdict(fft, top, opts);
  for (; v.state == GateState::Owed && taken < 30; v = f.verdict(fft, top, opts)) {
    UseConfig const which = v.owesReference ? accuracyReference(nvidia(), fft, opts) : opts;
    f.read(fft, v.owedAt, which, zAt(v.owesReference, v.owedAt), checkOk(v.owedAt));
    ++taken;
  }
  CHECK(taken < 30);
  return {v, taken};
}

// z rising by one for every `slope` bits per word below `top`, where it reads `z`.
double linear(const FFTConfig& fft, u64 top, double z, double slope, u64 E) {
  return z + (double(top) - double(E)) / double(fft.size()) / slope;
}

}  // namespace

TEST(a_set_at_default_accuracy_short_of_the_floor_is_published_up_to_a_reach_derived_at_28) {
  // Shaped after a Tesla P100's readings of 2:512:8:512:212, taken when the table's top was 31.66 bits per word: z 6.1
  // and a failed check there, 13.1 at 31.5, 21.6 at 31.3 and 33.4 at 31.1 -- the table gives every variant of the shape
  // one reach. The model reads z 6.1 and fails at the top the table gives now.
  Fixture f;
  FFTConfig const fft{"2:512:8:512:212"};
  Interval const span = interval(fft, 131'952'797);
  u64 const top = gateExponent(span);
  CHECK_EQ(top, u64(131'952'797));

  auto const [v, taken] = settle(
    f, fft, top, {}, [&](bool, u64 E) { return linear(fft, top, 6.1, 0.021, E); }, [&](u64 E) { return E != top; });
  CHECK(v.state == GateState::Passed);
  CHECK(v.derived);
  CHECK(v.evidence == Evidence::Confirmed);
  CHECK(taken >= 3 && taken <= 6);

  // Where the model reads 28 or better, and within a guard band of where it reads exactly 28.
  CHECK(linear(fft, top, 6.1, 0.021, v.reach) >= 28);
  double const at28 = double(top) / fft.size() - (28 - 6.1) * 0.021;
  CHECK(double(v.reach) / fft.size() > at28 - REACH_GUARD_BPW);
  CHECK(v.reach >= span.lo && v.reach < top);
}

TEST(a_set_that_spends_accuracy_derives_the_reach_where_it_reads_what_its_defaults_read_at_the_top) {
  // Measured on a Tesla P100 at 512:8:512: the defaults read 24.44 at the table's top, and MM2_CHAIN=1 22.34.
  Fixture f;
  FFTConfig const fft{"512:15:512:212"};
  UseConfig const moved{{"MM2_CHAIN", "1"}};
  CHECK(movesAccuracy(nvidia(), fft, moved));
  u64 const top = 143'413'741;

  auto const [v, taken] = settle(f, fft, top, moved, [&](bool reference, u64 E) {
    return reference ? linear(fft, top, 24.44, 0.015, E) : linear(fft, top, 22.34, 0.015, E);
  });
  CHECK(v.state == GateState::Passed);
  CHECK(v.derived);
  CHECK(taken >= 3 && taken <= 4);

  // 2.1 z below its defaults is about 0.03 bits per word of reach at this slope: the reach reads no worse than the
  // defaults' 24.44, less the slack, and no more than a guard band under where it reads 24.44 exactly.
  CHECK(linear(fft, top, 22.34, 0.015, v.reach) >= 24.44 - ACCURACY_SLACK_Z);
  double const matching = double(top) / fft.size() - (24.44 - 22.34) * 0.015;
  CHECK(double(v.reach) / fft.size() <= matching + ACCURACY_SLACK_Z * 0.015);
  CHECK(double(v.reach) / fft.size() > matching - REACH_GUARD_BPW);
}

TEST(a_set_whose_arithmetic_is_its_references_is_published_to_the_same_reach) {
  // MM_CHAIN away from its default read bit-identical to the defaults on the P100: nothing to derive, nothing lost.
  Fixture f;
  FFTConfig const fft{"512:15:512:212"};
  UseConfig const moved{{"MM_CHAIN", "0"}};
  u64 const top = 143'413'741;
  if (!movesAccuracy(nvidia(), fft, moved)) { return; }

  auto const [v, taken] = settle(f, fft, top, moved, [&](bool, u64 E) { return linear(fft, top, 24.44, 0.015, E); });
  CHECK(v.state == GateState::Passed);
  CHECK(!v.derived);
  CHECK_EQ(v.reach, interval(fft, top).hi);
  CHECK_EQ(taken, 2u);
}

TEST(a_set_no_reach_can_be_confirmed_for_is_rejected) {
  Fixture f;
  FFTConfig const fft{"2:512:8:512:212"};
  u64 const top = 131'952'797;

  // Wrong wherever it is read.
  auto const [v, taken] =
    settle(f, fft, top, {}, [&](bool, u64 E) { return linear(fft, top, 6.1, 0.021, E); }, [](u64) { return false; });
  CHECK(v.state == GateState::Rejected);
  CHECK(!v.derivable);
  CHECK(v.why.starts_with("its Gerbicz check failed, and "));
  CHECK(taken > 2);

  // Or reading the same however far down it goes.
  Fixture g;
  auto const [flat, n] = settle(g, fft, top, {}, [](bool, u64) { return 5.0; });
  CHECK(flat.state == GateState::Rejected);
  CHECK_EQ(n, MAX_DERIVE_READINGS);
}

TEST(a_derived_reach_survives_a_reload_of_the_database) {
  // z as the device reports it, with more digits than a roe row is written with.
  Fixture f;
  FFTConfig const fft{"2:512:8:512:212"};
  u64 const top = 131'952'797;
  auto const [v, taken] = settle(
    f, fft, top, {}, [&](bool, u64 E) { return linear(fft, top, 6.1234567, 0.021, E) + 0.0012345; },
    [&](u64 E) { return E != top; });
  CHECK(v.state == GateState::Passed);
  CHECK(v.derived);

  TuneDB reloaded;
  CHECK(reloaded.parse(f.db.text(), "reloaded"));
  Interval const span = interval(fft, top);
  GateVerdict const again = Gates{reloaded, f.env, nvidia()}(fft, span, {});
  CHECK(again.state == GateState::Passed);
  CHECK_EQ(again.reach, v.reach);
  CHECK_EQ(again.owedAt, v.owedAt);
}
