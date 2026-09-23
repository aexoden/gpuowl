// Copyright (C) Jason Lynch

// Tests the value model as a pure function: the gain distribution and what it learns, the saving a configuration
// would bring, and what one more call on a contested pair is worth.  Each counterexample the model exists to get right
// has a fixture of its own.

#include "Value.h"

#include "test.h"

#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

using namespace tune;

namespace {

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol * std::max(1.0, std::abs(b)); }

double total(const GainDist& g) { return std::accumulate(g.p.begin(), g.p.end(), 0.0); }

constexpr u64 E0 = 118'063'003;

ObjectivePoint point(u64 exponent, double weight, double cStar) {
  return {
    .kind = TestKind::PRP, .exponent = exponent, .weight = weight, .cost = Cost{.us = cStar, .entry = "x", .fft = "x"}};
}

Interval everywhere() { return {.lo = 1, .hi = ~u64(0), .regime = {}}; }

// A concluded option set eligible over [lo, hi], its mean `mean` over `calls` calls of four blocks each, with the block
// spread chosen so that its standard error is `se`.
OptionSet optionSet(const std::string& id, double mean, double se, u32 calls, u64 lo = 1, u64 hi = ~u64(0)) {
  Measurement m{
    .mean = mean, .stddev = 0, .blocks = 4 * calls, .calls = calls, .drift = 1, .status = Status::Ok, .ts = 0};
  double const unit = standardError(Measurement{.mean = mean, .stddev = 1, .blocks = 4 * calls, .calls = calls});
  m.stddev = se / unit;
  return {.entry = {.id = id,
                    .cost = pessimisticCost(m),
                    .fft = "512:15:512:202",
                    .kind = TestKind::PRP,
                    .emin = lo,
                    .reach = hi,
                    .regime = {},
                    .evidence = Evidence::Unvalidated,
                    .opts = {}},
          .m = m};
}

}  // namespace

TEST(the_gain_prior_is_a_distribution_with_a_tail) {
  CHECK(near(total(GAIN_PRIOR), 1));
  for (size_t i = 1; i < GAIN_BINS; ++i) { CHECK(GAIN_AT[i] > GAIN_AT[i - 1]); }
  CHECK(GAIN_PRIOR.p.back() > 0);
}

TEST(the_forty_percent_off_family_scores_above_zero) {
  std::vector<ObjectivePoint> const points{point(E0, 1, 100)};

  // A point estimate of the gain -- the prior's mean -- puts a candidate 40% off the pace at 100 * 1.4 * (1 - 0.018),
  // above the frontier, and so at exactly nothing.
  CHECK_EQ(saving(points, TestKind::PRP, everywhere(), 140 * (1 - GAIN_PRIOR.mean())), 0.0);

  // The distribution does not: the 32% and 64% bins both reach under the frontier.
  double const prior = expectedSaving(points, TestKind::PRP, everywhere(), 140, GAIN_PRIOR);
  CHECK(near(prior, 0.01 * (100 - 140 * 0.68) + 0.002 * (100 - 140 * 0.36)));
  CHECK(prior > 0);

  // However long the device goes on showing nothing, the value falls but never reaches zero...
  GainCounts nothing;
  for (int i = 0; i < 1000; ++i) { nothing.observe(0); }
  double const flat = expectedSaving(points, TestKind::PRP, everywhere(), 140, nothing.posterior(GAIN_PRIOR));
  CHECK(flat > 0);
  CHECK(flat < prior / 50);

  // ...and it rises when gains elsewhere turn out to have a fatter tail than the prior said.
  GainCounts fat;
  for (int i = 0; i < 4; ++i) { fat.observe(0.4); }
  CHECK(expectedSaving(points, TestKind::PRP, everywhere(), 140, fat.posterior(GAIN_PRIOR)) > 2 * prior);

  // Twice the frontier is still within the tail; three times is past any gain the bins hold.
  CHECK(GAIN_PRIOR.expectedSaving(100, 200) > 0);
  CHECK_EQ(GAIN_PRIOR.expectedSaving(100, 300), 0.0);
}

TEST(saving_counts_only_the_points_the_configuration_could_serve) {
  std::vector<ObjectivePoint> points{point(100, 0.25, 100), point(200, 0.5, 120), point(300, 0.25, 90)};
  points.push_back(
    {.kind = TestKind::LL, .exponent = 200, .weight = 1, .cost = Cost{.us = 500, .entry = "y", .fft = "y"}});
  points.push_back({.kind = TestKind::PRP, .exponent = 250, .weight = 1, .cost = std::nullopt});

  // At 95 over [150, 350]: 0.5 * (120 - 95) at 200, nothing at 300 where production already runs cheaper, nothing
  // for the other kind or where no FFT runs.
  Interval const band{.lo = 150, .hi = 350, .regime = {}};
  CHECK(near(saving(points, TestKind::PRP, band, 95), 12.5));

  // Its expectation is the saving at each gain, weighted: at 110, every bin from 16% reaches below 90 as well.
  double expected = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) {
    expected += GAIN_PRIOR.p[i] * saving(points, TestKind::PRP, band, 110 * (1 - GAIN_AT[i]));
  }
  CHECK(near(expectedSaving(points, TestKind::PRP, band, 110, GAIN_PRIOR), expected));
  CHECK(expected > 0.5 * (120 - 110));
}

TEST(an_observed_gain_is_split_between_the_bins_either_side) {
  GainCounts c;
  c.observe(0.001);
  CHECK(near(c.counts()[0], 0.8));
  CHECK(near(c.counts()[1], 0.2));

  c.observe(0.03);
  CHECK(near(c.counts()[3], 0.5));
  CHECK(near(c.counts()[4], 0.5));

  // A loss is no gain, and a gain past the last bin counts there.
  c.observe(-0.1);
  CHECK(near(c.counts()[0], 1.8));
  c.observe(0.9);
  CHECK(near(c.counts().back(), 1));

  // One count each, and the counts' mean gain is the observed mean (with the last capped).
  CHECK(near(c.n(), 4));
  CHECK(near(std::accumulate(c.counts().begin(), c.counts().end(), 0.0), 4));
  double mean = 0;
  for (size_t i = 0; i < GAIN_BINS; ++i) { mean += c.counts()[i] * GAIN_AT[i]; }
  CHECK(near(mean, 0.001 + 0.03 + 0 + 0.64));

  // Exactly on a bin, all of it there.
  GainCounts exact;
  exact.observe(0.08);
  CHECK(near(exact.counts()[5], 1));
}

TEST(the_posterior_moves_from_the_prior_toward_what_is_observed) {
  GainCounts c;
  for (size_t i = 0; i < GAIN_BINS; ++i) { CHECK(near(c.posterior(GAIN_PRIOR).p[i], GAIN_PRIOR.p[i])); }

  // As many observations as the prior is worth: halfway.
  for (int i = 0; i < 8; ++i) { c.observe(0.04); }
  GainDist const half = c.posterior(GAIN_PRIOR);
  CHECK(near(total(half), 1));
  CHECK(near(half.p[4], (GAIN_PRIOR.p[4] + 1) / 2));
  CHECK(near(half.p[0], GAIN_PRIOR.p[0] / 2));
}

TEST(an_entry_whose_moves_keep_finding_nothing_sinks_but_is_not_removed) {
  EntryKey const dull{"512:15:512:202", TestKind::PRP, "short32"};
  EntryKey const lively{"1:512:8:512:202", TestKind::PRP, "short32"};

  GainModel model;
  for (int i = 0; i < 40; ++i) {
    model.observe(dull, 0);
    model.observe(lively, 0.04);
  }

  GainDist const device = model.global();
  GainDist const d = model.forEntry(dull);
  GainDist const l = model.forEntry(lively);
  CHECK(near(total(d), 1));
  CHECK(near(total(l), 1));

  // Each entry's own record moves it from what the device shows, in its own direction.
  CHECK(d.expectedSaving(100, 100) < device.expectedSaving(100, 100));
  CHECK(l.expectedSaving(100, 100) > device.expectedSaving(100, 100));

  // But only so far: the dull entry keeps at least half of the device's tail, so it can float again.
  CHECK(d.p.back() >= (1 - ENTRY_MIX) * device.p.back());
  CHECK(d.expectedSaving(100, 140) > 0);

  // An entry with no record of its own is the device's.
  GainDist const fresh = model.forEntry({"4K:15:512:202", TestKind::PRP, "short32"});
  for (size_t i = 0; i < GAIN_BINS; ++i) { CHECK(near(fresh.p[i], device.p[i])); }
}

TEST(gains_are_replayed_from_each_entrys_rows_in_the_order_they_were_taken) {
  TuneDB db;
  u32 const env = db.internEnv(dbEnvOf(Env{.isNvidia = true, .computeCapability = 806}));
  u32 const sess = db.beginSession(env, "", 0, 1'753'471'200);
  u32 const elsewhere = db.beginSession(db.internEnv(dbEnvOf(Env{.isAmd = true})), "", 0, 1'753'471'300);

  auto add = [&](u32 s, const std::string& fft, const UseConfig& opts, double mean, u32 calls,
                 Status status = Status::Ok) {
    CHECK(db.add(RunRow{
      .sess = s,
      .fft = fft,
      .kind = TestKind::PRP,
      .exponent = E0,
      .regime = regimeOf(FFTConfig{fft}, E0),
      .cfg = db.internCfg(opts),
      .m = {.mean = mean, .stddev = 0.1, .blocks = 4 * calls, .calls = calls, .drift = 1, .status = status, .ts = 0}}));
  };

  std::string const x = "512:15:512:202";
  add(sess, x, {}, 100, 2);                // the first: what the rest are measured against
  add(sess, x, {{"WMUL", "1"}}, 98, 2);    // 2% under the best so far
  add(sess, x, {{"LOADS", "3"}}, 99, 2);   // over the new best: nothing
  add(sess, x, {{"STORES", "2"}}, 50, 1);  // one call concludes nothing, however cheap
  add(sess, x, {{"MULTI_Q", "1"}}, 0, 0, Status::NoCompile);
  add(sess, x, {{"LDSPAD_W", "0"}}, 93.1, 2);     // 5% under 98
  add(sess, "1:512:8:512:202", {}, 1450, 2);      // another entry's first row
  add(elsewhere, x, {{"UNROLL_W", "1"}}, 10, 2);  // another device's

  GainModel const model = gainsOf(db, env);
  CHECK(near(model.all().n(), 3));

  const auto& c = model.all().counts();
  CHECK(near(c[0], 1));     // 99 against 98
  CHECK(near(c[3], 1));     // 2%
  CHECK(near(c[4], 0.75));  // 5%, a quarter of the way from 4% to 8%
  CHECK(near(c[5], 0.25));

  // A row's calls pool across sessions, as emission pools them, and the row keeps the place its first call gave it:
  // concluded now, STORES=2 gained 1 - 50/98 against the rows before it, and the 93.1 after it gained nothing.
  u32 const later = db.beginSession(env, "", 0, 1'753'481'200);
  add(later, x, {{"STORES", "2"}}, 50, 1);
  GainModel const pooled = gainsOf(db, env);
  CHECK(near(pooled.all().n(), 4));
  const auto& p = pooled.all().counts();
  CHECK(near(p[0], 2));
  CHECK(near(p[4], 0));
  CHECK(near(p[7] * GAIN_AT[7] + p[8] * GAIN_AT[8], 1 - 50 / 98.0));
}

TEST(an_option_set_measured_again_is_not_another_move) {
  TuneDB db;
  Env const device{.isNvidia = true, .computeCapability = 806};
  u32 const env = db.internEnv(dbEnvOf(device));
  u32 const sess = db.beginSession(env, "", 0, 1'753'471'200);

  std::string const x = "512:15:512:202";
  FFTConfig const fft{x};
  auto add = [&](const UseConfig& opts, u64 exponent, double mean) {
    CHECK(db.add(RunRow{.sess = sess,
                        .fft = x,
                        .kind = TestKind::PRP,
                        .exponent = exponent,
                        .regime = regimeOf(fft, exponent),
                        .cfg = db.internCfg(opts),
                        .m = {.mean = mean,
                              .stddev = 0.1,
                              .blocks = 4 * MIN_CALLS,
                              .calls = MIN_CALLS,
                              .drift = 1,
                              .status = Status::Ok,
                              .ts = 0}}));
  };

  // One move, 2% off the defaults at first sight; then both option sets measured again at twenty other exponents of
  // the same regime, as resumed or re-anchored readings are.
  add({}, E0, 100);
  add({{"WMUL", "1"}}, E0, 98);
  for (u64 i = 1; i <= 20; ++i) {
    CHECK(regimeOf(fft, E0 + 2 * i) == regimeOf(fft, E0));
    add({{"WMUL", "1"}}, E0 + 2 * i, 97);
    add({}, E0 + 2 * i, i % 2 ? 100.2 : 99.8);
  }

  // And the defaults spelt out, which is the same configuration.
  const Option* const wmul = findOption("WMUL");
  CHECK(wmul != nullptr);
  add({{"WMUL", std::to_string(wmul->defaultFor(device, fft, {}))}}, E0 + 100, 100);

  // One observation, of the move as all its readings describe it: 1 - (98 * 2 + 97 * 40) / 42 against the defaults'
  // pooled 100.
  GainModel const model = gainsOf(db, env);
  CHECK(near(model.all().n(), 1));
  double const gain = 1 - (98.0 * 2 + 97.0 * 40) / 42 / 100;
  const auto& c = model.all().counts();
  CHECK(near(c[3] + c[4], 1));
  CHECK(near(c[3] * GAIN_AT[3] + c[4] * GAIN_AT[4], gain));
  CHECK(near(model.global().mean(), (GAIN_PRIOR_WEIGHT * GAIN_PRIOR.mean() + gain) / (GAIN_PRIOR_WEIGHT + 1)));
}

TEST(regret_is_greatest_at_a_tie_and_falls_away_with_the_gap) {
  CHECK(near(regret(0, 2), 2 / std::sqrt(2 * std::numbers::pi)));
  CHECK(near(regret(1.5, 2), regret(-1.5, 2)));
  CHECK(regret(1, 2) < regret(0, 2));
  CHECK(regret(3, 2) < regret(1, 2));
  CHECK(regret(20, 1) < 1e-80);
  CHECK_EQ(regret(1, 0), 0.0);

  // It never goes negative, which a careless tail term could make it at a wide gap.
  for (double mu = 0; mu < 40; mu += 0.5) { CHECK(regret(mu, 1) >= 0); }
}

TEST(one_more_call_is_worth_the_regret_it_is_expected_to_remove) {
  // Integrated directly: after one more call on a side with error `se` over `calls` calls, the difference's mean is
  // distributed N(mu, s^2) with s^2 the fall in its variance, and the regret left is regret(mu', sigma') averaged over
  // that.  The other side's error `otherSe` is in sigma and sigma' alike.
  auto integrated = [](double mu, double se, u32 calls, double otherSe) {
    double const narrowed = se * std::sqrt(double(calls) / (calls + 1));
    double const sigma = std::hypot(se, otherSe);
    double const after = std::hypot(narrowed, otherSe);
    double const s = std::sqrt(sigma * sigma - after * after);
    double left = 0;
    int const steps = 20'000;
    for (int i = 0; i < steps; ++i) {
      double const x = -12 + 24 * (i + 0.5) / steps;
      left += std::exp(-0.5 * x * x) / std::sqrt(2 * std::numbers::pi) * regret(mu + s * x, after) * 24 / steps;
    }
    return regret(mu, sigma) - left;
  };

  // A difference of 3 against errors of 2 and 1: one call is worth ~0.0017 of the weight, not the 0.045 that narrowing
  // the error while holding the mean still would say.
  CHECK(near(refineValue(1, 3, 2, 2), integrated(3, 2, 2, 1), 1e-6));
  CHECK(near(refineValue(1, 3, 2, 2), 0.0017007268876, 1e-9));
  CHECK(near(refineValue(1, 3, 2, 2), integrated(3, 2, 2, 5), 1e-6));

  // At a tie, the spread the call can move the mean by, over sqrt(2 pi).
  CHECK(near(refineValue(0.5, 0, 2, 2), 0.5 * (2 / std::sqrt(3.0)) / std::sqrt(2 * std::numbers::pi)));
  CHECK(near(refineValue(0.5, 0, 2, 2), 0.5 * integrated(0, 2, 2, 1), 1e-6));

  // Worth most where the bar is widest, the calls fewest and the gap narrowest; nothing where no weight rides on it.
  CHECK(refineValue(1, 0, 2, 2) > refineValue(1, 0, 1, 2));
  CHECK(refineValue(1, 0, 2, 2) > refineValue(1, 0, 2, 8));
  CHECK(refineValue(1, 0, 2, 2) > refineValue(1, 3, 2, 2));
  CHECK_EQ(refineValue(0, 0, 2, 2), 0.0);
}

TEST(a_tie_no_exponent_cares_about_scores_nothing) {
  // Two option sets tied at 100 +- 1, over the same band.
  std::vector<OptionSet> const tied{optionSet("a", 100, 1, 2), optionSet("b", 100.1, 1, 2)};

  // Where the workload weighs them, the tie is a contest worth a call on either side.
  std::vector<ObjectivePoint> const weighed{point(E0, 0.6, 100), point(E0 + 2, 0.4, 100)};
  std::vector<Contest> const contested = contests(tied, weighed);
  CHECK_EQ(contested.size(), size_t(1));
  CHECK_EQ(contested[0].chosen, size_t(0));
  CHECK_EQ(contested[0].runnerUp, size_t(1));
  CHECK(near(contested[0].weight, 1));
  CHECK(refineValue(contested[0], tied, 0) > 0.04);
  CHECK(refineValue(contested[0], tied, 1) > 0.04);

  // Where no point with weight falls in their band, there is nothing to decide.
  std::vector<ObjectivePoint> const unweighed{point(E0, 0, 100)};
  CHECK(contests(tied, unweighed).empty());
  std::vector<OptionSet> const outside{optionSet("a", 100, 1, 2, 1, E0 - 1), optionSet("b", 100.1, 1, 2, 1, E0 - 1)};
  CHECK(contests(outside, weighed).empty());

  // Nor where a third configuration is cheaper everywhere they are eligible: production never runs either.
  std::vector<OptionSet> beaten = tied;
  beaten.push_back(optionSet("c", 90, 0.1, 2));
  std::vector<Contest> const behind = contests(beaten, weighed);
  CHECK_EQ(behind.size(), size_t(1));
  CHECK_EQ(behind[0].chosen, size_t(2));

  // Their contest with the leader is ten sigmas wide, which one more call cannot change: worth ~0.
  CHECK(refineValue(behind[0], beaten, behind[0].runnerUp) < 1e-20);
  CHECK(refineValue(behind[0], beaten, behind[0].chosen) < 1e-20);
}

TEST(the_pair_contested_at_each_point_is_the_cheapest_two_eligible_there) {
  // a serves only the low band and b only the high one; c everywhere between them in cost.
  std::vector<OptionSet> const sets{optionSet("a", 90, 1, 2, 1, 150), optionSet("b", 95, 1, 2, 151, 1000),
                                    optionSet("c", 100, 1, 2)};
  std::vector<ObjectivePoint> const points{point(100, 0.3, 90), point(120, 0.2, 90), point(200, 0.5, 95)};

  std::vector<Contest> const out = contests(sets, points);
  CHECK_EQ(out.size(), size_t(2));
  CHECK_EQ(out[0].chosen, size_t(0));
  CHECK_EQ(out[0].runnerUp, size_t(2));
  CHECK(near(out[0].weight, 0.5));
  CHECK_EQ(out[1].chosen, size_t(1));
  CHECK_EQ(out[1].runnerUp, size_t(2));
  CHECK(near(out[1].weight, 0.5));
}
