// Copyright (C) Jason Lynch

// Tests the value model as a pure function: the gain distribution and what it learns, the saving a configuration
// would bring, and what one more call on a contested pair is worth.  Each counterexample the model exists to get right
// has a fixture of its own.

#include "Value.h"

#include "Bootstrap.h"

#include "test.h"

#include <algorithm>
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

TEST(the_runner_up_is_the_set_most_likely_to_be_cheaper_not_the_second_production_ranks) {
  std::vector<ObjectivePoint> const points{point(E0, 1, 100)};

  // a is chosen on six calls; b is a close second by the same standard; c has the cheapest mean, but on two calls with
  // a wide bar its pessimistic cost ranks it last.  A call on c is what could change production's answer.
  std::vector<OptionSet> const sets{optionSet("a", 100, 0.2, 6), optionSet("b", 100.5, 0.2, 6),
                                    optionSet("c", 99, 3, 2)};
  CHECK(sets[2].entry.cost > sets[1].entry.cost);

  std::vector<Contest> const out = contests(sets, points);
  CHECK_EQ(out.size(), size_t(1));
  if (out.size() != 1) { return; }
  CHECK_EQ(out[0].chosen, size_t(0));
  CHECK_EQ(out[0].runnerUp, size_t(2));
  CHECK(undecided(out[0], sets));

  std::vector<double> const worth = refineValues(sets, points);
  CHECK(worth[2] > 0);
  CHECK(worth[0] > 0);
  CHECK_EQ(worth[1], 0.0);

  // With the same bar on every side, it is the second by cost after all.
  std::vector<OptionSet> const even{optionSet("a", 100, 0.2, 6), optionSet("b", 100.5, 0.2, 6),
                                    optionSet("c", 101, 0.2, 6)};
  std::vector<Contest> const plain = contests(even, points);
  CHECK(plain.size() == 1 && plain[0].chosen == 0 && plain[0].runnerUp == 1);
}

TEST(a_contest_is_refined_only_while_the_race_rule_leaves_it_undecided) {
  std::vector<ObjectivePoint> const points{point(E0, 1, 100)};
  auto worthless = [](const std::vector<double>& worth) {
    return std::ranges::all_of(worth, [](double w) { return w == 0; });
  };
  auto contested = [&](const std::vector<OptionSet>& sets) {
    std::vector<Contest> const out = contests(sets, points);
    CHECK_EQ(out.size(), size_t(1));
    return out.front();
  };

  // 1% apart, their two-standard-error intervals overlapping: undecided, and a call on either side is worth something.
  std::vector<OptionSet> const close{optionSet("a", 100, 0.4, 2), optionSet("b", 101, 0.4, 2)};
  CHECK(undecided(contested(close), close));
  std::vector<double> const worth = refineValues(close, points);
  CHECK(worth[0] > 0 && worth[1] > 0);
  CHECK(near(worth[0], refineValue(contested(close), close, 0)));

  // Apart at two standard errors: decided, and worth nothing more.
  std::vector<OptionSet> const apart{optionSet("a", 100, 0.2, 2), optionSet("b", 101, 0.2, 2)};
  CHECK(!undecided(contested(apart), apart));
  CHECK(worthless(refineValues(apart, points)));

  // Within RACE_MARGIN of each other: not worth the calls it would take to say which.
  std::vector<OptionSet> const tied{optionSet("a", 100, 1, 2), optionSet("b", 100.2, 1, 2)};
  CHECK(!undecided(contested(tied), tied));
  CHECK(worthless(refineValues(tied, points)));

  // A side that has had RACE_MAX_CALLS calls is tied by exhaustion; the other can still be called.
  std::vector<OptionSet> const spent{optionSet("a", 100, 0.4, RACE_MAX_CALLS), optionSet("b", 101, 0.4, 2)};
  std::vector<double> const left = refineValues(spent, points);
  CHECK_EQ(left[0], 0.0);
  CHECK(left[1] > 0);
}

TEST(a_set_in_two_contests_is_worth_both) {
  // b is the runner-up to a over the low band and the choice over c over the high one.
  std::vector<OptionSet> const sets{optionSet("a", 100, 0.4, 2, 1, 150), optionSet("b", 100.8, 0.4, 2),
                                    optionSet("c", 101.6, 0.4, 2, 151, 1000)};
  std::vector<ObjectivePoint> const low{point(100, 0.3, 100)};
  std::vector<ObjectivePoint> const high{point(200, 0.7, 100.8)};
  std::vector<ObjectivePoint> both = low;
  both.push_back(high.front());

  CHECK(near(refineValues(sets, both)[1], refineValues(sets, low)[1] + refineValues(sets, high)[1]));
  CHECK(refineValues(sets, high)[1] > refineValues(sets, low)[1]);
}

TEST(a_restart_teaches_its_entry_but_not_the_device) {
  TuneDB db;
  Env const device{.isNvidia = true, .computeCapability = 806};
  u32 const env = db.internEnv(dbEnvOf(device));
  u32 const sess = db.beginSession(env, "", 0, 1'753'471'200);

  std::string const x = "512:15:512:202";
  auto add = [&](const UseConfig& opts, double mean) {
    CHECK(
      db.add(RunRow{.sess = sess,
                    .fft = x,
                    .kind = TestKind::PRP,
                    .exponent = E0,
                    .regime = regimeOf(FFTConfig{x}, E0),
                    .cfg = db.internCfg(opts),
                    .m = {.mean = mean, .stddev = 0.1, .blocks = 8, .calls = 2, .drift = 1, .status = Status::Ok}}));
  };
  add({}, 100);
  add({{"WMUL", "1"}}, 98);
  add({{"LDSPAD_W", "0"}, {"LOADS", "3"}, {"ZEROHACK_W", "0"}}, 120);

  EntryKey const entry{x, TestKind::PRP, regimeOf(FFTConfig{x}, E0).label()};
  GainModel const moves = gainsOf(db, env);

  // Declared as a restart, spelt otherwise than its row (a key at its default): the set is what is recognised.
  CHECK(db.add(JumpRow{.sess = sess,
                       .fft = x,
                       .kind = TestKind::PRP,
                       .regime = regimeOf(FFTConfig{x}, E0),
                       .cfg = db.internCfg({{"LDSPAD_W", "0"}, {"LOADS", "3"}, {"ZEROHACK_W", "0"}, {"WMUL", "2"}}),
                       .k = 0,
                       .ts = 1}));
  GainModel const jumped = gainsOf(db, env);

  // Both see the entry's two observations; only the moves teach the device, so the restart's nothing costs every other
  // entry nothing.
  CHECK(near(moves.all().n(), 2));
  CHECK(near(jumped.all().n(), 1));
  CHECK(near(jumped.all().counts()[3], 1));
  CHECK(jumped.forEntry(entry).mean() < jumped.global().mean());
  CHECK(jumped.global().mean() > moves.global().mean());
}

TEST(a_combination_teaches_the_combination_gains_and_not_the_move_gains) {
  TuneDB db;
  Env const device{.isNvidia = true, .computeCapability = 806};
  u32 const env = db.internEnv(dbEnvOf(device));
  u32 const sess = db.beginSession(env, "", 0, 1'753'471'200);

  std::string const x = "512:15:512:212";
  Regime const regime = regimeOf(FFTConfig{x}, E0);
  auto add = [&](const UseConfig& opts, double mean) {
    CHECK(
      db.add(RunRow{.sess = sess,
                    .fft = x,
                    .kind = TestKind::PRP,
                    .exponent = E0,
                    .regime = regime,
                    .cfg = db.internCfg(opts),
                    .m = {.mean = mean, .stddev = 0.1, .blocks = 8, .calls = 2, .drift = 1, .status = Status::Ok}}));
  };
  add({}, 100);
  add({{"TAIL_KERNELS", "3"}}, 100.4);
  add({{"ZEROHACK_H", "0"}}, 100.2);
  add({{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}}, 97);

  EntryKey const entry{x, TestKind::PRP, regime.label()};
  GainModel const undeclared = gainsOf(db, env);
  CHECK(near(undeclared.all().n(), 3));
  CHECK(near(undeclared.combos().n(), 0));

  // Declared as a combination, however spelt: the 3% it found is the combinations' alone.
  CHECK(db.add(ComboRow{.sess = sess,
                        .fft = x,
                        .kind = TestKind::PRP,
                        .regime = regime,
                        .cfg = db.internCfg({{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}, {"WMUL", "2"}}),
                        .tier = 2,
                        .ts = 1}));
  GainModel const declared = gainsOf(db, env);
  CHECK(near(declared.all().n(), 2));
  CHECK(near(declared.combos().n(), 1));
  CHECK(declared.global().mean() < undeclared.global().mean());
  CHECK(declared.globalCombo().mean() > GAIN_COMBO_PRIOR.mean());

  // The entry's own combinations over the device's, mixed back: one observation moves it by 1/18 of the way.
  GainDist const own = declared.comboForEntry(entry);
  CHECK(own.mean() > declared.globalCombo().mean());
  CHECK(near(declared.comboForEntry(EntryKey{"1K:8:1K:202", TestKind::PRP, "short32"}).mean(),
             declared.globalCombo().mean()));
}

TEST(the_combination_prior_is_rarer_with_a_tail_as_long) {
  double total = 0;
  for (double const p : GAIN_COMBO_PRIOR.p) { total += p; }
  CHECK(near(total, 1));
  CHECK(GAIN_COMBO_PRIOR.p[0] > GAIN_PRIOR.p[0]);
  CHECK_EQ(GAIN_COMBO_PRIOR.p.back(), GAIN_PRIOR.p.back());
  CHECK(near(GainModel{}.globalCombo().mean(), GAIN_COMBO_PRIOR.mean()));
}

TEST(an_entry_whose_jumps_keep_finding_nothing_is_worth_jumping_from_less_and_less) {
  EntryKey const entry{"512:15:512:202", TestKind::PRP, "short32"};
  GainModel model;
  for (int i = 0; i < 20; ++i) { model.observe(entry, 0.004); }

  // With no jump of its own yet, a jump is worth what a move is.
  GainDist const before = model.restartForEntry(entry);
  for (size_t i = 0; i < GAIN_BINS; ++i) { CHECK(near(before.p[i], model.forEntry(entry).p[i])); }

  double previous = before.expectedSaving(100, 100);
  for (int n = 1; n <= 200; ++n) {
    model.observe(entry, 0, GainSource::Restart);
    double const now = model.restartForEntry(entry).expectedSaving(100, 100);
    CHECK(now < previous);
    previous = now;
  }
  CHECK(near(total(model.restartForEntry(entry)), 1));

  // The moves keep their floor, and the jumps do not: a jump is drawn from the same space however often it came up
  // empty, so there is nothing to float it again, and past the floor is where a run can stop jumping.
  GainDist const moves = model.forEntry(entry);
  CHECK(moves.expectedSaving(100, 100) >= (1 - ENTRY_MIX) * model.global().expectedSaving(100, 100));
  CHECK(previous < 0.1 * moves.expectedSaving(100, 100));
}

TEST(the_chance_of_a_gain_is_the_weight_at_or_past_it) {
  CHECK(near(GAIN_PRIOR.chanceOfAtLeast(0), total(GAIN_PRIOR)));
  CHECK(near(GAIN_PRIOR.chanceOfAtLeast(0.16), 0.03 + 0.01 + 0.002));
  CHECK(near(GAIN_PRIOR.chanceOfAtLeast(0.10), 0.03 + 0.01 + 0.002));
  CHECK(near(GAIN_PRIOR.chanceOfAtLeast(0.64), 0.002));
  CHECK_EQ(GAIN_PRIOR.chanceOfAtLeast(0.65), 0.0);
}

TEST(the_gain_an_item_needs_is_where_its_saving_reaches_what_it_has_to_be_worth) {
  // Three points in the band, one dear point outside it that must not count.
  std::vector<ObjectivePoint> const points{point(E0, 0.5, 100), point(E0 + 2, 0.3, 90), point(E0 + 4, 0.2, 80),
                                           point(E0 + 100, 0.9, 500)};
  Interval const band{.lo = E0, .hi = E0 + 10, .regime = {}};
  double const cost = 120;

  for (double const worth : {0.01, 1.0, 4.0, 12.0, 25.0, 60.0}) {
    std::optional<double> const g = requiredGain(points, TestKind::PRP, band, cost, worth);
    CHECK(g.has_value());
    CHECK(near(saving(points, TestKind::PRP, band, cost * (1 - *g)), worth, 1e-9));
    CHECK(saving(points, TestKind::PRP, band, cost * (1 - *g + 1e-6)) < worth);
  }

  // Everything in the band costing nothing would save 0.5*100 + 0.3*90 + 0.2*80 = 93, and no more.
  CHECK(near(*requiredGain(points, TestKind::PRP, band, cost, 93), 1, 1e-9));
  CHECK(!requiredGain(points, TestKind::PRP, band, cost, 93.01));

  // Worth nothing in particular: the gain at which it starts to save anything, which is where it undercuts the
  // dearest point of its band.
  CHECK(near(*requiredGain(points, TestKind::PRP, band, cost, 0), 1 - 100.0 / 120));
  CHECK_EQ(saving(points, TestKind::PRP, band, 100), 0.0);

  // Already cheaper than enough of the band: no gain is needed.
  CHECK_EQ(*requiredGain(points, TestKind::PRP, band, 50, 1), 0.0);

  // Nothing weighs the band, or another kind's grid.
  CHECK(!requiredGain(points, TestKind::PRP, {.lo = 1, .hi = 2, .regime = {}}, cost, 1));
  CHECK(!requiredGain(points, TestKind::LL, band, cost, 1));
}
