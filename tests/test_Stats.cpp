// Copyright (C) Jason Lynch

// GPU-free tests of the timing statistics (src/Stats.cpp), over synthetic sample vectors.

#include "test.h"

#include "Stats.h"

#include <cmath>
#include <vector>

using namespace tune;

namespace {

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

Measurement callOf(const std::vector<double>& blocks, double drift = 1, u64 ts = 0) {
  return measurementOf(coreStats(blocks), drift, ts);
}

// `calls` calls of `blocks` blocks each: call i sits at `mean + i * step`, and its blocks straddle their call's mean by
// `spread`, so the within-call and between-call scatter are separately dialled.
std::vector<std::vector<double>> synthetic(u32 calls, u32 blocks, double mean, double step, double spread) {
  std::vector<std::vector<double>> out;
  for (u32 c = 0; c < calls; ++c) {
    std::vector<double> one;
    for (u32 b = 0; b < blocks; ++b) {
      double const offset = (blocks == 1) ? 0 : (2.0 * b / (blocks - 1) - 1);
      one.push_back(mean + c * step + offset * spread);
    }
    out.push_back(one);
  }
  return out;
}

}  // namespace

TEST(stats_of_empty_and_single) {
  Stats const none = statsOf({});
  CHECK_EQ(none.n, 0u);
  CHECK(none.mean == 0 && none.sd == 0 && none.sem() == 0 && none.relSem() == 0);

  std::vector<double> const one{1000.0};
  Stats const s = statsOf(one);
  CHECK_EQ(s.n, 1u);
  CHECK(near(s.mean, 1000));
  CHECK(s.sd == 0 && s.sem() == 0);
}

TEST(stats_of_uses_the_sample_standard_deviation) {
  std::vector<double> const samples{2, 4, 4, 4, 5, 5, 7, 9};
  Stats const s = statsOf(samples);
  CHECK_EQ(s.n, 8u);
  CHECK(near(s.mean, 5));
  CHECK(near(s.sd, std::sqrt(32.0 / 7)));  // the population sd here is 2, so n-1 is visible
  CHECK(near(s.sem(), s.sd / std::sqrt(8.0)));
  CHECK(near(s.relSem(), s.sem() / 5));
}

TEST(spikes_are_dropped_and_the_core_survives) {
  std::vector<double> samples(19, 1000.0);
  samples.push_back(1140.0);  // one block in twenty, upward, large
  CoreStats const core = coreStats(samples);
  CHECK_EQ(core.dropped, 1u);
  CHECK(!core.declined);
  CHECK_EQ(core.stats.n, 19u);
  CHECK(near(core.stats.mean, 1000));
  CHECK(core.stats.sd == 0);

  // Left in, one spike in twenty carries the mean and invents the entire spread.
  Stats const raw = statsOf(samples);
  CHECK(raw.mean > 1006);
  CHECK(raw.sd > 30);
}

TEST(the_cut_is_inclusive_and_one_sided) {
  std::vector<double> samples(10, 1000.0);
  samples.push_back(1020.0);  // exactly median * SPIKE_CUT
  samples.push_back(1020.001);
  CoreStats const core = coreStats(samples);
  CHECK_EQ(core.dropped, 1u);
  CHECK_EQ(core.stats.n, 11u);

  // Nothing is ever dropped downward: a low block is a real reading of a fast run, not an interrupt.
  std::vector<double> low(11, 1000.0);
  low.push_back(600.0);
  CoreStats const lowCore = coreStats(low);
  CHECK_EQ(lowCore.dropped, 0u);
  CHECK_EQ(lowCore.stats.n, 12u);
}

TEST(too_many_spikes_decline_the_model) {
  // A third of the blocks above the cut is not "core plus spikes"; the data keeps whatever shape it has.
  std::vector<double> samples(8, 1000.0);
  samples.insert(samples.end(), 4, 1100.0);
  CoreStats const core = coreStats(samples);
  CHECK(core.declined);
  CHECK_EQ(core.dropped, 0u);
  CHECK_EQ(core.stats.n, 12u);
  CHECK(near(core.stats.mean, statsOf(samples).mean));

  // Exactly at the fraction is still a rejection: the rule is "more than".
  std::vector<double> edge(9, 1000.0);
  edge.insert(edge.end(), 3, 1100.0);
  CoreStats const edgeCore = coreStats(edge);
  CHECK(!edgeCore.declined);
  CHECK_EQ(edgeCore.dropped, 3u);
}

TEST(too_few_blocks_to_have_a_median) {
  std::vector<double> const two{1000.0, 1400.0};
  CoreStats const core = coreStats(two);
  CHECK_EQ(core.dropped, 0u);
  CHECK(!core.declined);
  CHECK_EQ(core.stats.n, 2u);
}

TEST(a_call_is_one_observation) {
  Measurement const m = callOf({1000, 1002, 998, 1000}, 1, 77);
  CHECK_EQ(m.blocks, 4u);
  CHECK_EQ(m.calls, 1u);
  CHECK_EQ(m.ts, 77u);
  CHECK(m.ok());

  // One call is never enough to conclude from, however tight its blocks are.
  CHECK(!concluded(m));
  CHECK(concluded(merged(m, m)));

  CHECK(!measurementOf(coreStats({})).ok());
  CHECK_EQ(measurementOf(coreStats({})).calls, 0u);
}

TEST(merging_two_calls_matches_the_pooled_samples) {
  std::vector<double> const a{1000, 1004, 996, 1000};
  std::vector<double> const b{1010, 1014, 1006, 1010};
  Measurement const m = merged(callOf(a, 1, 5), callOf(b, 1, 9));

  std::vector<double> all = a;
  all.insert(all.end(), b.begin(), b.end());
  Stats const pooled = statsOf(all);

  CHECK_EQ(m.blocks, 8u);
  CHECK_EQ(m.calls, 2u);
  CHECK_EQ(m.ts, 9u);
  CHECK(near(m.mean, pooled.mean, 1e-12));
  CHECK(near(m.stddev, pooled.sd, 1e-12));
}

TEST(merging_is_order_independent_and_absorbs_the_empty) {
  Measurement const a = callOf({1000, 1004, 996, 1000});
  Measurement const b = callOf({1010, 1014, 1006, 1010});
  Measurement const c = callOf({990, 994, 986, 990});

  Measurement const left = merged(merged(a, b), c);
  Measurement const right = merged(merged(c, b), a);
  CHECK(near(left.mean, right.mean, 1e-12));
  CHECK(near(left.stddev, right.stddev, 1e-12));
  CHECK_EQ(left.calls, right.calls);

  Measurement const empty;
  CHECK(near(merged(empty, a).mean, a.mean));
  CHECK_EQ(merged(empty, a).calls, 1u);
  CHECK(near(merged(a, empty).mean, a.mean));
  CHECK_EQ(merged(a, empty).calls, 1u);
}

TEST(merging_corrects_for_drift_before_pooling) {
  std::vector<double> const blocks{1000, 1004, 996, 1000};
  // The same device, 5% slower: every block time, and so every deviation, reads 5% high.
  std::vector<double> slow;
  for (double const x : blocks) { slow.push_back(x * 1.05); }

  Measurement const fast = callOf(blocks, 1.0);
  Measurement const later = callOf(slow, 1.05);
  CHECK(near(later.cost(), fast.cost(), 1e-12));
  CHECK(near(later.costStddev(), fast.costStddev(), 1e-12));

  Measurement const m = merged(fast, later);
  CHECK(near(m.drift, 1));
  CHECK(near(m.mean, fast.mean, 1e-12));

  // The slow call contributes nothing but its own shape once the ratio is out, so the merged row is the two corrected
  // sample sets pooled -- which is not either call's own sd, since a pooled spread is over its own degrees of freedom.
  std::vector<double> both = blocks;
  both.insert(both.end(), blocks.begin(), blocks.end());
  CHECK(near(m.stddev, statsOf(both).sd, 1e-9));
}

TEST(a_failure_is_not_a_sample) {
  Measurement const good = callOf({1000, 1000, 1000, 1000});
  Measurement bad;
  bad.status = Status::Err;
  bad.ts = 42;

  Measurement const spoiled = merged(good, bad);
  CHECK(spoiled.status == Status::Err);
  CHECK(!spoiled.ok());
  CHECK(!concluded(spoiled));

  // And it stays that way: a later ok does not average with it.
  Measurement const after = merged(spoiled, good);
  CHECK(after.status == Status::Err);
  CHECK_EQ(after.ts, 42u);

  for (Status s : {Status::NoCompile, Status::Unsupported, Status::Lost}) {
    Measurement failed;
    failed.status = s;
    CHECK(merged(good, failed).status == s);
    CHECK(parseStatus(toString(s)) == s);
  }
  CHECK(!parseStatus("ok "));
}

TEST(the_error_bar_counts_calls_not_blocks) {
  auto const calls = synthetic(8, BLOCKS_PER_CALL, 1000, 0, 2);
  Measurement m;
  for (const auto& one : calls) { mergeInto(m, callOf(one)); }

  CHECK_EQ(m.blocks, 8 * BLOCKS_PER_CALL);
  CHECK_EQ(m.calls, 8u);

  // Counting blocks would divide by 32 rather than by the 8 independent observations, and would take the pooled
  // spread's block degrees of freedom with it.
  double const perBlock = m.stddev / std::sqrt(double(m.blocks));
  CHECK(standardError(m) > std::sqrt(double(BLOCKS_PER_CALL)) * perBlock);
}

TEST(the_bar_is_never_below_the_spread_of_the_call_means) {
  // Two calls whose blocks are each perfectly steady: every bit of the scatter is between the calls, so the bar is the
  // standard error of the two call means and nothing else. Dividing the pooled block spread by sqrt(calls) instead
  // would report 3.7796 here, under a row the minimum call count already lets the tuner conclude from.
  Measurement const m =
    merged(callOf(std::vector<double>(BLOCKS_PER_CALL, 1000.0)), callOf(std::vector<double>(BLOCKS_PER_CALL, 1010.0)));
  CHECK(concluded(m));
  CHECK(near(m.mean, 1005));
  CHECK(near(m.stddev, std::sqrt(200.0 / 7)));  // pooled over eight blocks, seven degrees of freedom

  std::vector<double> const callMeans{1000, 1010};
  CHECK(near(standardError(m), statsOf(callMeans).sd / std::sqrt(2.0)));
  CHECK(near(standardError(m), 5));
  CHECK(near(pessimisticCost(m), 1015));

  // The same holds at every call count, since the bound is exact whenever the blocks inside a call agree.
  for (u32 calls = 2; calls <= 6; ++calls) {
    Measurement row;
    std::vector<double> means;
    for (u32 c = 0; c < calls; ++c) {
      double const mean = 1000 + 7.0 * c;
      means.push_back(mean);
      mergeInto(row, callOf(std::vector<double>(BLOCKS_PER_CALL, mean)));
    }
    CHECK(near(standardError(row), statsOf(means).sd / std::sqrt(double(calls)), 1e-9));
  }
}

TEST(a_thin_row_is_penalised_more_than_a_thick_one) {
  // The same spread and the same mean, sampled over two calls and over sixteen: the thin row's bar is wider, which is
  // what stops it from outranking the thick one on luck.
  auto const rowOf = [](u32 calls) {
    Measurement m;
    m.mean = 1000;
    m.stddev = 10;
    m.blocks = calls * BLOCKS_PER_CALL;
    m.calls = calls;
    return m;
  };
  CHECK(standardError(rowOf(2)) > 2 * standardError(rowOf(16)));
  CHECK(pessimisticCost(rowOf(2)) > pessimisticCost(rowOf(16)));

  // A row from a single call has no between-call spread to speak of, so its whole pooled spread stands as the bar.
  Measurement one = rowOf(1);
  one.blocks = BLOCKS_PER_CALL;
  CHECK(near(standardError(one), 10));
}

TEST(between_call_scatter_of_two_and_a_half) {
  // What the call-based bar exists for: blocks inside a call scatter 2.5x less than the calls themselves.
  constexpr u32 CALLS = 8;
  constexpr double WITHIN = 2.0;
  auto const spread = [](const std::vector<double>& v) { return statsOf(v).sd; };

  auto const calls = synthetic(CALLS, BLOCKS_PER_CALL, 1000, 0, WITHIN);
  double const withinSd = spread(calls[0]);

  // Shift each call bodily, by offsets scaled so that the call means scatter exactly 2.5x wider than one call's blocks.
  std::vector<double> offsets;
  for (u32 c = 0; c < CALLS; ++c) { offsets.push_back(double(c) - (CALLS - 1) / 2.0); }
  double const scale = 2.5 * withinSd / spread(offsets);

  std::vector<std::vector<double>> scattered;
  std::vector<double> means;
  for (u32 c = 0; c < CALLS; ++c) {
    std::vector<double> one;
    for (double const x : calls[c]) { one.push_back(x + offsets[c] * scale); }
    scattered.push_back(one);
    means.push_back(statsOf(one).mean);
  }

  double const betweenSd = spread(means);
  CHECK(betweenSd > 2.4 * withinSd && betweenSd < 2.6 * withinSd);

  Measurement m;
  for (const auto& one : scattered) { mergeInto(m, callOf(one)); }

  // The pooled bar over calls is the honest one: it lands within a few percent of the standard error of the call
  // means, which is the quantity a race actually needs.
  double const trueSem = betweenSd / std::sqrt(double(CALLS));
  CHECK(standardError(m) > trueSem);
  CHECK(standardError(m) < 1.1 * trueSem);

  // Over blocks it would have claimed half that, which is the error this whole rule removes.
  CHECK(m.stddev / std::sqrt(double(m.blocks)) < 0.55 * trueSem);
}

TEST(pessimistic_cost_prefers_the_well_supported_row) {
  // Two rows a hair apart, one of them thinly sampled and noisy: the mean says the noisy one won.
  Measurement thin;
  thin.mean = 1000;
  thin.stddev = 20;
  thin.blocks = 2 * BLOCKS_PER_CALL;
  thin.calls = 2;

  Measurement solid;
  solid.mean = 1002;
  solid.stddev = 1;
  solid.blocks = 16 * BLOCKS_PER_CALL;
  solid.calls = 16;

  CHECK(thin.cost() < solid.cost());
  CHECK(pessimisticCost(thin) > pessimisticCost(solid));
  CHECK(near(pessimisticCost(solid), solid.mean + PESSIMISM_SIGMA * standardError(solid)));

  // A row with no spread at all is its own pessimistic cost.
  Measurement exact;
  exact.mean = 500;
  exact.blocks = 8;
  exact.calls = 2;
  CHECK(near(pessimisticCost(exact), 500));
  CHECK(relStandardError(exact) == 0);
}

TEST(the_pessimistic_cost_is_on_the_drift_corrected_scale) {
  Measurement m;
  m.mean = 1050;
  m.stddev = 21;
  m.blocks = 4 * BLOCKS_PER_CALL;
  m.calls = 4;
  m.drift = 1.05;

  CHECK(near(m.cost(), 1000));
  CHECK(near(m.costStddev(), 20));
  CHECK(near(standardError(m), 20 * std::sqrt(15.0 / 12) / std::sqrt(4.0)));
  CHECK(near(pessimisticCost(m), 1000 + 2 * standardError(m)));
}
