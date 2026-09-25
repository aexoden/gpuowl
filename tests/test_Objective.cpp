// Copyright (C) Jason Lynch

// Tests the objective as a pure function of a hand-written database and a hand-built grid: that it is finite before
// anything has been measured, that an entry covers exactly its own interval and no band beside it, that the cheapest
// covering entry of the right kind is the one c* reports, and that the prior is fitted to the nearest measured size of
// the same type and stays below what it was fitted to.

#include "Objective.h"

#include "FFTVariants.h"

#include "test.h"

#include <cmath>
#include <string>

using namespace tune;

namespace {

// One FP64 shape measured once in its short-carry, 32-bit-carry regime, which runs from 78643196 (10 bits per word,
// below which the carry is long) to 143413744 (above which the carry needs 64 bits); a pure NTT shape covering some of
// the same exponents at a higher cost; an LL reading; and rows that must count for nothing -- another env's, a single
// call's, and a failure's.  The FP64 set's accuracy has been read at the top of its interval; the NTT rounds nothing.
const char* const DB =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"Tesla P100-PCIE-16GB\" name=\"Tesla P100-PCIE-16GB\" drv=550.163.01 vendor=nvidia be=ocl cc=600"
  " noasm=0 pdl=0 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "cfg   21 INPLACE=1,PAD=256\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  9 env=2 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 ll 100000000 short32 17 1760.000 3.000 16 4 1.0000 ok 1753471284\n"
  "run   4 3:1K:8:512:202 prp 100000000 short32 21 2000.000 5.000 16 4 1.0000 ok 1753471294\n"
  "run   4 1K:8:1K:202 prp 200000000 short32 21 100.000 1.000 4 1 1.0000 ok 1753471304\n"
  "run   4 4K:8:1K:202 prp 900000000 short32 21 100.000 1.000 16 4 1.0000 err 1753471314\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 900.000 1.000 16 4 1.0000 ok 1753471324\n"
  "roe   4 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471330\n";

constexpr u64 SHORT32_LO = 78'643'196;
constexpr u64 SHORT32_HI = 143'413'744;

TuneDB loaded(const char* text = DB) {
  TuneDB db;
  CHECK(db.parse(text, "fixture"));
  return db;
}

// A grid of one kind over the given points, weights as given.
RunScope scopeOver(std::vector<GridPoint> points, TestKind kind = TestKind::PRP) {
  RunScope out;
  out.lo = points.front().exponent;
  out.hi = points.back().exponent;
  out.probe = points.front().exponent;
  out.grids.push_back({.kind = kind, .fromWorktodo = true, .points = std::move(points)});
  return out;
}

RunScope defaultScope() {
  return makeScope(ScopeArgs{.lo = 100'000'000, .hi = 400'000'000, .kinds = {TestKind::PRP, TestKind::LL}}, {});
}

bool near(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(std::abs(a), std::abs(b)); }

SelectionEntry entryOf(const Objective& objective, const std::string& fft, TestKind kind = TestKind::PRP) {
  for (const SelectionEntry& entry : objective.entries()) {
    if (entry.fft == fft && entry.kind == kind) { return entry; }
  }
  CHECK(!"no such entry");
  return {};
}

}  // namespace

TEST(with_nothing_measured_T_is_finite_and_every_point_is_priced) {
  RunScope const scope = defaultScope();

  // Both ways of having nothing: no database at all, and an env with no rows.
  TuneDB const empty = loaded("# prpll tunedb v1\n"
                              "env   1 gpu=\"X\" name=\"X\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0 machine=-"
                              " build=9a3f21c0d1e2f304\n");
  for (const Objective& objective : {Objective{Env{}, scope}, Objective{empty, 1, scope}}) {
    CHECK(std::isfinite(objective.T()));
    CHECK(objective.T() > 0);
    CHECK_EQ(objective.measured(), 0.0);
    CHECK_EQ(objective.unservable(), 0.0);
    CHECK(objective.entries().empty());

    CHECK_EQ(objective.points().size(), scope.grids[0].points.size() + scope.grids[1].points.size());
    double T = 0;
    for (const ObjectivePoint& point : objective.points()) {
      CHECK(point.cost.has_value());
      CHECK(!point.cost->measured());
      T += point.weight * point.cost->us;
    }
    CHECK(near(objective.T(), T));
  }

  CHECK(near(Objective(Env{}, scope).T(), Objective(empty, 1, scope).T()));
}

TEST(an_entry_whose_interval_excludes_a_band_does_not_cover_it) {
  TuneDB const db = loaded();
  RunScope const scope =
    scopeOver({{SHORT32_LO - 1, 0.25}, {SHORT32_LO, 0.25}, {SHORT32_HI, 0.25}, {SHORT32_HI + 1, 0.25}});
  Objective const objective{db, 1, scope};

  SelectionEntry const entry = entryOf(objective, "512:15:512:212");
  CHECK_EQ(entry.emin, SHORT32_LO);
  CHECK_EQ(entry.reach, SHORT32_HI);

  // Inside, at both ends, c* is the entry and at the cost it is published at.
  for (u64 E : {SHORT32_LO, u64(100'000'000), SHORT32_HI}) {
    std::optional<Cost> const cost = objective.cStar(TestKind::PRP, E);
    CHECK(cost && cost->measured());
    CHECK_EQ(cost->entry, entry.id);
    CHECK_EQ(cost->us, entry.cost);
  }

  // One exponent outside, on either side, is another regime's band: the same shape runs there, but not the kernels
  // that were measured.  Below, nothing else covers it and the prior stands in; above, the costlier NTT entry does.
  std::optional<Cost> const below = objective.cStar(TestKind::PRP, SHORT32_LO - 1);
  CHECK(below && !below->measured());
  CHECK(std::isfinite(below->us));

  std::optional<Cost> const above = objective.cStar(TestKind::PRP, SHORT32_HI + 1);
  CHECK(above && above->measured());
  CHECK_EQ(above->entry, entryOf(objective, "3:1K:8:512:202").id);

  CHECK(near(objective.measured(), 0.75));
}

TEST(c_star_is_the_cheapest_entry_of_its_own_kind_that_covers_the_exponent) {
  TuneDB const db = loaded();
  RunScope const scope = scopeOver({{100'000'000, 1}});
  Objective const objective{db, 1, scope};

  // Two PRP entries cover 100M; the FP64 one is cheaper, whatever order they were measured in.
  SelectionEntry const fp64 = entryOf(objective, "512:15:512:212");
  SelectionEntry const ntt = entryOf(objective, "3:1K:8:512:202");
  CHECK(fp64.cost < ntt.cost);
  CHECK(ntt.emin <= 100'000'000 && 100'000'000 <= ntt.reach);
  std::optional<Cost> const prp = objective.cStar(TestKind::PRP, 100'000'000);
  CHECK(prp.has_value());
  CHECK_EQ(prp->entry, fp64.id);
  CHECK(near(objective.T(), fp64.cost));

  // The LL reading is cheaper than either, and covers the LL grid only.
  SelectionEntry const ll = entryOf(objective, "512:15:512:212", TestKind::LL);
  CHECK(ll.cost < fp64.cost);
  std::optional<Cost> const llCost = objective.cStar(TestKind::LL, 100'000'000);
  CHECK(llCost.has_value());
  CHECK_EQ(llCost->entry, ll.id);

  Objective const llOnly{db, 1, scopeOver({{100'000'000, 1}}, TestKind::LL)};
  CHECK(near(llOnly.T(), ll.cost));
}

TEST(a_kind_with_nothing_measured_of_its_own_is_priced_by_the_prior) {
  TuneDB const db = loaded("# prpll tunedb v1\n"
                           "env   1 gpu=\"X\" name=\"X\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0 machine=-"
                           " build=9a3f21c0d1e2f304\n"
                           "cfg   17 INPLACE=1\n"
                           "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
                           "run   4 512:15:512:212 ll 100000000 short32 17 1760.000 3.000 16 4 1.0000 ok 1\n");
  Objective const objective{db, 1, scopeOver({{100'000'000, 1}})};

  std::optional<Cost> const cost = objective.cStar(TestKind::PRP, 100'000'000);
  CHECK(cost && !cost->measured());
  CHECK_EQ(objective.measured(), 0.0);
}

TEST(rows_that_conclude_nothing_or_belong_elsewhere_count_for_nothing) {
  TuneDB const db = loaded();
  Objective const objective{db, 1, scopeOver({{100'000'000, 0.5}, {200'000'000, 0.5}})};

  // Env 2's 900 us/it is another card: it neither covers 100M for env 1 nor pulls env 1's prior down.
  std::optional<Cost> const at100M = objective.cStar(TestKind::PRP, 100'000'000);
  CHECK(at100M.has_value());
  CHECK_EQ(at100M->entry, entryOf(objective, "512:15:512:212").id);

  // The single call at 1K:8:1K is not a concluded row, so no entry covers 200M and it is still the prior's -- though
  // the prior, which only orders what is measured next, has learnt from that call.
  std::optional<Cost> const at200M = objective.cStar(TestKind::PRP, 200'000'000);
  CHECK(at200M && !at200M->measured());
  CHECK(at200M->us < 1000);
}

TEST(an_entry_the_gate_still_owes_a_reading_counts_only_for_valuing) {
  std::string text = DB;
  std::string const roe = "roe   4 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471330\n";
  text.erase(text.find(roe), roe.size());
  TuneDB const db = loaded(text.c_str());
  RunScope const scope = scopeOver({{100'000'000, 1}});

  // Published, 100M is the NTT's; to the search, which takes the reading next, it is already the FP64 set's.
  Objective const published{db, 1, scope};
  Objective const valuing{db, 1, scope, {}, Gating::Assumed};
  CHECK_EQ(published.cStar(TestKind::PRP, 100'000'000)->fft, std::string{"3:1K:8:512:202"});
  CHECK_EQ(valuing.cStar(TestKind::PRP, 100'000'000)->fft, std::string{"512:15:512:212"});
  CHECK(valuing.T() < published.T());
}

TEST(a_point_no_fft_can_run_is_left_out_of_T) {
  TuneDB const db = loaded();
  RunScope const scope = scopeOver({{1000, 0.25}, {100'000'000, 0.75}});
  Objective const objective{db, 1, scope};

  CHECK(!objective.cStar(TestKind::PRP, 1000).has_value());
  CHECK(!objective.prior(1000).has_value());
  CHECK(!objective.prior(20'000'000'000).has_value());

  CHECK(near(objective.unservable(), 0.25));
  CHECK(std::isfinite(objective.T()));
  CHECK(near(objective.T(), 0.75 * entryOf(objective, "512:15:512:212").cost));
}

TEST(the_prior_is_fitted_to_the_nearest_measured_size_of_the_type) {
  Prior prior;
  FFTShape const small{"256:2:256"};
  FFTShape const middle{"512:15:512"};
  FFTShape const large{"4K:16:1K"};
  FFTShape const ntt{"3:1K:8:512"};

  CHECK(!prior.fitted(FFT64));
  CHECK_EQ(prior.k(middle), statedPriorK(FFT64));

  prior.add(middle, 1800);
  CHECK(prior.fitted(FFT64));
  CHECK(!prior.fitted(FFT61));
  CHECK_EQ(prior.k(ntt), statedPriorK(FFT61));

  // One measured size fits every size of the type, and at its own size the prior is the measurement, made optimistic.
  double const kMiddle = 1800 * 1e6 / priorWork(middle.size());
  CHECK(near(prior.k(middle), kMiddle));
  CHECK(near(prior.k(small), kMiddle));
  CHECK(near(prior.k(large), kMiddle));
  CHECK(near(prior.cost(middle), PRIOR_OPTIMISM * 1800));
  CHECK(prior.cost(middle) < 1800);
  CHECK(near(prior.cost(large) / prior.cost(middle), priorWork(large.size()) / priorWork(middle.size())));

  // A second, smaller size takes over the shapes nearer to it.
  prior.add(small, 30);
  double const kSmall = 30 * 1e6 / priorWork(small.size());
  CHECK(near(prior.k(small), kSmall));
  CHECK(near(prior.k(FFTShape{"256:4:256"}), kSmall));
  CHECK(near(prior.k(FFTShape{"1K:12:512"}), kMiddle));
  CHECK(near(prior.k(large), kMiddle));

  // At a size read twice the cheaper reading is the fit, whichever came first.
  prior.add(middle, 2000);
  CHECK(near(prior.k(middle), kMiddle));
  prior.add(middle, 1500);
  CHECK(near(prior.k(middle), 1500 * 1e6 / priorWork(middle.size())));
}

TEST(a_size_equally_far_from_two_measurements_takes_the_cheaper) {
  Prior prior;
  // 256:4:256 is twice 256:2:256 and half 256:8:256, so it is one octave from each.
  prior.add(FFTShape{"256:2:256"}, 100);
  prior.add(FFTShape{"256:8:256"}, 300);
  double const kLow = 100 * 1e6 / priorWork(FFTShape{"256:2:256"}.size());
  double const kHigh = 300 * 1e6 / priorWork(FFTShape{"256:8:256"}.size());
  CHECK(kHigh < kLow);
  CHECK(near(prior.k(FFTShape{"256:4:256"}), kHigh));
}

TEST(a_tie_off_a_power_of_two_is_still_a_tie) {
  // 256:6:256 is 1.5 times 256:4:256 and two thirds of 256:9:256, so it is exactly as far from each -- which a
  // difference of two logarithms does not reliably say.
  FFTShape const low{"256:4:256"};
  FFTShape const middle{"256:6:256"};
  FFTShape const high{"256:9:256"};
  CHECK_EQ(u64(middle.size()) * middle.size(), u64(low.size()) * high.size());

  Prior prior;
  prior.add(low, 900);
  prior.add(high, 100);
  double const kLow = 900 * 1e6 / priorWork(low.size());
  double const kHigh = 100 * 1e6 / priorWork(high.size());
  CHECK(kHigh < kLow);
  CHECK(near(prior.k(middle), kHigh));

  // And the other way round, so that neither side wins the tie by position.
  Prior reversed;
  reversed.add(low, 100);
  reversed.add(high, 900);
  CHECK(near(reversed.k(middle), 100 * 1e6 / priorWork(low.size())));

  // Off the tie, the nearer size wins however cheap the farther one is: 256:5:256 is nearer 256:4:256.
  CHECK(near(prior.k(FFTShape{"256:5:256"}), kLow));
}

TEST(the_prior_is_fitted_from_the_env_it_is_asked_about) {
  TuneDB const db = loaded();

  Prior const one{db, 1};
  CHECK(one.fitted(FFT64));
  CHECK(one.fitted(FFT61));
  CHECK(!one.fitted(FFT3161));

  // The cheapest FP64 reading at 512:15:512 on env 1 is the LL one, 1760 -- a prior prices a shape, whichever kind
  // runs on it.  The one-call 100 at 1K:8:1K is a reading too, so it fits its own size and 4K:8:1K, its nearest; the
  // failed 100 at 4K:8:1K is not a cost and fits nothing, even at its own size.
  double const k = 1760 * 1e6 / priorWork(FFTShape{"512:15:512"}.size());
  double const one1K = 100 * 1e6 / priorWork(FFTShape{"1K:8:1K"}.size());
  CHECK(near(one.k(FFTShape{"512:15:512"}), k));
  CHECK(near(one.k(FFTShape{"1K:8:1K"}), one1K));
  CHECK(near(one.k(FFTShape{"4K:8:1K"}), one1K));

  Prior const two{db, 2};
  CHECK(near(two.k(FFTShape{"512:15:512"}), 900 * 1e6 / priorWork(FFTShape{"512:15:512"}.size())));
  CHECK(!two.fitted(FFT61));
}

TEST(every_type_production_can_choose_has_a_stated_prior) {
  for (const FFTShape& shape : FFTShape::allShapes()) {
    double const k = statedPriorK(shape.fft_type);
    CHECK(std::isfinite(k));
    CHECK(k > 0);
  }
}

TEST(the_prior_at_an_exponent_is_the_cheapest_shape_that_can_run_it) {
  Objective const objective{Env{}, scopeOver({{100'000'000, 1}})};

  std::optional<Cost> const at = objective.prior(100'000'000);
  CHECK(at.has_value());

  // Nothing any type has that can run 100M is priced below it.
  Prior const nothing;
  for (const FFTShape& shape : FFTShape::allShapes()) {
    FFTConfig const fft{shape, defaultVariant(shape), CARRY_AUTO};
    if (isEligible(fft, 100'000'000) && fft.maxExp() >= 100'000'000) { CHECK(nothing.cost(shape) >= at->us); }
  }
  CHECK(near(at->us, nothing.cost(FFTShape{at->fft})));
}
