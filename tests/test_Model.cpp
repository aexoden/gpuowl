// Copyright (C) Jason Lynch

#include "test.h"

#include "Axes.h"
#include "FFTVariants.h"
#include "Model.h"

#include <cmath>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

Subject subject(const char* spec) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft, .kind = TestKind::PRP, .regime = regimeOf(fft, 67'000'000)};
}

Observation obs(const Subject& s, UseConfig options, double cost) {
  return {.fft = s.fft, .kind = s.kind, .regime = s.regime, .options = std::move(options), .cost = cost};
}

double us(double log) { return std::exp(log); }

}  // namespace

TEST(model_recovers_an_effect_and_carries_it_to_another_entry) {
  Subject const a = subject("1K:13:256:202");
  Subject const b = subject("512:15:512:202");

  // ZEROHACK_W=0 costs 5% on both entries; b is only ever read at its defaults.
  std::vector<Observation> data;
  for (int i = 0; i < 6; ++i) {
    data.push_back(obs(a, {}, 1000));
    data.push_back(obs(a, {{"ZEROHACK_W", "0"}}, 1050));
    data.push_back(obs(b, {}, 800));
  }
  Model m{nvidia()};
  m.fit(data);

  Draw const mean = m.mean();
  CHECK(std::abs(us(m.predict(mean, a, {})) - 1000) < 5);
  CHECK(std::abs(us(m.predict(mean, a, {{"ZEROHACK_W", "0"}})) - 1050) < 5);
  CHECK(std::abs(us(m.predict(mean, b, {})) - 800) < 5);

  // Never measured on b: predicted from what a showed, shrunk a little toward no effect.
  double const moved = us(m.predict(mean, b, {{"ZEROHACK_W", "0"}})) / 800 - 1;
  CHECK(moved > 0.03 && moved < 0.055);
}

TEST(model_energy_agrees_with_its_predictions) {
  Subject const a = subject("1K:13:256:202");
  std::vector<Observation> data{obs(a, {}, 1000), obs(a, {{"ZEROHACK_W", "0"}}, 1050),
                                obs(a, {{"TAIL_KERNELS", "0"}}, 990)};
  Model m{nvidia()};
  m.fit(data);

  for (u64 seed : {0ull, 7ull}) {
    Draw const d = seed ? m.draw(seed) : m.mean();
    UseConfig const options{{"TAIL_KERNELS", "0"}, {"ZEROHACK_W", "0"}};
    std::vector<Axis> const axes = axesOf(nvidia(), a.fft, options);
    Energy const e = m.energy(d, a, options, axes);

    double sum = e.constant;
    for (size_t i = 0; i < axes.size(); ++i) { sum += e.unary[i][axes[i].current]; }
    for (const Energy::Pair& p : e.pairs) {
      sum += p.table[axes[p.a].current * axes[p.b].values.size() + axes[p.b].current];
    }

    // The energy leaves out the configuration's residual, which a configuration never measured has only in a draw.
    double const predicted = m.predict(d, a, options);
    CHECK(std::abs(predicted - sum) < (seed ? 0.05 : 1e-9));
  }
}

TEST(model_draws_spread_where_nothing_was_measured) {
  Subject const a = subject("1K:13:256:202");
  Subject const unseen = subject("2:512:4:512:212");
  std::vector<Observation> data;
  for (int i = 0; i < 4; ++i) { data.push_back(obs(a, {}, 1000)); }
  Model m{nvidia()};
  m.fit(data);

  double sumA = 0, sumA2 = 0, sumU = 0, sumU2 = 0;
  int const n = 40;
  for (int i = 1; i <= n; ++i) {
    Draw const d = m.draw(u64(i));
    double const pa = m.predict(d, a, {});
    double const pu = m.predict(d, unseen, {});
    sumA += pa;
    sumA2 += pa * pa;
    sumU += pu;
    sumU2 += pu * pu;
  }
  double const sdA = std::sqrt(sumA2 / n - (sumA / n) * (sumA / n));
  double const sdU = std::sqrt(sumU2 / n - (sumU / n) * (sumU / n));
  CHECK(sdA < 0.02);
  CHECK(sdU > 0.2);
}

TEST(model_draws_are_reproducible_and_scaled) {
  Subject const a = subject("1K:13:256:202");
  std::vector<Observation> data{obs(a, {}, 1000), obs(a, {{"ZEROHACK_W", "0"}}, 1050)};
  Model m{nvidia()};
  m.fit(data);

  UseConfig const other{{"TAIL_KERNELS", "0"}};
  double const d1 = m.predict(m.draw(5), a, other);
  CHECK_EQ(d1, m.predict(m.draw(5), a, other));

  double const centre = m.predict(m.mean(), a, other);
  double const wide = m.predict(m.draw(5, 2), a, other);
  CHECK(std::abs((wide - centre) - 2 * (d1 - centre)) < 1e-9);
}

TEST(model_pairs_only_within_a_group_or_a_shared_kernel) {
  Env const env = nvidia();
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  std::vector<Axis> const axes = axesOf(env, fft, {});
  auto find = [&](const std::string& name) -> const Axis& {
    for (const Axis& a : axes) {
      if (a.name == name) { return a; }
    }
    throw "no axis " + name;
  };
  CHECK(Model::interact(env, fft, {}, find("ZEROHACK_W"), find("SHUFL_BYTES_W")));
  CHECK(!Model::interact(env, fft, {}, find("ZEROHACK_W"), find("INPLACE")));
}

TEST(a_few_huge_effects_do_not_widen_every_prior) {
  // Twelve options a percent either way, and two that cost 40% -- a register limit that spills: the effects' spread is
  // learnt as the small ones show it.
  Subject const a = subject("1K:13:256:202");
  std::vector<Axis> const axes = axesOf(nvidia(), a.fft, {});
  CHECK(axes.size() >= 14);
  std::vector<Observation> data;
  for (int i = 0; i < 4; ++i) { data.push_back(obs(a, {}, 1000)); }
  for (size_t k = 0; k < 14 && k < axes.size(); ++k) {
    UseConfig options;
    place(options, axes[k], axes[k].reference == 0 ? 1 : 0);
    double const cost = k < 12 ? 1000 * (k % 2 ? 1.01 : 0.99) : 1400;
    for (int i = 0; i < 4; ++i) { data.push_back(obs(a, fitted(nvidia(), a.fft, options), cost)); }
  }
  Model m{nvidia()};
  m.fit(data);
  CHECK(m.effectSpread() < 0.03);
  // The huge ones are fitted all the same.
  UseConfig huge;
  place(huge, axes[12], axes[12].reference == 0 ? 1 : 0);
  CHECK(std::abs(us(m.predict(m.mean(), a, fitted(nvidia(), a.fft, huge))) / 1400 - 1) < 0.02);
}

TEST(a_spill_is_the_spills_not_its_options) {
  // ZEROHACK_W=0 saves 4% on both FFTs, and REGCF64=7 nothing; on a, a build with both spilled and cost four times as
  // much.
  Subject const a = subject("1K:13:256:202");
  Subject const b = subject("512:15:512:202");
  auto fitWith = [&](bool marked) {
    std::vector<Observation> data;
    for (int i = 0; i < 4; ++i) {
      data.push_back(obs(a, {}, 1000));
      data.push_back(obs(a, {{"ZEROHACK_W", "0"}}, 960));
      data.push_back(obs(a, {{"REGCF64", "7"}}, 1000));
      data.push_back(obs(b, {}, 800));
      Observation spill = obs(a, {{"REGCF64", "7"}, {"ZEROHACK_W", "0"}}, 3840);
      if (marked) { spill.spilled = {"REGCF64"}; }
      data.push_back(spill);
    }
    Model m{nvidia()};
    m.fit(data);
    return m;
  };
  Model const unmarked = fitWith(false);
  Model const marked = fitWith(true);
  auto saving = [&](const Model& m) { return 1 - us(m.predict(m.mean(), b, {{"ZEROHACK_W", "0"}})) / 800; };
  // Unmarked, the spill is shared with ZEROHACK_W=0, which then looks dearer than it is elsewhere.
  CHECK(saving(unmarked) < 0.03);
  CHECK(saving(marked) > 0.03 && saving(marked) < 0.05);
  // And the configuration that spilled is predicted as it was read.
  CHECK(std::abs(us(marked.predict(marked.mean(), a, {{"REGCF64", "7"}, {"ZEROHACK_W", "0"}})) / 3840 - 1) < 0.05);
}
