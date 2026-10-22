// Copyright (C) Jason Lynch

#include "test.h"

#include "FFTVariants.h"
#include "Optimize.h"

#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

Subject subject(const char* spec) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft, .kind = TestKind::PRP, .regime = regimeOf(fft, 67'000'000)};
}

}  // namespace

TEST(optimize_finds_a_pair_that_loses_alone) {
  Subject const s = subject("1K:13:256:202");
  auto obs = [&](UseConfig o, double cost) {
    return Observation{.fft = s.fft, .kind = s.kind, .regime = s.regime, .options = std::move(o), .cost = cost};
  };

  std::vector<Observation> data;
  for (int i = 0; i < 8; ++i) {
    data.push_back(obs({}, 1000));
    data.push_back(obs({{"ZEROHACK_W", "0"}}, 1020));
    data.push_back(obs({{"SHUFL_BYTES_W", "16"}}, 1020));
    data.push_back(obs({{"SHUFL_BYTES_W", "16"}, {"ZEROHACK_W", "0"}}, 950));
  }
  Model m{nvidia()};
  m.fit(data);

  Optimum const o = optimize(m, m.mean(), s, {});
  CHECK(o.exact);
  // Under the mean every untried position is a tie with the default, which wins it.
  CHECK(o.options == (UseConfig{{"SHUFL_BYTES_W", "16"}, {"ZEROHACK_W", "0"}}));
}

TEST(optimize_returns_a_configuration_the_table_offers) {
  Subject const s = subject("2:512:4:512:212");
  Model m{nvidia()};
  m.fit({});
  for (u64 seed = 1; seed <= 5; ++seed) {
    Optimum const o = optimize(m, m.draw(seed), s, {.tailQueues = TailQueues::Ntt});
    CHECK(o.options == fitted(nvidia(), s.fft, o.options));
  }
}
