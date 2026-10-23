// Copyright (C) Jason Lynch

#include "test.h"

#include "Acquire.h"
#include "FFTVariants.h"
#include "Model.h"

#include <random>
#include <vector>

using namespace tune;

namespace {

Subject subject(const char* spec) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft, .kind = TestKind::LL, .regime = regimeOf(fft, 69'000'000)};
}

}  // namespace

TEST(every_subject_starts_in_either_branch_about_half_the_time) {
  std::vector<Subject> subjects;
  for (const char* spec : {"512:7:512:012", "512:7:512:010", "1K:13:256:202", "2:512:4:512:212", "4:256:11:512",
                           "1:512:8:512", "1K:7:256:110", "512:8:512:000"}) {
    subjects.push_back(subject(spec));
  }
  // Subjects whose hashes differ in parity, which a seed of fixed parity would send one way each.
  bool odd = false;
  bool even = false;
  for (const Subject& s : subjects) {
    (termHash(s.fft.spec() + toString(s.kind) + s.regime.label()) & 1 ? odd : even) = true;
  }
  CHECK(odd && even);

  for (const Subject& s : subjects) {
    std::mt19937_64 rng{12'345};
    u32 other = 0;
    constexpr u32 N = 10'000;
    for (u32 i = 0; i < N; ++i) { other += startsInOtherBranch(drawSeed(rng), s); }
    // Within about four standard deviations of half.
    CHECK(other > N / 2 - 200 && other < N / 2 + 200);
  }
}

TEST(a_draw_seed_keeps_both_parities_and_is_never_the_means) {
  std::mt19937_64 rng{1};
  u32 odd = 0;
  for (u32 i = 0; i < 1000; ++i) {
    u64 const seed = drawSeed(rng);
    CHECK(seed != 0);
    odd += seed & 1;
  }
  CHECK(odd > 400 && odd < 600);
}
