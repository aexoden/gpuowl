// Copyright (C) Jason Lynch

#include "test.h"

#include "Gpu.h"
#include "Measure.h"

#include <string>
#include <vector>

using namespace tune;

namespace {

Call reading(double mean, u64 res64 = 0) {
  Call c;
  c.core.stats = {.mean = mean, .sd = 0.1, .n = 4};
  c.res64 = res64;
  return c;
}

Call failing() {
  Call c;
  c.status = Status::Err;
  return c;
}

}  // namespace

TEST(summarize_drops_spikes_and_marks_a_failed_check) {
  IterSamples const samples{.usPerIt = {100.0, 100.1, 99.9, 130.0}, .checkOk = true, .res64 = 7, .iters = 5000};
  Call const c = summarize(samples);
  CHECK(c.ok());
  CHECK_EQ(c.core.dropped, 1u);
  CHECK_EQ(c.core.stats.n, 3u);
  CHECK(c.mean() > 99.9 && c.mean() < 100.1);
  CHECK_EQ(c.usPerIt.size(), size_t{4});
  CHECK_EQ(c.res64, u64{7});

  IterSamples bad = samples;
  bad.checkOk = false;
  CHECK(summarize(bad).status == Status::Err);
  CHECK(!summarize(bad).ok());
}

TEST(take_over_sets_aside_every_tunable_key) {
  Args shared{true};
  shared.flags = {{"INPLACE", "0"}, {"NO_ASM", "1"}, {"PAD", "64"}, {"NOT_A_KEY", "3"}};
  shared.perFftConfig["512:15:512"] = {{"WMUL", "1"}};
  shared.carry = CARRY_64;

  Takeover const t = takeOver(shared);
  CHECK(t.args.flags == (std::map<std::string, std::string>{{"NO_ASM", "1"}}));
  CHECK(t.args.perFftConfig.empty());
  CHECK(t.args.carry == CARRY_AUTO);
  CHECK(t.carry);
  CHECK(t.setAside == (std::set<std::string>{"INPLACE", "NOT_A_KEY", "PAD", "WMUL"}));

  // The shared args are untouched.
  CHECK_EQ(shared.flags.size(), size_t{4});
}

TEST(a_failed_check_is_read_again_before_it_is_believed) {
  Call const err = failing();
  Call const good = reading(100);

  u32 reads = 0;
  auto againGood = [&] {
    ++reads;
    return good;
  };
  CHECK(checkedTwice(err, againGood).ok());
  CHECK_EQ(reads, 1u);

  reads = 0;
  CHECK(checkedTwice(good, againGood).ok());
  CHECK_EQ(reads, 0u);

  CHECK(checkedTwice(err, failing).status == Status::Err);
}

TEST(an_ll_residue_is_read_again_before_it_is_believed) {
  Call const right = reading(100, 42);
  Call const wrong = reading(100, 41);

  CHECK_EQ(checkedAgainst(42, right, [] { return Call{}; }).res64, u64{42});
  CHECK(checkedAgainst(42, wrong, [&] { return right; }).ok());
  Call const twice = checkedAgainst(42, wrong, [&] { return wrong; });
  CHECK(twice.status == Status::Err);
}

TEST(failures_are_classified_by_what_they_mean_for_the_configuration) {
  CHECK(classify("stop requested").stop);
  CHECK(!classify("stop requested").fatal);
  CHECK(classify("CL_DEVICE_NOT_AVAILABLE loading fftw.cl").fatal);
  CHECK(classify("Can't compile carryfused.cl").status == Status::NoCompile);
  CHECK(classify("workgroup size too large").status == Status::Unsupported);
}

TEST(a_call_is_counted_over_its_warm_up_too) { CHECK_EQ(callIterations(4, 1000), u64{5000}); }
