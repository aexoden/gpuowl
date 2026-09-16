// Copyright (C) Jason Lynch

#include "test.h"

#include "Args.h"
#include "Gpu.h"

#include <type_traits>

static_assert(std::is_same_v<decltype(Gpu::args), Args>, "Gpu must hold Args by value, not by reference");
static_assert(std::is_copy_constructible_v<Args>);

TEST(args_copy_is_independent) {
  Args original;
  original.flags["L2_STRIPING"] = "8";
  original.perFftConfig["512:15:512"] = {{"PAD", "256"}};
  original.verbose = 1;

  Args copy = original;
  copy.flags["L2_STRIPING"] = "0";
  copy.flags["WMUL"] = "1";
  copy.perFftConfig["512:15:512"].clear();
  copy.verbose = 2;

  CHECK_EQ(original.value("L2_STRIPING", -1), 8);
  CHECK_EQ(original.value("WMUL", -1), -1);
  CHECK_EQ(original.perFftConfig["512:15:512"].size(), size_t{1});
  CHECK_EQ(original.verbose, 1);

  CHECK_EQ(copy.value("L2_STRIPING", -1), 0);
  CHECK_EQ(copy.value("WMUL", -1), 1);
  CHECK(copy.perFftConfig["512:15:512"].empty());
}
