// Copyright (C) Jason Lynch

// GPU-free tests of the state one generation hands to the next. The exec itself is not testable in process.

#include "test.h"

#include "Restart.h"

#include <cstdlib>
#include <string>

TEST(the_generation_count_survives_the_environment) {
  CHECK_EQ(restart::parseCarry("0"), 0u);
  CHECK_EQ(restart::parseCarry("7"), 7u);
}

TEST(a_carry_value_that_makes_no_sense_starts_over_rather_than_refusing_to_run) {
  // The variable is this program's own, but a user can set anything, and a generation count that will not parse is no
  // reason to refuse to measure: the restart cap is a backstop, not the mechanism that ends the loop.
  CHECK_EQ(restart::parseCarry(nullptr), 0u);
  CHECK_EQ(restart::parseCarry(""), 0u);
  CHECK_EQ(restart::parseCarry("not a number"), 0u);
  CHECK_EQ(restart::parseCarry("3 and some trailing text"), 3u);
}

#ifndef _WIN32
TEST(a_process_that_cannot_restart_still_gives_back_what_it_was_showing) {
  // The hook is what hands a terminal back; a process past the restart limit is about to end, so it needs it as much.
  const char* const before = getenv("PRPLL_MAX_RESTARTS");
  std::string const saved = before ? before : "";
  setenv("PRPLL_MAX_RESTARTS", "0", 1);

  bool ran = false;
  restart::beforeExec([&] { ran = true; });
  std::string const why = restart::reexec();
  restart::beforeExec(nullptr);

  if (before) {
    setenv("PRPLL_MAX_RESTARTS", saved.c_str(), 1);
  } else {
    unsetenv("PRPLL_MAX_RESTARTS");
  }
  CHECK(ran);
  CHECK(why.find("which is the limit") != std::string::npos);
}
#endif
