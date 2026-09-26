// Copyright (C) Jason Lynch

// Tests what a status reads of a database file that a run may be appending to, and how it says how long ago.  What it
// makes of the database is tested against the queue itself, in test_Scheduler.

#include "Status.h"

#include "test.h"

#include <string>
#include <string_view>

using namespace tune;

TEST(a_status_reads_only_the_lines_a_run_has_finished_writing) {
  CHECK_EQ(completeLines("# prpll tunedb v1\nrun   4 512:15:512:212 prp 1180"),
           std::string_view{"# prpll tunedb v1\n"});
  CHECK_EQ(completeLines("# prpll tunedb v1\ncfg   1 -\n"), std::string_view{"# prpll tunedb v1\ncfg   1 -\n"});
  CHECK_EQ(completeLines("# prpll tu"), std::string_view{});
  CHECK_EQ(completeLines(""), std::string_view{});
}

TEST(how_long_ago_is_said_in_the_unit_that_reads_best) {
  CHECK_EQ(ago(0), std::string{"0 s"});
  CHECK_EQ(ago(119), std::string{"119 s"});
  CHECK_EQ(ago(120), std::string{"2 min"});
  CHECK_EQ(ago(2 * 3600 - 1), std::string{"119 min"});
  CHECK_EQ(ago(2 * 3600), std::string{"2 h"});
  CHECK_EQ(ago(48 * 3600 - 1), std::string{"47 h"});
  CHECK_EQ(ago(48 * 3600), std::string{"2 days"});
}
