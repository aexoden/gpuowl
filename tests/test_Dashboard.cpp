// Copyright (C) Jason Lynch

#include "test.h"

#include "Dashboard.h"

#include <string>

using namespace tune;

namespace {

Board busyBoard() {
  Board b;
  b.device = "NVIDIA RTX A4000";
  b.scope = "ll 67000000-72000000, probe 69454999";
  b.elapsed = 3725;
  b.started = true;
  b.what = "ll 1:256:7:512:202 -use INPLACE=0";
  b.state = {.T = 633.8,
             .coverage = 1,
             .regret = 0.0057,
             .calls = 177,
             .lines = {"ll   633.8 1:256:7:512:202  to 72904352 INPLACE=0,LOADS=3200"},
             .observations = 170,
             .spreads = "type 0.5000 kind 0.1000",
             .faults = {"failed: 3:1K:8:256:202 ll -use LOADS=5"}};
  b.T = {690, 650, 640, 634};
  b.regret = {0.05, 0.01, 0.0057};
  for (u32 n = 1; n <= 30; ++n) {
    b.recent.push_front({.n = n,
                         .kind = "call",
                         .what = "ll 1:256:7:512:202",
                         .options = "INPLACE=0",
                         .result = "640.0 us/it",
                         .measured = 640,
                         .predicted = 630});
  }
  b.events = {"21:00:00 wrote tune-ll.txt: 1 lines"};
  return b;
}

}  // namespace

TEST(the_frame_fits_any_terminal) {
  Board const b = busyBoard();
  for (size_t const rows : {1, 10, 24, 60}) {
    for (size_t const cols : {2, 40, 80, 200}) {
      std::vector<Row> const f = frame(b, rows, cols);
      CHECK(f.size() <= rows);
      for (const Row& row : f) { CHECK(columns(row) < cols); }
    }
  }
}

TEST(the_frame_shows_what_matters) {
  std::string all;
  for (const Row& row : frame(busyBoard(), 60, 200)) { all += plain(row) + '\n'; }
  CHECK(all.find("T 633.80 us/it") != std::string::npos);
  CHECK(all.find("expected regret 0.57%") != std::string::npos);
  CHECK(all.find("FAULTS") != std::string::npos);
  CHECK(all.find("PUBLISHED") != std::string::npos);
  CHECK(all.find("predicted/measured 0.984") != std::string::npos);
  CHECK(all.find("wrote tune-ll.txt") != std::string::npos);
}

TEST(item_lines_are_not_events) {
  CHECK(!eventOf("20261022 06:03:36  tune: 12. ll 1:256:7:512:202 -use x at 1: 640.0 us/it").has_value());
  CHECK_EQ(*eventOf("20261022 06:03:36  tune: wrote tune-ll.txt: 1 lines"),
           std::string{"06:03:36 wrote tune-ll.txt: 1 lines"});
  CHECK(!eventOf("20261022 06:03:36   ").has_value());
}
