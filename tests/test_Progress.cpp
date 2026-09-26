// Copyright (C) Jason Lynch

// Tests what a run shows while it runs: the periodic line, the heartbeat, the line drawn at the bottom of a terminal
// and the bytes that keep it there while log() writes above it, all as pure functions of what the queue reports and of
// the clock; and that log()'s stdout copy can be taken.

#include "Progress.h"

#include "log.h"

#include "test.h"

#include <string>
#include <vector>

using namespace tune;

namespace {

Item item(ItemKind kind, double value, double seconds) {
  Item out;
  out.kind = kind;
  out.value = value;
  out.seconds = seconds;
  return out;
}

// A run 19:41 in, with figures of the size a run at 24M gives.
RunProgress midRun() {
  return {.items = 212,
          .anchors = 4,
          .spent = {{ItemKind::Baseline, {.items = 60, .seconds = 186}},
                    {ItemKind::Probe, {.items = 140, .seconds = 420}},
                    {ItemKind::Gate, {.items = 12, .seconds = 30}}},
          .startT = 217.453,
          .T = 199.518,
          .measured = 0.914,
          .stop = 0.001,
          .floor = 0.1995,
          .mostWorth = 0.363,
          .worthRunning = 420,
          .worthSeconds = 1320};
}

}  // namespace

TEST(a_span_reads_as_a_clock) {
  CHECK_EQ(clockText(0), std::string{"0:00"});
  CHECK_EQ(clockText(59.9), std::string{"0:59"});
  CHECK_EQ(clockText(61), std::string{"1:01"});
  CHECK_EQ(clockText(3599), std::string{"59:59"});
  CHECK_EQ(clockText(3600), std::string{"1:00:00"});
  CHECK_EQ(clockText(3725), std::string{"1:02:05"});
  CHECK_EQ(clockText(-3), std::string{"0:00"});
}

TEST(what_is_worth_running_is_the_stopping_rules_own_view_of_the_ranking) {
  QueueReport sofar;
  sofar.items = 7;
  sofar.anchors = 1;
  sofar.startT = 210;
  sofar.spent[ItemKind::Probe] = {.items = 7, .seconds = 70};

  // A bootstrap call and a gate reading run by rule whatever they are valued at, so neither says how near the stop is.
  std::vector<Item> const ranked{item(ItemKind::Bootstrap, 90, 5),  item(ItemKind::Gate, 50, 3),
                                 item(ItemKind::Probe, 0.5, 20),    item(ItemKind::Restart, 0.3, 10),
                                 item(ItemKind::Baseline, 0.1, 30), item(ItemKind::Refine, 0, 4)};
  RunProgress const p = progressOf(sofar, 200, 0.75, ranked, 0.001, 0.2, true);

  CHECK_EQ(p.items, 7u);
  CHECK_EQ(p.anchors, 1u);
  CHECK_EQ(p.spent.at(ItemKind::Probe).items, 7u);
  CHECK_EQ(p.startT, 210.0);
  CHECK_EQ(p.T, 200.0);
  CHECK_EQ(p.measured, 0.75);
  CHECK_EQ(p.mostWorth, 0.5);
  CHECK_EQ(p.worthRunning, 4u);
  CHECK_EQ(p.worthSeconds, 38.0);
  CHECK(p.bootstrapComplete);

  // What stages listed in part have past what is listed counts where the item that says it is worth running.
  std::vector<Item> listed = ranked;
  listed[2].unlisted = 843'000'000;
  listed[4].unlisted = 1000;
  CHECK_EQ(progressOf(sofar, 200, 0.75, listed, 0.001, 0.2, true).unlisted, u64(843'000'000));
  CHECK_EQ(progressOf(sofar, 200, 0.75, listed, 0, 0, true).unlisted, u64(843'001'000));

  // With no stop fraction everything worth anything runs, and nothing is worth running where nothing is valued.
  CHECK_EQ(progressOf(sofar, 200, 0.75, ranked, 0, 0, true).worthRunning, 5u);
  CHECK_EQ(progressOf(sofar, 200, 0.75, {item(ItemKind::Refine, 0, 4)}, 0, 0, true).worthRunning, 0u);
}

TEST(the_periodic_line_says_where_the_run_stands_against_its_start_and_its_stop) {
  CHECK_EQ(progressLine(midRun(), 1181),
           std::string{"19:41 in, 212 items and 4 anchor readings: 60 baseline (3.1 min), 140 probe (7.0 min), 12 gate "
                       "(0.5 min); T 217.453 -> 199.518 us/it, 91.4% of the weight on measured entries; the most an "
                       "item is worth is 0.3630 us/it, 1.8x the stop floor of 0.1995 (stop=0.1% of T); 420 items are "
                       "worth running now, ~22 min by the queue's estimates"});

  RunProgress unbounded = midRun();
  unbounded.stop = 0;
  unbounded.floor = 0;
  unbounded.worthRunning = 1;
  unbounded.worthSeconds = 40;
  CHECK(progressLine(unbounded, 3725).starts_with("1:02:05 in, "));
  CHECK(progressLine(unbounded, 3725)
          .ends_with("; the most an item is worth is 0.3630 us/it, and stop=0 runs until stopped; 1 item is worth "
                     "running now, ~40 s by the queue's estimates"));

  // Stages listed a window at a time say how much more they could list.
  RunProgress lifted = unbounded;
  lifted.unlisted = 5'906'934'895;
  CHECK(progressLine(lifted, 3725)
          .ends_with("; 1 item is worth running now, ~40 s by the queue's estimates, and up to 5906934895 more not "
                     "listed yet"));

  // Before anything is valued only what runs by rule is worth running, and at the end nothing is.
  RunProgress byRule = midRun();
  byRule.mostWorth = 0;
  byRule.worthRunning = 3;
  byRule.worthSeconds = 150;
  CHECK(progressLine(byRule, 60)
          .ends_with("measured entries; 3 items are worth running now, ~2 min by the queue's "
                     "estimates"));

  // The first T is the prior's where nothing is measured, which is why the line says how much is.
  RunProgress fresh;
  fresh.anchors = 1;
  fresh.startT = 150.333;
  fresh.T = 150.333;
  CHECK_EQ(progressLine(fresh, 300),
           std::string{"5:00 in, 0 items and 1 anchor reading; T 150.333 -> 150.333 us/it, 0.0% of the weight on "
                       "measured entries; nothing is worth running now"});
}

TEST(a_heartbeat_names_the_call_and_how_long_it_has_run) {
  CHECK_EQ(heartbeatLine("213. probe 512:5:256:202 prp short32 PAD=512 at 24698117", 125),
           std::string{"still measuring 213. probe 512:5:256:202 prp short32 PAD=512 at 24698117, 2:05 so far"});
}

TEST(the_drawn_line_never_wraps_and_keeps_both_ends_of_what_it_names) {
  std::string const what = "213. probe 512:5:256:202 prp short32 Placement 2 IN_SIZEX=32,OUT_SIZEX=32 at 24698117";

  std::string const wide = liveLine(midRun(), what, 1184, 3, 200);
  CHECK_EQ(wide, "[19:44] " + what + " (0:03) | T 199.518 us/it (91% measured) | worth 1.8x stop");

  for (size_t cols : {200u, 120u, 80u, 60u, 40u, 20u, 2u, 1u, 0u}) {
    std::string const line = liveLine(midRun(), what, 1184, 3, cols);
    CHECK(line.size() + 1 <= std::max<size_t>(cols, 1));
    CHECK(line.find('\n') == std::string::npos);
    CHECK(line.find('\r') == std::string::npos);
  }

  // Narrowed, what is being measured keeps its room: the figures beside it are given up from the last, and then its
  // middle goes, keeping its entry and its exponent.
  std::string const narrow = liveLine(midRun(), what, 1184, 3, 100);
  CHECK_EQ(narrow.size(), size_t{99});
  CHECK(narrow.starts_with("[19:44] 213. probe 512:5"));
  CHECK(narrow.find("...") != std::string::npos);
  CHECK(narrow.ends_with("at 24698117 (0:03) | T 199.518 us/it (91% measured)"));

  CHECK_EQ(liveLine(midRun(), "31. baseline 256:10:256:102 prp short32 at 24691343", 119, 2, 80),
           std::string{"[1:59] 31. baseline 256:10:256:102 prp short32 at 24691343 (0:02)"});

  // Between calls, and before the first figures, it says so.
  CHECK_EQ(liveLine(midRun(), "", 1184, 0, 200),
           std::string{"[19:44] choosing the next item | T 199.518 us/it (91% measured) | worth 1.8x stop"});
  CHECK_EQ(liveLine(RunProgress{}, "the drift anchor", 5, 5, 200), std::string{"[0:05] the drift anchor (0:05)"});

  // A short name is not cut to make room, and gives up only what does not fit beside it.
  CHECK_EQ(liveLine(midRun(), "the drift anchor", 8, 7, 80),
           std::string{"[0:08] the drift anchor (0:07) | T 199.518 us/it (91% measured)"});

  RunProgress unbounded = midRun();
  unbounded.stop = 0;
  unbounded.floor = 0;
  CHECK(liveLine(unbounded, "", 1184, 0, 200).ends_with("| worth 0.3630 us/it"));
}

TEST(the_periodic_line_and_the_heartbeat_are_due_on_their_own_clocks) {
  CHECK(!progressDue(299.9, 0));
  CHECK(progressDue(300, 0));
  CHECK(!progressDue(500, 300));
  CHECK(progressDue(600, 300));

  CHECK_EQ(heartbeatsDue(0), 0u);
  CHECK_EQ(heartbeatsDue(59.9), 0u);
  CHECK_EQ(heartbeatsDue(60), 1u);
  CHECK_EQ(heartbeatsDue(150), 2u);
  CHECK_EQ(heartbeatsDue(-1), 0u);
}

TEST(a_log_line_is_written_above_the_drawn_line) {
  CHECK_EQ(redraw("[0:05] x"), std::string{"\r[0:05] x\x1b[K"});
  CHECK_EQ(redraw(""), std::string{"\r\x1b[K"});

  // Drawn: cleared, the text, and drawn again below it.
  Paint const over = repaint("20260925 19:25:19  tune: 1. baseline\n", true, "[0:05] x");
  CHECK_EQ(over.bytes, std::string{"\r\x1b[K20260925 19:25:19  tune: 1. baseline\n\r[0:05] x\x1b[K"});
  CHECK(over.drawn);

  // Not drawn yet: nothing to clear.
  Paint const first = repaint("a\n", false, "[0:00] y");
  CHECK_EQ(first.bytes, std::string{"a\n\r[0:00] y\x1b[K"});
  CHECK(first.drawn);

  // Text that has not ended its line is left to end it; the line is not drawn onto the end of it.
  Paint const partial = repaint("a", true, "[0:00] y");
  CHECK_EQ(partial.bytes, std::string{"\r\x1b[Ka"});
  CHECK(!partial.drawn);

  // Nothing to draw.
  Paint const none = repaint("a\n", false, "");
  CHECK_EQ(none.bytes, std::string{"a\n"});
  CHECK(!none.drawn);
}

TEST(logs_stdout_copy_can_be_taken_and_given_back) {
  std::vector<std::string> taken;
  setStdoutSink([&](std::string_view s) { taken.emplace_back(s); });
  log("progress sink %d\n", 42);
  setStdoutSink(nullptr);

  CHECK_EQ(taken.size(), size_t{1});
  CHECK(taken.front().ends_with(" progress sink 42\n"));
}
