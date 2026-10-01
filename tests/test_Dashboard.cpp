// Copyright (C) Jason Lynch

// Tests the dashboard's frame as a pure function of a Board and the terminal's size: that it never outgrows the
// terminal, that faults sit under the header whenever there are any, what each panel holds, and the bytes that paint
// it; and the pure parts that fill the Board -- the stretches production runs, the phase, the log's lines as events.

#include "Dashboard.h"

#include "test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tune;

namespace {

ObjectivePoint point(u64 exponent, double weight, const std::string& entry, const std::string& fft, double us) {
  return {
    .kind = TestKind::PRP, .exponent = exponent, .weight = weight, .cost = Cost{.us = us, .entry = entry, .fft = fft}};
}

Finished timing(u32 n, const std::string& label, double us, double before, double after) {
  Finished f;
  f.n = n;
  f.kind = ItemKind::Probe;
  f.label = label;
  f.exponent = 67'513'549;
  f.completed = true;
  f.seconds = 6.2;
  f.usPerIt = us;
  f.before = before;
  f.after = after;
  return f;
}

// What the events panel makes of a line, or "-" for nothing.
std::string event(std::string_view line) { return eventOf(line).value_or("-"); }

// A run five minutes into its third session on a P100, with one configuration that took the device down.
Board board(bool unicode) {
  Board b;
  b.device = "Tesla P100-PCIE-16GB, OpenCL, device 1, env 2";
  b.scope = "prp 67000000-80000000, probe 67513549";
  b.generation = 1;
  b.elapsed = 300;
  b.before = 3300;
  b.sessions = 3;
  b.started = true;
  b.phase = "searching";
  b.progress.items = 12;
  b.progress.T = 1124.006;
  b.progress.measured = 1;
  b.progress.stop = 0.001;
  b.progress.floor = 1.124;
  b.progress.mostWorth = 2.9;
  b.progress.worthRunning = 40;
  b.progress.worthSeconds = 900;
  b.what = "13. probe 512:7:512:202 prp short32 Memory 2 LOADS=42552,STORES=3 at 67513549 (call 2)";
  b.onIt = 17;
  b.benefit = {.kind = TestKind::PRP,
               .probe = 67'513'549,
               .now = Cost{.us = 502.712, .entry = "e1", .fft = "512:7:512:202"},
               .untunedFft = "1K:7:256:212",
               .untuned = 523.907};
  for (int k = 0; k <= 12; ++k) {
    b.history.push_back({.active = 300.0 * k,
                         .T = 1300 - 15.0 * k,
                         .measured = k < 2 ? 0.5 : 1,
                         .probe = k < 1 ? std::nullopt : std::optional<double>{523.907 - 1.7 * k}});
  }
  b.production = {{.kind = TestKind::PRP,
                   .lo = 67'000'000,
                   .hi = 68'364'640,
                   .weight = 0.55,
                   .fft = "512:7:512:202",
                   .entry = "e1",
                   .measured = true,
                   .us = 502.712,
                   .was = 523.907,
                   .moved = true},
                  {.kind = TestKind::PRP,
                   .lo = 68'412'189,
                   .hi = 80'000'000,
                   .weight = 0.45,
                   .fft = "256:15:512:101",
                   .entry = "e2",
                   .measured = true,
                   .us = 594.645,
                   .was = 561.0,
                   .moved = true,
                   .wasMeasured = false}};
  b.samples = {{.kind = TestKind::PRP,
                .exponent = 67'000'000,
                .probe = false,
                .untuned = Cost{.us = 523.907, .entry = "u1", .fft = "1K:7:256:212"},
                .now = Cost{.us = 502.712, .entry = "e1", .fft = "512:7:512:202"}},
               {.kind = TestKind::PRP,
                .exponent = 67'513'549,
                .probe = true,
                .untuned = Cost{.us = 523.907, .entry = "u1", .fft = "1K:7:256:212"},
                .now = Cost{.us = 502.712, .entry = "e1", .fft = "512:7:512:202"}},
               {.kind = TestKind::PRP,
                .exponent = 80'000'000,
                .probe = false,
                .untuned = std::nullopt,
                .now = Cost{.us = 594.645, .entry = "e2", .fft = "256:15:512:101"}}};
  b.next = {
    {.kind = ItemKind::Probe, .label = "512:7:512:202 prp short32 Memory 3 STORES=4", .value = 2.9, .seconds = 22},
    {.kind = ItemKind::Gate, .label = "256:15:512:101 prp short32 LOADS=1", .byRule = true, .seconds = 5}};
  b.recent = {
    {.item = timing(12, "512:7:512:202 prp short32 Memory 2 LOADS=30050", 509.799, 1125.1, 1124.006), .at = "16:49:47"},
    {.item = timing(11, "512:7:512:202 prp short32 Memory 2 LOADS=30049", 511.2, 1125.1, 1125.1), .at = "16:49:40"}};
  b.recent.front().item.best = "LOADS=30050,SHUFL_BYTES_H=16, 509.799 us/it";
  b.events = {"16:49:47 512:7:512:202 prp short32 is now best at LOADS=30050,SHUFL_BYTES_H=16, 509.799 us/it",
              "16:44:12 bootstrap complete; the defaults are MULTI_Q=1,SHUFL_BYTES_H=16"};
  b.faults = {{.what = Fault::What::Lost,
               .fft = "1K:7:256:212",
               .kind = TestKind::PRP,
               .exponent = 67'513'549,
               .options = "L1CUDA=1,LOADS=4000",
               .ts = 0,
               .gen = 0}};
  b.unicode = unicode;
  return b;
}

std::vector<std::string> text(const std::vector<Line>& lines) {
  std::vector<std::string> out;
  for (const Line& line : lines) { out.push_back(plain(line)); }
  return out;
}

}  // namespace

TEST(a_glyph_is_one_column_however_many_bytes_it_takes) {
  CHECK_EQ(columns(std::string_view{"▁▂…"}), size_t{3});
  CHECK_EQ(columns(std::string_view{"T 1124.006"}), size_t{10});
  Line const line{{"ab", Style::Bold}, {"│c", Style::Dim}};
  CHECK_EQ(columns(line), size_t{4});
  CHECK_EQ(plain(line), std::string{"ab│c"});
}

TEST(the_logs_lines_are_listed_as_events_but_for_what_the_panels_already_show) {
  CHECK_EQ(event("20260927 16:49:47  tune: 12. probe 512:7:512:202 prp at 67513549: 509.799 us/it\n"),
           std::string{"-"});
  CHECK_EQ(event("20260927 16:49:47  tune: progress: 5:00 in, 110 items\n"), std::string{"-"});
  CHECK_EQ(event("20260927 16:49:47  tune: still measuring 13. probe ..., 1:00 so far\n"), std::string{"-"});
  CHECK_EQ(event("20260927 16:49:47 \n"), std::string{"-"});
  CHECK_EQ(event(""), std::string{"-"});

  CHECK_EQ(event("20260927 16:49:47  tune: 512:7:512:202 prp short32 is now best at LOADS=1, 509.799 us/it\n"),
           std::string{"16:49:47 512:7:512:202 prp short32 is now best at LOADS=1, 509.799 us/it"});
  CHECK_EQ(event("20260927 16:49:47  measure: anchor 1K:10:256:212@89843291: 622.862 us/it\n"),
           std::string{"16:49:47 measure: anchor 1K:10:256:212@89843291: 622.862 us/it"});
  // "tune: 3.5x ..." is not an item's line: an item is numbered in whole numbers.
  CHECK_EQ(event("20260927 16:49:47  tune: 3.5x. odd\n"), std::string{"16:49:47 3.5x. odd"});
  // The lines after the first of one log() call carry no time of their own.
  CHECK_EQ(event("    -fft 1K:7:256:212 -use LOADS=1\n"), std::string{"             -fft 1K:7:256:212 -use LOADS=1"});
}

TEST(production_is_grouped_into_stretches_of_one_entry_and_set_against_a_reference) {
  std::vector<ObjectivePoint> const start{
    point(100, 0.2, "a", "512:7:512:202", 520), point(200, 0.2, "a", "512:7:512:202", 520),
    point(300, 0.2, "", "256:15:512:101", 600), point(400, 0.2, "", "256:15:512:101", 600),
    point(500, 0.2, "c", "1K:8:1K:212", 700)};
  std::vector<ObjectivePoint> now{point(100, 0.2, "a", "512:7:512:202", 520),
                                  point(200, 0.2, "a", "512:7:512:202", 520),
                                  point(300, 0.2, "b", "256:15:512:101", 590),
                                  point(400, 0.2, "", "256:15:512:101", 600), point(500, 0.2, "c", "1K:8:1K:212", 690)};
  // A point the grid holds that no FFT can run carries no cost, and is no part of any stretch.
  now.push_back({.kind = TestKind::PRP, .exponent = 600, .weight = 0, .cost = std::nullopt});

  std::vector<Stretch> const s = stretchesOf(now, start);
  CHECK_EQ(s.size(), size_t{4});
  if (s.size() != 4) { return; }
  // With no entries to say where each stops serving, the stretches meet just before the next one's first point.
  CHECK(s[0].lo == 100 && s[0].hi == 299 && s[0].measured && !s[0].moved && s[0].wasMeasured);
  CHECK(std::abs(s[0].weight - 0.4) < 1e-12);
  // Measured where the prior stood at the start: a first measurement, not a change.
  CHECK(s[1].lo == 300 && s[1].hi == 399 && s[1].measured && s[1].moved && !s[1].wasMeasured);
  CHECK(s[2].lo == 400 && s[2].hi == 499 && !s[2].measured && !s[2].moved);
  CHECK(s[3].moved && s[3].wasMeasured && s[3].was == 700 && s[3].us == 690);

  // A point that weighs nothing (a probe with probeWeight=0) costs what the entry costs, not what dividing by its
  // weight would make it.
  std::vector<ObjectivePoint> const probe{point(100, 0.5, "a", "512:7:512:202", 520),
                                          point(150, 0, "a", "512:7:512:202", 520),
                                          point(200, 0.5, "a", "512:7:512:202", 520)};
  std::vector<Stretch> const one = stretchesOf(probe, probe);
  CHECK(one.size() == 1 && std::abs(one[0].us - 520) < 1e-9 && std::abs(one[0].was - 520) < 1e-9);
  std::vector<ObjectivePoint> const weightless{point(150, 0, "a", "512:7:512:202", 520)};
  std::vector<Stretch> const alone = stretchesOf(weightless, {});
  CHECK(alone.size() == 1 && std::abs(alone[0].us - 520) < 1e-9 && alone[0].moved);
}

TEST(the_phase_is_what_runs_ahead_by_rule) {
  auto items = [](ItemKind kind, bool cover = false, bool bootstrap = false) {
    Item i;
    i.kind = kind;
    i.cover = cover;
    i.bootstrap = bootstrap;
    return std::vector<Item>{i};
  };
  CHECK_EQ(phaseOf(items(ItemKind::Probe, false, true), true),
           std::string{"bootstrap: searching each FFT type's fastest FFT"});
  CHECK_EQ(phaseOf(items(ItemKind::Gate), true), std::string{"accuracy gate: reading what the table owes"});
  CHECK_EQ(phaseOf(items(ItemKind::Baseline, true), true), std::string{"covering the workload"});
  CHECK_EQ(phaseOf(items(ItemKind::Probe), true), std::string{"searching"});
  CHECK_EQ(phaseOf({}, false), std::string{"nothing left worth running"});
}

TEST(a_frame_never_outgrows_the_terminal) {
  std::vector<std::pair<size_t, size_t>> const sizes{{60, 224}, {50, 160}, {49, 159}, {40, 120}, {24, 80},
                                                     {15, 60},  {8, 30},   {3, 20},   {1, 1},    {0, 0}};
  for (bool const unicode : {true, false}) {
    for (bool const faults : {true, false}) {
      Board b = board(unicode);
      if (!faults) { b.faults.clear(); }
      for (auto const& [rows, cols] : sizes) {
        std::vector<Line> const lines = frame(b, rows, cols);
        CHECK(lines.size() <= rows);
        for (const Line& line : lines) {
          if (columns(line) + 1 > std::max<size_t>(cols, 1)) {
            printf("%zux%zu: '%s'\n", rows, cols, plain(line).c_str());
            CHECK(!"a line reaches the last column");
          }
        }
      }
    }
  }

  // Before anything has been ranked, and with nothing in any panel.
  Board empty;
  for (auto const& [rows, cols] : sizes) {
    std::vector<Line> const lines = frame(empty, rows, cols);
    CHECK(lines.size() <= rows);
    for (const Line& line : lines) { CHECK(columns(line) < std::max<size_t>(cols, 1)); }
  }
}

TEST(faults_sit_under_the_header_whenever_there_are_any) {
  for (auto const& [rows, cols] : std::vector<std::pair<size_t, size_t>>{{60, 224}, {24, 80}}) {
    std::vector<std::string> const lines = text(frame(board(true), rows, cols));
    CHECK(lines.size() > 4 && lines[3].starts_with("FAULTS  1 configuration took the device down"));
    CHECK(lines.size() > 4 && lines[4].starts_with("  lost   -fft 1K:7:256:212 -use L1CUDA=1,LOADS=4000"));

    Board clean = board(true);
    clean.faults.clear();
    for (const std::string& line : text(frame(clean, rows, cols))) { CHECK(!line.starts_with("FAULTS")); }
  }
}

TEST(the_wide_frame_holds_every_panel_in_two_columns) {
  std::vector<std::string> const lines = text(frame(board(true), 60, 224));
  auto find = [&](std::string_view what) {
    return std::ranges::any_of(lines, [&](const std::string& l) { return l.find(what) != std::string::npos; });
  };
  CHECK(lines.size() > 1 && lines[0].starts_with("PRPLL -tune  Tesla P100-PCIE-16GB, OpenCL, device 1, env 2"));
  CHECK(lines.size() > 1 && lines[0].ends_with("generation 1"));
  CHECK(find("NOW  13. probe 512:7:512:202 prp short32 Memory 2 LOADS=42552,STORES=3 at 67513549 (call 2)  (0:17)"));
  CHECK(find("BENEFIT  at 67513549 production runs 512:7:512:202 at 502.712 us/it, 4.0% less per iteration"));
  CHECK(find("T, the time per iteration over the workload: now 1120.000 us/it, from 1270.000"));
  CHECK(find("what production runs at the probe 67513549: now 503.507 us/it, from 522.207"));
  CHECK(find("^ this run"));
  CHECK(find("SAMPLES  across the workload"));
  CHECK(find("*   67513549  1K:7:256:212   523.907  →  512:7:512:202   502.712  -4.05%"));
  CHECK(find("    80000000  not read                →  256:15:512:101   594.645"));
  CHECK(std::ranges::any_of(
    lines, [](const std::string& l) { return l.starts_with("PRODUCTION") && l.find("│ NEXT") != std::string::npos; }));
  CHECK(find("67000000-68364640   55.0%  512:7:512:202     502.712 us/it  -4.05%"));
  CHECK(find("68412189-80000000   45.0%  256:15:512:101    594.645 us/it  not read at the defaults"));
  CHECK(find("now best at LOADS=30050,SHUFL_BYTES_H=16, 509.799 us/it"));
  CHECK(find("by rule, ~5 s"));
  CHECK(find("16:44:12 bootstrap complete; the defaults are MULTI_Q=1,SHUFL_BYTES_H=16"));
}

TEST(a_chart_says_where_it_started_however_long_the_run_grows) {
  // T measured whole from the first second, falling a little with every item; the first column of any chart wide
  // enough to draw holds many of these points.
  Board b = board(true);
  b.history.clear();
  for (int k = 0; k <= 600; ++k) {
    b.history.push_back({.active = double(k), .T = 1300 - 0.1 * k, .measured = 1, .probe = 520 - 0.01 * k});
  }
  for (double const total : {600.0, 6000.0, 60000.0}) {
    b.before = total - b.elapsed;
    b.history.push_back({.active = total, .T = 1200, .measured = 1, .probe = 500});
    std::vector<std::string> const lines = text(frame(b, 60, 224));
    auto find = [&](std::string_view what) {
      return std::ranges::any_of(lines, [&](const std::string& l) { return l.find(what) != std::string::npos; });
    };
    CHECK(find("T, the time per iteration over the workload: now 1200.000 us/it, from 1300.000"));
    CHECK(find("what production runs at the probe 67513549: now 500.000 us/it, from 520.000"));
  }
}

TEST(a_timing_says_what_its_row_is_ranked_at) {
  Board b = board(true);
  Finished f = timing(13, "512:7:512:202 prp short32 Memory 2 LOADS=30051", 509.1, 1124.006, 1124.006);
  f.ranked = 512.345;
  f.calls = 2;
  b.recent.push_front({.item = f, .at = "16:50:01"});
  std::vector<std::string> const lines = text(frame(b, 60, 224));
  CHECK(std::ranges::any_of(
    lines, [](const std::string& l) { return l.find("509.100 us/it, ranked 512.345 (2)") != std::string::npos; }));
}

TEST(the_samples_are_spread_over_the_workload_with_the_probe_among_them) {
  std::vector<ObjectivePoint> points;
  for (u64 k = 0; k < 20; ++k) { points.push_back(point(60'000'000 + k * 1'000'000, 0.05, "e", "512:7:512:202", 500)); }
  // The probe, weighing nothing, is still followed; a point no FFT can run, or of another kind, is not.
  points.push_back(point(67'513'549, 0, "e", "512:7:512:202", 500));
  points.push_back({.kind = TestKind::PRP, .exponent = 99'000'000, .weight = 0.05, .cost = std::nullopt});
  ObjectivePoint ll = point(70'500'000, 0.05, "e", "512:7:512:202", 500);
  ll.kind = TestKind::LL;
  points.push_back(ll);

  std::vector<u64> const seven = sampleExponents(points, TestKind::PRP, 67'513'549, 7);
  CHECK_EQ(seven.size(), size_t(7));
  CHECK(std::ranges::is_sorted(seven));
  CHECK_EQ(seven.front(), u64(60'000'000));
  CHECK_EQ(seven.back(), u64(79'000'000));
  CHECK(std::ranges::find(seven, u64(67'513'549)) != seven.end());
  CHECK(std::ranges::find(seven, u64(70'500'000)) == seven.end());

  // Fewer points than samples: every one.
  std::vector<ObjectivePoint> const few(points.begin(), points.begin() + 3);
  CHECK(sampleExponents(few, TestKind::PRP, 1, 7) == (std::vector<u64>{60'000'000, 61'000'000, 62'000'000}));
  CHECK(sampleExponents(points, TestKind::PRP, 67'513'549, 0).empty());
}

TEST(a_stretch_production_runs_as_it_did_untuned_says_so) {
  // Production's reference is the fastest at the built-in defaults: where it runs that very entry the stretch has not
  // moved, and where the reference is another FFT or another set, the change is against it.
  std::vector<ObjectivePoint> const now{point(67'000'000, 0.5, "d1", "1K:7:256:212", 520),
                                        point(68'000'000, 0.5, "t1", "512:7:512:201", 480)};
  std::vector<ObjectivePoint> const untuned{point(67'000'000, 0.5, "d1", "1K:7:256:212", 520),
                                            point(68'000'000, 0.5, "d1", "1K:7:256:212", 520)};
  std::vector<Stretch> const s = stretchesOf(now, untuned);
  CHECK_EQ(s.size(), size_t(2));
  CHECK(!s[0].moved && s[0].wasMeasured);
  CHECK(s[1].moved && s[1].wasMeasured && std::abs(s[1].was - 520) < 1e-9);
}

TEST(stretches_meet_where_one_entry_stops_serving_and_cover_the_whole_workload) {
  // The 5070 Ti's LL table in miniature: 2:512:4:512:202 serves up to 67004008, between the grid's first two points,
  // and 1:256:4:1K:202 from there to 70500000; then a stretch the prior prices.
  std::vector<ObjectivePoint> const now{
    point(67'000'000, 0.3, "a", "2:512:4:512:202", 172.8), point(67'046'600, 0.3, "b", "1:256:4:1K:202", 223.4),
    point(70'000'000, 0.2, "b", "1:256:4:1K:202", 223.4), point(71'000'000, 0.2, "", "256:16:512", 300)};
  std::vector<SelectionEntry> entries(2);
  entries[0].id = "a";
  entries[0].emin = 39'845'888;
  entries[0].reach = 67'004'008;
  entries[1].id = "b";
  entries[1].emin = 39'845'888;
  entries[1].reach = 70'500'000;

  std::vector<Stretch> const s = stretchesOf(now, now, entries);
  CHECK_EQ(s.size(), size_t(3));
  CHECK(s[0].lo == 67'000'000 && s[0].hi == 67'004'008);
  CHECK(s[1].lo == 67'004'009 && s[1].hi == 70'500'000);
  CHECK(s[2].lo == 70'500'001 && s[2].hi == 71'000'000);

  // Without the entries, the stretches still meet, just before each one's first point.
  std::vector<Stretch> const plain = stretchesOf(now, now);
  CHECK(plain[0].hi + 1 == plain[1].lo && plain[1].hi + 1 == plain[2].lo);
  CHECK_EQ(plain[0].hi, u64(67'046'599));
}

TEST(a_narrow_frame_is_one_column_drawn_in_ascii_where_the_terminal_wants_it) {
  std::vector<std::string> const lines = text(frame(board(false), 34, 100));
  std::string all;
  for (const std::string& l : lines) { all += l + "|\n"; }
  std::string const golden =
    "PRPLL -tune  Tesla P100-PCIE-16GB, OpenCL, device 1, env 2           this run 5:00  |  generation 1|\n"
    "searching  |  T 1124.006 us/it, 100.0% measured  |  best item worth 2.9000 us/it, 2.6x the stop ...|\n"
    "|\n"
    "FAULTS  1 configuration took the device down: kernel or driver bugs, held out of the search; -tu...|\n"
    "  lost   -fft 1K:7:256:212 -use L1CUDA=1,LOADS=4000   (prp at 67513549)|\n"
    "|\n"
    "NOW  13. probe 512:7:512:202 prp short32 Memory 2 LOADS=42552,STORES=3 at 67513549 (call 2)  (0:17)|\n"
    "|\n"
    "BENEFIT  at 67513549 production runs 512:7:512:202 at 502.712 us/it, 4.0% less per iteration than|\n"
    "         the best measured there at the built-in defaults (1K:7:256:212, 523.907 us/it)|\n"
    "  T, the time per iteration over the workload: now 1120.000 us/it, from 1270.000|\n"
    "                 #######*************+++++++=============~~~~~~~-------............._  1120-1270|\n"
    "    0:00                                                             this run ^      |\n"
    "|\n"
    "  what production runs at the probe 67513549: now 503.507 us/it, from 522.207|\n"
    "          #######*************+++++++==============~~~~~~~~~~~~~-------............._  504-522|\n"
    "    0:00                                                             this run ^      |\n"
    "|\n"
    "PRODUCTION  what selection.txt runs over the workload; change against the fastest untuned there|\n"
    "  67000000-68364640   55.0%  512:7:512:202     502.712 us/it  -4.05%|\n"
    "  68412189-80000000   45.0%  256:15:512:101    594.645 us/it  not read at the defaults|\n"
    "|\n"
    "SAMPLES  across the workload: the fastest at the built-in defaults, and what runs there now|\n"
    "    67000000  1K:7:256:212   523.907  ->  512:7:512:202   502.712  -4.05%|\n"
    "*   67513549  1K:7:256:212   523.907  ->  512:7:512:202   502.712  -4.05%|\n"
    "|\n"
    "RECENT  items this run, newest first; ranked as selection.txt ranks the row (calls)|\n"
    "    12  16:49:47  probe 512:7:5...ory 2 LOADS=30050 at 67513549    509.799 us/it    6.2 s  T -1.094|\n"
    "|\n"
    "NEXT  what the queue ranks highest, best rate first|\n"
    "   1. probe 512:7:512:202 prp short32 Memory 3 STORES=4                         2.9000 us/it, ~22 s|\n"
    "|\n"
    "EVENTS  the rest of the log, newest first|\n"
    "  16:49:47 512:7:512:202 prp short32 is now best at LOADS=30050,SHUFL_BYTES_H=16, 509.799 us/it|\n";
  if (all != golden) { printf("%s", all.c_str()); }
  CHECK_EQ(all, golden);
}

TEST(a_frame_is_painted_from_the_top_without_ever_scrolling) {
  std::vector<Line> const lines{{{"a", Style::Plain}}, {{"b", Style::Bad}, {"c", Style::Plain}}, {}};
  CHECK_EQ(paint(lines, false), std::string{"\x1b[Ha\x1b[K\r\nbc\x1b[K\r\n\x1b[K\x1b[J"});
  CHECK_EQ(paint(lines, true), std::string{"\x1b[Ha\x1b[K\r\n\x1b[1;31mb\x1b[0mc\x1b[K\r\n\x1b[K\x1b[J"});
  CHECK_EQ(paint({}, false), std::string{"\x1b[H\x1b[J"});
}
