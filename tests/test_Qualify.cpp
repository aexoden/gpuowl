// Copyright (C) Jason Lynch

// -measure's arithmetic and verdicts, and what it reads off a card, without one.

#include "test.h"

#include "Lanes.h"
#include "Qualify.h"
#include "Telemetry.h"

#include <cmath>
#include <string>
#include <vector>

using namespace tune;

namespace {

std::string refusal(const char* text) {
  try {
    (void)parseMeasureArgs(text);
  } catch (const std::string& mes) { return mes; }
  return {};
}

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

bool has(const std::vector<std::string>& lines, const char* text) {
  for (const std::string& line : lines) {
    if (line.find(text) != std::string::npos) { return true; }
  }
  return false;
}

}  // namespace

TEST(measure_reads_an_fft_and_its_settings) {
  MeasureArgs const plain = parseMeasureArgs("512:7:512:000");
  CHECK_EQ(plain.fft, std::string{"512:7:512:000"});
  CHECK(plain.kind == TestKind::PRP);
  CHECK_EQ(plain.exponent, u64{0});
  CHECK_EQ(plain.bursts, 6u);
  CHECK(near(plain.idle, 5) && near(plain.soak, 900) && near(plain.settle, 0.001));

  MeasureArgs const set = parseMeasureArgs(
    "2:512:4:512:202,kind=ll,exp=67513549,bursts=3,idle=10s,soak=20m,settle=0.2%,minutes=8,gap=2%,spread=1%");
  CHECK(set.kind == TestKind::LL);
  CHECK_EQ(set.exponent, u64{67'513'549});
  CHECK_EQ(set.bursts, 3u);
  CHECK_EQ(set.minutes, 8u);
  CHECK(near(set.idle, 10) && near(set.soak, 1200));
  CHECK(near(set.settle, 0.002) && near(set.gap, 0.02) && near(set.spread, 0.01));
  CHECK(near(parseMeasureArgs("512:7:512:000,idle=0").idle, 0));

  for (const char* bad : {"", "nonsense", "512:7:512:000,kind=cert", "512:7:512:000,bursts=0", "512:7:512:000,gap=1",
                          "512:7:512:000,soak=15", "512:7:512:000,exp=1.5", "512:7:512:000,fast=1", "512:7:512:000,"}) {
    CHECK(!refusal(bad).empty());
  }
}

TEST(a_span_of_time_may_be_in_seconds) {
  CHECK(parseDuration("30s") && near(*parseDuration("30s"), 30));
  CHECK(parseDuration("1m30s") && near(*parseDuration("1m30s"), 90));
  CHECK(!parseDuration("30"));
}

TEST(minutes_are_whole_and_weighted_by_their_blocks) {
  // Chunks ending every 20 s from t=100: minute 1 is 100-160, minute 2 is 160-220.
  std::vector<Chunk> const chunks{{.end = 120, .us = 100, .blocks = 1}, {.end = 140, .us = 103, .blocks = 2},
                                  {.end = 159, .us = 100, .blocks = 1}, {.end = 180, .us = 104, .blocks = 1},
                                  {.end = 200, .us = 104, .blocks = 1}};
  std::vector<double> const minutes = minuteMeans(chunks, 100);
  CHECK_EQ(minutes.size(), size_t{1});  // the second minute has not ended
  if (minutes.empty()) { return; }
  CHECK(near(minutes[0], (100 + 206 + 100) / 4.0));

  std::vector<Chunk> more = chunks;
  more.push_back({.end = 225, .us = 104, .blocks = 1});
  CHECK_EQ(minuteMeans(more, 100).size(), size_t{2});
  CHECK(minuteMeans({}, 0).empty());
}

TEST(a_soak_settles_at_the_first_minute_like_the_one_before) {
  std::vector<double> const warming{100, 101, 101.5, 101.55, 101.56};
  CHECK(!settledAt(warming, 0.00005).has_value());
  CHECK(settledAt(warming, 0.0001) == std::optional<size_t>{4});
  CHECK(settledAt(warming, 0.001) == std::optional<size_t>{3});
  CHECK(!settledAt(std::vector<double>{100}, 0.01).has_value());
}

TEST(the_verdict_names_optimistic_and_wandering_cards) {
  MeasureArgs const args = parseMeasureArgs("512:7:512:000");
  auto result = [](std::vector<double> bursts, std::vector<double> sustained, std::vector<double> hot) {
    MeasureResult r;
    r.bursts = std::move(bursts);
    r.sustained = std::move(sustained);
    r.hot = std::move(hot);
    return r;
  };
  MeasureResult stable = result({435, 435.2, 434.9}, {436, 436.3, 436.1}, {435.8});
  stable.settled = 2;
  MeasureVerdict const s = judge(stable, args, "nvidia", "03:00.0");
  CHECK(!s.optimistic && !s.wandering);
  CHECK(has(s.lines, "verdict: stable"));
  CHECK(!has(s.lines, "nvidia-smi"));

  // The 4070 SUPER's case: calls at 167 us/it, production at 182.
  MeasureResult hot = result({166.5, 167, 167.2}, {181.8, 182, 182.3}, {176});
  hot.settled = 4;
  hot.sustainedCard.smMedian = 2512;
  MeasureVerdict const h = judge(hot, args, "nvidia", "01:00.0");
  CHECK(h.optimistic && !h.wandering);
  CHECK(near(h.burstGap, 182.0 / 167 - 1));
  CHECK(has(h.lines, "verdict: optimistic"));
  CHECK(has(h.lines, "sudo nvidia-smi -i 0000:01:00.0 -lgc 2505,2505"));
  CHECK(has(h.lines, "tag=2505mhz"));

  MeasureResult const wander = result({600}, {600, 610, 604}, {600});
  MeasureVerdict const w = judge(wander, args, "amd", "");
  CHECK(!w.optimistic && w.wandering);
  CHECK(has(w.lines, "verdict: wandering"));
  CHECK(has(w.lines, "ran out of time"));
  CHECK(has(w.lines, "lock the card's clocks"));  // no clock read to name

  std::string const json = measureJson(args, "", 67'425'529, "none", hot, h);
  CHECK(json.find("\"optimistic\": true") != std::string::npos);
  CHECK(json.find("\"sustainedMinutesUsPerIt\": [181.8, 182, 182.3]") != std::string::npos);
}

TEST(amdgpu_marks_its_current_clock_level) {
  CHECK(currentDpmMHz("0: 500Mhz \n1: 1200Mhz \n2: 1801Mhz *\n") == std::optional<double>{1801});
  CHECK(currentDpmMHz("0: 96Mhz *\n1: 1000Mhz\n") == std::optional<double>{96});
  CHECK(!currentDpmMHz("0: 500Mhz\n1: 1200Mhz\n").has_value());
  CHECK(!currentDpmMHz("").has_value());
}

TEST(a_window_of_card_readings_is_summarized) {
  std::vector<CardSample> const samples{{.smMHz = 1300, .tempC = 70, .powerW = 200, .capW = 250, .held = 0x4},
                                        {.smMHz = 1328, .tempC = 74, .powerW = 210, .capW = 250, .held = 0x1},
                                        {.smMHz = 1328, .tempC = 76, .powerW = 220, .capW = 250, .held = 0x24}};
  CardSummary const s = summarizeCard(samples);
  CHECK_EQ(s.n, 3u);
  CHECK(near(s.smMin, 1300) && near(s.smMax, 1328) && near(s.smMedian, 1328));
  CHECK(std::isnan(s.memMedian));
  CHECK(near(s.tempMax, 76) && near(s.powerMedian, 210) && near(s.capW, 250));
  CHECK_EQ(s.held.size(), size_t{2});  // the idle bit is not a reason
  std::string const text = toString(s, true);
  CHECK(text.find("SM 1328 MHz (1300-1328)") != std::string::npos);
  CHECK(text.find("210/250 W") != std::string::npos);
  CHECK(text.find("held back by power cap 67%, thermal 33% of samples") != std::string::npos);
  CHECK(toString(s, false).find("held") == std::string::npos);
  CHECK(toString(summarizeCard({}), true).empty());
  CHECK_EQ(heldText(0x1 | 0x4 | 0x40), std::string{"power cap, hardware thermal"});
}

TEST(a_card_with_nothing_to_read_reads_nothing) {
  std::string why;
  CHECK(openTelemetry("", false, false, why) == nullptr);
  CHECK(!why.empty());
  Sampler const none{nullptr, 0.01};
  CHECK(none.between(0, 1e9).empty());
}
