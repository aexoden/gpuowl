// Copyright (C) Jason Lynch

// What production sustains, for the tuner to set against what it measured.  A tuner's calls read 4.5-9% faster than
// production ran the identical configuration on three of four cards, so the published costs are checked against the one
// reading that is production: as a test runs, each stretch of it appends a line to tune-feedback.txt beside tune.txt,
//
//   181.6 2:512:4:512:202 ll short64 67249139 61 61000000 8f3a2c01d4e5b677 1761674436 HOIST_W=1,LOADS=5502,...
//
// (us/it, FFT, kind, regime, exponent, steps, iterations, build, time, the -use settings the Gpu was built with, as
// configText() writes them), and
// -tune status and tune-report.json say, for each published line production has run, what it sustained there.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "Readings.h"
#include "TestKind.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

inline constexpr const char* FEEDBACK_FILE = "tune-feedback.txt";

// A stretch is written once it has this many log steps besides the first, which carries the load and the card's warm-up.
inline constexpr u32 FEEDBACK_MIN_STEPS = 3;

struct FeedbackRow {
  double usPerIt = 0;  // the median of the stretch's log steps, the first left out
  std::string fft{};
  TestKind kind = TestKind::PRP;
  std::string regime{};
  u64 exponent = 0;
  u32 steps = 0;  // the steps the median is of
  u64 iters = 0;  // the iterations they hold
  u64 build = 0;  // buildFingerprint() of the kernels production ran
  u64 ts = 0;     // when the stretch ended, seconds since the epoch
  std::map<std::string, std::string> settings{};  // every -use setting the Gpu was built with
};

[[nodiscard]] std::string feedbackLine(const FeedbackRow& row);
[[nodiscard]] std::optional<FeedbackRow> parseFeedbackLine(std::string_view line);

// Every line of `path` that parses; none where there is no file.
[[nodiscard]] std::vector<FeedbackRow> readFeedback(const std::filesystem::path& path);

// One test's stretch, as production runs it: fed each log step, it appends its line to FEEDBACK_FILE in the working
// directory when it goes out of scope, or when restart() begins another stretch (a reload from a savefile).
class ProductionRate {
public:
  ProductionRate(u64 E, const FFTConfig& fft, TestKind kind, const std::map<std::string, std::string>& settings,
                 u32 logStep);
  ~ProductionRate();
  ProductionRate(const ProductionRate&) = delete;
  ProductionRate& operator=(const ProductionRate&) = delete;

  // A log step that ended at iteration `k`, at `secsPerIt` since the one before.
  void step(u64 k, double secsPerIt);

  // The test was (re)loaded at iteration `k`: what was gathered is written, and the next step is a warm-up again.
  void restart(u64 k);

  // The line the steps so far would write; nothing below FEEDBACK_MIN_STEPS.
  [[nodiscard]] std::optional<FeedbackRow> row() const;

private:
  void flush();

  FeedbackRow base_;
  u32 logStep_;
  u64 lastK_ = 0;
  bool warm_ = false;
  std::vector<double> rates_;  // us/it of the steps counted
  u64 iters_ = 0;
};

// What production sustained on each configuration, over the lines of `rows` from kernels of `build`: the iteration-
// weighted mean of its stretches' rates.
struct Sustained {
  double usPerIt = 0;
  u32 stretches = 0;
  u64 iters = 0;
};

[[nodiscard]] std::map<ConfigKey, Sustained> sustainedOf(const std::vector<FeedbackRow>& rows, u64 build,
                                                         const Env& device);

}  // namespace tune
