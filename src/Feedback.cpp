// Copyright (C) Jason Lynch

#include "Feedback.h"

#include "BuildId.h"
#include "Eligibility.h"
#include "File.h"
#include "log.h"
#include "OptionSpace.h"
#include "TuneDB.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>

namespace tune {

std::string feedbackLine(const FeedbackRow& row) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%.2f %s %s %s %" PRIu64 " %u %" PRIu64 " %016" PRIx64 " %" PRIu64 " ", row.usPerIt,
                row.fft.c_str(), toString(row.kind), row.regime.c_str(), row.exponent, row.steps, row.iters, row.build,
                row.ts);
  return buf + configText(row.settings);
}

std::optional<FeedbackRow> parseFeedbackLine(std::string_view line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.remove_suffix(1); }
  auto const fields = splitFields(line);
  if (!fields || fields->size() != 10) { return {}; }
  const std::vector<std::string>& f = *fields;
  auto const us = parseDouble(f[0]);
  auto const kind = parseTestKind(f[2]);
  auto const exponent = parseInt<u64>(f[4]);
  auto const steps = parseInt<u32>(f[5]);
  auto const iters = parseInt<u64>(f[6]);
  auto const build = parseInt<u64>(f[7], 16);
  auto const ts = parseInt<u64>(f[8]);
  auto const settings = parseConfigText(f[9]);
  if (!us || *us <= 0 || !kind || !exponent || !steps || !iters || !build || !ts || !settings) { return {}; }
  return FeedbackRow{.usPerIt = *us,
                     .fft = f[1],
                     .kind = *kind,
                     .regime = f[3],
                     .exponent = *exponent,
                     .steps = *steps,
                     .iters = *iters,
                     .build = *build,
                     .ts = *ts,
                     .settings = *settings};
}

std::vector<FeedbackRow> readFeedback(const std::filesystem::path& path) {
  std::vector<FeedbackRow> out;
  File fi = File::openRead(path);
  if (!fi) { return out; }
  fi.allowUnterminatedLastLine();
  for (const std::string& line : fi) {
    if (line.starts_with('#')) { continue; }
    if (auto row = parseFeedbackLine(line)) { out.push_back(std::move(*row)); }
  }
  return out;
}

ProductionRate::ProductionRate(u64 E, const FFTConfig& fft, TestKind kind,
                               const std::map<std::string, std::string>& settings, u32 logStep)
  : base_{.fft = fft.spec(),
          .kind = kind,
          .regime = regimeOf(fft, E).label(),
          .exponent = E,
          .build = buildFingerprint(),
          .settings = settings},
    logStep_{logStep} {}

ProductionRate::~ProductionRate() {
  try {
    flush();
  } catch (...) {}
}

void ProductionRate::step(u64 k, double secsPerIt) {
  u64 const iters = k > lastK_ ? k - lastK_ : 0;
  lastK_ = k;
  // The first step after a load carries it and the card's warm-up; a short one (a PRP test's first check, or its last
  // step) is mostly the fixed cost of the log and the check around it.
  if (!std::exchange(warm_, true) || iters * 2 < logStep_ || !(secsPerIt > 0)) { return; }
  rates_.push_back(secsPerIt * 1e6);
  iters_ += iters;
}

void ProductionRate::restart(u64 k) {
  flush();
  lastK_ = k;
  warm_ = false;
}

std::optional<FeedbackRow> ProductionRate::row() const {
  if (rates_.size() < FEEDBACK_MIN_STEPS) { return {}; }
  std::vector<double> sorted = rates_;
  std::ranges::sort(sorted);
  size_t const n = sorted.size();
  FeedbackRow out = base_;
  out.usPerIt = n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  out.steps = u32(n);
  out.iters = iters_;
  out.ts = u64(std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count());
  return out;
}

void ProductionRate::flush() {
  std::optional<FeedbackRow> const out = row();
  rates_.clear();
  iters_ = 0;
  if (!out) { return; }
  try {
    File::append(FEEDBACK_FILE, feedbackLine(*out) + '\n');
  } catch (...) {
    log("Can't append to %s\n", FEEDBACK_FILE);
  }
}

std::map<ConfigKey, Sustained> sustainedOf(const std::vector<FeedbackRow>& rows, u64 build, const Env& device) {
  std::map<ConfigKey, Sustained> out;
  std::map<ConfigKey, double> weighted;
  for (const FeedbackRow& row : rows) {
    if (row.build != build) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    if (!fft) { continue; }
    Subject const subject{.fft = *fft, .kind = row.kind, .regime = regimeOf(*fft, row.exponent)};
    ConfigKey const key = Readings::keyOf(subject, canonicalConfig(device, *fft, row.settings));
    Sustained& s = out[key];
    ++s.stretches;
    s.iters += row.iters;
    weighted[key] += row.usPerIt * double(row.iters);
  }
  for (auto& [key, s] : out) { s.usPerIt = weighted.at(key) / double(s.iters); }
  return out;
}

}  // namespace tune
