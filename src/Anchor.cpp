// Copyright (C) Jason Lynch

#include "Anchor.h"

#include "Eligibility.h"
#include "FFTVariants.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <string>

namespace tune {

const char* toString(DriftLevel level) {
  switch (level) {
  case DriftLevel::Steady: return "steady";
  case DriftLevel::Warn: return "drifting";
  case DriftLevel::Alarm: return "alarm";
  }
  return "steady";
}

DriftLevel driftLevelOf(double ratio) {
  if (!(ratio > 0)) { return DriftLevel::Steady; }
  double const moved = std::abs(ratio - 1);
  if (moved >= DRIFT_ALARM) { return DriftLevel::Alarm; }
  return moved >= DRIFT_WARN ? DriftLevel::Warn : DriftLevel::Steady;
}

std::string AnchorSpec::text() const { return fft + '@' + to_string(exponent); }

std::optional<AnchorSpec> parseAnchorSpec(std::string_view text) {
  size_t const at = text.rfind('@');
  if (at == std::string_view::npos || at == 0 || at + 1 == text.size()) { return {}; }

  std::string_view const digits = text.substr(at + 1);
  if (digits.find_first_not_of("0123456789") != std::string_view::npos) { return {}; }

  AnchorSpec out{.fft = std::string{text.substr(0, at)},
                 .exponent = strtoull(std::string{digits}.c_str(), nullptr, 10)};
  if (!out.exponent) { return {}; }

  // A spec that does not name a configuration names nothing; the caller has to build it.
  try {
    FFTConfig const fft{out.fft};
    out.fft = fft.spec();
  } catch (...) { return {}; }

  return out;
}

std::vector<AnchorSpec> anchorCandidates(u64 exponent) {
  if (!exponent) { return {}; }

  // allShapes() is in ascending size, so the first eligible shape of a type is its smallest.
  std::map<enum FFT_TYPES, FFTConfig> smallest;
  for (const FFTShape& shape : FFTShape::allShapes()) {
    if (smallest.contains(shape.fft_type)) { continue; }

    // Both ends: isEligible() answers only the bits-per-word floor the Gpu constructor enforces, and an anchor above
    // the top of its own range is not a configuration anything would run.
    FFTConfig const fft{shape, defaultVariant(shape), CARRY_AUTO};
    if (!interval(fft, exponent).empty()) { smallest.emplace(shape.fft_type, fft); }
  }

  std::vector<AnchorSpec> out;
  for (const auto& [type, fft] : smallest) { out.push_back({.fft = fft.spec(), .exponent = exponent}); }
  return out;
}

std::optional<AnchorSpec> raceWinner(const std::vector<AnchorReading>& readings) {
  const AnchorReading* best = nullptr;
  for (const AnchorReading& r : readings) {
    if (r.us > 0 && (!best || r.us < best->us)) { best = &r; }
  }
  if (!best) { return {}; }
  return best->anchor;
}

DriftLevel AnchorState::observe(double mean) {
  if (!(mean > 0)) { return driftLevelOf(ratio); }

  latest = mean;
  ++readings;
  if (!(baseline > 0)) { baseline = mean; }
  ratio = mean / baseline;
  return driftLevelOf(ratio);
}

}  // namespace tune
