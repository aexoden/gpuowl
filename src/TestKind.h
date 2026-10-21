// Copyright (C) Jason Lynch

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace tune {

// The kind of test a configuration is timed and published for. LL and PRP run different carry kernels, and on some
// cards and FFT types the difference in cost is large, so each kind is tuned on its own readings.
enum class TestKind : std::uint8_t { PRP, LL };

[[nodiscard]] constexpr const char* toString(TestKind kind) { return kind == TestKind::LL ? "ll" : "prp"; }

[[nodiscard]] constexpr std::optional<TestKind> parseTestKind(std::string_view text) {
  if (text == "prp") { return TestKind::PRP; }
  if (text == "ll") { return TestKind::LL; }
  return {};
}

}  // namespace tune
