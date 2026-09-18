// Copyright (C) Jason Lynch

// Which -use options a Gpu runs with: the '!' config-line selectors, the precedence between every source of options,
// and the takeover that sets the config files aside for a tuning run.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "OptionSpace.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class Args;

namespace tune {

// PRP and LL run different carry kernels.
enum class TestKind : std::uint8_t { PRP, LL };

[[nodiscard]] const char* toString(TestKind kind);
[[nodiscard]] std::optional<TestKind> parseTestKind(std::string_view text);

// The left-hand side of a '!' config line: an optional test kind followed by an FFT spec truncated after any part.
//   ll                  every LL test
//   0                   every FFT of type 0 (FP64)
//   512:15:512          one FP64 shape, any variant and carry
//   1:512:8:512:202     one shape of type 1 and one variant
//   256:2:256:101:0     one shape and variant whose spec pins the carry (0 = 32-bit, 1 = 64-bit); never an FFT with an
//                       automatic carry, whatever width it ends up using at its exponent
//   prp:1K:8:256        the same forms, restricted to one test kind
struct FFTSelector {
  std::optional<TestKind> kind;
  std::optional<FFT_TYPES> type;
  u32 width = 0;  // 0: any shape
  u32 middle = 0;
  u32 height = 0;
  std::optional<u32> variant;
  std::optional<CARRY_KIND> carry;

  // Throws on anything that is not a selector, after logging why.
  [[nodiscard]] static FFTSelector parse(std::string_view text);

  [[nodiscard]] bool matches(const FFTConfig& fft, TestKind testKind) const;

  // Lines are applied in increasing rank, then in the order read, so the highest rank wins. A kind-qualified selector
  // outranks every unqualified one; within that, each further FFT part narrows the match and raises the rank.
  [[nodiscard]] u32 rank() const;

  [[nodiscard]] std::string spec() const;
};

struct UseLine {
  FFTSelector selector;
  std::vector<std::pair<std::string, std::string>> uses;
};

// Parses "! <selector> <KEY=VALUE,...> [# comment]". Throws on anything else, after logging the line and why.
[[nodiscard]] UseLine parseUseLine(std::string_view line);

// What the selection file contributes to the precedence below. Empty until a selection file is read.
struct SelectionLayers {
  std::vector<std::pair<std::string, std::string>> global;
  std::vector<UseLine> family;
  std::vector<std::pair<std::string, std::string>> entry;
};

// The -use options in effect for one FFT under one test kind. Weakest first, each overriding the ones before:
//   built-in defaults (absent from the result)
//   the selection file's global line, then its family lines, then the selected entry's option set
//   -use lines in the config files
//   '!' lines in the config files, by rank, then the last read
//   -use on the command line
[[nodiscard]] UseConfig resolveConfig(const Args& args, const FFTConfig& fft, TestKind kind,
                                      const SelectionLayers& selection = {});

// Replaces args' flags with what resolveConfig() gives, with `extraConf` on top, and drops the '!' lines they came
// from.
void resolveInto(Args& args, const FFTConfig& fft, TestKind kind,
                 const std::vector<std::pair<std::string, std::string>>& extraConf);

// The keys a tuning run set aside, each list sorted.
struct Takeover {
  std::vector<std::string> configKeys;
  std::vector<std::string> commandLineKeys;
};

// Sets aside every -use and '!' setting from the config files, and every command-line -use key other than the ones the
// tuner never varies (NO_ASM, DEBUG, ...), so that a measurement depends on the built-in defaults and the tuner's own
// choices alone.
[[nodiscard]] Takeover takeOverConfig(Args& args);

[[nodiscard]] std::string describe(const Takeover& takeover);

}  // namespace tune
