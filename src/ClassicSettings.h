// Copyright (C) Jason Lynch

// The settings of classic mode (-tune mode=classic, Classic.h): upstream's own tuner options, read as this tuner reads
// its settings, and the thresholds upstream hard-codes.

#pragma once

#include "common.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// Which FFT types the option sweeps and the shape sweep cover: upstream's fp64 and ntt words, or neither, where an FP64
// FFT is timed against an M31+M61 NTT and a type within `typeMargin` of the other is tuned too.
enum class ClassicTypes : u8 { Auto, Fp64, Ntt, Both };

// How the width and height variants of an FP64 shape are pruned (upstream's skip_some_WH_variants 2, 1, 0): only the
// fastest variant of each width and height, or also any as accurate as it, or every variant.
enum class WhVariants : u8 { Fastest, Accurate, All };

// An optional group of shapes: not timed, timed, or timed alone (upstream's 0, 1 and 2).
enum class Group3 : u8 { Off, On, Only };

struct ClassicSettings {
  ClassicTypes types = ClassicTypes::Auto;

  // Whether the option sweeps run at all (upstream's noconfig turns them off).
  bool sweeps = true;

  // Whether only the in-place layout is timed (upstream's inplace word, the default on nVidia); empty: as upstream.
  std::optional<bool> inplaceOnly{};

  Group3 fp32 = Group3::On;
  Group3 fp6431 = Group3::Off;
  Group3 k1x256 = Group3::Off;
  Group3 m61 = Group3::Off;
  Group3 pfa = Group3::On;

  // Upstream's regs, variants and fine sub-modes; fine is a fraction, 0 for off.
  bool regsOnly = false;
  bool variantsOnly = false;
  double fine = 0;

  // tk=: 0 never tries the other TAIL_KERNELS choices, -1 tries them all (auto), N > 0 only N.
  int tk = 0;

  // How much better a kept option value has to read than the value it replaces, beyond its error bars; empty for
  // upstream's own threshold of each option (0 to 0.3%).
  std::optional<double> keepMargin{};

  // The auto type check: the slower kind of FFT is tuned too where within this fraction of the faster.
  double typeMargin = 0.20;

  WhVariants whVariants = WhVariants::Fastest;

  // The most calls a reading is taken to before an undecided comparison is settled for what it is.
  u32 maxCalls = 6;

  // A TAIL_KERNELS choice is no longer tried for an FFT type once it has lost this many times without a win, or half
  // as many times by more than `tkBad`.
  u32 tkLosses = 4;
  double tkBad = 0.05;

  [[nodiscard]] bool operator==(const ClassicSettings&) const = default;
};

// The settings as the words of a -tune command, each named whatever its value: what a session records.
[[nodiscard]] std::vector<std::string> classicWords(const ClassicSettings& settings);

// Reads one word of a classic run into `settings`: upstream's own (noconfig, fp64, ntt, inplace, nofp32, fp32=, fp6431,
// 1k256, m61, pfa, regs, variants, fine, tk=) or this mode's (types=, sweeps=, keepMargin=, typesMargin=, whVariants=,
// maxCalls=, tkLosses=, tkBad=).  False for a word it does not know; throws a message for a value it cannot read.
[[nodiscard]] bool parseClassicWord(std::string_view word, ClassicSettings& settings);

}  // namespace tune
