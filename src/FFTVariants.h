// Copyright (C) Jason Lynch

// Establishes one name per configuration, folding every spelling onto one canonical member.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <vector>

namespace tune {

struct Env;

struct FFTParts {
  bool fp64 = false;
  bool fp32 = false;
  bool gf31 = false;
  bool gf61 = false;
};

[[nodiscard]] FFTParts fftParts(enum FFT_TYPES type);

// The digits naming distinct kernels for this shape, ascending.
//
// Width and height: only the FP64 body of fft_common branches on them (digit 0 broadcasts, 1 is the generic path, 2 the
// extra-FMA path), and at size 1024 digits 1 and 2 share the one "WG == 128 && RADIX == 8" body, so 1 folds onto 2
// there. Digit 0 cannot compile above 1024. With no FP64 part every digit compiles the same kernels and folds onto 2.
//
// Middle: the digit only chooses the defaults of MM_CHAIN and MM2_CHAIN, so it names distinct kernels where there is a
// floating-point middle that reads them, i.e. not for a pure NTT, nor at MIDDLE == 2 where both are no-ops.
[[nodiscard]] std::vector<u32> widthDigits(const FFTShape& shape);
[[nodiscard]] std::vector<u32> middleDigits(const FFTShape& shape);
[[nodiscard]] std::vector<u32> heightDigits(const FFTShape& shape);

// Every canonical variant of this shape, ascending.
[[nodiscard]] std::vector<u32> allVariants(const FFTShape& shape);

// The member of allVariants() that compiles the same kernels as `variant`. Digits outside the range cl/base.cl accepts,
// and width digit 0 above 1024, compile nothing and are left unchanged.
[[nodiscard]] u32 canonicalVariant(const FFTShape& shape, u32 variant);

// What a spec that names no variant means: the highest digit of each list, which is the most accurate variant and the
// canonical form of LAST_VARIANT.
[[nodiscard]] u32 defaultVariant(const FFTShape& shape);

// Whether the carry setting reaches the kernels. CARRY64 only chooses the carry type of the single-arithmetic
// carryFused bodies; every hybrid body uses a fixed carry type.
[[nodiscard]] bool carryAffectsKernels(enum FFT_TYPES type);

// The canonical form of `carry` that names these kernels.
[[nodiscard]] enum CARRY_KIND canonicalCarry(const FFTShape& shape, u32 variant, enum CARRY_KIND carry);

// The canonical variants `env` can compile.
[[nodiscard]] std::vector<u32> runnableVariants(const Env& env, const FFTShape& shape);

// Checks the folding rules over every shape FFTShape::allShapes() produces. Returns the number of failed checks.
[[nodiscard]] u32 variantSelfCheck();

}  // namespace tune
