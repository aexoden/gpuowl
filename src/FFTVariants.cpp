// Copyright (C) Jason Lynch

#include "FFTVariants.h"

#include "log.h"
#include "OptionSpace.h"

#include <algorithm>
#include <string>

namespace tune {

namespace {

bool contains(const std::vector<u32>& v, u32 x) { return std::ranges::find(v, x) != v.end(); }

bool legalDigits(u32 variant) {
  return variant <= LAST_VARIANT && variant_W(variant) < N_VARIANT_W && variant_M(variant) < N_VARIANT_M &&
    variant_H(variant) < N_VARIANT_H;
}

std::vector<u32> passDigits(const FFTShape& shape, u32 size) {
  if (!fftParts(shape.fft_type).fp64) { return {2}; }
  if (size > 1024) { return {1, 2}; }
  if (size == 1024) { return {0, 2}; }
  return {0, 1, 2};
}

u32 foldDigit(const std::vector<u32>& digits, u32 digit) {
  if (contains(digits, digit)) { return digit; }
  return digits.back();
}

string where(const FFTShape& shape, u32 variant) {
  return shape.spec() + ":" + std::to_string(variant_W(variant)) + std::to_string(variant_M(variant)) +
    std::to_string(variant_H(variant));
}

} // namespace

FFTParts fftParts(enum FFT_TYPES type) {
  switch (type) {
  case FFT64: return {.fp64 = true};
  case FFT3161: return {.gf31 = true, .gf61 = true};
  case FFT3261: return {.fp32 = true, .gf61 = true};
  case FFT61: return {.gf61 = true};
  case FFT323161: return {.fp32 = true, .gf31 = true, .gf61 = true};
  case FFT3231: return {.fp32 = true, .gf31 = true};
  case FFT6431: return {.fp64 = true, .gf31 = true};
  case FFT31: return {.gf31 = true};
  case FFT32: return {.fp32 = true};
  }

  return {};
}

std::vector<u32> widthDigits(const FFTShape& shape) { return passDigits(shape, shape.width); }

std::vector<u32> heightDigits(const FFTShape& shape) { return passDigits(shape, shape.height); }

std::vector<u32> middleDigits(const FFTShape& shape) {
  FFTParts const parts = fftParts(shape.fft_type);
  if (!(parts.fp64 || parts.fp32) || shape.middle == 2) { return {0}; }
  return {0, 1};
}

std::vector<u32> allVariants(const FFTShape& shape) {
  std::vector<u32> variants;

  for (u32 w : widthDigits(shape)) {
    for (u32 m : middleDigits(shape)) {
      for (u32 h : heightDigits(shape)) { variants.push_back(variant_WMH(w, m, h)); }
    }
  }

  return variants;
}

u32 canonicalVariant(const FFTShape& shape, u32 variant) {
  if (!legalDigits(variant)) { return variant; }
  if (fftParts(shape.fft_type).fp64 && shape.width > 1024 && variant_W(variant) == 0) { return variant; }

  return variant_WMH(foldDigit(widthDigits(shape), variant_W(variant)),
                     foldDigit(middleDigits(shape), variant_M(variant)),
                     foldDigit(heightDigits(shape), variant_H(variant)));
}

u32 defaultVariant(const FFTShape& shape) {
  return variant_WMH(widthDigits(shape).back(), middleDigits(shape).back(), heightDigits(shape).back());
}

bool carryAffectsKernels(enum FFT_TYPES type) {
  return type == FFT64 || type == FFT32 || type == FFT31 || type == FFT61;
}

enum CARRY_KIND canonicalCarry(const FFTShape& shape, u32 variant, enum CARRY_KIND carry) {
  if (carry == CARRY_AUTO || !carryAffectsKernels(shape.fft_type)) { return CARRY_AUTO; }
  if (carry == CARRY_64) { return CARRY_64; }

  float const autoBpw = FFTConfig{shape, variant, CARRY_AUTO}.maxBpw();
  return shape.carry32BPW() < autoBpw ? CARRY_32 : CARRY_AUTO;
}

std::vector<u32> runnableVariants(const Env& env, const FFTShape& shape) {
  bool const broadcast = env.isAmd && !env.cudaBackend && !env.noAsm;

  std::vector<u32> out;

  for (u32 v : allVariants(shape)) {
    if (!broadcast && (variant_W(v) == 0 || variant_H(v) == 0)) { continue; }
    out.push_back(v);
  }

  return out;
}

u32 variantSelfCheck() {
  u32 problems = 0;
  auto fail = [&problems](const string& mes) {
    log("FFTVariants: %s\n", mes.c_str());
    ++problems;
  };

  for (const FFTShape& shape : FFTShape::allShapes()) {
    std::vector<u32> const variants = allVariants(shape);
    FFTParts const parts = fftParts(shape.fft_type);
    FFTConfig const probe{shape, defaultVariant(shape), CARRY_AUTO};

    if (probe.FFT_FP64 != parts.fp64 || probe.FFT_FP32 != parts.fp32 || probe.NTT_GF31 != parts.gf31 ||
        probe.NTT_GF61 != parts.gf61) {
      fail("fftParts disagrees with FFTConfig for " + shape.spec());
    }

    if (!contains(variants, defaultVariant(shape))) { fail("the default variant is not offered for " + shape.spec()); }

    if (canonicalVariant(shape, LAST_VARIANT) != defaultVariant(shape)) {
      fail("a spec without a variant does not mean the default for " + shape.spec());
    }

    for (u32 v = 0; v <= LAST_VARIANT; v = next_variant(v)) {
      u32 const c = canonicalVariant(shape, v);
      bool const compiles = !(parts.fp64 && shape.width > 1024 && variant_W(v) == 0);

      if (compiles && !contains(variants, c)) {
        fail(where(shape, v) + " folds onto " + where(shape, c) + ", not offered");
      }

      if (canonicalVariant(shape, c) != c) { fail(where(shape, v) + " does not fold idempotently"); }

      if (!compiles) { continue; }

      for (enum CARRY_KIND carry : {CARRY_32, CARRY_64, CARRY_AUTO}) {
        FFTConfig const fft{shape, v, carry};
        if (fft.variant != c) { fail(where(shape, v) + ": FFTConfig does not canonicalize the variant"); }

        if (canonicalCarry(shape, c, fft.carry) != fft.carry) {
          fail(fft.spec() + ": the carry fold is not idempotent");
        }

        if (carry == CARRY_64 && carryAffectsKernels(shape.fft_type) && fft.carry != CARRY_64) {
          fail(where(shape, v) + ": the 64-bit carry folded away");
        }

        if (fft.carry == CARRY_32 && fft.maxBpw() >= FFTConfig{shape, v, CARRY_AUTO}.maxBpw()) {
          fail(fft.spec() + ": the 32-bit carry stays distinct but costs no bits per word");
        }

        if (FFTConfig{fft.spec()}.spec() != fft.spec()) { fail(fft.spec() + " does not survive its own spec"); }
      }
    }
  }

  return problems;
}

} // namespace tune
