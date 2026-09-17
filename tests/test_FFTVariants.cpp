// Copyright (C) Jason Lynch

// GPU-free tests of variant and carry canonicalization (src/FFTVariants.cpp). The facts the folding rests on are read
// out of the bundled kernel source rather than restated, so an upstream kernel change that invalidates a fold fails
// here instead of silently giving two configurations one name.

#include "test.h"

#include "FFTVariants.h"
#include "OptionSpace.h"

#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <string>

const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

using namespace tune;

namespace {

std::string clSource(const char* name) {
  for (size_t i = 0; i < getClFileNames().size(); ++i) {
    if (strcmp(getClFileNames()[i], name) == 0) { return getClFiles()[i]; }
  }
  return "";
}

// The (WG, RADIX) pairs for which fft_common's FP64 body has a variant-2 specialization of its own. Everywhere else
// digit 2 runs the digit-1 code.
const std::set<std::pair<u32, u32>>& fp64Variant2Bodies() {
  static const std::set<std::pair<u32, u32>> bodies = [] {
    std::string const src = clSource("fftbase.cl");
    auto const begin = src.find("#if FFT_FP64");
    auto const end = src.find("#if FFT_FP32", begin);
    std::string const fp64 = src.substr(begin, end - begin);

    std::set<std::pair<u32, u32>> out;
    std::regex const body{R"(#elif WG == (\d+) && RADIX == (\d+) && VARIANT == 2)"};
    for (std::sregex_iterator it{fp64.begin(), fp64.end(), body}, last; it != last; ++it) {
      out.emplace(u32(std::stoul((*it)[1])), u32(std::stoul((*it)[2])));
    }
    return out;
  }();
  return bodies;
}

// The FFT types whose carryFused body declares its carries as CFcarry, the one type CARRY64 chooses.
const std::set<int>& carryTypedTypes() {
  static const std::set<int> types = [] {
    static const std::map<std::string, int> ids{{"FFT64", FFT64},     {"FFT3161", FFT3161},     {"FFT3261", FFT3261},
                                                {"FFT61", FFT61},     {"FFT323161", FFT323161}, {"FFT3231", FFT3231},
                                                {"FFT6431", FFT6431}, {"FFT31", FFT31},         {"FFT32", FFT32}};
    std::string const src = clSource("carryfused.cl");
    std::regex const section{R"(#(?:el)?if FFT_TYPE == (\w+))"};

    std::set<int> out;
    std::vector<std::pair<size_t, std::string>> starts;
    for (std::sregex_iterator it{src.begin(), src.end(), section}, last; it != last; ++it) {
      starts.emplace_back(size_t(it->position()), (*it)[1]);
    }
    for (size_t i = 0; i < starts.size(); ++i) {
      size_t const end = i + 1 < starts.size() ? starts[i + 1].first : src.size();
      if (src.substr(starts[i].first, end - starts[i].first).find("CFcarry") != std::string::npos) {
        out.insert(ids.at(starts[i].second));
      }
    }
    return out;
  }();
  return types;
}

// An independent description of the kernels a spelling compiles, from the kernel facts above: which fft_common body
// each pass runs, the middle chain defaults where a floating-point middle reads them, and when CARRY64 is defined.
std::string kernelKey(const FFTShape& shape, u32 v, CARRY_KIND carry) {
  FFTConfig const parts{shape, 202, CARRY_AUTO};

  // cl/base.cl: WG = G_W = WIDTH / NW and RADIX = NW, and likewise for the height.
  auto pass = [&](u32 size, u32 radix, u32 digit) {
    if (!parts.FFT_FP64) { return std::string{"-"}; }
    if (digit == 2 && !fp64Variant2Bodies().contains({size / radix, radix})) { return std::string{"1"}; }
    return std::to_string(digit);
  };

  std::string key = pass(shape.width, shape.nW(), variant_W(v)) + "/";
  key += ((parts.FFT_FP64 || parts.FFT_FP32) && shape.middle != 2) ? std::to_string(variant_M(v)) : "-";
  key += "/" + pass(shape.height, shape.nH(), variant_H(v)) + "/";

  if (!carryTypedTypes().contains(shape.fft_type)) { return key + "-"; }
  if (carry == CARRY_64) { return key + "64"; }

  // CARRY_AUTO takes the long carry only above the cap, so over a reach below it, it is the 32-bit carry.
  u64 const reach = u64(FFTConfig{shape, v, CARRY_AUTO}.maxBpw() * shape.size());
  bool const autoGoesLong = shape.needsLargeCarry(reach);
  if (carry == CARRY_32) { return key + (autoGoesLong ? "32" : "short"); }
  return key + (autoGoesLong ? "auto" : "short");
}

bool compiles(const FFTShape& shape, u32 v) {
  return !(shape.fft_type == FFT64 || shape.fft_type == FFT6431) || shape.width <= 1024 || variant_W(v) != 0;
}

std::vector<FFTShape> everyShape() {
  std::vector<FFTShape> shapes = FFTShape::allShapes();
  // Not enumerated by allShapes(), but valid FFT types all the same.
  shapes.emplace_back(FFT3231, 512, 8, 512);
  shapes.emplace_back(FFT31, 512, 8, 512);
  shapes.emplace_back(FFT32, 512, 8, 512);
  return shapes;
}

std::string canonicalSpec(const std::string& spec) { return FFTConfig{spec}.spec(); }

Env nvidia() { return {.isNvidia = true, .computeCapability = 806}; }

std::string variants(const std::vector<u32>& vs) {
  std::string s;
  for (u32 v : vs) {
    s += (s.empty() ? "" : ",") + std::to_string(v / 100) + std::to_string(v / 10 % 10) + std::to_string(v % 10);
  }
  return s;
}

} // namespace

TEST(kernel_facts_are_found) {
  // Guards the parsers above: were they to find nothing, every key would collapse and the main test would prove little.
  // (256, 4) is a radix-4 1024 body that no shape has produced since 1024 became radix 8.
  CHECK(fp64Variant2Bodies() == (std::set<std::pair<u32, u32>>{{64, 4}, {64, 8}, {256, 4}, {512, 8}}));
  CHECK(carryTypedTypes() == (std::set<int>{FFT64, FFT32, FFT31, FFT61, FFT3231}));
  for (int t = 0; t < 60; ++t) { CHECK_EQ(carryAffectsKernels(FFT_TYPES(t)), carryTypedTypes().contains(t)); }

  // The middle digit reaches the kernels only as the MM_CHAIN / MM2_CHAIN defaults.
  for (size_t i = 0; i < getClFileNames().size(); ++i) {
    std::string const name = getClFileNames()[i];
    if (name == "base.cl" || name == "fft-middle.cl") { continue; }
    CHECK(!strstr(getClFiles()[i], "FFT_VARIANT_M"));
  }
  std::string const middle = clSource("fft-middle.cl");
  CHECK(middle.find("#define MM_CHAIN (FFT_VARIANT_M == 0 ? 0 : 1)") != std::string::npos);
  CHECK(middle.find("#define MM2_CHAIN (FFT_VARIANT_M == 0 ? 0 : 2)") != std::string::npos);
}

TEST(one_name_per_configuration) {
  u32 shapes = 0;
  for (const FFTShape& shape : everyShape()) {
    std::map<std::string, std::set<std::string>> keysOfName, namesOfKey;
    std::string const where = " at " + shape.spec();

    for (u32 v = 0; v <= LAST_VARIANT; v = next_variant(v)) {
      if (!compiles(shape, v)) { continue; }
      for (CARRY_KIND carry : {CARRY_32, CARRY_64, CARRY_AUTO}) {
        std::string const name = FFTConfig{shape, v, carry}.spec();
        std::string const key = kernelKey(shape, v, carry);
        keysOfName[name].insert(key);
        namesOfKey[key].insert(name);
      }
    }

    for (const auto& [name, keys] : keysOfName) {
      if (keys.size() != 1) {
        testing::fail(__FILE__, __LINE__, name + " names " + std::to_string(keys.size()) + " kernel sets" + where);
      }
    }
    for (const auto& [key, names] : namesOfKey) {
      if (names.size() != 1) {
        testing::fail(__FILE__, __LINE__,
                      "kernels " + key + " have " + std::to_string(names.size()) + " names, e.g. " + *names.begin() +
                        where);
      }
    }

    size_t names = 0;
    for (u32 v : allVariants(shape)) {
      names += carryAffectsKernels(shape.fft_type) ? (canonicalCarry(shape, v, CARRY_32) == CARRY_32 ? 3 : 2) : 1;
    }
    CHECK_EQ(keysOfName.size(), names);
    ++shapes;
  }
  CHECK_EQ(shapes, u32(FFTShape::allShapes().size() + 3));
}

TEST(variant_folds) {
  // Width and height 1024: digits 1 and 2 share the WG = 128 body.
  CHECK_EQ(canonicalSpec("1K:8:1K:101"), std::string{"1K:8:1K:202"});
  CHECK_EQ(canonicalSpec("1K:8:256:101"), std::string{"1K:8:256:201"});
  CHECK_EQ(canonicalSpec("512:8:1K:111"), std::string{"512:8:1K:112"});
  CHECK_EQ(canonicalSpec("1K:8:1K:000"), std::string{"1K:8:1K:000"});
  CHECK_EQ(canonicalSpec("4K:12:512:100"), std::string{"4K:12:512:100"});
  CHECK_EQ(canonicalSpec("512:15:512:101"), std::string{"512:15:512:101"});
  CHECK_EQ(canonicalSpec("51:1K:8:512:111"), std::string{"51:1K:8:512:211"});

  // MIDDLE == 2: neither chain is read.
  CHECK_EQ(canonicalSpec("256:2:256:111"), std::string{"256:2:256:101"});

  // No FP64 part: the width and height digits select nothing, and without a float part neither does the middle.
  CHECK_EQ(canonicalSpec("1:512:8:512:010"), std::string{"1:512:8:512:202"});
  CHECK_EQ(canonicalSpec("3:1K:8:256:212"), std::string{"3:1K:8:256:202"});
  CHECK_EQ(canonicalSpec("2:512:8:512:010"), std::string{"2:512:8:512:212"});
  CHECK_EQ(canonicalSpec("50:512:8:512:000"), std::string{"50:512:8:512:202"});

  // A spec without a variant means the default, and so does its canonical spelling.
  CHECK_EQ(canonicalSpec("512:15:512"), std::string{"512:15:512:212"});
  CHECK_EQ(canonicalSpec("1K:8:1K"), std::string{"1K:8:1K:212"});
  CHECK_EQ(canonicalSpec("1:512:8:512"), std::string{"1:512:8:512:202"});
  CHECK_EQ(canonicalSpec("2:512:8:512"), std::string{"2:512:8:512:212"});

  // Width digit 0 cannot compile above 1024, so it is left for the kernels to refuse.
  CHECK_EQ(FFTConfig(FFTShape{FFT64, 4096, 12, 512}, 12, CARRY_AUTO).variant, 12u);

  CHECK_EQ(variants(allVariants(FFTShape{FFT64, 512, 15, 512})),
           std::string{"000,001,002,010,011,012,100,101,102,110,111,112,200,201,202,210,211,212"});
  CHECK_EQ(variants(allVariants(FFTShape{FFT64, 1024, 4, 1024})), std::string{"000,002,010,012,200,202,210,212"});
  CHECK_EQ(variants(allVariants(FFTShape{FFT64, 4096, 2, 512})), std::string{"100,101,102,200,201,202"});
  CHECK_EQ(variants(allVariants(FFTShape{FFT323161, 512, 8, 512})), std::string{"202,212"});
  CHECK_EQ(variants(allVariants(FFTShape{FFT61, 512, 8, 512})), std::string{"202"});
}

TEST(carry_folds) {
  // The hybrids whose carryFused has fixed carry types.
  CHECK_EQ(canonicalSpec("4:1K:8:256:202:1"), std::string{"4:1K:8:256:202"});
  CHECK_EQ(canonicalSpec("51:512:8:512:101:0"), std::string{"51:512:8:512:101"});

  // FFT3231's carryFused takes its carry type from CARRY64 like the single-arithmetic types, so its
  // spellings no longer fold: the 32-bit carry caps 256:2:256 below the reach CARRY_AUTO has there.
  CHECK_EQ(canonicalSpec("50:256:2:256:202:0"), std::string{"50:256:2:256:202:0"});
  CHECK(FFTConfig{"50:256:2:256:202:0"}.maxBpw() < FFTConfig{"50:256:2:256:202"}.maxBpw());
  CHECK_EQ(canonicalSpec("50:256:2:256:202:1"), std::string{"50:256:2:256:202:1"});

  // FP64: the 32-bit carry caps 256:2:256 below its table reach, but not 1K:5:256.
  CHECK_EQ(canonicalSpec("256:2:256:202:0"), std::string{"256:2:256:202:0"});
  CHECK(FFTConfig{"256:2:256:202:0"}.maxBpw() < FFTConfig{"256:2:256:202"}.maxBpw());
  CHECK_EQ(canonicalSpec("1K:5:256:202:0"), std::string{"1K:5:256:202"});
  CHECK_EQ(canonicalSpec("1K:5:256:202:1"), std::string{"1K:5:256:202:1"});

  // Folding follows the canonical variant's reach: 1K:5:1K reaches further at digit 2 than at digit 1, above the cap.
  CHECK_EQ(canonicalSpec("1K:5:1K:101:0"), std::string{"1K:5:1K:202:0"});

  // A pure GF61 NTT reaches far beyond the 32-bit cap, where CARRY_AUTO takes the long carry.
  CHECK_EQ(canonicalSpec("3:1K:8:256:202:0"), std::string{"3:1K:8:256:202:0"});
}

TEST(runnable_variants) {
  FFTShape const fp64{FFT64, 1024, 8, 256};
  CHECK_EQ(variants(runnableVariants(Env{.isAmd = true}, fp64)), variants(allVariants(fp64)));
  CHECK_EQ(variants(runnableVariants(nvidia(), fp64)), std::string{"201,202,211,212"});
  CHECK_EQ(variants(runnableVariants(Env{.isAmd = true, .noAsm = true}, fp64)), std::string{"201,202,211,212"});
  CHECK_EQ(variants(runnableVariants(Env{.isAmd = true, .cudaBackend = true}, fp64)), std::string{"201,202,211,212"});

  FFTShape const wide{FFT64, 4096, 12, 512};
  CHECK_EQ(variants(runnableVariants(Env{.isAmd = true}, wide)),
           std::string{"100,101,102,110,111,112,200,201,202,210,211,212"});

  FFTShape const ntt{FFT3161, 512, 8, 512};
  CHECK_EQ(variants(runnableVariants(nvidia(), ntt)), std::string{"202"});
}

TEST(variant_self_check_is_clean) { CHECK_EQ(variantSelfCheck(), 0u); }