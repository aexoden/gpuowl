// Copyright (C) Jason Lynch

#include "OptionSpace.h"

#include "Args.h"
#include "clwrap.h"
#include "Context.h"
#include "FFTVariants.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>

#ifdef CUDA_BACKEND
#include <cuda.h>
#endif

namespace tune {

int useValue(const UseConfig& config, const string& key, int valNotFound) {
  auto it = config.find(key);
  return it == config.end() ? valNotFound : atoi(it->second.c_str());
}

string Env::label() const {
  string s = isAmd ? "amd" : (isNvidia ? "nvidia" : "other");
  s += cudaBackend ? ",cuda" : ",ocl";
  if (noAsm) { s += ",noasm"; }
  if (isNvidia) { s += ",cc" + to_string(computeCapability); }
  if (pdlLaunch) { s += ",pdl"; }
  return s;
}

Env detectEnv(const Context& context, const Args& args) {
  auto const id = context.deviceId();
  Env env;
  env.isAmd = isAmdGpu(id);
  env.isNvidia = isNvidiaGpu(id);
#ifdef CUDA_BACKEND
  env.cudaBackend = true;
#if CUDA_VERSION >= 12'000
  env.pdlLaunch = true;
#endif
#endif
  env.noAsm = args.value("NO_ASM", 0) != 0;
  if (env.isNvidia) { env.computeCapability = getNvidiaComputeCapability(id); }
  env.deviceName = getDeviceName(id);
  env.driverVersion = getDriverVersion(id);
  return env;
}

namespace {

//
// Predicates shared by several options.
//

bool openclBackend(const Env& e, const FFTConfig&, const UseConfig&) { return !e.cudaBackend; }
bool cudaBackend(const Env& e, const FFTConfig&, const UseConfig&) { return e.cudaBackend; }

bool hasFP64(const Env&, const FFTConfig& f, const UseConfig&) { return f.FFT_FP64; }
bool hasFP32(const Env&, const FFTConfig& f, const UseConfig&) { return f.FFT_FP32; }
bool hasGF31(const Env&, const FFTConfig& f, const UseConfig&) { return f.NTT_GF31; }
bool hasGF61(const Env&, const FFTConfig& f, const UseConfig&) { return f.NTT_GF61; }
bool hasFloat(const Env&, const FFTConfig& f, const UseConfig&) { return f.FFT_FP64 || f.FFT_FP32; }
bool hasNtt(const Env&, const FFTConfig& f, const UseConfig&) { return f.NTT_GF31 || f.NTT_GF61; }

// The middle-buffer geometry is compiled only under "#if !INPLACE".
bool inplaceOff(const Env&, const FFTConfig&, const UseConfig& d) { return useValue(d, "INPLACE", 0) == 0; }

u32 numDataTypes(const FFTConfig& f) {
  return ((f.FFT_FP64 || f.FFT_FP32) ? 1u : 0u) + (f.NTT_GF31 ? 1u : 0u) + (f.NTT_GF61 ? 1u : 0u);
}

//
// LDS budget
//

u32 shuflBytesW(const UseConfig& d) { return u32(useValue(d, "SHUFL_BYTES_W", 8)); }
u32 shuflBytesH(const UseConfig& d) { return u32(useValue(d, "SHUFL_BYTES_H", 8)); }

u32 maxWmul(const FFTConfig& f, const UseConfig& d) {
  u32 m = LDS_BUDGET / (f.shape.width * shuflBytesW(d));
  if (m > 2 && f.shape.width >= 1024) { m = 2; }
  if (m > 1 && f.shape.width >= 4096) { m = 1; }
  return m;
}

u32 effectiveWmul(const FFTConfig& f, const UseConfig& d) {
  return std::min(u32(useValue(d, "WMUL", 2)), maxWmul(f, d));
}

bool ldsPadWFits(const FFTConfig& f, const UseConfig& d) {
  return f.shape.width * shuflBytesW(d) * effectiveWmul(f, d) < LDS_BUDGET;
}

int effectiveLdsPadW(const FFTConfig& f, const UseConfig& d) {
  return ldsPadWFits(f, d) ? useValue(d, "LDSPAD_W", 1) : 0;
}

u32 heightLdsBlocks(const UseConfig& d) { return useValue(d, "TAIL_KERNELS", 2) < 2 ? 1 : 2; }

u32 maxStriping(const FFTConfig& f, const UseConfig& d) {
  return f.shape.width / (useValue(d, "MULTI_Q", 0) ? 128 : 64);
}

//
// Inert Options
//

// UNROLL is read only by the generic "for (s = 1; s < WG; s *= RADIX)" loop of each type's fft_common (cl/fftbase.cl).
// Every "WG == 128 && RADIX == 8" body -- width or height 1024 -- bypasses that loop in all four types.  In FP64,
// variant digit 0 has its own loop and digit 2 a specialisation at every size a shape can produce, so a pure FFT64
// reads UNROLL only at digit 1; an FP32, GF31 or GF61 part keeps the generic loop live away from 1024.
bool unrollInert(u32 size, u32 digit, const FFTConfig& f) {
  return size == 1024 || (f.shape.fft_type == FFT64 && digit != 1);
}

bool fp64PassReadsTabMulChain(u32 size, u32 digit) { return digit == 1 || (digit == 2 && size == 1024); }

u32 tabMulChainTouches(const Env&, const FFTConfig& f, const UseConfig&) {
  u32 touches = 0;
  if (fp64PassReadsTabMulChain(f.shape.width, variant_W(f.variant))) { touches |= KG_WIDTH; }
  if (fp64PassReadsTabMulChain(f.shape.height, variant_H(f.variant))) { touches |= KG_HEIGHT | KG_TAIL; }
  return touches;
}

//
// Middle Chains
//

bool middleDigitPinsChains(const FFTConfig& f) { return variant_M(f.variant) == 1; }

int mmChainDefault(const FFTConfig& f) { return middleDigitPinsChains(f) ? 1 : 0; }
int mm2ChainDefault(const FFTConfig& f) { return middleDigitPinsChains(f) ? 2 : 0; }

bool mm2ChainInert(const FFTConfig& f) { return f.shape.middle < 5; }

//
// CUDA Registers
//

const vector<int> REG_LADDER{0, 1, 2, 4, 48, 56, 64, 72, 80, 88, 96, 104, 112, 128};

// NOREG=1 makes numCudaRegisters() return early, so no REG* key reaches the compiler.
bool regUsable(const Env& e, const UseConfig& d) { return e.cudaBackend && useValue(d, "NOREG", 0) == 0; }

template<enum FFT_TYPES T> bool regCarryApplies(const Env& e, const FFTConfig& f, const UseConfig& d) {
  return regUsable(e, d) && f.shape.fft_type == T;
}

bool regFP64(const Env& e, const FFTConfig& f, const UseConfig& d) { return regUsable(e, d) && f.FFT_FP64; }
bool regFP32(const Env& e, const FFTConfig& f, const UseConfig& d) { return regUsable(e, d) && f.FFT_FP32; }
bool regGF31(const Env& e, const FFTConfig& f, const UseConfig& d) { return regUsable(e, d) && f.NTT_GF31; }
bool regGF61(const Env& e, const FFTConfig& f, const UseConfig& d) { return regUsable(e, d) && f.NTT_GF61; }

void addRegOptions(vector<Option>& t) {
  struct RegKey {
    const char* key;
    u32 touches;
    Predicate applies;
  };

  static const RegKey keys[] = {
    // carryFused, one per FFT type
    {"REGCF64", KG_CARRY, regCarryApplies<FFT64>},
    {"REGCF3161", KG_CARRY, regCarryApplies<FFT3161>},
    {"REGCF3261", KG_CARRY, regCarryApplies<FFT3261>},
    {"REGCF61", KG_CARRY, regCarryApplies<FFT61>},
    {"REGCF323161", KG_CARRY, regCarryApplies<FFT323161>},
    {"REGCF3231", KG_CARRY, regCarryApplies<FFT3231>},
    {"REGCF6431", KG_CARRY, regCarryApplies<FFT6431>},
    {"REGCF31", KG_CARRY, regCarryApplies<FFT31>},
    {"REGCF32", KG_CARRY, regCarryApplies<FFT32>},

    // fftMiddleIn / fftMiddleOut / tailSquare, one per arithmetic part present
    {"REGMI64", KG_MIDDLE_IN, regFP64},
    {"REGMI32", KG_MIDDLE_IN, regFP32},
    {"REGMI31", KG_MIDDLE_IN, regGF31},
    {"REGMI61", KG_MIDDLE_IN, regGF61},
    {"REGMO64", KG_MIDDLE_OUT, regFP64},
    {"REGMO32", KG_MIDDLE_OUT, regFP32},
    {"REGMO31", KG_MIDDLE_OUT, regGF31},
    {"REGMO61", KG_MIDDLE_OUT, regGF61},
    {"REGTS64", KG_TAIL, regFP64},
    {"REGTS32", KG_TAIL, regFP32},
    {"REGTS31", KG_TAIL, regGF31},
    {"REGTS61", KG_TAIL, regGF61},
  };

  for (const RegKey& r : keys) {
    t.push_back({.key = r.key,
                 .scope = Scope::Shape,
                 .group = Group::Cuda,
                 .touches = r.touches,
                 .dependsOn = {"NOREG"},
                 .applies = r.applies,
                 .values = REG_LADDER,
                 .defaultValue = 0});
  }
}

//
// Option Table
//

vector<Option> buildTable() {
  vector<Option> t;

  // Placement: where data lives in the buffers. INPLACE is structural: with it on, the middle-buffer layout ceases to
  // exist and L2_STRIPING comes into existence.
  t.push_back({.key = "INPLACE",
               .group = Group::Placement,
               .touches = KG_GLOBAL,
               .structural = true,
               .values = {0, 1},
               .defaultValue = 0});

  t.push_back({.key = "L2_STRIPING",
               .group = Group::Placement,
               .touches = KG_GLOBAL,
               .dependsOn = {"INPLACE", "MULTI_Q"},
               .applies = [](const Env&, const FFTConfig& f,
                             const UseConfig& d) { return useValue(d, "INPLACE", 0) != 0 && maxStriping(f, d) >= 1; },
               .valuesFn =
                 [](const Env&, const FFTConfig& f, const UseConfig& d) {
                   vector<int> v{0};
                   for (u32 s = 1; s <= maxStriping(f, d); s *= 2) { v.push_back(int(s)); }
                   return v;
                 },
               .defaultValue = 0});

  // MiddleIn / MiddleOut
  t.push_back({.key = "IN_WG",
               .group = Group::Placement,
               .touches = KG_MIDDLE_IN | KG_TAIL | KG_HEIGHT,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {64, 128, 256},
               .defaultValue = 128});
  t.push_back({.key = "IN_SIZEX",
               .group = Group::Placement,
               .touches = KG_MIDDLE_IN | KG_TAIL | KG_HEIGHT,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {8, 16, 32},
               .defaultValue = 16});
  t.push_back({.key = "OUT_WG",
               .group = Group::Placement,
               .touches = KG_MIDDLE_OUT | KG_CARRY | KG_WIDTH,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {64, 128, 256},
               .defaultValue = 128});
  t.push_back({.key = "OUT_SIZEX",
               .group = Group::Placement,
               .touches = KG_MIDDLE_OUT | KG_CARRY | KG_WIDTH,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {8, 16, 32},
               .defaultValue = 16});
  t.push_back({.key = "PAD",
               .group = Group::Placement,
               .touches = KG_WIDTH | KG_HEIGHT | KG_MIDDLE_IN | KG_MIDDLE_OUT | KG_TAIL | KG_CARRY,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {0, 64, 128, 256, 512},
               .defaultFn = [](const Env& e, const FFTConfig&, const UseConfig&) { return e.isAmd ? 256 : 0; }});

  // Middle Transposes
  t.push_back({.key = "MIDDLE_IN_LDS_TRANSPOSE",
               .group = Group::Middle,
               .touches = KG_MIDDLE_IN,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {0, 1},
               .defaultValue = 1});
  t.push_back({.key = "MIDDLE_OUT_LDS_TRANSPOSE",
               .group = Group::Middle,
               .touches = KG_MIDDLE_OUT,
               .dependsOn = {"INPLACE"},
               .applies = inplaceOff,
               .values = {0, 1},
               .defaultValue = 1});

  t.push_back({.key = "MM_CHAIN",
               .group = Group::Middle,
               .touches = KG_MIDDLE_IN | KG_MIDDLE_OUT,
               .accuracyImpact = AccuracyImpact::Suspected,
               .applies = hasFloat,
               .valuesFn =
                 [](const Env&, const FFTConfig& f, const UseConfig&) {
                   // With MM2_CHAIN inert, MM_CHAIN=1 is middle digit 1's pair.
                   if (middleDigitPinsChains(f)) { return vector<int>{1}; }
                   return mm2ChainInert(f) ? vector<int>{0} : vector<int>{0, 1};
                 },
               .defaultFn = [](const Env&, const FFTConfig& f, const UseConfig&) { return mmChainDefault(f); },
               // Both arms reduce to the shared WADD(1, w) plus a zero-iteration loop.
               .inert = [](const Env&, const FFTConfig& f, const UseConfig&) { return f.shape.middle == 2; },
               .inertWhen = "MIDDLE == 2"});
  t.push_back({.key = "MM2_CHAIN",
               .group = Group::Middle,
               .touches = KG_MIDDLE_IN | KG_MIDDLE_OUT,
               .accuracyImpact = AccuracyImpact::Suspected,
               .dependsOn = {"MM_CHAIN"},
               .applies = hasFloat,
               .valuesFn =
                 [](const Env&, const FFTConfig& f, const UseConfig& d) {
                   if (middleDigitPinsChains(f)) { return vector<int>{2}; }
                   return useValue(d, "MM_CHAIN", mmChainDefault(f)) == 1 ? vector<int>{0, 1} : vector<int>{0, 1, 2};
                 },
               .defaultFn = [](const Env&, const FFTConfig& f, const UseConfig&) { return mm2ChainDefault(f); },
               // Every branch sits in middleMul2's "MIDDLE >= SHARP_MIDDLE" arm.
               .inert = [](const Env&, const FFTConfig& f, const UseConfig&) { return mm2ChainInert(f); },
               .inertWhen = "MIDDLE < 5"});
  t.push_back({.key = "MIDDLE_CHAIN",
               .group = Group::Middle,
               .touches = KG_MIDDLE_IN | KG_MIDDLE_OUT,
               .applies = hasNtt,
               .values = {0, 1},
               .defaultValue = 0,
               // Both branches compute the same value and return.
               .inert = [](const Env&, const FFTConfig& f, const UseConfig&) { return f.shape.middle == 2; },
               .inertWhen = "MIDDLE == 2"});

  t.push_back({.key = "LOADS",
               .scope = Scope::Device,
               .group = Group::Memory,
               .touches = KG_GLOBAL,
               .compound = true,
               .defaultValue = 0});
  t.push_back({.key = "STORES",
               .scope = Scope::Device,
               .group = Group::Memory,
               .touches = KG_GLOBAL,
               .compound = true,
               .defaultValue = 0});
  // Marks the trig and weight table pointers restrict, which lets the compiler hoist their loads at the cost of
  // registers.  Upstream measured it a loss on nVidia at the default register caps and a small win alongside a
  // raised one, so which way it goes is a timing question rather than a rule.
  t.push_back({.key = "ENABLE_RESTRICT",
               .scope = Scope::Device,
               .group = Group::Memory,
               .touches = KG_GLOBAL,
               .values = {0, 1},
               .defaultValue = 0});

  t.push_back({.key = "FAST_BARRIER",
               .scope = Scope::Device,
               .group = Group::Queues,
               .touches = KG_GLOBAL,
               .applies = openclBackend,
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "OLD_FENCE",
               .scope = Scope::Device,
               .group = Group::Queues,
               .touches = KG_CARRY,
               .values = {0, 1},
               .defaultFn = [](const Env& e, const FFTConfig&, const UseConfig&) { return e.isAmd ? 0 : 1; }});
  t.push_back({.key = "ENABLE_BARSYNC",
               .scope = Scope::Device,
               .group = Group::Queues,
               .touches = KG_GLOBAL,
               .applies = [](const Env& e, const FFTConfig&, const UseConfig&) { return e.hasPtx(200); },
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "MULTI_Q",
               .group = Group::Queues,
               .touches = KG_GLOBAL,
               .applies = [](const Env&, const FFTConfig& f, const UseConfig&) { return numDataTypes(f) >= 2; },
               .values = {0, 1},
               .defaultValue = 0});

  t.push_back({.key = "TAIL_KERNELS",
               .scope = Scope::Variant,
               .group = Group::Tail,
               .touches = KG_TAIL,
               .values = {0, 1, 2, 3},
               .defaultValue = 2});
  t.push_back({.key = "TAIL_TRIGS",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL,
               .applies = hasFP64,
               .values = {0, 1, 2},
               .defaultValue = 2});
  t.push_back({.key = "TAIL_TRIGS31",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL,
               .applies = hasGF31,
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "TAIL_TRIGS32",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL,
               .accuracyImpact = AccuracyImpact::Yes,
               .applies = hasFP32,
               .values = {0, 1, 2},
               .defaultValue = 2});
  t.push_back({.key = "TAIL_TRIGS61",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL,
               .applies = hasGF61,
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back(
    {.key = "TABMUL_CHAIN",
     .scope = Scope::Family,
     .group = Group::Tail,
     .touchesFn = tabMulChainTouches,
     .applies = hasFP64,
     .values = {0, 1},
     .defaultValue = 0,
     .inert = [](const Env& e, const FFTConfig& f, const UseConfig& d) { return tabMulChainTouches(e, f, d) == 0; },
     .inertWhen = "width and height each at variant 0, or at variant 2 away from 1024"});
  t.push_back({.key = "TABMUL_CHAIN31",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL | KG_WIDTH | KG_HEIGHT,
               .applies = hasGF31,
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "TABMUL_CHAIN32",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL | KG_WIDTH | KG_HEIGHT,
               .accuracyImpact = AccuracyImpact::Yes,
               .applies = hasFP32,
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "TABMUL_CHAIN61",
               .scope = Scope::Family,
               .group = Group::Tail,
               .touches = KG_TAIL | KG_WIDTH | KG_HEIGHT,
               .applies = hasGF61,
               .values = {0, 1},
               .defaultValue = 0});

  t.push_back({.key = "MODM31",
               .scope = Scope::Family,
               .group = Group::Arith,
               .touches = KG_GLOBAL,
               .applies = hasGF31,
               .values = {0, 1, 2},
               .defaultValue = 0});
  t.push_back(
    {.key = "DISABLE_MUL64",
     .scope = Scope::Family,
     .group = Group::Arith,
     .touches = KG_GLOBAL,
     .structural = true,
     .applies = [](const Env& e, const FFTConfig& f, const UseConfig&) { return f.NTT_GF61 && e.hasPtx(100); },
     .values = {0, 1},
     .defaultValue = 0});
  t.push_back(
    {.key = "ENABLE_ALT_MUL64",
     .scope = Scope::Family,
     .group = Group::Arith,
     .touches = KG_GLOBAL,
     .dependsOn = {"DISABLE_MUL64"},
     .applies = [](const Env& e, const FFTConfig& f,
                   const UseConfig& d) { return f.NTT_GF61 && e.hasPtx(200) && useValue(d, "DISABLE_MUL64", 0) != 0; },
     .values = {0, 1},
     .defaultValue = 0});
  t.push_back(
    {.key = "ENABLE_MAD64",
     .scope = Scope::Family,
     .group = Group::Arith,
     .touches = KG_GLOBAL,
     .applies = [](const Env& e, const FFTConfig& f, const UseConfig&) { return f.NTT_GF61 && e.hasPtx(200); },
     .values = {0, 1},
     .defaultValue = 0});

  t.push_back({.key = "SHUFL_BYTES_W",
               .group = Group::Width,
               .touches = KG_WIDTH | KG_CARRY,
               .structural = true,
               .valuesFn =
                 [](const Env&, const FFTConfig& f, const UseConfig&) {
                   vector<int> v;
                   for (u32 b : {4, 8, 16}) {
                     if (f.shape.width * b <= LDS_BUDGET) { v.push_back(int(b)); }
                   }
                   return v;
                 },
               .defaultValue = 8});
  t.push_back({.key = "LDSPAD_W",
               .group = Group::Width,
               .touches = KG_WIDTH | KG_CARRY,
               .structural = true,
               .dependsOn = {"SHUFL_BYTES_W", "WMUL"},
               .applies = [](const Env&, const FFTConfig& f, const UseConfig& d) { return ldsPadWFits(f, d); },
               .values = {0, 1},
               .defaultValue = 1});
  t.push_back({.key = "LDSSWIZ_W",
               .group = Group::Width,
               .touches = KG_WIDTH | KG_CARRY,
               .dependsOn = {"SHUFL_BYTES_W", "LDSPAD_W", "WMUL"},
               .applies = [](const Env&, const FFTConfig& f,
                             const UseConfig& d) { return shuflBytesW(d) >= 8 && effectiveLdsPadW(f, d) == 0; },
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "UNROLL_W",
               .scope = Scope::Variant,
               .group = Group::Width,
               .touches = KG_WIDTH,
               .applies = openclBackend,
               .values = {0, 1},
               .defaultFn = [](const Env& e, const FFTConfig&, const UseConfig&) { return e.isAmd ? 0 : 1; },
               .inert = [](const Env&, const FFTConfig& f,
                           const UseConfig&) { return unrollInert(f.shape.width, variant_W(f.variant), f); },
               .inertWhen = "width 1024, or FFT64 with variant W != 1"});
  t.push_back({.key = "ZEROHACK_W",
               .scope = Scope::Variant,
               .group = Group::Width,
               .touches = KG_CARRY,
               .values = {0, 1},
               .defaultValue = 1});
  t.push_back(
    {.key = "WMUL",
     .group = Group::Width,
     .touches = KG_CARRY,
     .dependsOn = {"SHUFL_BYTES_W"},
     .valuesFn =
       [](const Env&, const FFTConfig& f, const UseConfig& d) {
         vector<int> v;
         for (u32 w : {1, 2, 4}) {
           if (w <= maxWmul(f, d)) { v.push_back(int(w)); }
         }
         return v;
       },
     .defaultFn = [](const Env&, const FFTConfig& f, const UseConfig& d) { return int(std::min(2u, maxWmul(f, d))); }});

  t.push_back({.key = "SHUFL_BYTES_H",
               .group = Group::Height,
               .touches = KG_HEIGHT | KG_TAIL,
               .structural = true,
               .dependsOn = {"TAIL_KERNELS"},
               .valuesFn =
                 [](const Env&, const FFTConfig& f, const UseConfig& d) {
                   vector<int> v;
                   for (u32 b : {4, 8, 16}) {
                     if (f.shape.height * b * heightLdsBlocks(d) <= LDS_BUDGET) { v.push_back(int(b)); }
                   }
                   return v;
                 },
               .defaultValue = 8});
  t.push_back({.key = "LDSPAD_H",
               .group = Group::Height,
               .touches = KG_HEIGHT | KG_TAIL,
               .structural = true,
               .dependsOn = {"SHUFL_BYTES_H"},
               .values = {0, 1},
               .defaultValue = 1});
  t.push_back({.key = "LDSSWIZ_H",
               .group = Group::Height,
               .touches = KG_HEIGHT | KG_TAIL,
               .dependsOn = {"SHUFL_BYTES_H", "LDSPAD_H"},
               .applies = [](const Env&, const FFTConfig&,
                             const UseConfig& d) { return shuflBytesH(d) >= 8 && useValue(d, "LDSPAD_H", 1) == 0; },
               .values = {0, 1},
               .defaultValue = 0});
  t.push_back({.key = "UNROLL_H",
               .scope = Scope::Variant,
               .group = Group::Height,
               .touches = KG_HEIGHT,
               .applies = openclBackend,
               .values = {0, 1},
               .defaultFn = [](const Env& e, const FFTConfig& f,
                               const UseConfig&) { return (e.isAmd && f.shape.height >= 1024) ? 0 : 1; },
               .inert = [](const Env&, const FFTConfig& f,
                           const UseConfig&) { return unrollInert(f.shape.height, variant_H(f.variant), f); },
               .inertWhen = "height 1024, or FFT64 with variant H != 1"});
  t.push_back({.key = "ZEROHACK_H",
               .scope = Scope::Variant,
               .group = Group::Height,
               .touches = KG_TAIL,
               .values = {0, 1},
               .defaultValue = 1});

  t.push_back({.key = "GRAPHS",
               .scope = Scope::Device,
               .group = Group::Cuda,
               .touches = KG_GLOBAL,
               .applies = cudaBackend,
               .values = {0, 1},
               .defaultValue = 1});
  t.push_back({.key = "L1CUDA",
               .scope = Scope::Device,
               .group = Group::Cuda,
               .touches = KG_GLOBAL,
               .applies = cudaBackend,
               .values = {0, 1, 2, 3},
               .defaultValue = 0});
  t.push_back({.key = "NOREG",
               .scope = Scope::Device,
               .group = Group::Cuda,
               .touches = KG_GLOBAL,
               .structural = true,
               .applies = cudaBackend,
               .values = {0, 1},
               .defaultValue = 0});
  addRegOptions(t);
  t.push_back({.key = "PDL",
               .scope = Scope::Device,
               .group = Group::Cuda,
               .touches = KG_GLOBAL,
               .applies = [](const Env& e, const FFTConfig&,
                             const UseConfig&) { return e.cudaBackend && e.pdlLaunch && e.hasPtx(900); },
               .values = {0, 1},
               .defaultValue = 0});

  // Recognized but never searched keys.
  for (const char* key : {
         "STATS", "DEBUG", "NO_ASM", "CARRY64", "ROUNDOFF_LIMIT",       // Debugging and correctness options
         "SHOULD_BE_FASTER", "ENABLE_BETTER_ONEPAIRSQ", "TEST_KERNEL",  // Upstream A/B options
         "TRY_SHL30", "TRY_SHL31", "TRY_SQRT2",                         // GF61 experiments, correctness unverified
         "WAVEFRONT",                                                   // Set too high, bar() skips needed barriers
         "USE_REGISTER_BARSYNC",   // Picks barsync()'s body, reached only through LDSMUL > 1
         "LDSMUL_W", "LDSMUL_H",   // Shared LDS: slower where tried upstream, and unsafe with variant 2
         "ENABLE_FP32_VARIANT_2",  // Would widen the FP32 variant space
       }) {
    t.push_back({.key = key, .kind = Kind::Fixed});
  }

  t.push_back({.key = "BIGLIT", .kind = Kind::Deprecated});
  t.push_back({.key = "NONTEMPORAL", .kind = Kind::Deprecated});

  return t;
}

vector<AccessClass> buildAccessClasses() {
  return {
    {.name = "FFT data", .digit = 0, .loadModes = {0, 1, 2, 3, 4, 5}, .storeModes = {0, 1, 2, 3}},
    {.name = "carry shuttle", .digit = 1, .pairs = {{0, 0}, {1, 1}, {4, 2}, {5, 0}}},
    {.name = "trig, frequently reused", .digit = 2, .loadModes = {0, 5}},
    {.name = "trig, several uses", .digit = 3, .loadModes = {0, 1, 2, 3, 4, 5}},
    {.name = "trig, used once", .digit = 4, .loadModes = {0, 1, 2, 3, 4, 5}},
  };
}

const char* toString(Scope s) {
  switch (s) {
  case Scope::Device: return "device";
  case Scope::Family: return "family";
  case Scope::Shape: return "shape";
  case Scope::Variant: return "variant";
  }
  return "?";
}

const char* toString(AccuracyImpact a) {
  switch (a) {
  case AccuracyImpact::None: return "";
  case AccuracyImpact::Suspected: return "?";
  case AccuracyImpact::Yes: return "yes";
  }
  return "?";
}

string touchesString(u32 touches) {
  if (touches & KG_GLOBAL) { return "global"; }
  string s;
  auto add = [&s](const char* name) { s += (s.empty() ? "" : "+") + string(name); };
  if (touches & KG_WIDTH) { add("W"); }
  if (touches & KG_HEIGHT) { add("H"); }
  if (touches & KG_MIDDLE_IN) { add("MidIn"); }
  if (touches & KG_MIDDLE_OUT) { add("MidOut"); }
  if (touches & KG_TAIL) { add("Tail"); }
  if (touches & KG_CARRY) { add("Carry"); }
  return s.empty() ? "-" : s;
}

string join(const vector<int>& v) {
  string s;
  for (int x : v) { s += (s.empty() ? "" : ",") + to_string(x); }
  return s.empty() ? "-" : s;
}

string join(const vector<string>& v, const char* sep) {
  string s;
  for (const string& x : v) { s += (s.empty() ? "" : sep) + x; }
  return s;
}

string join(const vector<Group>& groups) {
  vector<string> names;
  for (Group g : groups) { names.push_back(toString(g)); }
  return join(names, " ");
}

bool contains(const vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

}  // namespace

const vector<Option>& allOptions() {
  static const vector<Option> table = buildTable();
  return table;
}

const Option* findOption(const string& key) {
  for (const Option& o : allOptions()) {
    if (o.key == key) { return &o; }
  }
  return nullptr;
}

bool isKnownKey(const string& key) { return findOption(key) != nullptr; }

const vector<Group>& allGroups() {
  static const vector<Group> groups{
    Group::Placement, Group::Middle, Group::Memory, Group::Queues, Group::Tail,
    Group::Width,     Group::Height, Group::Arith,  Group::Cuda,
  };
  return groups;
}

const char* toString(Group group) {
  switch (group) {
  case Group::None: return "-";
  case Group::Placement: return "Placement";
  case Group::Middle: return "Middle";
  case Group::Memory: return "Memory";
  case Group::Queues: return "Queues";
  case Group::Tail: return "Tail";
  case Group::Width: return "Width";
  case Group::Height: return "Height";
  case Group::Arith: return "Arith";
  case Group::Cuda: return "Cuda";
  }
  return "?";
}

u32 kernelGroupOf(const string& name) {
  struct Entry {
    const char* prefix;
    u32 groups;
  };

  static const Entry table[] = {
    {"kfftMidIn", KG_MIDDLE_IN},
    {"kfftMidOut", KG_MIDDLE_OUT},
    {"kfftHin", KG_HEIGHT},
    {"ktailSquare", KG_TAIL | KG_HEIGHT},
    {"ktailMul", KG_TAIL | KG_HEIGHT},
    {"kfftW", KG_WIDTH},
    {"kfftP", KG_WIDTH},
    {"kCarryFused", KG_CARRY | KG_WIDTH},
    {"kCarry", KG_CARRY},
    {"carryB", KG_CARRY},
  };

  for (const Entry& e : table) {
    if (name.compare(0, strlen(e.prefix), e.prefix) == 0) { return e.groups; }
  }

  return 0;
}

vector<const Option*> applicableOptions(const Env& env, const FFTConfig& fft, const UseConfig& decided) {
  vector<const Option*> out;
  for (const Option& o : allOptions()) {
    if (o.kind != Kind::Tunable || !o.appliesTo(env, fft, decided) || o.isInert(env, fft, decided)) { continue; }
    if (!o.compound && o.valuesFor(env, fft, decided).size() < 2) { continue; }
    out.push_back(&o);
  }
  return out;
}

ClusterGraph clusterGraph(const Env& env, const FFTConfig& fft, const UseConfig& decided) {
  ClusterGraph g;
  for (const Option* o : applicableOptions(env, fft, decided)) {
    g.touches[o->group] |= o->touchesFor(env, fft, decided);
  }

  vector<Group> local;
  for (Group group : allGroups()) {
    auto it = g.touches.find(group);
    if (it == g.touches.end()) { continue; }

    if (it->second & KG_GLOBAL) {
      g.topTier.push_back(group);
    } else {
      local.push_back(group);
    }
  }

  vector<bool> placed(local.size());
  for (size_t i = 0; i < local.size(); ++i) {
    if (placed[i]) { continue; }

    vector<Group> cluster;
    vector<size_t> pending{i};
    placed[i] = true;

    while (!pending.empty()) {
      size_t const j = pending.back();
      pending.pop_back();
      cluster.push_back(local[j]);
      for (size_t k = 0; k < local.size(); ++k) {
        if (!placed[k] && (g.touches.at(local[j]) & g.touches.at(local[k]))) {
          placed[k] = true;
          pending.push_back(k);
        }
      }
    }

    std::ranges::sort(cluster);
    g.clusters.push_back(cluster);
  }

  return g;
}

string clusterPictureMismatch(const ClusterGraph& g) {
  static const std::set<Group> top{Group::Placement, Group::Memory, Group::Queues, Group::Arith, Group::Cuda};

  for (Group group : g.topTier) {
    if (!top.contains(group)) { return string(toString(group)) + " is in the top tier"; }
  }

  std::map<Group, size_t> clusterOf;

  for (size_t i = 0; i < g.clusters.size(); ++i) {
    const vector<Group>& cluster = g.clusters[i];
    string const name = "cluster {" + join(cluster) + "}";
    if (cluster.size() > MAX_PERMUTE) { return name + " has more than " + to_string(MAX_PERMUTE) + " groups"; }

    bool const hasWidth = std::ranges::find(cluster, Group::Width) != cluster.end();
    for (Group group : cluster) {
      clusterOf[group] = i;
      if (group == Group::Queues) {
        if (g.touches.at(group) != KG_CARRY || !hasWidth) { return name + " holds Queues other than OLD_FENCE's"; }
      } else if (top.contains(group)) {
        return name + " contains " + toString(group);
      } else if (group == Group::Middle && cluster.size() > 1) {
        return name + " joins the middle kernels to another pass";
      }
    }
  }

  auto const tail = clusterOf.find(Group::Tail);
  auto const height = clusterOf.find(Group::Height);
  if (tail != clusterOf.end() && height != clusterOf.end() && tail->second != height->second) {
    return "Tail and Height, which share the tail kernels, are in different clusters";
  }

  return "";
}

const vector<AccessClass>& accessClasses() {
  static const vector<AccessClass> classes = buildAccessClasses();
  return classes;
}

bool loadModeExists(const Env& env, int mode) {
  switch (mode) {
  // LOAD
  case 0: return true;

  // NTLOAD
  case 1: return env.hasNontemporal();

  // L2LOAD, EFLOAD, LULOAD
  case 2:
  case 3:
  case 4: return env.hasPtx(200);

  // NCLOAD
  case 5: return env.hasPtx(500);
  }

  return false;
}

bool storeModeExists(const Env& env, int mode) {
  switch (mode) {
  // STORE
  case 0: return true;

  // NTSTORE
  case 1: return env.hasNontemporal();

  // L2STORE, EFSTORE
  case 2:
  case 3: return env.hasPtx(200);
  }
  return false;
}

vector<int> usableLoadModes(const Env& env, const AccessClass& cls) {
  vector<int> out;
  for (int m : cls.loadModes) {
    if (loadModeExists(env, m)) { out.push_back(m); }
  }
  return out;
}

vector<int> usableStoreModes(const Env& env, const AccessClass& cls) {
  vector<int> out;
  for (int m : cls.storeModes) {
    if (storeModeExists(env, m)) { out.push_back(m); }
  }
  return out;
}

vector<pair<int, int>> usablePairs(const Env& env, const AccessClass& cls) {
  vector<pair<int, int>> out;
  for (auto [load, store] : cls.pairs) {
    if (loadModeExists(env, load) && storeModeExists(env, store)) { out.emplace_back(load, store); }
  }
  return out;
}

u32 getDigit(u32 packed, u32 digit) {
  for (u32 i = 0; i < digit; ++i) { packed /= 10; }
  return packed % 10;
}

u32 setDigit(u32 packed, u32 digit, u32 value) {
  u32 scale = 1;
  for (u32 i = 0; i < digit; ++i) { scale *= 10; }
  return packed - getDigit(packed, digit) * scale + value * scale;
}

vector<MatrixPoint> selfCheckMatrix() {
  vector<Env> envs;

  for (bool amd : {true, false}) {
    for (bool cuda : {false, true}) {
      for (bool noAsm : {false, true}) {
        for (u32 cc : {0u, 700u, 900u}) {
          if (amd && cc) { continue; }
          for (bool pdl : {false, true}) {
            if (pdl && !(cuda && cc >= 900)) { continue; }

            envs.push_back({.isAmd = amd,
                            .isNvidia = !amd,
                            .cudaBackend = cuda,
                            .noAsm = noAsm,
                            .computeCapability = cc,
                            .pdlLaunch = pdl});
          }
        }
      }
    }
  }

  vector<FFTConfig> ffts;
  for (FFTShape shape : {FFTShape{FFT64, 512, 15, 512}, FFTShape{FFT64, 4096, 12, 512}, FFTShape{FFT64, 256, 4, 256},
                         FFTShape{FFT64, 1024, 8, 1024}, FFTShape{FFT64, 512, 8, 1024}}) {
    ffts.emplace_back(shape, 101, CARRY_AUTO);
    ffts.emplace_back(shape, 212, CARRY_AUTO);
  }
  ffts.emplace_back(FFTShape{FFT3161, 512, 4, 512}, 202, CARRY_AUTO);
  ffts.emplace_back(FFTShape{FFT3261, 512, 4, 512}, 202, CARRY_AUTO);
  ffts.emplace_back(FFTShape{FFT3161, 1024, 16, 1024}, 202, CARRY_AUTO);

  vector<vector<KeyVal>> const scenarios{
    {},
    {{"LDSPAD_W", "0"}, {"LDSPAD_H", "0"}},              // opens LDSSWIZ_*
    {{"SHUFL_BYTES_W", "4"}, {"SHUFL_BYTES_H", "4"}},    // closes LDSSWIZ_*
    {{"SHUFL_BYTES_W", "16"}, {"SHUFL_BYTES_H", "16"}},  // tightens WMUL, may close LDSPAD_W
    {{"WMUL", "4"}},
    {{"NOREG", "1"}},          // closes REG*
    {{"DISABLE_MUL64", "1"}},  // opens ENABLE_ALT_MUL64
  };

  vector<MatrixPoint> points;
  for (const Env& env : envs) {
    for (const FFTConfig& fft : ffts) {
      for (int inplace : {0, 1}) {
        for (const vector<KeyVal>& scenario : scenarios) {
          MatrixPoint p{.env = env, .fft = fft, .decided = {{"INPLACE", to_string(inplace)}}};
          p.label = env.label() + " " + fft.spec() + " INPLACE=" + to_string(inplace);

          for (const auto& [key, val] : scenario) {
            const Option* const o = findOption(key);
            if (!o || !o->appliesTo(env, fft, p.decided) ||
                !contains(o->valuesFor(env, fft, p.decided), atoi(val.c_str()))) {
              continue;
            }
            p.decided[key] = val;
            p.label += " " + key + "=" + val;
          }

          points.push_back(std::move(p));
        }
      }
    }
  }

  return points;
}

u32 selfCheck() {
  u32 problems = 0;
  auto fail = [&problems](const string& mes) {
    log("OptionSpace: %s\n", mes.c_str());
    ++problems;
  };

  const vector<Option>& table = allOptions();

  // Every key once; tunable keys carry search metadata, the others none.
  std::set<string> seen;
  std::set<Group> usedGroups;

  for (const Option& o : table) {
    if (!seen.insert(o.key).second) { fail(o.key + " is declared twice"); }
    if (o.kind == Kind::Tunable) {
      usedGroups.insert(o.group);
      if (o.group == Group::None) { fail(o.key + " is tunable but belongs to no group"); }
      if (o.compound != (o.values.empty() && !o.valuesFn)) {
        fail(o.key + (o.compound ? " is compound but lists values" : " is tunable but has no values"));
      }
      if (!std::ranges::is_sorted(o.values) || std::ranges::adjacent_find(o.values) != o.values.end()) {
        fail(o.key + "'s value list is not strictly ascending");
      }
      if (o.inert && !*o.inertWhen) { fail(o.key + " has an inert rule but does not say when"); }
      if (!o.touchesFn && o.touches == 0) { fail(o.key + " touches no kernel"); }
    } else if (o.group != Group::None || o.structural || o.compound || o.accuracyImpact != AccuracyImpact::None ||
               !o.dependsOn.empty() || o.applies || o.touchesFn || o.valuesFn || !o.values.empty() || o.defaultFn ||
               o.inert) {
      fail(o.key + " is never searched but declares search metadata");
    }
  }

  for (Group group : allGroups()) {
    if (!usedGroups.contains(group)) { fail(string("group ") + toString(group) + " has no tunable key"); }
  }

  // dependsOn names real keys and the graph is acyclic.
  {
    std::map<string, int> state;  // 1 while on the DFS stack, 2 once finished
    auto visit = [&](auto&& self, const Option& o) -> void {
      int& s = state[o.key];
      if (s == 2) { return; }
      if (s == 1) {
        fail("dependency cycle through " + o.key);
        return;
      }
      s = 1;
      for (const string& d : o.dependsOn) {
        const Option* const dep = findOption(d);
        if (!dep) {
          fail(o.key + " depends on unknown key " + d);
          continue;
        }
        if (dep->kind != Kind::Tunable) { fail(o.key + " depends on " + d + ", which is never searched"); }
        self(self, *dep);
      }
      state[o.key] = 2;
    };
    for (const Option& o : table) { visit(visit, o); }
  }

  // The access classes cover distinct digits and offer mode 0, the compound default, on every machine.
  {
    std::set<u32> digits;
    for (const AccessClass& a : accessClasses()) {
      if (!digits.insert(a.digit).second) { fail("two access classes share digit " + to_string(a.digit)); }
      if (a.digit > 9) { fail(a.name + " uses an out-of-range digit"); }
      bool const coupled = !a.pairs.empty();
      if (coupled == !(a.loadModes.empty() && a.storeModes.empty())) {
        fail(a.name + " must list either modes or pairs, not both or neither");
      }
      bool const offersDefault = coupled
        ? a.pairs.front() == pair{0, 0}
        : (a.loadModes.empty() || contains(a.loadModes, 0)) && (a.storeModes.empty() || contains(a.storeModes, 0));
      if (!offersDefault) { fail(a.name + " does not offer mode 0 first"); }
    }
    for (const Option& o : table) {
      if (o.compound && o.defaultValue != 0) { fail(o.key + ": a compound default must be 0, every class at mode 0"); }
    }
  }

  problems += variantSelfCheck();

  // Resolve the table at every point of the matrix.
  for (const MatrixPoint& p : selfCheckMatrix()) {
    string const where = " [" + p.label + "]";

    for (const Option& o : table) {
      if (o.kind != Kind::Tunable || o.compound || !o.appliesTo(p.env, p.fft, p.decided)) { continue; }
      vector<int> const vals = o.valuesFor(p.env, p.fft, p.decided);
      if (vals.empty()) {
        fail(o.key + " applies but offers no values" + where);
        continue;
      }
      if (!std::ranges::is_sorted(vals) || std::ranges::adjacent_find(vals) != vals.end()) {
        fail(o.key + " offers values out of order {" + join(vals) + "}" + where);
      }
      int const d = o.defaultFor(p.env, p.fft, p.decided);
      if (!contains(vals, d)) {
        fail(o.key + " default " + to_string(d) + " is not among {" + join(vals) + "}" + where);
      }
      if (!o.isInert(p.env, p.fft, p.decided) && o.touchesFor(p.env, p.fft, p.decided) == 0) {
        fail(o.key + " applies but touches no kernel" + where);
      }
    }

    ClusterGraph const g = clusterGraph(p.env, p.fft, p.decided);
    if (string const mismatch = clusterPictureMismatch(g); !mismatch.empty()) {
      fail("the cluster graph departs from the expected picture: " + mismatch + where);
    }

    // No present group is orphaned: each lands exactly once in the top tier or in one cluster.
    std::map<Group, int> placements;
    for (Group group : g.topTier) { ++placements[group]; }
    for (const vector<Group>& cluster : g.clusters) {
      for (Group group : cluster) { ++placements[group]; }
    }
    for (const auto& [group, _] : g.touches) {
      if (placements[group] != 1) {
        fail(string(toString(group)) + " is placed " + to_string(placements[group]) + " times" + where);
      }
    }
  }

  return problems;
}

u32 dumpOptionSpace(const Env& env, const FFTConfig& fft) {
  log("Option space for %s (%s, driver %s)\n",
      env.deviceName.empty() ? "an unidentified device" : env.deviceName.c_str(), env.label().c_str(),
      env.driverVersion.empty() ? "?" : env.driverVersion.c_str());
  log("Resolved for FFT %s.  S: structural, the search branches on it.  ACC: affects accuracy (? = suspected).\n",
      fft.spec().c_str());

  for (int inplace : {0, 1}) {
    UseConfig const decided{{"INPLACE", to_string(inplace)}};
    log("\n");
    log("-- branch INPLACE=%d --\n", inplace);
    log("%-24s %-10s %-7s %-12s %-1s %-3s %7s  %-22s %s\n", "KEY", "GROUP", "SCOPE", "TOUCHES", "S", "ACC", "DEFAULT",
        "VALUES", "DEPENDS ON");

    vector<string> single, inert;
    for (const Option& o : allOptions()) {
      if (o.kind != Kind::Tunable || !o.appliesTo(env, fft, decided)) { continue; }
      if (o.isInert(env, fft, decided)) {
        inert.push_back(o.key + " (" + o.inertWhen + ")");
        continue;
      }
      vector<int> const vals = o.valuesFor(env, fft, decided);
      int const d = o.defaultFor(env, fft, decided);
      if (!o.compound && vals.size() < 2) {
        single.push_back(o.key + "=" + to_string(d));
        continue;
      }
      log("%-24s %-10s %-7s %-12s %-1s %-3s %7d  %-22s %s\n", o.key.c_str(), toString(o.group), toString(o.scope),
          touchesString(o.touchesFor(env, fft, decided)).c_str(), o.structural ? "*" : "", toString(o.accuracyImpact),
          d, o.compound ? "(access classes)" : join(vals).c_str(), join(o.dependsOn, ",").c_str());
    }
    if (!single.empty()) { log("One value only here: %s\n", join(single, ", ").c_str()); }
    if (!inert.empty()) { log("Offered but inert in the kernels here: %s\n", join(inert, ", ").c_str()); }

    ClusterGraph const g = clusterGraph(env, fft, decided);
    log("Combo tiers: top tier {%s}\n", join(g.topTier).c_str());
    for (const vector<Group>& cluster : g.clusters) {
      u32 mask = 0;
      for (Group group : cluster) { mask |= g.touches.at(group); }
      log("             cluster  {%s}, touching %s\n", join(cluster).c_str(), touchesString(mask).c_str());
    }
    string const mismatch = clusterPictureMismatch(g);
    log("             %s\n",
        mismatch.empty() ? "(matches the expected picture)"
                         : ("departs from the expected picture: " + mismatch).c_str());
  }

  {
    vector<string> fixed, deprecated;
    for (const Option& o : allOptions()) {
      if (o.kind == Kind::Fixed) { fixed.push_back(o.key); }
      if (o.kind == Kind::Deprecated) { deprecated.push_back(o.key); }
    }
    log("\n");
    log("Recognised, never searched: %s\n", join(fixed, " ").c_str());
    log("Deprecated: %s\n", join(deprecated, " ").c_str());
  }

  log("\n");
  log("Memory access classes (decimal digits of LOADS / STORES), modes usable here:\n");
  for (const AccessClass& a : accessClasses()) {
    string modes;
    if (!a.pairs.empty()) {
      vector<string> pairs;
      for (auto [load, store] : usablePairs(env, a)) { pairs.push_back(to_string(load) + "/" + to_string(store)); }
      modes = "load/store pairs " + join(pairs, ", ");
    } else {
      modes = "load " + join(usableLoadModes(env, a));
      if (!a.storeModes.empty()) { modes += ", store " + join(usableStoreModes(env, a)); }
    }
    log("  digit %u %-24s %s\n", a.digit, a.name.c_str(), modes.c_str());
  }

  u32 const problems = selfCheck();
  log("\n");
  log("Self-check over %u matrix points: %s\n", u32(selfCheckMatrix().size()),
      problems ? (to_string(problems) + " problem(s)").c_str() : "consistent");
  return problems;
}

}  // namespace tune
