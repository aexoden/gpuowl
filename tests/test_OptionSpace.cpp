// Copyright (C) Jason Lynch

// GPU-free tests of the option table (src/OptionSpace.cpp): the inventory, the inert and gating rules, the dependency
// declarations, the combo clusters, and the memory access classes.

#include "test.h"

#include "FFTVariants.h"
#include "OptionSpace.h"

#include <algorithm>
#include <fstream>
#include <regex>
#include <set>

using namespace tune;

namespace {

// Compute capability as clDefines() passes it: 806 is sm_86 (Ampere), 600 sm_60 (Pascal).
Env nvidia(u32 cc = 806, bool cuda = false, bool noAsm = false) {
  return {.isNvidia = true, .cudaBackend = cuda, .noAsm = noAsm, .computeCapability = cc};
}

Env amd() { return {.isAmd = true}; }

bool applicable(const Env& env, const string& spec, const UseConfig& decided, const string& key) {
  for (const Option* o : applicableOptions(env, FFTConfig{spec}, decided)) {
    if (o->key == key) { return true; }
  }

  return false;
}

bool inert(const Env& env, const string& spec, const string& key) {
  FFTConfig const fft{spec};
  const Option* const o = findOption(key);
  return o && o->appliesTo(env, fft, {}) && o->isInert(env, fft, {});
}

string valuesOf(const Env& env, const string& spec, const UseConfig& decided, const string& key) {
  string s;
  for (int v : findOption(key)->valuesFor(env, FFTConfig{spec}, decided)) {
    s += (s.empty() ? "" : ",") + to_string(v);
  }
  return s;
}

int defaultOf(const Env& env, const string& spec, const UseConfig& decided, const string& key) {
  return findOption(key)->defaultFor(env, FFTConfig{spec}, decided);
}

string names(const vector<Group>& groups) {
  string s;
  for (Group g : groups) { s += (s.empty() ? "" : " ") + string(toString(g)); }
  return s;
}

string describe(const ClusterGraph& g) {
  string s = "top{" + names(g.topTier) + "}";
  for (const vector<Group>& c : g.clusters) { s += " {" + names(c) + "}"; }
  return s;
}

string graphOf(const Env& env, const string& spec, const UseConfig& decided) {
  return describe(clusterGraph(env, FFTConfig{spec}, decided));
}

}  // namespace

TEST(self_check_is_clean) { CHECK_EQ(selfCheck(), 0u); }

TEST(structural_and_accuracy_flags) {
  const set<string> structural{"INPLACE",       "NOREG",    "DISABLE_MUL64", "SHUFL_BYTES_W",
                               "SHUFL_BYTES_H", "LDSPAD_W", "LDSPAD_H"};
  for (const Option& o : allOptions()) { CHECK_EQ(o.structural, structural.contains(o.key)); }

  CHECK(findOption("TAIL_TRIGS32")->accuracyImpact == AccuracyImpact::Yes);
  CHECK(findOption("TABMUL_CHAIN32")->accuracyImpact == AccuracyImpact::Yes);
  CHECK(findOption("MM_CHAIN")->accuracyImpact == AccuracyImpact::Suspected);
  CHECK(findOption("MM2_CHAIN")->accuracyImpact == AccuracyImpact::Suspected);
  CHECK(findOption("MODM31")->group == Group::Arith);

  for (const char* key : {"INPLACE", "L2_STRIPING", "IN_WG", "IN_SIZEX", "OUT_WG", "OUT_SIZEX", "PAD"}) {
    CHECK(findOption(key)->group == Group::Placement);
  }
}

TEST(known_keys) {
  // Every key the hardcoded list in Gpu.cpp accepted before this table replaced it.
  for (const char* key : {"FAST_BARRIER",
                          "STATS",
                          "IN_SIZEX",
                          "IN_WG",
                          "OUT_SIZEX",
                          "OUT_WG",
                          "UNROLL_H",
                          "UNROLL_W",
                          "ZEROHACK_H",
                          "ZEROHACK_W",
                          "NO_ASM",
                          "DEBUG",
                          "CARRY64",
                          "BIGLIT",
                          "NONTEMPORAL",
                          "INPLACE",
                          "PAD",
                          "MIDDLE_IN_LDS_TRANSPOSE",
                          "MIDDLE_OUT_LDS_TRANSPOSE",
                          "TAIL_KERNELS",
                          "TAIL_TRIGS",
                          "TAIL_TRIGS31",
                          "TAIL_TRIGS32",
                          "TAIL_TRIGS61",
                          "TABMUL_CHAIN",
                          "TABMUL_CHAIN31",
                          "TABMUL_CHAIN32",
                          "TABMUL_CHAIN61",
                          "MODM31",
                          "LOADS",
                          "STORES",
                          "NOREG",
                          "WMUL",
                          "MULTI_Q",
                          "GRAPHS",
                          "L1CUDA"}) {
    CHECK(isKnownKey(key));
  }
  // Keys that list missed, so a valid -use drew "unrecognized" while still reaching the compiler.
  for (const char* key : {"MM_CHAIN", "MM2_CHAIN", "MIDDLE_CHAIN", "SHUFL_BYTES_W", "SHUFL_BYTES_H", "LDSPAD_W",
                          "LDSPAD_H", "LDSSWIZ_W", "LDSSWIZ_H", "L2_STRIPING", "OLD_FENCE", "ENABLE_BARSYNC", "REGCF64",
                          "REGTS61", "PDL", "TRY_SQRT2", "WAVEFRONT", "ROUNDOFF_LIMIT", "USE_REGISTER_BARSYNC"}) {
    CHECK(isKnownKey(key));
  }
  // Recognized so -use accepts them, but never offered for search.
  for (const char* key : {"USE_REGISTER_BARSYNC", "LDSMUL_W", "LDSMUL_H", "WAVEFRONT", "TRY_SQRT2", "NO_ASM"}) {
    CHECK(isKnownKey(key) && findOption(key)->kind == Kind::Fixed);
    CHECK(!applicable(nvidia(900, true), "512:15:512:101", {}, key));
  }
  // Removed from the kernels by upstream 6cf0dcd, an environment variable, and misspellings.
  for (const char* key : {"NOWG2", "NOLDS2", "TRY_LDS_CARVEOUT", "inplace", "FOO", ""}) { CHECK(!isKnownKey(key)); }
}

TEST(inert_middle_chains) {
  Env const e = nvidia();
  CHECK(inert(e, "256:4:256:101", "MM2_CHAIN"));  // MIDDLE < 5
  CHECK(!inert(e, "512:5:512:101", "MM2_CHAIN"));
  CHECK(applicable(e, "512:5:512:101", {}, "MM2_CHAIN"));
  CHECK(!applicable(e, "256:4:256:101", {}, "MM2_CHAIN"));
  CHECK(inert(e, "2:512:4:512:202", "MM2_CHAIN"));  // FP32 reads it too

  CHECK(inert(e, "256:2:256:101", "MM_CHAIN"));  // MIDDLE == 2
  CHECK(applicable(e, "512:15:512:101", {}, "MM_CHAIN"));
  CHECK(applicable(e, "2:512:8:512:202", {}, "MM_CHAIN"));   // FP32 as well as FP64
  CHECK(!applicable(e, "3:512:4:512:202", {}, "MM_CHAIN"));  // no float part

  CHECK(inert(e, "3:256:2:256:202", "MIDDLE_CHAIN"));
  CHECK(applicable(e, "3:512:4:512:202", {}, "MIDDLE_CHAIN"));
  CHECK(!applicable(e, "512:15:512:101", {}, "MIDDLE_CHAIN"));

  // Defaults come off the M digit.
  CHECK_EQ(defaultOf(e, "512:15:512:101", {}, "MM_CHAIN"), 0);
  CHECK_EQ(defaultOf(e, "512:15:512:111", {}, "MM_CHAIN"), 1);
  CHECK_EQ(defaultOf(e, "512:15:512:111", {}, "MM2_CHAIN"), 2);
}

// The middle digit only defaults the chains, so each effective (MM_CHAIN, MM2_CHAIN) pair is offered under exactly one
// middle digit: digit 1 keeps its own defaults, digit 0 offers the rest.
TEST(middle_digit_partitions_chains) {
  Env const e = nvidia();
  const Option& mm = *findOption("MM_CHAIN");
  const Option& mm2 = *findOption("MM2_CHAIN");

  struct Row {
    const char* shape;
    size_t pairs;
  };
  for (const Row& row : {Row{"512:15:512", 6}, Row{"256:4:256", 2}, Row{"256:2:256", 1}, Row{"2:512:8:512", 6},
                         Row{"51:512:8:512", 6}, Row{"4:512:4:512", 2}}) {
    FFTShape const shape{row.shape};
    std::map<string, u32> digitOfPair;

    for (u32 variant : allVariants(shape)) {
      if (variant_W(variant) != 2 || variant_H(variant) != 2) { continue; }
      FFTConfig const fft{shape, variant, CARRY_AUTO};

      for (int a : mm.valuesFor(e, fft, {})) {
        UseConfig const d{{"MM_CHAIN", to_string(a)}};
        for (int b : mm2.valuesFor(e, fft, d)) {
          string const pair = (mm.isInert(e, fft, d) ? string("-") : to_string(a)) + "," +
            (mm2.isInert(e, fft, d) ? string("-") : to_string(b));
          auto const [it, fresh] = digitOfPair.emplace(pair, variant_M(variant));
          if (!fresh && it->second != variant_M(variant)) {
            testing::fail(__FILE__, __LINE__,
                          string(row.shape) + ": chains " + pair + " offered under both middle digits");
          }
        }
      }
    }
    CHECK_EQ(digitOfPair.size(), row.pairs);
  }

  CHECK_EQ(valuesOf(e, "512:15:512:212", {}, "MM_CHAIN"), string("1"));
  CHECK_EQ(valuesOf(e, "512:15:512:202", {{"MM_CHAIN", "1"}}, "MM2_CHAIN"), string("0,1"));
  CHECK_EQ(valuesOf(e, "256:4:256:202", {}, "MM_CHAIN"), string("0"));
  CHECK(!applicable(e, "256:4:256:212", {}, "MM_CHAIN"));
}

TEST(inert_unroll) {
  Env const e = nvidia();
  CHECK(applicable(e, "512:15:512:101", {}, "UNROLL_W"));
  CHECK(applicable(e, "512:15:512:101", {}, "UNROLL_H"));
  CHECK(inert(e, "512:15:512:202", "UNROLL_W"));  // FFT64 digit 2 is specialised
  CHECK(inert(e, "512:15:512:001", "UNROLL_W"));  // FFT64 digit 0 has its own loop
  CHECK(!inert(e, "512:15:512:001", "UNROLL_H"));
  CHECK(inert(e, "1K:8:1K:101", "UNROLL_W"));  // WG = 128 bodies bypass the loop
  CHECK(inert(e, "1K:8:1K:101", "UNROLL_H"));
  CHECK(!inert(e, "1K:8:512:101", "UNROLL_H"));
  CHECK(!inert(e, "51:512:8:512:202", "UNROLL_W"));  // FP64 + GF31: the GF31 loop reads it
  CHECK(!inert(e, "2:512:4:512:202", "UNROLL_W"));   // FP32's generic loop reads it
  CHECK(inert(e, "3:1K:8:1K:202", "UNROLL_W"));
  CHECK(!findOption("UNROLL_W")->appliesTo(nvidia(806, true), FFTConfig{"512:15:512:101"}, {}));  // NVRTC ignores it

  CHECK_EQ(defaultOf(amd(), "512:8:1K:101", {}, "UNROLL_H"), 0);
  CHECK_EQ(defaultOf(amd(), "1K:8:512:101", {}, "UNROLL_H"), 1);
  CHECK_EQ(defaultOf(e, "512:8:1K:101", {}, "UNROLL_H"), 1);
}

// A middle buffer's layout is compiled into both the kernel that writes it and the one that reads it.
TEST(layout_touches_both_sides) {
  Env const e = nvidia();
  FFTConfig const fft{"512:15:512:101"};
  auto touches = [&](const char* key) { return findOption(key)->touchesFor(e, fft, {}); };
  CHECK_EQ(touches("IN_WG"), u32(KG_MIDDLE_IN | KG_TAIL | KG_HEIGHT));
  CHECK_EQ(touches("IN_SIZEX"), u32(KG_MIDDLE_IN | KG_TAIL | KG_HEIGHT));
  CHECK_EQ(touches("OUT_WG"), u32(KG_MIDDLE_OUT | KG_CARRY | KG_WIDTH));
  CHECK_EQ(touches("OUT_SIZEX"), u32(KG_MIDDLE_OUT | KG_CARRY | KG_WIDTH));
  CHECK_EQ(touches("PAD"), u32(KG_WIDTH | KG_HEIGHT | KG_MIDDLE_IN | KG_MIDDLE_OUT | KG_TAIL | KG_CARRY));
  CHECK_EQ(touches("MIDDLE_IN_LDS_TRANSPOSE"), u32(KG_MIDDLE_IN));
}

TEST(inert_tabmul_chain) {
  Env const e = nvidia();
  auto touches = [&](const string& spec) { return findOption("TABMUL_CHAIN")->touchesFor(e, FFTConfig{spec}, {}); };
  CHECK(applicable(e, "512:15:512:101", {}, "TABMUL_CHAIN"));
  CHECK_EQ(touches("512:15:512:101"), u32(KG_WIDTH | KG_HEIGHT | KG_TAIL));
  CHECK(inert(e, "512:15:512:212", "TABMUL_CHAIN"));  // partial_tabMul8 at WG = 64
  CHECK(!applicable(e, "512:15:512:212", {}, "TABMUL_CHAIN"));
  CHECK(inert(e, "256:4:256:202", "TABMUL_CHAIN"));
  CHECK(inert(e, "4K:12:512:212", "TABMUL_CHAIN"));
  CHECK(inert(amd(), "512:15:512:002", "TABMUL_CHAIN"));  // digit 0 broadcasts
  CHECK(!inert(e, "1K:8:1K:212", "TABMUL_CHAIN"));        // the WG = 128 body calls tabMul
  CHECK_EQ(touches("512:8:1K:212"), u32(KG_HEIGHT | KG_TAIL));
  CHECK_EQ(touches("512:15:512:102"), u32(KG_WIDTH));
  // The other types have no variant-2 bodies that skip tabMul.
  CHECK(!inert(e, "2:512:4:512:202", "TABMUL_CHAIN32"));
  CHECK(!inert(e, "3:512:4:512:202", "TABMUL_CHAIN61"));
}

TEST(placement_and_queue_gates) {
  Env const e = nvidia();
  UseConfig const inplace{{"INPLACE", "1"}};
  for (const char* key :
       {"IN_WG", "IN_SIZEX", "OUT_WG", "OUT_SIZEX", "PAD", "MIDDLE_IN_LDS_TRANSPOSE", "MIDDLE_OUT_LDS_TRANSPOSE"}) {
    CHECK(applicable(e, "512:15:512:101", {}, key));
    CHECK(!applicable(e, "512:15:512:101", inplace, key));
  }
  CHECK(!applicable(e, "512:15:512:101", {}, "L2_STRIPING"));
  CHECK_EQ(valuesOf(e, "512:15:512:101", inplace, "L2_STRIPING"), string("0,1,2,4,8"));
  CHECK_EQ(valuesOf(e, "512:15:512:101", {{"INPLACE", "1"}, {"MULTI_Q", "1"}}, "L2_STRIPING"), string("0,1,2,4"));
  CHECK_EQ(valuesOf(e, "256:4:256:101", {{"INPLACE", "1"}, {"MULTI_Q", "1"}}, "L2_STRIPING"), string("0,1,2"));

  CHECK(!applicable(e, "512:15:512:101", {}, "MULTI_Q"));   // one data type
  CHECK(applicable(e, "1:512:4:512:202", {}, "MULTI_Q"));   // GF31 + GF61
  CHECK(applicable(e, "51:512:8:512:202", {}, "MULTI_Q"));  // FP64 + GF31
  CHECK(applicable(e, "2:512:4:512:202", {}, "MULTI_Q"));   // FP32 + GF61 is two as well

  CHECK(applicable(e, "512:15:512:101", {}, "FAST_BARRIER"));
  CHECK(!applicable(nvidia(806, true), "512:15:512:101", {}, "FAST_BARRIER"));
  CHECK(!applicable(nvidia(806, false, true), "512:15:512:101", {}, "ENABLE_BARSYNC"));
  CHECK(!applicable(nvidia(100), "512:15:512:101", {}, "ENABLE_BARSYNC"));
  CHECK_EQ(defaultOf(amd(), "512:15:512:101", {}, "OLD_FENCE"), 0);
  CHECK_EQ(defaultOf(e, "512:15:512:101", {}, "OLD_FENCE"), 1);
  CHECK_EQ(defaultOf(amd(), "512:15:512:101", {}, "PAD"), 256);
  CHECK_EQ(defaultOf(e, "512:15:512:101", {}, "INPLACE"), 0);
}

TEST(arith_and_cuda_gates) {
  CHECK(applicable(nvidia(100), "3:512:4:512:202", {}, "DISABLE_MUL64"));
  CHECK(!applicable(nvidia(100), "3:512:4:512:202", {{"DISABLE_MUL64", "1"}}, "ENABLE_ALT_MUL64"));
  CHECK(!applicable(nvidia(806), "3:512:4:512:202", {}, "ENABLE_ALT_MUL64"));
  CHECK(applicable(nvidia(806), "3:512:4:512:202", {{"DISABLE_MUL64", "1"}}, "ENABLE_ALT_MUL64"));
  CHECK(!applicable(amd(), "3:512:4:512:202", {}, "DISABLE_MUL64"));
  CHECK(!applicable(nvidia(806), "512:15:512:101", {}, "ENABLE_MAD64"));

  Env const cuda = nvidia(806, true);
  CHECK(applicable(cuda, "512:15:512:101", {}, "REGCF64"));
  CHECK(!applicable(cuda, "512:15:512:101", {{"NOREG", "1"}}, "REGCF64"));
  CHECK(!applicable(nvidia(806), "512:15:512:101", {}, "REGCF64"));
  // Exactly one carryFused budget per FFT type.
  int regcf = 0;
  for (const Option* o : applicableOptions(cuda, FFTConfig{"1:512:4:512:202"}, {})) {
    if (o->key.starts_with("REGCF")) {
      ++regcf;
      CHECK_EQ(o->key, string("REGCF3161"));
    }
  }
  CHECK_EQ(regcf, 1);
  CHECK(applicable(cuda, "1:512:4:512:202", {}, "REGMI31"));
  CHECK(!applicable(cuda, "1:512:4:512:202", {}, "REGMI64"));

  auto pdl = [](Env env) {
    env.pdlLaunch = true;
    return env;
  };

  CHECK(applicable(pdl(nvidia(900, true)), "512:15:512:101", {}, "PDL"));
  CHECK(!applicable(nvidia(900, true), "512:15:512:101", {}, "PDL"));
  CHECK(!applicable(pdl(nvidia(806, true)), "512:15:512:101", {}, "PDL"));
  CHECK(!applicable(nvidia(900, false), "512:15:512:101", {}, "PDL"));
  CHECK(!applicable(pdl(nvidia(900, true, true)), "512:15:512:101", {}, "PDL"));
}

TEST(lds_budget) {
  Env const e = nvidia();
  CHECK_EQ(valuesOf(e, "4K:12:512:101", {}, "SHUFL_BYTES_W"), string("4,8"));
  CHECK_EQ(valuesOf(e, "1K:8:1K:101", {}, "SHUFL_BYTES_W"), string("4,8,16"));
  CHECK_EQ(valuesOf(e, "1K:8:1K:101", {}, "SHUFL_BYTES_H"), string("4,8,16"));
  CHECK_EQ(valuesOf(e, "1K:8:1K:101", {{"TAIL_KERNELS", "0"}}, "SHUFL_BYTES_H"), string("4,8,16"));

  // WMUL: min(2, maxWmul), capped at 2 from width 1K and 1 at 4K.
  CHECK_EQ(valuesOf(e, "4K:12:512:101", {}, "WMUL"), string("1"));
  CHECK(!applicable(e, "4K:12:512:101", {}, "WMUL"));
  CHECK_EQ(defaultOf(e, "4K:12:512:101", {}, "WMUL"), 1);
  CHECK_EQ(valuesOf(e, "4K:12:512:101", {{"SHUFL_BYTES_W", "4"}}, "WMUL"), string("1"));
  CHECK_EQ(valuesOf(e, "1K:8:1K:101", {{"SHUFL_BYTES_W", "16"}}, "WMUL"), string("1,2"));
  CHECK_EQ(valuesOf(e, "512:15:512:101", {}, "WMUL"), string("1,2,4"));
  CHECK_EQ(defaultOf(e, "512:15:512:101", {}, "WMUL"), 2);

  // At 1K * 16 * 2 the width row fills the budget, so clDefines() forces LDSPAD_W=0 and swizzling is
  // offered instead.
  UseConfig const full{{"SHUFL_BYTES_W", "16"}};
  CHECK(!applicable(e, "1K:8:1K:101", full, "LDSPAD_W"));
  CHECK(applicable(e, "1K:8:1K:101", full, "LDSSWIZ_W"));
  CHECK(applicable(e, "1K:8:1K:101", {{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}}, "LDSPAD_W"));
  CHECK(!applicable(e, "1K:8:1K:101", {{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}}, "LDSSWIZ_W"));
  CHECK(applicable(e, "1K:8:1K:101", {{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}, {"LDSPAD_W", "0"}}, "LDSSWIZ_W"));
  CHECK(applicable(e, "4K:12:512:101", {}, "LDSSWIZ_W"));  // 4K * 8 * 1 fills it too
  CHECK(!applicable(e, "512:15:512:101", {{"SHUFL_BYTES_W", "4"}, {"LDSPAD_W", "0"}}, "LDSSWIZ_W"));

  CHECK(applicable(e, "512:15:512:101", {}, "LDSPAD_H"));
  CHECK(!applicable(e, "512:15:512:101", {}, "LDSSWIZ_H"));
  CHECK(applicable(e, "512:15:512:101", {{"LDSPAD_H", "0"}}, "LDSSWIZ_H"));
  CHECK(!applicable(e, "512:15:512:101", {{"LDSPAD_H", "0"}, {"SHUFL_BYTES_H", "4"}}, "LDSSWIZ_H"));
}

// Every key that applies, touchesFn, valuesFn, defaultFn or inert reads must be in dependsOn: moving any other key
// through every value it offers must leave the option exactly as it was.
TEST(depends_on_is_complete) {
  struct Snapshot {
    bool applies, inert;
    u32 touches;
    vector<int> values;
    int defaultValue;
    bool operator==(const Snapshot&) const = default;
  };
  auto snap = [](const Option& o, const Env& e, const FFTConfig& f, const UseConfig& d) {
    return Snapshot{o.appliesTo(e, f, d), o.isInert(e, f, d), o.touchesFor(e, f, d), o.valuesFor(e, f, d),
                    o.defaultFor(e, f, d)};
  };

  // From every matrix point, not only the all-defaults ones: some reads only show once a scenario has
  // moved a structural key (LDSSWIZ_W reads WMUL only where SHUFL_BYTES_W=16 fills the LDS budget).
  std::set<string> reported;
  for (const MatrixPoint& p : selfCheckMatrix()) {
    for (const Option& moved : allOptions()) {
      if (moved.kind != Kind::Tunable || moved.compound) { continue; }
      for (int v : moved.valuesFor(p.env, p.fft, p.decided)) {
        UseConfig d = p.decided;
        d[moved.key] = to_string(v);
        for (const Option& o : allOptions()) {
          if (o.kind != Kind::Tunable || &o == &moved) { continue; }
          if (std::ranges::find(o.dependsOn, moved.key) != o.dependsOn.end()) { continue; }
          if (!(snap(o, p.env, p.fft, p.decided) == snap(o, p.env, p.fft, d)) &&
              reported.insert(o.key + "<-" + moved.key).second) {
            testing::fail(__FILE__, __LINE__,
                          o.key + " changes with " + moved.key + "=" + to_string(v) + " but does not declare it [" +
                            p.label + "]");
          }
        }
      }
    }
  }
}

TEST(clusters_match_expected_picture) {
  Env const ocl = nvidia();
  CHECK_EQ(graphOf(ocl, "512:15:512:101", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues} {Middle} {Tail Width Height}"));
  CHECK_EQ(graphOf(ocl, "512:15:512:101", {{"INPLACE", "1"}}),
           string("top{Placement Memory Queues} {Middle} {Tail Width Height}"));
  // Where TABMUL_CHAIN is inert, or read by the height pass alone, nothing couples the width pass to the tail.
  CHECK_EQ(graphOf(ocl, "512:15:512:212", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues} {Middle} {Tail Height} {Width}"));
  CHECK_EQ(graphOf(ocl, "512:8:1K:212", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues} {Middle} {Tail Height} {Width}"));
  // MODM31 in Arith keeps Tail in its cluster on a GF31 shape.
  CHECK_EQ(graphOf(ocl, "1:512:4:512:202", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues Arith} {Middle} {Tail Width Height}"));
  CHECK_EQ(graphOf(nvidia(806, true), "1:512:4:512:202", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues Arith Cuda} {Middle} {Tail Width Height}"));
  // Where OLD_FENCE is the only Queues key, Queues touches carryFused alone and joins Width.
  ClusterGraph const noAsm = clusterGraph(nvidia(806, true, true), FFTConfig{"512:15:512:101"}, {{"INPLACE", "0"}});
  CHECK_EQ(describe(noAsm), string("top{Placement Memory Cuda} {Middle} {Queues Tail Width Height}"));
  CHECK_EQ(clusterPictureMismatch(noAsm), string(""));
  ClusterGraph const noAsmSplit =
    clusterGraph(nvidia(806, true, true), FFTConfig{"512:15:512:212"}, {{"INPLACE", "0"}});
  CHECK_EQ(describe(noAsmSplit), string("top{Placement Memory Cuda} {Middle} {Queues Width} {Tail Height}"));
  CHECK_EQ(clusterPictureMismatch(noAsmSplit), string(""));
  CHECK_EQ(graphOf(amd(), "512:15:512:101", {{"INPLACE", "0"}}),
           string("top{Placement Memory Queues} {Middle} {Tail Width Height}"));
}

TEST(cluster_picture_check_catches_departures) {
  ClusterGraph const good = clusterGraph(nvidia(), FFTConfig{"512:15:512:101"}, {{"INPLACE", "0"}});
  CHECK_EQ(describe(good), string("top{Placement Memory Queues} {Middle} {Tail Width Height}"));
  CHECK_EQ(clusterPictureMismatch(good), string(""));

  ClusterGraph widthAlone = good;  // allowed: Width on its own
  widthAlone.clusters[1] = {Group::Tail, Group::Height};
  widthAlone.clusters.push_back({Group::Width});
  CHECK_EQ(clusterPictureMismatch(widthAlone), string(""));

  ClusterGraph merged = good;  // Memory merged into a cluster
  merged.clusters[1].push_back(Group::Memory);
  CHECK(!clusterPictureMismatch(merged).empty());

  ClusterGraph split = good;  // Height split from Tail
  split.clusters[1] = {Group::Tail, Group::Width};
  split.clusters.push_back({Group::Height});
  CHECK(!clusterPictureMismatch(split).empty());

  ClusterGraph blob = good;  // the middle kernels joined to another pass
  blob.clusters = {{Group::Middle, Group::Tail, Group::Height}, {Group::Width}};
  CHECK(!clusterPictureMismatch(blob).empty());

  ClusterGraph middleOnTop = good;
  middleOnTop.topTier.push_back(Group::Middle);
  middleOnTop.clusters.erase(middleOnTop.clusters.begin());
  CHECK(!clusterPictureMismatch(middleOnTop).empty());

  ClusterGraph queuesAway = widthAlone;  // OLD_FENCE's Queues in a cluster without the width pass
  queuesAway.touches[Group::Queues] = KG_CARRY;
  queuesAway.topTier = {Group::Placement, Group::Memory};
  queuesAway.clusters[1].insert(queuesAway.clusters[1].begin(), Group::Queues);
  CHECK(!clusterPictureMismatch(queuesAway).empty());

  ClusterGraph queues = good;  // Queues in a cluster while it touches global
  queues.topTier = {Group::Placement, Group::Memory};
  queues.clusters[1].insert(queues.clusters[1].begin(), Group::Queues);
  CHECK(!clusterPictureMismatch(queues).empty());

  ClusterGraph wide = good;  // more than MAX_PERMUTE groups
  wide.touches[Group::Queues] = KG_CARRY;
  wide.topTier = {Group::Placement, Group::Memory};
  wide.clusters[1] = {Group::Queues, Group::Tail, Group::Width, Group::Height, Group::Height};
  CHECK(!clusterPictureMismatch(wide).empty());
}

TEST(access_classes) {
  const vector<AccessClass>& classes = accessClasses();
  CHECK_EQ(classes.size(), size_t(5));
  const AccessClass& data = classes[0];
  const AccessClass& shuttle = classes[1];
  const AccessClass& freq = classes[2];

  auto str = [](const vector<int>& v) {
    string s;
    for (int x : v) { s += to_string(x); }
    return s;
  };
  auto strp = [](const vector<pair<int, int>>& v) {
    string s;
    for (auto [l, st] : v) { s += to_string(l) + to_string(st) + " "; }
    return s;
  };

  CHECK_EQ(str(usableLoadModes(nvidia(806), data)), string("02345"));
  CHECK_EQ(str(usableStoreModes(nvidia(806), data)), string("023"));
  CHECK_EQ(strp(usablePairs(nvidia(806), shuttle)), string("00 42 50 "));
  CHECK_EQ(str(usableLoadModes(nvidia(806), freq)), string("05"));

  CHECK_EQ(str(usableLoadModes(nvidia(300), data)), string("0234"));
  CHECK_EQ(strp(usablePairs(nvidia(300), shuttle)), string("00 42 "));
  CHECK_EQ(str(usableLoadModes(nvidia(300), freq)), string("0"));

  CHECK_EQ(str(usableLoadModes(amd(), data)), string("01"));
  CHECK_EQ(str(usableStoreModes(amd(), data)), string("01"));
  CHECK_EQ(strp(usablePairs(amd(), shuttle)), string("00 11 "));

  CHECK_EQ(str(usableLoadModes(nvidia(806, false, true), data)), string("0"));
  CHECK_EQ(strp(usablePairs(nvidia(806, true, true), shuttle)), string("00 "));

  CHECK_EQ(getDigit(12'345, 0), 5u);
  CHECK_EQ(getDigit(12'345, 3), 2u);
  CHECK_EQ(setDigit(12'345, 2, 9), 12'945u);
  CHECK_EQ(setDigit(0, 1, 4), 40u);
  CHECK_EQ(setDigit(40, 1, 0), 0u);
}

// The kernel names kernelGroupOf() knows must be the ones Gpu.h declares: every kernel either maps to
// a group or is listed here as not part of an iteration, so a kernel upstream adds fails this test
// until it is classified.  Run from the repository root, as "make check" does.
TEST(kernel_groups_cover_gpu_h) {
  std::ifstream in{"src/Gpu.h"};
  CHECK(in.good());
  const set<string> notIteration{"transpIn", "transpOut", "readResidue", "kernIsEqual", "sum64",   "testTrig",
                                 "testFFT4", "testFFT14", "testFFT15",   "testFFT",     "testTime"};
  std::regex const decl{R"(^\s*Kernel\s+([^;]+);)"};
  std::regex const ident{R"([A-Za-z_][A-Za-z0-9_]*)"};
  set<string> declared;
  for (string line; std::getline(in, line);) {
    std::smatch m;
    if (!std::regex_search(line, m, decl)) { continue; }
    string const list = m[1];
    for (std::sregex_iterator it{list.begin(), list.end(), ident}, end; it != end; ++it) { declared.insert(it->str()); }
  }
  CHECK(declared.size() > 30);
  for (const string& name : declared) {
    if (notIteration.contains(name)) {
      CHECK_EQ(kernelGroupOf(name), 0u);
    } else if (kernelGroupOf(name) == 0) {
      testing::fail(__FILE__, __LINE__, "Gpu.h kernel " + name + " belongs to no kernel group");
    }
  }
  for (const string& name : notIteration) {
    if (!declared.contains(name)) { testing::fail(__FILE__, __LINE__, name + " is no longer declared in Gpu.h"); }
  }
  CHECK_EQ(kernelGroupOf("kCarryFusedLL"), u32(KG_CARRY | KG_WIDTH));
  CHECK_EQ(kernelGroupOf("ktailMulLowGF61"), u32(KG_TAIL | KG_HEIGHT));
  CHECK_EQ(kernelGroupOf("kCarryMROE"), u32(KG_CARRY));
}
