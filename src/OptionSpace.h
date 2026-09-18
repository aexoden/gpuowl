// Copyright (C) Jason Lynch

// The declarative inventory of -use options: which keys exist, which values are worth trying, when a key applies, when
// the kernels cannot read it, how widely one measured answer is expected to generalize, and which kernels it affects.

#pragma once

#include "common.h"
#include "FFTConfig.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

class Args;
class Context;

namespace tune {

// The maximum number of keys permuted in one bin.
constexpr u32 MAX_PERMUTE = 4;

// The LDS budget, in bytes, that clDefines() in Gpu.cpp clamps WMUL and LDSPAD_W against.
constexpr u32 LDS_BUDGET = 32'768;

// A -use assignment, KEY -> VALUE, as Args::flags holds one.
using UseConfig = std::map<std::string, std::string>;

// The integer value of `key` in `config`, or `valNotFound`.
[[nodiscard]] int useValue(const UseConfig& config, const string& key, int valNotFound);

// Facts about the machine that decide which options exist and what they default to.
struct Env {
  bool isAmd = false;
  bool isNvidia = false;
  bool cudaBackend = false;  // built with CUDA_BACKEND; independent of the GPU vendor
  bool noAsm = false;        // -use NO_ASM

  // What clDefines() passes as -DCC on NVIDIA, 0 elsewhere. base.cl makes this HAS_PTX.
  u32 computeCapability = 0;

  // The CUDA shim's launcher can ask for programmatic stream serialization, which it does for exactly those kernels
  // whose compiled code waits on the predecessor. It needs cuLaunchKernelEx, so a build against CUDA below 12.0 cannot
  // honor PDL however the kernels were compiled.
  bool pdlLaunch = false;

  string deviceName{};
  string driverVersion{};

  // Inline PTX guarded by "#if HAS_PTX >= n": HAS_PTX is CC on NVIDIA without NO_ASM, else 0.
  [[nodiscard]] bool hasPtx(u32 atLeast) const { return isNvidia && !noAsm && computeCapability >= atLeast; }

  // __builtin_nontemporal_load / _store (access mode 1). In practice, only AMD's compiler has them.
  [[nodiscard]] bool hasNontemporal() const { return isAmd; }

  // A short description, e.g. "nvidia,ocl,cc806"; used to say where a self-check failed.
  [[nodiscard]] string label() const;
};

// The environment this process is running in.
[[nodiscard]] Env detectEnv(const Context& context, const Args& args);

enum class Kind {
  Tunable,
  Fixed,
  Deprecated,
};

enum class Scope { Device, Family, Shape, Variant };

// Whether a key moves the usable bits per word.
enum class AccuracyImpact { None, Suspected, Yes };

// The kernels an option affects, as a bit mask.
enum KernelGroup : u32 {
  KG_WIDTH = 1u << 0,       // fftW, fftP, carryFused
  KG_HEIGHT = 1u << 1,      // fftHin, tailSquare/tailMul
  KG_MIDDLE_IN = 1u << 2,   // fftMiddleIn
  KG_MIDDLE_OUT = 1u << 3,  // fftMiddleOut
  KG_TAIL = 1u << 4,        // tailSquare, tailMul
  KG_CARRY = 1u << 5,       // carryFused, carryA, carryB
  KG_GLOBAL = 1u << 6,      // every kernel, or not separable
};

// Which kernel groups a Gpu kernel belongs to; 0 for anything that is not part of an iteration.
[[nodiscard]] u32 kernelGroupOf(const string& kernelName);

enum class Group {
  None,
  Placement,
  Middle,
  Memory,
  Queues,
  Tail,
  Width,
  Height,
  Arith,
  Cuda,
};

// Every group except None, in declaration order.
[[nodiscard]] const vector<Group>& allGroups();

[[nodiscard]] const char* toString(Group group);

using Predicate = bool (*)(const Env&, const FFTConfig&, const UseConfig&);
using ValuesFn = vector<int> (*)(const Env&, const FFTConfig&, const UseConfig&);
using DefaultFn = int (*)(const Env&, const FFTConfig&, const UseConfig&);
using TouchesFn = u32 (*)(const Env&, const FFTConfig&, const UseConfig&);

struct Option {
  string key;
  Kind kind = Kind::Tunable;
  Scope scope = Scope::Shape;
  Group group = Group::None;

  // The kernel groups the key affects: `touchesFn` if set, for a key whose reach depends on the FFT, else `touches`.
  u32 touches = KG_GLOBAL;
  TouchesFn touchesFn = nullptr;

  // Whether the option value determines which keys apply.
  bool structural = false;

  // Searched through accessClasses() rather than through values.
  bool compound = false;

  AccuracyImpact accuracyImpact = AccuracyImpact::None;

  // Every key that applies, touchesFn, valuesFn, defaultFn or inert reads. Must name real keys and be acyclic.
  vector<string> dependsOn{};

  // Whether the key is offered at all; nullptr means always.
  Predicate applies = nullptr;

  // The values worth trying, ascending. `valueList` unless `values` is set.
  vector<int> values{};
  ValuesFn valuesFn = nullptr;

  // The default value to use; `defaultValue` unless `defaultFn` is set.
  int defaultValue = 0;
  DefaultFn defaultFn = nullptr;

  // Offered by the rules above, but a no-op in the kernels; nullptr means never.
  Predicate inert = nullptr;
  const char* inertWhen = "";

  [[nodiscard]] bool appliesTo(const Env& e, const FFTConfig& f, const UseConfig& d) const {
    return !applies || applies(e, f, d);
  }

  [[nodiscard]] bool isInert(const Env& e, const FFTConfig& f, const UseConfig& d) const {
    return inert && inert(e, f, d);
  }

  [[nodiscard]] u32 touchesFor(const Env& e, const FFTConfig& f, const UseConfig& d) const {
    return touchesFn ? touchesFn(e, f, d) : touches;
  }

  [[nodiscard]] vector<int> valuesFor(const Env& e, const FFTConfig& f, const UseConfig& d) const {
    return valuesFn ? valuesFn(e, f, d) : values;
  }

  [[nodiscard]] int defaultFor(const Env& e, const FFTConfig& f, const UseConfig& d) const {
    return defaultFn ? defaultFn(e, f, d) : defaultValue;
  }
};

// The table, in declaration order.
[[nodiscard]] const vector<Option>& allOptions();

// The option with this key, or nullptr.
[[nodiscard]] const Option* findOption(const string& key);

// Every -use key this build understands, whatever its kind.
[[nodiscard]] bool isKnownKey(const string& key);

// The tunable keys worth searching for one FFT given the options decided so far: those that apply, are not inert, and
// offer more than one value (or are compound).
[[nodiscard]] vector<const Option*> applicableOptions(const Env& env, const FFTConfig& fft, const UseConfig& decided);

// The combo hierarchy for one (entry, structural branch). A group is present when at least one of its keys is
// applicable. A present group whose applicable keys touch KG_GLOBAL goes to the top tier; the rest are joined when
// their touches overlap, and each cluster is a connected component of that relation. Groups appear in declaration
// order.
struct ClusterGraph {
  vector<Group> topTier;
  vector<vector<Group>> clusters;
  std::map<Group, u32> touches;
};

[[nodiscard]] ClusterGraph clusterGraph(const Env& env, const FFTConfig& fft, const UseConfig& decided);

// Where `graph` departs from the expected cluster picture: empty when it matches, else a description. The picture is a
// top tier drawn from Placement, Memory, Queues, Arith and Cuda; Middle in a cluster of its own; Tail and Height in one
// cluster, which Width joins where a key couples the width pass to them and otherwise leaves; and Queues in the top
// tier, except that it joins Width's cluster where its only applicable key is OLD_FENCE, which touches carryFused
// alone.
[[nodiscard]] string clusterPictureMismatch(const ClusterGraph& graph);

// LOADS and STORES pack one access mode per class of memory traffic into their decimal digits.
struct AccessClass {
  string name;
  u32 digit;                       // 0 = ones, 1 = tens, ...
  vector<int> loadModes{};         // empty: no load side
  vector<int> storeModes{};        // empty: no store side
  vector<pair<int, int>> pairs{};  // non-empty: load and store are chosen together, as these pairs
};

[[nodiscard]] const vector<AccessClass>& accessClasses();

// Whether this machine's compiler can emit an access mode.
[[nodiscard]] bool loadModeExists(const Env& env, int mode);
[[nodiscard]] bool storeModeExists(const Env& env, int mode);

// The modes that exist on `env`.
[[nodiscard]] vector<int> usableLoadModes(const Env& env, const AccessClass& cls);
[[nodiscard]] vector<int> usableStoreModes(const Env& env, const AccessClass& cls);
[[nodiscard]] vector<pair<int, int>> usablePairs(const Env& env, const AccessClass& cls);

// Read and write one decimal digit of a digit-packed value.
[[nodiscard]] u32 getDigit(u32 packed, u32 digit);
[[nodiscard]] u32 setDigit(u32 packed, u32 digit, u32 value);

struct MatrixPoint {
  Env env;
  FFTConfig fft;
  UseConfig decided;
  string label{};
};

[[nodiscard]] vector<MatrixPoint> selfCheckMatrix();

// Checks the table against itself and at every point of selfChecKMatrix(). Logs each problem and returns how many were
// found.
[[nodiscard]] u32 selfCheck();

u32 dumpOptionSpace(const Env& env, const FFTConfig& fft);

}  // namespace tune
