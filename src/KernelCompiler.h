// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>
#include <future>
#include <map>
#include <set>

class Args;
class Context;

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  std::string linkArgs;
  std::string baseArgs;
  const bool useCache;
  const bool useMemCache;   // -tune, -oldtune, -measure, -ztune, -ctune, -carryTune: reuse this process's compiles (see memCache in KernelCompiler.cpp)
  const int verbose;
  const bool saveTemps; // The AMD OpenCL runtime writes -save-temps files into the current directory (AMD_OCL_BUILD_OPTIONS_APPEND or
                        // AMD_OCL_LINK_OPTIONS_APPEND has -save-temps, set by -v 10 or by the user): they are deleted after each compile.
  const bool asmDump;   // -v 10 or -v 11: pick up the compiler's assembly (AMD OpenCL). -v 10 reports only
                        // the "important" kernels (see isImportantKernel in KernelCompiler.cpp); -v 11 reports all of them.

  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};

  // -v 10 assembly dump: some kernelNames (e.g. "carryFused", "carry") are compiled several times
  // under this same name with different defines (plain/-DROE=1/-DMUL3=1/...). The args of every
  // kernel declared under each kernelName, to tell those apart by the defines that differ; and the
  // .s names used so far.
  std::map<std::string, std::vector<std::string>> declaredArgs;
  mutable std::set<std::string> asmDumpNames;
  mutable bool asmMissingNoted = false;
  mutable bool asmCacheNoted = false;
  int maxWaves = 0;   // waves per SIMD, AMD; 0 when not known
  // For the -v 10 LDS-occupancy check (AMD only; 0 when not known/available):
  int simdPerCU = 0;        // SIMDs per compute unit
  int wavefrontWidth = 0;   // threads per wavefront
  u64 ldsPerCU = 0;         // LDS bytes per compute unit (a workgroup's LDS is shared by the whole CU)

  // load=false: on CUDA only the compile, NVRTC's PTX and CUBIN with no module loaded (OpenCL builds and links either way)
  [[nodiscard]] Program build(const string& fileName, const string& args, bool load = true) const;
  [[nodiscard]] Program compile(const string& fileName, const string& kernelName, const string& args, bool load = true) const;
  [[nodiscard]] KernelHolder loadAux(const string& fileName, const string& kernelName, const string& args) const;
  [[nodiscard]] bool compileAheadAux(const string& fileName, const string& kernelName, const string& args) const;

public:
  const cl_device_id deviceId;

  KernelCompiler(const Args& args, const Context* context, const string& clArgs);

  ~KernelCompiler();
  
  // Called by each Kernel on construction, before any is loaded (see declaredArgs).
  void declare(const string& kernelName, const string& args);

  [[nodiscard]] std::future<KernelHolder> load(const string& fileName, const string& kernelName, const string& args) const;

  // Compiles the kernel into this process's in-memory cache, unless it is there already, so that load() later finds it there:
  // without loading anything into the context or running anything on the device.  On CUDA that is NVRTC alone, and a register
  // limit is applied by the driver's linker rather than at the load.  A tuner compiles a Gpu's kernels so while the card still
  // squares another (Bench::compileAhead).  Concurrent on CUDA, as load() is; done before returning on OpenCL.  True where the
  // kernel is ready; false where it could not be compiled, or where no in-memory cache is kept (only while tuning).
  [[nodiscard]] std::future<bool> compileAhead(const string& fileName, const string& kernelName, const string& args) const;

  // What this process's kernel loads and compiles ahead have taken, for a tuner to say where a call's build went.
  struct Tally {
    u32 compiled = 0;     // loads that compiled (NVRTC, and on CUDA the module load)
    u32 fromCache = 0;    // loads of a binary from memory or from -cache
    u32 limited = 0;      // CUDA: register limits applied at the load (a JIT of the PTX)
    u32 ahead = 0;        // kernels compiled ahead (compileAhead), register limits included
    double compiledSec = 0, fromCacheSec = 0, limitedSec = 0, aheadSec = 0;

    [[nodiscard]] Tally operator-(const Tally& o) const {
      return {compiled - o.compiled, fromCache - o.fromCache, limited - o.limited, ahead - o.ahead,
              compiledSec - o.compiledSec, fromCacheSec - o.fromCacheSec, limitedSec - o.limitedSec, aheadSec - o.aheadSec};
    }
  };
  [[nodiscard]] static Tally tally();
};
