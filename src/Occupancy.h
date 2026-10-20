// Copyright (C) Jason Lynch

// The register limits worth timing for one kernel, from the occupancy its compiled resources give it: what the search
// offers each REG* key.  The model is upstream's register tuning (Tune::regTuneEntry): occupancy changes only where the
// registers a thread may use cross a boundary of the register file, so the candidates are the occupancies around the
// compiler's own and the register counts at their boundaries.  Pure but for deviceLimits(), so that a database read
// off the device answers the same.

#pragma once

#include "clwrap.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tune {

// What the search offers each REG* key of one entry beside the compiler's default: the occupancy model's candidates, by
// key.  A key it does not name is offered the default alone, whose reading records the kernel's resources.
using RegisterOffer = std::map<std::string, std::vector<int>>;

// What limits a kernel's occupancy on one device: per SM under CUDA, per SIMD on AMD.
struct OccupancyLimits {
  bool amd = false;
  int regsPerUnit = 0;     // CUDA: registers per SM; AMD: VGPRs per lane of a SIMD
  int regsPerBlock = 0;    // CUDA only
  int maxRegs = 0;         // AMD: most VGPRs per lane for one wave
  int granule = 0;         // AMD: VGPR allocation unit
  int maxThreads = 0;      // CUDA: threads per SM
  int maxBlocks = 0;       // CUDA: blocks per SM; AMD: waves per SIMD
  int sharedPerUnit = 0;   // CUDA: shared memory per SM; AMD: LDS per CU
  int reservedShared = 0;  // CUDA: shared memory the system takes per block
  int simdPerCU = 0;       // AMD
  int waveSize = 0;        // AMD
};

// A kernel's occupancy as its registers set it: blocks per SM under CUDA, waves per SIMD on AMD.
struct Occupancy {
  bool amd = false;
  int k0 = 0;         // with the compiler's own registers; 0 where the resources say nothing
  int maxBlocks = 0;  // what every limit but the registers allows
  int warps = 0;      // per block (CUDA)
  int maxRegs = 0;
  int regsPerUnit = 0;
  int granule = 0;

  // The most registers a thread may use at this occupancy.
  [[nodiscard]] int regsFor(int blocks) const;

  // Whether a minimum occupancy of `blocks` is worth a timing.  CUDA: from one below the compiler's, never one block
  // below it; AMD: only above it, since a minimum of waves cannot lower the occupancy.  Neither so high that a thread
  // keeps too few registers to be compiled sensibly.
  [[nodiscard]] bool canTry(int blocks) const;
};

[[nodiscard]] Occupancy occupancyOf(const KernelResources& resources, const OccupancyLimits& limits);

// The REG* values worth timing for a kernel built at the compiler's own registers, ascending, the default (0) not
// among them.  CUDA: launch bounds of one block below the compiler's occupancy to two above, and the register count at
// each of those.  AMD: one and two waves above, and -1 where `defaultCap`, the host's own cap of a kernel left at 0,
// is to be lifted.  Empty where the resources give no occupancy.
[[nodiscard]] std::vector<int> registerCandidates(const KernelResources& resources, const OccupancyLimits& limits,
                                                  bool defaultCap);

// The open device's limits, where its register limits can be modelled: CUDA, and AMD where the runtime reports a
// kernel's VGPRs.
[[nodiscard]] std::optional<OccupancyLimits> deviceLimits(cl_device_id id);

}  // namespace tune
