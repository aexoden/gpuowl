// Copyright (C) Jason Lynch

#include "Occupancy.h"

#include "Gpu.h"

#include <algorithm>
#include <set>

namespace tune {

int Occupancy::regsFor(int blocks) const {
  if (blocks <= 0) { return 0; }
  if (amd) { return std::min(maxRegs, regsPerUnit / blocks / granule * granule); }
  return std::min(maxRegs, regsPerUnit / (blocks * warps) / 256 * 256 / 32);
}

bool Occupancy::canTry(int blocks) const {
  int const most = std::min(maxBlocks, Gpu::REG_MAX_OCCUPANCY);
  if (amd) { return blocks > k0 && blocks <= most && regsFor(blocks) >= 32; }
  return blocks >= 1 && (blocks >= 2 || blocks == k0) && blocks <= most && regsFor(blocks) >= 24;
}

// Registers are allocated per warp in units of 256 under CUDA, by the granule per lane on AMD.
Occupancy occupancyOf(const KernelResources& r, const OccupancyLimits& l) {
  Occupancy o{.amd = l.amd};
  if (r.threads <= 0 || r.regs <= 0) { return o; }

  if (l.amd) {
    if (l.simdPerCU <= 0 || l.waveSize <= 0 || l.granule <= 0) { return o; }
    o.warps = (r.threads + l.waveSize - 1) / l.waveSize;
    o.maxRegs = l.maxRegs;
    o.regsPerUnit = l.regsPerUnit;
    o.granule = l.granule;
    o.maxBlocks = l.maxBlocks;
    // The workgroups whose LDS fits in a CU, as waves per SIMD: 7 workgroups of 2 waves over 4 SIMDs is 3.
    if (r.sharedBytes) { o.maxBlocks = std::min(o.maxBlocks, l.sharedPerUnit / r.sharedBytes * o.warps / l.simdPerCU); }
    o.k0 = std::min(l.regsPerUnit / ((r.regs + l.granule - 1) / l.granule * l.granule), l.maxBlocks);
    return o;
  }

  o.warps = (r.threads + 31) / 32;
  o.maxRegs = std::min(255, l.regsPerBlock / r.threads);
  o.regsPerUnit = l.regsPerUnit;
  o.maxBlocks = std::min(l.maxBlocks, l.maxThreads / r.threads);
  if (r.sharedBytes) { o.maxBlocks = std::min(o.maxBlocks, l.sharedPerUnit / (r.sharedBytes + l.reservedShared)); }
  o.k0 = std::min(l.regsPerUnit / ((r.regs * 32 + 255) / 256 * 256 * o.warps), o.maxBlocks);
  return o;
}

std::vector<int> RegisterPlan::values() const {
  std::set<int> out;
  for (auto const& [blocks, regs] : occupancies) {
    out.insert(blocks);
    if (regs > Gpu::REG_MAX_OCCUPANCY) { out.insert(regs); }
  }
  if (lift) { out.insert(-1); }
  return {out.begin(), out.end()};
}

RegisterPlan registerPlan(const KernelResources& resources, const OccupancyLimits& limits, bool defaultCap) {
  Occupancy const occ = occupancyOf(resources, limits);
  RegisterPlan out{.k0 = occ.k0, .occupancies = {}, .lift = limits.amd && defaultCap};
  if (occ.k0 <= 0) { return out; }
  for (int const blocks : {occ.k0 - 1, occ.k0, occ.k0 + 1, occ.k0 + 2}) {
    if (!occ.canTry(blocks)) { continue; }
    int const regs = occ.amd ? 0 : occ.regsFor(blocks);
    out.occupancies.emplace_back(blocks, regs > Gpu::REG_MAX_OCCUPANCY ? regs : 0);
  }
  return out;
}

std::optional<OccupancyLimits> deviceLimits([[maybe_unused]] cl_device_id id) {
#ifdef CUDA_BACKEND
  CudaSmLimits const sm = cudaSmLimits();
  return OccupancyLimits{.regsPerUnit = sm.regsPerSM,
                         .regsPerBlock = sm.regsPerBlock,
                         .maxThreads = sm.maxThreadsPerSM,
                         .maxBlocks = sm.maxBlocksPerSM,
                         .sharedPerUnit = sm.sharedPerSM,
                         .reservedShared = sm.reservedSharedPerBlock};
#else
  if (!isAmdGpu(id) || !amdHasKernelInfo(id)) { return {}; }
  AmdSimdLimits const l = amdSimdLimits(id);
  return OccupancyLimits{.amd = true,
                         .regsPerUnit = l.vgprs,
                         .maxRegs = l.maxRegs,
                         .granule = l.granule,
                         .maxBlocks = l.maxWaves,
                         .sharedPerUnit = l.ldsPerCU,
                         .simdPerCU = l.simdPerCU,
                         .waveSize = l.waveSize};
#endif
}

}  // namespace tune
