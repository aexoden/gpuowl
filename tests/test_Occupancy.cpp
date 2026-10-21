// Copyright (C) Jason Lynch

// Tests that the occupancy model offers each register limit what upstream's register tuning would time: the
// occupancies around the compiler's own and the register counts at them under CUDA, more waves on AMD.

#include "Occupancy.h"

#include "test.h"

#include <vector>

using namespace tune;

namespace {

// A Tesla P100: 64K registers and 2048 threads an SM, 64 KB of shared memory.
OccupancyLimits p100() {
  return {.regsPerUnit = 65'536,
          .regsPerBlock = 65'536,
          .maxThreads = 2048,
          .maxBlocks = 32,
          .sharedPerUnit = 65'536,
          .reservedShared = 0};
}

// A Radeon VII (gfx906): 256 VGPRs a lane in units of 4, up to 10 waves a SIMD, four 64-lane SIMDs and 64 KB of LDS a
// CU.
OccupancyLimits gfx906() {
  return {.amd = true,
          .regsPerUnit = 256,
          .maxRegs = 256,
          .granule = 4,
          .maxBlocks = 10,
          .sharedPerUnit = 65'536,
          .simdPerCU = 4,
          .waveSize = 64};
}

}  // namespace

// 96 registers for 256 threads is 3072 a warp, 24576 a block: two blocks an SM, eight at most for the threads.
TEST(cuda_occupancy_is_the_blocks_the_register_file_holds) {
  Occupancy const occ = occupancyOf({.regs = 96, .localBytes = 0, .sharedBytes = 0, .threads = 256}, p100());
  CHECK_EQ(occ.k0, 2);
  CHECK_EQ(occ.maxBlocks, 8);
  CHECK_EQ(occ.regsFor(1), 255);
  CHECK_EQ(occ.regsFor(2), 128);
  CHECK_EQ(occ.regsFor(3), 80);
  CHECK_EQ(occ.regsFor(4), 64);
  CHECK(!occ.canTry(1));  // never one block below the compiler's but where that is its own
  CHECK(occ.canTry(2) && occ.canTry(4) && !occ.canTry(9));

  // Launch bounds of 2, 3 and 4 blocks, and the register count at each.
  CHECK(registerPlan({.regs = 96, .localBytes = 0, .sharedBytes = 0, .threads = 256}, p100(), false).values() ==
        (std::vector<int>{2, 3, 4, 64, 80, 128}));
  RegisterPlan const plan =
    registerPlan({.regs = 96, .localBytes = 0, .sharedBytes = 0, .threads = 256}, p100(), false);
  CHECK_EQ(plan.k0, 2);
  CHECK(plan.occupancies == (std::vector<std::pair<int, int>>{{2, 128}, {3, 80}, {4, 64}}));
  CHECK(!plan.lift);

  // Shared memory caps the blocks as the registers do: 20 KB a block leaves room for three.
  Occupancy const shared = occupancyOf({.regs = 32, .localBytes = 0, .sharedBytes = 20'480, .threads = 256}, p100());
  CHECK_EQ(shared.maxBlocks, 3);
  CHECK_EQ(shared.k0, 3);
  CHECK(registerPlan({.regs = 32, .localBytes = 0, .sharedBytes = 20'480, .threads = 256}, p100(), false).values() ==
        (std::vector<int>{2, 3, 80, 128}));
}

// 133 VGPRs is one wave a SIMD; 16 KB of LDS a workgroup of four waves fits four workgroups a CU, four waves a SIMD.
TEST(amd_occupancy_is_the_waves_the_vgprs_and_the_lds_allow) {
  KernelResources const middle{.regs = 133, .localBytes = 0, .sharedBytes = 16'384, .threads = 256};
  Occupancy const occ = occupancyOf(middle, gfx906());
  CHECK_EQ(occ.k0, 1);
  CHECK_EQ(occ.maxBlocks, 4);
  CHECK_EQ(occ.regsFor(2), 128);
  CHECK_EQ(occ.regsFor(3), 84);
  CHECK(!occ.canTry(1));  // a minimum of waves cannot lower the occupancy

  // Only more waves, and -1 where the host caps the kernel of its own accord.
  CHECK(registerPlan(middle, gfx906(), false).values() == (std::vector<int>{2, 3}));
  CHECK(registerPlan(middle, gfx906(), true).values() == (std::vector<int>{-1, 2, 3}));
  RegisterPlan const plan = registerPlan(middle, gfx906(), true);
  CHECK_EQ(plan.k0, 1);
  CHECK(plan.occupancies == (std::vector<std::pair<int, int>>{{2, 0}, {3, 0}}));
  CHECK(plan.lift);
}

TEST(a_kernel_whose_resources_say_nothing_is_offered_nothing) {
  CHECK(registerPlan({.regs = -1, .localBytes = 0, .sharedBytes = 0, .threads = 256}, p100(), false).values().empty());
  CHECK(registerPlan({.regs = 64, .localBytes = 0, .sharedBytes = 0, .threads = 0}, p100(), false).values().empty());
  CHECK(
    registerPlan({.regs = 64, .localBytes = 0, .sharedBytes = 0, .threads = 64}, OccupancyLimits{.amd = true}, false)
      .values()
      .empty());
}
