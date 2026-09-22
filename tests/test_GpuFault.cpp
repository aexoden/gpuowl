// Copyright (C) Jason Lynch

// GPU-free tests of the part of a fault report that is a pure function of what the runtime hands over. Taking the
// notification over, and what is done with it, needs a device that faults.

#include "test.h"

#include "GpuFault.h"

#include <string>

using gpufault::detail::faultReasonText;

TEST(a_fault_reason_reads_as_words) {
  CHECK_EQ(faultReasonText(1u << 0), "page not present");
  CHECK_EQ(faultReasonText(1u << 1), "write to read-only memory");
  CHECK_EQ(faultReasonText(1u << 31), "the device hung");
}

TEST(a_fault_with_several_reasons_names_all_of_them) {
  // The mask a write past the end of a buffer produces on gfx906.
  CHECK_EQ(faultReasonText(0x3), "page not present, write to read-only memory");
  CHECK_EQ(faultReasonText((1u << 0) | (1u << 31)), "page not present, the device hung");
}

// The two ECC failures are separate flags in ROCm's own enum, and the card this was written for reports itself as
// 'gfx906:sramecc+': one standing for the other would name the wrong memory in the one report anybody would keep.
TEST(the_two_ecc_failures_are_told_apart) {
  CHECK_EQ(faultReasonText(1u << 4), "uncorrectable DRAM ECC error");
  CHECK_EQ(faultReasonText(1u << 6), "uncorrectable SRAM ECC error, in registers rather than at an address");
  CHECK(faultReasonText(1u << 6) != faultReasonText(1u << 4));
}

// Every bit the enum defines is named, so that a real fault never reads as a bare number.
TEST(no_defined_reason_falls_through_to_its_number) {
  for (u32 const bit : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 31u}) {
    CHECK(faultReasonText(1u << bit).find("reason 0x") == std::string::npos);
  }
}

TEST(a_fault_reason_nobody_has_seen_is_still_reported) {
  // Worth keeping legible: a mask this does not know is exactly the case where the number is the whole of the
  // evidence, and dropping it would leave a fault report that says nothing at all.
  CHECK_EQ(faultReasonText(0), "reason 0x0");
  CHECK_EQ(faultReasonText(1u << 20), "reason 0x100000");
}
