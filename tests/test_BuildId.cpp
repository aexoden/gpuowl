// Copyright (C) Jason Lynch

// GPU-free tests of the kernel build fingerprint (src/BuildId.cpp).

#include "test.h"

#include "BuildId.h"

#include <array>
#include <string>
#include <vector>

using namespace tune;

namespace {

u64 hashOf(const std::vector<const char*>& names, const std::vector<const char*>& sources) {
  return fingerprintOf(names, sources);
}

}  // namespace

TEST(fingerprint_is_stable_and_nonzero) {
  CHECK(buildFingerprint() != 0);
  CHECK_EQ(buildFingerprint(), buildFingerprint());
  CHECK_EQ(hashOf({"a.cl", "b.cl"}, {"one", "two"}), hashOf({"a.cl", "b.cl"}, {"one", "two"}));
}

TEST(fingerprint_follows_the_sources) {
  u64 const base = hashOf({"a.cl", "b.cl"}, {"one", "two"});

  CHECK(hashOf({"a.cl", "b.cl"}, {"one", "TWO"}) != base);              // an edit to a kernel
  CHECK(hashOf({"a.cl", "c.cl"}, {"one", "two"}) != base);              // a rename
  CHECK(hashOf({"b.cl", "a.cl"}, {"two", "one"}) != base);              // a reordering
  CHECK(hashOf({"a.cl", "b.cl", "c.cl"}, {"one", "two", ""}) != base);  // a new, empty file
  CHECK(hashOf({"a.cl"}, {"one"}) != base);                             // a deletion
}

TEST(fingerprint_separates_the_fields) {
  // Without a separator between a name and what follows it, moving a character across the boundary would leave the
  // byte string -- and so the fingerprint -- unchanged.
  CHECK(hashOf({"ab"}, {"c"}) != hashOf({"a"}, {"bc"}));
  CHECK(hashOf({"a.cl", "b.cl"}, {"", "x"}) != hashOf({"a.cl", "b.cl"}, {"x", ""}));
}

TEST(fingerprint_ignores_a_source_without_a_name) {
  // The bundle provides the two lists in step; a short one is a bug elsewhere, and hashing past the end of either is
  // worse than ignoring the tail.
  CHECK_EQ(hashOf({"a.cl"}, {"one", "two"}), hashOf({"a.cl"}, {"one"}));
}
