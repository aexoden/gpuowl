// Copyright (C) Jason Lynch

#include "BuildId.h"

#include "Sha3Hash.h"

#include <algorithm>
#include <string>
#include <vector>

// Implemented in bundle.cpp
const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

namespace tune {

u64 fingerprintOf(std::span<const char* const> names, std::span<const char* const> sources) {
  SHA3 hasher;
  size_t const n = std::min(names.size(), sources.size());
  for (size_t i = 0; i < n; ++i) {
    // A separator after each field: without it "ab" + "c" and "a" + "bc" are one byte string, and a file renamed into
    // its neighbour's contents would keep the fingerprint it had.
    hasher.update(std::string{names[i]} + '\0');
    hasher.update(std::string{sources[i]} + '\0');
  }
  return std::move(hasher).finish()[0];
}

u64 buildFingerprint() { return fingerprintOf(getClFileNames(), getClFiles()); }

}  // namespace tune
