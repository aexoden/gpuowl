// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <filesystem>

inline fs::path operator+(fs::path p, const std::string& tail) {
  p += tail;
  return p;
}

u64 fileSize(const fs::path& path);

void fancyRename(const fs::path& src, const fs::path& dst);

bool deleteLine(const fs::path& path, const string& targetLine, u64 initialSize = 0);

// Where a failure's evidence goes -- the failure log, a failed build's log, the source a compiler refused -- so that none
// of it clutters the run directory: failures/ in the working directory.
inline constexpr const char* FAILURES_DIR = "failures";

// A new file in FAILURES_DIR for `name` (the directory made on first use), its name prefixed with the local time and made
// unique, so that no failure's evidence overwrites another's.
fs::path failurePath(const string& name);
