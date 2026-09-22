// Copyright (C) Jason Lynch

// Recovers the application by attempting to restart it after a GPU fault.

#pragma once

#include "common.h"

namespace restart {

// The generation count a parent passes to its child.
[[nodiscard]] u32 parseCarry(const char* text);

// Initializes the restart mechanism with the command-line arguments.
void init(int argc, char** argv);

// Returns the number of times the application has been restarted.
[[nodiscard]] u32 generation();

// The cap on restarts, from PRPLL_MAX_RESTARTS.
[[nodiscard]] u32 maxRestarts();

// Replaces the application with a fresh one running the same command.
[[nodiscard]] string reexec();

}  // namespace restart
