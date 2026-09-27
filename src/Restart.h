// Copyright (C) Jason Lynch

// Recovers the application by attempting to restart it after a GPU fault.

#pragma once

#include "common.h"

#include <functional>

namespace restart {

// The generation count a parent passes to its child.
[[nodiscard]] u32 parseCarry(const char* text);

// Initializes the restart mechanism with the command-line arguments.
void init(int argc, char** argv);

// Returns the number of times the application has been restarted.
[[nodiscard]] u32 generation();

// The cap on restarts, from PRPLL_MAX_RESTARTS.
[[nodiscard]] u32 maxRestarts();

// What reexec() does first, on whichever thread calls it and whether or not it goes on to restart, or nullptr for
// nothing: something the image is showing that the next one, or the shell once this one ends, would otherwise inherit
// half-drawn.
void beforeExec(std::function<void()> f);

// Replaces the application with a fresh one running the same command.
[[nodiscard]] string reexec();

}  // namespace restart
