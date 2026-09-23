// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"

#include <filesystem>
#include <optional>
#include <vector>

class Task;
class Args;

class Worktodo {
public:
  static std::optional<Task> getTask(Args &args, i32 instance);
  static bool deleteTask(const Task &task, i32 instance);

  // Every task the named files hold, in the order they are read. For a caller that wants the whole of the pending
  // work rather than the next piece of it; a missing file contributes nothing, as does a line that does not parse, but
  // a file that is there and cannot be read throws, as it does for getTask().
  static std::vector<Task> pending(const std::vector<fs::path>& files);
};
