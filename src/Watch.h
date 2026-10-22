// Copyright (C) Jason Lynch

// What the tuning loop tells whoever is watching it: what it is measuring, each item it finishes, and where things
// stand after each publication.  The log gets its own lines either way.

#pragma once

#include "common.h"

#include <limits>
#include <string>
#include <vector>

namespace tune {

struct ItemReport {
  u32 n = 0;
  std::string kind{};  // "call", "second call", "gate"
  std::string what{};  // "ll 512:7:512:212"
  std::string options{};
  u64 exponent = 0;
  std::string result{};  // "505.4 us/it", "z 26.73", "nocompile"
  double measured = 0;   // microseconds per iteration; 0 where the item read none
  double predicted = 0;  // what the model said before, where it said anything
  std::string why{};
};

struct StateReport {
  double T = 0;  // the published lines' weighted cost, over the points they cover
  double coverage = 0;
  double regret = std::numeric_limits<double>::quiet_NaN();  // the latest estimate
  u32 calls = 0;
  u32 gates = 0;
  std::vector<std::string> lines{};  // what is published, one string per line, kind first
  u32 observations = 0;
  std::string spreads{};
  std::vector<std::string> faults{};  // configurations that failed or that a process died holding
};

class Watch {
public:
  virtual ~Watch() = default;
  virtual void measuring(const std::string& what) = 0;
  virtual void finished(const ItemReport& item) = 0;
  virtual void state(const StateReport& state) = 0;
};

}  // namespace tune
