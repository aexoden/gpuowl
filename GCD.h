#pragma once

#include "common.h"
#include "timeutil.h"

#include <chrono>
#include <future>
#include <string>
#include <vector>

class GCD {
  std::future<std::string> gcdFuture;
  Timer timer;
  u32 E;

public:
  void start(u32 E, const std::vector<u32> &bits, u32 sub);
  bool isOngoing() { return gcdFuture.valid(); }
  bool isReady() {
    return isOngoing() &&
           gcdFuture.wait_for(std::chrono::steady_clock::duration::zero()) ==
               std::future_status::ready;
  }
  std::string get();
  void wait() { gcdFuture.wait(); }
};
