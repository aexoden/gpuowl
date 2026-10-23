// Copyright (C) Jason Lynch

// The tuner's CPU work spread over the machine's threads, out of the way of anything else that wants them.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tune {

// The model's work yields to anything else wanting the CPU: another tuner's timed loop on the same machine busy-waits
// on its device, and a stall there is read as the configuration's cost.
inline void lowerPriority() {
#ifdef __linux__
  (void)setpriority(PRIO_PROCESS, pid_t(syscall(SYS_gettid)), 19);
#endif
}

// Runs f(0) .. f(n-1) across the machine's threads.
template<typename F> void parallelFor(size_t n, F&& f) {
  size_t const threads = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
  if (threads <= 1) {
    for (size_t i = 0; i < n; ++i) { f(i); }
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::jthread> pool;
  for (size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      lowerPriority();
      for (size_t i; (i = next++) < n;) { f(i); }
    });
  }
}

}  // namespace tune
