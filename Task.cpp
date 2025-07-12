#include "Task.h"

#include "Gpu.h"
#include "Result.h"
#include "args.h"

#include <cstdio>
#include <cmath>

std::vector<std::string> getDevices() {
  std::vector<std::string> ret;
  for (auto id : getDeviceIDs(false)) { ret.push_back(getLongInfo(id)); }
  return ret;
}

bool Task::execute(const Args &args) {
  assert(kind == PRP);
  auto gpu = Gpu::make(exponent, args);
  return gpu->isPrimePRP(exponent, args, B1, B2).write(args, *this, gpu->getFFTSize());
}
