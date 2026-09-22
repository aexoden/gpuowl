// Copyright (C) Jason Lynch

#include "GpuFault.h"

#include "clwrap.h"
#include "log.h"
#include "Restart.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace gpufault {

namespace {

std::mutex attemptMutex;
std::string attempt;      // guarded by attemptMutex
int faultExitStatus = 1;  // set by arm(), read by the handler

#ifndef _WIN32

// The log file of the thread that armed this, borrowed. log() writes to a thread-local handle, and the handler runs on
// a thread of the runtime's own, which has none -- so without this the fault report reaches the terminal and never the
// log file, which is the one a user keeps. Set by arm(), which runs on the thread that opened the log.
LogLink faultLog;

// Mirrored from ROCm's <hsa/hsa_ext_amd.h> rather than included from it.
constexpr int STATUS_SUCCESS = 0;
constexpr int STATUS_ERROR = 0x1000;
constexpr int MEMORY_FAULT_EVENT = 0;

struct Agent {
  uint64_t handle;
};

struct MemoryFaultInfo {
  Agent agent;
  uint64_t virtualAddress;
  uint32_t faultReasonMask;
};

struct Event {
  int type;
  union {
    MemoryFaultInfo memoryFault;
  };
};

using EventHandler = int (*)(const Event*, void*);
using RegisterHandler = int (*)(EventHandler, void*);

std::string currentAttempt() {
  std::lock_guard const lock{attemptMutex};
  return attempt;
}

// Reached from a thread of the runtime's own while the thread that dispatched the work is still inside a driver wait
// that will never return. That thread holds no lock this needs -- it is blocked in the driver, not in the log or the
// allocator -- so ordinary code is safe here; what is not safe is returning, which is why every path ends the process.
int onEvent(const Event* event, void* /*data*/) {
  // Adopt the log of the thread that armed this, for as long as this call lasts -- which is to say until the process
  // ends, since every path below ends it.
  LogLinkScope const logScope{faultLog};

  // Only the memory fault is claimed. A hardware exception and a memory error arrive the same way, but neither has
  // been measured, and handing them back leaves whatever the runtime does about them today unchanged.
  if (!event || event->type != MEMORY_FAULT_EVENT) { return STATUS_ERROR; }

  // Two devices can fault at once, and the handler is called for each. The first one here ends the process; a second
  // has nothing left to do but wait for that to happen.
  static std::atomic_flag handling = ATOMIC_FLAG_INIT;
  if (handling.test_and_set()) { return STATUS_SUCCESS; }

  char address[32] = {};
  snprintf(address, sizeof address, "0x%016llx", (unsigned long long)event->memoryFault.virtualAddress);
  string const why =
    "a memory fault at "s + address + " (" + detail::faultReasonText(event->memoryFault.faultReasonMask) + ")";
  markContextLost(why.c_str());

  string const what = currentAttempt();
  if (what.empty()) {
    // Nothing was declared, so nothing would keep a new image from doing exactly what this one just did.
    log("Nothing was being measured, so there is no record that would keep a restart from repeating it.\n");
  } else {
    log("It was lost during:\n    %s\nThat attempt is recorded and will not be made again on this device.\n\n",
        what.c_str());
    log("The device was lost; restarting to recover it.\n");
    string const reason = restart::reexec();  // returns only if the restart could not be attempted
    log("Could not restart: %s\n", reason.c_str());
  }

  // _exit, not exit: the static destructors still to run would release objects belonging to a device that is gone,
  // and the point of being here at all is that those calls do not come back.
  fflush(nullptr);
  _exit(faultExitStatus);
}

#endif  // _WIN32

}  // namespace

namespace detail {

std::string faultReasonText(u32 mask) {
  // ROCm's hsa_amd_memory_fault_reason_t, bit for bit. The two ECC failures are distinct and a card whose name carries
  // 'sramecc+' can report either, so neither may stand for the other.
  static constexpr std::pair<u32, const char*> NAMES[] = {
    {1u << 0, "page not present"},  // or a supervisor-privilege violation, which shares the flag
    {1u << 1, "write to read-only memory"},
    {1u << 2, "execute of non-executable memory"},
    {1u << 3, "host memory only"},
    {1u << 4, "uncorrectable DRAM ECC error"},
    {1u << 5, "imprecise"},
    {1u << 6, "uncorrectable SRAM ECC error, in registers rather than at an address"},
    {1u << 31, "the device hung"},
  };

  string out;
  for (auto const& [bit, name] : NAMES) {
    if (mask & bit) {
      if (!out.empty()) { out += ", "; }
      out += name;
    }
  }
  if (out.empty()) {
    char hex[16] = {};
    snprintf(hex, sizeof hex, "0x%x", mask);
    out = "reason "s + hex;
  }
  return out;
}

}  // namespace detail

void arm([[maybe_unused]] cl_device_id device, [[maybe_unused]] int exitStatus) {
#ifndef _WIN32
  if (!isAmdGpu(device)) { return; }

  // RTLD_NOLOAD takes the handle only if the runtime is already in this process, which it is exactly when the device
  // chosen is reached through it. An AMD card driven by something else is left as it was rather than having ROCm
  // loaded underneath it. The handle is deliberately never released: the handler outlives every caller.
  void* const rocr = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!rocr) { return; }

  auto const registerHandler = reinterpret_cast<RegisterHandler>(dlsym(rocr, "hsa_amd_register_system_event_handler"));
  if (!registerHandler) { return; }

  faultExitStatus = exitStatus;
  faultLog = logLink();
  if (int const status = registerHandler(onEvent, nullptr); status != STATUS_SUCCESS) {
    log("Could not take over what happens on a GPU fault (%d); a fault on this device will abort the process "
        "instead of restarting it.\n",
        status);
  }
#endif
}

std::string attempting(std::string what) {
  std::lock_guard const lock{attemptMutex};
  std::string previous = std::move(attempt);
  attempt = std::move(what);
  return previous;
}

}  // namespace gpufault
