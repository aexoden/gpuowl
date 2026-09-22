// Copyright (C) Jason Lynch

// Error handling for GPU faults that aren't reported as errors (on ROCm).

#pragma once

#ifdef CUDA_BACKEND
#include "tinycuda.h"
#else
#include "tinycl.h"
#endif

#include "common.h"
#include "log.h"

#include <string>

namespace gpufault {

// Takes over the answer to a fault on this device, where the runtime behind it offers one. `exitStatus` is what the
// process ends with when a fault cannot be recovered from. Call once, after the device is chosen.
void arm(cl_device_id device, int exitStatus);

// Names what is being attempted, for the message a fault leaves behind. It also marks the window in which a restart
// can make progress: a fault restarts only while an attempt is declared, because the record of that attempt is what
// stops the next generation from repeating it. An empty string closes the window.
//
// Returns the declaration it replaced, so that a caller inside another one can put it back: clearing unconditionally
// would leave a fault during the outer attempt reporting that nothing was being measured, and declining to restart.
[[nodiscard]] std::string attempting(std::string what);

namespace detail {

// The fault reasons ROCm reports, as text. Pure; exposed for testing.
[[nodiscard]] std::string faultReasonText(u32 mask);

}  // namespace detail

}  // namespace gpufault