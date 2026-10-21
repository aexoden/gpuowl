// Copyright (C) Jason Lynch

// Identifies the kernel build: a hash of every bundled kernel source.  Editing a kernel changes it, which is what tells
// a later run that the rows it is looking at were measured against different code.

#pragma once

#include "common.h"

#include <span>

namespace tune {

// The fingerprint of a set of (file name, file contents) pairs, in order.
[[nodiscard]] u64 fingerprintOf(std::span<const char* const> names, std::span<const char* const> sources);

// The fingerprint of the kernels this binary carries.
//
// Deliberately computable without a GPU, so that emission needs no device, and deliberately not the program version:
// that is `git describe --dirty`, so it moves on every commit and on the first edit to a clean tree, which would put
// every host-side change in an env of its own and strand every measurement taken before it.  It also leaves out the
// device-dependent compiler flags and the per-configuration defines, which the env row and the cfg row carry
// respectively.
//
// What that gives up is a host-side change that moves a timing without moving a kernel -- the block size, how a call is
// timed, what a cost is normalized by.  Those are rare and deliberate, so each one bumps MEASUREMENT_PROTOCOL, which
// the fingerprint takes in: rows timed the old way then sit in an env of their own rather than pooling with new ones.
[[nodiscard]] u64 buildFingerprint();

// 1 until calls heated the device first (CALL_HEAT_SEC).
inline constexpr u32 MEASUREMENT_PROTOCOL = 2;

}  // namespace tune
