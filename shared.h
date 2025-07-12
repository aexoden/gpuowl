// included from both C++ and OpenCL.

#pragma once

#ifdef __OPENCL_VERSION__
typedef uint u32;
typedef ulong u64;
#else
#include "common.h"
#endif

inline u32 bitposToWord(u32 E, u32 N, u32 offset) { return offset * ((u64) N) / E; }
inline u32 wordToBitpos(u32 E, u32 N, u32 word) { return (word * ((u64) E) + (N - 1)) / N; }
