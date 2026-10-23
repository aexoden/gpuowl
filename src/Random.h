// Copyright (C) Jason Lynch

// Independent random streams derived from one seed.  Each of the tuner's random choices draws from a stream of its own,
// keyed by what the choice is for and what it is about, so that no two of them share bits by accident: a coin tossed
// on one bit of a seed whose other uses fix that bit always lands the same way.

#pragma once

#include "common.h"

#include <string_view>

namespace tune {

// SplitMix64: a bijection whose every output bit depends on every input bit.
[[nodiscard]] constexpr u64 splitmix(u64 x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// 64-bit FNV-1a.
[[nodiscard]] constexpr u64 fnv1a(std::string_view text) {
  u64 h = 0xcbf29ce484222325ull;
  for (unsigned char c : text) { h = (h ^ c) * 0x100000001b3ull; }
  return h;
}

// The seed of the stream `purpose` under `seed`, for the item `salt` names (a subject's hash, an exponent, an index).
[[nodiscard]] constexpr u64 streamSeed(u64 seed, std::string_view purpose, u64 salt = 0) {
  return splitmix(splitmix(splitmix(seed) ^ fnv1a(purpose)) ^ salt);
}

}  // namespace tune
