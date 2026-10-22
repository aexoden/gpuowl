// Copyright (C) Jason Lynch

// The dimensions the tuner searches an FFT's options along.  An axis is one key, or for LOADS and STORES one access
// class -- one digit of one of them, or of both where the class chooses its load and store together -- so a search
// never moves the packed integer as a whole, and never offers a mode the compiler would quietly build as another.

#pragma once

#include "common.h"
#include "FFTConfig.h"
#include "Occupancy.h"
#include "OptionSpace.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tune {

// Which FFTs are offered TAIL_KERNELS 11 and 13, the tail*Zero kernels in a queue of their own.  Upstream measured them
// faster only for NTTs on nVidia (2-7% on a TITAN V); FP64 lost 3% there and a Radeon VII 60-70%.
enum class TailQueues : u8 { Ntt, All, None };

[[nodiscard]] const char* toString(TailQueues queues);
[[nodiscard]] std::optional<TailQueues> parseTailQueues(std::string_view text);

struct Axis {
  const Option* option = nullptr;  // LOADS for a coupled class

  // An access class: the digit it occupies, and whether STORES moves with LOADS.
  std::optional<u32> digit{};
  bool coupled = false;

  // The positions the axis can take, ascending: a key's value, a class's mode, or a coupled class's (load, store).
  std::vector<std::pair<int, int>> values{};
  size_t current = 0;

  // Where the axis is when nothing sets it: the key's default for this FFT and device, or mode 0 for a class.
  size_t reference = 0;

  // "WMUL", "LOADS[trig, used once]"
  std::string name{};

  // The keys it writes.
  [[nodiscard]] std::vector<std::string> keys() const;

  // "WMUL=2", "LOADS[trig, used once]=5", "LOADS+STORES[carry]=4/2"
  [[nodiscard]] std::string label(size_t index) const;
};

// What a search offers of the table's values: TAIL_KERNELS 11 and 13 where `tailQueues` says, and for a REG* key the
// compiler's default and the occupancy model's candidates `registers` names.  Without `registers` every value the
// table offers is a position.
struct AxisFilter {
  TailQueues tailQueues = TailQueues::All;
  const RegisterOffer* registers = nullptr;
};

// Every axis of `background` with somewhere to move: the applicable tunable keys in declaration order, LOADS and
// STORES taken apart into their access classes, each offering only the modes `env` can emit.
[[nodiscard]] std::vector<Axis> axesOf(const Env& env, const FFTConfig& fft, const UseConfig& background,
                                       const AxisFilter& filter = {});

// Where `config` puts `axis`.
[[nodiscard]] std::pair<int, int> positionOf(const Env& env, const FFTConfig& fft, const UseConfig& config,
                                             const Axis& axis);

// Puts `axis` at its `index`th position in `config`, leaving every other digit of a packed key where it was.
void place(UseConfig& config, const Axis& axis, size_t index);

// `config` with every key the table would not offer at its value alongside the rest returned to its default, to a
// fixpoint, and canonical: a structural move can take a dependent's value out of its own list, as SHUFL_BYTES_W=16
// does to WMUL at a wide width, and the host would only clamp it back.
[[nodiscard]] UseConfig fitted(const Env& env, const FFTConfig& fft, UseConfig config);

// The structural keys' effective values under `config`: which branch of the search it lies in.
[[nodiscard]] UseConfig branchOf(const Env& env, const FFTConfig& fft, const UseConfig& config);

}  // namespace tune
