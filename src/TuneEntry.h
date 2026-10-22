// Copyright (C) Mihai Preda

#pragma once

#include "FFTConfig.h"

#include <algorithm>
#include <vector>

class Args;

class TuneEntry {
public:
  double cost;
  FFTConfig fft;
  vector<KeyVal> uses;     // -use settings for this FFT alone, from the end of its tune.txt line
  u64 maxExpCap{};         // the line's "# <maxExp>" where it is below the FFT's own, which it then lowers; else 0

  // The largest exponent this entry serves: the FFT's own, or the lower one its line names (e.g. where its settings were
  // measured to be accurate only that far).
  [[nodiscard]] u64 maxExp() const { return maxExpCap ? std::min(maxExpCap, fft.maxExp()) : fft.maxExp(); }

  bool update(std::vector<TuneEntry>&) const;
  [[nodiscard]] bool willUpdate(const vector<TuneEntry>&) const;

  // ll: for an LL test, which reads tune-ll.txt where there is one (LL and PRP run different carry kernels, whose costs
  // can differ by FFT type and device), and tune.txt otherwise.
  static vector<TuneEntry> readTuneFile(const Args& args, bool ll = false);
  static vector<KeyVal> usesFor(const Args& args, const FFTConfig& fft, bool ll = false);
  static void writeTuneFile(const vector<TuneEntry>&);
};
