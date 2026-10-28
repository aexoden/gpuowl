// Copyright (C) Mihai Preda

#pragma once

#include "FFTConfig.h"

#include <algorithm>
#include <string>
#include <vector>

class Args;

// What opens a tune.txt the tuner wrote, after "# ", followed by ", build <its kernels' fingerprint>".
inline constexpr const char* TUNE_FILE_MAGIC = "prpll tune v2";

class TuneEntry {
public:
  double cost;
  FFTConfig fft;
  vector<KeyVal> uses;     // -use settings for this FFT alone, from the end of its tune.txt line
  u64 maxExpCap{};         // the line's "# <maxExp>" where it is below the FFT's own, which it then lowers; else 0

  // What a line of the tuner's own tune.txt also names (see tune::tuneFileText): the test kind it was measured for
  // ("prp" or "ll"), the carry regime it runs in, and the lowest exponent it holds.  Empty and 0 on a line that names
  // none of them, which holds every exponent its FFT does up to its maxExp.
  std::string kind{};
  std::string regime{};
  u64 minExp{};
  bool fallback{};         // a PRP line offered to an LL test that no LL line holds

  // The largest exponent this entry serves: the FFT's own, or the lower one its line names (e.g. where its settings were
  // measured to be accurate only that far).
  [[nodiscard]] u64 maxExp() const { return maxExpCap ? std::min(maxExpCap, fft.maxExp()) : fft.maxExp(); }

  bool update(std::vector<TuneEntry>&) const;
  [[nodiscard]] bool willUpdate(const vector<TuneEntry>&) const;

  // -oldtune's frontier.  ll: for an LL test, which reads tune-ll.txt where there is one (LL and PRP run different carry
  // kernels, whose costs can differ by FFT type and device), and tune.txt otherwise.
  static vector<TuneEntry> readTuneFile(const Args& args, bool ll = false);

  // The lines production chooses from for a test, cheapest first: those of its own kind, then, for an LL test, the PRP
  // lines marked as a fallback.  A tune.txt the tuner wrote holds both kinds; otherwise an LL test reads tune-ll.txt
  // where there is one, with tune.txt as its fallback.
  static vector<TuneEntry> tunedLines(const Args& args, bool ll = false);

  // The line production runs at E: the first of `lines` (tunedLines()) that holds E, in the regime it was measured in.
  // `log`: say why a line that holds E is passed over, and when a fallback line is taken.
  static const TuneEntry* pick(const vector<TuneEntry>& lines, u64 E, double overdrive, bool fp64, bool log);

  // The settings of the line for exactly this FFT, for an FFT given with -fft: the cheapest that holds E, else the
  // cheapest of the FFT's lines.
  static vector<KeyVal> usesFor(const Args& args, const FFTConfig& fft, u64 E, bool ll = false);

  // Whether tune.txt is the tuner's own (tune::TUNE_FILE_MAGIC), which -oldtune must not rewrite.
  static bool tunerFile(const Args& args);

  static void writeTuneFile(const vector<TuneEntry>&);
};
