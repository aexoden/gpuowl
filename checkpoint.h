// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <vector>

u64 residue(const std::vector<u32> &words);

// Shift-related functions
std::vector<u32> applyShift(const std::vector<u32> &words, u32 shift, u32 E);
std::vector<u32> removeShift(const std::vector<u32> &words, u32 shift, u32 E);
u32 getOrGenerateShift(i64 userShift, u32 E);
u32 computeCumulativeShift(u32 initialShift, u32 iterations, u32 E);

class PRPState {
  // Exponent, iteration, B1, block-size, res64.
  static constexpr const char *HEADER_v7 = "OWL PRP 7 %u %u %u %u %016llx\n";

  // Exponent, iteration, B1, block-size, res64, stage, nBitsBase
  static constexpr const char *HEADER_v8 =
      "OWL PRP 8 %u %u %u %u %016llx %u %u\n";

  // Exponent, iteration, B1, block-size, res64, stage, nBitsBase, shift
  static constexpr const char *HEADER_v9 =
      "OWL PRP 9x %u %u %u %u %016llx %u %u %u\n";

  static constexpr const char *SUFFIX = "";

  // bool loadV7(u32 E, u32 B1, u32 iniBlockSize);
  void loadInt(u32 E, u32 B1, u32 iniBlockSize, u32 initialShift);
  bool saveImpl(u32 E, const std::string &name);
  std::string durableName();

public:
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  u32 stage;
  u32 shift;

  std::vector<bool> basePower; // Stage-0 P-1 powerSmooth(B1).

  std::vector<u32> check;
  std::vector<u32> base;
  std::vector<u32> gcdAcc;

  static PRPState load(u32 E, u32 B1, u32 iniBlockSize, u32 initialShift) {
    PRPState prp;
    prp.loadInt(E, B1, iniBlockSize, initialShift);
    return prp;
  }

  void save(u32 E);

  PRPState initStage1(u32 iniB1, u32 iniBlockSize,
                      const std::vector<u32> &iniBase, u32 initialShift);
};
