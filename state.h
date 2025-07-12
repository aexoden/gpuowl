// Copyright 2017 Mihai Preda.

#include "common.h"
#include <vector>

#pragma once

std::vector<u32> compactBits(const std::vector<int> &dataVect, int E);
std::vector<int> expandBits(const std::vector<u32> &compactBits, int N, int E);
u64 residueFromRaw(u32 E, u32 N, const std::vector<int> &words);

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
std::pair<std::vector<double>, std::vector<double>> genWeights(int E, int W, int H);
