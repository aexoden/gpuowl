// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <gmp.h>

using namespace std::string_literals;

// Residue from compacted words.
u64 residue(const std::vector<u32> &words) {
  return (u64(words[1]) << 32) | words[0];
}

// Generate a random shift
static u32 generateRandomShift(u32 E) {
  srand(time(NULL));
  return (rand() % (E - 1)) + 1;
}

// Shift a number left by s bits modulo 2^E - 1
static std::vector<u32> shiftLeft(const std::vector<u32> &words, u32 shift,
                                  u32 E) {
  if (shift == 0)
    return words;

  // Reduce shift modulo E (since shifting by E bits is equivalent to no shift
  // mod 2^E - 1)
  shift = shift % E;
  if (shift == 0)
    return words;

  mpz_t num, result, modulus;
  mpz_init(num);
  mpz_init(result);
  mpz_init(modulus);

  // Import the number from words
  mpz_import(num, words.size(), -1, sizeof(u32), 0, 0, words.data());

  // Calculate 2^E - 1
  mpz_set_ui(modulus, 1);
  mpz_mul_2exp(modulus, modulus, E);
  mpz_sub_ui(modulus, modulus, 1);

  // Shift left by multiplying by 2^shift
  mpz_mul_2exp(result, num, shift);

  // Reduce modulo 2^E - 1
  mpz_mod(result, result, modulus);

  // Export back to vector
  std::vector<u32> shifted(words.size());
  size_t count;
  mpz_export(shifted.data(), &count, -1, sizeof(u32), 0, 0, result);

  // Clear any remaining words that weren't filled
  for (size_t i = count; i < shifted.size(); ++i) {
    shifted[i] = 0;
  }

  mpz_clear(num);
  mpz_clear(result);
  mpz_clear(modulus);

  return shifted;
}

// Shift a number right by s bits modulo 2^E - 1 (inverse of left shift)
static std::vector<u32> shiftRight(const std::vector<u32> &words, u32 shift,
                                   u32 E) {
  if (shift == 0)
    return words;

  // Reduce shift modulo E
  shift = shift % E;
  if (shift == 0)
    return words;

  // Right shift by s is equivalent to left shift by E - s
  return shiftLeft(words, E - shift, E);
}

static std::string fileName(int E, const std::string &suffix) {
  return std::to_string(E) + suffix + ".owl";
}

void PRPState::save(u32 E) {
  std::string tempFile = fileName(E, "-temp"s + SUFFIX);
  if (!saveImpl(E, tempFile)) {
    throw "can't save";
  }

  std::string prevFile = fileName(E, "-prev"s + SUFFIX);
  remove(prevFile.c_str());

  std::string saveFile = fileName(E, SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());

  std::string persist = durableName();
  if (!persist.empty() && !saveImpl(E, fileName(E, persist + SUFFIX))) {
    throw "can't save";
  }
}

static bool write(FILE *fo, const std::vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, std::vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

static void powerSmooth(mpz_t a, u32 exp, u32 B1, u32 B2 = 0) {
  if (B2 == 0) {
    B2 = B1;
  }
  assert(B2 >= sqrt(B1));

  mpz_set_ui(a, exp);
  mpz_mul_2exp(a, a, 20); // boost 2s.

  mpz_t b;
  mpz_init(b);

  for (int k = log2(B1); k > 1; --k) {
    u32 limit = pow(B1, 1.0 / k);
    mpz_primorial_ui(b, limit);
    mpz_mul(a, a, b);
  }

  mpz_primorial_ui(b, B2);
  mpz_mul(a, a, b);
  mpz_clear(b);
}

// "Rev" means: most significant bit first (at index 0).
static std::vector<bool> powerSmoothBitsRev(u32 exp, u32 B1) {
  mpz_t a;
  mpz_init(a);
  powerSmooth(a, exp, B1);
  int nBits = mpz_sizeinbase(a, 2);
  std::vector<bool> bits;
  for (int i = nBits - 1; i >= 0; --i) {
    bits.push_back(mpz_tstbit(a, i));
  }
  assert(int(bits.size()) == nBits);
  mpz_clear(a);
  return bits;
}

static std::vector<u32> makeVect(u32 size, u32 elem0) {
  std::vector<u32> v(size);
  v[0] = elem0;
  return v;
}

PRPState PRPState::initStage1(u32 iniB1, u32 iniBlockSize,
                              const std::vector<u32> &iniBase,
                              u32 initialShift) {
  stage = 1;
  k = 0;
  B1 = iniB1;
  blockSize = iniBlockSize;
  base = iniBase;
  res64 = residue(base);
  u32 nWords = iniBase.size(); // (E - 1) / 32 + 1;
  check = gcdAcc = makeVect(nWords, 1);
  shift = initialShift;
  basePower.clear();
  return *this;
}

void PRPState::loadInt(u32 E, u32 wantB1, u32 iniBlockSize, u32 initialShift) {
  u32 nWords = (E - 1) / 32 + 1;
  std::string name = fileName(E, SUFFIX);
  auto fi{openRead(name)};
  if (!fi) {
    log("%s not found, starting from the beginning.\n", name.c_str());
    k = 0;
    B1 = wantB1;
    blockSize = iniBlockSize;

    if (B1 > 0) {
      stage = 0;
      res64 = 1;
      shift = 0;
      base = makeVect(nWords, 1);
      base[0] = 1;
      basePower = powerSmoothBitsRev(E, B1);
      log("powerSmooth(%u, %u) has %u bits\n", E, B1, u32(basePower.size()));
    } else {
      shift = initialShift;
      auto base = makeVect(nWords, 3);
      base = applyShift(base, shift, E);
      initStage1(B1, blockSize, std::move(base), shift);
    }
    return;
  }

  char line[256];
  if (!fgets(line, sizeof(line), fi.get())) {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");
  }

  stage = 1;
  u32 fileE = 0;
  u32 nBaseBits = 0;
  shift = 0;

  if (sscanf(line, HEADER_v7, &fileE, &k, &B1, &blockSize, &res64) == 5) {
    assert(E == fileE);
    if (B1 != wantB1) {
      log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1,
          name.c_str(), wantB1);
    }
    if (!read(fi.get(), nWords, &check)) {
      throw("load: error read check");
    }
    assert(stage == 1);
    if (B1 == 0) {
      base = makeVect(nWords, 3);
      gcdAcc = makeVect(nWords, 1);
    } else {
      bool ok = read(fi.get(), nWords, &base);
      assert(ok);
      gcdAcc = makeVect(nWords, 1);
    }
  } else if (sscanf(line, HEADER_v8, &fileE, &k, &B1, &blockSize, &res64,
                    &stage, &nBaseBits) == 7) {
    assert(E == fileE);
    if (B1 != wantB1) {
      log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1,
          name.c_str(), wantB1);
    }
    if (!read(fi.get(), nWords, &check)) {
      throw("load: error read check");
    }

    if (stage == 0) {
      std::swap(check, base);
      assert(res64 == residue(base));
      assert(B1 != 0);
      assert(k > 0 && k < nBaseBits);
      basePower = powerSmoothBitsRev(E, B1);
      assert(nBaseBits == basePower.size());
    } else {
      assert(stage == 1);
      if (B1 == 0) {
        base = makeVect(nWords, 3);
        gcdAcc = makeVect(nWords, 1);
      } else {
        bool ok =
            read(fi.get(), nWords, &base) && read(fi.get(), nWords, &gcdAcc);
        assert(ok);
      }
    }
  } else if (sscanf(line, HEADER_v9, &fileE, &k, &B1, &blockSize, &res64,
                    &stage, &nBaseBits, &shift) == 8) {
    assert(E == fileE);
    if (B1 != wantB1) {
      log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1,
          name.c_str(), wantB1);
    }
    if (!read(fi.get(), nWords, &check)) {
      throw("load: error read check");
    }

    if (stage == 0) {
      std::swap(check, base);
      assert(res64 == residue(base));
      assert(B1 != 0);
      assert(k > 0 && k < nBaseBits);
      basePower = powerSmoothBitsRev(E, B1);
      assert(nBaseBits == basePower.size());
    } else {
      assert(stage == 1);
      if (B1 == 0) {
        base = makeVect(nWords, 3);
        gcdAcc = makeVect(nWords, 1);
      } else {
        bool ok =
            read(fi.get(), nWords, &base) && read(fi.get(), nWords, &gcdAcc);
        assert(ok);
      }
    }
  } else {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");
  }

  log("%s loaded: k %u, B1 %u, block %u, res64 %016llx, stage %u, baseBits "
      "%u, shift %u\n",
      name.c_str(), k, B1, blockSize, res64, stage, nBaseBits, shift);
}

bool PRPState::saveImpl(u32 E, const std::string &name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);

  auto fo(openWrite(name));
  return fo &&
         fprintf(fo.get(), HEADER_v9, E, k, B1, blockSize, res64, stage,
                 u32(basePower.size()), shift) > 0 &&
         write(fo.get(), check) &&
         (B1 == 0 || stage == 0 ||
          (write(fo.get(), base) && write(fo.get(), gcdAcc)));
}

std::string PRPState::durableName() {
  if (k == 0 && B1 != 0) {
    return ".0";
  }
  if (k && (k % 20'000'000 == 0)) {
    return "."s + std::to_string(k / 1'000'000) + "M";
  }
  return "";
}

// Helper functions for shifting operations
std::vector<u32> applyShift(const std::vector<u32> &words, u32 shift, u32 E) {
  return shiftLeft(words, shift, E);
}

std::vector<u32> removeShift(const std::vector<u32> &words, u32 shift, u32 E) {
  return shiftRight(words, shift, E);
}

u32 getOrGenerateShift(i64 userShift, u32 E) {
  if (userShift >= 0) {
    return static_cast<u32>(userShift);
  }
  return generateRandomShift(E);
}

// Compute shift * 2^iterations mod E efficiently
u32 computeCumulativeShift(u32 initialShift, u32 iterations, u32 E) {
  if (initialShift == 0)
    return 0;

  // Compute 2^iterations mod E using binary exponentiation
  u64 power = 1;
  u64 base = 2;
  u32 exp = iterations;

  while (exp > 0) {
    if ((exp & 1) > 0) {
      power = (power * base) % E;
    }
    base = (base * base) % E;
    exp >>= 1;
  }

  return (static_cast<u64>(initialShift) * power) % E;
}
