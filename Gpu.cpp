// Copyright 2017 Mihai Preda.

#include "Gpu.h"

#include "GCD.h"
#include "Primes.h"
#include "Result.h"
#include "Signal.h"
#include "Stats.h"
#include "args.h"
#include "checkpoint.h"
#include "state.h"
#include "timeutil.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

using namespace std::string_literals;

using double2 = std::pair<double, double>;

static_assert(sizeof(double2) == 16, "size double2");

// Returns the primitive root of unity of order N, to the power k.
static double2 root1(u32 N, u32 k) {
  long double angle = -TAU / N * k;
  return double2{double(cosl(angle)), double(sinl(angle))};
}

static double2 *smallTrigBlock(int W, int H, double2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}

static cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) {
    p = smallTrigBlock(w, std::min(radix, size / w), p);
  }
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

static void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W,
                         int H, int E) {
  int N = 2 * W * H;
  auto weights = genWeights(E, W, H);
  bufA.reset(
      makeBuf(context, BUF_CONST, sizeof(double) * N, weights.first.data()));
  bufI.reset(
      makeBuf(context, BUF_CONST, sizeof(double) * N, weights.second.data()));
}

Gpu::~Gpu() {}

Gpu::Gpu(u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
         cl_program program, cl_device_id device, cl_context context,
         bool timeKernels, bool useLongCarry)
    : E(E), N(W * BIG_H * 2), hN(N / 2), nW(nW), nH(nH),
      bufSize(N * sizeof(double)), useLongCarry(useLongCarry),
      useMiddle(BIG_H != SMALL_H), gcd(std::make_unique<GCD>()),
      queue(makeQueue(device, context)),

#define LOAD(name, workGroups)                                                 \
  name(program, queue.get(), device, workGroups, #name, timeKernels)
      LOAD(carryFused, BIG_H + 1), LOAD(carryFusedMul, BIG_H + 1),
      LOAD(fftP, BIG_H), LOAD(fftW, BIG_H), LOAD(fftH, (hN / SMALL_H)),
      LOAD(fftMiddleIn, hN / (256 * (BIG_H / SMALL_H))),
      LOAD(fftMiddleOut, hN / (256 * (BIG_H / SMALL_H))),
      LOAD(carryA, nW * (BIG_H / 16)), LOAD(carryM, nW * (BIG_H / 16)),
      LOAD(carryB, nW * (BIG_H / 16)),
      LOAD(transposeW, (W / 64) * (BIG_H / 64)),
      LOAD(transposeH, (W / 64) * (BIG_H / 64)),
      LOAD(transposeIn, (W / 64) * (BIG_H / 64)),
      LOAD(transposeOut, (W / 64) * (BIG_H / 64)), LOAD(square, hN / SMALL_H),
      LOAD(multiply, hN / SMALL_H), LOAD(multiplySub, hN / SMALL_H),
      LOAD(tailFused, (hN / SMALL_H) / 2), LOAD(readResidue, 1),
      LOAD(isNotZero, 256), LOAD(isEqual, 256),
#undef LOAD

      bufData(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
      bufCheck(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
      bufAux(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
      bufBase(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
      bufAcc(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),

      bufTrigW(genSmallTrig(context, W, nW)),
      bufTrigH(genSmallTrig(context, SMALL_H, nH)),
      buf1{makeBuf(context, BUF_RW, bufSize)},
      buf2{makeBuf(context, BUF_RW, bufSize)},
      buf3{makeBuf(context, BUF_RW, bufSize)},
      bufCarry{makeBuf(context, BUF_RW, bufSize / 2)},
      bufReady{makeBuf(context, BUF_RW, BIG_H * sizeof(int))},
      bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int))),
      bufBaseDown(makeBuf(context, BUF_RW, bufSize)) {
  setupWeights(context, bufA, bufI, W, BIG_H, E);

  carryFused.setFixedArgs(3, bufA, bufI, bufTrigW);
  carryFusedMul.setFixedArgs(3, bufA, bufI, bufTrigW);

  fftP.setFixedArgs(2, bufA, bufTrigW);
  fftW.setFixedArgs(1, bufTrigW);
  fftH.setFixedArgs(1, bufTrigH);

  carryA.setFixedArgs(3, bufI);
  carryM.setFixedArgs(3, bufI);
  tailFused.setFixedArgs(1, bufTrigH);

  queue.zero(bufReady, BIG_H * sizeof(int));
  queue.zero(bufAcc, N * sizeof(int));
  queue.write(bufAcc, std::vector<u32>{1});
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  struct Info {
    std::string name;
    double totalTime;
    double avgTime;
    u32 n;
  };

  double total = 0;
  std::vector<Info> infos;
  for (Kernel *k : kerns) {
    StatsInfo s = k->resetStats();
    Info info{k->getName(), s.msPerSq * s.nSq, s.msPerSq, s.nSq};
    infos.push_back(info);
    total += info.totalTime;
  }

  // std::sort(infos.begin(), infos.end(), [](const Info &a, const Info &b) {
  // return a.stats.sum >= b.stats.sum; });

  for (Info info : infos) {
    float percent = 100 / total * info.totalTime;
    if (true || percent >= .1f) {
      log("%4.1f%% %-14s : %6.0f us/call x %5d calls\n", percent,
          info.name.c_str(), info.avgTime, info.n);
    }
  }
  log("\n");
}

struct FftConfig {
  u32 width, height, middle;
  u32 fftSize;
  u32 maxExp;

  static u32 getMaxExp(u32 fftSize) {
    return fftSize * (17.77 + 0.33 * (24 - log2(fftSize)));
  }

  FftConfig(u32 width, u32 height, u32 middle)
      : width(width), height(height), middle(middle),
        fftSize(width * height * middle * 2),
        // 17.88 + 0.36 * (24 - log2(n)); Update after feedback on 86700001, FFT
        // 4608 (18.37b/w) being insufficient.
        maxExp(getMaxExp(fftSize)) {
    assert(width == 64 || width == 256 || width == 512 || width == 1024 ||
           width == 2048 || width == 4096);
    assert(height == 64 || height == 256 || height == 512 || height == 1024 ||
           height == 2048);
    assert(middle == 1 || middle == 3 || middle == 5 || middle == 9);
  }
};

static std::vector<FftConfig> genConfigs() {
  std::vector<FftConfig> configs;
  for (u32 width : {64, 256, 512, 1024, 2048, 4096}) {
    for (u32 height : {64, 256, 512, 1024, 2048}) {
      for (u32 middle : {1, 3, 5, 9}) {
        configs.push_back(FftConfig(width, height, middle));
      }
    }
  }
  std::sort(configs.begin(), configs.end(),
            [](const FftConfig &a, const FftConfig &b) {
              if (a.fftSize != b.fftSize) {
                return (a.fftSize < b.fftSize);
              }
              assert(a.width != b.width);
              if (a.width == 1024) {
                return true;
              }
              if (b.width == 1024) {
                return false;
              }
              return (a.width < b.width);
            });
  return configs;
}

static FftConfig getFftConfig(const std::vector<FftConfig> &configs, u32 E,
                              int argsFftSize) {
  int i = 0;
  int n = int(configs.size());
  if (argsFftSize < 10) { // fft delta or not specified.
    while (i < n - 1 && configs[i].maxExp < E) {
      ++i;
    }
    i = std::max(0, std::min(i + argsFftSize, n - 1));
  } else { // user-specified fft size.
    while (i < n - 1 && u32(argsFftSize) > configs[i].fftSize) {
      ++i;
    }
  }
  return configs[i];
}

std::vector<int> Gpu::readSmall(Buffer &buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return queue.read<int>(bufSmallOut, 128);
}

static std::string numberK(u32 n) {
  return (n % (1024 * 1024) == 0) ? std::to_string(n / (1024 * 1024)) + "M"
         : (n % 1024 == 0)        ? std::to_string(n / 1024) + "K"
                                  : std::to_string(n);
}

static std::string configName(u32 width, u32 height, u32 middle) {
  return numberK(width) + '-' + numberK(height) +
         ((middle != 1) ? "-"s + numberK(middle) : ""s);
}

std::unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  std::vector<FftConfig> configs = genConfigs();
  if (args.listFFT) {
    std::string variants;
    u32 activeSize = 0;
    for (auto c : configs) {
      if (c.fftSize != activeSize) {
        if (!variants.empty()) {
          log("FFT %5s [%6.2fM - %7.2fM] %s\n", numberK(activeSize).c_str(),
              activeSize * 1.5 / 1'000'000,
              FftConfig::getMaxExp(activeSize) / 1'000'000.0, variants.c_str());
          variants.clear();
        }
      }
      activeSize = c.fftSize;
      variants += " "s + configName(c.width, c.height, c.middle);
    }
    if (!variants.empty()) {
      log("FFT %5s [%6.2fM - %7.2fM] %s\n", numberK(activeSize).c_str(),
          activeSize * 1.5 / 1'000'000,
          FftConfig::getMaxExp(activeSize) / 1'000'000.0, variants.c_str());
    }
  }

  FftConfig config = getFftConfig(configs, E, args.fftSize);
  int WIDTH = config.width;
  int SMALL_HEIGHT = config.height;
  int MIDDLE = config.middle;
  int N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  std::string configName = (N % (1024 * 1024))
                               ? std::to_string(N / 1024) + "K"
                               : std::to_string(N / (1024 * 1024)) + "M";

  int nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  int nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

  float bitsPerWord = E / float(N);
  std::string strMiddle =
      (MIDDLE == 1) ? "" : (std::string(", Middle ") + std::to_string(MIDDLE));
  log("%u FFT %dK: Width %dx%d, Height %dx%d%s; %.2f bits/word\n", E, N / 1024,
      WIDTH / nW, nW, SMALL_HEIGHT / nH, nH, strMiddle.c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 1.5) {
    log("FFT size too large for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too large";
  }

  bool useLongCarry = (bitsPerWord < 14.5f) ||
                      (args.carry == Args::CARRY_LONG) ||
                      (args.carry == Args::CARRY_AUTO && WIDTH >= 2048);

  log("using %s carry kernels\n", useLongCarry ? "long" : "short");

  std::string clArgs = args.clArgs;
  if (!args.dump.empty()) {
    clArgs += " -save-temps=" + args.dump + "/" + configName;
  }

  bool timeKernels = args.timeKernels;

  cl_device_id device = getDevice(args.device);
  if (!device) {
    throw "No OpenCL device";
  }

  log("%s\n", getLongInfo(device).c_str());
  // if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

  Context context(createContext(device));
  Holder<cl_program> program(compile(device, context.get(), "gpuowl", clArgs,
                                     {{"EXP", E},
                                      {"WIDTH", WIDTH},
                                      {"SMALL_HEIGHT", SMALL_HEIGHT},
                                      {"MIDDLE", MIDDLE}},
                                     args.usePrecompiled));
  if (!program) {
    throw "OpenCL compilation";
  }

  return std::make_unique<Gpu>(E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT,
                               nW, nH, program.get(), device, context.get(),
                               timeKernels, useLongCarry);
}

std::vector<u32> Gpu::readData() { return compactBits(readOut(bufData), E); }
std::vector<u32> Gpu::readCheck() { return compactBits(readOut(bufCheck), E); }
std::vector<u32> Gpu::readAcc() { return compactBits(readOut(bufAcc), E); }

std::vector<u32> Gpu::writeData(const std::vector<u32> &v) {
  writeIn(v, bufData);
  return v;
}

std::vector<u32> Gpu::writeCheck(const std::vector<u32> &v) {
  writeIn(v, bufCheck);
  return v;
}

// The modular multiplication io *= in.
void Gpu::modMul(Buffer &in, Buffer &io) {
  fftP(in, buf1);
  tW(buf1, buf3);

  fftP(io, buf1);
  tW(buf1, buf2);

  fftH(buf2);
  fftH(buf3);
  multiply(buf2, buf3);
  fftH(buf2);

  tH(buf2, buf1);

  fftW(buf1);
  carryA(buf1, io, bufCarry);
  carryB(io, bufCarry);
};

void Gpu::writeState(const std::vector<u32> &check,
                     const std::vector<u32> &base, const std::vector<u32> &acc,
                     u32 blockSize) {
  assert(blockSize > 0);

  writeCheck(check);
  queue.copy<int>(bufCheck, bufData, N);
  queue.copy<int>(bufCheck, bufBase, N);

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    dataLoopMul(std::vector<bool>(n));
    modMul(bufBase, bufData);
    queue.copy<int>(bufData, bufBase, N);
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);

  blockSize /= n;
  for (u32 i = 0; i < blockSize - 1; ++i) {
    dataLoopMul(std::vector<bool>(n));
    modMul(bufBase, bufData);
  }

  writeBase(base);
  modMul(bufBase, bufData);

  writeIn(acc, bufAcc);
}

void Gpu::updateCheck() { modMul(bufData, bufCheck); }

bool Gpu::doCheck(int blockSize) {
  queue.copy<int>(bufCheck, bufAux, N);
  modSqLoopMul(bufAux, std::vector<bool>(blockSize));
  modMul(bufBase, bufAux);
  updateCheck();
  return equalNotZero(bufCheck, bufAux);
}

static u32 countOnBits(const std::vector<bool> &bits) {
  u32 n = 0;
  for (bool b : bits) {
    n += b;
  }
  return n;
}

u32 Gpu::dataLoopAcc(u32 kBegin, u32 kEnd, const std::vector<bool> &kset) {
  assert(kEnd > kBegin);
  std::vector<bool> accs(kset.begin() + kBegin, kset.begin() + kEnd);
  dataLoopAcc(accs);
  return countOnBits(accs);
}
/*
  vector<bool> accs;
  u32 nAcc = 0;
  for (u32 k = kBegin; k < kEnd; ++k) {
    bool on = kset.count(k);
    accs.push_back(on);
    nAcc += on;
  }
  assert(accs.size() == kEnd - kBegin);
  modSqLoopAcc(bufData, accs);
  return nAcc;
}
*/

void Gpu::logTimeKernels() {
  ::logTimeKernels({&carryFused,  &carryFusedMul, &fftP,         &fftW,
                    &fftH,        &fftMiddleIn,   &fftMiddleOut, &carryA,
                    &carryM,      &carryB,        &transposeW,   &transposeH,
                    &transposeIn, &transposeOut,  &square,       &multiply,
                    &multiplySub, &tailFused,     &readResidue,  &isNotZero,
                    &isEqual});
}

void Gpu::tW(Buffer &in, Buffer &out) {
  transposeW(in, out);
  if (useMiddle) {
    fftMiddleIn(out);
  }
}

void Gpu::tH(Buffer &in, Buffer &out) {
  if (useMiddle) {
    fftMiddleOut(in);
  }
  transposeH(in, out);
}

std::vector<u32> Gpu::writeBase(const std::vector<u32> &v) {
  writeIn(v, bufBase);
  fftP(bufBase, buf1);
  tW(buf1, bufBaseDown);
  fftH(bufBaseDown);
  return v;
}

std::vector<int> Gpu::readOut(Buffer &buf) {
  transposeOut(buf, bufAux);
  return queue.read<int>(bufAux, N);
}

void Gpu::writeIn(const std::vector<u32> &words, Buffer &buf) {
  writeIn(expandBits(words, N, E), buf);
}

void Gpu::writeIn(const std::vector<int> &words, Buffer &buf) {
  queue.write(bufAux, words);
  transposeIn(bufAux, buf);
}

void Gpu::modSqLoopMul(Buffer &io, const std::vector<bool> &muls) {
  assert(!muls.empty());
  bool dataIsOut = true;

  for (auto it = muls.begin(), end = muls.end(); it < end; ++it) {
    if (dataIsOut) {
      fftP(io, buf1);
    }
    tW(buf1, buf2);
    tailFused(buf2);
    tH(buf2, buf1);

    dataIsOut = useLongCarry || it == prev(end);
    if (dataIsOut) {
      fftW(buf1);
      *it ? carryM(buf1, io, bufCarry) : carryA(buf1, io, bufCarry);
      carryB(io, bufCarry);
    } else {
      *it ? carryFusedMul(buf1, bufCarry, bufReady)
          : carryFused(buf1, bufCarry, bufReady);
    }
  }
}

void Gpu::exitKerns(Buffer &buf, Buffer &bufWords) {
  fftW(buf);
  carryA(buf, bufWords, bufCarry);
  carryB(bufWords, bufCarry);
}

void Gpu::modSqLoopAcc(Buffer &io, const std::vector<bool> &accs) {
  assert(!accs.empty());
  bool dataIsOut = true;
  bool accIsOut = true;

  for (auto it = accs.begin(), end = accs.end(); it < end; ++it) {
    if (dataIsOut) {
      fftP(io, buf1);
    }
    tW(buf1, buf2);

    if (*it) {
      fftH(buf2);
      if (accIsOut) {
        fftP(bufAcc, buf3);
      }
      tW(buf3, buf1);
      fftH(buf1);
      multiplySub(buf1, buf2, bufBaseDown);
      fftH(buf1);
      tH(buf1, buf3);
      square(buf2);
      fftH(buf2);

      accIsOut =
          useLongCarry || !any_of(next(it), end, [](bool on) { return on; });
      if (accIsOut) {
        exitKerns(buf3, bufAcc);
      } else {
        carryFused(buf3, bufCarry, bufReady);
      }
    } else {
      tailFused(buf2);
    }

    tH(buf2, buf1);

    dataIsOut = useLongCarry || it == prev(end);
    if (dataIsOut) {
      exitKerns(buf1, io);
    } else {
      carryFused(buf1, bufCarry, bufReady);
    }
  }
}

bool Gpu::equalNotZero(Buffer &buf1, Buffer &buf2) {
  queue.zero(bufSmallOut, sizeof(int));
  u32 sizeBytes = N * sizeof(int);
  isNotZero(sizeBytes, buf1, bufSmallOut);
  isEqual(sizeBytes, buf1, buf2, bufSmallOut);
  return queue.read<int>(bufSmallOut, 1)[0];
}

u64 Gpu::bufResidue(Buffer &buf) {
  u32 earlyStart = N / 2 - 32;
  std::vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(E, N, readBuf);
}

static std::string makeLogStr(u32 E, std::string status, int k, u64 res,
                              const StatsInfo &info, u32 nIters) {
  int etaMins = (nIters - k) * info.msPerIt * (1 / 60000.f) + .5f;
  int days = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins = etaMins % 60;

  char buf[256];
  std::string ghzStr;

  snprintf(
      buf, sizeof(buf),
      "%u %2s %8d %5.2f%%; %.2f ms/sq, %4u MULs;%s ETA %dd %02d:%02d; %016llx",
      E, status.c_str(), k, k / float(nIters) * 100, info.msPerSq, info.nMul,
      ghzStr.c_str(), days, hours, mins, res);
  return buf;
}

static void doLog(int E, int k, u32 timeCheck, u64 res, bool checkOK,
                  Stats &stats, u32 nIters, u32 shift) {
  std::string resLabel = (shift > 0) ? " (shifted)" : "";
  std::string logStr =
      makeLogStr(E, checkOK ? "OK" : "EE", k, res, stats.reset(), nIters);

  size_t resPos = logStr.find_last_of(' ');
  if (resPos != std::string::npos && shift > 0) {
    logStr.insert(resPos, resLabel);
  }

  log("%s (check %.2fs)\n", logStr.c_str(), timeCheck * .001f);
  stats.reset();
}

static void doLogWithBothResidues(int E, int k, u32 timeCheck, u64 shiftedRes,
                                  u64 unshiftedRes, bool checkOK, Stats &stats,
                                  u32 nIters, u32 cumulativeShift) {
  std::string logStr = makeLogStr(E, checkOK ? "OK" : "EE", k, unshiftedRes,
                                  stats.reset(), nIters);

  size_t resPos = logStr.find_last_of(' ');
  if (resPos != std::string::npos) {
    std::string prefix = logStr.substr(0, resPos + 1); // Include the space
    log("%s%016llx (unshifted), %016llx (shifted, shift %u) (check %.2fs)\n",
        prefix.c_str(), unshiftedRes, shiftedRes, cumulativeShift,
        timeCheck * .001f);
  } else {
    log("%s; unshifted: %016llx, shifted: %016llx (shift %u) (check %.2fs)\n",
        logStr.c_str(), unshiftedRes, shiftedRes, cumulativeShift,
        timeCheck * .001f);
  }

  stats.reset();
}

static void doSmallLog(int E, int k, u64 res, Stats &stats, u32 nIters,
                       u32 shift) {
  std::string resLabel = (shift > 0) ? " (shifted)" : "";
  std::string logStr = makeLogStr(E, "", k, res, stats.reset(), nIters);

  // Insert the shift label before the residue
  size_t resPos = logStr.find_last_of(' ');
  if (resPos != std::string::npos && shift > 0) {
    logStr.insert(resPos, resLabel);
  }

  log("%s\n", logStr.c_str());
  stats.reset();
}

static std::vector<u32> bitNeg(const std::vector<u32> &v) {
  std::vector<u32> ret;
  ret.reserve(v.size());
  for (auto x : v) {
    ret.push_back(~x);
  }
  return ret;
}

// Checks whether a == bitNeg(b) ignoring the last word.
// (this is because the last word is often only partially filled with bits of
// exponent E) 'a' passed by value intentional.
static bool equalNeg(std::vector<u32> a, const std::vector<u32> &b) {
  assert(!a.empty() && !b.empty());
  auto c = bitNeg(b);
  a.pop_back();
  c.pop_back();
  return a == c;
}

PRPState Gpu::loadPRP(u32 E, u32 iniB1, u32 iniBlockSize, i64 userShift) {
  auto initialShift = getOrGenerateShift(userShift, E);
  auto loaded = PRPState::load(E, iniB1, iniBlockSize, initialShift);
  if (loaded.stage == 0) {

    doStage0(loaded.k, loaded.B1, loaded.blockSize, std::move(loaded.base),
             std::move(loaded.basePower), initialShift);
    loaded = PRPState::load(E, iniB1, iniBlockSize, initialShift);
  }

  assert(loaded.stage == 1);

  std::vector<u32> unshiftedBase = loaded.base;

  if (loaded.shift > 0) {
    unshiftedBase = removeShift(loaded.base, loaded.shift, E);
  }

  writeState(loaded.check, loaded.base, loaded.gcdAcc, loaded.blockSize);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);
  updateCheck();
  if (!ok) {
    std::string resLabel = (loaded.shift > 0) ? " (shifted)" : "";
    log("%u EE loaded: %d, B1 %u, blockSize %d, %016llx%s (expected "
        "%016llx%s)\n",
        E, loaded.k, loaded.B1, loaded.blockSize, res64, resLabel.c_str(),
        loaded.res64, resLabel.c_str());
    throw "error on load";
  }

  if (loaded.B1 != iniB1) {
    log("B1 mismatch %u %u\n", iniB1, loaded.B1);
    throw "B1 mismatch";
  }

  return loaded;
}

static std::pair<std::vector<bool>, u32> kselect(u32 E, u32 blockSize, u32 B1,
                                                 u32 B2, u32 shift) {
  u32 lastIteration = ((E - 2) / blockSize + 1) * blockSize;

  if (!B1) {
    return make_pair(std::vector<bool>(lastIteration + 1), 0);
  }

  // If shift is non-zero, disable P-1 accumulation due to incompatibility
  if (shift > 0) {
    log("%u P-1 accumulation disabled due to non-zero shift (%u) - B1=%u B2=%u "
        "specified but no P-1 accumulation will be performed\n",
        E, shift, B1, B2);
    u32 effectiveB2 = (B2 == 0) ? E : B2; // Use default B2 logic for reporting
    return make_pair(std::vector<bool>(lastIteration + 1, false), effectiveB2);
  }

  // log("Starting P-1 selection: exp %u, B1 %u, B2 %u\n", E, B1, B2);
  Timer timer;

  Primes primes(B2 + 1);
  std::vector<bool> covered(lastIteration + 1);
  std::vector<bool> on(lastIteration + 1);

  u32 reportB2 = 0;

  for (u32 p : primes.from(B1)) {
    u32 z = primes.zn2(p);
    if (z <= lastIteration) {
      if (!covered[z]) {
        // assert(!on[z]);
        for (u32 d : primes.divisors(z)) {
          covered[d] = true;
          on[d] = false;
        }
        on[z] = true;
      }
    } else if (reportB2 == 0) {
      reportB2 = p - 1;
    }
  }
  if (reportB2 == 0) {
    reportB2 = B2;
  }

  on[1] = true; // special-case to allow testing P-1 first-stage early, as:
                // base^2 - 1 = (base - 1) * (base + 1)
  log("%u B1=%u B2=%u (effective B2=%u) selected %u P-1 points in %.2fs\n", E,
      B1, B2, reportB2, countOnBits(on), timer.deltaMillis() * (1.0 / 1000));
  return make_pair(on, reportB2);
}

void Gpu::doStage0(u32 k, u32 B1, u32 blockSize, std::vector<u32> &&base,
                   std::vector<bool> &&basePower, u32 shift) {
  writeData(base);
  u32 kEnd = basePower.size();
  assert(k < kEnd);

  Stats stats;
  Timer timer;
  Signal signal;
  while (k < kEnd) {
    u32 nIts = std::min(u32(kEnd - k), blockSize);
    dataLoopMul(std::vector<bool>(basePower.begin() + k,
                                  basePower.begin() + (k + nIts)));
    queue.finish();
    stats.add(timer.deltaMillis(), nIts, 0);
    k += nIts;

    bool doStop = signal.stopRequested();
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    if (k % 10000 == 0 || doStop) {
      auto data = readData();
      u64 res64 = residue(data);
      log("%s\n",
          makeLogStr(E, "P-1", k, res64, stats.reset(), basePower.size())
              .c_str());
      stats.reset();
      PRPState state;
      state.k = k;
      state.B1 = B1;
      state.blockSize = blockSize;
      state.res64 = res64;
      state.stage = 0;
      state.shift = 0;
      state.basePower = basePower;
      state.check = data;
      state.save(E);
    }

    if (doStop) {
      throw "stop requested";
    }
  }

  auto stage1_base = readData();

  if (shift > 0) {
    log("Applying shift %u to base\n", shift);
    stage1_base = applyShift(stage1_base, shift, E);
  }

  PRPState{}.initStage1(B1, blockSize, stage1_base, shift).save(E);
}

PRPResult Gpu::isPrimePRP(u32 E, const Args &args, u32 B1, u32 B2) {
  // u32 N = this->getFFTSize();

  assert(B2 == 0 || B2 >= B1);
  if (B1 != 0 && B2 == 0) {
    B2 = E * 1.1; // by default test a some primes above E as well.
  }
  // log("PRP M(%d), FFT %dK, %.2f bits/word, B1 %u, B2 %u\n", E, N/1024, E /
  // float(N), B1, B2);

  PRPState loaded = loadPRP(E, B1, args.blockSize, args.shift);

  u32 k = loaded.k;
  u32 blockSize = loaded.blockSize;
  assert(blockSize > 0 && 10000 % blockSize == 0);

  u32 shift = loaded.shift;
  std::vector<u32> base = loaded.base;

  const u32 kEnd =
      E - 1; // Type-4 per
             // http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  auto kselectRet = kselect(E, blockSize, B1, B2, shift);
  std::vector<bool> kset = kselectRet.first;
  u32 effectiveB2 = kselectRet.second;

  const u32 checkStep = blockSize * blockSize;

  u32 startK = k;

  Signal signal;
  Stats stats;

  // Number of sequential errors (with no success in between). If this ever gets
  // high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  int nGcdAcc = (B1 > 0 && shift == 0) ? 1 : 0;
  u64 finalRes64 = 0;
  u32 nTotalIters = ((kEnd - 1) / blockSize + 1) * blockSize;

  // Store original unshifted base for final comparison
  std::vector<u32> originalBase;
  if (B1 == 0) {
    // For pure PRP, original base is 3
    u32 nWords = (E - 1) / 32 + 1;
    originalBase.resize(nWords);
    originalBase[0] = 3;
    for (u32 i = 1; i < nWords; ++i) {
      originalBase[i] = 0;
    }
  } else {
    // For PRP/P-1, remove shift from current base to get original
    originalBase = removeShift(base, shift, E);
  }

  while (true) {
    assert(k % blockSize == 0);
    u32 nAcc = 0;
    if (k < kEnd && k + blockSize >= kEnd) {
      nAcc = dataLoopAcc(k, kEnd, kset);
      auto words = this->roundtripData();

      // Remove shift from final result for comparison
      // After kEnd iterations, the cumulative shift is shift * 2^kEnd mod E
      u32 cumulativeShift = computeCumulativeShift(shift, kEnd, E);
      auto unshiftedWords = removeShift(words, cumulativeShift, E);

      finalRes64 = residue(unshiftedWords);
      isPrime = (unshiftedWords == originalBase) ||
                equalNeg(unshiftedWords, originalBase);

      u64 shiftedRes64 = residue(words);
      if (shift > 0) {
        log("%s %8d / %d, %016llx (unshifted), %016llx (shifted, cumulative "
            "shift %u), base %016llx\n",
            isPrime ? "PP" : "CC", kEnd, E, finalRes64, shiftedRes64,
            cumulativeShift, residue(originalBase));
      } else {
        log("%s %8d / %d, %016llx, base %016llx\n", isPrime ? "PP" : "CC", kEnd,
            E, finalRes64, residue(originalBase));
      }

      int itersLeft = blockSize - (kEnd - k);
      if (itersLeft > 0) {
        nAcc += dataLoopAcc(kEnd, kEnd + itersLeft, kset);
      }
    } else {
      nAcc = dataLoopAcc(k, k + blockSize, kset);
    }
    nGcdAcc += nAcc;
    k += blockSize;

    queue.finish();

    if (shift == 0 && gcd->isReady()) {
      std::string factor = gcd->get();
      if (!factor.empty()) {
        // log("GCD: %s\n", factor.c_str());
        return PRPResult{factor,      false, 0, residue(originalBase),
                         effectiveB2, shift};
      }
    }

    stats.add(timer.deltaMillis(), blockSize, nAcc);
    bool doStop = signal.stopRequested();
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    bool doCheck = (k % checkStep == 0) ||
                   (k >= kEnd && k < kEnd + blockSize) || doStop ||
                   (k - startK == 2 * blockSize);

    if (!doCheck) {
      this->updateCheck();
      if (k % 10000 == 0) {
        doSmallLog(E, k, dataResidue(), stats, nTotalIters, shift);
        if (args.timeKernels) {
          this->logTimeKernels();
        }
      }
      continue;
    }

    std::vector<u32> check = this->roundtripCheck();
    bool ok = this->doCheck(blockSize);

    u64 res64 = dataResidue();
    std::vector<u32> gcdAcc = B1 ? readAcc() : std::vector<u32>();

    // the check time (above) is accounted separately, not added to iteration
    // time.
    if (shift > 0) {
      // For shifted runs, show both residues at check time
      auto currentData = this->roundtripData();
      u32 cumulativeShift = computeCumulativeShift(shift, k, E);
      auto unshiftedData = removeShift(currentData, cumulativeShift, E);
      u64 unshiftedRes = residue(unshiftedData);
      doLogWithBothResidues(E, k, timer.deltaMillis(), res64, unshiftedRes, ok,
                            stats, nTotalIters, cumulativeShift);
    } else {
      doLog(E, k, timer.deltaMillis(), res64, ok, stats, nTotalIters, 0);
    }

    if (ok) {
      if (k < kEnd) {
        PRPState state;
        state.k = k;
        state.B1 = B1;
        state.blockSize = blockSize;
        state.res64 = res64;
        state.stage = 1;
        state.shift = shift;
        state.check = check;
        state.base = base;
        state.gcdAcc = gcdAcc;
        state.save(E);
      }
      if (shift == 0 && k % 1'000'000 < checkStep && nGcdAcc &&
          !gcd->isOngoing() && !doStop) {
        gcd->start(E, gcdAcc, 0);
        nGcdAcc = 0;
      }
      if (isPrime || k >= kEnd) {
        return PRPResult{
            "", isPrime, finalRes64, residue(originalBase), effectiveB2, shift};
      }
      nSeqErrors = 0;
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        throw "too many errors";
      }

      auto loaded = loadPRP(E, B1, blockSize, args.shift);
      k = loaded.k;
      assert(blockSize == loaded.blockSize);
      assert(base == loaded.base);
      assert(B1 == loaded.B1);
      shift = loaded.shift;
      nGcdAcc = (B1 > 0 && shift == 0) ? 1 : 0;
    }
    if (args.timeKernels) {
      this->logTimeKernels();
    }
    if (doStop) {
      throw "stop requested";
    }
  }
}
