// Copyright 2017 Mihai Preda.

#pragma once

#include "clwrap.h"
#include "common.h"
#include "kernel.h"

#include <vector>
#include <string>
#include <memory>

struct Args;
struct PRPResult;
struct PRPState;
class GCD;

class Gpu {
  u32 E;
  u32 N;

  int hN, nW, nH, bufSize;
  bool useLongCarry;
  bool useMiddle;

  std::unique_ptr<GCD> gcd;
  
  Queue queue;
  
  Kernel carryFused;
  Kernel carryFusedMul;
  Kernel fftP;
  Kernel fftW;
  Kernel fftH;
  Kernel fftMiddleIn;
  Kernel fftMiddleOut;
  
  Kernel carryA;
  Kernel carryM;
  Kernel carryB;
  
  Kernel transposeW, transposeH;
  Kernel transposeIn, transposeOut;

  Kernel square;
  Kernel multiply;
  Kernel multiplySub;
  Kernel tailFused;
  Kernel readResidue;
  Kernel isNotZero;
  Kernel isEqual;
  
  Buffer bufData, bufCheck, bufAux, bufBase, bufAcc;
  Buffer bufTrigW, bufTrigH;
  Buffer bufA, bufI;
  Buffer buf1, buf2, buf3;
  Buffer bufCarry;
  Buffer bufReady;
  Buffer bufSmallOut;
  Buffer bufBaseDown;

  std::vector<u32> computeBase(u32 E, u32 B1);
  std::pair<std::vector<u32>, std::vector<u32>> seedPRP(u32 E, u32 B1);
  
  std::vector<int> readSmall(Buffer &buf, u32 start);

  void tW(Buffer &in, Buffer &out);
  void tH(Buffer &in, Buffer &out);
  void exitKerns(Buffer &buf, Buffer &bufWords);
  
  void copyFromTo(Buffer &from, Buffer &to);
  
  std::vector<int> readOut(Buffer &buf);
  void writeIn(const std::vector<u32> &words, Buffer &buf);
  void writeIn(const std::vector<int> &words, Buffer &buf);
  
  void modSqLoopMul(Buffer &io, const std::vector<bool> &muls);
  void modSqLoopAcc(Buffer &io, const std::vector<bool> &muls);
  
  void modMul(Buffer &in, Buffer &io);
  bool equalNotZero(Buffer &bufCheck, Buffer &bufAux);
  u64 bufResidue(Buffer &buf);
  
  std::vector<u32> writeBase(const std::vector<u32> &v);

  PRPState loadPRP(u32 E, u32 iniB1, u32 iniBlockSize);
  void doStage0(u32 k, u32 B1, u32 blockSize, std::vector<u32> &&base, std::vector<bool> &&basePower);
  
public:
  static std::unique_ptr<Gpu> make(u32 E, const Args &args);
  
  Gpu(u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
      cl_program program, cl_device_id device, cl_context context,
      bool timeKernels, bool useLongCarry);

  ~Gpu();
  
  void writeState(const std::vector<u32> &check, const std::vector<u32> &base, const std::vector<u32> &gcdAcc, u32 blockSize);
  
  std::vector<u32> roundtripData()  { return writeData(readData()); }
  std::vector<u32> roundtripCheck() { return writeCheck(readCheck()); }

  std::vector<u32> writeData(const std::vector<u32> &v);
  std::vector<u32> writeCheck(const std::vector<u32> &v);
  
  u64 dataResidue()  { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
    
  bool doCheck(int blockSize);
  void updateCheck();

  void dataLoopMul(const std::vector<bool> &muls) { modSqLoopMul(bufData, muls); }
  void dataLoopAcc(const std::vector<bool> &accs) { modSqLoopAcc(bufData, accs); }
  u32 dataLoopAcc(u32 begin, u32 end, const std::vector<bool> &kset);
  
  void finish();

  void logTimeKernels();

  std::vector<u32> readCheck();
  std::vector<u32> readData();
  std::vector<u32> readAcc();

  PRPResult isPrimePRP(u32 E, const Args &args, u32 B1, u32 B2);
  u32 getFFTSize() { return N; }
};
