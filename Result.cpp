// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Result.h"

#include "Task.h"
#include "args.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cassert>
#include <string>

static bool writeResult(const std::string &part, u32 E, const char *workType, const std::string &status,
                        const std::string &AID, const std::string &user, const std::string &cpu) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  
  char buf[512];
  snprintf(buf, sizeof(buf), "{\"exponent\":\"%u\", \"worktype\":\"%s\", \"status\":\"%s\", "
           "\"program\":{\"name\":\"%s\", \"version\":\"%s\"}, \"timestamp\":\"%s\"%s%s%s}",
           E, workType, status.c_str(), PROGRAM, VERSION, timeStr().c_str(), uid.c_str(), aidJson.c_str(), part.c_str());
  
  log("%s\n", buf);
  auto fo = openAppend("results.txt");
  if (!fo) { return false; }

  fprintf(fo.get(), "%s\n", buf);
  return true;
}

static std::string factorStr(const std::string &factor) { return factor.empty() ? "" : (", \"factors\":[\"" + factor + "\"]"); }

static std::string resStr(u64 res64) {
  char buf[64];
  snprintf(buf, sizeof(buf), ", \"res64\":\"%016llx\"", res64);
  return buf;
}

bool PRPResult::write(const Args &args, const Task &task, u32 fftSize) {
  u32 B1 = task.B1;
  
  bool hasFactor = !factor.empty();
  if (hasFactor) {
    assert(!isPrime);
    assert(res64 == 0);
    assert(B1 != 0);
  }

  std::string status = isPrime ? "P" : (hasFactor ? "F" : "C");
  std::string fftLength = std::string(", \"fft-length\":") + std::to_string(fftSize);
  
  char buf[256];
  if (B1 == 0) {
    assert(task.B2 == 0);
    assert(baseRes64 == 3);
    std::string str = resStr(res64) + ", \"residue-type\":4";
    return writeResult(fftLength + str, task.exponent, "PRP-3", status, task.AID, args.user, args.cpu);
  }
  
  std::string r1 = hasFactor ? "" : resStr(res64);
  std::string r2 = resStr(baseRes64);

  // When B1!=0: B2==0 means "use default B2", which is ==Exponent.
  // If B2!=0, the reported B2 won't be larger then the Exponent in any case.
  u32 E = task.exponent;
  // u32 B2 = (task.B2==0) ? E : min(E, task.B2);
  
  snprintf(buf, sizeof(buf), "%s%s, \"b2\":\"%u\", \"base\":{\"b1\":\"%u\", \"bias\":{\"2\":19}%s}",
           factorStr(factor).c_str(), r1.c_str(),
           B2, B1, r2.c_str());
  return writeResult(fftLength + buf, E, "PRP,P-1", status, task.AID, args.user, args.cpu);
}
