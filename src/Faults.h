// Copyright (C) Jason Lynch

// The configurations an env holds out because they did something no configuration should: took the device down, or
// computed a wrong answer.  Either is a bug in a kernel or a driver, not a slow configuration, so a run names every one
// of them where it starts and where it ends, and a dashboard keeps them on screen, each as the command line that
// reproduces it.

#pragma once

#include "common.h"
#include "TuneDB.h"

#include <string>
#include <vector>

namespace tune {

struct Fault {
  enum class What : u8 {
    Lost,   // a process died holding the attempt, or the device went away during the call
    Wrong,  // a wrong residue or a failed Gerbicz check
  };
  What what = What::Lost;

  std::string fft;
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;

  // The option set as the attempt or the row names it, "-" for none, "?" where the database does not declare it.
  std::string options;

  // When it happened, 0 where the database does not say, and the restart generation of the session it happened in.
  u64 ts = 0;
  u32 gen = 0;
};

// Every fault on `env`, each configuration and exponent once, oldest first.  An attempt a live session of this process
// holds is not one: it is the call in flight.
[[nodiscard]] std::vector<Fault> faultsOf(const TuneDB& db, u32 env);

// "-fft <spec> -use <options>   (<kind> at <exponent>)"
[[nodiscard]] std::string reproduce(const Fault& fault);

// "took the device down" or "computed a wrong answer"
[[nodiscard]] const char* faultText(Fault::What what);

// The faults as the log lists them, each kind under a line saying what it means, prefixed with `prefix`; nothing where
// there are none.
[[nodiscard]] std::vector<std::string> faultLines(const std::vector<Fault>& faults);

void logFaults(const std::vector<Fault>& faults, const char* prefix);

}  // namespace tune
