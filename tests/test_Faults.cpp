// Copyright (C) Jason Lynch

// Tests which configurations an env holds out as faults -- a death an attempt was left standing for, a row that went
// away with the device, a wrong answer -- and how each is said so that it can be reproduced.

#include "Faults.h"

#include "test.h"

#include <string>
#include <vector>

using namespace tune;

namespace {

// Env 1: session 4 died holding cfg 17 at 143400073; session 5 answered its attempt (a written file says so with a
// `done` after the `try`); session 6 recorded a wrong answer twice (in two regimes) and a call the device went away
// during.  Env 2 died holding one of its own.
constexpr const char* DB = "# prpll tunedb v1\n"
                           "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia"
                           " be=ocl cc=806 noasm=0 pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
                           "env   2 gpu=\"Tesla P100-PCIE-16GB\" name=\"Tesla P100-PCIE-16GB\" drv=550.163.01"
                           " vendor=nvidia be=ocl cc=600 noasm=0 pdl=0 fp64=1 builtins=1 machine=4d:00.0"
                           " build=9a3f21c0d1e2f304\n"
                           "cfg   1 -\n"
                           "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
                           "cfg   18 LOADS=1\n"
                           "sess  4 env=1 start=1753471200 gen=0 anchor=-\n"
                           "sess  5 env=1 start=1753481200 gen=1 anchor=-\n"
                           "sess  6 env=1 start=1753491200 gen=0 anchor=-\n"
                           "sess  7 env=2 start=1753491300 gen=0 anchor=-\n"
                           "run   5 512:15:512:212 prp 143400073 short32 1 1774.230 2.100 24 6 1.0000 ok 1753481300\n"
                           "run   6 256:14:512:212 prp 68000033 short32 18 1024.000 1.000 8 2 1.0000 err 1753491400\n"
                           "run   6 256:14:512:212 prp 68000033 long32 18 1030.000 1.000 8 2 1.0000 err 1753491410\n"
                           "run   6 1K:8:1K:112 ll 118063003 short32 1 900.000 1.000 8 1 1.0000 lost 1753491500\n"
                           "try   4 512:15:512:212 prp 143400073 17 1753471250\n"
                           "try   5 512:15:512:212 prp 143400073 1 1753481250\n"
                           "done  5 1753481300\n"
                           "try   7 512:15:512:212 prp 143400073 1 1753491350\n";

TuneDB loaded() {
  TuneDB db;
  CHECK(db.parse(DB, "fixture"));
  return db;
}

}  // namespace

TEST(an_env_holds_out_what_took_its_device_down_and_what_answered_wrongly_oldest_first) {
  TuneDB const db = loaded();
  std::vector<Fault> const faults = faultsOf(db, 1);
  CHECK_EQ(faults.size(), size_t{3});
  if (faults.size() != 3) { return; }

  CHECK(faults[0].what == Fault::What::Lost);
  CHECK_EQ(faults[0].fft, std::string{"512:15:512:212"});
  CHECK_EQ(faults[0].options, std::string{"INPLACE=1,PAD=256,TAIL_KERNELS=3"});
  CHECK_EQ(faults[0].ts, u64{1'753'471'250});
  CHECK_EQ(faults[0].gen, 0u);

  // One wrong answer in two regimes is one configuration to report.
  CHECK(faults[1].what == Fault::What::Wrong);
  CHECK_EQ(faults[1].fft, std::string{"256:14:512:212"});
  CHECK_EQ(faults[1].options, std::string{"LOADS=1"});

  CHECK(faults[2].what == Fault::What::Lost);
  CHECK(faults[2].kind == TestKind::LL);
  CHECK_EQ(faults[2].options, std::string{"-"});

  // Another env's faults are a fact about another card.
  std::vector<Fault> const other = faultsOf(db, 2);
  CHECK_EQ(other.size(), size_t{1});
  CHECK(faultsOf(db, 3).empty());
}

TEST(a_fault_is_said_as_the_command_line_that_reproduces_it) {
  TuneDB const db = loaded();
  std::vector<Fault> const faults = faultsOf(db, 1);
  CHECK_EQ(faults.size(), size_t{3});
  if (faults.size() != 3) { return; }
  CHECK_EQ(reproduce(faults[0]),
           std::string{"-fft 512:15:512:212 -use INPLACE=1,PAD=256,TAIL_KERNELS=3   (prp at "
                       "143400073)"});
  CHECK_EQ(reproduce(faults[2]), std::string{"-fft 1K:8:1K:112   (ll at 118063003)"});

  std::vector<std::string> const lines = faultLines(faults);
  CHECK_EQ(lines.size(), size_t{5});
  if (lines.size() != 5) { return; }
  CHECK(lines[0].starts_with("2 configurations took the device down"));
  CHECK_EQ(lines[1], "  " + reproduce(faults[0]));
  CHECK_EQ(lines[2], "  " + reproduce(faults[2]));
  CHECK(lines[3].starts_with("1 configuration computed a wrong answer"));
  CHECK_EQ(lines[4], "  " + reproduce(faults[1]));

  CHECK(faultLines({}).empty());
}
