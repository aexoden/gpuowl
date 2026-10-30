// Copyright (C) Jason Lynch

// Failures worth a person's look: a configuration that took the device down, computed a wrong answer, or would not
// build.  Each is appended as an entry to failures/failures.log -- what was being done, what was seen, and the command
// line that reproduces it -- beside whatever evidence was saved with it (fs.h failurePath: a build log, the source a
// compiler refused).  A run says at its start and end how many configurations the env holds out, and -tune status
// does too, so the log is found rather than happened upon.

#pragma once

#include "common.h"
#include "fs.h"
#include "TestKind.h"
#include "TuneDB.h"

#include <filesystem>
#include <set>
#include <string>

namespace tune {

inline constexpr const char* FAILURES_LOG = "failures.log";

enum class FailureKind : u8 { DeviceLost, Wrong, NoBuild };

// "took the device down", "computed a wrong answer", "would not build"
[[nodiscard]] const char* toString(FailureKind kind);

struct FailureEntry {
  FailureKind kind = FailureKind::Wrong;
  std::string fft{};
  TestKind test = TestKind::PRP;
  u64 exponent = 0;
  std::string options{};  // configText(): "-", or empty, for the built-in defaults
  std::string device{};   // the card, as the env names it
  std::string detail{};   // what was seen
  u64 at = 0;             // when it happened, seconds since the epoch; 0 for now
  std::string id{};       // what the entry is of, where it must be written once: "attempt <session> <ts>" for a death
};

// The entry as failures.log holds it: a heading line with the time and what happened, then indented lines.
[[nodiscard]] std::string failureText(const FailureEntry& entry);

// Appends the entry to `dir`/failures.log, making `dir` where it is missing; false where it could not be written.
bool recordFailure(const FailureEntry& entry, const std::filesystem::path& dir = FAILURES_DIR);

// The ids the log in `dir` already holds.
[[nodiscard]] std::set<std::string> recordedFailureIds(const std::filesystem::path& dir = FAILURES_DIR);

// The id of the entry for a failed call, and for an attempt a process died holding.
[[nodiscard]] std::string failureId(const CallRow& call);
[[nodiscard]] std::string failureId(const TryRow& attempt);

// An entry for each failure `env` records that the log lacks -- an attempt a process died holding (TuneDB::deaths()), a
// call that computed wrongly or would not build -- so that the log is whole: a hard fault leaves no process to write
// one, and a database can predate the log.  How many it wrote.
u32 recordMissingFailures(const TuneDB& db, u32 env, const std::filesystem::path& dir = FAILURES_DIR);

// The configurations `env` holds out, by why: each counted once, by FFT, test kind and settings.
struct FailureTally {
  u32 deviceLost = 0;
  u32 wrong = 0;
  u32 noBuild = 0;

  [[nodiscard]] u32 total() const { return deviceLost + wrong + noBuild; }
};
[[nodiscard]] FailureTally failureTally(const TuneDB& db, u32 env);

// "3 configurations held out: 1 took the device down, 2 computed a wrong answer; each is in failures/failures.log",
// or empty where there are none.
[[nodiscard]] std::string toString(const FailureTally& tally);

}  // namespace tune
