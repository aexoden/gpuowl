// Copyright (C) Jason Lynch

// Where a tuning database stands, asked from outside any run: what is measured and what is not, what a run started
// now would measure next and what that is worth, what the accuracy gate has made of what is published, and which
// configurations a device fault holds out.  Read without the database's lock, so that it can be asked beside a run
// that holds it.
//
// A pure function of the database, of a queue built as a run builds its own, and of what the caller saw of the file,
// so that every figure is one the queue itself computes: the family counts are the run summary's, and the items are
// ranked by admissible().

#pragma once

#include "common.h"
#include "Scheduler.h"
#include "Summary.h"
#include "TuneDB.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// How many of the items a run would take next are named.
inline constexpr size_t STATUS_NEXT = 5;

// What the caller saw of the database file, which the database itself does not say.
struct Activity {
  // Whether a process holds the right to write it; nothing where that cannot be told.
  std::optional<bool> held;

  // When it was last written to, and now, in seconds since the epoch.
  u64 written = 0;
  u64 now = 0;
};

// One attempt: a configuration at one exponent, as its `try` row declared it.
struct Attempt {
  TryRow row;
  std::string options;
};

struct TuneStatus {
  Activity activity;

  // The latest session on the env, and what the process holding the database is measuring: the attempt the newest
  // session in the file has open, which is that process's own.
  std::optional<SessRow> latest;
  std::optional<Attempt> measuring;

  // T over what is published, and the share of the weight on measured entries.
  double T = 0;
  double measured = 0;

  // The T items are valued against, which counts the sets the gate still owes a reading, and so what an item has to
  // be worth to run.
  double valuedT = 0;
  double stop = 0;
  double floor = 0;

  std::vector<RunSummary::Family> families;
  std::string heldBy;

  struct Next {
    Item item;
    std::string label;
  };
  // What a run would take, best rate first, as far as STATUS_NEXT; then how many more it would run, and how many it
  // would not.
  std::vector<Next> next;
  u32 moreWorth = 0;
  u32 notWorth = 0;

  // Past those, at most how many more points worth running the stages listed in part have.
  u64 unlisted = 0;

  // The entries the selection file publishes, by what the gate made of them, and the sets it still owes a reading or
  // has refused.
  struct Accuracy {
    u32 entries = 0;
    u32 exact = 0;
    u32 confirmed = 0;
    u32 unvalidated = 0;
    u32 belowTable = 0;
    u32 aboveTable = 0;
    u32 owed = 0;
    u32 rejected = 0;
  };
  Accuracy accuracy;

  // Attempts a process on the env died holding, which no run makes again.
  std::vector<Attempt> faults;
};

// A database file's text up to the end of its last complete line: a run appends one line at a time, and a reader can
// arrive part-way through one.
[[nodiscard]] std::string_view completeLines(std::string_view text);

// A span of seconds in whole seconds, minutes, hours or days.
[[nodiscard]] std::string ago(u64 seconds);

// Where `env` stands in `db`, against `stop`, as `scheduler` ranks it; built as a run builds its own, a fresh one ranks
// what a run started now would take first.  An unattached `db` has the attempt the holder is measuring answered in it,
// which is how the holder will leave it.
[[nodiscard]] TuneStatus statusOf(const Scheduler& scheduler, TuneDB& db, u32 env, double stop,
                                  const Activity& activity);

// `valuedAs` says whose settings the items were valued with.
void logStatus(const TuneStatus& status, const DbEnv& env, const std::string& valuedAs);

}  // namespace tune
