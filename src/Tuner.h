// Copyright (C) Jason Lynch

// The -tune command line.
//
// Four of its subcommands read and rewrite the measurement database and nothing else: they open no device, build no
// kernels and take no readings, so they run wherever the database is rather than only on the card it was measured on.

#pragma once

#include "common.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace tune {

class TuneDB;

enum class DbVerb : u8 {
  Emit,     // publish the selection file the database supports
  Reset,    // drop what was measured on an env, or on one shape of it
  Adopt,    // restamp another env's rows as this one's
  Compact,  // fold duplicate rows and drop the option sets nothing names
};

[[nodiscard]] const char* toString(DbVerb verb);

struct DbCommand {
  DbVerb verb = DbVerb::Emit;

  // The env the command acts on; 0 to take the one this binary's kernels were measured on.
  u32 env = 0;

  // `adopt` only: the env whose rows are being taken over.  0 to take the most recent env that is this card under
  // other kernels.
  u32 from = 0;

  // `reset` only: one shape of the env rather than all of it.
  std::string fft;
};

// The database-only subcommand `text` asks for, or nothing where it asks for something else -- upstream's own tuner
// takes the same flag, and its option words are not these.  Throws a message for a subcommand it recognises and then
// cannot read, so a mistyped setting is a usage error rather than a silent fall-through to the other tuner.
[[nodiscard]] std::optional<DbCommand> parseDbCommand(std::string_view text);

// Which env the command runs on: the one it names, or the single env whose rows were measured against `build`.  0
// where there is no such env or more than one, having said which envs there are -- with no device open there is
// nothing here that can tell two cards apart, and publishing the wrong card's measurements is worse than asking.
[[nodiscard]] u32 commandEnv(const TuneDB& db, const DbCommand& command, u64 build);

// Everything `command` changes about a loaded database, `emit` aside -- that one reads.  False with the reason logged.
[[nodiscard]] bool rewriteFor(TuneDB& db, const DbCommand& command, u32 env);

// Runs one database-only subcommand against the database in `dir`, reporting whether it did what it was asked.
[[nodiscard]] bool runDbCommand(const DbCommand& command, const fs::path& dir);

}  // namespace tune
