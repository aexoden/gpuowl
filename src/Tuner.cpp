// Copyright (C) Jason Lynch

#include "Tuner.h"

#include "BuildId.h"
#include "Emit.h"
#include "log.h"
#include "TuneDB.h"

#include <cstdio>
#include <ctime>
#include <vector>

namespace tune {

namespace {

constexpr const char* SELECTION_NAME = "selection.txt";

[[nodiscard]] u32 asEnvId(std::string_view text, const char* what) {
  std::optional<u32> const id = parseInt<u32>(text);
  if (!id || *id == 0) { throw "-tune: " + std::string{what} + " takes the number of an env in the database"; }
  return *id;
}

[[nodiscard]] std::string hex16(u64 value) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return buf;
}

// Enough to tell one env from another by eye: which card it is, where that card sits, and which kernels its rows were
// measured against.
[[nodiscard]] std::string describe(const DbEnv& env) {
  return "  env " + std::to_string(env.id) + ": " + (env.gpu.empty() ? "unnamed device" : env.gpu) +
    (env.machine.empty() ? "" : " at " + env.machine) + ", " + (env.cudaBackend ? "cuda" : "ocl") + ", kernels " +
    hex16(env.build) + "\n";
}

void listEnvs(const std::vector<const DbEnv*>& envs) {
  for (const DbEnv* const env : envs) { log("%s", describe(*env).c_str()); }
}

}  // namespace

const char* toString(DbVerb verb) {
  switch (verb) {
  case DbVerb::Emit: return "emit";
  case DbVerb::Reset: return "reset";
  case DbVerb::Adopt: return "adopt";
  case DbVerb::Compact: return "compact";
  }
  return "?";
}

std::optional<DbCommand> parseDbCommand(std::string_view text) {
  size_t const firstComma = text.find(',');
  std::string_view const verb = text.substr(0, firstComma);

  DbCommand out;
  if (verb == "emit") {
    out.verb = DbVerb::Emit;
  } else if (verb == "reset") {
    out.verb = DbVerb::Reset;
  } else if (verb == "adopt") {
    out.verb = DbVerb::Adopt;
  } else if (verb == "compact") {
    out.verb = DbVerb::Compact;
  } else {
    return {};
  }

  for (size_t at = firstComma == std::string_view::npos ? text.size() + 1 : firstComma + 1; at <= text.size();) {
    size_t const comma = text.find(',', at);
    std::string_view const token = text.substr(at, comma == std::string_view::npos ? comma : comma - at);
    at = comma == std::string_view::npos ? text.size() + 1 : comma + 1;

    size_t const eq = token.find('=');
    if (eq == std::string_view::npos) {
      throw "-tune " + std::string{verb} + ": '" + std::string{token} + "' is not a <key>=<value> setting";
    }

    std::string_view const key = token.substr(0, eq);
    std::string_view const val = token.substr(eq + 1);

    bool const wantsEnv = out.verb != DbVerb::Compact;

    if (key == "env" && wantsEnv) {
      out.env = asEnvId(val, "env=");
    } else if (key == "into" && out.verb == DbVerb::Adopt) {
      out.env = asEnvId(val, "into=");
    } else if (key == "from" && out.verb == DbVerb::Adopt) {
      out.from = asEnvId(val, "from=");
    } else if (key == "fft" && out.verb == DbVerb::Reset) {
      if (val.empty()) { throw std::string{"-tune reset: fft= takes an FFT specification"}; }
      out.fft = std::string{val};
    } else {
      std::string accepted = "nothing";
      switch (out.verb) {
      case DbVerb::Emit: accepted = "env=<id>"; break;
      case DbVerb::Reset: accepted = "env=<id>, fft=<spec>"; break;
      case DbVerb::Adopt: accepted = "into=<id> (or env=<id>), from=<id>"; break;
      case DbVerb::Compact: break;
      }
      throw "-tune " + std::string{verb} + ": '" + std::string{key} + "=' is not understood. Accepted: " + accepted;
    }
  }

  return out;
}

u32 commandEnv(const TuneDB& db, const DbCommand& command, u64 build) {
  if (command.env) {
    if (!db.findEnv(command.env)) {
      log("tune: there is no env %u in the database\n", command.env);
      return 0;
    }
    return command.env;
  }

  std::vector<const DbEnv*> matching;
  std::vector<const DbEnv*> all;
  for (const DbEnv& env : db.envs()) {
    all.push_back(&env);
    if (env.build == build) { matching.push_back(&env); }
  }

  if (matching.size() == 1) { return matching.front()->id; }

  if (matching.empty()) {
    log("tune: nothing in the database was measured against the kernels this binary carries (%s), so there is no env"
        " to work on; name one with env=<id>, or adopt it into one\n",
        hex16(build).c_str());
    listEnvs(all);
  } else {
    log("tune: %zu envs were measured against the kernels this binary carries, and nothing here opens a device to"
        " tell their cards apart; name one with env=<id>\n",
        matching.size());
    listEnvs(matching);
  }

  return 0;
}

bool rewriteFor(TuneDB& db, const DbCommand& command, u32 env) {
  switch (command.verb) {
  case DbVerb::Emit: return true;

  case DbVerb::Compact: return db.compact();

  case DbVerb::Reset: return db.reset(env, command.fft);

  case DbVerb::Adopt: {
    u32 const from = command.from ? command.from : db.adoptCandidate(env);
    if (!from) {
      log("tune: env %u has no earlier env of the same card to adopt; name one with from=<id>\n", env);
      return false;
    }
    if (from == env) {
      log("tune: env %u cannot adopt itself\n", env);
      return false;
    }
    if (!db.adopt(from, env)) { return false; }

    log("tune: env %u has taken over the rows of env %u\n", env, from);
    return true;
  }
  }

  return false;
}

bool runDbCommand(const DbCommand& command, const fs::path& dir) {
  fs::path const dbPath = dir / TuneDB::DEFAULT_NAME;

  TuneDB db;
  // Before the load and held for the rest of the command: a session appending beside this would be writing rows into
  // a file about to be rewritten from what was read here.
  if (!db.lockForWriting(dbPath)) { return false; }
  if (!db.load(dbPath)) { return false; }

  u32 env = 0;
  if (command.verb != DbVerb::Compact) {
    env = commandEnv(db, command, buildFingerprint());
    if (!env) { return false; }
  }

  if (command.verb == DbVerb::Emit) {
    Provenance const from{.ts = u64(time(nullptr)), .db = TuneDB::DEFAULT_NAME, .env = env};

    std::optional<SelectionFile> const file = emit(db, {}, from);
    if (!file) {
      log("tune: emit: the database gave nothing that could be published\n");
      return false;
    }

    fs::path const out = dir / SELECTION_NAME;
    writeSelection(out, *file);
    log("tune: published %zu %s of env %u to %s\n", file->entries.size(),
        file->entries.size() == 1 ? "entry" : "entries", env, out.string().c_str());
    return true;
  }

  if (!rewriteFor(db, command, env)) { return false; }

  db.save(dbPath);
  log("tune: %s rewrote %s\n", toString(command.verb), dbPath.string().c_str());
  return true;
}

}  // namespace tune
