// Copyright (C) Jason Lynch

// Tests the database-only -tune subcommands: that the grammar reads what it should and refuses what it should, that
// an env is chosen only where the choice is unambiguous, that each command changes the database the way it says, and
// that all four run against files alone -- no device is opened anywhere below.

#include "Tuner.h"

#include "Selection.h"
#include "TuneDB.h"

#include "test.h"

#include <cstdio>
#include <string>

using namespace tune;

namespace {

constexpr u64 BUILD = 0x9a3f21c0d1e2f304;
constexpr u64 OTHER_BUILD = 0x1111222233334444;

// Two cards under one set of kernels, and the first card again under an earlier set.  Env 1 has two readings of one
// configuration -- which is what `compact` folds -- and one of another shape.
const char* const DB =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"Tesla P100-PCIE-16GB\" name=\"Tesla P100-PCIE-16GB\" drv=550.163.01 vendor=nvidia be=ocl cc=600"
  " noasm=0 pdl=0 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
  "env   3 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 machine=01:00.0 build=1111222233334444\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "cfg   18 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  9 env=3 start=1753471100 gen=0 anchor=512:15:512:212@100000000\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 12 3 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1776.000 2.000 12 3 1.0000 ok 1753471284\n"
  "run   4 1K:8:1K:202 prp 200000000 short32 18 3100.000 4.000 16 4 1.0000 ok 1753471354\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 1900.000 2.000 16 4 1.0000 ok 1753471204\n";

TuneDB loaded(const char* text = DB) {
  TuneDB db;
  CHECK(db.parse(text, "fixture"));
  return db;
}

DbCommand parsed(const char* text) {
  std::optional<DbCommand> const command = parseDbCommand(text);
  CHECK(command.has_value());
  return *command;
}

// The message a malformed subcommand is refused with, or "" where it was accepted.
std::string refusal(const char* text) {
  try {
    (void)parseDbCommand(text);
  } catch (const std::string& mes) { return mes; } catch (const char* mes) {
    return mes;
  }
  return {};
}

size_t runsOn(const TuneDB& db, u32 env) {
  size_t n = 0;
  for (const RunRow& row : db.runs()) {
    if (db.envOf(row.sess) == env) { ++n; }
  }
  return n;
}

// A directory of its own for each command that touches files, since the database is addressed by name within it.
struct Dir {
  fs::path path;

  explicit Dir(const char* name) : path{fs::temp_directory_path() / name} {
    fs::remove_all(path);
    fs::create_directories(path);
  }

  ~Dir() { fs::remove_all(path); }

  Dir(const Dir&) = delete;
  Dir& operator=(const Dir&) = delete;

  void write(const char* name, const std::string& text) const {
    FILE* const f = fopen((path / name).string().c_str(), "wb");
    CHECK(f != nullptr);
    CHECK(fwrite(text.data(), 1, text.size(), f) == text.size());
    CHECK(fclose(f) == 0);
  }

  [[nodiscard]] bool has(const char* name) const { return fs::exists(path / name); }
};

}  // namespace

TEST(a_subcommand_of_another_tuner_is_not_one_of_these) {
  // Upstream's own -tune takes the same flag, and none of its option words is a subcommand here.
  CHECK(!parseDbCommand("").has_value());
  CHECK(!parseDbCommand("noconfig,fp64").has_value());
  CHECK(!parseDbCommand("quick=5").has_value());
  CHECK(!parseDbCommand("emitter").has_value());
}

TEST(each_subcommand_reads_its_own_settings) {
  CHECK(parsed("emit").verb == DbVerb::Emit);
  CHECK_EQ(parsed("emit").env, 0u);
  CHECK_EQ(parsed("emit,env=7").env, 7u);

  CHECK(parsed("compact").verb == DbVerb::Compact);

  CHECK(parsed("reset,env=2,fft=1K:8:1K:202").verb == DbVerb::Reset);
  CHECK_EQ(parsed("reset,env=2,fft=1K:8:1K:202").env, 2u);
  CHECK_EQ(parsed("reset,fft=1K:8:1K:202").fft, std::string{"1K:8:1K:202"});

  CHECK(parsed("adopt,from=3,into=1").verb == DbVerb::Adopt);
  CHECK_EQ(parsed("adopt,from=3,into=1").from, 3u);
  CHECK_EQ(parsed("adopt,from=3,into=1").env, 1u);
  CHECK_EQ(parsed("adopt,env=1").env, 1u);
}

TEST(a_setting_that_belongs_to_another_subcommand_is_a_usage_error) {
  CHECK(!refusal("emit,fft=1K:8:1K:202").empty());
  CHECK(!refusal("emit,from=2").empty());
  CHECK(!refusal("compact,env=1").empty());
  CHECK(!refusal("reset,from=2").empty());
  CHECK(!refusal("emit,env=").empty());
  CHECK(!refusal("emit,env=0").empty());
  CHECK(!refusal("emit,env=two").empty());
  CHECK(!refusal("reset,fft=").empty());
  CHECK(!refusal("emit,verbose").empty());
  CHECK(refusal("adopt,from=3,into=1").empty());
}

TEST(the_env_is_the_one_these_kernels_measured) {
  TuneDB const db = loaded();

  // Envs 1 and 2 carry BUILD, so only naming one resolves it; env 3 is the sole holder of OTHER_BUILD.
  CHECK_EQ(commandEnv(db, parsed("emit"), BUILD), 0u);
  CHECK_EQ(commandEnv(db, parsed("emit"), OTHER_BUILD), 3u);
  CHECK_EQ(commandEnv(db, parsed("emit,env=2"), BUILD), 2u);

  // A build nothing was measured against has no env of its own, and a named env that is not there is not one either.
  CHECK_EQ(commandEnv(db, parsed("emit"), 0xdeadbeef), 0u);
  CHECK_EQ(commandEnv(db, parsed("emit,env=9"), BUILD), 0u);
}

TEST(compact_folds_the_duplicate_readings) {
  TuneDB db = loaded();
  CHECK_EQ(db.runs().size(), 4u);

  CHECK(rewriteFor(db, parsed("compact"), 0));

  // The two readings of 512:15:512:212 under cfg 17 on env 1 are one row now; the other two stand.
  CHECK_EQ(db.runs().size(), 3u);
  CHECK_EQ(runsOn(db, 1), 2u);
  CHECK_EQ(runsOn(db, 3), 1u);
}

TEST(reset_drops_an_env_or_one_shape_of_it) {
  {
    TuneDB db = loaded();
    CHECK(rewriteFor(db, parsed("reset,env=1"), 1));
    CHECK_EQ(runsOn(db, 1), 0u);
    CHECK_EQ(runsOn(db, 3), 1u);
    // The env and its session stay, so it keeps the anchor it is pinned to.
    CHECK(db.findEnv(1) != nullptr);
    CHECK_EQ(db.envAnchor(1), std::string{"512:15:512:212@100000000"});
  }

  {
    TuneDB db = loaded();
    CHECK(rewriteFor(db, parsed("reset,env=1,fft=1K:8:1K:202"), 1));
    CHECK_EQ(runsOn(db, 1), 2u);
  }

  {
    TuneDB db = loaded();
    CHECK(!rewriteFor(db, parsed("reset,env=1,fft=not-a-spec"), 1));
    CHECK_EQ(runsOn(db, 1), 3u);
  }
}

TEST(adopt_takes_the_earlier_env_of_the_same_card) {
  TuneDB db = loaded();

  // Env 3 is the A4000 under other kernels, so it is what env 1 adopts when nothing is named.  Env 2 is a different
  // card and is never a candidate.
  CHECK_EQ(db.adoptCandidate(1), 3u);
  CHECK(rewriteFor(db, parsed("adopt"), 1));
  CHECK_EQ(runsOn(db, 1), 4u);
  CHECK(db.findEnv(3) == nullptr);
}

TEST(adopt_refuses_what_was_never_this_card) {
  {
    TuneDB db = loaded();
    CHECK(!rewriteFor(db, parsed("adopt,from=2"), 1));
    CHECK_EQ(runsOn(db, 1), 3u);
  }

  {
    TuneDB db = loaded();
    CHECK(!rewriteFor(db, parsed("adopt,from=1"), 1));
  }

  {
    // Env 2 is the only P100 in the database, so there is nothing for it to adopt.
    TuneDB db = loaded();
    CHECK(!rewriteFor(db, parsed("adopt"), 2));
  }
}

TEST(emit_publishes_the_selection_file_beside_the_database) {
  Dir const dir{"prpll-test-tuner-emit"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  // Env 3's rows are the only ones a binary carrying OTHER_BUILD can see, and the one env carrying it needs no naming
  // -- but this test cannot choose the fingerprint the binary was built with, so it names the env instead.
  CHECK(runDbCommand(parsed("emit,env=1"), dir.path));
  CHECK(dir.has("selection.txt"));

  std::optional<SelectionFile> const file = readSelection(dir.path / "selection.txt");
  CHECK(file.has_value());
  CHECK_EQ(file->entries.size(), 2u);
  CHECK_EQ(file->entries.front().fft, std::string{"512:15:512:212"});
  CHECK_EQ(configText(file->entries.front().opts), std::string{"INPLACE=1,PAD=256,TAIL_KERNELS=3"});

  // The provenance says which database and env it came from, and carries no workload: nothing here weighed one.
  CHECK(file->provenance.find("from tunedb.txt env 1") != std::string::npos);
  CHECK(file->provenance.find("workload") == std::string::npos);

  // Nothing it read was changed.
  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.runs().size(), 4u);
}

TEST(a_command_that_rewrites_writes_the_database_back) {
  Dir const dir{"prpll-test-tuner-compact"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  CHECK(runDbCommand(parsed("compact"), dir.path));

  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.runs().size(), 3u);
  CHECK(!dir.has("selection.txt"));
}

TEST(a_command_says_so_rather_than_writing_anything_it_cannot) {
  Dir const dir{"prpll-test-tuner-refused"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  CHECK(!runDbCommand(parsed("emit,env=9"), dir.path));
  CHECK(!dir.has("selection.txt"));

  CHECK(!runDbCommand(parsed("reset,env=2,fft=not-a-spec"), dir.path));

  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.runs().size(), 4u);
}

TEST(a_database_that_is_not_there_is_an_empty_one) {
  Dir const dir{"prpll-test-tuner-absent"};

  // Nothing was measured, so there is no env to work on and nothing is published -- but the command reads a missing
  // file as empty rather than as a failure to read.
  CHECK(!runDbCommand(parsed("emit"), dir.path));
  CHECK(!dir.has("selection.txt"));

  // `compact` needs no env, so an empty database is one it can rewrite.
  CHECK(runDbCommand(parsed("compact"), dir.path));
  CHECK(dir.has(TuneDB::DEFAULT_NAME));
}
