// Copyright (C) Jason Lynch

// Tests the device-free -tune subcommands: that the grammar reads what it should and refuses what it should, that an
// env is chosen only where the choice is unambiguous, that each command changes the database the way it says, and
// that the scope derived from a worktodo is the one the run would work within.  No device is opened anywhere below.

#include "Tuner.h"

#include "Args.h"
#include "BuildId.h"
#include "Emit.h"
#include "File.h"
#include "Objective.h"
#include "Selection.h"
#include "TuneDB.h"

#include "test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tune;

namespace {

constexpr u64 BUILD = 0x9a3f21c0d1e2f304;
constexpr u64 OTHER_BUILD = 0x1111222233334444;

// Two cards under one set of kernels, and the first card again under an earlier set.  Env 1 has two readings of one
// configuration -- which is what `compact` folds -- and one of another shape.
const char* const DB =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"Tesla P100-PCIE-16GB\" name=\"Tesla P100-PCIE-16GB\" drv=550.163.01 vendor=nvidia be=ocl cc=600"
  " noasm=0 pdl=0 fp64=1 builtins=1 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
  "env   3 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=1111222233334444\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "cfg   18 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  9 env=3 start=1753471100 gen=0 anchor=512:15:512:212@100000000\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 12 3 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1776.000 2.000 12 3 1.0000 ok 1753471284\n"
  "run   4 1K:8:1K:202 prp 200000000 short32 18 3100.000 4.000 16 4 1.0000 ok 1753471354\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 1900.000 2.000 16 4 1.0000 ok 1753471204\n"
  "roe   4 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471294\n"
  "roe   4 1K:8:1K:202 296960407 18 25.10 2150 0.3021 ok - 1753471364\n";

TuneDB loaded(const char* text = DB) {
  TuneDB db;
  CHECK(db.parse(text, "fixture"));
  return db;
}

TuneCommand parsed(const char* text) { return parseTuneCommand(text); }

// The message a malformed subcommand is refused with, or "" where it was accepted.
std::string refusal(const char* text) {
  try {
    (void)parseTuneCommand(text);
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

TEST(an_option_of_the_previous_tuner_is_refused_naming_its_flag) {
  for (const char* text : {"noconfig,fp64", "quick=5", "minexp=100000000,maxexp=200000000", "fp6431",
                           "workload=100M-400M,inplace", "emit,ntt", "status,maxexp=200000000"}) {
    CHECK(refusal(text).find("-oldtune") != std::string::npos);
  }
  CHECK(refusal("fp65").find("-oldtune") == std::string::npos);
}

TEST(every_setting_the_help_names_is_accepted_where_it_says) {
  for (const char* text :
       {"",
        "workload=100M-140M,probe=118063003",
        "workload=118063003",
        "probeWeight=0",
        "probeWeight=1",
        "kinds=prp",
        "kinds=ll",
        "kinds=prp+ll",
        "stop=0.5%",
        "stop=0",
        "bootstrap=0",
        "strategy=hybrid",
        "strategy=groups",
        "strategy=single",
        "strategy=permute:PAD+TAIL_KERNELS+IN_SIZEX",
        "comboTop=2,comboTiers=1",
        "maxPermute=2,maxPoints=all",
        "strategy=groups,maxPermute=all,maxPoints=500",
        "tunetxt=1",
        "scope,workload=330M-340M,probe=335M,probeWeight=0.5,kinds=prp,env=1",
        "scope,strategy=groups,maxPermute=all,maxPoints=200",
        "scope,maxPermute=3,comboTop=2,comboTiers=2",
        "status,stop=1%,env=1,workload=100M-140M,probe=118063003,probeWeight=0.5,kinds=prp+ll,bootstrap=1,"
        "strategy=hybrid,maxPermute=4,maxPoints=64,comboTop=3,comboTiers=3",
        "emit,tunetxt=1,env=1,workload=100M-140M,probe=118063003,probeWeight=0.5,kinds=prp",
        "reset",
        "reset,env=2",
        "reset,fft=1K:13:256:212",
        "adopt",
        "adopt,from=1,into=3",
        "adopt,env=3",
        "compact",
        "accuracy,workload=100M-140M,probe=118063003,fft=1K:13:256,groups=Tail+Middle"}) {
    CHECK_EQ(refusal(text), std::string{});
  }
}

TEST(settings_alone_or_nothing_at_all_is_a_tuning_run) {
  CHECK(parsed("").verb == TuneVerb::Run);
  CHECK(parsed("workload=100M-400M").verb == TuneVerb::Run);
  CHECK_EQ(parsed("workload=100M-400M,probe=136279841").scope.probe, u64(136'279'841));
  CHECK_EQ(parsed("probeWeight=0.25").scope.probeWeight, 0.25);
  CHECK(parsed("kinds=prp").scope.kinds == std::vector<TestKind>{TestKind::PRP});

  // Anything else is a mistyped command rather than a word for the other tuner.
  CHECK(!refusal("emitter").empty());
  CHECK(!refusal("worklaod=100M-400M").empty());
  CHECK(!refusal("workload=100M-400M,verbose").empty());

  // The env of a run is the device it opens.
  CHECK(!refusal("env=1").empty());

  // A run tunes either kind, or both.
  CHECK(parsed("kinds=ll").scope.kinds == std::vector<TestKind>{TestKind::LL});
  std::vector<TestKind> const both{TestKind::PRP, TestKind::LL};
  CHECK(parsed("kinds=prp+ll").scope.kinds == both);
  CHECK(refusal("scope,kinds=prp+ll").empty());
  CHECK(!refusal("kinds=pr").empty());

  // The two settings still have to agree.
  CHECK(!refusal("workload=100M-400M,probe=500000003").empty());
}

TEST(the_combo_settings_shape_hybrid_in_either_order) {
  Strategy const byDefault = parsed("").strategy;
  CHECK(byDefault.kind == Strategy::Kind::Hybrid);
  CHECK_EQ(byDefault.comboTop, COMBO_TOP);
  CHECK_EQ(byDefault.comboTiers, COMBO_TIERS);

  Strategy const set = parsed("comboTop=2,comboTiers=1").strategy;
  CHECK_EQ(set.comboTop, 2u);
  CHECK_EQ(set.comboTiers, 1u);

  // Given before the strategy they belong to, they still shape it.
  Strategy const before = parsed("comboTiers=2,strategy=hybrid").strategy;
  CHECK_EQ(before.comboTiers, 2u);
  CHECK_EQ(before.comboTop, COMBO_TOP);

  // Only hybrid combines, so for any other strategy they are a mistyped command.
  CHECK(!refusal("strategy=groups,comboTiers=1").empty());
  CHECK(!refusal("comboTop=4,strategy=single").empty());
  CHECK(!refusal("comboTiers=0").empty());
  CHECK(!refusal("comboTiers=4").empty());
  CHECK(!refusal("comboTop=0").empty());
  CHECK(!refusal("comboTop=three").empty());
  CHECK(!refusal("emit,comboTop=3").empty());
}

TEST(the_group_limits_are_the_users_to_raise_or_lower) {
  Strategy const byDefault = parsed("").strategy;
  CHECK_EQ(byDefault.maxPermute, MAX_PERMUTE);
  CHECK_EQ(byDefault.maxPoints, MAX_POINTS);

  Strategy const raised = parsed("maxPermute=all,maxPoints=1000").strategy;
  CHECK_EQ(raised.maxPermute, NO_LIMIT);
  CHECK_EQ(raised.maxPoints, 1000u);

  // Given before the strategy they belong to, they still shape it, and the one not named keeps its default.
  Strategy const groups = parsed("maxPoints=all,strategy=groups").strategy;
  CHECK(groups.kind == Strategy::Kind::Groups);
  CHECK_EQ(groups.maxPoints, NO_LIMIT);
  CHECK_EQ(groups.maxPermute, MAX_PERMUTE);
  CHECK_EQ(parsed("maxPermute=1").strategy.maxPermute, 1u);

  // single and permute: search no groups, so for them the limits are a mistyped command.
  CHECK(!refusal("strategy=single,maxPermute=2").empty());
  CHECK(!refusal("maxPoints=10,strategy=permute:PAD+IN_SIZEX").empty());
  CHECK(!refusal("maxPermute=0").empty());
  CHECK(!refusal("maxPoints=0").empty());
  CHECK(!refusal("maxPoints=-1").empty());
  CHECK(!refusal("maxPoints=4294967295").empty());
  CHECK(!refusal("maxPermute=every").empty());
  CHECK(!refusal("emit,maxPoints=all").empty());
}

TEST(stop_is_a_percentage_of_T_or_nothing) {
  CHECK_EQ(parsed("").stop, STOP);
  CHECK_EQ(parsed("stop=0.1%").stop, 0.001);
  CHECK_EQ(parsed("stop=2%,workload=100M-400M").stop, 0.02);
  CHECK_EQ(parsed("stop=0").stop, 0.0);
  CHECK_EQ(parsed("stop=0%").stop, 0.0);

  // A bare fraction could be read either way, so it is refused rather than guessed at.
  CHECK(!refusal("stop=0.1").empty());
  CHECK(!refusal("stop=1").empty());
  CHECK(!refusal("stop=100%").empty());
  CHECK(!refusal("stop=-1%").empty());
  CHECK(!refusal("stop=%").empty());
  CHECK(!refusal("stop=a%").empty());
  CHECK(!refusal("emit,stop=0.1%").empty());
}

TEST(each_subcommand_reads_its_own_settings) {
  CHECK(parsed("emit").verb == TuneVerb::Emit);
  CHECK_EQ(parsed("emit").env, 0u);
  CHECK_EQ(parsed("emit,env=7").env, 7u);

  CHECK(parsed("compact").verb == TuneVerb::Compact);

  CHECK(parsed("reset,env=2,fft=1K:8:1K:202").verb == TuneVerb::Reset);
  CHECK_EQ(parsed("reset,env=2,fft=1K:8:1K:202").env, 2u);
  CHECK_EQ(parsed("reset,fft=1K:8:1K:202").fft, std::string{"1K:8:1K:202"});

  CHECK(parsed("adopt,from=3,into=1").verb == TuneVerb::Adopt);
  CHECK_EQ(parsed("adopt,from=3,into=1").from, 3u);
  CHECK_EQ(parsed("adopt,from=3,into=1").env, 1u);
  CHECK_EQ(parsed("adopt,env=1").env, 1u);

  CHECK_EQ(parsed("scope,env=2").env, 2u);
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

  // A scope says how far a search reaches, not when a run stops or where it starts.
  CHECK(!refusal("scope,stop=1%").empty());
  CHECK(!refusal("scope,bootstrap=0").empty());
  CHECK(!refusal("emit,maxPoints=all").empty());
}

TEST(the_search_report_says_what_each_group_offers_and_what_its_bins_hold) {
  Env const nvidia{.isNvidia = true, .computeCapability = 806};
  FFTConfig const fft{"1K:7:256:212"};
  SearchSize const size = searchSize(nvidia, fft, {}, {});
  std::vector<std::string> const lines = searchReport(nvidia, {fft}, {}, true);
  CHECK(!lines.empty());
  if (lines.empty()) { return; }

  // One line for the FFT, naming what it offers, what its bins hold where that is more, and what whole groups would.
  CHECK(
    lines.front().starts_with("FFT64 1K:7:256:212: " + std::to_string(size.offered()) + " steps from each best set ("));
  CHECK(lines.front().find("Memory 93 of 128") != std::string::npos);
  CHECK(lines.front().ends_with("), " + std::to_string(size.whole()) + " with maxPermute=all"));

  // Then one per group.
  CHECK(std::ranges::find(lines,
                          std::string{"  Memory: 7 options in bins of 99 and 29 points, 64 and 29 offered; 2999 "
                                      "with maxPermute=all"}) != lines.end());
  CHECK(std::ranges::find(lines,
                          std::string{"  Height: 2 options in a bin of 7 points, all offered; 3 structural "
                                      "steps"}) != lines.end());
  CHECK_EQ(searchReport(nvidia, {fft}, {}, false).size(), size_t(1));
  CHECK(searchReport(nvidia, {fft}, {.kind = Strategy::Kind::Single}, true).empty());
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

// The A4000 of DB after an update: every env it holds was measured against other kernels than the binary's.
const char* const UPDATED =
  "# prpll tunedb v1\n"
  "env   3 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=1111222233334444\n"
  "env   5 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=5555666677778888\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "sess  9 env=3 start=1753471100 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  10 env=5 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "anchor 10 512:15:512:212 100000000 17 1774.000 1.0000 1753471210\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 1900.000 2.000 16 4 1.0000 ok 1753471204\n"
  "run   10 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 12 3 1.0000 ok 1753471274\n";

TEST(adopt_after_an_update_takes_the_latest_env_over_under_the_new_kernels) {
  TuneDB db = loaded(UPDATED);

  // Nothing has been measured with BUILD, so there is no env for it until adopt makes one, for the one card there is.
  u32 const into = adoptTarget(db, parsed("adopt"), BUILD);
  CHECK(into != 0u);
  CHECK(db.findEnv(into) && db.findEnv(into)->build == BUILD);
  CHECK(db.findEnv(into)->sameCard(*db.findEnv(5)));

  // The most recent earlier env is the one taken over, as when the env already existed.
  CHECK(rewriteFor(db, parsed("adopt"), into));
  CHECK(db.findEnv(5) == nullptr);
  CHECK(db.findEnv(3) != nullptr);
  CHECK_EQ(runsOn(db, into), 1u);

  // Its anchor readings were taken under the kernels being adopted, so they go; the configuration it is pinned to
  // stays.
  CHECK(db.anchors().empty());
  CHECK_EQ(db.envAnchor(into), std::string{"512:15:512:212@100000000"});
}

TEST(adopt_after_an_update_takes_the_env_named) {
  TuneDB db = loaded(UPDATED);
  u32 const into = adoptTarget(db, parsed("adopt,from=3"), BUILD);
  CHECK(into != 0u);
  CHECK(rewriteFor(db, parsed("adopt,from=3"), into));
  CHECK(db.findEnv(3) == nullptr);
  CHECK(db.findEnv(5) != nullptr);
}

TEST(adopt_goes_to_the_card_from_names) {
  // Env 3 is the A4000 under other kernels, and env 1 the A4000 under these: from=3 names its target by the card, even
  // though the P100 has an env under these kernels too.
  TuneDB db = loaded();
  CHECK_EQ(adoptTarget(db, parsed("adopt,from=3"), BUILD), 1u);
  CHECK_EQ(db.envs().size(), 3u);

  // An env this build measured has nothing to take over from itself.
  CHECK_EQ(adoptTarget(db, parsed("adopt,from=1"), BUILD), 1u);
  CHECK(!rewriteFor(db, parsed("adopt,from=1"), 1));

  CHECK_EQ(adoptTarget(db, parsed("adopt,from=7"), BUILD), 0u);
}

TEST(adopt_does_not_guess_between_cards) {
  // Every env is under other kernels and there are two cards, so which one this is cannot be told without a device.
  TuneDB db = loaded();
  CHECK_EQ(adoptTarget(db, parsed("adopt"), 0x0123456789abcdefull), 0u);
  CHECK_EQ(db.envs().size(), 3u);

  // Two envs under these kernels cannot be told apart either; naming one settles it.
  CHECK_EQ(adoptTarget(db, parsed("adopt"), BUILD), 0u);
  CHECK_EQ(adoptTarget(db, parsed("adopt,into=1"), BUILD), 1u);
}

TEST(adopt_after_an_update_rewrites_the_database) {
  Dir const dir{"prpll-test-tuner-adopt"};
  dir.write(TuneDB::DEFAULT_NAME, UPDATED);

  CHECK(runTuneCommand(parsed("adopt"), Args{}, dir.path));

  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  auto const current = std::ranges::find_if(after.envs(), [](const DbEnv& e) { return e.build == buildFingerprint(); });
  CHECK(current != after.envs().end());
  CHECK(after.findEnv(5) == nullptr);
  CHECK_EQ(runsOn(after, current->id), 1u);
}

TEST(a_failed_adopt_leaves_no_env_behind) {
  Dir const dir{"prpll-test-tuner-adopt-refused"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  // Two cards and nothing named: refused, and the file is as it was.
  CHECK(!runTuneCommand(parsed("adopt"), Args{}, dir.path));
  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.envs().size(), 3u);
}

TEST(emit_publishes_the_selection_file_beside_the_database) {
  Dir const dir{"prpll-test-tuner-emit"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  // Env 3's rows are the only ones a binary carrying OTHER_BUILD can see, and the one env carrying it needs no naming
  // -- but this test cannot choose the fingerprint the binary was built with, so it names the env instead.
  CHECK(runTuneCommand(parsed("emit,env=1"), Args{}, dir.path));
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

TEST(emit_writes_tune_txt_only_when_asked) {
  Dir const dir{"prpll-test-tuner-tunetxt"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  CHECK(runTuneCommand(parsed("emit,env=1"), Args{}, dir.path));
  CHECK(dir.has("selection.txt"));
  CHECK(!dir.has("tune.txt"));

  CHECK(runTuneCommand(parsed("emit,env=1,tunetxt=1"), Args{}, dir.path));
  CHECK(dir.has("tune.txt"));

  std::optional<SelectionFile> const file = readSelection(dir.path / "selection.txt");
  TuneDB db;
  CHECK(db.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK(file.has_value() && db.findEnv(1));
  if (!file || !db.findEnv(1)) { return; }
  CHECK_EQ(File::openRead(dir.path / "tune.txt").readAll(),
           compatibilityText(compatibilityView(*file, db.findEnv(1)->toEnv())));
}

TEST(tune_txt_is_written_by_a_run_or_by_emit) {
  CHECK(!parseTuneCommand("workload=100M-400M").tuneTxt);
  CHECK(parseTuneCommand("workload=100M-400M,tunetxt=1").tuneTxt);
  CHECK(parseTuneCommand("emit,tunetxt=1").tuneTxt);
  CHECK(!parseTuneCommand("emit,tunetxt=0").tuneTxt);

  CHECK(!refusal("tunetxt=2").empty());
  CHECK(!refusal("emit,tunetxt=").empty());
  CHECK(!refusal("scope,tunetxt=1").empty());
  CHECK(!refusal("compact,tunetxt=1").empty());
}

TEST(a_command_that_rewrites_writes_the_database_back) {
  Dir const dir{"prpll-test-tuner-compact"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  CHECK(runTuneCommand(parsed("compact"), Args{}, dir.path));

  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.runs().size(), 3u);
  CHECK(!dir.has("selection.txt"));
}

TEST(a_command_says_so_rather_than_writing_anything_it_cannot) {
  Dir const dir{"prpll-test-tuner-refused"};
  dir.write(TuneDB::DEFAULT_NAME, DB);

  CHECK(!runTuneCommand(parsed("emit,env=9"), Args{}, dir.path));
  CHECK(!dir.has("selection.txt"));

  CHECK(!runTuneCommand(parsed("reset,env=2,fft=not-a-spec"), Args{}, dir.path));

  TuneDB after;
  CHECK(after.load(dir.path / TuneDB::DEFAULT_NAME));
  CHECK_EQ(after.runs().size(), 4u);
}

TEST(a_database_that_is_not_there_is_an_empty_one) {
  Dir const dir{"prpll-test-tuner-absent"};

  // Nothing was measured, so there is no env to work on and nothing is published -- but the command reads a missing
  // file as empty rather than as a failure to read.
  CHECK(!runTuneCommand(parsed("emit"), Args{}, dir.path));
  CHECK(!dir.has("selection.txt"));

  // `compact` needs no env, so an empty database is one it can rewrite.
  CHECK(runTuneCommand(parsed("compact"), Args{}, dir.path));
  CHECK(dir.has(TuneDB::DEFAULT_NAME));
}

TEST(scope_reports_the_objective_over_what_is_there) {
  Dir const dir{"prpll-test-tuner-objective"};

  // With no database the prior prices everything, and a directory checked before any run is left as it was found.
  CHECK(runTuneCommand(parsed("scope,workload=100M-400M"), Args{}, dir.path));
  CHECK(!dir.has(TuneDB::DEFAULT_NAME));
  CHECK(!dir.has("tunedb.txt.lock"));

  // With one, the env is chosen as every other subcommand chooses it: named, or refused where the kernels match none.
  dir.write(TuneDB::DEFAULT_NAME, DB);
  CHECK(runTuneCommand(parsed("scope,workload=100M-400M,env=1"), Args{}, dir.path));
  CHECK(!runTuneCommand(parsed("scope,env=9"), Args{}, dir.path));
  CHECK(!runTuneCommand(parsed("scope"), Args{}, dir.path));

  // Naming an env where there is no database is a question about an env that is not there.
  Dir const bare{"prpll-test-tuner-objective-bare"};
  CHECK(!runTuneCommand(parsed("scope,env=1"), Args{}, bare.path));
}

// ---------------------------------------------------------------------------------------------------------------
// Scope: the range, the probe, the grid and the weights.

namespace {

// The total weight of a grid, which is what the mixture is normalised to.
double total(const Grid& grid) {
  double sum = 0;
  for (const GridPoint& point : grid.points) { sum += point.weight; }
  return sum;
}

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

ScopeArgs scopeOf(const char* text) { return parsed(text).scope; }

// A worktodo the way a real one is written: PRP assignments with and without an AID, a double check, a comment, a
// line for another number entirely, and an exponent that is not prime.
const char* const WORKTODO = "# my work\n"
                             "PRP=FEEE9DCD59A0855711265C1165C4C693,1,2,124647911,-1,77,0\n"
                             "PRP=1,2,124647911,-1,77,0\n"
                             "PRP=B2EE67DC0A514753E488794C9DD6F6BD,1,2,131088689,-1,78,0\n"
                             "PRP=1,2,3,332192831,-1,77,0\n"
                             "DoubleCheck=E0F583710728343C61643028FBDBA0FB,286472227,75,1\n"
                             "PRP=1,2,124647912,-1,77,0\n"
                             "PRP=1,2,124647911,-1,77,0\n";

}  // namespace

TEST(scope_settings_are_read_off_the_subcommand) {
  CHECK(parsed("scope").verb == TuneVerb::Scope);
  CHECK_EQ(scopeOf("scope").lo, 0u);
  CHECK_EQ(scopeOf("scope").probe, 0u);
  CHECK(near(scopeOf("scope").probeWeight, 0.5));
  CHECK(scopeOf("scope").kinds == std::vector<TestKind>{TestKind::PRP});

  CHECK_EQ(scopeOf("scope,workload=100M-400M").lo, 100'000'000u);
  CHECK_EQ(scopeOf("scope,workload=100M-400M").hi, 400'000'000u);
  CHECK_EQ(scopeOf("scope,workload=100000000-400000000").lo, 100'000'000u);
  CHECK_EQ(scopeOf("scope,workload=1G-2G").hi, 2'000'000'000u);
  CHECK_EQ(scopeOf("scope,workload=136279841").hi, 136'279'841u);

  CHECK_EQ(scopeOf("scope,probe=136279841").probe, 136'279'841u);
  CHECK(near(scopeOf("scope,probeWeight=0").probeWeight, 0));
  CHECK(near(scopeOf("scope,probeWeight=1").probeWeight, 1));

  CHECK(scopeOf("scope,kinds=ll").kinds == std::vector<TestKind>{TestKind::LL});
  CHECK(scopeOf("scope,kinds=prp+ll").kinds == (std::vector<TestKind>{TestKind::PRP, TestKind::LL}));
  CHECK(scopeOf("scope,kinds=ll+ll").kinds == std::vector<TestKind>{TestKind::LL});
}

TEST(a_scope_setting_that_cannot_describe_a_scope_is_a_usage_error) {
  CHECK(!refusal("scope,workload=400M-100M").empty());
  CHECK(!refusal("scope,workload=").empty());
  CHECK(!refusal("scope,workload=100X-400X").empty());
  CHECK(!refusal("scope,probe=0").empty());
  CHECK(!refusal("scope,probeWeight=2").empty());
  CHECK(!refusal("scope,probeWeight=-1").empty());
  CHECK(!refusal("scope,kinds=cert").empty());
  CHECK(!refusal("scope,kinds=").empty());
  CHECK(!refusal("scope,env=0").empty());

  // The scope's settings belong to `scope`, and to `emit`, whose default lines are the races run at the scope's
  // probe; the other subcommands' belong to theirs.
  CHECK(refusal("emit,workload=100M-400M,probe=118063003").empty());
  CHECK(!refusal("reset,workload=100M-400M").empty());
  CHECK(!refusal("compact,probe=118063003").empty());
  CHECK(!refusal("scope,fft=1K:8:1K:202").empty());
}

TEST(with_no_pending_work_the_grid_spreads_over_the_default_range) {
  RunScope const scope = makeScope(scopeOf("scope"), {});

  CHECK_EQ(scope.lo, DEFAULT_WORKLOAD_LO);
  CHECK_EQ(scope.hi, DEFAULT_WORKLOAD_HI);

  // The geometric centre of 100M-400M is 200M, and the probe is the prime at or below it.
  CHECK_EQ(scope.probe, 199'999'991u);

  const Grid* const grid = scope.grid(TestKind::PRP);
  CHECK(grid != nullptr);
  CHECK_EQ(grid->points.size(), size_t{GRID_POINTS} + 1);  // the probe is not one of the spread points
  CHECK_EQ(grid->points.front().exponent, DEFAULT_WORKLOAD_LO);
  CHECK_EQ(grid->points.back().exponent, DEFAULT_WORKLOAD_HI);
  CHECK(near(total(*grid), 1));

  // Half the weight on the probe, the other half spread evenly.
  CHECK(near(grid->weight(scope.probe), 0.5));
  CHECK(near(grid->weight(DEFAULT_WORKLOAD_LO), 0.5 / GRID_POINTS));

  // Log-spaced, so the gaps grow.
  CHECK(grid->points[1].exponent - grid->points[0].exponent <
        grid->points[GRID_POINTS - 1].exponent - grid->points[GRID_POINTS - 2].exponent);

  CHECK(scope.grid(TestKind::LL) == nullptr);
}

TEST(the_range_and_the_probe_come_from_the_pending_work) {
  Dir const dir{"prpll-test-tuner-worktodo"};
  dir.write("worktodo-0.txt", WORKTODO);

  Args args;
  std::vector<fs::path> const files = worktodoFiles(args, dir.path);
  CHECK_EQ(files.size(), size_t{1});

  std::vector<PendingWork> const pending = scanWorktodo(files);

  // Four PRP assignments (three of one exponent) and one LL; the base-3 line, the comment and the composite exponent
  // are not work this could run.
  CHECK_EQ(pending.size(), size_t{5});
  CHECK_EQ(std::ranges::count(pending, PendingWork{TestKind::PRP, 124'647'911}), 3);
  CHECK_EQ(std::ranges::count(pending, PendingWork{TestKind::LL, 286'472'227}), 1);

  RunScope const scope = makeScope(scopeOf("scope"), pending);

  // From 5% below the PRP work to 25% above it, the LL exponent not being a kind this run is tuning for: a worktodo is
  // a few days of work, and the assignments a tune serves move upward.
  CHECK_EQ(scope.lo, u64(124'647'911 * 0.95));
  CHECK_EQ(scope.hi, u64(131'088'689 * 1.25));

  // The three assignments at 124647911 outnumber the one at 131088689, and they are a 2% bin apart.
  CHECK_EQ(scope.probe, 124'647'911u);

  // The range is weighed evenly, not at the assignments: only the probe carries weight of its own.
  const Grid* const grid = scope.grid(TestKind::PRP);
  CHECK(grid != nullptr);
  CHECK_EQ(grid->points.size(), size_t{GRID_POINTS} + 1);
  CHECK_EQ(grid->points.front().exponent, scope.lo);
  CHECK_EQ(grid->points.back().exponent, scope.hi);
  CHECK(near(total(*grid), 1));
  CHECK(near(grid->weight(124'647'911), 0.5));
  CHECK(near(grid->weight(131'088'689), 0));
  CHECK(near(grid->weight(scope.lo), 0.5 / GRID_POINTS));
}

TEST(each_kind_has_its_own_grid) {
  Dir const dir{"prpll-test-tuner-kinds"};
  dir.write("worktodo-0.txt", WORKTODO);

  std::vector<PendingWork> const pending = scanWorktodo(worktodoFiles(Args{}, dir.path));
  RunScope const scope = makeScope(scopeOf("scope,kinds=prp+ll,workload=100M-400M"), pending);

  const Grid* const prp = scope.grid(TestKind::PRP);
  const Grid* const ll = scope.grid(TestKind::LL);
  CHECK(prp != nullptr);
  CHECK(ll != nullptr);
  CHECK(near(total(*prp), 1));
  CHECK(near(total(*ll), 1));

  // An LL entry and a PRP entry are never compared, so each kind is weighed over the whole range on its own, whatever
  // work of it is pending.  The probe is the one exponent both kinds share: it is where the user is, whichever kind is
  // running there.
  CHECK(prp->points == ll->points);
  CHECK(near(ll->weight(286'472'227), 0));
  CHECK_EQ(scope.probe, 124'647'911u);
  CHECK(near(ll->weight(scope.probe), 0.5));
  CHECK(near(prp->weight(scope.probe), 0.5));
}

TEST(a_cert_is_prp_work) {
  Dir const dir{"prpll-test-tuner-cert"};
  dir.write("worktodo-0.txt", "Cert=B2EE67DC0A514753E488794C9DD6F6BD,1,2,124647911,-1,162105\n");

  std::vector<PendingWork> const pending = scanWorktodo(worktodoFiles(Args{}, dir.path));
  CHECK_EQ(pending.size(), size_t{1});
  CHECK(pending.front().kind == TestKind::PRP);
  CHECK_EQ(pending.front().exponent, 124'647'911u);
}

TEST(a_worktodo_that_cannot_be_read_is_an_error) {
  Dir const dir{"prpll-test-tuner-unreadable"};
  dir.write("worktodo-0.txt", WORKTODO);
  std::vector<fs::path> const files = worktodoFiles(Args{}, dir.path);
  CHECK_EQ(files.size(), size_t{1});

  fs::permissions(files.front(), fs::perms::none);
  // Root, and Windows, read the file all the same, and there is then nothing to test.
  if (!File::openRead(files.front())) {
    bool threw = false;
    try {
      (void)scanWorktodo(files);
    } catch (const char*) { threw = true; }
    CHECK(threw);
  }
  fs::permissions(files.front(), fs::perms::owner_all);
}

TEST(a_named_workload_is_what_bounds_the_grid) {
  std::vector<PendingWork> const pending{
    {TestKind::PRP, 90'000'049}, {TestKind::PRP, 124'647'911}, {TestKind::PRP, 500'000'003}};

  RunScope const scope = makeScope(scopeOf("scope,workload=100M-400M"), pending);
  CHECK_EQ(scope.lo, 100'000'000u);
  CHECK_EQ(scope.hi, 400'000'000u);

  // Spread evenly across the range named, whatever is pending: the work in hand is a few days of what the tune is for.
  // The exponents outside the range are outside the workload, which is what "outside the workload" is meant to mean.
  const Grid* const grid = scope.grid(TestKind::PRP);
  CHECK_EQ(grid->points.size(), size_t{GRID_POINTS} + 1);
  CHECK_EQ(grid->points.front().exponent, 100'000'000u);
  CHECK_EQ(grid->points.back().exponent, 400'000'000u);
  CHECK(near(total(*grid), 1));
  CHECK_EQ(scope.probe, 124'647'911u);
  CHECK(near(grid->weight(scope.probe), 0.5));
}

TEST(a_probe_outside_a_named_workload_is_a_usage_error) {
  CHECK(!refusal("scope,workload=100M-400M,probe=500000003").empty());

  bool threw = false;
  try {
    (void)makeScope(ScopeArgs{.lo = 100'000'000, .hi = 400'000'000, .probe = 500'000'003}, {});
  } catch (const std::string&) { threw = true; }
  CHECK(threw);
}

TEST(a_named_probe_widens_a_range_that_was_only_derived) {
  // The worktodo says where the work is now, and the probe where it is headed; a range nobody named covers both.
  RunScope const scope = makeScope(scopeOf("scope,probe=500000003"), {{TestKind::PRP, 124'647'911}});

  CHECK(scope.lo <= 124'647'911u);
  CHECK(scope.hi >= 500'000'003u);
  CHECK_EQ(scope.probe, 500'000'003u);
  CHECK(scope.rangeSource.find("widened to the probe") != std::string::npos);
  CHECK(near(total(*scope.grid(TestKind::PRP)), 1));
}

TEST(a_probe_is_always_an_exponent_the_tuner_could_time) {
  // 124647912 is even; the probe is the prime at or below whatever is asked for.
  RunScope const scope = makeScope(scopeOf("scope,workload=100M-400M,probe=124647912"), {});
  CHECK_EQ(scope.probe, 124'647'911u);
  CHECK(scope.probeSource.find("124647912") != std::string::npos);

  // An exponent that is already prime is left alone.
  CHECK_EQ(makeScope(scopeOf("scope,workload=100M-400M,probe=124647911"), {}).probe, 124'647'911u);
}

TEST(the_probe_weight_moves_all_of_the_weight_and_none_of_it) {
  std::vector<PendingWork> const pending{{TestKind::PRP, 124'647'911}, {TestKind::PRP, 131'088'689}};

  RunScope const all = makeScope(scopeOf("scope,probeWeight=1"), pending);
  CHECK(near(all.grid(TestKind::PRP)->weight(all.probe), 1));
  CHECK(near(total(*all.grid(TestKind::PRP)), 1));

  RunScope const none = makeScope(scopeOf("scope,probeWeight=0"), pending);
  CHECK(near(none.grid(TestKind::PRP)->weight(none.probe), 0));
  CHECK(near(none.grid(TestKind::PRP)->weight(none.lo), 1.0 / GRID_POINTS));
  CHECK(near(total(*none.grid(TestKind::PRP)), 1));
}

TEST(every_worktodo_a_run_would_read_is_read) {
  Dir const dir{"prpll-test-tuner-files"};
  Dir const pool{"prpll-test-tuner-pool"};

  dir.write("worktodo-0.txt", "PRP=1,2,124647911,-1,77,0\n");
  dir.write("worktodo-1.txt", "PRP=1,2,131088689,-1,78,0\n");
  dir.write("worktodo.txt", "DoubleCheck=E0F583710728343C61643028FBDBA0FB,286472227,75,1\n");
  pool.write("worktodo.txt", "PRP=1,2,143413741,-1,79,0\n");

  Args args;
  args.workers = 2;
  args.masterDir = pool.path;

  std::vector<fs::path> const files = worktodoFiles(args, dir.path);
  CHECK_EQ(files.size(), size_t{4});

  // A worker's file that is not there contributes nothing rather than failing the scan.
  args.workers = 4;
  CHECK_EQ(worktodoFiles(args, dir.path).size(), size_t{4});

  std::vector<PendingWork> const pending = scanWorktodo(files);
  CHECK_EQ(pending.size(), size_t{4});
  CHECK_EQ(std::ranges::count(pending, PendingWork{TestKind::LL, 286'472'227}), 1);
  CHECK_EQ(std::ranges::count(pending, PendingWork{TestKind::PRP, 143'413'741}), 1);
}

TEST(a_single_exponent_of_pending_work_still_gives_a_range) {
  RunScope const scope = makeScope(scopeOf("scope"), {{TestKind::PRP, 124'647'911}});

  CHECK(scope.lo < 124'647'911u);
  CHECK(scope.hi > 124'647'911u);
  CHECK_EQ(scope.probe, 124'647'911u);
  CHECK(near(total(*scope.grid(TestKind::PRP)), 1));

  // Named as one exponent, it is widened the same way rather than leaving a grid of one point.
  RunScope const named = makeScope(scopeOf("scope,workload=124647911"), {});
  CHECK(named.lo < 124'647'911u);
  CHECK(named.hi > 124'647'911u);
}

TEST(an_exponent_the_prime_test_cannot_answer_for_is_a_usage_error) {
  // Both ends of the band, each of which used to run the sieve off its own supported range and never return.
  CHECK(!refusal("scope,probe=1").empty());
  CHECK(!refusal("scope,probe=999").empty());
  CHECK(!refusal("scope,probe=20G").empty());
  CHECK(!refusal("scope,workload=1-2").empty());
  CHECK(!refusal("scope,workload=15G-20G").empty());

  // The suffix is applied to an exponent that would wrap past the band rather than back into it.
  CHECK(!refusal("scope,probe=99999999999G").empty());

  CHECK_EQ(scopeOf("scope,probe=10G").probe, 10'000'000'000u);
  CHECK_EQ(scopeOf("scope,workload=1000-2000").lo, 1000u);
}

TEST(a_scope_given_exponents_the_parser_never_saw_refuses_them_too) {
  // makeScope is the entry point the tuner will use, not only the command line, so it holds the band itself.
  for (const ScopeArgs& args :
       {ScopeArgs{.lo = 1, .hi = 2}, ScopeArgs{.lo = 100'000'000, .hi = 400'000'000, .probe = 1}}) {
    bool threw = false;
    try {
      (void)makeScope(args, {});
    } catch (const std::string&) { threw = true; }
    CHECK(threw);
  }

  // Pending work outside the band is not work this can be scoped to, and does not drag the range out with it.
  RunScope const scope = makeScope(ScopeArgs{}, {{TestKind::PRP, 124'647'911}, {TestKind::PRP, 20'000'000'000}});
  CHECK(scope.hi <= MAX_EXPONENT);
  CHECK_EQ(scope.probe, 124'647'911u);
}

TEST(a_range_with_no_prime_in_it_is_refused) {
  // 124647912 and 124647913 are both composite, so the range names no exponent the tuner could time.
  bool threw = false;
  try {
    (void)makeScope(ScopeArgs{.lo = 124'647'912, .hi = 124'647'913}, {});
  } catch (const std::string&) { threw = true; }
  CHECK(threw);

  // One that does hold a prime, but not below the exponent asked for, reaches up for it instead.
  RunScope const up = makeScope(ScopeArgs{.lo = 124'647'912, .hi = 124'647'960}, {});
  CHECK_EQ(up.probe, 124'647'953u);
}

TEST(one_worktodo_reached_two_ways_is_read_once) {
  Dir const dir{"prpll-test-tuner-equivalent"};
  dir.write("worktodo.txt", "PRP=1,2,124647911,-1,77,0\n");
  dir.write("worktodo-0.txt", "PRP=1,2,131088689,-1,78,0\n");

  // A pool naming the run directory by another spelling of the same path: the same file, and its assignments must
  // not be counted twice.
  Args args;
  args.masterDir = dir.path / ".";

  std::vector<fs::path> const files = worktodoFiles(args, dir.path);
  CHECK_EQ(files.size(), size_t{2});
  CHECK_EQ(scanWorktodo(files).size(), size_t{2});

  RunScope const scope = makeScope(ScopeArgs{}, scanWorktodo(files));
  CHECK_EQ(scope.lo, u64(124'647'911 * 0.95));
  CHECK_EQ(scope.hi, u64(131'088'689 * 1.25));
  CHECK_EQ(scope.rangeSource, std::string{"2 assignments pending, from 5% below to 25% above"});
}

TEST(the_bootstrap_is_on_unless_a_run_turns_it_off) {
  CHECK(parseTuneCommand("workload=100M-400M").bootstrap);
  CHECK(!parseTuneCommand("workload=100M-400M,bootstrap=0").bootstrap);
  CHECK(parseTuneCommand("bootstrap=1").bootstrap);

  CHECK(!refusal("bootstrap=2").empty());
  CHECK(!refusal("bootstrap=").empty());

  // It is a run's setting: nothing else races anything.
  CHECK(!refusal("scope,bootstrap=0").empty());
  CHECK(!refusal("emit,bootstrap=0").empty());
}

// ---------------------------------------------------------------------------------------------------------------
// Status: a run's settings, recorded with its session and read back.

namespace {

bool sameGrids(const RunScope& a, const RunScope& b) {
  if (a.grids.size() != b.grids.size()) { return false; }
  for (size_t i = 0; i < a.grids.size(); ++i) {
    const Grid& x = a.grids[i];
    const Grid& y = b.grids[i];
    if (x.kind != y.kind || x.points.size() != y.points.size()) { return false; }
    for (size_t j = 0; j < x.points.size(); ++j) {
      if (x.points[j].exponent != y.points[j].exponent || !near(x.points[j].weight, y.points[j].weight)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

TEST(a_runs_settings_read_back_as_the_same_run) {
  std::vector<PendingWork> const pending{
    {TestKind::PRP, 124'647'911}, {TestKind::PRP, 131'088'689}, {TestKind::LL, 286'472'227}};

  // Resolved from the pending work, the range and the probe are named in the word.
  CHECK_EQ(runSettings(makeScope(parsed("").scope, pending), parsed("")),
           std::string{"workload=118415515-163860861,probe=124647911,probeWeight=0.5,kinds=prp,bootstrap=1,"
                       "strategy=hybrid,maxPermute=4,maxPoints=64,comboTop=3,comboTiers=3,contenders=16,roundCalls=16,"
                       "halvings=2,lookCalls=16,typeMargin=100%,lookMargin=50%,linesSweep=1,stop=0.1%"});

  for (const char* const text :
       {"", "workload=100M-400M,probe=136279841,stop=0", "kinds=prp+ll,probeWeight=0.3,bootstrap=0,stop=0.25%",
        "strategy=permute:PAD+IN_SIZEX", "comboTop=2,comboTiers=1", "strategy=single,stop=2%",
        "maxPermute=all,maxPoints=200", "strategy=groups,maxPermute=2,maxPoints=all", "contenders=0",
        "contenders=4,roundCalls=40", "halvings=1", "halvings=5", "lookCalls=0", "lookCalls=40,typeMargin=12.5%",
        "typeMargin=all,linesSweep=0", "lookMargin=all", "lookMargin=35%,typeMargin=60%"}) {
    TuneCommand const command = parsed(text);
    RunScope const scope = makeScope(command.scope, pending);
    std::string const word = runSettings(scope, command);

    TuneCommand const again = parsed(word.c_str());
    RunScope const back = makeScope(again.scope, pending);
    CHECK_EQ(runSettings(back, again), word);
    CHECK_EQ(back.lo, scope.lo);
    CHECK_EQ(back.hi, scope.hi);
    CHECK_EQ(back.probe, scope.probe);
    CHECK(sameGrids(back, scope));
    CHECK(again.bootstrap == command.bootstrap);
    CHECK_EQ(again.strategy.text(), command.strategy.text());
    CHECK_EQ(again.strategy.maxPermute, command.strategy.maxPermute);
    CHECK_EQ(again.strategy.maxPoints, command.strategy.maxPoints);
    CHECK_EQ(again.strategy.comboTop, command.strategy.comboTop);
    CHECK_EQ(again.strategy.comboTiers, command.strategy.comboTiers);
    CHECK_EQ(again.halving.contenders, command.halving.contenders);
    CHECK_EQ(again.halving.roundCalls, command.halving.roundCalls);
    CHECK_EQ(again.halving.halvings, command.halving.halvings);
    CHECK_EQ(again.exploration.lookCalls, command.exploration.lookCalls);
    CHECK(again.exploration.typeMargin == command.exploration.typeMargin);
    CHECK(again.exploration.lookMargin == command.exploration.lookMargin);
    CHECK_EQ(again.exploration.linesSweep, command.exploration.linesSweep);
    CHECK(near(again.stop, command.stop));
  }

  // A count, or for roundCalls= and halvings= one of at least 1; a percentage or all; 0 or 1; and a run's settings,
  // not an emit's.
  for (const char* const bad :
       {"contenders=", "contenders=-1", "contenders=all", "roundCalls=0", "roundCalls=x", "halvings=0",
        "halvings=", "halvings=all", "lookCalls=", "lookCalls=all", "typeMargin=100", "typeMargin=-5%",
        "typeMargin=", "lookMargin=50", "lookMargin=", "linesSweep=2", "emit,lookCalls=4", "emit,lookMargin=50%"}) {
    bool refused = false;
    try {
      (void)parsed(bad);
    } catch (const std::string&) { refused = true; }
    CHECK(refused);
  }
  CHECK_EQ(parsed("contenders=0").halving.on(), false);
  CHECK_EQ(parsed("").halving.halvings, HALVINGS);
  CHECK_EQ(parsed("halvings=1").halving.halvings, 1u);
  CHECK_EQ(parsed("").exploration.lookCalls, LOOK_CALLS);
  CHECK(parsed("").exploration.typeMargin == TYPE_MARGIN);
  CHECK(parsed("").exploration.linesSweep);
  CHECK(near(parsed("typeMargin=40%").exploration.typeMargin, 0.4));
  CHECK(std::isinf(parsed("typeMargin=all").exploration.typeMargin));
  CHECK(parsed("").exploration.lookMargin == LOOK_MARGIN);
  CHECK(near(parsed("lookMargin=35%").exploration.lookMargin, 0.35));
  CHECK(std::isinf(parsed("lookMargin=all").exploration.lookMargin));
}

TEST(a_run_is_weighted_by_the_work_it_recorded_whatever_the_worktodo_says_now) {
  std::vector<PendingWork> const pending{{TestKind::PRP, 131'088'689},
                                         {TestKind::PRP, 124'647'911},
                                         {TestKind::LL, 286'472'227},
                                         {TestKind::PRP, 124'647'911}};
  TuneDB db = loaded();
  CHECK(db.add(SessRow{.id = 10, .env = 1, .start = 1, .gen = 0, .anchor = {}, .tune = "x", .alarmed = false}));
  for (const WorkRow& row : workRows(10, pending)) { CHECK(db.add(row)); }
  CHECK_EQ(db.works().size(), size_t{3});

  // Read back from the file, as status reads it.
  TuneDB back;
  CHECK(back.parse(db.text(), "status"));
  std::vector<PendingWork> const recorded = pendingOf(back, 10);
  CHECK_EQ(recorded.size(), pending.size());
  CHECK(pendingOf(back, 4).empty());

  // The worktodo has since emptied; the run's own settings still give the grid and the T it had.  Once the range and
  // the probe are named, as a run's settings name them, the grid no longer depends on the work pending at all.
  for (const char* const text : {"", "kinds=prp+ll,probeWeight=0.2"}) {
    TuneCommand const command = parsed(text);
    RunScope const then = makeScope(command.scope, pending);
    TuneCommand const saved = parsed(runSettings(then, command).c_str());
    RunScope const now = makeScope(saved.scope, {});
    RunScope const restored = makeScope(saved.scope, recorded);
    CHECK(sameGrids(restored, then));
    CHECK(sameGrids(now, then));
    CHECK_EQ(Objective(back, 1, restored).T(), Objective(back, 1, then).T());
  }

  // Which is the work a status takes for the run, whatever the worktodo beside it holds; with no run, that worktodo.
  Dir const dir{"prpll-test-tuner-status-work"};
  dir.write("worktodo.txt", "PRP=1,2,332192831,-1,77,0\n");
  CHECK(statusWork(back, back.findSession(10), Args{}, dir.path) == recorded);
  std::vector<PendingWork> const beside{{TestKind::PRP, 332'192'831}};
  CHECK(statusWork(back, nullptr, Args{}, dir.path) == beside);
}

TEST(a_status_takes_the_latest_runs_settings_with_its_own_in_their_place) {
  std::string const run = "workload=100000000-400000000,probe=136279841,probeWeight=0.5,kinds=prp,bootstrap=1,"
                          "strategy=hybrid,maxPermute=4,maxPoints=64,comboTop=3,comboTiers=3,stop=0.1%";
  CHECK_EQ(statusSettings(run, ""), run);
  CHECK_EQ(statusSettings(run, "kinds=ll"),
           std::string{"workload=100000000-400000000,probe=136279841,probeWeight=0.5,bootstrap=1,strategy=hybrid,"
                       "maxPermute=4,maxPoints=64,comboTop=3,comboTiers=3,stop=0.1%,kinds=ll"});

  // The run's range and probe were resolved together, so naming either replaces both.
  CHECK_EQ(statusSettings(run, "probe=200000033"),
           std::string{"probeWeight=0.5,kinds=prp,bootstrap=1,strategy=hybrid,maxPermute=4,maxPoints=64,comboTop=3,"
                       "comboTiers=3,stop=0.1%,probe=200000033"});
  CHECK_EQ(statusSettings(run, "workload=50M-60M").find("probe=136279841"), std::string::npos);

  // Another strategy takes hybrid's group and combination settings with it, which would otherwise refuse it.
  std::string const single = statusSettings(run, "strategy=single");
  CHECK_EQ(single,
           std::string{"workload=100000000-400000000,probe=136279841,probeWeight=0.5,kinds=prp,bootstrap=1,"
                       "stop=0.1%,strategy=single"});
  CHECK(parsed(single.c_str()).strategy.kind == Strategy::Kind::Single);

  // With no run, the status's own settings are all there is.
  CHECK_EQ(statusSettings("", "kinds=ll"), std::string{"kinds=ll"});
  CHECK_EQ(statusSettings("", ""), std::string{});
}

TEST(status_reads_a_runs_settings_and_an_env) {
  TuneCommand const plain = parsed("status");
  CHECK(plain.verb == TuneVerb::Status);
  CHECK(!opensDevice(plain.verb));
  CHECK(plain.settings.empty());

  TuneCommand const named = parsed("status,env=2,stop=0,strategy=single,workload=100M-400M");
  CHECK_EQ(named.env, 2u);
  CHECK_EQ(named.settings, std::string{"stop=0,strategy=single,workload=100M-400M"});

  CHECK(!refusal("status,tunetxt=1").empty());
  CHECK(!refusal("status,fft=1K:8:1K:202").empty());
  CHECK(!refusal("status,comboTop=2,strategy=single").empty());
  CHECK(!refusal("status,maxPoints=2,strategy=single").empty());
  CHECK_EQ(parsed("status,maxPoints=all").strategy.maxPoints, NO_LIMIT);
}

TEST(status_reads_the_database_beside_a_run_that_holds_it) {
  Dir const dir{"prpll-test-tuner-status"};

  // Nothing measured, and nothing created by asking.
  CHECK(runTuneCommand(parsed("status"), Args{}, dir.path));
  CHECK(!dir.has(TuneDB::DEFAULT_NAME));
  CHECK(!dir.has("tunedb.txt.lock"));

  // Another process holding the database, part-way through appending a row.
  dir.write(TuneDB::DEFAULT_NAME, std::string{DB} + "run   4 512:15:512:212 prp 1000");
  {
    TuneDB writer;
    CHECK(writer.lockForWriting(dir.path / TuneDB::DEFAULT_NAME));
    CHECK(runTuneCommand(parsed("status,env=1,workload=100M-400M"), Args{}, dir.path));
    CHECK(!runTuneCommand(parsed("status,env=9"), Args{}, dir.path));
  }

  // A run's recorded settings are what it is valued with, and ones this build cannot read are refused, not guessed at.
  dir.write(TuneDB::DEFAULT_NAME,
            std::string{DB} +
              "sess  10 env=1 start=1753481200 gen=0 anchor=- tune=workload=100000000-400000000,probe=136279841\n");
  CHECK(runTuneCommand(parsed("status,env=1"), Args{}, dir.path));
  CHECK(runTuneCommand(parsed("status,env=1,stop=0"), Args{}, dir.path));
  dir.write(TuneDB::DEFAULT_NAME,
            std::string{DB} + "sess  10 env=1 start=1753481200 gen=0 anchor=- tune=strategy=best\n");
  CHECK(!runTuneCommand(parsed("status,env=1"), Args{}, dir.path));
}
