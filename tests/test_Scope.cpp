// Copyright (C) Jason Lynch

// The -tune settings word, and the scope a run derives from it and from the pending work.  No device is opened.

#include "test.h"

#include "Args.h"
#include "File.h"
#include "TuneCommand.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tune;

namespace {

TuneCommand parsed(const char* text) { return parseTuneCommand(text); }

// The message a malformed word is refused with, or "" where it was accepted.
std::string refusal(const char* text) {
  try {
    (void)parseTuneCommand(text);
  } catch (const std::string& mes) { return mes; } catch (const char* mes) {
    return mes;
  }
  return {};
}

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

TEST(an_option_of_upstreams_tuner_is_refused_naming_its_flag) {
  for (const char* text : {"noconfig,fp64", "quick=5", "minexp=100000000", "fp6431", "workload=100M-400M,inplace",
                           "emit,ntt", "status,maxexp=200000000", "regs", "variants,exp=140M-142M"}) {
    CHECK(refusal(text).find("-oldtune") != std::string::npos);
  }
  CHECK(refusal("fp65").find("-oldtune") == std::string::npos);
}

TEST(every_setting_is_accepted_where_it_belongs) {
  for (const char* text : {"", "workload=100M-140M,probe=118063003", "probeWeight=0", "kinds=prp+ll", "pfaShapes=all",
                           "tailQueues=none", "stop=0.5%", "stop=0", "dashboard=1", "seed=42", "sweep=0", "sweep=0.3",
                           "status", "status,stop=1%", "emit,kinds=ll", "scope,workload=60M-80M"}) {
    CHECK_EQ(refusal(text), std::string{});
  }
  CHECK(parsed("").verb == TuneVerb::Run);
  CHECK(parsed("workload=60M-80M").verb == TuneVerb::Run);
  CHECK(parsed("status").verb == TuneVerb::Status);
  CHECK(near(parsed("stop=0.5%").stop, 0.005));
  CHECK(!refusal("stop=0.1").empty());
  CHECK(near(parsed("").sweep, SWEEP));
  CHECK(near(parsed("sweep=0.3").sweep, 0.3));
  CHECK(!refusal("sweep=2").empty());
  CHECK(!refusal("status,sweep=0.3").empty());
  CHECK(parsed("tailQueues=all").tailQueues == TailQueues::All);
  CHECK(parsed("lanes=model").lanes == *parseTimetable("model"));
  CHECK(!refusal("lanes=70/20/10").empty());
  CHECK(!refusal("status,lanes=model").empty());
  CHECK(near(parsed("").explore.contenders, CONTENDERS));
  CHECK(near(parsed("contenders=5%").explore.contenders, 0.05));
  CHECK(std::isinf(parsed("contenders=all").explore.contenders));
  CHECK(parsed("").explore.reach == Reach{});
  CHECK(near(parsed("budget=2h").budget, 7200));
  CHECK(near(parsed("").budget, 0));
  CHECK(near(parsed("reserve=10%").reserve, 0.1));
  CHECK(near(parsed("").target, TARGET));
  CHECK(near(parsed("").newFloor, NEW_FLOOR));
  CHECK(near(parsed("").explore.sweepBand, SWEEP_BAND));
  CHECK(near(parsed("frontSweep=0").explore.sweepBand, 0));
  CHECK(near(parsed("frontSweep=10%").explore.sweepBand, 0.1));
  CHECK(!refusal("frontSweep=10").empty());
  CHECK(near(parsed("newFloor=0").newFloor, 0));
  CHECK(!refusal("newFloor=2").empty());
  CHECK(near(parsed("").explore.restartCut, RESTART_CUT));
  CHECK(near(parsed("restartCut=30%").explore.restartCut, 0.3));
  CHECK(std::isinf(parsed("restartCut=all").explore.restartCut));
  CHECK(!refusal("restartCut=0.3").empty());
  CHECK(near(parsed("").shortCalls, SHORT_CALLS));
  CHECK(near(parsed("shortCalls=3x").shortCalls, 3));
  CHECK(near(parsed("shortCalls=0").shortCalls, 0));
  CHECK(!refusal("shortCalls=0.5x").empty());
  CHECK(!refusal("shortCalls=2").empty());
  CHECK(near(parsed("target=0.1%").target, 0.001));
  CHECK(!refusal("target=0.1").empty());
  CHECK(!refusal("budget=2").empty());
  CHECK(!refusal("reserve=100%").empty());
  CHECK(parsed("randomReach=2x@1h").explore.reach == (Reach{.start = 2, .doubling = 3600}));
  CHECK(std::isinf(parsed("randomReach=all").explore.reach.start));
  for (const char* bad : {"randomReach=0.5x@30m", "randomReach=2x", "randomReach=2x@0m", "randomReach=x@1h"}) {
    CHECK(!refusal(bad).empty());
  }
  CHECK(!refusal("contenders=0.05").empty());
  CHECK_EQ(parsed("tag=875mhz").tag, std::string{"875mhz"});
  CHECK_EQ(parsed("emit,tag=a.b_c-1").tag, std::string{"a.b_c-1"});
  CHECK(parsed("").tag.empty());
  CHECK(!refusal("tag=875 mhz").empty());
  CHECK(!refusal("tag=a/b").empty());
  CHECK(near(parsed("").powerHold, POWER_HOLD));
  CHECK(near(parsed("powerHold=0.9").powerHold, 0.9));
  CHECK(near(parsed("powerHold=off").powerHold, 0));
  CHECK(near(parsed("status,powerHold=off").powerHold, 0));
  CHECK(!refusal("powerHold=1.5").empty());
  CHECK(!refusal("powerHold=95%").empty());
  for (const char* word : {"powerHold=0.875", "powerHold=off"}) {
    TuneCommand const held = parsed(word);
    CHECK(near(parsed(settingsWord(makeScope(held.scope, {}), held).c_str()).powerHold, held.powerHold));
  }
  CHECK(settingsWord(makeScope(parsed("").scope, {}), parsed("")).find("powerHold") == std::string::npos);

  CHECK(near(parsed("").callHeat, CALL_HEAT_SEC) && parsed("").callBlocks == BLOCKS_PER_CALL);
  CHECK(near(parsed("heat=5s").callHeat, 5) && near(parsed("heat=0").callHeat, 0));
  CHECK(parsed("blocks=8").callBlocks == 8u);
  for (const char* bad : {"heat=5", "heat=1h", "blocks=0", "blocks=65", "status,heat=5s", "emit,blocks=8"}) {
    CHECK(!refusal(bad).empty());
  }
  TuneCommand const shaped = parsed("heat=2.5s,blocks=8");
  TuneCommand const back = parsed(settingsWord(makeScope(shaped.scope, {}), shaped).c_str());
  CHECK(near(back.callHeat, 2.5) && back.callBlocks == 8u);
  CHECK(settingsWord(makeScope(parsed("").scope, {}), parsed("")).find("blocks") == std::string::npos);
  CHECK(parsed("").heater && !parsed("heater=off").heater && parsed("heater=on").heater);
  for (const char* bad : {"heater=0", "heater=yes", "status,heater=off"}) { CHECK(!refusal(bad).empty()); }
  CHECK(!parsed(settingsWord(makeScope(parsed("heater=off").scope, {}), parsed("heater=off")).c_str()).heater);
  CHECK(settingsWord(makeScope(parsed("").scope, {}), parsed("")).find("heater") == std::string::npos);
  CHECK(parsed("").compileAhead && !parsed("compileAhead=off").compileAhead && parsed("compileAhead=on").compileAhead);
  for (const char* bad : {"compileAhead=0", "status,compileAhead=off"}) { CHECK(!refusal(bad).empty()); }
  CHECK(!parsed(settingsWord(makeScope(parsed("compileAhead=off").scope, {}), parsed("compileAhead=off")).c_str())
           .compileAhead);
  CHECK(settingsWord(makeScope(parsed("").scope, {}), parsed("")).find("compileAhead") == std::string::npos);
  // A run records its tag with its settings, which a later status reads back.
  TuneCommand const tagged = parsed("workload=60M-80M,tag=875mhz");
  CHECK_EQ(parsed(settingsWord(makeScope(tagged.scope, {}), tagged).c_str()).tag, std::string{"875mhz"});
  CHECK(settingsWord(makeScope(parsed("").scope, {}), parsed("")).find("tag=") == std::string::npos);

  // The search, and the staged search's settings, recorded only where the run was staged.
  CHECK(parsed("").search == Search::Model && parsed("search=staged").search == Search::Staged);
  CHECK(parsed("search=model").search == Search::Model);
  CHECK(parsed("").staged == StagedSettings{});
  CHECK(near(parsed("margin=10%").staged.margin, 0.1) && std::isinf(parsed("margin=all").staged.margin));
  CHECK(near(parsed("restarts=0").staged.restarts, 0) && near(parsed("restarts=25%").staged.restarts, 0.25));
  for (const char* bad : {"search=bayes", "margin=0.25", "margin=-1%", "restarts=0.1", "restarts=101%",
                          "status,search=staged", "emit,margin=10%"}) {
    CHECK(!refusal(bad).empty());
  }
  CHECK(settingsWord(makeScope(parsed("margin=10%").scope, {}), parsed("margin=10%")).find("search") ==
        std::string::npos);
  for (const char* word : {"search=staged", "search=staged,margin=10%,restarts=0", "search=staged,margin=all"}) {
    TuneCommand const staged = parsed(word);
    TuneCommand const again = parsed(settingsWord(makeScope(staged.scope, {}), staged).c_str());
    CHECK(again.search == Search::Staged);
    CHECK(again.staged == staged.staged);
  }
  CHECK(settingsWord(makeScope(parsed("search=staged").scope, {}), parsed("search=staged")).find("margin") ==
        std::string::npos);
}

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

  CHECK(refusal("emit,workload=100M-400M,probe=118063003").empty());
  CHECK(!refusal("scope,fft=1K:8:1K:202").empty());
  CHECK(!refusal("scope,stop=1%").empty());
  CHECK(!refusal("status,dashboard=1").empty());
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
