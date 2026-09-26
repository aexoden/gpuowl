// Copyright (C) Jason Lynch

// The bootstrap's pure half: what a configuration is called, which moves a race is between, when a race is decided and
// for whom, which families are worth racing at all, and what the winners come to as lines -- each from fixed readings,
// with no GPU.

#include "Bootstrap.h"

#include "Anchor.h"
#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"
#include "Scheduler.h"

#include "test.h"

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .computeCapability = 806}; }

Env amd() { return {.isAmd = true}; }

constexpr u64 PROBE = 118'063'003;

// A reading of `calls` calls whose blocks scatter by `sd` about `mean`.
Measurement reading(double mean, double sd, u32 calls) {
  return {.mean = mean,
          .stddev = sd,
          .blocks = BLOCKS_PER_CALL * calls,
          .calls = calls,
          .drift = 1,
          .status = Status::Ok,
          .ts = 1};
}

RaceEntry entry(UseConfig config, std::optional<Measurement> m, bool out = false) {
  return {.config = std::move(config), .text = {}, .m = m, .out = out};
}

Family familyOf(const std::string& spec) {
  FFTConfig const fft{spec};
  return {.type = fft.shape.fft_type, .fft = fft};
}

// A database with the env and one session on it.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;

  explicit Fixture(const Env& device = nvidia()) {
    env = db.internEnv(dbEnvOf(device));
    sess = db.beginSession(env, "", 0, 1'753'471'200);
    CHECK(env && sess);
  }

  void add(const FFTConfig& fft, const UseConfig& options, const Measurement& m) {
    CHECK(db.add(RunRow{.sess = sess,
                        .fft = fft.spec(),
                        .kind = TestKind::PRP,
                        .exponent = PROBE,
                        .regime = regimeOf(fft, PROBE),
                        .cfg = db.internCfg(options),
                        .m = m}));
  }
};

}  // namespace

TEST(one_configuration_has_one_option_set) {
  FFTConfig const fft{"2:1K:8:256:212"};

  // WMUL=2 and INPLACE=1 are this FFT's defaults on NVIDIA, and NO_ASM is not something the tuner varies.
  CHECK_EQ(configText(
             canonicalConfig(nvidia(), fft, {{"WMUL", "2"}, {"INPLACE", "1"}, {"NO_ASM", "1"}, {"TAIL_KERNELS", "3"}})),
           std::string{"TAIL_KERNELS=3"});

  // L2_STRIPING does nothing out of place, so a set naming it there is the set without it.
  CHECK_EQ(configText(canonicalConfig(nvidia(), fft, {{"INPLACE", "0"}, {"L2_STRIPING", "2"}})),
           std::string{"INPLACE=0"});

  // The defaults are the vendor's: INPLACE=1 is a move on AMD, where it is not the default.
  CHECK_EQ(configText(canonicalConfig(amd(), fft, {{"INPLACE", "1"}})), std::string{"INPLACE=1"});
  CHECK(canonicalConfig(nvidia(), fft, {}).empty());
}

TEST(a_move_is_one_step_within_one_group) {
  FFTConfig const fft{"2:1K:8:256:212"};

  std::vector<Move> const width = movesWithin(nvidia(), fft, {}, Group::Width);
  std::vector<std::string> texts;
  for (const Move& m : width) { texts.push_back(m.text); }
  std::vector<std::string> const oneKeyEach{"SHUFL_BYTES_W=4", "SHUFL_BYTES_W=16", "LDSPAD_W=0", "ZEROHACK_W=0",
                                            "WMUL=1"};
  CHECK(texts == oneKeyEach);

  // Each moves exactly the key it names, from a background that is not itself offered.
  for (const Move& m : width) {
    CHECK_EQ(m.config.size(), size_t(1));
    CHECK(m.config.contains(m.key));
  }

  // From a moved background, the move back to the default is offered and the background is not.
  std::vector<Move> const from = movesWithin(nvidia(), fft, {{"WMUL", "1"}}, Group::Width);
  CHECK(std::ranges::any_of(from, [](const Move& m) { return m.text == "WMUL=2" && !m.config.contains("WMUL"); }));
  CHECK(std::ranges::none_of(from, [](const Move& m) { return configText(m.config) == "WMUL=1"; }));
  CHECK(std::ranges::all_of(from, [](const Move& m) { return m.key == "WMUL" || m.config.at("WMUL") == "1"; }));

  // A key the kernels cannot read is not a move: UNROLL_H is inert on FP64 away from height variant 1.
  std::vector<Move> const fp64 = movesWithin(nvidia(), FFTConfig{"512:15:512:212"}, {}, Group::Height);
  CHECK(std::ranges::none_of(fp64, [](const Move& m) { return m.key == "UNROLL_H"; }));

  // Nothing of another group.
  CHECK(movesWithin(nvidia(), fft, {}, Group::Cuda).empty());
}

TEST(a_structural_move_opens_its_dependents_to_the_next_round) {
  FFTConfig const fft{"1K:13:256:212"};

  // On NVIDIA the default is in place, where only L2_STRIPING is offered; out of place, the buffer layout is.
  std::vector<Move> const inPlace = movesWithin(nvidia(), fft, {}, Group::Placement);
  CHECK(std::ranges::none_of(inPlace, [](const Move& m) { return m.key == "IN_WG" || m.key == "PAD"; }));

  std::vector<Move> const outOfPlace = movesWithin(nvidia(), fft, {{"INPLACE", "0"}}, Group::Placement);
  for (const char* key : {"IN_WG", "IN_SIZEX", "OUT_WG", "OUT_SIZEX", "PAD"}) {
    CHECK(std::ranges::any_of(outOfPlace, [&](const Move& m) { return m.key == key; }));
  }
  CHECK(std::ranges::none_of(outOfPlace, [](const Move& m) { return m.key == "L2_STRIPING"; }));
}

TEST(memory_moves_one_access_class_at_a_time) {
  FFTConfig const fft{"2:1K:8:256:212"};

  // NVIDIA with inline PTX: FFT-data loads 2-5 and stores 2-3, the shuttle pairs (4,2) and (5,0), trig loads 5 for the
  // frequently reused class and 2-5 for the other two, and ENABLE_RESTRICT.
  std::vector<Move> const nv = movesWithin(nvidia(), fft, {}, Group::Memory);
  CHECK_EQ(nv.size(), size_t(4 + 2 + 2 + 1 + 4 + 4 + 1));

  // Every LOADS move changes one digit, and a coupled pair changes the same digit of both keys.
  for (const Move& m : nv) {
    u32 const loads = u32(useValue(m.config, "LOADS", 0));
    u32 const stores = u32(useValue(m.config, "STORES", 0));
    u32 moved = 0;
    for (u32 d = 0; d < 5; ++d) { moved += getDigit(loads, d) != 0 || getDigit(stores, d) != 0; }
    CHECK(moved <= 1);
  }
  CHECK(std::ranges::any_of(nv, [](const Move& m) { return m.text == "LOADS=40,STORES=20" && m.key.empty(); }));

  // AMD has the nontemporal mode and no PTX: mode 1 in each class that has it, and (1,1) for the shuttle.
  std::vector<Move> const a = movesWithin(amd(), fft, {}, Group::Memory);
  std::vector<std::string> texts;
  for (const Move& m : a) { texts.push_back(m.text); }
  std::vector<std::string> const nontemporal{"LOADS=1",     "LOADS=10,STORES=10", "LOADS=1000",
                                             "LOADS=10000", "STORES=1",           "ENABLE_RESTRICT=1"};
  CHECK(texts == nontemporal);
}

TEST(every_candidate_is_called_before_any_is_compared) {
  std::vector<RaceEntry> const entries{entry({}, reading(100, 0.1, 2)), entry({{"WMUL", "1"}}, reading(90, 0.1, 1)),
                                       entry({{"LDSPAD_W", "0"}}, std::nullopt)};
  RaceResult const r = decideRace(entries);
  CHECK(r.how == RaceHow::Pending);
  std::vector<size_t> const unsampledFirst{2, 1};
  CHECK(r.next == unsampledFirst);
}

TEST(a_race_is_decided_by_separation) {
  std::vector<RaceEntry> const entries{entry({}, reading(100, 0.1, 2)), entry({{"WMUL", "1"}}, reading(97, 0.1, 2)),
                                       entry({{"LDSPAD_W", "0"}}, reading(104, 0.1, 2))};
  RaceResult const r = decideRace(entries);
  CHECK(r.how == RaceHow::Separated);
  CHECK_EQ(*r.winner, size_t(1));
  CHECK(r.next.empty());
}

TEST(a_tie_goes_to_the_candidate_nearest_the_defaults) {
  // 0.1% apart and overlapping: within the margin, so the move buys nothing measurable and the incumbent keeps it, even
  // though the move read lower.
  std::vector<RaceEntry> const within{entry({}, reading(100.1, 1, 2)), entry({{"WMUL", "1"}}, reading(100, 1, 2))};
  RaceResult const r = decideRace(within);
  CHECK(r.how == RaceHow::Margin);
  CHECK_EQ(*r.winner, size_t(0));

  // 1% apart and still overlapping: not settled, so both are called again, fewest calls first.
  std::vector<RaceEntry> const open{entry({}, reading(101, 4, 3)), entry({{"WMUL", "1"}}, reading(100, 4, 2))};
  RaceResult const o = decideRace(open);
  CHECK(o.how == RaceHow::Pending);
  std::vector<size_t> const bothAgain{1, 0};
  CHECK(o.next == bothAgain);

  // The same after RACE_MAX_CALLS calls of the rival is a tie by exhaustion, which again goes to the defaults.
  std::vector<RaceEntry> const spent{entry({}, reading(101, 4, RACE_MAX_CALLS)),
                                     entry({{"WMUL", "1"}}, reading(100, 4, 3))};
  RaceResult const e = decideRace(spent);
  CHECK(e.how == RaceHow::Margin);
  CHECK_EQ(*e.winner, size_t(0));

  // But a leader that has had its calls does not settle a rival that has not: the rival is called on, with the leader
  // offered after it so that the rival's calls are not taken back to back.
  std::vector<RaceEntry> const fresh{entry({}, reading(100, 4, RACE_MAX_CALLS)),
                                     entry({{"WMUL", "1"}}, reading(101, 4, 3))};
  RaceResult const f = decideRace(fresh);
  CHECK(f.how == RaceHow::Pending);
  std::vector<size_t> const rivalThenLeader{1, 0};
  CHECK(f.next == rivalThenLeader);
}

TEST(a_tie_between_equals_stays_with_the_better_supported) {
  // Two one-key moves 0.05% apart, the incumbent well behind: a tie by margin, and the move with more calls keeps it
  // though it reads dearer -- once decided, a race's winner is every later race's incumbent and gathers calls, and
  // letting those move a verdict the margin already called a tie would re-race everything after it for nothing.
  std::vector<RaceEntry> const entries{entry({}, reading(103, 0.1, 2)), entry({{"WMUL", "1"}}, reading(100.05, 1, 9)),
                                       entry({{"LDSPAD_W", "0"}}, reading(100, 1, 2))};
  RaceResult const r = decideRace(entries);
  CHECK(r.how == RaceHow::Margin);
  CHECK_EQ(*r.winner, size_t(1));
}

TEST(a_candidate_that_failed_takes_no_part) {
  std::vector<RaceEntry> const entries{entry({}, reading(100, 0.1, 2)), entry({{"WMUL", "1"}}, std::nullopt, true)};
  RaceResult const r = decideRace(entries);
  CHECK(r.how == RaceHow::Alone);
  CHECK_EQ(*r.winner, size_t(0));

  CHECK(!decideRace({entry({}, std::nullopt, true)}).winner);
}

TEST(the_families_are_the_smallest_shape_of_each_type_the_workload_reaches) {
  std::vector<Family> const all =
    bootstrapFamilies(nvidia(), PROBE, {FFTConfig{"512:15:512:101"}, FFTConfig{"1:512:8:512:202"}});
  CHECK_EQ(all.size(), size_t(2));
  CHECK_EQ(all[0].fft.spec(), std::string{"1K:13:256:212"});
  CHECK_EQ(all[1].fft.spec(), std::string{"1:1K:8:256:202"});

  for (const Family& f : all) {
    CHECK(std::ranges::any_of(anchorCandidates(PROBE), [&](const AnchorSpec& a) { return a.fft == f.fft.spec(); }));
  }
}

TEST(every_family_is_read_before_any_is_raced) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};

  BootstrapState const cold = b.state(f.db, f.env);
  CHECK(!cold.complete);
  CHECK_EQ(cold.turns.size(), size_t(2));
  for (const Turn& t : cold.turns) { CHECK(t.config.empty()); }

  // One family read: the other is still wanted first, and nothing is raced.
  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, 1));
  BootstrapState const half = b.state(f.db, f.env);
  CHECK_EQ(half.turns.size(), size_t(1));
  CHECK_EQ(half.turns.front().family, size_t(1));
}

TEST(a_family_the_gain_prior_cannot_bring_level_is_not_raced) {
  Fixture f;
  Bootstrap const b{
    nvidia(), PROBE, {familyOf("1K:13:256:212"), familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};

  // The A4000's anchor race: FP64 at five times FFT3261 is past any gain the prior allows; FFT3161 is not.
  f.add(FFTConfig{"1K:13:256:212"}, {}, reading(8088, 1, 1));
  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, 1));
  f.add(FFTConfig{"1:1K:8:256:202"}, {}, reading(1640, 1, 1));

  BootstrapState const s = b.state(f.db, f.env);
  CHECK(s.families[0].phase == FamilyPhase::Skipped);
  CHECK(s.families[1].phase == FamilyPhase::Racing);
  CHECK(s.families[2].phase == FamilyPhase::Waiting);

  // The cheaper family races first, and its first race is its first group's.
  CHECK(!s.turns.empty());
  for (const Turn& t : s.turns) { CHECK_EQ(t.family, size_t(1)); }
  CHECK_EQ(s.families[1].stage, std::string{"Placement"});
  CHECK(!s.complete);
}

TEST(a_reading_counts_for_its_candidate_however_its_options_were_spelled) {
  // The anchor race and a session under a user's own -use both record what the kernels were built with, which can name
  // keys at their defaults or keys the tuner never varies.  Still the defaults, so the family is read.
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212")}};
  f.add(FFTConfig{"2:1K:8:256:212"}, {{"WMUL", "2"}, {"INPLACE", "1"}, {"NO_ASM", "0"}}, reading(1576, 1, 1));

  BootstrapState const s = b.state(f.db, f.env);
  CHECK_EQ(s.families[0].reading, 1576.0);
  CHECK(s.families[0].phase == FamilyPhase::Racing);
}

TEST(a_family_is_judged_against_the_cheapest_as_tuned) {
  // The A4000's FFT61 at 2257 us/it is within RACE_GAIN of FFT3261's defaults at 1576, and out of it once FFT3261 is
  // tuned past 1535 (2257 x 0.68): racing it then would be for a gain too rare to repay the race.
  Fixture f;
  Family const fp32 = familyOf("2:1K:8:256:212");
  Family const gf61 = familyOf("3:1K:16:256:202");
  Bootstrap const b{nvidia(), PROBE, {fp32, gf61}};
  f.add(fp32.fft, {}, reading(1576, 1, 1));
  f.add(gf61.fft, {}, reading(2257, 1, 1));

  BootstrapState const before = b.state(f.db, f.env);
  CHECK(before.families[1].phase == FamilyPhase::Waiting);

  // WMUL=1 takes 9% off; every other move costs 1%.
  auto cost = [](const UseConfig& c) {
    double us = 1576;
    for (const auto& [key, value] : c) { us *= key == "WMUL" && value == "1" ? 0.91 : 1.01; }
    return us;
  };

  u32 calls = 0;
  for (BootstrapState s = b.state(f.db, f.env); !s.turns.empty(); s = b.state(f.db, f.env)) {
    const Turn& t = s.turns.front();
    CHECK_EQ(t.family, size_t(0));
    f.add(fp32.fft, t.config, reading(cost(t.config), cost(t.config) * 0.0005, 1));
    // A rule that never lets the races finish would otherwise loop for ever.
    if (++calls >= 1000) {
      CHECK(false);
      break;
    }
  }

  BootstrapState const after = b.state(f.db, f.env);
  CHECK(after.complete);
  CHECK(after.families[0].phase == FamilyPhase::Done);
  CHECK(after.families[1].phase == FamilyPhase::Skipped);
  CHECK_EQ(configText(after.defaults.global), std::string{"WMUL=1"});
}

TEST(a_family_that_cannot_run_at_its_defaults_is_held) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};
  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, 1));
  f.add(FFTConfig{"1:1K:8:256:202"}, {}, Measurement{.status = Status::NoCompile, .ts = 1});

  BootstrapState const s = b.state(f.db, f.env);
  CHECK(s.families[1].phase == FamilyPhase::Held);
  CHECK(s.families[0].phase == FamilyPhase::Racing);
}

TEST(turned_off_it_decides_nothing) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212")}, false};
  BootstrapState const s = b.state(f.db, f.env);
  CHECK(s.complete);
  CHECK(s.turns.empty());
  CHECK(s.families[0].phase == FamilyPhase::Skipped);
  CHECK(s.defaults.global.empty() && s.defaults.family.empty());
}

TEST(a_family_raced_to_the_end_is_a_transcript_of_its_races) {
  // Every call answered at once from a fixed rule: WMUL=1 and TAIL_KERNELS=3 each save 2%, every other move costs 1%,
  // and anything else is the defaults.  Replaying the turns until none is left is what a run does.  The groups alone.
  Fixture f;
  Family const family = familyOf("2:1K:8:256:212");
  Bootstrap const b{nvidia(), PROBE, {family}, true, 1};

  auto cost = [](const UseConfig& c) {
    double us = 1576;
    for (const auto& [key, value] : c) {
      us *= (key == "WMUL" && value == "1") || (key == "TAIL_KERNELS" && value == "3") ? 0.98 : 1.01;
    }
    return us;
  };

  u32 calls = 0;
  for (BootstrapState s = b.state(f.db, f.env); !s.turns.empty(); s = b.state(f.db, f.env)) {
    const Turn& t = s.turns.front();
    f.add(family.fft, t.config, reading(cost(t.config), cost(t.config) * 0.0005, 1));
    // A rule that never lets the races finish would otherwise loop for ever.
    if (++calls >= 1000) {
      CHECK(false);
      break;
    }
  }

  BootstrapState const done = b.state(f.db, f.env);
  CHECK(done.complete);
  CHECK(done.families[0].phase == FamilyPhase::Done);
  CHECK_EQ(configText(done.families[0].decided), std::string{"TAIL_KERNELS=3,WMUL=1"});

  // One decision per group the family has a move in, and a second round of each group a move won -- Tail and Width --
  // which the incumbent then holds.
  CHECK_EQ(done.families[0].decisions.size(), size_t(10));
  for (const Decision& d : done.families[0].decisions) { CHECK(d.how == RaceHow::Separated); }

  // One family agrees with itself, so everything it decided is the global line.
  CHECK_EQ(configText(done.defaults.global), std::string{"TAIL_KERNELS=3,WMUL=1"});
  CHECK(done.defaults.family.empty());

  // Two calls for each of the 44 moves of its eight first rounds, and two for the defaults: each later race's incumbent
  // is the previous one's winner, whose calls it already has.  The 49 moves there are less the five that change the
  // rounding -- TAIL_KERNELS to 0 or 1, TAIL_TRIGS32 to 0 or 1, TABMUL_CHAIN32 to 1 -- which no line carries;
  // TAIL_KERNELS=3 rounds as the default does, and is raced.  Then the second rounds: Tail's from TAIL_KERNELS=3 offers
  // three moves it may race, of which the one back to TAIL_KERNELS=2 was measured in the first; Width's from WMUL=1
  // offers five, of which WMUL=2 was.
  CHECK_EQ(calls, 2u * 44 + 2 + 2 * 2 + 2 * 4);
}

TEST(no_line_carries_a_key_that_changes_the_rounding) {
  // TAIL_TRIGS32=0 and TABMUL_CHAIN32=1 would each save 10%; every other move costs 1%.  Production applies the lines
  // where no gate ever read them, so neither is raced, in the groups or in the combinations.
  Fixture f;
  Family const family = familyOf("2:1K:8:256:212");
  Bootstrap const b{nvidia(), PROBE, {family}, true};

  auto cost = [](const UseConfig& c) {
    double us = 1576;
    for (const auto& [key, value] : c) {
      us *= (key == "TAIL_TRIGS32" && value == "0") || (key == "TABMUL_CHAIN32" && value == "1") ? 0.9 : 1.01;
    }
    return us;
  };

  u32 calls = 0;
  for (BootstrapState s = b.state(f.db, f.env); !s.turns.empty() && calls < 2000; s = b.state(f.db, f.env)) {
    const Turn& t = s.turns.front();
    CHECK(!movesAccuracy(nvidia(), family.fft, t.config));
    f.add(family.fft, t.config, reading(cost(t.config), cost(t.config) * 0.0005, 1));
    ++calls;
  }

  BootstrapState const done = b.state(f.db, f.env);
  CHECK(done.complete);
  CHECK(done.defaults.global.empty());
  CHECK(done.defaults.family.empty());
}

TEST(what_the_families_agree_on_is_global_and_the_rest_is_theirs) {
  Family const fp32 = familyOf("2:1K:8:256:212");
  Family const gf31 = familyOf("1:1K:8:256:202");

  Defaults const d = defaultLines(nvidia(),
                                  {{fp32, {{"TAIL_KERNELS", "3"}, {"WMUL", "1"}, {"TAIL_TRIGS32", "1"}}},
                                   {gf31, {{"TAIL_KERNELS", "3"}, {"MODM31", "2"}}}});

  // Both moved TAIL_KERNELS to 3; TAIL_TRIGS32 and MODM31 reach one family each, so nobody disagrees about them.
  CHECK_EQ(configText(d.global), std::string{"MODM31=2,TAIL_KERNELS=3,TAIL_TRIGS32=1"});

  // WMUL reaches both, and the hybrid kept its default of 2: a line for the family that moved it, and none for the
  // one that did not, which the global line leaves at its default.
  CHECK_EQ(d.family.size(), size_t(1));
  if (d.family.size() == 1) {
    std::vector<std::pair<std::string, std::string>> const wmul{{"WMUL", "1"}};
    CHECK(d.family[0].selector.type == FFT3261);
    CHECK(d.family[0].uses == wmul);
  }

  // And what each family runs at under the lines is exactly what it decided.
  CHECK_EQ(configText(underDefaults(nvidia(), fp32.fft, TestKind::PRP, d)),
           std::string{"TAIL_KERNELS=3,TAIL_TRIGS32=1,WMUL=1"});
  CHECK_EQ(configText(underDefaults(nvidia(), gf31.fft, TestKind::PRP, d)), std::string{"MODM31=2,TAIL_KERNELS=3"});

  // A family nobody raced runs at the global line, less what does not reach it.
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"512:15:512:212"}, TestKind::PRP, d)),
           std::string{"TAIL_KERNELS=3"});
}

namespace {

// A published entry of `spec` over the band that holds `E`.
SelectionEntry publishedAt(const std::string& spec, u64 E, double cost, UseConfig opts, TestKind kind = TestKind::PRP) {
  FFTConfig const fft{spec};
  Interval const band = interval(fft, E);
  SelectionEntry e{.id = {},
                   .cost = cost,
                   .fft = fft.spec(),
                   .kind = kind,
                   .emin = band.lo,
                   .reach = band.hi,
                   .regime = band.regime,
                   .evidence = Evidence::Unvalidated,
                   .opts = std::move(opts)};
  e.id = entryId(e.fft, e.kind, e.regime, e.opts);
  return e;
}

}  // namespace

TEST(the_lines_follow_the_best_set_published_for_each_type) {
  u64 const probe = 118'063'003;
  Family const fp64 = familyOf("512:15:512:212");
  Family const fp32 = familyOf("2:1K:8:256:212");

  // Both families raced, and both moved WMUL to 1.
  BootstrapState bootstrap;
  bootstrap.families = {{.family = fp64, .phase = FamilyPhase::Done, .decided = {{"WMUL", "1"}}},
                        {.family = fp32, .phase = FamilyPhase::Done, .decided = {{"WMUL", "1"}}}};
  bootstrap.defaults = defaultLines(nvidia(), {{fp64, {{"WMUL", "1"}}}, {fp32, {{"WMUL", "1"}}}});
  CHECK_EQ(linesText(publishedLines(nvidia(), probe, TestKind::PRP, {}, bootstrap)), std::string{"WMUL=1"});

  // FP64's evidence is now what production runs at the probe: the cheapest entry there, not a dearer one there, a
  // cheaper one of another size that does not reach it, or one of the other kind.  Its TABMUL_CHAIN=1 changes the
  // rounding, which nothing reads for what the lines reach, so it is left out.  The hybrid has nothing published, so
  // its race still speaks for it.
  std::vector<SelectionEntry> const published{
    publishedAt("512:15:512:101", probe, 1700, {{"TABMUL_CHAIN", "1"}, {"TAIL_KERNELS", "3"}}),
    publishedAt("512:15:512:212", probe, 1800, {{"WMUL", "1"}}),
    publishedAt("256:13:512:101", 60'000'000, 900, {{"WMUL", "1"}}),
    publishedAt("512:15:512:102", probe, 1500, {{"WMUL", "1"}}, TestKind::LL)};
  Defaults const d = publishedLines(nvidia(), probe, TestKind::PRP, published, bootstrap);
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"512:15:512:101"}, TestKind::PRP, d)),
           std::string{"TAIL_KERNELS=3"});
  CHECK_EQ(configText(underDefaults(nvidia(), fp32.fft, TestKind::PRP, d)), std::string{"WMUL=1"});

  // Where nothing of a type covers the probe, the entry nearest it speaks for the type, however much it costs.
  CHECK(maxExp(FFTConfig{"256:14:512:101"}) < probe);
  std::vector<SelectionEntry> const below{publishedAt("256:13:512:101", 60'000'000, 900, {{"WMUL", "1"}}),
                                          publishedAt("256:14:512:101", 64'000'000, 950, {{"TAIL_KERNELS", "3"}})};
  Defaults const nearest = publishedLines(nvidia(), probe, TestKind::PRP, below, bootstrap);
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"512:15:512:212"}, TestKind::PRP, nearest)),
           std::string{"TAIL_KERNELS=3"});
}

TEST(a_key_only_amd_and_nvidia_can_choose_is_not_raced_elsewhere) {
  // Elsewhere the host builds OLD_FENCE=1 whatever is asked for.
  FFTConfig const fft{"2:1K:8:256:212"};
  auto offersFence = [&](const Env& e) {
    std::vector<Move> const moves = movesWithin(e, fft, {}, Group::Queues);
    return std::ranges::any_of(moves, [](const Move& m) { return m.key == "OLD_FENCE"; });
  };
  CHECK(offersFence(nvidia()));
  CHECK(offersFence(amd()));
  CHECK(!offersFence(Env{}));
}

TEST(a_line_reaches_only_the_shapes_that_can_take_it) {
  // SHUFL_BYTES_W=16 won at width 1K; at 4K it is past the LDS budget, and the host would derive WMUL=0 from it.
  Defaults const d{.global = {{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}}, .family = {}};
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"2:1K:8:256:212"}, TestKind::PRP, d)),
           std::string{"SHUFL_BYTES_W=16,WMUL=1"});
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"2:4K:8:256:212"}, TestKind::PRP, d)), std::string{"-"});
}

namespace {

// Replays the turns of `b` until none is left, answering each call from `cost` at once.  The turns taken, in order.
std::vector<Turn> raceToTheEnd(Fixture& f, const Bootstrap& b, const std::function<double(const UseConfig&)>& cost) {
  std::vector<Turn> out;
  for (BootstrapState s = b.state(f.db, f.env); !s.turns.empty(); s = b.state(f.db, f.env)) {
    const Turn& t = s.turns.front();
    f.add(b.families()[t.family].fft, t.config, reading(cost(t.config), cost(t.config) * 0.0005, 1));
    out.push_back(t);
    // A rule that never lets the races finish would otherwise loop for ever.
    if (out.size() >= 2000) {
      CHECK(false);
      break;
    }
  }
  return out;
}

// TAIL_KERNELS=3 and ZEROHACK_H=0, in two groups that share the tail kernels, each cost a little alone -- less than
// any other move, each of which costs 1% -- and save 3% together.
double crossGroupPair(const UseConfig& c) {
  bool const tail = useValue(c, "TAIL_KERNELS", 2) == 3;
  bool const height = useValue(c, "ZEROHACK_H", 1) == 0;
  double us = 1700 * (tail && height ? 0.97 : tail ? 1.004 : height ? 1.002 : 1);
  for (const auto& [key, value] : c) {
    if (!(key == "TAIL_KERNELS" && value == "3") && !(key == "ZEROHACK_H" && value == "0")) { us *= 1.01; }
  }
  return us;
}

}  // namespace

TEST(a_cross_group_pair_is_found_by_the_combinations_and_not_by_the_groups) {
  // FP64 at variant 212 away from 1024: Tail and Height share the tail kernels, so they are combined at tier 2.
  Family const family = familyOf("512:15:512:212");

  Fixture groupsOnly;
  Bootstrap const groups{nvidia(), PROBE, {family}, true, 1};
  std::vector<Turn> const plain = raceToTheEnd(groupsOnly, groups, crossGroupPair);
  BootstrapState const g = groups.state(groupsOnly.db, groupsOnly.env);
  CHECK(g.complete);
  CHECK_EQ(configText(g.families[0].decided), std::string{"-"});
  CHECK(std::ranges::all_of(plain, [](const Turn& t) { return t.tier == 1; }));

  Fixture whole;
  Bootstrap const tree{nvidia(), PROBE, {family}};
  std::vector<Turn> const turns = raceToTheEnd(whole, tree, crossGroupPair);
  BootstrapState const t = tree.state(whole.db, whole.env);
  CHECK(t.complete);
  CHECK_EQ(configText(t.families[0].decided), std::string{"TAIL_KERNELS=3,ZEROHACK_H=0"});
  CHECK_EQ(configText(t.defaults.global), std::string{"TAIL_KERNELS=3,ZEROHACK_H=0"});

  // The groups raced exactly as before, and the combinations after them.
  CHECK(turns.size() > plain.size());
  for (size_t i = 0; i < plain.size() && i < turns.size(); ++i) { CHECK(turns[i].config == plain[i].config); }
  CHECK(std::all_of(turns.begin() + ptrdiff_t(std::min(plain.size(), turns.size())), turns.end(), [](const Turn& turn) {
    return turn.tier > 1 || turn.text.find("the incumbent") != std::string::npos;
  }));

  // Found at tier 2, by combining the two groups' best answers, and held at tier 3.
  auto const pair =
    std::ranges::find_if(t.families[0].decisions, [](const Decision& d) { return d.stage == "Tail+Height"; });
  CHECK(pair != t.families[0].decisions.end());
  if (pair != t.families[0].decisions.end()) {
    CHECK_EQ(pair->winner, std::string{"TAIL_KERNELS=3,ZEROHACK_H=0"});
    CHECK(pair->how == RaceHow::Separated);
  }
  CHECK(t.families[0].decisions.back().stage == "all");
  CHECK_EQ(t.families[0].decisions.back().winner, std::string{"the incumbent"});
}

TEST(each_combination_stage_is_raced_once_from_the_background_as_it_stands) {
  // No combination pays: every stage holds its incumbent, and the lines are what the groups decided.
  Fixture f;
  Family const family = familyOf("2:1K:8:256:212");
  Bootstrap const b{nvidia(), PROBE, {family}};
  auto cost = [](const UseConfig& c) {
    double us = 1576;
    for (const auto& [key, value] : c) {
      us *= (key == "WMUL" && value == "1") || (key == "TAIL_KERNELS" && value == "3") ? 0.98 : 1.01;
    }
    return us;
  };
  std::vector<Turn> const turns = raceToTheEnd(f, b, cost);

  BootstrapState const done = b.state(f.db, f.env);
  CHECK(done.complete);
  CHECK_EQ(configText(done.families[0].decided), std::string{"TAIL_KERNELS=3,WMUL=1"});

  // The ten group races, then one race per stage, each once.
  const std::vector<Decision>& decisions = done.families[0].decisions;
  CHECK(decisions.size() > 10);
  std::set<std::string> stages;
  for (size_t i = 10; i < decisions.size(); ++i) {
    CHECK(stages.insert(decisions[i].stage).second);
    CHECK_EQ(decisions[i].winner, std::string{"the incumbent"});
  }
  CHECK(stages.contains("all"));
  CHECK(std::ranges::any_of(stages, [](const std::string& s) { return s != "all"; }));

  // Every combination point is a set the groups' races never measured, called as a tier-2 or tier-3 turn.
  std::set<std::string> raced;
  for (const Turn& t : turns) {
    if (t.tier == 1) { raced.insert(configText(t.config)); }
  }
  for (const Turn& t : turns) {
    if (t.tier > 1 && t.text.find("the incumbent") == std::string::npos) {
      CHECK(!raced.contains(configText(t.config)));
      CHECK(t.tier == 2 || t.tier == 3);
    }
  }
}

TEST(what_the_entry_later_measures_does_not_reopen_the_combinations) {
  // Once the bootstrap is done, its configuration is searched as an entry like any other, and those rows are at the
  // same spec and exponent.  They are not answers of the bootstrap's races, so its stages keep their candidates.
  Fixture f;
  Family const family = familyOf("512:15:512:212");
  Bootstrap const b{nvidia(), PROBE, {family}};
  (void)raceToTheEnd(f, b, crossGroupPair);
  BootstrapState const before = b.state(f.db, f.env);
  CHECK(before.complete);

  // Cheap readings of two-key moves within one group, which a combination would take as its best answers.
  f.add(family.fft, {{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}, {"TAIL_TRIGS", "1"}}, reading(1640, 0.5, 2));
  f.add(family.fft, {{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}, {"LOADS", "2"}, {"STORES", "2"}},
        reading(1641, 0.5, 2));
  BootstrapState const after = b.state(f.db, f.env);
  CHECK(after.complete);
  CHECK(after.turns.empty());
  CHECK(after.families[0].decided == before.families[0].decided);
}
