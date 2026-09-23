// Copyright (C) Jason Lynch

// The bootstrap's pure half: what a configuration is called, which moves a race is between, when a race is decided and
// for whom, which families are worth racing at all, and what the winners come to as lines -- each from fixed readings,
// with no GPU.

#include "Bootstrap.h"

#include "Anchor.h"
#include "FFTVariants.h"

#include "test.h"

#include <algorithm>
#include <functional>
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
  std::vector<std::string> const oneKeyEach{"SHUFL_BYTES_W=4", "SHUFL_BYTES_W=16", "LDSPAD_W=0", "HOIST_W=1",
                                            "HOIST_W=2",       "ZEROHACK_W=0",     "WMUL=1"};
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
  CHECK(s.families[1].group == Group::Placement);
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
  // The A4000's FFT61 at 2257 us/it is within the prior's reach of FFT3261's defaults at 1576, and out of it once
  // FFT3261 is tuned past 1535 (2257 x 0.68): racing it then could not change what production runs.
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
  // and anything else is the defaults.  Replaying the turns until none is left is what a run does.
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

  // Two calls for each of the 53 moves of its eight first rounds, and two for the defaults: each later race's incumbent
  // is the previous one's winner, whose calls it already has.  Then the second rounds: Tail's from TAIL_KERNELS=3
  // offers eight moves, of which the three back to TAIL_KERNELS 0, 1 and 2 were measured in the first; Width's from
  // WMUL=1 offers seven, of which WMUL=2 was.
  CHECK_EQ(calls, 2u * 53 + 2 + 2 * 5 + 2 * 6);
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
