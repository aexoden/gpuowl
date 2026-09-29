// Copyright (C) Jason Lynch

// The bootstrap's pure half: what a configuration is called, what one step within a group is, which configuration each
// family is searched on and whether it is worth searching, and what the best entries come to as lines -- each from
// fixed readings, with no GPU.

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

  void add(const FFTConfig& fft, const UseConfig& options, const Measurement& m, TestKind kind = TestKind::PRP,
           u64 exponent = PROBE) {
    CHECK(db.add(RunRow{.sess = sess,
                        .fft = fft.spec(),
                        .kind = kind,
                        .exponent = exponent,
                        .regime = regimeOf(fft, exponent),
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

TEST(every_family_is_read_before_any_is_searched) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};

  BootstrapState const cold = b.state(f.db, f.env, 64);
  CHECK(!cold.complete);
  CHECK_EQ(cold.budget, u64(64));
  for (const FamilyState& s : cold.families) { CHECK(s.phase == FamilyPhase::Unread); }

  // A reading under other options, or not yet concluded, is not a reading of the defaults.
  f.add(FFTConfig{"2:1K:8:256:212"}, {{"WMUL", "1"}}, reading(1500, 1, MIN_CALLS));
  f.add(FFTConfig{"1:1K:8:256:202"}, {}, reading(1640, 1, 1));
  CHECK(b.state(f.db, f.env, 64).families[0].phase == FamilyPhase::Unread);

  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, MIN_CALLS));
  BootstrapState const half = b.state(f.db, f.env, 64);
  CHECK(half.families[0].phase == FamilyPhase::Owed);
  CHECK_EQ(half.families[0].reading, 1576.0);
  CHECK_EQ(half.families[0].best, 1500.0);
  CHECK(half.families[1].phase == FamilyPhase::Unread);
  CHECK(!half.complete);
}

TEST(a_family_the_gain_prior_cannot_bring_level_is_not_searched) {
  Fixture f;
  Bootstrap const b{
    nvidia(), PROBE, {familyOf("1K:13:256:212"), familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};

  // The A4000's anchor race: FP64 at five times FFT3261 is past any gain the prior allows; FFT3161 is not.
  f.add(FFTConfig{"1K:13:256:212"}, {}, reading(8088, 1, MIN_CALLS));
  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, MIN_CALLS));
  f.add(FFTConfig{"1:1K:8:256:202"}, {}, reading(1640, 1, MIN_CALLS));

  BootstrapState const s = b.state(f.db, f.env, 64);
  CHECK(s.families[0].phase == FamilyPhase::Skipped);
  CHECK(s.families[1].phase == FamilyPhase::Owed);
  CHECK(s.families[2].phase == FamilyPhase::Owed);
  CHECK(!s.complete);
}

TEST(a_reading_counts_for_its_family_however_its_options_were_spelled) {
  // The anchor race and a session under a user's own -use both record what the kernels were built with, which can name
  // keys at their defaults or keys the tuner never varies.  Still the defaults, so the family is read.
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212")}};
  f.add(FFTConfig{"2:1K:8:256:212"}, {{"WMUL", "2"}, {"INPLACE", "1"}, {"NO_ASM", "0"}}, reading(1576, 1, MIN_CALLS));

  BootstrapState const s = b.state(f.db, f.env, 64);
  CHECK_EQ(s.families[0].reading, 1576.0);
  CHECK_EQ(s.families[0].calls, u64(0));
  CHECK(s.families[0].phase == FamilyPhase::Owed);
}

TEST(a_family_is_judged_against_the_cheapest_as_tuned) {
  // The A4000's FFT61 at 2257 us/it is within BOOTSTRAP_GAIN of FFT3261's defaults at 1576, and out of it once FFT3261
  // is tuned past 1535 (2257 x 0.68): searching it then would be for a gain too rare to repay the search.
  Fixture f;
  Family const fp32 = familyOf("2:1K:8:256:212");
  Family const gf61 = familyOf("3:1K:16:256:202");
  Bootstrap const b{nvidia(), PROBE, {fp32, gf61}};
  f.add(fp32.fft, {}, reading(1576, 1, MIN_CALLS));
  f.add(gf61.fft, {}, reading(2257, 1, MIN_CALLS));
  CHECK(b.state(f.db, f.env, 64).families[1].phase == FamilyPhase::Owed);

  f.add(fp32.fft, {{"WMUL", "1"}}, reading(1540, 1, MIN_CALLS));
  CHECK(b.state(f.db, f.env, 64).families[1].phase == FamilyPhase::Owed);
  f.add(fp32.fft, {{"WMUL", "1"}, {"TAIL_KERNELS", "3"}}, reading(1500, 1, MIN_CALLS));
  BootstrapState const after = b.state(f.db, f.env, 64);
  CHECK(after.families[0].phase == FamilyPhase::Owed);
  CHECK(after.families[1].phase == FamilyPhase::Skipped);
}

TEST(a_family_has_its_calls_of_search_and_is_served) {
  Fixture f;
  Family const fp32 = familyOf("2:1K:8:256:212");
  Bootstrap const b{nvidia(), PROBE, {fp32}};

  // Calls at the defaults are not calls of search; calls anywhere in the band are, a failure as one.
  f.add(fp32.fft, {}, reading(1576, 1, 3));
  f.add(fp32.fft, {{"WMUL", "1"}}, reading(1540, 1, 2));
  u64 const elsewhere = PROBE + 20'000;
  CHECK(regimeOf(fp32.fft, elsewhere).label() == regimeOf(fp32.fft, PROBE).label());
  f.add(fp32.fft, {{"TAIL_KERNELS", "3"}}, reading(1560, 1, 1), TestKind::PRP, elsewhere);
  BootstrapState const three = b.state(f.db, f.env, 4);
  CHECK_EQ(three.families[0].calls, u64(3));
  CHECK(three.families[0].phase == FamilyPhase::Owed);

  f.add(fp32.fft, {{"ZEROHACK_W", "0"}}, Measurement{.status = Status::Err, .ts = 1});
  BootstrapState const served = b.state(f.db, f.env, 4);
  CHECK_EQ(served.families[0].calls, u64(4));
  CHECK(served.families[0].phase == FamilyPhase::Served);
  CHECK(served.complete);
}

TEST(a_family_is_read_and_searched_in_the_kind_the_bootstrap_is_in) {
  Fixture f;
  Family const fp32 = familyOf("2:1K:8:256:212");
  f.add(fp32.fft, {}, reading(1576, 1, MIN_CALLS));
  f.add(fp32.fft, {{"WMUL", "1"}}, reading(1540, 1, 64));

  Bootstrap const prp{nvidia(), PROBE, {fp32}, true, TestKind::PRP};
  CHECK(prp.state(f.db, f.env, 64).families[0].phase == FamilyPhase::Served);

  Bootstrap const ll{nvidia(), PROBE, {fp32}, true, TestKind::LL};
  BootstrapState const s = ll.state(f.db, f.env, 64);
  CHECK(s.kind == TestKind::LL);
  CHECK(s.families[0].phase == FamilyPhase::Unread);
  f.add(fp32.fft, {}, reading(1570, 1, MIN_CALLS), TestKind::LL);
  CHECK(ll.state(f.db, f.env, 64).families[0].phase == FamilyPhase::Owed);
}

TEST(a_family_that_cannot_run_at_its_defaults_is_held) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212"), familyOf("1:1K:8:256:202")}};
  f.add(FFTConfig{"2:1K:8:256:212"}, {}, reading(1576, 1, MIN_CALLS));
  f.add(FFTConfig{"1:1K:8:256:202"}, {}, Measurement{.status = Status::NoCompile, .ts = 1});

  BootstrapState const s = b.state(f.db, f.env, 64);
  CHECK(s.families[1].phase == FamilyPhase::Held);
  CHECK(s.families[0].phase == FamilyPhase::Owed);
}

TEST(turned_off_it_searches_nothing) {
  Fixture f;
  Bootstrap const b{nvidia(), PROBE, {familyOf("2:1K:8:256:212")}, false};
  BootstrapState const s = b.state(f.db, f.env, 64);
  CHECK(s.complete);
  CHECK(s.families[0].phase == FamilyPhase::Skipped);
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

  // And what each family runs at under the lines is exactly its own set.
  CHECK_EQ(configText(underDefaults(nvidia(), fp32.fft, TestKind::PRP, d)),
           std::string{"TAIL_KERNELS=3,TAIL_TRIGS32=1,WMUL=1"});
  CHECK_EQ(configText(underDefaults(nvidia(), gf31.fft, TestKind::PRP, d)), std::string{"MODM31=2,TAIL_KERNELS=3"});

  // A family with no set of its own runs at the global line, less what does not reach it.
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
  Family const fp32 = familyOf("2:1K:8:256:212");

  // Nothing published, no lines.
  CHECK_EQ(linesText(publishedLines(nvidia(), probe, TestKind::PRP, {})), std::string{"-"});

  // FP64's evidence is what production runs at the probe: the cheapest entry there, not a dearer one there, a cheaper
  // one of another size that does not reach it, or one of the other kind.  Its TABMUL_CHAIN=1 changes the rounding,
  // which nothing reads for what the lines reach, so it is left out.  The hybrid has nothing published, so it has no
  // line.
  std::vector<SelectionEntry> const published{
    publishedAt("512:15:512:101", probe, 1700, {{"TABMUL_CHAIN", "1"}, {"TAIL_KERNELS", "3"}}),
    publishedAt("512:15:512:212", probe, 1800, {{"WMUL", "1"}}),
    publishedAt("256:13:512:101", 60'000'000, 900, {{"WMUL", "1"}}),
    publishedAt("512:15:512:102", probe, 1500, {{"WMUL", "1"}}, TestKind::LL)};
  Defaults const d = publishedLines(nvidia(), probe, TestKind::PRP, published);
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"512:15:512:101"}, TestKind::PRP, d)),
           std::string{"TAIL_KERNELS=3"});
  CHECK_EQ(configText(underDefaults(nvidia(), fp32.fft, TestKind::PRP, d)), std::string{"TAIL_KERNELS=3"});

  // Where nothing of a type covers the probe, the entry nearest it speaks for the type, however much it costs.
  CHECK(maxExp(FFTConfig{"256:14:512:101"}) < probe);
  std::vector<SelectionEntry> const below{publishedAt("256:13:512:101", 60'000'000, 900, {{"WMUL", "1"}}),
                                          publishedAt("256:14:512:101", 64'000'000, 950, {{"TAIL_KERNELS", "3"}})};
  Defaults const nearest = publishedLines(nvidia(), probe, TestKind::PRP, below);
  CHECK_EQ(configText(underDefaults(nvidia(), FFTConfig{"512:15:512:212"}, TestKind::PRP, nearest)),
           std::string{"TAIL_KERNELS=3"});
}

TEST(an_entry_still_at_the_built_in_defaults_is_no_evidence_for_the_lines) {
  u64 const probe = 118'063'003;
  Family const fp64 = familyOf("512:15:512:212");
  std::vector<SelectionEntry> const searchedOnce{publishedAt("512:15:512:212", probe, 1680, {{"WMUL", "1"}})};
  CHECK_EQ(linesText(publishedLines(nvidia(), probe, TestKind::PRP, searchedOnce)), std::string{"WMUL=1"});

  // The cheapest entry at the probe has not been searched: its defaults say nothing.  Nor does the unsearched FFT61
  // entry beside it, which would otherwise pull WMUL off the global line.
  std::vector<SelectionEntry> unsearched = searchedOnce;
  unsearched.push_back(publishedAt("512:15:512:112", probe, 1650, {}));
  unsearched.push_back(publishedAt("3:1K:8:512:202", probe, 2200, {}));
  CHECK_EQ(linesText(publishedLines(nvidia(), probe, TestKind::PRP, unsearched)), std::string{"WMUL=1"});

  // Once it has been searched, what it found is the evidence.
  std::vector<SelectionEntry> searched = unsearched;
  searched.push_back(publishedAt("512:15:512:112", probe, 1600, {{"TAIL_KERNELS", "3"}}));
  Defaults const d = publishedLines(nvidia(), probe, TestKind::PRP, searched);
  CHECK_EQ(configText(underDefaults(nvidia(), fp64.fft, TestKind::PRP, d)), std::string{"TAIL_KERNELS=3"});
}

TEST(a_family_is_searched_on_its_types_cheapest_reading_at_the_defaults_and_stays_there) {
  // Two readings of FP64 at the probe at the built-in defaults, the cheaper not the smallest shape's default variant.
  Fixture f;
  Family const smallest = familyOf("512:15:512:212");
  Bootstrap const b{nvidia(), PROBE, {smallest}};
  CHECK_EQ(b.familiesIn(f.db, f.env).front().fft.spec(), smallest.fft.spec());

  f.add(FFTConfig{"512:15:512:212"}, {}, reading(1700, 1, MIN_CALLS));
  f.add(FFTConfig{"512:15:512:112"}, {}, reading(1650, 1, MIN_CALLS));
  // A cheaper reading under other options, or one not yet concluded, is not a reading of the defaults.
  f.add(FFTConfig{"512:15:512:202"}, {{"WMUL", "1"}}, reading(1500, 1, MIN_CALLS));
  f.add(FFTConfig{"512:15:512:201"}, {}, reading(1400, 1, 1));
  CHECK_EQ(b.familiesIn(f.db, f.env).front().fft.spec(), std::string{"512:15:512:112"});
  CHECK(!b.chosen(f.db, f.env));

  // Recorded, it stays there whatever is read later.
  CHECK(f.db.add(BootRow{.sess = f.sess, .fft = "512:15:512:112", .probe = PROBE, .ts = 2}));
  CHECK(b.chosen(f.db, f.env));
  CHECK(b.unrecorded(f.db, f.env).empty());
  f.add(FFTConfig{"512:15:512:211"}, {}, reading(1600, 1, MIN_CALLS));
  CHECK_EQ(b.familiesIn(f.db, f.env).front().fft.spec(), std::string{"512:15:512:112"});

  // A choice recorded at another probe is another bootstrap's.
  Bootstrap const other{nvidia(), PROBE + 2, {smallest}};
  CHECK(!other.chosen(f.db, f.env));
}

TEST(a_key_only_amd_and_nvidia_can_choose_is_not_a_move_elsewhere) {
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
