// Copyright (C) Jason Lynch

// The per-entry search's pure half: what a strategy offers as one step from a best option set, and which of those
// steps the rows already answer -- from the option table and fixed option sets, with no GPU.

#include "Probe.h"

#include "Bootstrap.h"
#include "TuneDB.h"

#include "test.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .computeCapability = 806}; }

Env amd() { return {.isAmd = true}; }

std::vector<std::string> textsOf(const ProbeList& list, const std::string& stage) {
  std::vector<std::string> out;
  for (const Probe& p : list.probes) {
    if (p.stage == stage) { out.push_back(p.text); }
  }
  return out;
}

std::map<std::string, size_t> stagesOf(const ProbeList& list) {
  std::map<std::string, size_t> out;
  for (const Probe& p : list.probes) { ++out[p.stage]; }
  return out;
}

const Probe* findProbe(const ProbeList& list, const std::string& text) {
  auto const it = std::ranges::find_if(list.probes, [&](const Probe& p) { return p.text == text; });
  return it == list.probes.end() ? nullptr : &*it;
}

}  // namespace

TEST(a_probe_list_is_a_pure_function_of_the_table_and_the_best_set) {
  FFTConfig const fft{"512:15:512:212"};
  ProbeList const list = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Groups});

  // Each group in declaration order; the width pass's structural keys one step at a time, then its other axes as one
  // bin, fewest moved first.  LDSSWIZ_W is not offered: it needs LDSPAD_W=0, which is a step of its own.
  std::vector<std::string> const width{"SHUFL_BYTES_W=4",
                                       "SHUFL_BYTES_W=16",
                                       "LDSPAD_W=0",
                                       "HOIST_W=1",
                                       "HOIST_W=2",
                                       "HOIST_W=3",
                                       "ZEROHACK_W=0",
                                       "WMUL=1",
                                       "WMUL=4",
                                       "HOIST_W=1,ZEROHACK_W=0",
                                       "HOIST_W=2,ZEROHACK_W=0",
                                       "HOIST_W=3,ZEROHACK_W=0",
                                       "HOIST_W=1,WMUL=1",
                                       "HOIST_W=1,WMUL=4",
                                       "HOIST_W=2,WMUL=1",
                                       "HOIST_W=2,WMUL=4",
                                       "HOIST_W=3,WMUL=1",
                                       "HOIST_W=3,WMUL=4",
                                       "WMUL=1,ZEROHACK_W=0",
                                       "WMUL=4,ZEROHACK_W=0",
                                       "HOIST_W=1,WMUL=1,ZEROHACK_W=0",
                                       "HOIST_W=1,WMUL=4,ZEROHACK_W=0",
                                       "HOIST_W=2,WMUL=1,ZEROHACK_W=0",
                                       "HOIST_W=2,WMUL=4,ZEROHACK_W=0",
                                       "HOIST_W=3,WMUL=1,ZEROHACK_W=0",
                                       "HOIST_W=3,WMUL=4,ZEROHACK_W=0"};
  CHECK(textsOf(list, "Width") == width);

  // Memory has seven axes on NVIDIA with PTX: two bins, the first cut at MAX_POINTS.
  std::map<std::string, size_t> const stages{{"Height", 10}, {"Memory 1", 64}, {"Memory 2", 29}, {"Placement", 5},
                                             {"Queues", 7},  {"Tail", 11},     {"Width", 26}};
  CHECK(stagesOf(list) == stages);

  // No duplicates, never the best set itself, every set canonical.
  std::set<std::string> seen;
  for (const Probe& p : list.probes) {
    CHECK(!p.config.empty());
    CHECK(seen.insert(configText(p.config)).second);
    CHECK(p.config == canonicalConfig(nvidia(), fft, p.config));
  }

  // The same inputs give the same list, and a different best set a list of its own: from WMUL=1, the move back is on
  // offer and WMUL=1 itself is not.
  ProbeList const again = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Groups});
  CHECK_EQ(again.probes.size(), list.probes.size());
  for (size_t i = 0; i < std::min(again.probes.size(), list.probes.size()); ++i) {
    CHECK(again.probes[i].config == list.probes[i].config);
  }
  std::vector<std::string> const fromWmul1 = textsOf(probesOf(nvidia(), fft, {{"WMUL", "1"}}, {}), "Width");
  CHECK(std::ranges::count(fromWmul1, std::string{"WMUL=2"}) == 1);
  CHECK(std::ranges::count(fromWmul1, std::string{"WMUL=1"}) == 0);
}

TEST(single_is_the_bootstraps_moves_over_every_group) {
  FFTConfig const fft{"2:1K:8:256:212"};
  ProbeList const list = probesOf(nvidia(), fft, {{"TAIL_KERNELS", "3"}}, {.kind = Strategy::Kind::Single});

  std::set<std::string> moves;
  for (Group const group : allGroups()) {
    for (const Move& m : movesWithin(nvidia(), fft, {{"TAIL_KERNELS", "3"}}, group)) {
      moves.insert(configText(m.config));
    }
  }

  std::set<std::string> probes;
  for (const Probe& p : list.probes) {
    CHECK_EQ(p.moves.size(), size_t(1));
    probes.insert(configText(p.config));
  }
  CHECK(probes == moves);
}

TEST(a_bin_holds_at_most_four_axes_and_a_stage_at_most_64_points) {
  ProbeList const list = probesOf(nvidia(), FFTConfig{"2:1K:8:256:212"}, {}, {});

  // Tail: TAIL_KERNELS, TAIL_TRIGS32, TAIL_TRIGS61, TABMUL_CHAIN32 are 4 x 3 x 2 x 2 points, the best one of them;
  // TABMUL_CHAIN61 is a bin of its own.  Memory's first bin would be 5 x 3 x 2 x 5 and is cut.
  std::map<std::string, size_t> const stages = stagesOf(list);
  CHECK_EQ(stages.at("Tail 1"), size_t(4 * 3 * 2 * 2 - 1));
  CHECK_EQ(stages.at("Tail 2"), size_t(1));
  CHECK_EQ(stages.at("Memory 1"), size_t(MAX_POINTS));
  CHECK_EQ(stages.at("Queues"), size_t(2 * 2 * 2 * 2 - 1));

  // Within a stage, fewer axes moved always comes first, so a cut keeps every one-step move.
  std::map<std::string, size_t> most;
  size_t memorySingles = 0;
  for (const Probe& p : list.probes) {
    CHECK(p.moves.size() <= MAX_PERMUTE);
    CHECK(p.moves.size() >= most[p.stage]);
    most[p.stage] = p.moves.size();
    if (p.stage == "Memory 1" && p.moves.size() == 1) { ++memorySingles; }
  }
  CHECK_EQ(memorySingles, size_t(4 + 2 + 1 + 4));
}

TEST(how_many_axes_a_bin_permutes_and_where_it_is_cut_are_the_strategys) {
  FFTConfig const fft{"2:1K:8:256:212"};

  // Two axes a bin: Tail's five are 4 x 3, 2 x 2 and 2.
  ProbeList const pairs = probesOf(nvidia(), fft, {}, {.maxPermute = 2});
  std::map<std::string, size_t> const paired = stagesOf(pairs);
  CHECK_EQ(paired.at("Tail 1"), size_t(4 * 3 - 1));
  CHECK_EQ(paired.at("Tail 2"), size_t(2 * 2 - 1));
  CHECK_EQ(paired.at("Tail 3"), size_t(1));
  CHECK(std::ranges::all_of(pairs.probes, [](const Probe& p) { return p.tier > 1 || p.moves.size() <= 2; }));

  // Uncut, Memory's first bin is its whole cross product, and its last points move every axis of the bin.
  std::vector<std::string> const uncut = textsOf(probesOf(nvidia(), fft, {}, {.maxPoints = NO_LIMIT}), "Memory 1");
  CHECK_EQ(uncut.size(), size_t(5 * 3 * 2 * 5 - 1));
  std::vector<std::string> const cut = textsOf(probesOf(nvidia(), fft, {}, {}), "Memory 1");
  CHECK(std::equal(cut.begin(), cut.end(), uncut.begin()));

  // A group kept whole is one stage, whose points reach every axis it has.
  ProbeList const whole = probesOf(nvidia(), fft, {}, {.maxPermute = NO_LIMIT, .maxPoints = NO_LIMIT});
  std::map<std::string, size_t> const stages = stagesOf(whole);
  CHECK(!stages.contains("Memory 1"));
  CHECK(!stages.contains("Tail 1"));
  CHECK_EQ(stages.at("Tail"), size_t(4 * 3 * 2 * 2 * 2 - 1));
  size_t const memoryAxes =
    size_t(std::ranges::count_if(whole.axes, [](const Axis& a) { return a.option->group == Group::Memory; }));
  size_t most = 0;
  for (const Probe& p : whole.probes) {
    if (p.stage == "Memory") { most = std::max(most, p.moves.size()); }
  }
  CHECK_EQ(most, memoryAxes);
  CHECK(stages.at("Memory") > uncut.size());
}

TEST(a_structural_step_stands_alone_and_opens_its_dependents) {
  FFTConfig const fft{"1K:13:256:212"};

  // In place, INPLACE=0 is a single step, and none of the layout keys it would open is offered.
  ProbeList const inPlace = probesOf(nvidia(), fft, {}, {});
  CHECK(findProbe(inPlace, "INPLACE=0") != nullptr);
  CHECK(std::ranges::none_of(inPlace.probes, [](const Probe& p) { return p.text.find("IN_WG") != std::string::npos; }));

  // Out of place, the five layout keys are two bins in declaration order, and INPLACE=1 is a step back.
  ProbeList const outOfPlace = probesOf(nvidia(), fft, {{"INPLACE", "0"}}, {});
  std::map<std::string, size_t> const stages = stagesOf(outOfPlace);
  CHECK_EQ(stages.at("Placement 1"), size_t(MAX_POINTS));
  CHECK_EQ(stages.at("Placement 2"), size_t(4));
  CHECK(findProbe(outOfPlace, "INPLACE=1") != nullptr);
  CHECK(findProbe(outOfPlace, "IN_SIZEX=32") != nullptr);
  CHECK(findProbe(outOfPlace, "PAD=512") != nullptr);
}

TEST(memory_probes_move_access_classes_never_the_packed_integer) {
  FFTConfig const fft{"512:15:512:212"};

  for (const Env& env : {nvidia(), amd()}) {
    ProbeList const list = probesOf(env, fft, {}, {.kind = Strategy::Kind::Single});
    for (const Probe& p : list.probes) {
      u32 const loads = u32(useValue(p.config, "LOADS", 0));
      u32 const stores = u32(useValue(p.config, "STORES", 0));
      for (const AccessClass& cls : accessClasses()) {
        int const load = int(getDigit(loads, cls.digit));
        int const store = int(getDigit(stores, cls.digit));
        if (cls.pairs.empty()) {
          if (!cls.loadModes.empty()) { CHECK(std::ranges::count(usableLoadModes(env, cls), load) == 1); }
          if (!cls.storeModes.empty()) { CHECK(std::ranges::count(usableStoreModes(env, cls), store) == 1); }
        } else {
          CHECK(std::ranges::count(usablePairs(env, cls), std::pair{load, store}) == 1);
        }
      }
    }
  }

  // The carry shuttle moves its load and store together, so a build failure there is pinned on neither key.
  ProbeList const nv = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Single});
  const Probe* const shuttle = findProbe(nv, "LOADS=40,STORES=20");
  CHECK(shuttle != nullptr);
  if (shuttle) { CHECK(shuttle->key.empty()); }

  // AMD has the nontemporal mode and none of the PTX ones.
  ProbeList const radeon = probesOf(amd(), fft, {}, {.kind = Strategy::Kind::Single});
  CHECK(findProbe(radeon, "LOADS=1") != nullptr);
  CHECK(findProbe(radeon, "LOADS=2") == nullptr);
  CHECK(findProbe(nv, "LOADS=1") == nullptr);
  CHECK(findProbe(nv, "LOADS=2") != nullptr);
}

TEST(permute_takes_exactly_the_keys_named_and_is_not_cut) {
  FFTConfig const fft{"512:15:512:212"};

  ProbeList const two = probesOf(nvidia(), fft, {}, parseStrategy("permute:TAIL_KERNELS+WMUL"));
  CHECK_EQ(two.probes.size(), size_t(4 * 3 - 1));
  for (const Probe& p : two.probes) {
    CHECK_EQ(p.stage, std::string{"permute"});
    for (const auto& [key, value] : p.config) { CHECK(key == "TAIL_KERNELS" || key == "WMUL"); }
  }

  // LOADS brings every class it has a digit in, the coupled carry shuttle included: 5 x 3 x 2 x 5 x 5.
  ProbeList const loads = probesOf(nvidia(), fft, {}, parseStrategy("permute:LOADS"));
  CHECK_EQ(loads.probes.size(), size_t(5 * 3 * 2 * 5 * 5 - 1));
}

TEST(a_strategy_is_read_and_written_the_same_way) {
  for (const char* text : {"hybrid", "single", "groups", "permute:PAD+TAIL_KERNELS+IN_SIZEX"}) {
    CHECK_EQ(parseStrategy(text).text(), std::string{text});
  }
  CHECK(parseStrategy("permute:WMUL+WMUL").keys.size() == 1);

  auto refused = [](const char* text) {
    try {
      (void)parseStrategy(text);
    } catch (const std::string&) { return true; }
    return false;
  };
  CHECK(refused("fast"));
  CHECK(refused("permute:"));
  CHECK(refused("permute:WMUL+NOSUCHKEY"));
  CHECK(refused("permute:STATS"));
}

TEST(a_step_the_table_would_not_offer_takes_its_dependents_with_it) {
  // FFT3261 at width 1K offers L2_STRIPING up to width/64 = 16 alone, and up to width/128 = 8 beside MULTI_Q=1.  So
  // from L2_STRIPING=16 the step to MULTI_Q=1 returns L2_STRIPING to its default rather than asking the host to clamp
  // it, and says so.
  FFTConfig const fft{"2:1K:8:256:212"};
  ProbeList const list = probesOf(nvidia(), fft, {{"L2_STRIPING", "16"}}, {.kind = Strategy::Kind::Single});
  const Probe* const multiQ = findProbe(list, "L2_STRIPING=0,MULTI_Q=1");
  CHECK(multiQ != nullptr);
  if (multiQ) {
    CHECK_EQ(configText(multiQ->config), std::string{"MULTI_Q=1"});
    CHECK(multiQ->key.empty());
  }
}

TEST(a_probe_is_answered_by_a_row_that_agrees_on_what_it_depends_on) {
  FFTConfig const fft{"512:15:512:212"};
  UseConfig const best{{"WMUL", "1"}};
  ProbeList const list = probesOf(nvidia(), fft, best, {.kind = Strategy::Kind::Single});

  // ZEROHACK_W depends on nothing, so its reading from before WMUL moved still answers it.
  const Probe* const zerohack = findProbe(list, "ZEROHACK_W=0");
  CHECK(zerohack != nullptr);
  if (zerohack) {
    CHECK(answeredBy(nvidia(), fft, list, *zerohack, {{"ZEROHACK_W", "0"}}));
    CHECK(answeredBy(nvidia(), fft, list, *zerohack, {{"ZEROHACK_W", "0"}, {"TAIL_KERNELS", "3"}}));
    CHECK(!answeredBy(nvidia(), fft, list, *zerohack, {{"TAIL_KERNELS", "3"}}));
  }

  // LDSPAD_W depends on WMUL: a reading at the old WMUL does not answer it, one at the new WMUL does, whatever else
  // it ran with.
  const Probe* const ldspad = findProbe(list, "LDSPAD_W=0");
  CHECK(ldspad != nullptr);
  if (ldspad) {
    CHECK(!answeredBy(nvidia(), fft, list, *ldspad, {{"LDSPAD_W", "0"}}));
    CHECK(answeredBy(nvidia(), fft, list, *ldspad, {{"LDSPAD_W", "0"}, {"WMUL", "1"}}));
    CHECK(answeredBy(nvidia(), fft, list, *ldspad, {{"LDSPAD_W", "0"}, {"WMUL", "1"}, {"TAIL_KERNELS", "3"}}));
  }

  // An access class is answered by its own digit, whatever the other digits of the key were.
  const Probe* const trig = findProbe(list, "LOADS=50000");
  CHECK(trig != nullptr);
  if (trig) {
    CHECK(answeredBy(nvidia(), fft, list, *trig, {{"LOADS", "50003"}}));
    CHECK(!answeredBy(nvidia(), fft, list, *trig, {{"LOADS", "40000"}}));
  }
}

TEST(a_restart_is_a_fixed_draw_of_every_axis_the_table_offers) {
  FFTConfig const fft{"512:15:512:212"};
  std::string const entry = "512:15:512:212 prp short32";

  for (const Env& env : {nvidia(), amd()}) {
    // The same sequence however often it is asked for, and another for another entry.
    CHECK(restartOf(env, fft, entry, 0) == restartOf(env, fft, entry, 0));
    CHECK(restartOf(env, fft, entry, 0) != restartOf(env, fft, "512:15:512:212 prp long32", 0));

    constexpr u32 DRAWS = 1000;
    std::set<std::string> distinct;
    std::map<std::string, std::set<std::pair<int, int>>> offered;
    std::map<std::string, std::set<std::pair<int, int>>> drawn;
    for (u32 k = 0; k < DRAWS; ++k) {
      UseConfig const config = restartOf(env, fft, entry, k);
      distinct.insert(configText(config));

      // Canonical, and every key at a value the table offers it beside the rest.
      CHECK(canonicalConfig(env, fft, config) == config);
      for (const auto& [key, value] : config) {
        const Option* const option = findOption(key);
        if (option->compound) { continue; }
        std::vector<int> const values = option->valuesFor(env, fft, config);
        CHECK(std::ranges::find(values, parseInt<int>(value).value_or(-1)) != values.end());
      }

      for (const Axis& axis : axesOf(env, fft, config)) {
        offered[axis.name].insert(axis.values.begin(), axis.values.end());
        drawn[axis.name].insert(axis.values[axis.current]);
      }
    }

    // Nearly every draw a new point, and every position of every axis drawn somewhere: no assignment is out of reach.
    CHECK(distinct.size() > DRAWS * 99 / 100);
    CHECK(offered == drawn);
  }
}
