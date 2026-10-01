// Copyright (C) Jason Lynch

// The combo tiers and the structural branches: which groups combine, what each group's best answers are among an
// entry's readings, the points combining them and their order, and the branches an entry is searched in -- from the
// option table and fixed readings, with no GPU.

#include "Probe.h"

#include "Bootstrap.h"
#include "TuneDB.h"

#include "test.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .computeCapability = 806}; }

std::vector<std::string> textsOf(const ProbeList& list, const std::string& stage) {
  std::vector<std::string> out;
  for (const Probe& p : list.probes) {
    if (p.stage == stage) { out.push_back(p.text); }
  }
  return out;
}

std::vector<std::string> namesOf(const std::vector<Group>& groups) {
  std::vector<std::string> out;
  for (Group const g : groups) { out.emplace_back(toString(g)); }
  return out;
}

// FP64 512:15:512:212 on NVIDIA cc 8.6: the defaults, and one reading off them in each of four groups, each gain
// twice the last so that no two sums of them tie.  TAIL_KERNELS=1 is the third answer of Tail, one past comboTop.
std::vector<Reading> fp64Readings() {
  return {{{}, 100},
          {{{"ZEROHACK_H", "0"}}, 100.1},
          {{{"WMUL", "1"}}, 100.2},
          {{{"TAIL_KERNELS", "3"}}, 100.4},
          {{{"L2_STRIPING", "2"}}, 100.8},
          {{{"LOADS", "2"}}, 101.6},
          {{{"TAIL_TRIGS", "1"}}, 103.2},
          {{{"TAIL_KERNELS", "1"}}, 106.4}};
}

}  // namespace

TEST(the_cluster_graph_of_an_entry_follows_its_branch) {
  // FP64 at variant 212 away from 1024: TABMUL_CHAIN is inert, so nothing couples the width pass to the tail, and
  // Width is a cluster of its own.  M=1 leaves the middle chains one value each, so Middle is present only out of
  // place, where the two LDS transposes apply.
  FFTConfig const fp64{"512:15:512:212"};
  ClusterGraph const inPlace = clusterGraph(nvidia(), fp64, {});
  CHECK(namesOf(inPlace.topTier) == (std::vector<std::string>{"Placement", "Memory", "Queues"}));
  CHECK_EQ(inPlace.clusters.size(), size_t(2));
  CHECK(namesOf(inPlace.clusters.at(0)) == (std::vector<std::string>{"Tail", "Height"}));
  CHECK(namesOf(inPlace.clusters.at(1)) == (std::vector<std::string>{"Width"}));

  ClusterGraph const outOfPlace = clusterGraph(nvidia(), fp64, {{"INPLACE", "0"}});
  CHECK_EQ(outOfPlace.clusters.size(), size_t(3));
  CHECK(namesOf(outOfPlace.clusters.at(0)) == (std::vector<std::string>{"Middle"}));

  // At variant 101 the width pass reads TABMUL_CHAIN, which joins it to the tail.
  ClusterGraph const coupled = clusterGraph(nvidia(), FFTConfig{"512:15:512:101"}, {});
  CHECK(namesOf(coupled.clusters.back()) == (std::vector<std::string>{"Tail", "Width", "Height"}));

  // A hybrid with a GF61 part has Arith in the top tier.
  ClusterGraph const hybrid = clusterGraph(nvidia(), FFTConfig{"2:1K:8:256:212"}, {});
  CHECK(namesOf(hybrid.topTier) == (std::vector<std::string>{"Placement", "Memory", "Queues", "Arith"}));
}

TEST(a_combo_stage_is_the_cross_product_of_each_groups_best_answers) {
  FFTConfig const fft{"512:15:512:212"};
  ProbeList const list = probesOf(nvidia(), fft, {}, {}, fp64Readings());

  // Tier 2 is Tail+Height alone, Width being a cluster of one: Tail's answers are TAIL_KERNELS=3 and TAIL_TRIGS=1,
  // Height's ZEROHACK_H=0.  A point moving one group only is one the group's own stage already offers.
  CHECK(textsOf(list, "Tail+Height") ==
        (std::vector<std::string>{"TAIL_KERNELS=3,ZEROHACK_H=0", "TAIL_TRIGS=1,ZEROHACK_H=0"}));

  // Tier 3 combines Placement, Memory, and each cluster whole -- Queues has no answer but the defaults, and adds
  // nothing -- in falling order of summed gain.  Tail+Height taken whole has ZEROHACK_H=0 and TAIL_KERNELS=3 as its
  // two best answers, so the two never appear together here.
  std::vector<std::string> const all{"WMUL=1,ZEROHACK_H=0",                           // -0.3%
                                     "TAIL_KERNELS=3,WMUL=1",                         // -0.6%
                                     "L2_STRIPING=2,ZEROHACK_H=0",                    // -0.9%
                                     "L2_STRIPING=2,WMUL=1",                          // -1.0%
                                     "L2_STRIPING=2,WMUL=1,ZEROHACK_H=0",             // -1.1%
                                     "L2_STRIPING=2,TAIL_KERNELS=3",                  // -1.2%
                                     "L2_STRIPING=2,TAIL_KERNELS=3,WMUL=1",           // -1.4%
                                     "LOADS=2,ZEROHACK_H=0",                          // -1.7%
                                     "LOADS=2,WMUL=1",                                // -1.8%
                                     "LOADS=2,WMUL=1,ZEROHACK_H=0",                   // -1.9%
                                     "LOADS=2,TAIL_KERNELS=3",                        // -2.0%
                                     "LOADS=2,TAIL_KERNELS=3,WMUL=1",                 // -2.2%
                                     "L2_STRIPING=2,LOADS=2",                         // -2.4%
                                     "L2_STRIPING=2,LOADS=2,ZEROHACK_H=0",            // -2.5%
                                     "L2_STRIPING=2,LOADS=2,WMUL=1",                  // -2.6%
                                     "L2_STRIPING=2,LOADS=2,WMUL=1,ZEROHACK_H=0",     // -2.7%
                                     "L2_STRIPING=2,LOADS=2,TAIL_KERNELS=3",          // -2.8%
                                     "L2_STRIPING=2,LOADS=2,TAIL_KERNELS=3,WMUL=1"};  // -3.0%
  CHECK(textsOf(list, "all") == all);

  for (const Probe& p : list.probes) { CHECK_EQ(p.tier, p.stage == "all" ? 3u : p.stage == "Tail+Height" ? 2u : 1u); }

  // Once the cross point of Tail and Height is the best set, it is the background, and the other readings are
  // answers against it: the defaults' Tail and Height are now the runners-up, so the way back is a point too.
  std::vector<Reading> moved = fp64Readings();
  moved.insert(moved.begin(), {{{"TAIL_KERNELS", "3"}, {"ZEROHACK_H", "0"}}, 99});
  ProbeList const from = probesOf(nvidia(), fft, moved.front().config, {}, moved);
  CHECK(textsOf(from, "Tail+Height") ==
        (std::vector<std::string>{"TAIL_KERNELS=2,ZEROHACK_H=1", "TAIL_KERNELS=2,TAIL_TRIGS=1,ZEROHACK_H=1"}));
}

TEST(each_tier_the_search_is_given_adds_its_stages_and_nothing_else) {
  FFTConfig const fft{"512:15:512:212"};
  std::vector<Reading> const readings = fp64Readings();
  ProbeList const groups = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Groups}, readings);

  auto withTiers = [&](u32 tiers, u32 top = COMBO_TOP) {
    return probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Hybrid, .comboTop = top, .comboTiers = tiers},
                    readings);
  };

  // Every tier's list starts with the one below it, whole.
  auto prefixed = [](const ProbeList& shorter, const ProbeList& longer) {
    if (shorter.probes.size() > longer.probes.size()) { return false; }
    for (size_t i = 0; i < shorter.probes.size(); ++i) {
      if (shorter.probes[i].config != longer.probes[i].config || shorter.probes[i].stage != longer.probes[i].stage) {
        return false;
      }
    }
    return true;
  };
  ProbeList const one = withTiers(1);
  ProbeList const two = withTiers(2);
  ProbeList const three = withTiers(3);
  CHECK_EQ(one.probes.size(), groups.probes.size());
  CHECK(prefixed(groups, one) && prefixed(one, groups));
  CHECK_EQ(two.probes.size(), groups.probes.size() + 2);
  CHECK(prefixed(one, two));
  CHECK_EQ(three.probes.size(), two.probes.size() + 18);
  CHECK(prefixed(two, three));

  // comboTop=2 keeps the best answer besides the background's: TAIL_KERNELS=3 but not TAIL_TRIGS=1.  comboTop=1 keeps
  // only the background's, which combines into nothing.
  CHECK(textsOf(withTiers(2, 2), "Tail+Height") == std::vector<std::string>{"TAIL_KERNELS=3,ZEROHACK_H=0"});
  CHECK_EQ(withTiers(3, 1).probes.size(), groups.probes.size());
}

TEST(an_answer_is_ranked_by_what_it_did_against_the_same_settings_of_everything_else) {
  FFTConfig const fft{"512:15:512:212"};

  // Two readings put Tail where TAIL_KERNELS=3 does.  The cheaper also has WMUL=1, and no reading has WMUL=1 with Tail
  // where the background has it, so nothing says whether TAIL_KERNELS=3 or WMUL=1 made it cheap: TAIL_KERNELS=3 is
  // ranked by the dearer one alone, behind TAIL_KERNELS=1.
  std::vector<Reading> readings{{{}, 100},
                                {{{"ZEROHACK_H", "0"}}, 100.1},
                                {{{"TAIL_TRIGS", "1"}}, 100.2},
                                {{{"TAIL_KERNELS", "3"}, {"WMUL", "1"}}, 100.3},
                                {{{"TAIL_KERNELS", "1"}}, 100.5},
                                {{{"TAIL_KERNELS", "3"}}, 104}};
  ProbeList const list = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Hybrid, .comboTiers = 2}, readings);
  CHECK(textsOf(list, "Tail+Height") ==
        (std::vector<std::string>{"TAIL_TRIGS=1,ZEROHACK_H=0", "TAIL_KERNELS=1,ZEROHACK_H=0"}));

  // An answer read only beside another group's move is no answer at all.
  readings.pop_back();
  readings.pop_back();
  ProbeList const joint = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Hybrid, .comboTiers = 2}, readings);
  CHECK(textsOf(joint, "Tail+Height") == std::vector<std::string>{"TAIL_TRIGS=1,ZEROHACK_H=0"});

  // Out of place, only the readings of that branch count: the in-place ones have nothing to say about it.
  std::vector<Reading> outOfPlace{{{{"INPLACE", "0"}}, 101}, {{{"INPLACE", "0"}, {"TAIL_KERNELS", "1"}}, 101.5}};
  outOfPlace.insert(outOfPlace.end(), readings.begin(), readings.end());
  ProbeList const branch =
    probesOf(nvidia(), fft, {{"INPLACE", "0"}}, {.kind = Strategy::Kind::Hybrid, .comboTiers = 2}, outOfPlace, false);
  CHECK(std::ranges::none_of(branch.probes, [](const Probe& p) { return p.tier > 1; }));
}

TEST(an_answer_read_against_an_older_background_is_ranked_by_what_it_did_there) {
  // Tail was raced from the defaults, where TAIL_KERNELS=1 won; WMUL=1 then won from there, and TAIL_KERNELS=3 from
  // that.  TAIL_KERNELS=0 was read against the background as it stands, and TAIL_TRIGS=1 only against the defaults,
  // which are dearer: it is the cheaper of the two only against a background it never had.  Chained through the
  // contexts both were read in, TAIL_TRIGS=1 costs 0.4% more than TAIL_KERNELS=3 and TAIL_KERNELS=0 0.5%.
  FFTConfig const fft{"512:15:512:212"};
  std::vector<Reading> const readings{{{{"TAIL_KERNELS", "3"}, {"WMUL", "1"}}, 98.8},
                                      {{{"TAIL_KERNELS", "1"}, {"WMUL", "1"}}, 99},
                                      {{{"TAIL_KERNELS", "0"}, {"WMUL", "1"}}, 99.3},
                                      {{{"TAIL_KERNELS", "1"}}, 99.5},
                                      {{{"TAIL_TRIGS", "1"}}, 99.7},
                                      {{}, 100},
                                      {{{"ZEROHACK_H", "0"}}, 100.1}};
  ProbeList const list =
    probesOf(nvidia(), fft, readings.front().config, {.kind = Strategy::Kind::Hybrid, .comboTiers = 2}, readings);
  CHECK(textsOf(list, "Tail+Height") ==
        (std::vector<std::string>{"TAIL_KERNELS=1,ZEROHACK_H=0", "TAIL_KERNELS=2,TAIL_TRIGS=1,ZEROHACK_H=0"}));
}

TEST(an_answer_read_in_several_contexts_is_ranked_by_them_all_as_exactly_as_each_was_read) {
  // TAIL_TRIGS=1 costs 0.3% against the defaults, read closely, and 2% beside WMUL=1, read loosely; TAIL_KERNELS=1
  // costs 0.5% against the defaults.  Each context counts for as much as it was read exactly, so TAIL_TRIGS=1 is the
  // better answer: by the plain mean of the two it would not be.
  FFTConfig const fft{"512:15:512:212"};
  std::vector<Reading> const readings{{{}, 100.02, 0.01},
                                      {{{"ZEROHACK_H", "0"}}, 100.12, 0.01},
                                      {{{"TAIL_TRIGS", "1"}}, 100.32, 0.01},
                                      {{{"TAIL_KERNELS", "1"}}, 100.52, 0.01},
                                      {{{"WMUL", "1"}}, 101, 1},
                                      {{{"TAIL_TRIGS", "1"}, {"WMUL", "1"}}, 103.02, 1}};
  ProbeList const list =
    probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Hybrid, .comboTop = 2, .comboTiers = 2}, readings);
  CHECK(textsOf(list, "Tail+Height") == std::vector<std::string>{"TAIL_TRIGS=1,ZEROHACK_H=0"});
}

TEST(an_answer_the_readings_cannot_tell_from_the_last_one_carried_is_carried_too) {
  // Tail's two answers past the background's are TAIL_TRIGS=1 and TAIL_KERNELS=1.  TAIL_KERNELS=3 reads exactly what
  // TAIL_KERNELS=1 does, so which of the two comes first says nothing, and both are combined; TAIL_KERNELS=0, dearer
  // and read as exactly, is not.
  FFTConfig const fft{"512:15:512:212"};
  Strategy const strategy{.kind = Strategy::Kind::Hybrid, .comboTiers = 2};
  std::vector<Reading> readings{{{}, 100},
                                {{{"ZEROHACK_H", "0"}}, 100.1},
                                {{{"TAIL_TRIGS", "1"}}, 100.4},
                                {{{"TAIL_KERNELS", "1"}}, 100.8},
                                {{{"TAIL_KERNELS", "3"}}, 100.8},
                                {{{"TAIL_KERNELS", "0"}}, 101.6}};
  CHECK(textsOf(probesOf(nvidia(), fft, {}, strategy, readings), "Tail+Height") ==
        (std::vector<std::string>{"TAIL_TRIGS=1,ZEROHACK_H=0", "TAIL_KERNELS=1,ZEROHACK_H=0",
                                  "TAIL_KERNELS=3,ZEROHACK_H=0"}));

  // Read less exactly, TAIL_KERNELS=0 is within the noise of TAIL_KERNELS=1 -- its mean, 100.8, is 0.2 behind that
  // one's 100.6, well within twice their combined standard error -- while TAIL_TRIGS=0, as exact as TAIL_KERNELS=1 and
  // 2.2 behind it, is not.
  readings[3].error = 0.1;
  readings[4].error = 0.1;
  readings[5].error = 0.4;
  readings.push_back({{{"TAIL_TRIGS", "0"}}, 103, 0.1});
  CHECK(textsOf(probesOf(nvidia(), fft, {}, strategy, readings), "Tail+Height") ==
        (std::vector<std::string>{"TAIL_TRIGS=1,ZEROHACK_H=0", "TAIL_KERNELS=1,ZEROHACK_H=0",
                                  "TAIL_KERNELS=3,ZEROHACK_H=0", "TAIL_KERNELS=0,ZEROHACK_H=0"}));
}

TEST(an_answer_the_background_cannot_hold_is_passed_over) {
  // FFT3261 at width 1K offers L2_STRIPING up to 16 alone but only up to 8 beside MULTI_Q=1.  From MULTI_Q=1, the
  // best answer of Placement is L2_STRIPING=16, read against MULTI_Q=0, which that background cannot hold: its answers
  // are the next two instead, while MULTI_Q=0 is still Queues' answer.
  FFTConfig const fft{"2:1K:8:256:212"};
  std::vector<Reading> const readings{{{{"MULTI_Q", "1"}}, 100},
                                      {{}, 100.1},
                                      {{{"L2_STRIPING", "16"}}, 100.15},
                                      {{{"MULTI_Q", "1"}, {"LOADS", "2"}}, 100.2},
                                      {{{"MULTI_Q", "1"}, {"L2_STRIPING", "4"}}, 100.3},
                                      {{{"MULTI_Q", "1"}, {"L2_STRIPING", "2"}}, 100.5}};
  ProbeList const list = probesOf(nvidia(), fft, readings.front().config, {}, readings);
  CHECK(textsOf(list, "all") ==
        (std::vector<std::string>{"LOADS=2,MULTI_Q=0", "L2_STRIPING=4,MULTI_Q=0", "L2_STRIPING=4,LOADS=2",
                                  "L2_STRIPING=4,LOADS=2,MULTI_Q=0", "L2_STRIPING=2,MULTI_Q=0", "L2_STRIPING=2,LOADS=2",
                                  "L2_STRIPING=2,LOADS=2,MULTI_Q=0"}));
}

TEST(a_stage_is_cut_at_its_limit_in_falling_order_of_gain) {
  // Every one-step move of the entry read once, each a little dearer than the last: seven dimensions of three answers
  // at the top tier would be 3^7 - 1 points.
  FFTConfig const fft{"512:15:512:212"};
  ProbeList const steps = probesOf(nvidia(), fft, {}, {.kind = Strategy::Kind::Single});
  std::vector<Reading> readings{{{}, 100}};
  std::map<std::string, double> gainOf;
  for (size_t i = 0; i < steps.probes.size(); ++i) {
    const Probe& p = steps.probes[i];
    if (p.moves.size() != 1 || p.text.find(',') != std::string::npos) { continue; }
    double const cost = 100 + 0.01 * double(i + 1);
    readings.push_back({p.config, cost});
    gainOf[p.text] = 1 - cost / 100;
  }

  ProbeList const list = probesOf(nvidia(), fft, {}, {}, readings);
  std::vector<std::string> const all = textsOf(list, "all");
  CHECK_EQ(all.size(), size_t(MAX_POINTS));

  // A point's gain is the sum of its seeds', each a one-step reading, so it is the sum over the keys it sets.
  double last = 0;
  for (const std::string& text : all) {
    double gain = 0;
    for (size_t at = 0; at <= text.size();) {
      size_t const comma = std::min(text.find(',', at), text.size());
      std::string const part = text.substr(at, comma - at);
      CHECK(gainOf.contains(part));
      gain += gainOf[part];
      at = comma + 1;
    }
    CHECK(gain <= last + 1e-12);
    last = gain;
  }
  // The cut is the strategy's: a lower one keeps the head of the same order, and none keeps every point.
  std::vector<std::string> const ten = textsOf(probesOf(nvidia(), fft, {}, {.maxPoints = 10}, readings), "all");
  CHECK(ten == std::vector<std::string>(all.begin(), all.begin() + 10));
  std::vector<std::string> const every = textsOf(probesOf(nvidia(), fft, {}, {.maxPoints = NO_LIMIT}, readings), "all");
  CHECK(every.size() > all.size());
  CHECK(std::equal(all.begin(), all.end(), every.begin()));

  // Listed a few at a time, the same order, and what is left of the stage said.
  ProbeList const few = probesOf(nvidia(), fft, {}, {.maxPoints = NO_LIMIT}, readings, true, 10);
  CHECK(textsOf(few, "all") == std::vector<std::string>(every.begin(), every.begin() + 10));
  auto const rest = std::ranges::find(few.unlisted, std::string{"all"}, &ProbeList::Unlisted::stage);
  CHECK(rest != few.unlisted.end());
  if (rest != few.unlisted.end()) {
    CHECK_EQ(rest->tier, 3u);
    CHECK(rest->most >= every.size() - 10);
  }
}

TEST(the_bins_of_one_group_are_combined_at_the_second_tier) {
  // One answer read in each of Memory's two bins: no step of either bin moves both, and until bins were dimensions no
  // combination did either.
  FFTConfig const fft{"512:15:512:212"};
  ProbeList const steps = probesOf(nvidia(), fft, {}, {});
  auto const first = [&](const std::string& stage) {
    auto const at = std::ranges::find(steps.probes, stage, &Probe::stage);
    CHECK(at != steps.probes.end());
    return at == steps.probes.end() ? UseConfig{} : at->config;
  };
  UseConfig const one = first("Memory 1");
  UseConfig const two = first("Memory 2");
  std::vector<Reading> const readings{{{}, 100}, {one, 100.1}, {two, 100.2}};

  // Every axis where one of the two answers has it: both LOADS classes may be digits of the one key.
  auto isBoth = [&](const ProbeList& list, const UseConfig& config) {
    return std::ranges::all_of(list.axes, [&](const Axis& axis) {
      auto const background = positionOf(nvidia(), fft, {}, axis);
      auto const inOne = positionOf(nvidia(), fft, one, axis);
      auto const want = inOne != background ? inOne : positionOf(nvidia(), fft, two, axis);
      return positionOf(nvidia(), fft, config, axis) == want;
    });
  };
  auto combines = [&](const Strategy& strategy) {
    ProbeList const list = probesOf(nvidia(), fft, {}, strategy, readings);
    return std::ranges::any_of(list.probes, [&](const Probe& p) {
      return p.stage == "Memory combined" && p.tier == 2 && isBoth(list, p.config);
    });
  };
  CHECK(combines({}));

  // Likewise two register caps under CUDA, each a bin of its own.
  Env const cuda{.isNvidia = true, .cudaBackend = true, .computeCapability = 600, .pdlLaunch = true};
  FFTConfig const fp64{"1K:7:256:212"};
  UseConfig const caps{{"REGMI64", "72"}, {"REGMO64", "72"}};
  std::vector<Reading> const capReadings{{{}, 100}, {{{"REGMI64", "72"}}, 100.1}, {{{"REGMO64", "72"}}, 100.2}};
  ProbeList const capList = probesOf(cuda, fp64, {}, {}, capReadings);
  CHECK(std::ranges::any_of(capList.probes, [&](const Probe& p) {
    return p.stage == "Cuda combined" && p.config == canonicalConfig(cuda, fp64, caps);
  }));
}

TEST(branches_are_the_structural_values_the_readings_hold_cheapest_first) {
  FFTConfig const fft{"512:15:512:212"};
  std::vector<Reading> const readings{{{{"WMUL", "1"}}, 99},
                                      {{}, 100},
                                      {{{"INPLACE", "0"}, {"PAD", "64"}}, 100.3},
                                      {{{"INPLACE", "0"}}, 100.5},
                                      {{{"SHUFL_BYTES_W", "16"}}, 101},
                                      {{{"SHUFL_BYTES_W", "4"}}, 102},
                                      {{{"LDSPAD_W", "0"}}, 103},
                                      {{{"SHUFL_BYTES_H", "16"}}, 104},
                                      {{{"SHUFL_BYTES_H", "4"}}, 105},
                                      {{{"LDSPAD_H", "0"}}, 106},
                                      {{{"INPLACE", "0"}, {"SHUFL_BYTES_W", "16"}}, 107},
                                      {{{"LDSPAD_H", "0"}, {"LDSPAD_W", "0"}}, 108}};
  std::vector<Branch> const branches = branchesOf(nvidia(), fft, readings);
  CHECK_EQ(branches.size(), size_t(MAX_BRANCHES));
  if (branches.size() != MAX_BRANCHES) { return; }

  // The entry's best set heads the first, and each branch's best is its cheapest reading.
  CHECK(branches[0].best == (UseConfig{{"WMUL", "1"}}));
  CHECK(branches[1].best == (UseConfig{{"INPLACE", "0"}, {"PAD", "64"}}));
  CHECK_EQ(branches[1].cost, 100.3);
  CHECK_EQ(configText(branches[1].structure),
           std::string{"INPLACE=0,LDSPAD_H=1,LDSPAD_W=1,SHUFL_BYTES_H=8,"
                       "SHUFL_BYTES_W=8"});
  CHECK(branches.back().best == (UseConfig{{"LDSPAD_H", "0"}}));

  // A non-structural key never makes a branch, and a structural value the kernels do not see is not one either.
  CHECK(branchOf(nvidia(), fft, {{"WMUL", "1"}}) == branchOf(nvidia(), fft, {}));
  CHECK(!branchOf(nvidia(), fft, {}).contains("NOREG"));
}

TEST(only_the_best_branch_steps_into_others) {
  FFTConfig const fft{"512:15:512:212"};
  ProbeList const best = probesOf(nvidia(), fft, {}, {});
  ProbeList const other = probesOf(nvidia(), fft, {{"SHUFL_BYTES_W", "16"}}, {}, {}, false);

  auto structural = [](const ProbeList& list) {
    return std::ranges::count_if(list.probes, [&](const Probe& p) {
      return std::ranges::any_of(p.moves, [&](const auto& m) { return list.axes[m.first].option->structural; });
    });
  };
  CHECK_EQ(structural(best), 7);
  CHECK_EQ(structural(other), 0);

  // Within its branch it offers everything else, WMUL among it: at SHUFL_BYTES_W=16 WMUL=4 is still on offer.
  CHECK(std::ranges::count(textsOf(other, "Width"), std::string{"WMUL=1"}) == 1);
}

TEST(combo_tiers_1_is_groups_exactly) {
  // Over the self-check matrix, every ninth point, with and without readings to combine: the same probes, in the same
  // order, from the same stages.
  std::vector<MatrixPoint> const matrix = selfCheckMatrix();
  size_t compared = 0;
  for (size_t i = 0; i < matrix.size(); i += 9) {
    const MatrixPoint& p = matrix[i];
    ProbeList const groups = probesOf(p.env, p.fft, p.decided, {.kind = Strategy::Kind::Groups});
    std::vector<Reading> readings{{canonicalConfig(p.env, p.fft, p.decided), 100}};
    for (size_t j = 0; j < groups.probes.size(); j += 5) {
      readings.push_back({groups.probes[j].config, 100 + 0.01 * double(j + 1)});
    }

    for (bool const withReadings : {false, true}) {
      std::span<const Reading> const given =
        withReadings ? std::span<const Reading>{readings} : std::span<const Reading>{};
      ProbeList const one = probesOf(p.env, p.fft, p.decided, {.kind = Strategy::Kind::Hybrid, .comboTiers = 1}, given);
      ProbeList const grouped = probesOf(p.env, p.fft, p.decided, {.kind = Strategy::Kind::Groups}, given);
      bool same = one.probes.size() == grouped.probes.size();
      for (size_t k = 0; same && k < one.probes.size(); ++k) {
        const Probe& a = one.probes[k];
        const Probe& b = grouped.probes[k];
        same = a.config == b.config && a.stage == b.stage && a.moves == b.moves && a.text == b.text && a.key == b.key &&
          a.dependees == b.dependees && a.tier == b.tier;
      }
      if (!same) { printf("  differs at %s\n", p.label.c_str()); }
      CHECK(same);
      ++compared;
    }
  }
  CHECK(compared > 700);
}
