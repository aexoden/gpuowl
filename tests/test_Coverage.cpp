// Copyright (C) Jason Lynch

#include "test.h"

#include "Coverage.h"
#include "FFTVariants.h"

#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

const RegisterOffer NO_REGISTERS{};
const AxisFilter FILTER{.tailQueues = TailQueues::All, .registers = &NO_REGISTERS};

Reading reading(const char* spec, UseConfig options, double mean, double se = 1, u32 calls = 2) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft,
          .kind = TestKind::LL,
          .regime = regimeOf(fft, 69'000'000),
          .options = canonicalConfig(nvidia(), fft, options),
          .exponent = 69'000'000,
          .calls = calls,
          .mean = mean,
          .se = se,
          .failed = false};
}

std::vector<const Reading*> pointers(const std::vector<Reading>& rs) {
  std::vector<const Reading*> out;
  for (const Reading& r : rs) { out.push_back(&r); }
  return out;
}

// Every key `key` depends on, directly or through another.
std::set<std::string> ancestry(const std::string& key) {
  std::set<std::string> out;
  std::vector<std::string> todo{key};
  while (!todo.empty()) {
    std::string const at = todo.back();
    todo.pop_back();
    for (const std::string& parent : findOption(at)->dependsOn) {
      if (out.insert(parent).second) { todo.push_back(parent); }
    }
  }
  return out;
}

}  // namespace

TEST(the_archive_keeps_every_branch_however_slow) {
  std::vector<Reading> const rs{
    reading("512:7:512:012", {}, 500),
    reading("512:7:512:012", {{"INPLACE", "0"}}, 900),
    reading("512:7:512:012", {{"L2PERSIST", "1"}}, 700),
    reading("512:7:512:012", {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}, 650),
  };
  std::vector<BranchArchive> const a = archiveOf(nvidia(), pointers(rs));
  CHECK_EQ(a.size(), size_t{3});
  CHECK_EQ(a[0].seeds.front()->mean, 500.0);
  CHECK_EQ(a[1].seeds.front()->mean, 650.0);
  CHECK_EQ(a[1].configurations, size_t{2});
  CHECK_EQ(a[2].seeds.front()->mean, 900.0);
}

TEST(a_branch_seeds_the_fastest_one_within_error_and_the_most_unlike) {
  std::vector<Reading> const rs{
    reading("1K:13:256:202", {}, 500, 2),
    reading("1K:13:256:202", {{"ZEROHACK_W", "0"}}, 503, 2),  // within error
    reading("1K:13:256:202", {{"SHUFL_BYTES_W", "16"}, {"WMUL", "1"}, {"HOIST_W", "1"}}, 560),
    reading("1K:13:256:202", {{"HOIST_W", "1"}, {"LOADS", "3"}}, 540),
    reading("1K:13:256:202", {{"LOADS", "3"}}, 520),
    reading("1K:13:256:202", {{"ZEROHACK_W", "0"}, {"LOADS", "3"}}, 510, 0.5, 1),
  };
  std::vector<BranchArchive> const a = archiveOf(nvidia(), pointers(rs));
  // SHUFL_BYTES_W is structural: its set is a branch of its own.
  CHECK_EQ(a.size(), size_t{2});
  const BranchArchive& main = a[0];
  CHECK_EQ(main.configurations, size_t{5});
  CHECK_EQ(main.seeds.size(), size_t{4});
  CHECK_EQ(main.seeds[0]->mean, 500.0);
  CHECK_EQ(main.seeds[1]->mean, 503.0);
  // Then the set furthest from those two, two axes from either; then of the rest, each one axis from some seed, the
  // fastest.
  CHECK_EQ(main.seeds[2]->mean, 540.0);
  CHECK_EQ(main.seeds[3]->mean, 510.0);
}

TEST(axes_apart_counts_access_classes_one_by_one) {
  CHECK_EQ(axesApart({}, {}), size_t{0});
  CHECK_EQ(axesApart({{"WMUL", "2"}}, {}), size_t{1});
  CHECK_EQ(axesApart({{"WMUL", "2"}}, {{"WMUL", "4"}}), size_t{1});
  CHECK_EQ(axesApart({{"LOADS", "30045"}}, {}), size_t{3});
  CHECK_EQ(axesApart({{"LOADS", "30045"}}, {{"LOADS", "30005"}, {"HOIST_W", "1"}}), size_t{2});
}

TEST(position_coverage_counts_what_this_entry_read) {
  const char* const spec = "512:7:512:012";
  std::shared_ptr<const ConditionalSpace> const space = spaceOf(nvidia(), canonical(FFTConfig{spec}), FILTER);
  size_t active = 0;
  std::vector<size_t> const defaults = space->statesOf({});
  for (size_t v = 0; v < space->variables().size(); ++v) { active += defaults[v] != space->variables()[v].inactive(); }

  PositionCoverage const none = positionCoverage(*space, {});
  CHECK_EQ(none.covered, size_t{0});
  CHECK(none.total > active);

  std::vector<Reading> rs{reading(spec, {}, 500)};
  CHECK_EQ(positionCoverage(*space, pointers(rs)).covered, active);
  // L2PERSIST=1 and, in its context, L2PERSISTPCT=25; every other axis where the first reading had it.
  rs.push_back(reading(spec, {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}, 480));
  CHECK_EQ(positionCoverage(*space, pointers(rs)).covered, active + 2);
  // A reading of another FFT is none of this one's.
  rs.push_back(reading("512:8:512:000", {{"INPLACE", "0"}}, 480));
  CHECK_EQ(positionCoverage(*space, pointers(rs)).covered, active + 2);
  // A failed one covers nothing.
  rs.push_back(reading(spec, {{"WMUL", "4"}}, 480));
  rs.back().failed = true;
  CHECK_EQ(positionCoverage(*space, pointers(rs)).covered, active + 2);
}

TEST(legal_branches_are_those_a_walk_through_the_options_finds) {
  for (const char* spec : {"512:7:512:012", "1K:13:256:202"}) {
    FFTConfig const fft = canonical(FFTConfig{spec});
    std::shared_ptr<const ConditionalSpace> const space = spaceOf(nvidia(), fft, FILTER);

    // Independently of the space: every structural key, and every key one depends on, moved through every position
    // axesOf() offers it, one after another; the distinct branchOf() of what that reaches.
    std::set<std::string> keys;
    for (const Option& o : allOptions()) {
      if (o.kind != Kind::Tunable || !o.structural) { continue; }
      keys.insert(o.key);
      for (const std::string& k : ancestry(o.key)) { keys.insert(k); }
    }
    std::set<UseConfig> branches;
    std::set<std::pair<UseConfig, std::set<std::string>>> seen;
    auto walk = [&](auto&& self, const UseConfig& config, const std::set<std::string>& done) -> void {
      if (!seen.insert({config, done}).second) { return; }
      for (const Axis& a : axesOf(nvidia(), fft, config, FILTER)) {
        if (done.contains(a.name) || !keys.contains(a.option->key)) { continue; }
        std::set<std::string> next = done;
        next.insert(a.name);
        for (size_t p = 0; p < a.values.size(); ++p) {
          UseConfig moved = config;
          place(moved, a, p);
          self(self, fitted(nvidia(), fft, moved), next);
        }
        return;
      }
      branches.insert(branchOf(nvidia(), fft, config));
    };
    walk(walk, {}, {});

    BranchCount const count = space->legalBranches();
    CHECK(count.complete);
    if (count.legal != branches.size()) {
      testing::fail(__FILE__, __LINE__,
                    std::string{spec} + ": " + std::to_string(count.legal) + " legal branches, walk found " +
                      std::to_string(branches.size()));
    }
  }
}

TEST(every_context_a_configuration_has_is_legal) {
  FFTConfig const fft = canonical(FFTConfig{"512:7:512:012"});
  std::shared_ptr<const ConditionalSpace> const space = spaceOf(nvidia(), fft, FILTER);
  const std::vector<BranchVariable>& vars = space->variables();
  for (const UseConfig& config : std::vector<UseConfig>{
         {},
         {{"INPLACE", "0"}},
         {{"SHUFL_BYTES_W", "16"}},
         {{"TAIL_KERNELS", "1"}, {"SHUFL_BYTES_H", "16"}},
         {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "50"}},
       }) {
    std::vector<size_t> const states = space->statesOf(config);
    for (size_t v = 0; v < vars.size(); ++v) { CHECK(space->legalContext(v, space->contextOf(v, states))); }
  }
  // Some contexts cannot happen: an ancestor at a state its own ancestors rule out.
  size_t illegal = 0;
  for (size_t v = 0; v < vars.size(); ++v) {
    for (size_t c = 0; c < vars[v].contexts(); ++c) { illegal += !space->legalContext(v, c); }
  }
  CHECK(illegal > 0);
}

TEST(an_option_with_too_many_contexts_is_unsearched_only_where_it_can_apply) {
  // SMEM_SPILL depends on every register limit: seven of them on a two-part FFT, past MAX_CONTEXTS with four limits
  // offered each.  It needs a CUDA 13 driver.
  RegisterOffer offer;
  for (const Option& o : allOptions()) {
    if (o.key.starts_with("REG")) { offer[o.key] = {40, 48, 56, 64}; }
  }
  AxisFilter const filter{.tailQueues = TailQueues::All, .registers = &offer};
  FFTConfig const fft = canonical(FFTConfig{"1:256:7:512:202"});
  for (const char* driver : {"CUDA 12.4", "CUDA 13.0"}) {
    Env env = nvidia();
    env.computeCapability = 806;
    env.driverVersion = driver;
    ConditionalSpace const space{env, fft, filter};
    bool const cuda13 = std::string{driver} == "CUDA 13.0";
    CHECK_EQ(space.unsearched() == std::vector<std::string>{"SMEM_SPILL"}, cuda13);
    CHECK_EQ(space.unsearched().empty(), !cuda13);
  }
}
