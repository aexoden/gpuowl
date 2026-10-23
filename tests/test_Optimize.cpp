// Copyright (C) Jason Lynch

#include "test.h"

#include "Branches.h"
#include "FFTVariants.h"
#include "MinSum.h"
#include "Optimize.h"
#include "TuneDB.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

Subject subject(const char* spec) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft, .kind = TestKind::PRP, .regime = regimeOf(fft, 67'000'000)};
}

}  // namespace

TEST(optimize_finds_a_pair_that_loses_alone) {
  Subject const s = subject("1K:13:256:202");
  auto obs = [&](UseConfig o, double cost) {
    return Observation{.fft = s.fft, .kind = s.kind, .regime = s.regime, .options = std::move(o), .cost = cost};
  };

  std::vector<Observation> data;
  for (int i = 0; i < 8; ++i) {
    data.push_back(obs({}, 1000));
    data.push_back(obs({{"ZEROHACK_W", "0"}}, 1020));
    data.push_back(obs({{"SHUFL_BYTES_W", "16"}}, 1020));
    data.push_back(obs({{"SHUFL_BYTES_W", "16"}, {"ZEROHACK_W", "0"}}, 950));
  }
  Model m{nvidia()};
  m.fit(data);

  RegisterOffer const none{};
  Optimum const o = optimize(m, m.mean(), s, {.tailQueues = TailQueues::All, .registers = &none});
  CHECK(o.exact);
  // Under the mean every untried position is a tie with the default, which wins it.
  CHECK(o.options == (UseConfig{{"SHUFL_BYTES_W", "16"}, {"ZEROHACK_W", "0"}}));
}

TEST(optimize_returns_a_configuration_the_table_offers) {
  Subject const s = subject("2:512:4:512:212");
  Model m{nvidia()};
  m.fit({});
  for (u64 seed = 1; seed <= 5; ++seed) {
    Optimum const o = optimize(m, m.draw(seed), s, {.tailQueues = TailQueues::Ntt});
    CHECK(o.options == fitted(nvidia(), s.fft, o.options));
  }
}

namespace {

Subject ll(const char* spec) {
  FFTConfig const fft = canonical(FFTConfig{spec});
  return {.fft = fft, .kind = TestKind::LL, .regime = regimeOf(fft, 69'000'000)};
}

Observation observe(const Subject& s, UseConfig options, double cost) {
  return Observation{.fft = s.fft, .kind = s.kind, .regime = s.regime, .options = std::move(options), .cost = cost};
}

// No register limit offered: REG* stay at the compiler's default, which keeps the spaces below small enough to list.
const RegisterOffer NO_REGISTERS{};
const AxisFilter FILTER{.tailQueues = TailQueues::All, .registers = &NO_REGISTERS};

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

// By listing: every configuration reached from `around` by moving the axes `free` names, and any axis whose option
// depends on one of theirs, each to every position it has there; with `moved`, the names of every axis that moved.
std::set<UseConfig> listed(const Env& env, const FFTConfig& fft, const UseConfig& around,
                           const std::vector<std::string>& free, std::set<std::string>* moved = nullptr) {
  std::set<std::string> freeKeys;
  for (const Axis& a : axesOf(env, fft, around, FILTER)) {
    if (std::ranges::find(free, a.name) != free.end()) { freeKeys.insert(a.option->key); }
  }
  auto movable = [&](const Axis& a) {
    if (std::ranges::find(free, a.name) != free.end()) { return true; }
    return std::ranges::any_of(ancestry(a.option->key), [&](const std::string& k) { return freeKeys.contains(k); });
  };

  std::set<UseConfig> out;
  std::set<std::pair<UseConfig, std::set<std::string>>> seen;
  auto walk = [&](auto&& self, const UseConfig& config, const std::set<std::string>& done) -> void {
    if (!seen.insert({config, done}).second) { return; }
    for (const Axis& a : axesOf(env, fft, config, FILTER)) {
      if (done.contains(a.name) || !movable(a)) { continue; }
      std::set<std::string> next = done;
      next.insert(a.name);
      if (moved) { moved->insert(a.name); }
      for (size_t p = 0; p < a.values.size(); ++p) {
        UseConfig moved = config;
        place(moved, a, p);
        self(self, fitted(env, fft, moved), next);
      }
      return;
    }
    out.insert(canonicalConfig(env, fft, config));
  };
  walk(walk, fitted(env, fft, around), {});
  return out;
}

struct Case {
  const char* fft;
  UseConfig around;
  std::vector<std::string> free;
};

const std::vector<Case>& cases() {
  static const std::vector<Case> all{
    {"512:7:512:012", {}, {"L2PERSIST"}},
    {"512:7:512:012", {}, {"INPLACE"}},
    {"512:7:512:012", {{"INPLACE", "0"}}, {"SHUFL_BYTES_W"}},
    {"512:7:512:012", {}, {"TAIL_KERNELS"}},
    {"512:7:512:012", {}, {"L2PERSIST", "TAIL_KERNELS"}},
    {"1K:13:256:202", {}, {"SHUFL_BYTES_W", "ZEROHACK_W"}},
    {"4:256:11:512", {}, {"256RADIX8_W", "LDSPAD_H"}},
  };
  return all;
}

}  // namespace

TEST(optimize_searches_a_dependent_only_a_moved_parent_opens) {
  // L2PERSISTPCT applies only under L2PERSIST=1, whose default percentage loses: from L2PERSIST=0 the search never saw
  // the percentage that wins.
  Subject const s = ll("512:7:512:012");
  std::vector<Observation> data;
  for (int i = 0; i < 8; ++i) {
    data.push_back(observe(s, {}, 1000));
    data.push_back(observe(s, {{"L2PERSIST", "1"}}, 1050));
    data.push_back(observe(s, {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}, 900));
  }
  Model m{nvidia()};
  m.fit(data);
  UseConfig const best{{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}};

  Optimum const o = optimize(m, m.mean(), s, FILTER);
  CHECK(o.exact);
  CHECK(o.options == best);

  Subspace const one{.around = {}, .free = {"L2PERSIST"}};
  CHECK(optimize(m, m.mean(), s, FILTER, &one).options == best);
}

TEST(optimize_takes_a_dependents_effect_learnt_on_another_fft) {
  Subject const read = ll("512:7:512:012");
  Subject const target = ll("512:8:512:000");
  std::vector<Observation> data;
  for (int i = 0; i < 8; ++i) {
    data.push_back(observe(read, {}, 1000));
    data.push_back(observe(read, {{"L2PERSIST", "1"}}, 1050));
    data.push_back(observe(read, {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}, 900));
    data.push_back(observe(target, {}, 1100));
  }
  Model m{nvidia()};
  m.fit(data);
  CHECK(optimize(m, m.mean(), target, FILTER).options == (UseConfig{{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}));
}

TEST(the_space_reads_back_every_legal_configuration) {
  for (const Case& c : cases()) {
    Subject const s = ll(c.fft);
    ConditionalSpace const space{nvidia(), s.fft, FILTER};
    std::set<UseConfig> const all = listed(nvidia(), s.fft, c.around, c.free);
    CHECK(all.size() > 1);
    for (const UseConfig& config : all) {
      std::vector<size_t> const states = space.statesOf(config);
      CHECK(space.legal(states));
      CHECK(space.configOf(states) == config);
    }
  }
}

TEST(conditional_energy_is_predict_on_every_legal_configuration) {
  Model m{nvidia()};
  m.fit({});
  for (const Case& c : cases()) {
    Subject const s = ll(c.fft);
    for (u64 seed = 1; seed <= 3; ++seed) {
      Draw const d = m.draw(seed);
      for (const UseConfig& config : listed(nvidia(), s.fft, c.around, c.free)) {
        double const energy = conditionalEnergy(m, d, s, FILTER, config);
        double const predicted = m.predict(d, s, config, false);
        if (std::abs(energy - predicted) > 1e-6) {
          testing::fail(__FILE__, __LINE__,
                        std::string{c.fft} + " " + configText(config) + ": energy " + std::to_string(energy) +
                          ", predicted " + std::to_string(predicted));
        }
      }
    }
  }
}

TEST(optimize_is_exact_against_listing_every_branch) {
  Model m{nvidia()};
  m.fit({});
  for (const Case& c : cases()) {
    Subject const s = ll(c.fft);
    // Naming every axis that moves, so that the draw values all of them (Subspace).
    std::set<std::string> moved;
    std::set<UseConfig> const all = listed(nvidia(), s.fft, c.around, c.free, &moved);
    Subspace const sub{.around = c.around, .free = {moved.begin(), moved.end()}};
    for (u64 seed = 1; seed <= 10; ++seed) {
      Draw const d = m.draw(seed);
      double best = FORBIDDEN;
      for (const UseConfig& config : all) { best = std::min(best, m.predict(d, s, config, false)); }
      Optimum const o = optimize(m, d, s, FILTER, &sub);
      CHECK(o.exact);
      CHECK(all.contains(o.options));
      CHECK(std::abs(m.predict(d, s, o.options, false) - best) < 1e-6);
    }
  }
}

TEST(a_held_axis_keeps_a_position_the_space_has_no_state_for) {
  Subject const s = ll("512:7:512:012");
  RegisterOffer const offer{{"REGCF64", {128}}};
  AxisFilter const filter{.tailQueues = TailQueues::All, .registers = &offer};
  Model m{nvidia()};
  m.fit({});
  Subspace const sub{.around = {{"REGCF64", "77"}}, .free = {"L2PERSIST"}};
  for (u64 seed = 1; seed <= 5; ++seed) {
    CHECK(useValue(optimize(m, m.draw(seed), s, filter, &sub).options, "REGCF64", 0) == 77);
  }
}

TEST(what_a_free_parent_opens_moves_only_where_readings_say) {
  // Nothing measured: under the mean every dependent ties with its default, whatever the draw says of it.
  Subject const s = ll("512:7:512:012");
  Model m{nvidia()};
  m.fit({});
  for (u64 seed = 1; seed <= 10; ++seed) {
    for (const char* parent : {"INPLACE", "SHUFL_BYTES_W", "TAIL_KERNELS"}) {
      Subspace const sub{.around = {}, .free = {parent}};
      UseConfig const o = optimize(m, m.draw(seed), s, FILTER, &sub).options;
      CHECK(o.size() <= 1 && (o.empty() || o.begin()->first == parent));
    }
  }

  // Read L2PERSISTPCT=25 winning under L2PERSIST=1, and the draw frees L2PERSIST: the dependent follows the readings.
  std::vector<Observation> data;
  for (int i = 0; i < 8; ++i) {
    data.push_back(observe(s, {}, 1000));
    data.push_back(observe(s, {{"L2PERSIST", "1"}}, 1050));
    data.push_back(observe(s, {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}, 900));
  }
  m.fit(data);
  Subspace const sub{.around = {}, .free = {"L2PERSIST"}};
  for (u64 seed = 1; seed <= 10; ++seed) {
    CHECK(optimize(m, m.draw(seed), s, FILTER, &sub).options ==
          (UseConfig{{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}}));
  }
}
