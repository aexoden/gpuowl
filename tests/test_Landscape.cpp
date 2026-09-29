// Copyright (C) Jason Lynch

// How the tuner spends its time, judged against landscapes whose best configurations are planted: each scenario is one
// way a search can be starved or misled, and passes once what the run publishes is the planted best, within the time a
// run is given.

#include "Landscape.h"

#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace landscape;
using namespace tune;

namespace {

constexpr u64 PROBE = 118'063'003;

// With the probe at `probe`, or where nothing is named, at the workload's own centre.
RunScope around(u64 lo, u64 hi, u64 probe = PROBE) {
  return makeScope(ScopeArgs{.lo = lo, .hi = hi, .probe = probe}, {});
}

bool has(const UseConfig& options, const char* key, const char* value) {
  auto const at = options.find(key);
  return at != options.end() && at->second == value;
}

// Every key a set moves from where the landscape wants it costs 1%, but for those the scenario plants a gain on.
double others(const UseConfig& options, std::initializer_list<const char*> planted) {
  double out = 1;
  for (const auto& [key, value] : options) {
    if (std::ranges::none_of(planted, [&](const char* k) { return key == k; })) { out *= 1.01; }
  }
  return out;
}

// Three FFT64 shapes of one size, each serving the whole workload in the same regime; `a` is the cheapest at the
// built-in defaults, and nothing tunes it.
constexpr const char* A = "1K:15:256:202";
constexpr const char* B = "512:15:512:202";
constexpr const char* C = "256:15:1K:202";

Scenario scenario(RunScope scope, std::vector<std::string> ffts, double budget) {
  Scenario out;
  out.scope = std::move(scope);
  out.ffts = std::move(ffts);
  out.budget = budget;
  return out;
}

double flat(const FFTConfig& fft, const UseConfig& options, double base) {
  (void)fft;
  return base * others(options, {});
}

// B trails A by 3% and needs two steps, each 2.5%, to pass it: MM_CHAIN=1 early in its list, then TAIL_KERNELS=3 late.
Scenario twoSteps() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 4 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == A) { return flat(fft, o, 1000); }
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    double f = 1030 * others(o, {"MM_CHAIN", "TAIL_KERNELS"});
    if (has(o, "MM_CHAIN", "1")) {
      f *= 0.975;
    } else if (o.contains("MM_CHAIN")) {
      f *= 1.01;
    }
    if (has(o, "TAIL_KERNELS", "3")) {
      f *= 0.975;
    } else if (o.contains("TAIL_KERNELS")) {
      f *= 1.01;
    }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? 1000.0 : fft.spec() == C ? 1060.0 : 1030 * 0.975 * 0.975;
  };
  return s;
}

// B's gain is in a structural branch that loses alone: SHUFL_BYTES_W=16 costs 1%, and only there does WMUL=1 save 6%.
Scenario structuralBranch() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 8 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == A) { return flat(fft, o, 1000); }
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    bool const wide = has(o, "SHUFL_BYTES_W", "16");
    double f = 1030 * others(o, {"SHUFL_BYTES_W", "WMUL"});
    if (wide) {
      f *= 1.01;
    } else if (o.contains("SHUFL_BYTES_W")) {
      f *= 1.01;
    }
    if (has(o, "WMUL", "1")) {
      f *= wide ? 0.94 : 1.01;
    } else if (o.contains("WMUL")) {
      f *= 1.01;
    }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? 1000.0 : fft.spec() == C ? 1060.0 : 1030 * 1.01 * 0.94;
  };
  return s;
}

// B's gain needs two groups at once: TAIL_KERNELS=3 and ZEROHACK_H=0 each cost 1% alone and save 5% together.
Scenario crossBins() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 8 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == A) { return flat(fft, o, 1000); }
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    bool const tail = has(o, "TAIL_KERNELS", "3");
    bool const zero = has(o, "ZEROHACK_H", "0");
    double f = 1030 * others(o, {"TAIL_KERNELS", "ZEROHACK_H"});
    if (o.contains("TAIL_KERNELS")) { f *= 1.01; }
    if (o.contains("ZEROHACK_H")) { f *= 1.01; }
    if (tail && zero) { f *= 0.95 / (1.01 * 1.01); }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? 1000.0 : fft.spec() == C ? 1060.0 : 1030 * 0.95;
  };
  return s;
}

// B's gain is the last step its list offers: ZEROHACK_H=0 saves 5%.
Scenario lateGroup() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 4 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == A) { return flat(fft, o, 1000); }
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    double f = 1030 * others(o, {"ZEROHACK_H"});
    if (has(o, "ZEROHACK_H", "0")) { f *= 0.95; }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? 1000.0 : fft.spec() == C ? 1060.0 : 1030 * 0.95;
  };
  return s;
}

// A, whose prior is as cheap as anything's, fails to build whatever it is given; B has a 3% step.
Scenario failedPrior() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 2 * 3600);
  s.landscape.fails = [](const FFTConfig& fft, const UseConfig&) { return fft.spec() == A; };
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    double f = 1000 * others(o, {"MM_CHAIN"});
    if (has(o, "MM_CHAIN", "1")) {
      f *= 0.97;
    } else if (o.contains("MM_CHAIN")) {
      f *= 1.01;
    }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? std::numeric_limits<double>::infinity() : fft.spec() == C ? 1060.0 : 970.0;
  };
  return s;
}

// Two bands, each with its own FFT: 512:15:512 below 143M and 1K:8:1K above, each with a 3% step of its own.
Scenario disjointBands() {
  Scenario s = scenario(around(120'000'000, 160'000'000, 0), {B, "1K:8:1K:202"}, 4 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    bool const low = fft.spec() == B;
    const char* const key = low ? "MM_CHAIN" : "TAIL_KERNELS";
    const char* const value = low ? "1" : "3";
    double f = (low ? 1000 : 2100) * others(o, {key});
    if (has(o, key, value)) {
      f *= 0.97;
    } else if (o.contains(key)) {
      f *= 1.01;
    }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) { return fft.spec() == B ? 970.0 : 2100 * 0.97; };
  return s;
}

// What B's search finds holds on A too, and more: A finds MM_CHAIN=1 for 2% on its own early on; ZEROHACK_H=0, last in
// every list, saves 3% on either.
Scenario transfer() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B}, 4 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    bool const a = fft.spec() == A;
    double f = (a ? 1000 : 1010) * others(o, {"MM_CHAIN", "ZEROHACK_H"});
    if (has(o, "MM_CHAIN", "1")) {
      f *= a ? 0.98 : 1.01;
    } else if (o.contains("MM_CHAIN")) {
      f *= 1.01;
    }
    if (has(o, "ZEROHACK_H", "0")) { f *= 0.97; }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) { return fft.spec() == A ? 1000 * 0.98 * 0.97 : 1010 * 0.97; };
  return s;
}

// A finds 8% early in its list (MM_CHAIN=1), which puts B, 3% behind it at the defaults, outside the margin; B's own
// 12% is the last step it offers (ZEROHACK_H=0).
Scenario pulledAway() {
  Scenario s = scenario(around(110'000'000, 135'000'000), {A, B, C}, 12 * 3600);
  s.landscape.cost = [](const FFTConfig& fft, TestKind, const UseConfig& o) {
    if (fft.spec() == C) { return flat(fft, o, 1060); }
    if (fft.spec() == A) {
      double f = 1000 * others(o, {"MM_CHAIN"});
      if (has(o, "MM_CHAIN", "1")) {
        f *= 0.92;
      } else if (o.contains("MM_CHAIN")) {
        f *= 1.01;
      }
      return f;
    }
    double f = 1030 * others(o, {"ZEROHACK_H"});
    if (has(o, "ZEROHACK_H", "0")) { f *= 0.88; }
    return f;
  };
  s.landscape.best = [](const FFTConfig& fft, TestKind) {
    return fft.spec() == A ? 920.0 : fft.spec() == C ? 1060.0 : 1030 * 0.88;
  };
  return s;
}

struct Named {
  const char* name;
  Scenario (*make)();
};

const std::vector<Named>& scenarios() {
  static const std::vector<Named> all{{"two steps", twoSteps},       {"structural branch", structuralBranch},
                                      {"cross bins", crossBins},     {"late group", lateGroup},
                                      {"failed prior", failedPrior}, {"disjoint bands", disjointBands},
                                      {"transfer", transfer},        {"pulled away", pulledAway}};
  return all;
}

std::string minutes(std::optional<double> seconds) {
  if (!seconds) { return "never"; }
  char buf[32];
  snprintf(buf, sizeof(buf), "%.0f min", *seconds / 60);
  return buf;
}

}  // namespace

// Every scenario under the default policy run to the end, with the halving turned off, and at a run's default stop
// fraction, and how each went: set PRPLL_LANDSCAPES to run it, to part of a scenario's name for that scenario alone.
// It takes minutes, so the suite leaves it out.
TEST(landscape_report) {
  const char* const only = std::getenv("PRPLL_LANDSCAPES");
  if (!only) { return; }
  for (const Named& n : scenarios()) {
    if (std::string{n.name}.find(only) == std::string::npos) { continue; }
    Scenario const s = n.make();
    for (const auto& [label, policy] :
         {std::pair{"default", Policy{}}, std::pair{"no halving", Policy{.halving = {.contenders = 0}}},
          std::pair{"stop", Policy{.stop = STOP}}}) {
      Outcome const o = simulate(s, policy);
      std::optional<double> const r = o.regret();
      fprintf(stderr, "LANDSCAPE %-18s %-10s settled %-9s regret %s items %u in %.0f min;", n.name, label,
              minutes(o.settled()).c_str(), r ? std::to_string(*r * 100).c_str() : "uncovered", o.items,
              o.seconds / 60);
      for (const auto& [entry, calls] : o.calls) { fprintf(stderr, " %s=%u", entry.c_str(), calls); }
      fprintf(stderr, "\n");
      for (const std::string& r : o.rounds) { fprintf(stderr, "LANDSCAPE   round %s\n", r.c_str()); }
      for (const std::string& b : o.bootstrapped) { fprintf(stderr, "LANDSCAPE   bootstrap %s\n", b.c_str()); }
      for (const auto& [entry, set] : o.published) {
        fprintf(stderr, "LANDSCAPE   published %s %s\n", entry.c_str(), set.c_str());
      }
    }
  }
}

namespace {

// The scenarios at a scale the suite can afford: one key at a time, and rounds of 4 calls.
Policy small() {
  return {.strategy = {.kind = Strategy::Kind::Single}, .halving = {.contenders = CONTENDERS, .roundCalls = 4}};
}

}  // namespace

TEST(landscape_a_gain_last_in_a_trailing_ffts_search_is_found) {
  Scenario s = lateGroup();
  s.budget = 2 * 3600;
  Outcome o = simulate(s, small());
  CHECK(o.settled().has_value());
  CHECK_EQ(o.published[std::string{B} + " prp short32"], std::string{"ZEROHACK_H=0"});
}

TEST(landscape_an_fft_that_builds_nothing_holds_nothing_back) {
  Scenario s = failedPrior();
  s.budget = 3600;
  Outcome o = simulate(s, small());
  CHECK(o.settled().has_value());
  CHECK_EQ(o.calls[std::string{A} + " prp short32"], 1u);
}

TEST(landscape_each_band_has_its_own_fft_tuned) {
  Scenario s = disjointBands();
  s.budget = 2 * 3600;
  Outcome o = simulate(s, small());
  CHECK(o.settled().has_value());
  CHECK_EQ(o.published[std::string{B} + " prp short32"], std::string{"MM_CHAIN=1"});
  CHECK_EQ(o.published["1K:8:1K:202 prp long32"], std::string{"TAIL_KERNELS=3"});
}

TEST(landscape_each_band_has_its_own_fft_tuned_at_the_default_stop) {
  // The halving's rounds are what reach 1K:8:1K's step, since by then no step of it is worth the stop fraction on its
  // own.
  Scenario s = disjointBands();
  s.budget = 2 * 3600;
  Policy p = small();
  p.stop = STOP;
  Outcome o = simulate(s, p);
  CHECK(o.settled().has_value());
  CHECK_EQ(o.published["1K:8:1K:202 prp long32"], std::string{"TAIL_KERNELS=3"});
}
