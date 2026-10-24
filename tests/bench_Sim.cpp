// Copyright (C) Jason Lynch

// prpll-sim: the tuning loop against simulated cards whose optimum is known, each with a trap a search can fall into.
// For each scenario and seed it runs the loop for a span of simulated tuning time, in chunks, and prints how far what
// it publishes is from the optimum after each chunk.  No GPU; hours of CPU for a full run, so not part of `make check`.
//
//   prpll-sim [scenario|all] [seeds=N] [first=S] [minutes=M] [lanes=...]
//
// Output lines, one per chunk: "<scenario> <seed> <minutes> <gap>", the gap being T over the optimum's T, less one;
// then a summary per scenario.

#include "SimCard.h"

#include "Candidates.h"
#include "FFTVariants.h"
#include "log.h"
#include "Readings.h"
#include "Scope.h"
#include "TuneDB.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace tune;
using namespace sim;

namespace {

Env card() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

// A variant's own cost, which the stated prior cannot know: up to 3% either way, fixed per FFT.
double jittered(const FFTConfig& fft) {
  u64 const h = std::hash<std::string>{}(fft.spec()) * 0x9e3779b97f4a7c15ull;
  double const u = double(h >> 11) * 0x1.0p-53 * 2 - 1;
  return priorBase(fft) * (1 + 0.03 * u);
}

auto everywhere = [](const FFTConfig&) { return true; };
auto shapeIs(std::string spec) {
  return [spec](const FFTConfig& fft) { return fft.shape.spec() == spec; };
}
auto shapeIsNot(std::string spec) {
  return [spec](const FFTConfig& fft) { return fft.shape.spec() != spec; };
}

std::vector<Landscape> scenarios() {
  std::vector<Landscape> out;
  out.push_back(
    {.name = "broad",
     .base = jittered,
     .terms = {
       {.name = "ZEROHACK_W=0 saves 4%", .where = everywhere, .needs = {{"ZEROHACK_W", "0"}}, .factor = 0.96}}});
  out.push_back(
    {.name = "one-shape",
     .base = jittered,
     .terms = {
       {.name = "L1CUDA=1 costs 5%", .where = shapeIsNot("1:256:7:512"), .needs = {{"L1CUDA", "1"}}, .factor = 1.05},
       {.name = "L1CUDA=1 saves 10% on one shape",
        .where = shapeIs("1:256:7:512"),
        .needs = {{"L1CUDA", "1"}},
        .factor = 0.90}}});
  out.push_back(
    {.name = "late-branch",
     .base = jittered,
     .terms = {{.name = "L2PERSIST=1 costs 5%", .where = everywhere, .needs = {{"L2PERSIST", "1"}}, .factor = 1.05},
               {.name = "with L2PERSISTPCT=25 saves 10%",
                .where = everywhere,
                .needs = {{"L2PERSIST", "1"}, {"L2PERSISTPCT", "25"}},
                .factor = 0.90 / 1.05}}});
  out.push_back(
    {.name = "pair",
     .base = jittered,
     .terms = {
       {.name = "ENABLE_RESTRICT=1 costs 3%", .where = everywhere, .needs = {{"ENABLE_RESTRICT", "1"}}, .factor = 1.03},
       {.name = "OLD_FENCE=0 costs 3%", .where = everywhere, .needs = {{"OLD_FENCE", "0"}}, .factor = 1.03},
       {.name = "together they save 6%",
        .where = everywhere,
        .needs = {{"ENABLE_RESTRICT", "1"}, {"OLD_FENCE", "0"}},
        .factor = 0.94 / (1.03 * 1.03)}}});
  out.push_back({.name = "slow-fft",
                 .base = jittered,
                 .terms = {{.name = "INPLACE=0 saves 20% on a shape 15% behind",
                            .where = shapeIs("2:512:4:512"),
                            .needs = {{"INPLACE", "0"}},
                            .factor = 0.80}}});
  {
    UseConfig const five{
      {"HOIST_W", "1"}, {"ENABLE_RESTRICT", "1"}, {"L1CUDA", "1"}, {"OLD_FENCE", "0"}, {"MULTI_Q", "1"}};
    Landscape narrow{.name = "narrow", .base = jittered, .terms = {}};
    for (const auto& [key, value] : five) {
      narrow.terms.push_back({.name = key + "=" + value + " costs 1%",
                              .where = shapeIs("1K:7:256"),
                              .needs = {{key, value}},
                              .factor = 1.01});
    }
    narrow.terms.push_back(
      {.name = "all five save 10%", .where = shapeIs("1K:7:256"), .needs = five, .factor = 0.90 / std::pow(1.01, 5)});
    out.push_back(std::move(narrow));
  }
  return out;
}

struct Point {
  double minutes;
  double gap;
};

std::vector<Point> run(const Landscape& l, u64 seed, double minutes, const Timetable& lanes) {
  RunScope const scope =
    makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {});
  std::vector<std::vector<Candidate>> const candidates{candidatesOf(card(), scope, TestKind::PRP)};
  double const optimum = optimalT(l, card(), scope, candidates);

  TuneDB db;
  (void)db.parse(std::string{TuneDB::HEADER} + '\n', "sim");
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  Card device{l, card(), db, db.beginSession(env, "", 0, ""), seed};

  std::vector<Point> out;
  for (u32 chunk = 0; laneTime(db, env).total() < minutes * 60; ++chunk) {
    (void)runLoop(device, db, env, card(), scope,
                  {.stop = 0, .lanes = lanes, .seed = seed * 1000 + chunk, .regretEvery = 1000, .maxItems = 20});
    Readings const readings{db, env, card()};
    Publication const pub = publish(scope, candidates, readings);
    out.push_back({.minutes = laneTime(db, env).total() / 60, .gap = trueT(l, card(), scope, pub) / optimum - 1});
    std::printf("%s %llu %.1f %.5f\n", l.name.c_str(), static_cast<unsigned long long>(seed), out.back().minutes,
                out.back().gap);
    std::fflush(stdout);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string which = "all";
  u64 seeds = 5;
  u64 first = 1;
  double minutes = 60;
  Timetable lanes{};
  for (int i = 1; i < argc; ++i) {
    std::string_view const a = argv[i];
    if (a.starts_with("seeds=")) {
      seeds = std::stoull(std::string{a.substr(6)});
    } else if (a.starts_with("first=")) {
      first = std::stoull(std::string{a.substr(6)});
    } else if (a.starts_with("minutes=")) {
      minutes = std::stod(std::string{a.substr(8)});
    } else if (a.starts_with("lanes=")) {
      std::optional<Timetable> const t = parseTimetable(a.substr(6));
      if (!t) {
        std::fprintf(stderr, "bad lanes=\n");
        return 2;
      }
      lanes = *t;
    } else {
      which = a;
    }
  }
  // The loop's own lines would drown the results.
  setStdoutSink([](std::string_view) {});

  for (const Landscape& l : scenarios()) {
    if (which != "all" && which != l.name) { continue; }
    // A trap the card cannot spring tests nothing.
    bool sprung = false;
    for (const Candidate& c : candidatesOf(
           card(), makeScope({.lo = 67'400'000, .hi = 67'600'000, .probe = 67'500'000, .kinds = {TestKind::PRP}}, {}),
           TestKind::PRP)) {
      sprung = sprung || !best(l, card(), c.subject.fft).second.empty();
      if (sprung) { break; }
    }
    if (!sprung) {
      std::printf("%s: no FFT's best is other than its defaults: the scenario tests nothing\n", l.name.c_str());
      continue;
    }

    std::vector<double> at30;
    std::vector<double> atEnd;
    std::vector<double> toTarget;
    for (u64 seed = first; seed < first + seeds; ++seed) {
      std::vector<Point> const points = run(l, seed, minutes, lanes);
      double g30 = points.front().gap;
      double reached = -1;
      for (const Point& p : points) {
        if (p.minutes <= 30) { g30 = p.gap; }
        if (reached < 0 && p.gap <= 0.005) { reached = p.minutes; }
      }
      at30.push_back(g30);
      atEnd.push_back(points.back().gap);
      toTarget.push_back(reached);
    }
    auto worst = [](const std::vector<double>& v) { return *std::ranges::max_element(v); };
    size_t within30 = std::ranges::count_if(at30, [](double g) { return g <= 0.005; });
    size_t withinEnd = std::ranges::count_if(atEnd, [](double g) { return g <= 0.005; });
    std::printf("SUMMARY %s: within 0.5%% at 30 min %zu/%zu (worst %.2f%%), at %.0f min %zu/%zu (worst %.2f%%); "
                "minutes to 0.5%%:",
                l.name.c_str(), within30, at30.size(), 100 * worst(at30), minutes, withinEnd, atEnd.size(),
                100 * worst(atEnd));
    for (double const t : toTarget) {
      if (t < 0) {
        std::printf(" never");
      } else {
        std::printf(" %.0f", t);
      }
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
