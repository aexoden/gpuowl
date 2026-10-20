// Copyright (C) Jason Lynch

// Tests that classic mode reads upstream's words as upstream does, keeps an option value only where it beats the lines
// by more than its margin, times the shapes under what the sweeps settled, and, replayed over a database that lacks a
// reading, stops at that reading.

#include "Classic.h"

#include "Scheduler.h"
#include "test.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env cuda() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 806}; }

// Answers each reading from a cost model, with a spread small enough to decide any gap the model sets.
class FakeBench final : public ClassicBench {
public:
  explicit FakeBench(std::function<double(const FFTConfig&, const UseConfig&)> cost) : cost_{std::move(cost)} {}

  std::optional<Measurement> read(const FFTConfig& fft, TestKind, u64, const UseConfig& options, u32 calls) override {
    reads.push_back(fft.spec() + " " + configText(options));
    return Measurement{
      .mean = cost_(fft, options), .stddev = 0.01, .blocks = 4 * calls, .calls = calls, .drift = 1, .ts = 1};
  }
  bool gate(const FFTConfig&, u64, const UseConfig&) override { return false; }
  void declareBootstrap(const FFTConfig& fft, u64) override { boots.push_back(fft.spec()); }
  bool stopped() const override { return false; }
  void say(const std::string&) override {}

  std::vector<std::string> reads;
  std::vector<std::string> boots;

private:
  std::function<double(const FFTConfig&, const UseConfig&)> cost_;
};

RunScope scopeOf(u64 lo, u64 hi) { return makeScope(ScopeArgs{.lo = lo, .hi = hi, .probe = 0}, {}); }

}  // namespace

TEST(classic_settings_round_trip_as_words) {
  ClassicSettings s;
  s.types = ClassicTypes::Both;
  s.sweeps = false;
  s.inplaceOnly = false;
  s.fp32 = Group3::Off;
  s.fp6431 = Group3::Only;
  s.k1x256 = Group3::On;
  s.m61 = Group3::On;
  s.pfa = Group3::Off;
  s.regsOnly = true;
  s.fine = 0.05;
  s.tk = -1;
  s.keepMargin = 0.01;
  s.typeMargin = 0.3;
  s.whVariants = WhVariants::Accurate;
  s.maxCalls = 9;
  s.tkLosses = 6;
  s.tkBad = 0.1;

  ClassicSettings back;
  for (const std::string& word : classicWords(s)) { CHECK(parseClassicWord(word, back)); }
  CHECK(back == s);

  ClassicSettings defaults;
  for (const std::string& word : classicWords(ClassicSettings{})) { CHECK(parseClassicWord(word, defaults)); }
  CHECK(defaults == ClassicSettings{});
}

TEST(classic_mode_reads_upstreams_words) {
  ClassicSettings s;
  CHECK(parseClassicWord("fp64", s) && s.types == ClassicTypes::Fp64);
  CHECK(parseClassicWord("ntt", s) && s.types == ClassicTypes::Both);
  CHECK(parseClassicWord("noconfig", s) && !s.sweeps);
  CHECK(parseClassicWord("nofp32", s) && s.fp32 == Group3::Off);
  CHECK(parseClassicWord("pfa=2", s) && s.pfa == Group3::Only);
  CHECK(parseClassicWord("1k256", s) && s.k1x256 == Group3::On);
  CHECK(parseClassicWord("fine", s) && s.fine == 0.02);
  CHECK(parseClassicWord("fine=5", s) && s.fine == 0.05);
  CHECK(parseClassicWord("tk=auto", s) && s.tk == -1);
  CHECK(parseClassicWord("inplace", s) && s.inplaceOnly == true);
  CHECK(!parseClassicWord("strategy=single", s));
  for (const char* bad : {"tk=4", "pfa=3", "keepMargin=x", "fine=60", "maxCalls=1", "whVariants=some"}) {
    bool refused = false;
    try {
      (void)parseClassicWord(bad, s);
    } catch (const std::string&) { refused = true; }
    CHECK(refused);
  }

  // As a -tune word: upstream's range words name the workload, quick= is taken and set aside, and outside classic mode
  // those words still name -oldtune.
  TuneCommand const c = parseTuneCommand("mode=classic,fp64,exp=100M-120M,quick=7,tk=auto,keepMargin=1%");
  CHECK(c.classic);
  CHECK(c.classicSettings.types == ClassicTypes::Fp64 && c.classicSettings.tk == -1);
  CHECK(c.classicSettings.keepMargin == 0.01);
  CHECK_EQ(c.scope.lo, u64(100'000'000));
  CHECK_EQ(c.scope.hi, u64(120'000'000));
  TuneCommand const first = parseTuneCommand("fp64,mode=classic");
  CHECK(first.classic && first.classicSettings.types == ClassicTypes::Fp64);
  bool refused = false;
  try {
    (void)parseTuneCommand("fp64");
  } catch (const std::string& why) { refused = why.find("-oldtune") != std::string::npos; }
  CHECK(refused);

  // A run's recorded settings read back as the same run.
  RunScope const scope = makeScope(c.scope, {});
  std::string const word = runSettings(scope, c);
  CHECK(word.find(",mode=classic,") != std::string::npos);
  TuneCommand const again = parseTuneCommand(word);
  CHECK(again.classic && again.classicSettings == c.classicSettings);
  CHECK_EQ(runSettings(makeScope(again.scope, {}), again), word);
}

TEST(a_classic_sweep_keeps_what_beats_the_lines_by_its_margin_and_the_shapes_run_under_it) {
  // TAIL_TRIGS=0 is 3% faster, past upstream's 0.3%; UNROLL_W=0 0.1% faster, inside it; WMUL=1 slower.
  auto const cost = [](const FFTConfig& fft, const UseConfig& o) {
    double us = double(fft.size()) / 1000;
    if (o.contains("TAIL_TRIGS") && o.at("TAIL_TRIGS") == "0") { us *= 0.97; }
    if (o.contains("UNROLL_W") && o.at("UNROLL_W") == "0") { us *= 0.999; }
    if (o.contains("WMUL") && o.at("WMUL") == "1") { us *= 1.05; }
    return us;
  };
  TuneDB db;
  RunScope const scope = scopeOf(9'000'000, 10'000'000);
  Env const device = cuda();
  ClassicSettings settings;
  settings.types = ClassicTypes::Fp64;
  FakeBench bench{cost};
  ClassicOutcome const out = runClassic({.device = device,
                                         .db = db,
                                         .env = 1,
                                         .scope = scope,
                                         .kinds = {TestKind::PRP},
                                         .settings = settings,
                                         .tailQueues = TailQueues::Ntt},
                                        bench);
  CHECK(out.complete);
  CHECK_EQ(out.lines.global.count("TAIL_TRIGS"), size_t(1));
  CHECK(!out.lines.global.contains("UNROLL_W"));
  CHECK(!out.lines.global.contains("WMUL"));
  CHECK(!bench.boots.empty());

  // Every shape after the sweeps is read with what they kept, where it applies.
  size_t shapes = 0;
  bool sweepsDone = false;
  for (const std::string& r : bench.reads) {
    sweepsDone = sweepsDone || r.find("TAIL_TRIGS=0") != std::string::npos;
    if (sweepsDone && r.find("TAIL_TRIGS=0") != std::string::npos) { ++shapes; }
  }
  CHECK(shapes > 3);
}

TEST(a_classic_replay_stops_at_the_first_reading_the_rows_lack) {
  TuneDB db;
  RunScope const scope = scopeOf(9'000'000, 10'000'000);
  Env const device = cuda();
  ReplayBench replay{db, 1, device};
  ClassicOutcome const out = runClassic({.device = device,
                                         .db = db,
                                         .env = 1,
                                         .scope = scope,
                                         .kinds = {TestKind::PRP},
                                         .settings = {},
                                         .tailQueues = TailQueues::Ntt},
                                        replay);
  CHECK(!out.complete);
  CHECK(!replay.next().empty());
  CHECK(out.stage.find("type check") != std::string::npos);
}

TEST(the_register_pass_tries_upstreams_candidates_in_upstreams_order) {
  // One measured entry, and its carryFused's plan: two blocks with the compiler's registers.
  Env const device = cuda();
  TuneDB db;
  u32 const env = db.internEnv(dbEnvOf(device));
  u32 const sess = db.beginSession(env, "", 0, 1'753'471'200);
  FFTConfig const fft{"512:8:512:202"};
  u64 const exponent = 75'000'013;
  CHECK(db.add(RunRow{.sess = sess,
                      .fft = fft.spec(),
                      .kind = TestKind::PRP,
                      .exponent = exponent,
                      .regime = regimeOf(fft, exponent),
                      .cfg = db.internCfg({}),
                      .m = {.mean = 1000, .stddev = 0.01, .blocks = 8, .calls = 2, .drift = 1, .ts = 1}}));
  CHECK(db.add(RegsRow{.sess = sess,
                       .fft = fft.spec(),
                       .kind = TestKind::PRP,
                       .key = "REGCF64",
                       .resources = {.regs = 96, .localBytes = 0, .sharedBytes = 0, .threads = 256},
                       .plan = {.k0 = 2, .occupancies = {{2, 128}, {3, 80}, {4, 64}}, .lift = false},
                       .ts = 2}));

  // Three blocks is the fastest occupancy, and its register count faster still.
  auto const cost = [](const FFTConfig&, const UseConfig& o) {
    std::string const v = o.contains("REGCF64") ? o.at("REGCF64") : "";
    return v == "3" ? 950.0 : v == "80" ? 940.0 : v == "4" ? 990.0 : 1000.0;
  };
  ClassicSettings settings;
  settings.regsOnly = true;
  FakeBench bench{cost};
  RunScope const scope = scopeOf(70'000'000, 80'000'000);
  ClassicOutcome const out = runClassic(
    {.device = device, .db = db, .env = env, .scope = scope, .kinds = {TestKind::PRP}, .settings = settings}, bench);
  CHECK(out.complete);

  std::vector<std::string> tried;
  for (const std::string& r : bench.reads) {
    size_t const at = r.find("REGCF64=");
    if (at == std::string::npos) { continue; }
    std::string const v = r.substr(at + 8, r.find_first_of(", ", at + 8) - at - 8);
    if (tried.empty() || tried.back() != v) { tried.push_back(v); }
  }
  // k0 = 2 (k0 - 1 has no plan of its own), then 3, then 4 since 3 beat both, then 3 blocks' 80 registers.
  CHECK(tried.size() >= 4);
  if (tried.size() >= 4) {
    CHECK((std::vector<std::string>{tried.begin(), tried.begin() + 4}) ==
          (std::vector<std::string>{"2", "3", "4", "80"}));
  }
}
