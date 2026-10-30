// Copyright (C) Jason Lynch

#include "test.h"

#include "Contenders.h"
#include "Report.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

// The FP64 FFT the tests publish, which owes an accuracy reading, and two NTTs that owe none.  At 66M-70M the FP64
// one runs a long carry and the NTTs a short one, over the whole workload.
constexpr const char* FP64 = "1K:13:256:202";
constexpr const char* NTT8 = "3:1K:8:256:202";
constexpr const char* NTT9 = "3:1K:9:256:202";

const UseConfig NO_W{{"ZEROHACK_W", "0"}};
const UseConfig NO_H{{"ZEROHACK_H", "0"}};

// A database on one env with one session, as test_Publish's.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  RunScope scope;

  Fixture() {
    CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "fixture"));
    env = db.internEnv(DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
    sess = db.beginSession(env, "", 0, "");
    scope = makeScope({.lo = 66'000'000, .hi = 70'000'000, .probe = 67'000'000, .kinds = {TestKind::PRP}}, {});
  }

  void call(const char* spec, const UseConfig& options, double us, int times = 2) {
    u32 const cfg = db.internCfg(options);
    for (int i = 0; i < times; ++i) {
      CHECK(db.add(CallRow{.sess = sess,
                           .fft = spec,
                           .kind = TestKind::PRP,
                           .exponent = scope.probe,
                           .cfg = cfg,
                           .mean = us + 0.1 * i,
                           .sd = 0.2,
                           .blocks = 4,
                           .ts = 1}));
    }
  }

  // A safe reading at the FP64 FFT's table top, where its gate reads first.
  void safe(const UseConfig& options) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = FP64,
                        .exponent = top,
                        .cfg = db.internCfg(options),
                        .checkOk = true,
                        .z = 24.4,
                        .n = 9000,
                        .ts = 1}));
  }

  std::vector<std::vector<Candidate>> candidates;
  std::optional<Readings> readings;
  Publication publication;
  u64 top = 0;

  void publishNow() {
    candidates = {candidatesOf(nvidia(), scope, TestKind::PRP)};
    readings.emplace(db, env, nvidia());
    publication = publish(scope, candidates, *readings);
  }

  [[nodiscard]] Subject subject(const char* spec, const char* regime) const {
    for (const Candidate& c : candidates.front()) {
      if (c.subject.fft.spec() == spec && c.subject.regime.label() == regime) { return c.subject; }
    }
    CHECK(false);
    return candidates.front().front().subject;
  }

  [[nodiscard]] std::vector<StandingOf> standings(const char* spec, const char* regime) const {
    Subject const s = subject(spec, regime);
    return standingsOf(*publication.of(TestKind::PRP), s, heldCandidate(candidates.front(), s), *readings);
  }

  [[nodiscard]] std::vector<KindContest> table(ContestSettings settings = {}) const {
    return contendersOf(nvidia(), scope, candidates, *readings, publication, [](const Candidate&) { return AxisFilter{}; },
                        settings);
  }

  // The FP64 defaults published at 800 over the whole workload; ZEROHACK_W=0 dearer and as safe; ZEROHACK_H=0 cheaper
  // on one call; NTT8 at 820, undercut; NTT9 at 1000 on one call, which no exponent would take.
  Fixture& front() {
    call(FP64, {}, 800);
    publishNow();
    CHECK_EQ(publication.owedGates.size(), size_t{1});
    top = publication.owedGates.empty() ? 0 : publication.owedGates.front().exponent;
    safe({});
    call(FP64, NO_W, 850);
    safe(NO_W);
    call(FP64, NO_H, 790, 1);
    call(NTT8, {}, 820);
    call(NTT9, {}, 1000, 1);
    publishNow();
    return *this;
  }
};

const StandingOf* find(const std::vector<StandingOf>& standings, const UseConfig& options) {
  auto const it = std::ranges::find_if(standings, [&](const StandingOf& s) { return s.reading->options == options; });
  return it == standings.end() ? nullptr : &*it;
}

bool has(const std::vector<std::string>& lines, const std::string& text) {
  return std::ranges::any_of(lines, [&](const std::string& line) { return line.find(text) != std::string::npos; });
}

}  // namespace

TEST(each_reading_says_why_it_holds_a_line_or_does_not) {
  Fixture f;
  f.front();

  std::vector<StandingOf> const fp64 = f.standings(FP64, "long32");
  CHECK_EQ(fp64.size(), size_t{3});
  const StandingOf* const published = find(fp64, {});
  const StandingOf* const outreached = find(fp64, NO_W);
  const StandingOf* const short_ = find(fp64, NO_H);
  CHECK(published && outreached && short_);
  if (!published || !outreached || !short_) { return; }
  CHECK(published->standing == Standing::Published && published->line &&
        published->line->reading == published->reading);
  CHECK(outreached->standing == Standing::Outreached && outreached->by == published->reading);
  CHECK(short_->standing == Standing::ShortOfCalls);
  CHECK(fp64.front().reading == published->reading);  // the cheapest first, as the publisher counts: by its error bar

  // The NTT reaches as far, dearer: the FP64 line runs where the run weighs it most, its probe.
  std::vector<StandingOf> const ntt = f.standings(NTT8, "short32");
  CHECK_EQ(ntt.size(), size_t{1});
  CHECK(ntt.front().standing == Standing::Undercut);
  CHECK_EQ(ntt.front().at, f.scope.probe);
  CHECK(ntt.front().line == published->line);
  CHECK(whyText(ntt.front(), false)
          .starts_with("undercut: at " + std::to_string(f.scope.probe) + " production runs 1K:13:256:202 at "));

  // A second call is owed only where it might take a line.
  CHECK(whyText(*short_, true).find("the next is owed") != std::string::npos);
  CHECK(whyText(f.standings(NTT9, "short32").front(), false).find("none owed") != std::string::npos);
}

TEST(a_set_whose_gate_is_owed_says_where) {
  Fixture f;
  f.call(FP64, {}, 800);
  f.publishNow();
  std::vector<StandingOf> const s = f.standings(FP64, "long32");
  CHECK_EQ(s.size(), size_t{1});
  CHECK(s.front().standing == Standing::GateOwed);
  CHECK(s.front().at == f.publication.owedGates.front().exponent);
}

TEST(a_line_is_contested_by_the_rivals_near_it) {
  Fixture f;
  f.front();

  std::vector<KindContest> const table = f.table();
  CHECK_EQ(table.size(), size_t{1});
  if (table.empty()) { return; }
  // One line holds the whole workload.
  CHECK_EQ(table.front().contests.size(), size_t{1});
  const Contest& c = table.front().contests.front();
  CHECK(c.line && c.line->subject.fft.spec() == FP64);
  CHECK(c.lo == 66'000'000u && c.hi == 70'000'000u);
  CHECK(c.weight > 0.999 && c.weight < 1.001);
  CHECK(c.lineRead.positions.total > 0 && c.lineRead.branches == 1);

  // Its own FFT's cheaper set short of a call, then the NTT 2.5% dearer; the other NTT, 25% dearer, is past within=.
  CHECK_EQ(c.rivals.size(), size_t{2});
  if (c.rivals.size() != 2) { return; }
  CHECK(c.rivals[0].standing.reading->options == NO_H && c.rivals[0].ratio < 1 && c.rivals[0].owed);
  CHECK(c.rivals[1].subject.fft.spec() == NTT8 && c.rivals[1].standing.standing == Standing::Undercut);
  CHECK(c.rivals[1].ratio > 1.02 && c.rivals[1].ratio < 1.03);
  CHECK(c.rivals[1].lo < c.rivals[1].ratio && c.rivals[1].ratio < c.rivals[1].hi);
  CHECK(c.rivals[1].read.positions.total > 0);

  // within=all shows the far NTT; top=1 only the nearest.
  CHECK_EQ(f.table({.within = std::numeric_limits<double>::infinity(), .top = TOP}).front().contests.front().rivals.size(),
           size_t{3});
  std::vector<Rival> const one = f.table({.within = WITHIN, .top = 1}).front().contests.front().rivals;
  CHECK(one.size() == 1 && one.front().standing.reading->options == NO_H);

  std::vector<std::string> const lines = contestLines(table);
  CHECK(lines.size() == 3 && lines[0].starts_with("prp 66000000-70000000, 100% of the weight: "));
  CHECK(lines[2].find("3:1K:8:256:202 short32 -use the defaults: undercut, 2 calls; branches 1 of ") !=
        std::string::npos);

  std::string const json = reportJson({.publication = &f.publication, .contenders = &table});
  CHECK(json.find("\"contenders\": [\n    {\"kind\": \"prp\", \"minExponent\": 66000000") != std::string::npos);
  CHECK(json.find("\"standing\": \"undercut\"") != std::string::npos);
  CHECK(json.find("\"why\": \"short of calls: 1 of 2, the next is owed\"") != std::string::npos);
}

TEST(a_stretch_no_line_holds_ranks_its_rivals_against_the_cheapest) {
  Fixture f;
  f.call(NTT8, {}, 900, 1);
  f.call(NTT9, {}, 918, 1);
  f.publishNow();
  std::vector<KindContest> const table = f.table();
  CHECK(table.size() == 1 && table.front().contests.size() == 1);
  const Contest& c = table.front().contests.front();
  CHECK(!c.line);
  CHECK_EQ(c.rivals.size(), size_t{2});
  if (c.rivals.size() != 2) { return; }
  CHECK(c.rivals[0].subject.fft.spec() == NTT8 && c.rivals[0].ratio == 1);
  CHECK(c.rivals[1].ratio > 1.019 && c.rivals[1].ratio < 1.021);
  CHECK(contestLines(table).front().ends_with("no line"));
}

TEST(explain_says_why_each_set_on_an_fft_is_or_is_not_used) {
  Fixture f;
  f.front();
  auto explain = [&](ExplainQuery const& q) {
    return explainLines(nvidia(), q, f.scope, f.candidates, *f.readings, f.publication);
  };

  std::vector<std::string> const all = explain({.fft = FP64, .options = {}});
  CHECK(has(all, "1K:13:256:202 prp long32: a candidate over 66000000-70000000"));
  CHECK(has(all, "-use the defaults: published, "));
  CHECK(has(all, "-use ZEROHACK_W=0: outreached: the defaults is cheaper"));
  CHECK(has(all, "-use ZEROHACK_H=0: short of calls: 1 of 2"));

  // One set named: just it, or where it was never read, the nearest that were.
  std::vector<std::string> const one = explain({.fft = FP64, .options = NO_W});
  CHECK(has(one, "outreached") && !has(one, "published"));
  std::vector<std::string> const never = explain({.fft = FP64, .options = UseConfig{{"ZEROHACK_W", "0"}, {"ZEROHACK_H", "0"}}});
  CHECK(has(never, "-use ZEROHACK_H=0,ZEROHACK_W=0 was never read on it; the nearest sets that were:"));
  CHECK(has(never, "-use ZEROHACK_W=0: outreached"));

  // A shape alone takes in every variant; the variants nothing read are named on one line per regime.
  std::vector<std::string> const shape = explain({.fft = "3:1K:7:256", .options = {}});
  CHECK(has(shape, "prp short32, candidates of this workload never read: 3:1K:7:256:202"));
  CHECK(has(shape, "prp short64, candidates of this workload never read: 3:1K:7:256:202"));
  CHECK(has(explain({.fft = "3:1K:8:256", .options = {}}), "undercut"));
  CHECK(has(explain({.fft = "9:9:9", .options = {}}), "9:9:9: nothing read, and no candidate of this workload"));
  CHECK(matchesFft({.fft = "512:15:256", .options = {}}, "512:15:256:001"));
  CHECK(!matchesFft({.fft = "512:15:25", .options = {}}, "512:15:256:001"));
}
