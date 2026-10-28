// Copyright (C) Jason Lynch

#include "test.h"

#include "FFTVariants.h"
#include "Publish.h"

#include <optional>
#include <string>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

// A database on one env with one session, into which calls and readings are put by spec and options.
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

  void call(const char* spec, UseConfig options, double us, int times = 2, u64 exponent = 0) {
    u32 const cfg = db.internCfg(options);
    for (int i = 0; i < times; ++i) {
      CHECK(db.add(CallRow{.sess = sess,
                           .fft = spec,
                           .kind = TestKind::PRP,
                           .exponent = exponent ? exponent : scope.probe,
                           .cfg = cfg,
                           .mean = us + 0.1 * i,
                           .sd = 0.2,
                           .blocks = 4,
                           .ts = 1}));
    }
  }

  void roe(const char* spec, UseConfig options, u64 E, double z) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = spec,
                        .exponent = E,
                        .cfg = db.internCfg(options),
                        .checkOk = true,
                        .z = z,
                        .n = 9000,
                        .ts = 1}));
  }

  // The publication refers to the candidates and readings it was made from, which live here until the next one.
  std::vector<std::vector<Candidate>> candidates;
  std::optional<Readings> readings;

  [[nodiscard]] Publication published(const Carried* carried = nullptr) {
    candidates = {candidatesOf(nvidia(), scope, TestKind::PRP)};
    readings.emplace(db, env, nvidia());
    return publish(scope, candidates, *readings, carried);
  }

  // A run over a range the FFTs read so far hold none of.
  void moveAbove() {
    scope = makeScope({.lo = 150'000'000, .hi = 160'000'000, .probe = 155'000'000, .kinds = {TestKind::PRP}}, {});
  }

  // Where the gate first reads an FP64 set of 1K:13:256: its table top.
  [[nodiscard]] u64 top() {
    call("1K:13:256:202", {{"ZEROHACK_H", "0"}}, 2000);
    Publication const p = published();
    CHECK_EQ(p.owedGates.size(), size_t{1});
    return p.owedGates.empty() ? 0 : p.owedGates.front().exponent;
  }
};

}  // namespace

TEST(candidates_hold_the_workload_and_are_timed_at_the_probe) {
  Fixture const f;
  std::vector<Candidate> const cs = candidatesOf(nvidia(), f.scope, TestKind::PRP);
  CHECK(!cs.empty());
  bool sawProbe = false;
  for (const Candidate& c : cs) {
    CHECK(!c.points.empty());
    CHECK(c.interval.contains(c.timing));
    sawProbe = sawProbe || c.timing == f.scope.probe;
  }
  CHECK(sawProbe);
}

TEST(a_line_needs_its_calls_and_its_gate) {
  Fixture f;
  // An NTT needs no gate; an FP64 set owes its reading at the table top.
  f.call("3:1K:8:256:202", {}, 900, 1);
  f.call("1K:13:256:202", {}, 800);

  Publication const one = f.published();
  CHECK(one.of(TestKind::PRP)->lines.empty());
  CHECK_EQ(one.owedGates.size(), size_t{1});
  if (one.owedGates.empty()) { return; }
  CHECK_EQ(one.owedGates.front().fft.spec(), std::string{"1K:13:256:202"});
  CHECK_EQ(one.owedCalls.size(), size_t{1});

  f.call("3:1K:8:256:202", {}, 900, 1);
  f.roe("1K:13:256:202", {}, one.owedGates.front().exponent, 24.4);
  Publication const two = f.published();
  const KindPublication& prp = *two.of(TestKind::PRP);
  CHECK_EQ(prp.lines.size(), size_t{1});  // the NTT reaches further only at a higher cost, so it is on the frontier
  CHECK_EQ(prp.lines.front().subject.fft.spec(), std::string{"1K:13:256:202"});
  CHECK(two.owedGates.empty());
  CHECK(prp.T > 800 * prp.covered && prp.T < 820 * prp.covered);
}

TEST(the_tune_file_names_the_reach_and_the_options) {
  Fixture f;
  f.call("1K:13:256:202", {{"ZEROHACK_W", "0"}}, 800);
  Publication const owed = f.published();
  CHECK_EQ(owed.owedGates.size(), size_t{1});
  f.roe("1K:13:256:202", {{"ZEROHACK_W", "0"}}, owed.owedGates.at(0).exponent, 24.4);

  std::string const text = tuneFileText(*f.published().of(TestKind::PRP), "written by a test");
  CHECK(text.starts_with("# written by a test\n"));
  CHECK(text.find("1K:13:256:202 # ") != std::string::npos);
  CHECK(text.find(", ZEROHACK_W=0\n") != std::string::npos);
  CHECK_EQ(std::string{tuneFileName(TestKind::LL)}, std::string{"tune-ll.txt"});
}

TEST(a_failed_configuration_is_never_a_line) {
  Fixture f;
  u32 const cfg = f.db.internCfg({});
  CHECK(f.db.add(CallRow{.sess = f.sess,
                         .fft = "3:1K:8:256:202",
                         .kind = TestKind::PRP,
                         .exponent = f.scope.probe,
                         .cfg = cfg,
                         .status = Status::Err,
                         .ts = 1}));
  f.call("3:1K:8:256:202", {}, 900);
  CHECK(f.published().of(TestKind::PRP)->lines.empty());
}

TEST(a_line_outside_the_run_stays_published) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);
  f.moveAbove();

  Publication const p = f.published();
  const KindPublication& prp = *p.of(TestKind::PRP);
  CHECK_EQ(prp.written.size(), size_t{1});
  if (prp.written.empty()) { return; }
  CHECK_EQ(prp.written.front().subject.fft.spec(), std::string{"1K:13:256:202"});
  CHECK(prp.written.front().candidate == nullptr);
  CHECK(prp.covered == 0);
  CHECK(tuneFileText(prp, "").find("1K:13:256:202 # ") != std::string::npos);
}

TEST(an_fft_has_a_line_for_each_reach_a_cheaper_set_falls_short_of) {
  Fixture f;
  u64 const top = f.top();
  f.call("1K:13:256:202", {}, 800);
  f.call("1K:13:256:202", {{"ZEROHACK_W", "0"}}, 780);
  f.roe("1K:13:256:202", {}, top, 24.4);
  f.roe("1K:13:256:202", {{"ZEROHACK_W", "0"}}, top, 10);
  u64 const lowered = top - top / 200;
  f.roe("1K:13:256:202", {{"ZEROHACK_W", "0"}}, lowered, 24.4);

  for (bool const outside : {false, true}) {
    if (outside) { f.moveAbove(); }
    Publication const p = f.published();
    const std::vector<Line>& written = p.of(TestKind::PRP)->written;
    CHECK_EQ(written.size(), size_t{2});
    if (written.size() != 2) { return; }
    CHECK(written[0].reading->options == (UseConfig{{"ZEROHACK_W", "0"}}));
    CHECK_EQ(written[0].reach(), lowered);
    CHECK(written[1].reading->options.empty());
    CHECK(written[1].reach() > top);
  }
}

TEST(outside_the_run_an_unsettled_gate_publishes_its_highest_safe_reading) {
  Fixture f;
  u64 const top = f.top();
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, top, 10);
  f.roe("1K:13:256:202", {}, 100'000'000, 24.4);
  CHECK(f.published().of(TestKind::PRP)->written.empty());

  f.moveAbove();
  const std::vector<Line>& written = f.published().of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{1});
  if (written.empty()) { return; }
  CHECK_EQ(written.front().reach(), u64{100'000'000});
}

TEST(an_earlier_builds_line_is_written_until_this_build_has_one) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);

  f.env =
    f.db.internEnv(DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860, .build = 7});
  f.sess = f.db.beginSession(f.env, "", 0, "");
  Carried const carried = carriedLines(f.db, f.env, f.scope.grids);
  CHECK_EQ(carried.size(), size_t{1});

  Publication const before = f.published(&carried);
  const KindPublication& old = *before.of(TestKind::PRP);
  CHECK(old.lines.empty());
  CHECK(old.covered == 0);
  CHECK_EQ(old.written.size(), size_t{1});

  f.call("1K:13:256:202", {}, 900);
  f.roe("1K:13:256:202", {}, f.published(&carried).owedGates.at(0).exponent, 24.4);
  Publication const after = f.published(&carried);
  const std::vector<Line>& written = after.of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{1});
  if (written.empty()) { return; }
  CHECK(written.front().cost() > 850);
}
