// Copyright (C) Jason Lynch

#include "test.h"

#include "FFTVariants.h"
#include "Publish.h"
#include "Report.h"

#include <algorithm>
#include <iterator>
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

  void call(const char* spec, UseConfig options, double us, int times = 2, u64 exponent = 0,
            TestKind kind = TestKind::PRP) {
    u32 const cfg = db.internCfg(options);
    for (int i = 0; i < times; ++i) {
      CHECK(db.add(CallRow{.sess = sess,
                           .fft = spec,
                           .kind = kind,
                           .exponent = exponent ? exponent : scope.probe,
                           .cfg = cfg,
                           .mean = us + 0.1 * i,
                           .sd = 0.2,
                           .blocks = 4,
                           .ts = 1}));
    }
  }

  void roe(const char* spec, UseConfig options, u64 E, double z, bool ok = true) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = spec,
                        .exponent = E,
                        .cfg = db.internCfg(options),
                        .checkOk = ok,
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

TEST(the_tune_file_names_the_kind_the_regime_the_range_and_the_options) {
  Fixture f;
  f.call("1K:13:256:202", {{"ZEROHACK_W", "0"}}, 800);
  Publication const owed = f.published();
  CHECK_EQ(owed.owedGates.size(), size_t{1});
  f.roe("1K:13:256:202", {{"ZEROHACK_W", "0"}}, owed.owedGates.at(0).exponent, 24.4);

  std::string const text = tuneFileText(f.published(), "written by a test");
  CHECK(text.starts_with(std::string{"# "} + TUNE_FILE_MAGIC + ", build "));
  CHECK(text.find("\n# written by a test\n") != std::string::npos);
  // At 67M this FFT runs a long carry, which it does up to 80768068 and no further, whatever its table says.
  CHECK(text.find("1K:13:256:202 prp long32 20447231-80768068 # ZEROHACK_W=0\n") != std::string::npos);
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
  CHECK(tuneFileText(p, "").find("1K:13:256:202 prp long32 ") != std::string::npos);
}

TEST(the_report_names_a_line_the_run_does_not_hold) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);
  f.moveAbove();

  Publication const p = f.published();
  CHECK(!p.of(TestKind::PRP)->lines.empty());
  CHECK(reportJson({.state = "running", .publication = &p}).find("\"fft\": \"1K:13:256:202\"") != std::string::npos);
}

TEST(an_fft_has_a_line_for_each_reach_a_cheaper_set_falls_short_of) {
  Fixture f;
  u64 const top = f.top();
  // Read in the regime that runs up to the table top, above the long carry the run's own range is in.
  u64 const E = 100'000'000;
  // The cheaper set rounds otherwise, so its readings are its own and it is held to the defaults' at the top.
  UseConfig const cheaper{{"MM2_CHAIN", "1"}};
  f.call("1K:13:256:202", {}, 800, 2, E);
  f.call("1K:13:256:202", cheaper, 780, 2, E);
  f.roe("1K:13:256:202", {}, top, 24.4);
  f.roe("1K:13:256:202", cheaper, top, 10);
  u64 const lowered = top - top / 200;
  f.roe("1K:13:256:202", cheaper, lowered, 24.4);

  for (bool const outside : {false, true}) {
    if (outside) { f.moveAbove(); }
    Publication const p = f.published();
    // The regime they were read in; the set top() read is gated by the defaults' reading too, in its own regime.
    std::vector<Line> written;
    std::ranges::copy_if(p.of(TestKind::PRP)->written, std::back_inserter(written),
                         [](const Line& l) { return l.subject.regime.label() == std::string{"short32"}; });
    CHECK_EQ(written.size(), size_t{2});
    if (written.size() != 2) { return; }
    CHECK(written[0].reading->options == cheaper);
    CHECK_EQ(written[0].reach(), lowered);
    CHECK(written[1].reading->options.empty());
    CHECK(written[1].reach() > top);
    CHECK_EQ(written[0].lo(), written[1].lo());
  }
}

TEST(outside_the_run_an_unsettled_gate_publishes_its_highest_safe_reading) {
  Fixture f;
  u64 const top = f.top();
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, top, 10);
  f.roe("1K:13:256:202", {}, 75'000'000, 24.4);
  CHECK(f.published().of(TestKind::PRP)->written.empty());

  f.moveAbove();
  Publication const p = f.published();
  const std::vector<Line>& written = p.of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{1});
  if (written.empty()) { return; }
  CHECK_EQ(written.front().reach(), u64{75'000'000});
}

TEST(each_regime_of_an_fft_is_a_line_of_its_own) {
  Fixture f;
  // An NTT needs no gate.  Read in two carry regimes, it has two lines, each with its own cost and range.
  f.call("3:1K:8:256:202", {}, 900, 2, 60'000'000);
  f.call("3:1K:8:256:202", {}, 950, 2, 90'000'000);

  Publication const p = f.published();
  const std::vector<Line>& written = p.of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{2});
  if (written.size() != 2) { return; }
  CHECK_EQ(written[0].subject.regime.label(), std::string{"short32"});
  CHECK_EQ(written[0].lo(), u64{48'234'494});
  CHECK_EQ(written[0].reach(), u64{78'434'408});
  CHECK_EQ(written[1].subject.regime.label(), std::string{"short64"});
  CHECK_EQ(written[1].lo(), u64{78'434'409});
  CHECK(written[1].cost() > 949);
  const Line* const at70M = p.of(TestKind::PRP)->at(70'000'000);
  CHECK(at70M && at70M->subject.regime.label() == "short32");
  CHECK(p.of(TestKind::PRP)->at(30'000'000) == nullptr);  // its long carry was never read
  std::string const text = tuneFileText(p, "");
  CHECK(text.find("3:1K:8:256:202 prp short32 48234494-78434408\n") != std::string::npos);
  CHECK(text.find("3:1K:8:256:202 prp short64 78434409-99992208\n") != std::string::npos);
}

TEST(a_line_is_dropped_only_where_cheaper_lines_hold_all_of_it) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);
  // Held throughout by the cheaper long carry of 1K:13:256 (20447231-80768068).
  f.call("3:1K:8:256:202", {}, 900, 2, 60'000'000);
  // Above it: kept.
  f.call("3:1K:8:256:202", {}, 950, 2, 90'000'000);

  Publication const p = f.published();
  const std::vector<Line>& written = p.of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{2});
  if (written.size() != 2) { return; }
  CHECK_EQ(written[0].subject.fft.spec(), std::string{"1K:13:256:202"});
  CHECK_EQ(written[1].subject.regime.label(), std::string{"short64"});
}

TEST(a_kind_the_run_does_not_tune_keeps_its_lines) {
  Fixture f;
  f.call("3:1K:8:256:202", {}, 900, 2, 0, TestKind::LL);

  Publication const p = f.published();
  const KindPublication* const ll = p.of(TestKind::LL);
  CHECK(ll != nullptr);
  if (!ll) { return; }
  CHECK(!ll->tuned);
  CHECK_EQ(ll->written.size(), size_t{1});
  CHECK(p.of(TestKind::PRP)->tuned);
  CHECK(p.coverage() == p.of(TestKind::PRP)->covered);
  CHECK(tuneFileText(p, "").find("3:1K:8:256:202 ll short32 ") != std::string::npos);
}

TEST(an_earlier_builds_line_is_written_until_this_build_has_read_it) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);

  f.env =
    f.db.internEnv(DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860, .build = 7});
  f.sess = f.db.beginSession(f.env, "", 0, "");
  Carried const carried = carriedLines(f.db, f.env);
  CHECK_EQ(carried.size(), size_t{1});

  Publication const before = f.published(&carried);
  const KindPublication& old = *before.of(TestKind::PRP);
  CHECK(old.lines.empty());
  CHECK(old.covered == 0);
  CHECK_EQ(old.written.size(), size_t{1});
  if (old.written.empty()) { return; }
  CHECK(old.written.front().carried);
  // The run holds its FFT and regime, so it owes this build's own reading of it.
  CHECK_EQ(before.owedCalls.size(), size_t{1});
  if (before.owedCalls.empty()) { return; }
  CHECK(before.owedCalls.front().carried);
  CHECK(before.owedCalls.front().options.empty());

  // Another set of the same FFT read by this build leaves the earlier build's line standing.
  f.call("1K:13:256:202", {{"ZEROHACK_W", "0"}}, 1000);
  f.roe("1K:13:256:202", {{"ZEROHACK_W", "0"}}, f.published(&carried).owedGates.at(0).exponent, 24.4);
  Publication const middle = f.published(&carried);
  const std::vector<Line>& both = middle.of(TestKind::PRP)->written;
  CHECK_EQ(both.size(), size_t{1});
  if (both.empty()) { return; }
  CHECK(both.front().carried);

  // Its gate is settled already: the set read above rounds as it does.
  f.call("1K:13:256:202", {}, 900);
  CHECK(f.published(&carried).owedGates.empty());
  Publication const after = f.published(&carried);
  const std::vector<Line>& written = after.of(TestKind::PRP)->written;
  CHECK_EQ(written.size(), size_t{1});
  if (written.empty()) { return; }
  CHECK(!written.front().carried);
  CHECK(written.front().cost() > 850);
  CHECK(std::ranges::none_of(after.owedCalls, &ConcludeItem::carried));
}

TEST(lines_are_carried_only_under_the_same_tag) {
  Fixture f;
  f.call("1K:13:256:202", {}, 800);
  f.roe("1K:13:256:202", {}, f.published().owedGates.at(0).exponent, 24.4);

  u32 const other = f.db.internEnv(
    DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860, .build = 7, .tag = "875"});
  CHECK(other != f.env);
  CHECK_EQ(carriedLines(f.db, other).size(), size_t{0});
}

TEST(a_call_at_the_power_limit_costs_its_energy_over_what_the_card_holds) {
  Fixture f;
  u64 const E = 60'000'000;
  f.call("3:1K:8:256:202", {}, 900, 2, E);
  // 198 mJ an iteration at a 220 W limit: 198 / (0.95 * 220) ms, some 947 us, where the calls timed 900.
  CHECK(f.db.add(PowerRow{.sess = f.sess,
                          .fft = "3:1K:8:256:202",
                          .kind = TestKind::PRP,
                          .exponent = E,
                          .cfg = f.db.internCfg({}),
                          .mjPerIt = 198,
                          .watts = 220,
                          .capW = 220,
                          .smMHz = 2500,
                          .source = PowerSource::Instant,
                          .ts = 1}));
  FFTConfig const fft{"3:1K:8:256:202"};
  Subject const subject{.fft = fft, .kind = TestKind::PRP, .regime = regimeOf(fft, E)};

  Readings const held{f.db, f.env, nvidia(), HOLD_SHARE};
  const Reading* const r = held.find(subject, {});
  CHECK(r != nullptr);
  if (!r) { return; }
  CHECK(std::abs(r->mean - 198'000 / (HOLD_SHARE * 220)) < 1e-6);
  CHECK_EQ(r->powerBound, 2u);
  CHECK(std::abs(r->mjPerIt - 198) < 1e-9 && std::abs(r->watts - 220) < 1e-9);
  for (const Observation& o : held.observations()) { CHECK(o.cost > 940); }

  Readings const timed{f.db, f.env, nvidia(), 0};
  const Reading* const t = timed.find(subject, {});
  CHECK(t && t->mean < 901 && t->powerBound == 0);

  // Where the card can hold the whole limit, 198 mJ an iteration at 220 W is the 900 us the calls timed.
  Readings const loose{f.db, f.env, nvidia(), 1.0};
  CHECK(loose.find(subject, {}) && loose.find(subject, {})->mean < 901);

  // The floor is on by default.
  Readings const plain{f.db, f.env, nvidia()};
  CHECK(plain.find(subject, {}) && plain.find(subject, {})->powerBound == 2);
}

TEST(only_instantaneous_power_sets_a_cost) {
  Fixture f;
  u64 const E = 60'000'000;
  f.call("3:1K:8:256:202", {}, 900, 2, E);
  // The energy counter, which lags behind a call: recorded, never a floor.
  CHECK(f.db.add(PowerRow{.sess = f.sess,
                          .fft = "3:1K:8:256:202",
                          .kind = TestKind::PRP,
                          .exponent = E,
                          .cfg = f.db.internCfg({}),
                          .mjPerIt = 198,
                          .watts = 220,
                          .capW = 220,
                          .smMHz = 2500,
                          .source = PowerSource::Counter,
                          .ts = 1}));
  FFTConfig const fft{"3:1K:8:256:202"};
  Readings const readings{f.db, f.env, nvidia(), HOLD_SHARE};
  const Reading* const r = readings.find({.fft = fft, .kind = TestKind::PRP, .regime = regimeOf(fft, E)}, {});
  CHECK(r && r->mean < 901 && r->powerBound == 0 && std::abs(r->mjPerIt - 198) < 1e-9);
}

TEST(a_set_that_changes_the_rounding_owes_its_reference_reading_and_is_held_to_it) {
  Fixture f;
  UseConfig const moved{{"MM2_CHAIN", "1"}};
  f.call("1K:13:256:202", moved, 2000);

  Publication const own = f.published();
  CHECK_EQ(own.owedGates.size(), size_t{1});
  if (own.owedGates.empty()) { return; }
  u64 const top = own.owedGates.front().exponent;
  CHECK(own.owedGates.front().options == moved && own.owedGates.front().of == moved);

  // Read at the table top, it owes its reference's reading there, which settles its own gate.
  f.roe("1K:13:256:202", moved, top, 22.0);
  Publication const reference = f.published();
  CHECK_EQ(reference.owedGates.size(), size_t{1});
  if (reference.owedGates.empty()) { return; }
  CHECK(reference.owedGates.front().options.empty());
  CHECK(reference.owedGates.front().of == moved);
  CHECK_EQ(reference.owedGates.front().exponent, top);
  CHECK(reference.of(TestKind::PRP)->lines.empty());

  // 8% below its reference: no line at the table's reach, and a lower reading of its own is owed.
  f.roe("1K:13:256:202", {}, top, 24.0);
  Publication const cut = f.published();
  CHECK(cut.of(TestKind::PRP)->lines.empty());
  CHECK_EQ(cut.owedGates.size(), size_t{1});
  if (cut.owedGates.empty()) { return; }
  CHECK(cut.owedGates.front().options == moved);
  CHECK(cut.owedGates.front().exponent < top);

  // Lower down it is judged against the reference's reading at the top, so what it owes next is its own reading.
  f.roe("1K:13:256:202", moved, cut.owedGates.front().exponent, 30.0);
  for (const GateItem& g : f.published().owedGates) { CHECK(g.options == moved); }
}

TEST(a_rounding_reading_serves_every_set_that_rounds_alike_where_the_card_shares_them) {
  Fixture f;
  u64 const top = f.top();
  FFTConfig const fft = canonical(FFTConfig{"1K:13:256:202"});
  UseConfig const read = canonicalConfig(nvidia(), fft, {{"WMUL", "1"}});
  UseConfig const alike = canonicalConfig(nvidia(), fft, {{"ZEROHACK_H", "0"}});
  UseConfig const otherwise = canonicalConfig(nvidia(), fft, {{"TAIL_KERNELS", "0"}});
  f.roe("1K:13:256:202", read, top, 24.0);
  (void)f.published();

  // One set's reading is a reading of every set of the FFT that rounds as it does, and of no other.
  CHECK_EQ(f.readings->roes(fft, alike).size(), size_t{1});
  CHECK(f.readings->roes(fft, otherwise).empty());
  CHECK(verdictFor(*f.readings, fft, alike, f.scope.lo).state == Accuracy::Confirmed);
  CHECK(f.published().owedGates.empty() || f.published().owedGates.front().options != alike);

  // A failed check is the set's own: neither lent to the sets that round alike nor outweighed by theirs.
  f.roe("1K:13:256:202", alike, top, 24.0, false);
  f.roe("1K:13:256:202", read, top - 2, 24.0, false);
  (void)f.published();
  std::vector<RoeReading> const own = f.readings->roes(fft, alike);
  CHECK_EQ(own.size(), size_t{1});
  CHECK(!own.empty() && !own.front().checkOk);
  CHECK_EQ(f.readings->roes(fft, read).size(), size_t{2});

  // Not on a card the flags have not been measured on: there every set is read on its own.
  Readings const amd{f.db, f.env, Env{.isAmd = true}};
  CHECK(amd.roes(fft, canonicalConfig(Env{.isAmd = true}, fft, {{"ZEROHACK_H", "1"}})).empty());
}
