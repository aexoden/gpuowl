// Copyright (C) Jason Lynch

// Tests the history a database holds: the file as it stood at a moment, the stretches of time its env was measured, the
// order moments are rebuilt in, the objective over each moment, and what production runs at the probe against what
// the built-in defaults measured there.

#include "History.h"

#include "test.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

// Session 4 on env 1 measures an FP64 set at 1753471274, an NTT at 1753471294 and reads the FP64 set's accuracy at
// 1753471330, which is what lets the gate publish it; session 9 is another card's.  A failure written with no time
// says nothing about when.
const char* const DB =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"Tesla P100-PCIE-16GB\" name=\"Tesla P100-PCIE-16GB\" drv=550.163.01 vendor=nvidia be=ocl cc=600"
  " noasm=0 pdl=0 fp64=1 builtins=1 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
  "cfg   1 -\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "cfg   21 INPLACE=1,PAD=256\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  9 env=2 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "run   4 512:15:512:212 prp 100000000 short32 1 1900.000 2.000 24 6 1.0000 ok 1753471260\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
  "run   4 3:1K:8:512:202 prp 100000000 short32 21 2000.000 5.000 16 4 1.0000 ok 1753471294\n"
  "run   4 4K:8:1K:202 prp 900000000 short32 21 100.000 1.000 16 4 1.0000 unsupported 0\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 900.000 1.000 16 4 1.0000 ok 1753479999\n"
  "roe   4 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471330\n";

constexpr u64 START = 1'753'471'200;
constexpr u64 PROBE = 100'000'000;

RunScope probeScope() {
  RunScope out;
  out.lo = 90'000'000;
  out.hi = 110'000'000;
  out.probe = PROBE;
  out.grids.push_back({.kind = TestKind::PRP, .points = {{PROBE, 1.0}}});
  return out;
}

bool near(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(std::abs(a), std::abs(b)); }

}  // namespace

TEST(a_database_as_it_stood_leaves_out_every_row_stamped_later) {
  std::string const text = "# prpll tunedb v1\n"
                           "cfg   1 -\n"
                           "sess  4 env=1 start=100 gen=0 anchor=-\n"
                           "run   4 512:15:512:212 prp 1000 short32 1 10.000 1.000 4 1 1.0000 ok 150\n"
                           "try   4 512:15:512:212 prp 1000 1 200\n"
                           "done  4 201\n"
                           "run   4 512:15:512:212 prp 1000 short32 1 10.000 1.000 4 1 1.0000 nocompile 0\n"
                           "hint  4 from-a-later-build 999\n"
                           "milestone\n"
                           "# a note\n";
  CHECK_EQ(asOf(text, 200),
           std::string{"# prpll tunedb v1\n"
                       "cfg   1 -\n"
                       "sess  4 env=1 start=100 gen=0 anchor=-\n"
                       "run   4 512:15:512:212 prp 1000 short32 1 10.000 1.000 4 1 1.0000 ok 150\n"
                       "try   4 512:15:512:212 prp 1000 1 200\n"
                       "run   4 512:15:512:212 prp 1000 short32 1 10.000 1.000 4 1 1.0000 nocompile 0\n"
                       "hint  4 from-a-later-build 999\n"
                       "milestone\n"
                       "# a note\n"});
  CHECK_EQ(asOf(text, 99).find("run   4 512:15:512:212 prp 1000 short32 1 10.000 1.000 4 1 1.0000 ok"),
           std::string::npos);
  CHECK_EQ(asOf(text, 1000), text);

  // What is left reads back as a database.
  TuneDB db;
  CHECK(db.parse(asOf(DB, START + 80), "fixture"));
  CHECK_EQ(db.mergedRuns().size(), size_t{3});
}

TEST(an_env_was_measured_from_each_sessions_start_to_its_last_row) {
  std::vector<SessionSpan> const spans = spansOf(DB, 1);
  CHECK_EQ(spans.size(), size_t{1});
  if (spans.size() != 1) { return; }
  CHECK_EQ(spans[0].sess, 4u);
  CHECK_EQ(spans[0].start, START);
  CHECK_EQ(spans[0].end, START + 130);

  CHECK_EQ(spansOf(DB, 2).size(), size_t{1});
  CHECK(spansOf(DB, 3).empty());

  std::vector<SessionSpan> const two{{.sess = 1, .start = 100, .end = 200}, {.sess = 2, .start = 1000, .end = 1050}};
  CHECK_EQ(measuringTime(two), 150.0);
  CHECK_EQ(measuringTime(two, 2), 100.0);
  CHECK_EQ(momentAt(two, 0), u64{100});
  CHECK_EQ(momentAt(two, 100), u64{200});
  CHECK_EQ(momentAt(two, 120), u64{1020});
  CHECK_EQ(momentAt(two, 1e9), u64{1050});
  CHECK_EQ(momentAt({}, 5), u64{0});
}

TEST(moments_are_rebuilt_so_that_any_prefix_spans_the_whole_range) {
  CHECK(spreadOrder(0).empty());
  CHECK(spreadOrder(1) == std::vector<size_t>{0});
  CHECK(spreadOrder(2) == (std::vector<size_t>{0, 1}));
  CHECK(spreadOrder(5) == (std::vector<size_t>{0, 4, 2, 1, 3}));

  std::vector<size_t> const order = spreadOrder(64);
  CHECK_EQ(order.size(), size_t{64});
  CHECK_EQ(std::set<size_t>(order.begin(), order.end()).size(), size_t{64});
  CHECK(order.size() >= 3 && order[0] == 0 && order[1] == 63 && order[2] == 31);
}

TEST(each_moment_of_the_history_is_the_objective_over_the_database_as_it_stood) {
  RunScope const scope = probeScope();
  std::vector<HistoryPoint> const history = replay(DB, 1, scope, 14, [] { return true; });
  CHECK_EQ(history.size(), size_t{14});
  if (history.size() != 14) { return; }

  for (size_t k = 0; k < history.size(); ++k) {
    CHECK(near(history[k].active, 10.0 * double(k)));
    TuneDB db;
    CHECK(db.parse(asOf(DB, START + 10 * k), "fixture"));
    Objective const objective{db, 1, scope};
    CHECK(near(history[k].T, objective.T()));
    CHECK_EQ(history[k].measured, objective.measured());
  }

  // Nothing measured at first; the NTT once it is timed; the FP64 set once the gate has read it, cheaper.
  CHECK(!history.front().probe);
  CHECK_EQ(history.front().measured, 0.0);
  CHECK(history[10].probe && history[10].probe > 2000);
  CHECK(history.back().probe && *history.back().probe < 1800);
  CHECK(history.back().T < history[10].T);

  // Cut short, the history keeps both ends.
  size_t asked = 0;
  std::vector<HistoryPoint> const cut = replay(DB, 1, scope, 14, [&] { return asked++ < 2; });
  CHECK_EQ(cut.size(), size_t{2});
  CHECK(cut.size() == 2 && cut[0].active == 0 && near(cut[1].active, 130));

  CHECK(replay(DB, 3, scope, 14, [] { return true; }).empty());
}

TEST(what_production_runs_at_the_probe_is_set_against_the_built_in_defaults_there) {
  TuneDB db;
  CHECK(db.parse(DB, "fixture"));
  RunScope const scope = probeScope();
  Env const card = db.findEnv(1)->toEnv();
  Objective const objective{db, 1, scope};

  Benefit const benefit = benefitOf(db, 1, card, objective, TestKind::PRP, PROBE);
  CHECK(benefit.now && benefit.now->fft == "512:15:512:212");
  CHECK_EQ(benefit.untunedFft, std::string{"512:15:512:212"});
  // At the cost the selection file ranks by, as production's own is: the mean and two standard errors.
  CHECK(benefit.untuned > 1900 && benefit.untuned < 1902);
  CHECK(benefit.saving() && near(*benefit.saving(), (benefit.untuned - benefit.now->us) / benefit.untuned));
  std::string const text = benefitText(benefit);
  CHECK(text.starts_with("at 100000000 production runs 512:15:512:212 at "));
  CHECK_EQ(text,
           std::string{"at 100000000 production runs 512:15:512:212 at 1776.069 us/it, 6.6% less per iteration "
                       "than the best measured there at the built-in defaults (512:15:512:212, 1901.751 us/it)"});

  // Before anything is published, and with nothing at the built-in defaults to compare with.
  TuneDB early;
  CHECK(early.parse(asOf(DB, START + 70), "fixture"));
  Benefit const none = benefitOf(early, 1, card, Objective{early, 1, scope}, TestKind::PRP, PROBE);
  CHECK(!none.now && !none.saving());
  CHECK_EQ(benefitText(none), std::string{"at 100000000 nothing measured is published yet"});

  Benefit const elsewhere = benefitOf(db, 1, card, objective, TestKind::PRP, 100'000'007);
  CHECK(elsewhere.untunedFft.empty() && !elsewhere.saving());
  CHECK(benefitText(elsewhere).ends_with("; nothing was measured there at the built-in defaults to set it against"));
}

TEST(an_exponent_is_priced_untuned_by_the_cheapest_entry_read_at_the_built_in_defaults_that_serves_it) {
  TuneDB db;
  CHECK(db.parse(DB, "fixture"));
  Env const card = db.findEnv(1)->toEnv();
  Untuned const untuned{db, 1, card};

  // 512:15:512:212 was read at the defaults and under INPLACE=1,PAD=256,TAIL_KERNELS=3; the NTT only under
  // INPLACE=1,PAD=256, which is not its defaults.  So the defaults reading prices the probe, and at the published cost.
  std::optional<Cost> const at = untuned.at(TestKind::PRP, PROBE);
  CHECK(at && at->fft == "512:15:512:212" && at->measured());
  CHECK(at && at->us > 1900 && at->us < 1902);

  // Past what the FFT can serve, nothing read at the defaults does; and another kind has nothing at all.
  CHECK(!untuned.at(TestKind::PRP, 400'000'000));
  CHECK(!untuned.at(TestKind::LL, PROBE));

  // Another card's readings are not this env's.
  Untuned const other{db, 2, db.findEnv(2)->toEnv()};
  CHECK(!other.at(TestKind::PRP, PROBE));
}
