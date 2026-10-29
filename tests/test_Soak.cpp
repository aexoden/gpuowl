// Copyright (C) Jason Lynch

// Soaking finalists: the records, what readings make of them, which finalists a soak takes and when.

#include "test.h"

#include "Soak.h"
#include "TuneCommand.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

bool near(double a, double b, double tolerance = 1e-9) { return std::abs(a - b) < tolerance; }

// One env and session, a PRP run over 66-70M, and calls and soaks put in by candidate.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  RunScope scope;
  std::vector<std::vector<Candidate>> candidates;

  Fixture() {
    CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "fixture"));
    env = db.internEnv(DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
    sess = db.beginSession(env, "", 0, "");
    scope = makeScope({.lo = 66'000'000, .hi = 70'000'000, .probe = 67'000'000, .kinds = {TestKind::PRP}}, {});
    candidates = {candidatesOf(nvidia(), scope, TestKind::PRP)};
  }

  // Candidates of exact arithmetic, which owe no accuracy reading, that hold the probe: one per FFT.
  [[nodiscard]] std::vector<const Candidate*> exact(size_t n) const {
    std::vector<const Candidate*> out;
    for (const Candidate& c : candidates.front()) {
      if (out.size() < n && exactArithmetic(c.subject.fft) && c.interval.contains(scope.probe)) { out.push_back(&c); }
    }
    CHECK_EQ(out.size(), n);
    return out;
  }

  void call(const Candidate& c, double us) {
    for (int i = 0; i < 2; ++i) {
      CHECK(db.add(CallRow{.sess = sess,
                           .fft = c.subject.fft.spec(),
                           .kind = TestKind::PRP,
                           .exponent = c.timing,
                           .cfg = db.internCfg({}),
                           .mean = us + 0.1 * i,
                           .sd = 0.2,
                           .blocks = 4,
                           .ts = 1}));
    }
  }

  void soak(const Candidate& c, std::vector<double> rounds, u32 group = 1) {
    for (u32 r = 0; r < rounds.size(); ++r) {
      CHECK(db.add(SoakRow{.sess = sess,
                           .fft = c.subject.fft.spec(),
                           .kind = TestKind::PRP,
                           .exponent = scope.probe,
                           .cfg = db.internCfg({}),
                           .group = group,
                           .round = r,
                           .mean = rounds[r],
                           .sd = 1,
                           .blocks = 100,
                           .watts = 210,
                           .sec = 60,
                           .ts = 2}));
    }
  }

  std::optional<Readings> readings;
  [[nodiscard]] Publication published() {
    readings.emplace(db, env, nvidia());
    return publish(scope, candidates, *readings);
  }
};

}  // namespace

TEST(a_soak_slice_is_kept_as_written) {
  Fixture f;
  const Candidate& c = *f.exact(1).front();
  f.soak(c, {900, 945});
  CHECK(f.db.text().find("soak 1 " + c.subject.fft.spec() + " prp " + std::to_string(f.scope.probe) +
                         " 1 1 1 945.000 1.000 100 210.0 60.0 2\n") != std::string::npos);
  TuneDB back;
  CHECK(back.parse(f.db.text(), "soak"));
  CHECK_EQ(back.soaks().size(), size_t{2});
  CHECK_EQ(back.text(), f.db.text());
  CHECK_EQ(f.db.nextSoakGroup(), 2u);
  CHECK(near(soakTime(f.db, f.env), 120));
  SoakRow bad = f.db.soaks().front();
  bad.group = 0;
  CHECK(!f.db.add(bad));
}

TEST(a_soaked_configuration_costs_what_it_sustained_and_calibrates_the_rest) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(2);
  f.call(*cs[0], 900);
  f.call(*cs[1], 950);
  // The warm-up (round 0) reads nothing of the configuration.
  f.soak(*cs[0], {880, 944, 946});

  Readings const readings{f.db, f.env, nvidia()};
  const Reading* const soaked = readings.find(cs[0]->subject, {});
  const Reading* const other = readings.find(cs[1]->subject, {});
  CHECK(soaked && other);
  if (!soaked || !other) { return; }
  CHECK(soaked->isSoaked() && soaked->soakSlices == 2);
  CHECK(near(soaked->mean, 945));
  CHECK(near(soaked->timed, 900.05));
  double const k = 945 / 900.05;
  CHECK(near(readings.calibration(), k));
  CHECK(!other->isSoaked());
  CHECK(near(other->mean, 950.05 * k, 1e-6));
  CHECK(near(other->timed, 950.05));
  for (const Observation& o : readings.observations()) { CHECK(o.cost > 940); }

  Fixture none;
  none.call(*none.exact(1).front(), 900);
  CHECK(Readings(none.db, none.env, nvidia()).calibration() == 1);
}

TEST(a_soak_takes_a_line_and_its_closest_rivals_until_they_are_soaked) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(4);
  f.call(*cs[0], 900);
  f.call(*cs[1], 910);
  f.call(*cs[2], 920);
  f.call(*cs[3], 1000);  // more than 3% behind: not a finalist

  SoakSettings const settings{};
  std::optional<SoakPlan> const plan = planSoak(f.published(), f.candidates, *f.readings, settings);
  CHECK(plan.has_value());
  if (!plan) { return; }
  CHECK_EQ(plan->exponent, f.scope.probe);
  CHECK_EQ(plan->targets.size(), size_t{3});
  for (size_t i = 0; i < 3; ++i) { CHECK(plan->targets[i].candidate == cs[i]); }
  CHECK(!planSoak(f.published(), f.candidates, *f.readings, settings, {plan->key()}).has_value());
  CHECK(!planSoak(f.published(), f.candidates, *f.readings, SoakSettings{.share = 0}).has_value());
  CHECK_EQ(planSoak(f.published(), f.candidates, *f.readings, SoakSettings{.top = 2})->targets.size(), size_t{2});

  // Soaked, they reorder by what they sustain, and nothing is left to soak.
  f.soak(*cs[0], {900, 945, 945});
  f.soak(*cs[1], {950, 950}, 1);
  f.soak(*cs[2], {940, 940}, 1);
  Publication const after = f.published();
  CHECK(after.of(TestKind::PRP)->lines.front().reading->isSoaked());
  CHECK(!planSoak(after, f.candidates, *f.readings, settings).has_value());
}

TEST(a_soak_waits_until_its_share_of_the_tuning_time_pays_for_it) {
  SoakSettings const settings{};
  SoakPlan plan;
  plan.targets.resize(3);
  double const length = settings.length(3);  // 2 minutes of warm-up and three rounds of three minutes
  CHECK(near(length, 660));
  CHECK(!soakDue(0, 9 * length - 1, plan, settings));
  CHECK(soakDue(0, 9 * length, plan, settings));
  CHECK(!soakDue(length, 9 * length, plan, settings));
  CHECK(!soakDue(0, 1e9, SoakPlan{}, settings));
  CHECK(!soakDue(0, 1e9, plan, SoakSettings{.share = 0}));
}

TEST(soak_settings_are_read) {
  auto parsed = [](const char* text) { return parseTuneCommand(text); };
  auto refused = [](const char* text) {
    try {
      (void)parseTuneCommand(text);
    } catch (const std::string&) { return true; }
    return false;
  };
  CHECK(parsed("").soak == SoakSettings{});
  SoakSettings const s =
    parsed("soak=5%,soakTop=2,soakMargin=2%,soakWarm=90s,soakSlice=30s,soakRounds=4").soak;
  CHECK(near(s.share, 0.05) && s.top == 2 && near(s.margin, 0.02) && near(s.warm, 90) && near(s.slice, 30) &&
        s.rounds == 4);
  CHECK(near(parsed("soak=0").soak.share, 0) && near(parsed("soakWarm=0").soak.warm, 0));
  for (const char* bad : {"soak=5", "soak=100%", "soakTop=0", "soakRounds=17", "soakSlice=0", "soakSlice=0.5s",
                          "status,soak=5%"}) {
    CHECK(refused(bad));
  }
}
