// Copyright (C) Jason Lynch

// Duels: the records, and how a duel's outcome ranks the two configurations it compared.

#include "test.h"

#include "Publish.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

bool near(double a, double b, double tolerance = 1e-9) { return std::abs(a - b) < tolerance; }

// One env and session, a PRP run over 66-70M, and calls, soaks and duels put in by candidate.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  RunScope scope;
  std::vector<std::vector<Candidate>> candidates;
  std::optional<Readings> readings;

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

  void soak(const Candidate& c, double us) {
    for (u32 r = 0; r < 2; ++r) {
      CHECK(db.add(SoakRow{.sess = sess,
                           .fft = c.subject.fft.spec(),
                           .kind = TestKind::PRP,
                           .exponent = c.timing,
                           .cfg = db.internCfg({}),
                           .group = 1,
                           .round = r,
                           .mean = us,
                           .sd = 1,
                           .blocks = 100,
                           .watts = 210,
                           .sec = 60,
                           .ts = 2}));
    }
  }

  [[nodiscard]] DuelRow duelRow(const Candidate& incumbent, const Candidate& challenger, DuelOutcome outcome) {
    return DuelRow{.sess = sess,
                   .kind = TestKind::PRP,
                   .fft = incumbent.subject.fft.spec(),
                   .exponent = incumbent.timing,
                   .cfg = db.internCfg({}),
                   .challengerFft = challenger.subject.fft.spec(),
                   .challengerExponent = challenger.timing,
                   .challengerCfg = db.internCfg({}),
                   .pairs = 4,
                   .ratio = 1.002,
                   .lo = 0.999,
                   .hi = 1.005,
                   .outcome = outcome,
                   .sec = 50,
                   .ts = 3};
  }

  void duel(const Candidate& incumbent, const Candidate& challenger, DuelOutcome outcome) {
    CHECK(db.add(duelRow(incumbent, challenger, outcome)));
  }

  [[nodiscard]] Publication published() {
    readings.emplace(db, env, nvidia());
    return publish(scope, candidates, *readings);
  }

  // The configuration production would run at the probe.
  [[nodiscard]] const Candidate* atProbe() {
    Publication const p = published();
    const Line* const line = p.of(TestKind::PRP)->at(scope.probe);
    CHECK(line != nullptr);
    return line ? line->candidate : nullptr;
  }

  [[nodiscard]] const Reading& reading(const Candidate& c) {
    if (!readings) { readings.emplace(db, env, nvidia()); }
    const Reading* const r = readings->find(c.subject, {});
    CHECK(r != nullptr);
    return *r;
  }
};

}  // namespace

TEST(a_duel_record_is_kept_as_written) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(2);
  f.duel(*cs[0], *cs[1], DuelOutcome::Even);
  std::string const line = "duel 1 prp " + cs[0]->subject.fft.spec() + ' ' + std::to_string(cs[0]->timing) + " 1 " +
    cs[1]->subject.fft.spec() + ' ' + std::to_string(cs[1]->timing) + " 1 4 1.00200 0.99900 1.00500 even 50.0 3\n";
  CHECK(f.db.text().find(line) != std::string::npos);
  TuneDB back;
  CHECK(back.parse(f.db.text(), "duel"));
  CHECK_EQ(back.duels().size(), size_t{1});
  CHECK(back.duels().front().outcome == DuelOutcome::Even);
  CHECK_EQ(back.text(), f.db.text());

  for (DuelOutcome const o :
       {DuelOutcome::Challenger, DuelOutcome::Incumbent, DuelOutcome::Even, DuelOutcome::Undecided}) {
    CHECK(parseDuelOutcome(toString(o)) == o);
  }
  CHECK(!parseDuelOutcome("draw"));

  DuelRow bad = f.duelRow(*cs[0], *cs[1], DuelOutcome::Even);
  bad.pairs = 0;
  CHECK(!f.db.add(bad));
  bad = f.duelRow(*cs[0], *cs[1], DuelOutcome::Even);
  bad.hi = 0.99;
  CHECK(!f.db.add(bad));
  bad = f.duelRow(*cs[0], *cs[1], DuelOutcome::Even);
  bad.challengerFft = "nonsense";
  CHECK(!f.db.add(bad));
}

TEST(a_challenger_that_does_not_win_its_duel_ranks_behind_the_incumbent) {
  for (DuelOutcome const outcome : {DuelOutcome::Incumbent, DuelOutcome::Even, DuelOutcome::Undecided}) {
    Fixture f;
    std::vector<const Candidate*> const cs = f.exact(2);
    f.call(*cs[0], 910);
    f.call(*cs[1], 900);
    CHECK(f.atProbe() == cs[1]);
    CHECK(f.reading(*cs[0]).first < f.reading(*cs[1]).first);
    CHECK(!f.readings->dueled(Readings::keyOf(cs[0]->subject, {}), Readings::keyOf(cs[1]->subject, {})));

    f.duel(*cs[0], *cs[1], outcome);
    CHECK(f.atProbe() == cs[0]);
    const Reading& incumbent = f.reading(*cs[0]);
    const Reading& challenger = f.reading(*cs[1]);
    CHECK(f.readings->dueled(Readings::keyOf(cs[1]->subject, {}), Readings::keyOf(cs[0]->subject, {})));
    CHECK(incumbent.beaten == 0);
    CHECK(challenger.pessimistic() > incumbent.pessimistic());
    CHECK(near(challenger.pessimistic(), incumbent.pessimistic(), 1e-3));
    // Its own readings are as they were.
    CHECK(near(challenger.mean, 900.05));
  }
}

TEST(a_challenger_that_wins_its_duel_ranks_ahead_whatever_its_mean) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(2);
  f.call(*cs[0], 900);
  f.call(*cs[1], 905);
  CHECK(f.atProbe() == cs[0]);
  f.duel(*cs[0], *cs[1], DuelOutcome::Challenger);
  CHECK(f.atProbe() == cs[1]);
  CHECK(f.reading(*cs[1]).beaten == 0);
  CHECK(f.reading(*cs[0]).beaten > 0);
}

TEST(the_latest_duel_between_two_configurations_stands) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(2);
  f.call(*cs[0], 910);
  f.call(*cs[1], 900);
  f.duel(*cs[0], *cs[1], DuelOutcome::Incumbent);
  CHECK(f.atProbe() == cs[0]);
  // Read again the other way round, the challenger of the first now the incumbent.
  f.duel(*cs[1], *cs[0], DuelOutcome::Incumbent);
  CHECK(f.atProbe() == cs[1]);
}

TEST(duels_rank_through_a_chain) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(3);
  f.call(*cs[0], 920);
  f.call(*cs[1], 910);
  f.call(*cs[2], 900);
  // 0 beat 1, and 1 beat 2: 2, the cheapest by its calls, ranks behind both.
  f.duel(*cs[0], *cs[1], DuelOutcome::Incumbent);
  f.duel(*cs[1], *cs[2], DuelOutcome::Incumbent);
  CHECK(f.atProbe() == cs[0]);
  CHECK(f.reading(*cs[2]).pessimistic() > f.reading(*cs[1]).pessimistic());
  CHECK(f.reading(*cs[1]).pessimistic() > f.reading(*cs[0]).pessimistic());

  // A cycle settles too.
  f.duel(*cs[2], *cs[0], DuelOutcome::Incumbent);
  f.readings.reset();
  double const top = f.reading(*cs[0]).pessimistic();
  CHECK(near(f.reading(*cs[1]).pessimistic(), top, 1e-3) && near(f.reading(*cs[2]).pessimistic(), top, 1e-3));
}

TEST(what_two_soaked_configurations_sustained_settles_their_duel) {
  Fixture f;
  std::vector<const Candidate*> const cs = f.exact(2);
  f.call(*cs[0], 910);
  f.call(*cs[1], 900);
  f.soak(*cs[0], 950);
  f.soak(*cs[1], 940);
  f.duel(*cs[0], *cs[1], DuelOutcome::Incumbent);
  CHECK(f.atProbe() == cs[1]);
  CHECK(f.reading(*cs[1]).beaten == 0);
}
