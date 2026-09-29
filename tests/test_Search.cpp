// Copyright (C) Jason Lynch

// Tests one entry's search on its own: what it offers from fixed rows and readings, under a price the test sets, with
// no queue around it.

#include "Search.h"

#include "Scheduler.h"

#include "test.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace tune;

namespace {

constexpr const char* SPEC = "512:15:512:212";
constexpr u64 PROBE = 118'063'003;

Env nvidia() {
  Env env;
  env.isNvidia = true;
  env.computeCapability = 806;
  return env;
}

RunScope scope() { return makeScope(ScopeArgs{.lo = 110'000'000, .hi = 135'000'000, .probe = PROBE}, {}); }

Baseline entry() {
  std::vector<Baseline> const all = baselines(nvidia(), scope(), {FFTShape{"512:15:512"}});
  auto const at =
    std::ranges::find_if(all, [](const Baseline& b) { return b.fft.spec() == SPEC && b.band.contains(PROBE); });
  CHECK(at != all.end());
  return at != all.end() ? *at : all.front();
}

// A database with the env and one session on it, and the rows of the one entry searched.
struct Rows {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;

  Rows() {
    env = db.internEnv(dbEnvOf(nvidia()));
    sess = db.beginSession(env, std::string{SPEC} + "@118063003", 0, 1'753'471'200);
    CHECK(env && sess);
  }

  void add(const UseConfig& options, double mean, u32 calls = MIN_CALLS, Status status = Status::Ok,
           u64 exponent = PROBE) {
    CHECK(db.add(RunRow{
      .sess = sess,
      .fft = SPEC,
      .kind = TestKind::PRP,
      .exponent = exponent,
      .regime = regimeOf(FFTConfig{SPEC}, exponent),
      .cfg = db.internCfg(options),
      .m = {.mean = mean, .stddev = 0.1, .blocks = 4 * calls, .calls = calls, .drift = 1, .status = status, .ts = 0}}));
  }
};

// What one re-score gives the search, taken afresh from the rows each time it is asked.
struct Asking {
  Strategy strategy;
  Defaults lines{};
  bool restarts = false;

  std::vector<Candidate> operator()(EntrySearch& search, const Rows& rows, const std::vector<Reading>& readings,
                                    const Worth& worth) const {
    Env const device = nvidia();
    Progress const progress = progressOf(rows.db, rows.env, device);
    SearchContext const context{.device = device,
                                .strategy = strategy,
                                .db = rows.db,
                                .env = rows.env,
                                .progress = progress,
                                .lines = lines,
                                .restarts = restarts};
    return search.offers(context, readings, worth);
  }
};

double flat(Offer, double) { return 1; }

size_t countOf(const std::vector<Candidate>& offers, Offer kind) {
  return size_t(std::ranges::count_if(offers, [&](const Candidate& c) { return c.kind == kind; }));
}

const Candidate* find(const std::vector<Candidate>& offers, const std::string& options) {
  auto const at = std::ranges::find_if(offers, [&](const Candidate& c) { return configText(c.options) == options; });
  return at == offers.end() ? nullptr : &*at;
}

}  // namespace

TEST(an_entry_best_at_the_built_in_defaults_offers_the_lines_before_any_step) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}, .lines = {.global = {{"WMUL", "1"}}, .family = {}}};
  EntrySearch search{entry()};

  std::vector<Candidate> const first = ask(search, rows, {{.config = {}, .cost = 1700}}, flat);
  CHECK(!first.empty() && first.front().kind == Offer::Lines);
  CHECK_EQ(configText(first.front().options), std::string{"WMUL=1"});
  CHECK_EQ(first.front().what, std::string{"the default lines WMUL=1"});
  CHECK_EQ(countOf(first, Offer::Lines), size_t(1));
  CHECK(countOf(first, Offer::Probe) > 0);

  // The same set as a step of its own is not offered twice.
  CHECK_EQ(std::ranges::count_if(first, [](const Candidate& c) { return configText(c.options) == "WMUL=1"; }), 1);

  // Once measured it is answered, whether or not it won.
  rows.add({{"WMUL", "1"}}, 1750);
  std::vector<Candidate> const lost =
    ask(search, rows, {{.config = {}, .cost = 1700}, {.config = {{"WMUL", "1"}}, .cost = 1750}}, flat);
  CHECK_EQ(countOf(lost, Offer::Lines), size_t(0));

  // An entry whose best set is its own is offered the lines as well, and its set with the lines laid over it, which
  // is not offered again as a step of its own.
  Rows other;
  other.add({}, 1700);
  other.add({{"ZEROHACK_W", "0"}}, 1650);
  EntrySearch fresh{entry()};
  std::vector<Candidate> const moved =
    ask(fresh, other, {{.config = {{"ZEROHACK_W", "0"}}, .cost = 1650}, {.config = {}, .cost = 1700}}, flat);
  CHECK_EQ(countOf(moved, Offer::Lines), size_t(2));
  CHECK(moved.size() > 2 && configText(moved[0].options) == "WMUL=1" &&
        configText(moved[1].options) == "WMUL=1,ZEROHACK_W=0");
  CHECK(moved.size() > 2 && moved[1].what == "its best set under the default lines WMUL=1,ZEROHACK_W=0");
  CHECK(std::ranges::all_of(moved, [](const Candidate& c) { return c.cost == 1650; }));
  CHECK_EQ(
    std::ranges::count_if(moved, [](const Candidate& c) { return configText(c.options) == "WMUL=1,ZEROHACK_W=0"; }), 1);
  CHECK(countOf(moved, Offer::Probe) > 0);
}

TEST(an_entry_tries_the_lines_again_each_time_they_move_whatever_it_has_found) {
  Rows rows;
  rows.add({}, 1700);
  rows.add({{"ZEROHACK_W", "0"}}, 1650);
  rows.add({{"WMUL", "1"}}, 1690);
  rows.add({{"WMUL", "1"}, {"ZEROHACK_W", "0"}}, 1660);
  std::vector<Reading> const readings{{.config = {{"ZEROHACK_W", "0"}}, .cost = 1650},
                                      {.config = {{"WMUL", "1"}, {"ZEROHACK_W", "0"}}, .cost = 1660},
                                      {.config = {{"WMUL", "1"}}, .cost = 1690},
                                      {.config = {}, .cost = 1700}};
  EntrySearch search{entry()};

  // Both measured under the lines as they stood.
  Asking const stood{.strategy = {.kind = Strategy::Kind::Single}, .lines = {.global = {{"WMUL", "1"}}, .family = {}}};
  CHECK_EQ(countOf(stood(search, rows, readings, flat), Offer::Lines), size_t(0));

  // Another entry found a key the lines now carry.
  Asking const moved{.strategy = {.kind = Strategy::Kind::Single},
                     .lines = {.global = {{"TAIL_KERNELS", "3"}, {"WMUL", "1"}}, .family = {}}};
  std::vector<Candidate> const again = moved(search, rows, readings, flat);
  CHECK_EQ(countOf(again, Offer::Lines), size_t(2));
  CHECK(again.size() > 2 && configText(again[0].options) == "TAIL_KERNELS=3,WMUL=1");
  CHECK(again.size() > 2 && configText(again[1].options) == "TAIL_KERNELS=3,WMUL=1,ZEROHACK_W=0");
}

TEST(each_offer_is_priced_by_its_kind_from_the_best_set_it_steps_from_and_nothing_worth_nothing_is_offered) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}, .lines = {.global = {{"WMUL", "1"}}, .family = {}}};
  EntrySearch search{entry()};

  std::map<Offer, double> asked;
  std::vector<Candidate> const priced = ask(search, rows, {{.config = {}, .cost = 1700}}, [&](Offer kind, double cost) {
    asked[kind] = cost;
    return kind == Offer::Lines ? 3.0 : 2.0;
  });
  CHECK(asked.contains(Offer::Lines) && asked.contains(Offer::Probe));
  for (const auto& [kind, cost] : asked) { CHECK_EQ(cost, 1700.0); }
  for (const Candidate& c : priced) {
    CHECK_EQ(c.cost, 1700.0);
    CHECK_EQ(c.value, c.kind == Offer::Lines ? 3.0 : 2.0);
  }

  CHECK(ask(search, rows, {{.config = {}, .cost = 1700}}, [](Offer, double) { return 0.0; }).empty());
}

TEST(a_combination_is_offered_once_its_branch_has_nothing_of_a_lower_tier_left) {
  // Tail and Height each have an answer to combine.
  Rows rows;
  rows.add({}, 1700);
  rows.add({{"TAIL_KERNELS", "3"}}, 1710);
  rows.add({{"ZEROHACK_H", "0"}}, 1720);
  std::vector<Reading> const readings{{.config = {}, .cost = 1700},
                                      {.config = {{"TAIL_KERNELS", "3"}}, .cost = 1710},
                                      {.config = {{"ZEROHACK_H", "0"}}, .cost = 1720}};
  Asking const ask{.strategy = Strategy{}};

  // Their groups' other steps are not all answered yet.
  EntrySearch early{entry()};
  std::vector<Candidate> const waiting = ask(early, rows, readings, flat);
  CHECK(countOf(waiting, Offer::Probe) > 0);
  CHECK_EQ(countOf(waiting, Offer::Combo), size_t(0));

  // Where a step is worth nothing but a combination is, the combinations do not wait for steps that will never run.
  std::vector<Candidate> const opened =
    ask(early, rows, readings, [](Offer kind, double) { return kind == Offer::Combo ? 1.0 : 0.0; });
  CHECK(!opened.empty());
  CHECK_EQ(countOf(opened, Offer::Combo), opened.size());

  // Every step answered, none better than the defaults: the combinations, and only those of the lowest tier left.
  for (const Probe& p : probesOf(nvidia(), FFTConfig{SPEC}, {}, Strategy{}).probes) {
    if (p.tier == 1 && !p.config.empty()) { rows.add(p.config, 1800); }
  }
  EntrySearch late{entry()};
  std::vector<Candidate> const combined = ask(late, rows, readings, flat);
  CHECK(!combined.empty());
  CHECK_EQ(countOf(combined, Offer::Combo), combined.size());
  u32 const tier = combined.front().tier;
  CHECK(tier == 2 || tier == 3);
  CHECK(std::ranges::all_of(combined, [&](const Candidate& c) { return c.tier == tier; }));
}

TEST(what_a_row_answers_or_failed_is_not_offered_and_a_started_step_is_resumed_where_it_began) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  EntrySearch search{entry()};
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};

  std::vector<Candidate> const before = ask(search, rows, readings, flat);
  CHECK(before.size() >= 3);
  if (before.size() < 3) { return; }

  // Another exponent the band holds.
  u64 const elsewhere = PROBE + 20'000;
  CHECK(entry().band.contains(elsewhere));
  rows.add(before[0].options, 1750);
  rows.add(before[1].options, 0, 1, Status::Err);
  rows.add(before[2].options, 1760, 1, Status::Ok, elsewhere);

  std::vector<Candidate> const after = ask(search, rows, readings, flat);
  CHECK(!find(after, configText(before[0].options)));
  CHECK(!find(after, configText(before[1].options)));
  const Candidate* const resumed = find(after, configText(before[2].options));
  CHECK(resumed && resumed->exponent == elsewhere && resumed->calls == 1);
  CHECK_EQ(after.size(), before.size() - 2);
}

TEST(a_step_answered_from_an_earlier_best_set_is_offered_again_only_where_it_depends_on_what_moved) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  EntrySearch search{entry()};

  std::vector<Candidate> const first = ask(search, rows, {{.config = {}, .cost = 1700}}, flat);
  CHECK(find(first, "LDSPAD_W=0") && find(first, "ZEROHACK_W=0") && find(first, "WMUL=1"));
  rows.add({{"LDSPAD_W", "0"}}, 1720);
  rows.add({{"ZEROHACK_W", "0"}}, 1720);
  rows.add({{"WMUL", "1"}}, 1600);

  // LDSPAD_W depends on WMUL, ZEROHACK_W on nothing that moved.
  std::vector<Candidate> const next = ask(search, rows,
                                          {{.config = {{"WMUL", "1"}}, .cost = 1600},
                                           {.config = {}, .cost = 1700},
                                           {.config = {{"LDSPAD_W", "0"}}, .cost = 1720},
                                           {.config = {{"ZEROHACK_W", "0"}}, .cost = 1720}},
                                          flat);
  CHECK(find(next, "LDSPAD_W=0,WMUL=1"));
  CHECK(!find(next, "WMUL=1,ZEROHACK_W=0"));
  CHECK(std::ranges::all_of(next, [](const Candidate& c) { return c.cost == 1600; }));
}

TEST(a_configuration_tried_too_often_is_no_longer_offered_by_that_search) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};
  EntrySearch search{entry()};

  std::vector<Candidate> const before = ask(search, rows, readings, flat);
  CHECK(!before.empty());
  if (before.empty()) { return; }
  std::string const first = configText(before.front().options);

  for (u32 n = 1; n < MAX_ATTEMPTS; ++n) { search.tried(nvidia(), before.front().options); }
  CHECK(find(ask(search, rows, readings, flat), first));

  search.tried(nvidia(), before.front().options);
  std::vector<Candidate> const after = ask(search, rows, readings, flat);
  CHECK(!find(after, first));
  CHECK_EQ(after.size(), before.size() - 1);

  // What this process has tried is the search's own.
  EntrySearch other{entry()};
  CHECK(find(ask(other, rows, readings, flat), first));
}

TEST(a_restart_is_offered_only_where_no_step_is_and_carries_on_from_the_last_draw_declared) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}, .restarts = true};
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};
  auto const jumpsOnly = [](Offer kind, double) { return kind == Offer::Restart ? 1.0 : 0.0; };

  EntrySearch search{entry()};
  CHECK_EQ(countOf(ask(search, rows, readings, flat), Offer::Restart), size_t(0));

  std::vector<Candidate> const jump = ask(search, rows, readings, jumpsOnly);
  CHECK_EQ(jump.size(), size_t(1));
  if (jump.size() != 1) { return; }
  CHECK(jump.front().kind == Offer::Restart && jump.front().draw == 0 && jump.front().calls == 0);
  CHECK_EQ(jump.front().cost, 1700.0);

  Asking const still{.strategy = {.kind = Strategy::Kind::Single}};
  CHECK(ask(search, rows, readings, jumpsOnly).size() == 1);
  CHECK(still(search, rows, readings, jumpsOnly).empty());

  // A later process finds the third draw declared and started.
  Baseline const b = entry();
  UseConfig const third = canonicalConfig(nvidia(), b.fft, restartOf(nvidia(), b.fft, b.label(), 2));
  CHECK(rows.db.add(JumpRow{.sess = rows.sess,
                            .fft = SPEC,
                            .kind = TestKind::PRP,
                            .regime = b.band.regime,
                            .cfg = rows.db.internCfg(third),
                            .k = 2,
                            .ts = 0}));
  rows.add(third, 1800, 1);
  EntrySearch later{entry()};
  std::vector<Candidate> const resumed = ask(later, rows, readings, jumpsOnly);
  CHECK(resumed.size() == 1 && resumed.front().draw == 2 && resumed.front().calls == 1);
  CHECK(resumed.size() == 1 && resumed.front().options == third);
}
