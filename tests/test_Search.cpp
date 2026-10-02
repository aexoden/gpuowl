// Copyright (C) Jason Lynch

// Tests one entry's search on its own: what it offers from fixed rows and readings, under a price the test sets, with
// no queue around it.

#include "Search.h"

#include "Scheduler.h"

#include "test.h"

#include <algorithm>
#include <map>
#include <set>
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
           u64 exponent = PROBE, u64 ts = 0) {
    CHECK(db.add(RunRow{
      .sess = sess,
      .fft = SPEC,
      .kind = TestKind::PRP,
      .exponent = exponent,
      .regime = regimeOf(FFTConfig{SPEC}, exponent),
      .cfg = db.internCfg(options),
      .m = {
        .mean = mean, .stddev = 0.1, .blocks = 4 * calls, .calls = calls, .drift = 1, .status = status, .ts = ts}}));
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

  // An entry whose best set is its own is offered its set with the lines laid over it first, which is not offered
  // again as a step of its own, and then the lines alone: measured first, the lines alone could become the best set,
  // and what the entry found would no longer be laid under them.
  Rows other;
  other.add({}, 1700);
  other.add({{"ZEROHACK_W", "0"}}, 1650);
  EntrySearch fresh{entry()};
  std::vector<Candidate> const moved =
    ask(fresh, other, {{.config = {{"ZEROHACK_W", "0"}}, .cost = 1650}, {.config = {}, .cost = 1700}}, flat);
  CHECK_EQ(countOf(moved, Offer::Lines), size_t(2));
  CHECK(moved.size() > 2 && configText(moved[0].options) == "WMUL=1,ZEROHACK_W=0" &&
        configText(moved[1].options) == "WMUL=1");
  CHECK(moved.size() > 2 && moved[0].what == "its best set under the default lines WMUL=1,ZEROHACK_W=0");
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
  CHECK(again.size() > 2 && configText(again[0].options) == "TAIL_KERNELS=3,WMUL=1,ZEROHACK_W=0");
  CHECK(again.size() > 2 && configText(again[1].options) == "TAIL_KERNELS=3,WMUL=1");
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

TEST(a_combination_is_offered_once_the_groups_it_combines_have_nothing_of_a_lower_tier_left) {
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

  // Every step of Tail and Height answered: their combination, beside the other groups' steps, which it does not wait
  // for; nothing combining a group with steps left.
  std::vector<Probe> const all = probesOf(nvidia(), FFTConfig{SPEC}, {}, Strategy{}).probes;
  auto const within = [](const Probe& p, std::initializer_list<Group> groups) {
    return std::ranges::all_of(p.groups, [&](Group g) { return std::ranges::find(groups, g) != groups.end(); });
  };
  for (const Probe& p : all) {
    if (p.tier == 1 && !p.config.empty() && within(p, {Group::Tail, Group::Height})) { rows.add(p.config, 1800); }
  }
  EntrySearch pair{entry()};
  std::vector<Candidate> const paired = ask(pair, rows, readings, flat);
  CHECK(countOf(paired, Offer::Probe) > 0);
  CHECK(countOf(paired, Offer::Combo) > 0);
  for (const Candidate& c : paired) {
    if (c.kind == Offer::Combo) { CHECK(c.what.starts_with("Tail+Height ")); }
  }

  // Every step answered, none better than the defaults: the combinations, and only those of the lowest tier left.
  for (const Probe& p : all) {
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

TEST(the_stages_of_a_search_by_group_take_turns_a_step_at_a_time_counting_what_the_rows_took) {
  Rows rows;
  rows.add({}, 1700);
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};
  Asking const ask{.strategy = Strategy{}};
  ProbeList const list = probesOf(nvidia(), FFTConfig{SPEC}, {}, Strategy{});
  std::map<std::string, const Probe*> byText;
  for (const Probe& p : list.probes) { byText[configText(p.config)] = &p; }

  // Each step offered, whether it is structural and which turn of its stage it is, given what each stage has had.
  struct Turn {
    bool structural = false;
    u32 part = 0;
    size_t turn = 0;
  };
  auto const turnsOf = [&](const std::vector<Candidate>& offers, std::map<u32, size_t> had) {
    std::vector<Turn> out;
    for (const Candidate& c : offers) {
      auto const at = byText.find(configText(c.options));
      CHECK(at != byText.end());
      if (at == byText.end()) { continue; }
      out.push_back({.structural = at->second->structural, .part = at->second->part, .turn = had[at->second->part]++});
    }
    return out;
  };
  auto const inTurn = [](const std::vector<Turn>& turns) {
    auto const steps = std::ranges::find_if(turns, [](const Turn& t) { return !t.structural; });
    return std::all_of(steps, turns.end(), [](const Turn& t) { return !t.structural; }) &&
      std::is_sorted(steps, turns.end(), [](const Turn& a, const Turn& b) { return a.turn < b.turn; });
  };

  // Structural steps first, then every stage's first step before any stage's second.
  EntrySearch search{entry()};
  std::vector<Turn> const first = turnsOf(ask(search, rows, readings, flat), {});
  CHECK(!first.empty() && first.front().structural);
  CHECK(inTurn(first));
  CHECK(!first.empty() && first.back().turn > 0);

  // The largest stage's first three steps taken by the rows: it has had three turns, and goes behind the others'.
  std::map<u32, size_t> sizes;
  for (const Probe& p : list.probes) { ++sizes[p.part]; }
  u32 const big = std::ranges::max_element(sizes, {}, [](const auto& s) { return s.second; })->first;
  CHECK(sizes[big] > 4);
  u32 measured = 0;
  for (const Probe& p : list.probes) {
    if (p.part == big && measured < 3) {
      rows.add(p.config, 1800);
      ++measured;
    }
  }
  EntrySearch later{entry()};
  std::vector<Turn> const next = turnsOf(ask(later, rows, readings, flat), {{big, 3}});
  CHECK(inTurn(next));
  auto const own = std::ranges::find(next, big, &Turn::part);
  CHECK(own != next.end() && own->turn == 3);
  CHECK(std::none_of(own, next.end(), [&](const Turn& t) { return t.part != big && t.turn < 3; }));
}

TEST(other_branches_take_turns_beside_the_best_one_whose_stages_take_as_many_as_theirs_together) {
  Rows rows;
  rows.add({}, 1700);
  rows.add({{"SHUFL_BYTES_W", "16"}}, 1720);
  rows.add({{"INPLACE", "0"}}, 1730);
  std::vector<Reading> const readings{{.config = {}, .cost = 1700},
                                      {.config = {{"SHUFL_BYTES_W", "16"}}, .cost = 1720},
                                      {.config = {{"INPLACE", "0"}}, .cost = 1730}};
  Asking const ask{.strategy = Strategy{}};
  EntrySearch search{entry()};

  // The other branches' steps priced below the best branch's, by what the set they step from costs.
  std::vector<Candidate> const offers = ask(search, rows, readings, [](Offer, double cost) { return 1700 / cost; });
  std::vector<bool> other;
  for (const Candidate& c : offers) {
    if (c.kind == Offer::Probe) { other.push_back(c.cost > 1700); }
  }
  auto const first = std::ranges::find(other, true);
  CHECK(first != other.end());
  CHECK(std::find(first, other.end(), false) != other.end());

  // One stage every branch has: of two other branches, each takes a turn of it for every two the best branch takes.
  std::map<double, std::vector<size_t>> memory;
  for (size_t k = 0; k < offers.size(); ++k) {
    if (offers[k].kind == Offer::Probe && offers[k].what.starts_with("Memory 1 ")) {
      memory[offers[k].cost].push_back(k);
    }
  }
  CHECK_EQ(memory.size(), size_t(3));
  CHECK(std::ranges::all_of(memory, [](const auto& m) { return m.second.size() > 4; }));
  if (memory.size() != 3 || memory[1700].size() < 5) { return; }
  for (double const cost : {1720.0, 1730.0}) {
    if (memory[cost].size() < 3) { continue; }
    CHECK(memory[1700][2] < memory[cost][1] && memory[cost][1] < memory[1700][3]);
    CHECK(memory[1700][4] < memory[cost][2]);
  }
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

  // Ahead of every step not begun.
  CHECK(resumed && resumed == &after.front());
}

TEST(a_measurement_begun_is_finished_where_it_began_after_the_best_set_has_moved_on) {
  // ZEROHACK_W=0 has one call, at 1300, when WMUL=1 concludes at 1600 and becomes the best set.  The steps are now
  // taken from WMUL=1, which lists ZEROHACK_W=0 only with WMUL=1 beside it.
  Rows rows;
  rows.add({}, 1700);
  u64 const elsewhere = PROBE + 20'000;
  CHECK(entry().band.contains(elsewhere));
  rows.add({{"ZEROHACK_W", "0"}}, 1300, 1, Status::Ok, elsewhere);
  rows.add({{"WMUL", "1"}}, 1600);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  EntrySearch search{entry()};
  std::vector<Reading> const readings{{.config = {{"WMUL", "1"}}, .cost = 1600}, {.config = {}, .cost = 1700}};
  auto const byKind = [](Offer kind, double) { return kind == Offer::Probe ? 1.0 : 0.25; };

  std::vector<Candidate> const offers = ask(search, rows, readings, byKind);
  CHECK(find(offers, "WMUL=1,ZEROHACK_W=0"));
  const Candidate* const begun = find(offers, "ZEROHACK_W=0");
  CHECK(begun && begun->kind == Offer::Probe && begun->exponent == elsewhere && begun->calls == 1);
  CHECK(begun && begun->cost == 1600 && begun->value == 1.0 && begun->observed == 1300);
  CHECK(begun && begun->what == "unfinished ZEROHACK_W=0");
  CHECK(begun && begun == &offers.front());
  CHECK_EQ(std::ranges::count_if(offers, [](const Candidate& c) { return configText(c.options) == "ZEROHACK_W=0"; }),
           1);

  // Priced as the steps are, but offered where they are worth nothing, with what its call read to be valued by.
  auto const noSteps = [](Offer kind, double) { return kind == Offer::Probe ? 0.0 : 1.0; };
  std::vector<Candidate> const unpriced = ask(search, rows, readings, noSteps);
  const Candidate* const still = find(unpriced, "ZEROHACK_W=0");
  CHECK(still && still->value == 0 && still->observed == 1300 && still->calls == 1 && still == &unpriced.front());

  // Once it concludes it is answered.
  rows.add({{"ZEROHACK_W", "0"}}, 1310, MIN_CALLS, Status::Ok, elsewhere);
  CHECK(!find(ask(search, rows, readings, byKind), "ZEROHACK_W=0"));
}

TEST(a_step_taken_from_an_earlier_best_set_is_offered_again_after_every_step_never_taken) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  EntrySearch search{entry()};

  std::vector<Candidate> const first = ask(search, rows, {{.config = {}, .cost = 1700}}, flat);
  CHECK(find(first, "LDSPAD_W=0") && find(first, "ZEROHACK_W=0") && find(first, "WMUL=1"));
  rows.add({{"LDSPAD_W", "0"}}, 1720);
  rows.add({{"ZEROHACK_W", "0"}}, 1720);
  rows.add({{"WMUL", "1"}}, 1600);

  // LDSPAD_W depends on WMUL, so at WMUL=1 it is a step never taken; ZEROHACK_W depends on nothing that moved, so its
  // step was taken, but from the defaults, which says what it did there and nothing certain of what it does here.
  std::vector<Candidate> const next = ask(search, rows,
                                          {{.config = {{"WMUL", "1"}}, .cost = 1600},
                                           {.config = {}, .cost = 1700},
                                           {.config = {{"LDSPAD_W", "0"}}, .cost = 1720},
                                           {.config = {{"ZEROHACK_W", "0"}}, .cost = 1720}},
                                          flat);
  const Candidate* const fresh = find(next, "LDSPAD_W=0,WMUL=1");
  const Candidate* const again = find(next, "WMUL=1,ZEROHACK_W=0");
  CHECK(fresh && again);
  CHECK(std::ranges::all_of(next, [](const Candidate& c) { return c.cost == 1600; }));
  if (!again) { return; }
  CHECK_EQ(again, &next.back());

  // Priced as a jump is, the steps never taken as steps are.
  auto const byKind = [](Offer kind, double) { return kind == Offer::Restart ? 0.25 : 1.0; };
  std::vector<Candidate> const priced = ask(search, rows,
                                            {{.config = {{"WMUL", "1"}}, .cost = 1600},
                                             {.config = {}, .cost = 1700},
                                             {.config = {{"LDSPAD_W", "0"}}, .cost = 1720},
                                             {.config = {{"ZEROHACK_W", "0"}}, .cost = 1720}},
                                            byKind);
  const Candidate* const cheap = find(priced, "WMUL=1,ZEROHACK_W=0");
  const Candidate* const full = find(priced, "LDSPAD_W=0,WMUL=1");
  CHECK(cheap && cheap->value == 0.25 && cheap->kind == Offer::Probe);
  CHECK(full && full->value == 1.0);

  // Only a row of that very configuration answers it.
  rows.add({{"WMUL", "1"}, {"ZEROHACK_W", "0"}}, 1610);
  CHECK(!find(ask(search, rows,
                  {{.config = {{"WMUL", "1"}}, .cost = 1600},
                   {.config = {}, .cost = 1700},
                   {.config = {{"LDSPAD_W", "0"}}, .cost = 1720},
                   {.config = {{"ZEROHACK_W", "0"}}, .cost = 1720},
                   {.config = {{"WMUL", "1"}, {"ZEROHACK_W", "0"}}, .cost = 1610}},
                  flat),
              "WMUL=1,ZEROHACK_W=0"));
}

TEST(a_value_a_build_failed_with_is_offered_after_everything_else_and_holds_back_no_combination) {
  Rows rows;
  rows.add({}, 1700);
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  EntrySearch search{entry()};
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};

  std::vector<Candidate> const before = ask(search, rows, readings, flat);
  CHECK(find(before, "WMUL=1") && find(before, "ZEROHACK_W=0"));
  CHECK(rows.db.add(NogoRow{.sess = rows.sess, .fft = SPEC, .key = "WMUL", .val = "1", .ts = 1}));

  // Still offered, since a build that failed beside other settings is no verdict on this one: last.
  std::vector<Candidate> const after = ask(search, rows, readings, flat);
  CHECK_EQ(after.size(), before.size());
  CHECK(!after.empty() && configText(after.back().options) == "WMUL=1");

  // The configuration that failed is answered by its own row, whatever the pin says.
  rows.add({{"WMUL", "1"}}, 0, 1, Status::NoCompile);
  CHECK(!find(ask(search, rows, readings, flat), "WMUL=1"));

  // Under hybrid, a step that sets it does not keep the combinations of its branch waiting.
  Rows hybrid;
  hybrid.add({}, 1700);
  hybrid.add({{"TAIL_KERNELS", "3"}}, 1710);
  hybrid.add({{"ZEROHACK_H", "0"}}, 1720);
  std::vector<Reading> const seeds{{.config = {}, .cost = 1700},
                                   {.config = {{"TAIL_KERNELS", "3"}}, .cost = 1710},
                                   {.config = {{"ZEROHACK_H", "0"}}, .cost = 1720}};
  std::vector<Probe> const steps = probesOf(nvidia(), FFTConfig{SPEC}, {}, Strategy{}).probes;
  const Probe* pin = nullptr;
  for (const Probe& p : steps) {
    if (p.tier != 1 || p.config.empty()) { continue; }
    if (!pin && !p.key.empty() && p.config.size() == 1) {
      pin = &p;
      continue;
    }
    hybrid.add(p.config, 1800);
  }
  CHECK(pin != nullptr);
  if (!pin) { return; }
  CHECK(hybrid.db.add(
    NogoRow{.sess = hybrid.sess, .fft = SPEC, .key = pin->key, .val = pin->config.begin()->second, .ts = 1}));
  EntrySearch combined{entry()};
  std::vector<Candidate> const offers = Asking{.strategy = Strategy{}}(combined, hybrid, seeds, flat);
  CHECK(countOf(offers, Offer::Combo) > 0);
  CHECK(!offers.empty() && offers.back().options == pin->config);
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

TEST(a_restart_is_due_ahead_of_every_step_each_time_the_entry_has_measured_another_period_of_sets) {
  Rows rows;
  Baseline const b = entry();
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}, .restarts = true};
  auto const noJumps = [](Offer kind, double) { return kind == Offer::Restart ? 0.0 : 2.0; };

  // Sets that are neither steps from the defaults nor draws of the entry's own sequence, all dearer than the defaults.
  std::set<std::string> filled{"-"};
  u32 k = 0;
  auto fill = [&](u32 sets, u64 ts) {
    for (u32 added = 0; added < sets; ++k) {
      UseConfig const other = canonicalConfig(nvidia(), b.fft, restartOf(nvidia(), b.fft, "elsewhere", k));
      if (other.size() < 2 || !filled.insert(configText(other)).second) { continue; }
      rows.add(other, 1800, MIN_CALLS, Status::Ok, PROBE, ts);
      ++added;
    }
  };
  rows.add({}, 1700, MIN_CALLS, Status::Ok, PROBE, 100);
  fill(RESTART_PERIOD - 2, 100);

  EntrySearch search{b};
  std::vector<Candidate> const early = ask(search, rows, readings, noJumps);
  CHECK(!early.empty());
  CHECK_EQ(countOf(early, Offer::Restart), size_t(0));

  fill(1, 100);
  std::vector<Candidate> const due = ask(search, rows, readings, noJumps);
  CHECK(!due.empty() && due.front().kind == Offer::Restart && due.front().draw == 0);
  CHECK(!due.empty() && due.front().value == 2.0);
  CHECK(countOf(due, Offer::Probe) > 0);
  if (due.empty()) { return; }

  // Not without restarts at all.
  Asking const none{.strategy = {.kind = Strategy::Kind::Single}};
  CHECK_EQ(countOf(none(search, rows, readings, noJumps), Offer::Restart), size_t(0));

  // Declared: the next is due once as many sets again have been measured since.
  CHECK(rows.db.add(JumpRow{.sess = rows.sess,
                            .fft = SPEC,
                            .kind = TestKind::PRP,
                            .regime = b.band.regime,
                            .cfg = rows.db.internCfg(due.front().options),
                            .k = 0,
                            .ts = 200}));
  // Begun, it is finished before anything else, and then not offered again until as many sets again are measured.
  rows.add(due.front().options, 1800, 1, Status::Ok, PROBE, 201);
  std::vector<Candidate> const begun = ask(search, rows, readings, noJumps);
  CHECK(!begun.empty() && begun.front().kind == Offer::Restart && begun.front().draw == 0 && begun.front().calls == 1);
  rows.add(due.front().options, 1800, MIN_CALLS, Status::Ok, PROBE, 202);
  CHECK_EQ(countOf(ask(search, rows, readings, noJumps), Offer::Restart), size_t(0));
  fill(RESTART_PERIOD - 2, 300);
  CHECK_EQ(countOf(ask(search, rows, readings, noJumps), Offer::Restart), size_t(0));
  fill(1, 300);
  std::vector<Candidate> const next = ask(search, rows, readings, noJumps);
  CHECK(!next.empty() && next.front().kind == Offer::Restart && next.front().draw == 1);
}

TEST(a_restart_draw_answers_only_itself_and_says_nothing_of_its_steps) {
  Rows rows;
  rows.add({}, 1700);
  Baseline const b = entry();
  Asking const ask{.strategy = {.kind = Strategy::Kind::Single}};
  std::vector<Reading> const readings{{.config = {}, .cost = 1700}};

  // A row that moves WMUL among much else, measured and dear: the step WMUL=1 taken elsewhere, so offered last.
  UseConfig const drawn = canonicalConfig(nvidia(), b.fft, {{"WMUL", "1"}, {"ZEROHACK_W", "0"}, {"TAIL_KERNELS", "3"}});
  rows.add(drawn, 1900);
  EntrySearch undeclared{b};
  std::vector<Candidate> const elsewhere = ask(undeclared, rows, readings, flat);
  CHECK(!elsewhere.empty() && configText(elsewhere.back().options) == "WMUL=1");

  // Declared as a draw, it says nothing of what WMUL=1 alone does, and the step keeps its place; the point it is stays
  // answered.
  CHECK(rows.db.add(JumpRow{.sess = rows.sess,
                            .fft = SPEC,
                            .kind = TestKind::PRP,
                            .regime = b.band.regime,
                            .cfg = rows.db.internCfg(drawn),
                            .k = 0,
                            .ts = 0}));
  EntrySearch declared{b};
  std::vector<Candidate> const offers = ask(declared, rows, readings, flat);
  CHECK(find(offers, "WMUL=1") && configText(offers.back().options) != "WMUL=1");
  CHECK(!find(offers, configText(drawn)));
}
