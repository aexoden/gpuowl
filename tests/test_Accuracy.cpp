// Copyright (C) Jason Lynch

// Tests the accuracy sweep as a pure function of its readings: where it reads, what it plans from what it has read,
// and what the readings say about each key.

#include "Accuracy.h"

#include "Eligibility.h"
#include "FFTVariants.h"
#include "Primes.h"
#include "TuneDB.h"
#include "Tuner.h"

#include "test.h"

#include <algorithm>
#include <map>
#include <set>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .computeCapability = 806}; }

// Readings keyed as the database matches them; `failing` names sets whose check fails, `missing` sets that never
// give a reading (a build failure), `z` sets that round otherwise than the defaults.
struct Held {
  std::set<std::string> failing;
  std::set<std::string> missing;
  std::map<std::string, double> z;
  bool all = true;
  std::set<std::string> taken;

  [[nodiscard]] static std::string id(const FFTConfig& fft, const UseConfig& config) {
    return fft.spec() + " " + configText(config);
  }

  [[nodiscard]] SweepLookup lookup() const {
    return [this](const FFTConfig& fft, u64, const UseConfig& config) -> std::optional<SweepReading> {
      std::string const text = configText(config);
      if ((!all && !taken.contains(id(fft, config))) || missing.contains(id(fft, config))) { return {}; }
      auto const at = z.find(text);
      return SweepReading{.z = at != z.end() ? at->second : 24.44,
                          .n = 2150,
                          .maxRoe = at != z.end() ? 0.33 : 0.3098,
                          .checkOk = !failing.contains(id(fft, config)),
                          .fingerprint = at != z.end() ? std::hash<std::string>{}(text) | 1 : 1};
    };
  }
};

[[nodiscard]] bool has(const std::vector<SweepPoint>& plan, const std::string& text) {
  return std::ranges::any_of(plan, [&](const SweepPoint& p) { return p.text == text; });
}

[[nodiscard]] const SweepPoint* find(const std::vector<SweepPoint>& plan, const std::string& text) {
  auto const at = std::ranges::find_if(plan, [&](const SweepPoint& p) { return p.text == text; });
  return at == plan.end() ? nullptr : &*at;
}

}  // namespace

TEST(the_sweep_reads_at_the_top_of_an_ffts_range) {
  static const Primes primes;
  FFTConfig const fft{"512:8:512:212"};
  u64 const top = sweepExponent(fft);
  CHECK(primes.isPrime(top));
  CHECK(isEligible(fft, top) && top <= maxExp(fft));
  CHECK(primes.nextPrime(top) > maxExp(fft));
}

TEST(nothing_is_moved_before_the_set_it_moves_from_has_been_read) {
  FFTConfig const family{"512:8:512:212"};
  Held held;
  held.all = false;
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  std::vector<u32> const variants = runnableVariants(nvidia(), family.shape);
  CHECK_EQ(plan.size(), variants.size());
  CHECK(std::ranges::none_of(plan, [](const SweepPoint& p) { return p.background.has_value(); }));
  CHECK_EQ(plan.front().fft.spec(), family.spec());
}

TEST(each_move_is_read_once_against_the_set_it_moved_from_at_the_variant_that_offers_it) {
  FFTConfig const family{"512:8:512:212"};
  Held const held;
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  std::set<std::string> texts;
  for (const SweepPoint& p : plan) {
    if (!p.background) { continue; }
    CHECK(texts.insert(p.text).second);
  }

  // One exponent for the family, which every variant can run: the lowest top among them.
  u64 lowest = 0;
  for (u32 const v : runnableVariants(nvidia(), family.shape)) {
    u64 const e = sweepExponent({family.shape, v, family.carry});
    if (!lowest || e < lowest) { lowest = e; }
  }
  CHECK(lowest < sweepExponent(family));
  CHECK(std::ranges::all_of(plan, [&](const SweepPoint& p) { return p.exponent == lowest; }));

  // At M=1 the chains offer only their defaults, so they are read at an M=0 variant.
  const SweepPoint* const mm2 = find(plan, "MM2_CHAIN=1");
  CHECK(mm2 != nullptr);
  CHECK(mm2 && mm2->fft.variant / 10 % 10 == 0);
  CHECK(mm2 && mm2->background && mm2->background->empty());
  CHECK(has(plan, "MM_CHAIN=1"));
  CHECK(has(plan, "MM2_CHAIN=2"));

  // A structural move is itself read, and what it opens is read against it.
  const SweepPoint* const inplace = find(plan, "INPLACE=0");
  CHECK(inplace != nullptr);
  const SweepPoint* const pad = find(plan, "PAD=256");
  CHECK(pad && pad->background && configText(*pad->background) == "INPLACE=0");
}

TEST(every_key_offered_more_than_one_value_is_read) {
  FFTConfig const family{"512:8:512:212"};
  Held const held;
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  std::set<std::string> moved;
  for (const SweepPoint& p : plan) { moved.insert(p.key); }

  for (u32 const v : runnableVariants(nvidia(), family.shape)) {
    FFTConfig const fft{family.shape, v, canonicalCarry(family.shape, v, family.carry)};
    for (const Option& o : allOptions()) {
      if (o.kind != Kind::Tunable || !o.appliesTo(nvidia(), fft, {}) || o.isInert(nvidia(), fft, {})) { continue; }
      if (o.compound || o.valuesFor(nvidia(), fft, {}).size() < 2) { continue; }
      if (!moved.contains(o.key)) { testing::fail(__FILE__, __LINE__, o.key + " is never moved at " + fft.spec()); }
    }
  }
}

TEST(a_set_that_fails_its_check_leaves_its_moves_to_the_next_variant) {
  FFTConfig const family{"512:8:512:212"};
  Held held;
  held.failing.insert(Held::id(family, {}));
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  const SweepPoint* const tail = find(plan, "TAIL_KERNELS=3");
  CHECK(tail != nullptr);
  CHECK(tail && tail->fft.spec() != family.spec());
  CHECK(std::ranges::none_of(plan, [&](const SweepPoint& p) { return p.background && p.fft.spec() == family.spec(); }));
}

TEST(only_the_groups_asked_for_are_moved) {
  FFTConfig const family{"512:8:512:212"};
  Held const held;
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, {Group::Middle}, held.lookup());
  for (const SweepPoint& p : plan) {
    if (p.background) { CHECK(findOption(p.key)->group == Group::Middle); }
  }
  CHECK(has(plan, "MM2_CHAIN=1"));
}

TEST(a_reading_is_compared_by_the_fingerprint_of_its_samples) {
  SweepReading const bg{.z = 24.44, .n = 2150, .maxRoe = 0.3098, .checkOk = true, .fingerprint = 0x1234};
  CHECK(compare(bg, bg) == Change::Same);

  // Statistics that agree to every digit the database keeps say nothing: other samples round otherwise.
  SweepReading other = bg;
  other.fingerprint = 0x5678;
  CHECK(compare(other, bg) == Change::Differs);
  SweepReading lower = other;
  lower.z = 22.34;
  CHECK(compare(lower, bg) == Change::Differs);
  SweepReading failing = other;
  failing.checkOk = false;
  CHECK(compare(failing, bg) == Change::Fails);

  // Nothing to compare with: a reading missing, one with no fingerprint, or a background that failed its check.
  CHECK(compare(std::nullopt, bg) == Change::Unread);
  CHECK(compare(bg, std::nullopt) == Change::Unread);
  SweepReading bare = bg;
  bare.fingerprint = 0;
  CHECK(compare(bare, bg) == Change::Unread);
  CHECK(compare(bg, bare) == Change::Unread);
  SweepReading failed = bg;
  failed.checkOk = false;
  CHECK(compare(bg, failed) == Change::Unread);
}

TEST(a_structural_move_that_fails_its_check_is_offered_again_with_what_it_opens) {
  FFTConfig const family{"512:8:512:212"};
  Held held;
  held.failing.insert(Held::id(family, {{"INPLACE", "0"}}));
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  // Read, and failed, at the family's own variant; read again at the next, where what it opens is read against it.
  auto const inplace = std::ranges::count_if(plan, [](const SweepPoint& p) { return p.text == "INPLACE=0"; });
  CHECK_EQ(inplace, 2);
  const SweepPoint* const pad = find(plan, "PAD=256");
  CHECK(pad && pad->fft.spec() != family.spec() && configText(*pad->background) == "INPLACE=0");
}

TEST(a_move_that_gave_no_reading_is_offered_again_by_the_next_variant) {
  FFTConfig const family{"512:8:512:212"};
  Held held;
  held.missing.insert(Held::id(family, {{"TAIL_KERNELS", "0"}}));
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());

  std::vector<std::string> at;
  for (const SweepPoint& p : plan) {
    if (p.text == "TAIL_KERNELS=0") { at.push_back(p.fft.spec()); }
  }
  CHECK_EQ(at.size(), size_t(2));
  CHECK(!at.empty() && at.front() == family.spec());

  // One that was read, whatever it read, is not.
  CHECK_EQ(std::ranges::count_if(plan, [](const SweepPoint& p) { return p.text == "TAIL_KERNELS=1"; }), 1);
}

TEST(a_key_changes_the_rounding_if_any_value_read_otherwise_than_its_background) {
  FFTConfig const family{"512:8:512:212"};
  Held held;
  // As the P100 read it at 5.6: MM2_CHAIN=1 costs 2.10, and the rest are the defaults' arithmetic.
  held.z["MM2_CHAIN=1"] = 22.34;
  std::vector<SweepPoint> const plan = sweepPlan(nvidia(), family, allGroups(), held.lookup());
  std::vector<KeyVerdict> const verdicts = keyVerdicts(plan, held.lookup());

  auto const of = [&](const std::string& key) {
    auto const at = std::ranges::find_if(verdicts, [&](const KeyVerdict& v) { return v.key == key; });
    CHECK(at != verdicts.end());
    return *at;
  };

  KeyVerdict const mm2 = of("MM2_CHAIN");
  CHECK(mm2.measured() == AccuracyImpact::Yes);
  CHECK_EQ(mm2.differs, 1u);
  CHECK(std::abs(mm2.worstDz + 2.10) < 1e-9);
  CHECK(mm2.worstAt.ends_with("MM2_CHAIN=1"));
  CHECK(mm2.changedBy.at("MM2_CHAIN=1"));
  CHECK(!mm2.changedBy.at("MM2_CHAIN=2"));

  CHECK(of("MM_CHAIN").measured() == AccuracyImpact::None);
  CHECK(of("TAIL_KERNELS").measured() == AccuracyImpact::None);
  CHECK(std::ranges::is_sorted(verdicts, {}, &KeyVerdict::key));
}

TEST(a_key_nothing_was_read_for_has_no_verdict_and_a_failed_check_is_one) {
  KeyVerdict v;
  v.unread = 3;
  CHECK(!v.measured());
  v.same = 3;
  CHECK(v.measured() == AccuracyImpact::None);
  v.fails = 1;
  CHECK(v.measured() == AccuracyImpact::Yes);
}

TEST(the_accuracy_command_is_parsed) {
  std::optional<TuneCommand> const c = parseTuneCommand("accuracy,fft=512:8:512,groups=Middle+Tail,probe=100M");
  CHECK(c && c->verb == TuneVerb::Accuracy);
  CHECK(c && c->fft == "512:8:512");
  CHECK(c && c->groups == (std::vector<Group>{Group::Middle, Group::Tail}));
  CHECK(opensDevice(TuneVerb::Accuracy));
  CHECK(!opensDevice(TuneVerb::Emit));

  std::optional<TuneCommand> const bare = parseTuneCommand("accuracy");
  CHECK(bare && bare->verb == TuneVerb::Accuracy && bare->groups.empty() && bare->fft.empty());

  auto refused = [](std::string_view text) {
    try {
      (void)parseTuneCommand(text);
    } catch (const std::string&) { return true; }
    return false;
  };
  CHECK(refused("accuracy,groups=Middel"));
  CHECK(refused("accuracy,kinds=prp"));
  CHECK(refused("accuracy,env=1"));
  CHECK(refused("accuracy,fft="));
}
