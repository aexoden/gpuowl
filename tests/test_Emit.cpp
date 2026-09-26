// Copyright (C) Jason Lynch

// Tests that a hand-written database emits the selection file it should: that every published entry is a transcript of
// one row, that a costlier configuration reaching further is kept beside a cheaper one that does not reach as far,
// that one no exponent would choose is dropped, and that rows of another env, of a thinly measured call, or of a
// configuration that ever failed are not published at all.

#include "Emit.h"

#include "Args.h"
#include "Bootstrap.h"
#include "Production.h"

#include "test.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>

using namespace tune;

namespace {

// Ten rows on env 1 and one on env 2, over three FFTs: one FP64 shape measured under three option sets and at two
// exponents of one regime, a second FP64 shape in another regime, and a pure NTT shape.  Each set that is published
// has an accuracy reading at the top of its interval, but the NTT, which rounds nothing.  PAD does nothing on NVIDIA,
// so the gate reads the PAD=256 and PAD=128 sets as the one configuration they are, and either one's reading at or
// above the other's gate exponent counts for both.
const char* const DB =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
  "cfg   1 -\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "cfg   18 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
  "cfg   19 INPLACE=1,PAD=512,TAIL_KERNELS=3\n"
  "cfg   21 INPLACE=1,PAD=256\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "sess  9 env=2 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
  "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 prp 100000000 short32 18 1750.000 3.000 16 4 1.0000 ok 1753471284\n"
  "run   4 512:15:512:212 prp 120000000 short32 18 1760.000 3.000 16 4 1.0000 ok 1753471294\n"
  "run   4 512:15:512:212 prp 100000000 short32 19 1800.000 2.000 16 4 1.0000 ok 1753471304\n"
  "run   4 512:15:512:212 prp 100000000 short32 1 1700.000 2.000 4 1 1.0000 ok 1753471314\n"
  "run   4 512:15:512:212 ll 100000000 short32 17 1800.000 3.000 16 4 1.0000 ok 1753471324\n"
  "run   4 512:15:512:212 ll 100000000 short32 17 1805.000 3.000 16 4 1.0000 err 1753471334\n"
  "run   4 512:15:512:212 ll 100000000 short32 18 1850.000 3.000 16 4 1.0000 ok 1753471344\n"
  "run   4 1K:8:1K:202 prp 200000000 short32 21 3100.000 4.000 16 4 1.0000 ok 1753471354\n"
  "run   4 3:1K:8:512:202 prp 100000000 short32 21 2000.000 5.000 16 4 1.0000 ok 1753471364\n"
  "run   9 512:15:512:212 prp 100000000 short32 17 900.000 1.000 16 4 1.0000 ok 1753471374\n"
  "roe   4 512:15:512:212 143413741 18 24.40 2150 0.3098 ok - 1753471410\n"
  "roe   4 512:15:512:212 143498461 17 24.40 2150 0.3098 ok - 1753471420\n"
  "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 ok - 1753471430\n"
  "roe   9 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471440\n";

// The accuracy reading of the set published as 2e51eaf52a48bfc9.
const char* const PAD128_ROE = "roe   4 512:15:512:212 143413741 18 24.40 2150 0.3098 ok - 1753471410\n";

// What env 1 supports.  Costs are the mean plus two standard errors, so the six calls behind PAD=256 buy it a smaller
// penalty (+1.84) than the four behind PAD=128 (+3.35); PAD=256 at 1776.069 and PAD=512 at 1802.236 are dropped as
// nothing cheaper than PAD=128 anywhere they run; no ll entry is published, because one of the two ll rows under
// PAD=256 failed and that configuration -- which PAD=128, PAD doing nothing here, builds too -- is excluded; and the
// 1700.000 of one call is dropped as a reading nothing concluded.
const char* const SELECTION = "# prpll selection v1\n"
                              "# %PROVENANCE%\n"
                              "use   INPLACE=1,PAD=256\n"
                              "use ! 1 TAIL_KERNELS=3\n"
                              "entry 2e51eaf52a48bfc9 1753.354 512:15:512:212 prp 78643196 143413744 unvalidated\n"
                              "opts  2e51eaf52a48bfc9 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
                              "entry 688735b9ab069fec 2005.590 3:1K:8:512:202 prp 83886076 152674512 n/a\n"
                              "opts  688735b9ab069fec INPLACE=1,PAD=256\n"
                              "entry d5c85edcdda38629 3104.472 1K:8:1K:202 prp 167772152 296960416 unvalidated\n"
                              "opts  d5c85edcdda38629 INPLACE=1,PAD=256\n"
                              "exclude 512:15:512:212 ll short32 INPLACE=1,PAD=256,TAIL_KERNELS=3\n";

TuneDB loaded(const std::string& text) {
  TuneDB db;
  CHECK(db.parse(text, "fixture"));
  return db;
}

Defaults defaults() {
  return {.global = {{"INPLACE", "1"}, {"PAD", "256"}}, .family = {parseUseLine("! 1 TAIL_KERNELS=3")}};
}

Provenance provenance(u32 env = 1) {
  return {.ts = 1'753'471'500,
          .db = "tunedb.txt",
          .env = env,
          .T = 1801.4,
          .workloadLo = 100'000'000,
          .workloadHi = 400'000'000};
}

std::string expected(const Provenance& from) {
  std::string out = SELECTION;
  return out.replace(out.find("%PROVENANCE%"), std::string{"%PROVENANCE%"}.size(), provenanceOf(from));
}

std::string emitted(const TuneDB& db, const Provenance& from) {
  auto const file = emit(db, defaults(), from);
  CHECK(file.has_value());
  return file ? text(*file) : std::string{};
}

// The database with one record replaced, so a test says what it changes and nothing else.
std::string withRecord(const std::string& original, const std::string& replacement) {
  std::string text = DB;
  auto const at = text.find(original);
  CHECK(at != std::string::npos);
  return text.replace(at, original.size(), replacement);
}

bool publishes(const TuneDB& db, const std::string& fft, TestKind kind) {
  for (const SelectionEntry& e : entriesFor(db, 1)) {
    if (e.fft == fft && e.kind == kind) { return true; }
  }

  return false;
}

bool publishes(const TuneDB& db, const std::string& id) {
  for (const SelectionEntry& e : entriesFor(db, 1)) {
    if (e.id == id) { return true; }
  }

  return false;
}

}  // namespace

TEST(emit_writes_the_entries_its_rows_support) { CHECK_EQ(emitted(loaded(DB), provenance()), expected(provenance())); }

TEST(emit_publishes_what_production_can_read_back) {
  std::string const out = emitted(loaded(DB), provenance());

  auto const reread = parseSelection(out, "emitted");
  CHECK(reread.has_value());
  CHECK_EQ(text(reread.value_or(SelectionFile{})), out);
}

// The fold that pools duplicate rows is a view over the file, and compact() is what makes it the file's own content.
// Emission reads the view, so the two must not be able to disagree.
TEST(emit_is_unmoved_by_compacting_the_database) {
  TuneDB db = loaded(DB);
  std::string const before = emitted(db, provenance());

  CHECK(db.compact());
  CHECK_EQ(emitted(db, provenance()), before);
}

TEST(emit_keeps_each_env_to_its_own_rows) {
  TuneDB const db = loaded(DB);

  // The other card measured one configuration, and cheaply.  It is the only entry its own env publishes, and it is in
  // no other env's file.
  auto const other = entriesFor(db, 2);
  CHECK_EQ(other.size(), size_t{1});
  CHECK_EQ(other.at(0).opts.at("PAD"), std::string{"256"});
  CHECK(other.at(0).cost < 1000);

  for (const SelectionEntry& e : entriesFor(db, 1)) { CHECK(e.cost > 1000); }
}

// A row that names a regime its own exponent does not run in disagrees with this build about what the kernels do, and
// which of the two is right is not something emission can decide.
TEST(emit_refuses_a_row_that_contradicts_its_own_regime) {
  TuneDB const db = loaded(withRecord("3:1K:8:512:202 prp 100000000 short32", "3:1K:8:512:202 prp 100000000 long32"));

  CHECK(!publishes(db, "3:1K:8:512:202", TestKind::PRP));
  CHECK(publishes(db, "512:15:512:212", TestKind::PRP));
}

// An env with nothing measured publishes an empty table rather than nothing at all: production reads the file, finds
// no entry it can use, and falls back -- which is what a database that has been reset should make it do.
TEST(emit_publishes_an_empty_table_from_an_empty_database) {
  TuneDB const db = loaded("# prpll tunedb v1\n");

  auto const file = emit(db, defaults(), provenance());
  CHECK(file.has_value());
  CHECK(file && file->entries.empty());
  CHECK_EQ(emitted(db, provenance()),
           "# prpll selection v1\n# " + provenanceOf(provenance()) +
             "\nuse   INPLACE=1,PAD=256\nuse ! 1 TAIL_KERNELS=3\n");
}

TEST(emit_names_where_it_came_from) {
  CHECK(provenanceOf(provenance()).ends_with(" from tunedb.txt env 1; T=1801.4 over workload 100000000-400000000"));

  // A database-only emission has no workload to state, and states nothing rather than a placeholder.
  CHECK(provenanceOf({.ts = 1, .db = "db.txt", .env = 3}).ends_with(" from db.txt env 3"));
}

// The entry the fixture's second option set is published as, which is the one measured at two exponents.
const char* const TWICE_MEASURED = "2e51eaf52a48bfc9";

// A wrong residue is evidence about the kernels, and one regime is one set of kernels, so a configuration that failed
// at any exponent of a regime is published for none of it -- least of all for the exponent that caught it, which the
// interval it would otherwise carry contains.
TEST(emit_withdraws_a_configuration_that_answered_wrongly_in_the_regime) {
  CHECK(publishes(loaded(DB), TWICE_MEASURED));

  TuneDB const db = loaded(withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 err 1753471294"));
  CHECK(!publishes(db, TWICE_MEASURED));

  // And only that configuration: the same options in the other kind are untouched, where nothing failed there.
  std::string text = withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 err 1753471294");
  std::string const llErr = "1.0000 err 1753471334";
  text.replace(text.find(llErr), llErr.size(), "1.0000 ok 1753471334");
  CHECK(publishes(loaded(text), "512:15:512:212", TestKind::LL));
}

// PAD does nothing on NVIDIA, so PAD=128 builds the kernels that answered wrongly under PAD=256, and goes with them --
// as production, which matches an exclusion by the keys the tuner searches, would pass it over.
TEST(emit_withdraws_every_spelling_of_a_configuration_that_answered_wrongly) {
  CHECK(!publishes(loaded(DB), "512:15:512:212", TestKind::LL));
  CHECK(publishes(loaded(withRecord("1.0000 err 1753471334", "1.0000 ok 1753471334")), "512:15:512:212", TestKind::LL));
}

// The failure that follows a wrong answer is usually a refusal -- the next attempt's build, or a backend that no longer
// runs it -- and that says nothing the wrong answer did not, so the rows it merges into still withdraw the configuration
// though another exponent of the regime measured it cleanly.
TEST(emit_withdraws_a_wrong_answer_whatever_failure_follows_it) {
  for (const char* later : {"unsupported", "nocompile", "lost"}) {
    TuneDB db = loaded(withRecord("16 4 1.0000 ok 1753471294",
                                  "16 4 1.0000 err 1753471294\n"
                                  "run   4 512:15:512:212 prp 120000000 short32 18 0.000 0.000 0 0 1.0000 " +
                                    std::string{later} + " 1753471299"));
    CHECK(!publishes(db, TWICE_MEASURED));

    CHECK(db.compact());
    CHECK(!publishes(db, TWICE_MEASURED));
  }
}

// A build that would not compile and a run the backend refused say nothing about the answers the configuration
// computes, so neither reaches past the exponent it happened at.
TEST(emit_keeps_a_configuration_a_refusal_never_ran) {
  CHECK(
    publishes(loaded(withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 unsupported 1753471294")), TWICE_MEASURED));
  CHECK(publishes(loaded(withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 nocompile 1753471294")), TWICE_MEASURED));
}

// The lines move as entries are tuned, and an entry outranks them, so each entry names every key they would set to
// something its row did not run: production resolves exactly what was measured, whatever the lines have come to say.
TEST(emit_names_every_key_its_own_default_lines_would_change) {
  TuneDB const db =
    loaded(withRecord("run   4 1K:8:1K:202 prp 200000000 short32 21 ", "run   4 1K:8:1K:202 prp 200000000 short32 1 "));
  FFTConfig const fft{"1K:8:1K:202"};
  Env const nvidia = db.findEnv(1)->toEnv();

  for (const Defaults& lines : {Defaults{.global = {{"TAIL_KERNELS", "3"}}, .family = {}},
                                Defaults{.global = {}, .family = {parseUseLine("! 0 WMUL=1")}}, defaults()}) {
    auto const file = emit(db, lines, provenance());
    CHECK(file.has_value());
    if (!file) { continue; }

    auto const e =
      std::ranges::find_if(file->entries, [](const SelectionEntry& entry) { return entry.fft == "1K:8:1K:202"; });
    CHECK(e != file->entries.end());
    if (e == file->entries.end()) { continue; }

    // The row ran the built-in defaults, and that is what production builds for the entry.
    UseConfig const resolved =
      resolveConfig(Args{true}, fft, TestKind::PRP, fittedTo(file->layersFor(*e), nvidia, fft, TestKind::PRP));
    CHECK(canonicalConfig(nvidia, fft, resolved).empty());
    CHECK(shadowedKeys(Args{true}, nvidia, *file, *e, fft).empty());
  }
}

// One build written two ways is one configuration: its better supported row is published, and once, since the two
// spellings name the same keys once the lines' are named beside them.  PAD does nothing on NVIDIA.
TEST(emit_publishes_a_build_written_two_ways_once) {
  TuneDB const db = loaded(withRecord("19 1800.000 2.000 16 4 1.0000 ok", "19 1750.000 3.000 16 4 1.0000 ok"));

  auto const file = emit(db, defaults(), provenance());
  CHECK(file.has_value());
  if (!file) { return; }
  CHECK_EQ(
    std::ranges::count_if(file->entries,
                          [](const SelectionEntry& e) { return e.fft == "512:15:512:212" && e.kind == TestKind::PRP; }),
    ptrdiff_t{1});
}

namespace {

const char* const ROE_1K = "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 ok - 1753471430\n";

// The 1K:8:1K entry, and a second, dearer set of the same identity that the gate has passed.
std::string withSecondSet(const std::string& roe1K) {
  std::string text = DB;
  text.replace(text.find(ROE_1K), std::string{ROE_1K}.size(), roe1K);
  return text +
    "cfg   22 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
    "run   4 1K:8:1K:202 prp 200000000 short32 22 3200.000 4.000 16 4 1.0000 ok 1753471450\n"
    "roe   4 1K:8:1K:202 296960407 22 24.90 2150 0.3040 ok - 1753471460\n";
}

std::vector<std::string> published1K(const TuneDB& db, Gating gating = Gating::Required) {
  std::vector<std::string> out;
  for (const SelectionEntry& e : entriesFor(db, 1, gating)) {
    if (e.fft == "1K:8:1K:202") { out.push_back(configText(e.opts)); }
  }
  return out;
}

Env nvidia() {
  Env env;
  env.isNvidia = true;
  env.computeCapability = 806;
  return env;
}

}  // namespace

namespace {

// Takes every reading the gate owes env 1, as the queue would, each reading `zAt` gives at its exponent, until nothing
// is owed; the number taken.
u32 answerOwed(TuneDB& db, const std::function<RoeRow(const OptionSet&, u64)>& zAt) {
  u32 taken = 0;
  for (std::vector<OptionSet> owed = gatesOwed(db, 1); !owed.empty() && taken < 100; owed = gatesOwed(db, 1)) {
    const OptionSet& s = owed.front();
    FFTConfig const fft{s.entry.fft};
    RoeRow row = zAt(s, s.gate.owedAt);
    row.sess = 4;
    row.fft = s.entry.fft;
    row.exponent = s.gate.owedAt;
    row.cfg = db.internCfg(s.gate.owesReference ? accuracyReference(nvidia(), fft, s.entry.opts) : s.entry.opts);
    row.ts = 1'753'472'000 + ++taken;
    CHECK(db.add(row));
  }
  CHECK(taken < 100);
  return taken;
}

RoeRow roe(double z, bool checkOk = true) {
  return {.sess = 0, .fft = {}, .exponent = 0, .cfg = 0, .z = z, .n = 2150, .maxRoe = 0.3, .checkOk = checkOk, .ts = 0};
}

// z falling by one per 0.015 bits per word above `at`, where it reads `z`: -ztune's slope at 4M words.
std::function<RoeRow(const OptionSet&, u64)> sloped(double z, double at) {
  return [=](const OptionSet& s, u64 E) { return roe(z + (at - E / double(FFTConfig{s.entry.fft}.size())) / 0.015); };
}

}  // namespace

TEST(a_configuration_the_gate_rejects_never_reaches_the_selection_file) {
  std::vector<std::string> const cheapest{"INPLACE=1,PAD=256"};
  std::vector<std::string> const dearer{"INPLACE=1,PAD=256,TAIL_KERNELS=3"};
  CHECK(published1K(loaded(withSecondSet(ROE_1K))) == cheapest);

  // Read below the FP64 floor at the top of its interval, and with a failed Gerbicz check: either way a reach is
  // derived for the cheaper set, and where no exponent it is read at does any better, it is gone -- from the file and
  // from what the search works from -- and the set it kept out takes its place.
  for (bool const checkOk : {true, false}) {
    TuneDB db = loaded(withSecondSet(checkOk ? "roe   4 1K:8:1K:202 296960407 21 17.20 2150 0.4011 ok - 1753471430\n"
                                             : "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 fail - 1753471430\n"));
    CHECK(published1K(db) == dearer);
    CHECK_EQ(gatesOwed(db, 1).size(), size_t{1});

    CHECK(answerOwed(db, [&](const OptionSet&, u64) { return roe(checkOk ? 17.2 : 25.1, checkOk); }) > 1);
    CHECK(published1K(db) == dearer);
    CHECK(published1K(db, Gating::Assumed) == dearer);
    CHECK(emitted(db, provenance()).find("3104.472") == std::string::npos);
    for (const OptionSet& s : optionSetsFor(db, 1)) {
      CHECK(s.entry.fft != "1K:8:1K:202" || s.entry.opts.contains("TAIL_KERNELS"));
    }
  }
}

// What a derived reach buys: a set that cannot reach the top of the table is not discarded but published below where it
// stops, cheaper there than what reaches further, which is kept above it.
TEST(a_set_short_of_the_floor_is_published_up_to_the_reach_derived_for_it) {
  TuneDB db = loaded(withSecondSet("roe   4 1K:8:1K:202 296960407 21 17.20 2150 0.4011 ok - 1753471430\n"));
  FFTConfig const fft{"1K:8:1K:202"};
  double const top = 296'960'407 / double(fft.size());

  u32 const taken = answerOwed(db, sloped(17.2, top));
  CHECK(taken >= 2 && taken <= 4);

  std::vector<SelectionEntry> table;
  for (const SelectionEntry& e : entriesFor(db, 1)) {
    if (e.fft == fft.spec()) { table.push_back(e); }
  }
  CHECK_EQ(table.size(), size_t{2});
  if (table.size() != 2) { return; }

  // Cheapest first: the set derived a reach where the model reads 28, within a guard band, and confirmed by a reading
  // taken there; the dearer set keeps the table's reach.
  CHECK_EQ(configText(table[0].opts), std::string{"INPLACE=1,PAD=256"});
  CHECK(table[0].evidence == Evidence::Confirmed);
  double const reachBpw = table[0].reach / double(fft.size());
  CHECK(reachBpw <= top - (28 - 17.2) * 0.015 + 1e-9);
  CHECK(reachBpw > top - (28 - 17.2) * 0.015 - REACH_GUARD_BPW);
  CHECK(std::ranges::any_of(db.roes(), [&](const RoeRow& r) { return r.exponent == table[0].reach && r.z >= 28; }));

  CHECK_EQ(configText(table[1].opts), std::string{"INPLACE=1,PAD=256,TAIL_KERNELS=3"});
  CHECK_EQ(table[1].reach, u64(296'960'416));
  CHECK(table[1].evidence == Evidence::Unvalidated);
  CHECK_EQ(table[0].emin, table[1].emin);
}

TEST(a_set_the_gate_has_not_read_waits_for_its_reading) {
  TuneDB const db = loaded(withSecondSet(""));

  // Not published, and nothing dearer of its identity stands in for it where the search expects it to pass.
  CHECK(published1K(db) == std::vector<std::string>{"INPLACE=1,PAD=256,TAIL_KERNELS=3"});
  CHECK(published1K(db, Gating::Assumed) == std::vector<std::string>{"INPLACE=1,PAD=256"});

  std::vector<OptionSet> const owed = gatesOwed(db, 1);
  CHECK_EQ(owed.size(), size_t{1});
  CHECK_EQ(configText(owed.at(0).entry.opts), std::string{"INPLACE=1,PAD=256"});
  CHECK_EQ(owed.at(0).gate.owedAt, u64(296'960'407));
  CHECK(!owed.at(0).gate.owesReference);

  // Read where it would not count -- below the top of the interval -- it is still owed.
  TuneDB const low = loaded(withSecondSet("roe   4 1K:8:1K:202 250000013 21 25.10 2150 0.3021 ok - 1753471430\n"));
  CHECK_EQ(gatesOwed(low, 1).size(), size_t{1});
}

TEST(a_reading_at_the_fitted_standard_publishes_its_reach_as_confirmed) {
  TuneDB const db = loaded(withRecord(ROE_1K, "roe   4 1K:8:1K:202 296960407 21 28.30 2150 0.2711 ok - 1753471430\n"));
  bool seen = false;
  for (const SelectionEntry& e : entriesFor(db, 1)) {
    if (e.fft != "1K:8:1K:202") { continue; }
    CHECK(e.evidence == Evidence::Confirmed);
    seen = true;
  }
  CHECK(seen);
}

TEST(a_set_that_spends_accuracy_is_held_to_its_defaults_reading) {
  Env const env = nvidia();
  TuneDB db;
  u32 const id = db.internEnv(dbEnvOf(env));
  u32 const sess = db.beginSession(id, "512:15:512:212@118063003", 0, 1'753'471'200);

  FFTConfig const fft{"2:512:8:512:202"};
  u64 const at = 118'063'003;
  UseConfig const moved{{"TAIL_TRIGS32", "0"}};
  CHECK(db.add(RunRow{
    .sess = sess,
    .fft = fft.spec(),
    .kind = TestKind::PRP,
    .exponent = at,
    .regime = regimeOf(fft, at),
    .cfg = db.internCfg(moved),
    .m = {
      .mean = 1400, .stddev = 1, .blocks = 16, .calls = 4, .drift = 1, .status = Status::Ok, .ts = 1'753'471'300}}));

  u64 const top = gateExponent(interval(fft, at));
  u64 ts = 1'753'471'300;
  auto read = [&](const UseConfig& opts, double z) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = fft.spec(),
                        .exponent = top,
                        .cfg = db.internCfg(opts),
                        .z = z,
                        .n = 2150,
                        .maxRoe = 0.4,
                        .checkOk = true,
                        .ts = ++ts}));
  };

  // Its own reading clears the floor, and then the reading of its defaults is owed.
  read(moved, 12);
  std::vector<OptionSet> const owed = gatesOwed(db, id);
  CHECK_EQ(owed.size(), size_t{1});
  CHECK(owed.at(0).gate.owesReference);
  CHECK_EQ(owed.at(0).gate.owedAt, top);
  CHECK(entriesFor(db, id).empty());

  // Within ACCURACY_SLACK_Z of it, it is published; read again better, the defaults show what the set spent, and its
  // reach is derived lower down, where it reads what they read at the top.
  read(accuracyReference(env, fft, moved), 12.4);
  CHECK_EQ(entriesFor(db, id).size(), size_t{1});
  read({}, 13);
  CHECK(entriesFor(db, id).empty());
  std::vector<OptionSet> const deriving = gatesOwed(db, id);
  CHECK_EQ(deriving.size(), size_t{1});
  if (deriving.empty()) { return; }
  CHECK(!deriving.at(0).gate.owesReference);
  CHECK(deriving.at(0).gate.owedAt < top);
}

// A set that spends accuracy and cannot be confirmed anywhere is not published at all; what is, is the same set with
// those keys at their defaults, which the fitted table's reach describes.
TEST(a_rejected_set_falls_back_to_its_default_accuracy) {
  Env const env = nvidia();
  TuneDB db;
  u32 const id = db.internEnv(dbEnvOf(env));
  u32 const sess = db.beginSession(id, "512:15:512:212@118063003", 0, 1'753'471'200);

  FFTConfig const fft{"2:512:8:512:202"};
  u64 const at = 118'063'003;
  UseConfig const moved{{"TAIL_TRIGS32", "0"}};
  UseConfig const reference = accuracyReference(env, fft, moved);
  u64 ts = 1'753'471'300;
  auto time = [&](const UseConfig& opts, double mean) {
    CHECK(db.add(RunRow{
      .sess = sess,
      .fft = fft.spec(),
      .kind = TestKind::PRP,
      .exponent = at,
      .regime = regimeOf(fft, at),
      .cfg = db.internCfg(opts),
      .m = {.mean = mean, .stddev = 1, .blocks = 16, .calls = 4, .drift = 1, .status = Status::Ok, .ts = ++ts}}));
  };
  time(moved, 1400);
  time(reference, 1450);

  Interval const span = interval(fft, at);
  u64 const top = gateExponent(span);
  auto read = [&](const UseConfig& opts, u64 E, double z, bool checkOk) {
    CHECK(db.add(RoeRow{.sess = sess,
                        .fft = fft.spec(),
                        .exponent = E,
                        .cfg = db.internCfg(opts),
                        .z = z,
                        .n = 2150,
                        .maxRoe = 0.4,
                        .checkOk = checkOk,
                        .ts = ++ts}));
  };
  read(reference, top, 25.7, true);

  // Its Gerbicz check fails wherever it is read, so no reach can be confirmed for it.
  u32 taken = 0;
  for (std::vector<OptionSet> owed = gatesOwed(db, id); !owed.empty() && taken < 20; owed = gatesOwed(db, id)) {
    const OptionSet& s = owed.front();
    CHECK(canonicalConfig(env, fft, s.entry.opts) == canonicalConfig(env, fft, moved));
    CHECK(!s.gate.owesReference);
    read(moved, s.gate.owedAt, 6.1, false);
    ++taken;
  }
  CHECK(taken > 1 && taken < 20);

  std::vector<SelectionEntry> const table = entriesFor(db, id);
  CHECK_EQ(table.size(), size_t{1});
  if (table.empty()) { return; }
  CHECK(!movesAccuracy(env, fft, table[0].opts));
  CHECK(canonicalConfig(env, fft, table[0].opts) == canonicalConfig(env, fft, reference));
  CHECK_EQ(table[0].reach, span.hi);
  CHECK(std::ranges::none_of(optionSetsFor(db, id),
                             [&](const OptionSet& s) { return movesAccuracy(env, fft, s.entry.opts); }));
}

// A set whose reach was derived short of the table stays in the file once a cheaper set covers everything it does:
// nothing would choose it, but it is where production learns that its arithmetic stops there, and a user's setting can
// turn the cheaper set into exactly that arithmetic.
TEST(a_reduced_reach_is_published_though_a_cheaper_set_covers_it) {
  TuneDB db = loaded(withSecondSet("roe   4 1K:8:1K:202 296960407 21 17.20 2150 0.4011 ok - 1753471430\n"));
  FFTConfig const fft{"1K:8:1K:202"};
  u64 const top = 296'960'407;
  u64 const bandEnd = interval(fft, top).hi;
  CHECK(answerOwed(db, sloped(17.2, top / double(fft.size()))) >= 2);

  // Then the search finds a cheaper set off the defaults' rounding that reads well enough at the top to hold the
  // table's reach there.
  UseConfig const cheaper{{"INPLACE", "1"}, {"PAD", "256"}, {"TAIL_KERNELS", "1"}};
  u32 const cfg = db.internCfg(cheaper);
  CHECK(db.add(RunRow{
    .sess = 4,
    .fft = fft.spec(),
    .kind = TestKind::PRP,
    .exponent = 200'000'000,
    .regime = regimeOf(fft, 200'000'000),
    .cfg = cfg,
    .m = {
      .mean = 2900, .stddev = 4, .blocks = 16, .calls = 4, .drift = 1, .status = Status::Ok, .ts = 1'753'473'000}}));
  CHECK(db.add(RoeRow{.sess = 4,
                      .fft = fft.spec(),
                      .exponent = top,
                      .cfg = cfg,
                      .z = 28.5,
                      .n = 2150,
                      .maxRoe = 0.27,
                      .checkOk = true,
                      .ts = 1'753'473'010}));
  CHECK(gatesOwed(db, 1).empty());

  // The table the objective prices has only the cheaper set; the file keeps the derived one beside it.
  CHECK(published1K(db) == std::vector<std::string>{configText(cheaper)});

  auto const file = emit(db, defaults(), provenance());
  CHECK(file.has_value());
  if (!file) { return; }

  std::vector<SelectionEntry> both;
  for (const SelectionEntry& e : file->entries) {
    if (e.fft == fft.spec()) { both.push_back(e); }
  }
  CHECK_EQ(both.size(), size_t{2});
  if (both.size() != 2) { return; }
  CHECK_EQ(configText(both[0].opts), configText(cheaper));
  CHECK_EQ(both[0].reach, bandEnd);
  CHECK_EQ(configText(both[1].opts), std::string{"INPLACE=1,PAD=256"});
  u64 const reach = both[1].reach;
  CHECK(reach < bandEnd);

  // A config.txt that puts TAIL_KERNELS back to its default runs the cheaper entry at the defaults' arithmetic, which
  // is held to the reach derived for it, above which nothing in the file covers the exponent.
  Args args{true};
  args.parse("-use TAIL_KERNELS=2", true);
  std::optional<Choice> const below = chooseFrom(*file, args, nvidia(), reach, TestKind::PRP);
  CHECK(below.has_value());
  if (below) {
    CHECK_EQ(configText(below->entry->opts), configText(cheaper));
    CHECK_EQ(below->reach, reach);
  }
  CHECK(!chooseFrom(*file, args, nvidia(), reach + 2, TestKind::PRP));

  // Left alone, the cheaper entry serves its whole band.
  std::optional<Choice> const alone = chooseFrom(*file, Args{true}, nvidia(), bandEnd, TestKind::PRP);
  CHECK(alone.has_value());
  if (alone) { CHECK_EQ(alone->reach, bandEnd); }
}

namespace {

// The top regime's interval of `fft`, the one the table's reach ends in.
Interval topOf(const FFTConfig& fft) { return intervals(fft, minExp(fft), maxExp(fft)).back(); }

SelectionEntry entryOf(const std::string& spec, double cost, const UseConfig& opts, std::optional<u64> reach = {},
                       Evidence evidence = Evidence::Unvalidated) {
  FFTConfig const fft{spec};
  Interval const top = topOf(fft);
  return {.id = {},
          .cost = cost,
          .fft = fft.spec(),
          .kind = TestKind::PRP,
          .emin = top.lo,
          .reach = reach.value_or(top.hi),
          .regime = {},
          .evidence = evidence,
          .opts = opts};
}

SelectionFile fileOf(std::vector<SelectionEntry> entries, std::vector<Exclusion> excluded = {}) {
  SelectionFile file{.provenance = "written 1753471500 by test from tunedb.txt env 1",
                     .global = {},
                     .family = {},
                     .entries = std::move(entries),
                     .excluded = std::move(excluded),
                     .unknown = {}};
  CHECK(finalize(file));
  return file;
}

std::vector<std::pair<std::string, double>> linesOf(const std::vector<TuneEntry>& view) {
  std::vector<std::pair<std::string, double>> out;
  for (const TuneEntry& line : view) { out.emplace_back(line.fft.spec(), line.cost); }
  return out;
}

}  // namespace

TEST(tune_txt_offers_only_what_the_table_reach_holds_for_at_default_rounding) {
  std::string const fp64 = "512:15:512:212";
  std::string const shortOfTable = "1K:8:1K:202";
  std::string const ntt = "3:1K:8:512:202";
  std::string const hybrid = "2:512:8:512:202";
  std::string const small = "256:2:256:202";

  // An FFT whose table reach ends in one regime and that was only measured in another.
  FFTConfig const lowered{"1K:12:1K:202"};
  std::vector<Interval> const bands = intervals(lowered, minExp(lowered), maxExp(lowered));
  CHECK(bands.size() >= 2);
  SelectionEntry lowerBand = entryOf(lowered.spec(), 1500, {});
  lowerBand.emin = bands.front().lo;
  lowerBand.reach = bands.front().hi;

  // One measured to the table's reach at the top, but held short of the end of a lower band: an older binary would run
  // it through the whole of that band.
  std::string const restrictedLow = "1K:16:1K:202";
  FFTConfig const low{restrictedLow};
  std::vector<Interval> const lowBands = intervals(low, minExp(low), maxExp(low));
  CHECK(lowBands.size() >= 2);
  SelectionEntry heldLow = entryOf(restrictedLow, 1400, {});
  heldLow.emin = lowBands.front().lo;
  heldLow.reach = lowBands.front().lo + (lowBands.front().hi - lowBands.front().lo) / 2;
  heldLow.evidence = Evidence::Confirmed;

  SelectionFile const file = fileOf({
    // Two sets that round alike, at the table's reach: the cheaper one's cost.
    entryOf(fp64, 1750, {}),
    entryOf(fp64, 1740, {{"TAIL_KERNELS", "3"}}),
    // The defaults' arithmetic derived short of the table, so it is not offered, whatever else reaches it.
    entryOf(shortOfTable, 3000, {{"TAIL_KERNELS", "1"}}),
    entryOf(shortOfTable, 3100, {}, topOf(FFTConfig{shortOfTable}).hi - 1'000'000, Evidence::Confirmed),
    // Exact arithmetic, whatever it reaches.
    entryOf(ntt, 2000, {}, {}, Evidence::NotApplicable),
    // An older binary would not run the arithmetic that held the table's reach.
    entryOf(hybrid, 1600, {{"TAIL_TRIGS32", "0"}}),
    // Dearer than a line that reaches further, which upstream's reader would drop.
    entryOf(small, 1800, {}),
    lowerBand,
    entryOf(restrictedLow, 1450, {}),
    heldLow,
  });

  std::vector<TuneEntry> const view = compatibilityView(file, nvidia());
  CHECK(linesOf(view) == (std::vector<std::pair<std::string, double>>{{fp64, 1740}, {ntt, 2000}}));

  // Upstream's own reader keeps every line, and reads each at the table's reach.
  fs::path const dir = fs::temp_directory_path() / "prpll-test-emit-tunetxt";
  fs::remove_all(dir);
  fs::create_directories(dir);
  { File::openWrite(dir / "tune.txt").write(compatibilityText(view)); }
  fs::path const previous = fs::current_path();
  fs::current_path(dir);
  std::vector<TuneEntry> const read = TuneEntry::readTuneFile(Args{true});
  fs::current_path(previous);
  CHECK(linesOf(read) == linesOf(view));
  CHECK(compatibilityText(view).find(" # " + std::to_string(maxExp(FFTConfig{fp64})) + "\n") != std::string::npos);
  fs::remove_all(dir);
}

// An older binary runs a listed FFT at its defaults and reads no exclusion, so an FFT whose defaults computed a wrong
// answer is not listed, in whichever regime they did; one whose only exclusion is of another option set still is.
TEST(tune_txt_leaves_out_an_fft_whose_defaults_answered_wrongly) {
  std::string const fp64 = "512:15:512:212";
  std::string const ntt = "3:1K:8:512:202";
  FFTConfig const nttFft{ntt};

  auto exclusionOf = [](const std::string& spec, Regime regime, const UseConfig& opts) {
    return Exclusion{.fft = spec, .kind = TestKind::LL, .regime = regime, .opts = opts};
  };
  std::vector<SelectionEntry> const entries{entryOf(fp64, 1750, {}),
                                            entryOf(ntt, 2000, {}, {}, Evidence::NotApplicable)};

  CHECK(linesOf(compatibilityView(fileOf(entries), nvidia())).size() == 2);

  // Another option set, or the defaults spelled out (TAIL_KERNELS=2 is its default), make no difference to that.
  SelectionFile const other =
    fileOf(entries, {exclusionOf(fp64, topOf(FFTConfig{fp64}).regime, {{"TAIL_KERNELS", "3"}})});
  CHECK(linesOf(compatibilityView(other, nvidia())).size() == 2);

  SelectionFile const lower = fileOf(
    entries,
    {exclusionOf(ntt, intervals(nttFft, minExp(nttFft), maxExp(nttFft)).front().regime, {{"TAIL_KERNELS", "2"}})});
  CHECK(linesOf(compatibilityView(lower, nvidia())) == (std::vector<std::pair<std::string, double>>{{fp64, 1750}}));
}
