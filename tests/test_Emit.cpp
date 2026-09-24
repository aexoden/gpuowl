// Copyright (C) Jason Lynch

// Tests that a hand-written database emits the selection file it should: that every published entry is a transcript of
// one row, that a costlier configuration reaching further is kept beside a cheaper one that does not reach as far,
// that one no exponent would choose is dropped, and that rows of another env, of a thinly measured call, or of a
// configuration that ever failed are not published at all.

#include "Emit.h"

#include "test.h"

#include <algorithm>
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
  " pdl=0 machine=01:00.0 build=9a3f21c0d1e2f304\n"
  "env   2 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 machine=4d:00.0 build=9a3f21c0d1e2f304\n"
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
  "reach 4 512:15:512:212 prp short32 17 148000000 confirmed 1753471404\n"
  "roe   4 512:15:512:212 143413741 18 24.40 2150 0.3098 ok 1753471410\n"
  "roe   4 512:15:512:212 143498461 17 24.40 2150 0.3098 ok 1753471420\n"
  "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 ok 1753471430\n"
  "roe   9 512:15:512:212 143413741 17 24.40 2150 0.3098 ok 1753471440\n";

// The accuracy reading of the set published as 2e51eaf52a48bfc9 and 0a567e7dbc3e06d4, prp and ll alike.
const char* const PAD128_ROE = "roe   4 512:15:512:212 143413741 18 24.40 2150 0.3098 ok 1753471410\n";

// What env 1 supports.  Costs are the mean plus two standard errors, so the six calls behind the first entry buy it a
// smaller penalty (+1.84) than the four behind the second (+3.35); PAD=512 at 1802.236 is dropped as nothing cheaper
// than PAD=128 anywhere it runs; the ll rows under PAD=256 are dropped because one of the two failed; and the 1700.000
// of one call is dropped as a reading nothing concluded.  The first entry reaches past the fitted table because a
// reach row says so, and stops at 143498475 because that is where the carry regime it was measured in ends.
const char* const SELECTION = "# prpll selection v1\n"
                              "# %PROVENANCE%\n"
                              "use   INPLACE=1,PAD=256\n"
                              "use ! 1 TAIL_KERNELS=3\n"
                              "entry 2e51eaf52a48bfc9 1753.354 512:15:512:212 prp 78643196 143413744 unvalidated\n"
                              "opts  2e51eaf52a48bfc9 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
                              "entry adc5960c72b18307 1776.069 512:15:512:212 prp 78643196 143498475 confirmed\n"
                              "opts  adc5960c72b18307 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
                              "entry 0a567e7dbc3e06d4 1853.354 512:15:512:212 ll 78643196 143413744 unvalidated\n"
                              "opts  0a567e7dbc3e06d4 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
                              "entry 688735b9ab069fec 2005.590 3:1K:8:512:202 prp 83886076 152674512 n/a\n"
                              "opts  688735b9ab069fec INPLACE=1,PAD=256\n"
                              "entry d5c85edcdda38629 3104.472 1K:8:1K:202 prp 167772152 296960416 unvalidated\n"
                              "opts  d5c85edcdda38629 INPLACE=1,PAD=256\n";

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
  for (const SelectionEntry& e : entriesFor(db, 1, defaults())) {
    if (e.fft == fft && e.kind == kind) { return true; }
  }

  return false;
}

bool publishes(const TuneDB& db, const std::string& id) {
  for (const SelectionEntry& e : entriesFor(db, 1, defaults())) {
    if (e.id == id) { return true; }
  }

  return false;
}

// The entry the reach row in the fixture belongs to.
const char* const REACHED = "adc5960c72b18307";

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
  auto const other = entriesFor(db, 2, defaults());
  CHECK_EQ(other.size(), size_t{1});
  CHECK_EQ(other.at(0).opts.at("PAD"), std::string{"256"});
  CHECK(other.at(0).cost < 1000);

  for (const SelectionEntry& e : entriesFor(db, 1, defaults())) { CHECK(e.cost > 1000); }
}

// An entry is published for the exponents its row's own regime covers, so a reach that stops short of the exponent the
// row was measured at leaves nothing to publish rather than an interval the measurement says nothing about.
TEST(emit_drops_a_row_its_reach_no_longer_covers) {
  CHECK(publishes(loaded(DB), REACHED));
  CHECK(!publishes(loaded(withRecord("148000000 confirmed", "90000000 rejected")), REACHED));
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

  // And only that configuration: the others measured on the same FFT are untouched.
  CHECK(publishes(db, REACHED));
}

// A build that would not compile and a run the backend refused say nothing about the answers the configuration
// computes, so neither reaches past the exponent it happened at.
TEST(emit_keeps_a_configuration_a_refusal_never_ran) {
  CHECK(
    publishes(loaded(withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 unsupported 1753471294")), TWICE_MEASURED));
  CHECK(publishes(loaded(withRecord("16 4 1.0000 ok 1753471294", "16 4 1.0000 nocompile 1753471294")), TWICE_MEASURED));
}

// Reach rows are grouped by the id naming an option set and handed back in the order the ids first appeared, so a
// database that spells one set under two ids offers them in an order that says nothing about which reading is later.
TEST(emit_takes_the_newest_reach_however_its_options_were_named) {
  TuneDB const db = loaded(withRecord("reach 4 512:15:512:212 prp short32 17 148000000 confirmed 1753471404\n",
                                      "cfg   22 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
                                      "reach 4 512:15:512:212 prp short32 17 148000000 confirmed 1753471404\n"
                                      "reach 4 512:15:512:212 prp short32 22 143450000 confirmed 1753471414\n"
                                      "reach 4 512:15:512:212 prp short32 17 90000000 rejected 1753471424\n"));

  // The newest of the three rejects the configuration below the exponent it was measured at, so it publishes nothing
  // at all -- rather than the middle row's reach, which is what reading them in id order would have given.
  CHECK(!publishes(db, REACHED));
}

// An entry outranks the file's own default lines, so one that does not name a key they set would run under a value its
// measurement never saw.  A recorded option set is complete and cannot do this; a hand-written one can.
TEST(emit_drops_an_entry_its_own_default_lines_would_change) {
  TuneDB const db =
    loaded(withRecord("run   4 1K:8:1K:202 prp 200000000 short32 21 ", "run   4 1K:8:1K:202 prp 200000000 short32 1 "));

  auto publishedUnder = [&](const Defaults& lines) {
    return std::ranges::any_of(entriesFor(db, 1, lines),
                               [](const SelectionEntry& e) { return e.fft == "1K:8:1K:202"; });
  };

  // A line that moves a key the row left at its default changes what production would build.
  CHECK(!publishedUnder({.global = {{"TAIL_KERNELS", "3"}}, .family = {}}));
  CHECK(!publishedUnder({.global = {}, .family = {parseUseLine("! 0 WMUL=1")}}));

  // Lines that only name what the row ran anyway -- INPLACE=1 is NVIDIA's default, and PAD does nothing in place --
  // change nothing, so the row is published under them as it is under none.
  CHECK(publishes(db, "1K:8:1K:202", TestKind::PRP));

  // With nothing to shadow it, the same row is published.
  CHECK_EQ(entriesFor(db, 1).size(), entriesFor(loaded(DB), 1, defaults()).size());
}

namespace {

const char* const ROE_1K = "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 ok 1753471430\n";

// The 1K:8:1K entry, and a second, dearer set of the same identity that the gate has passed.
std::string withSecondSet(const std::string& roe1K) {
  std::string text = DB;
  text.replace(text.find(ROE_1K), std::string{ROE_1K}.size(), roe1K);
  return text +
    "cfg   22 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
    "run   4 1K:8:1K:202 prp 200000000 short32 22 3200.000 4.000 16 4 1.0000 ok 1753471450\n"
    "roe   4 1K:8:1K:202 296960407 22 24.90 2150 0.3040 ok 1753471460\n";
}

std::vector<std::string> published1K(const TuneDB& db, Gating gating = Gating::Required) {
  std::vector<std::string> out;
  for (const SelectionEntry& e : entriesFor(db, 1, defaults(), gating)) {
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

TEST(a_configuration_the_gate_rejects_never_reaches_the_selection_file) {
  std::vector<std::string> const cheapest{"INPLACE=1,PAD=256"};
  std::vector<std::string> const dearer{"INPLACE=1,PAD=256,TAIL_KERNELS=3"};
  CHECK(published1K(loaded(withSecondSet(ROE_1K))) == cheapest);

  // Read below the FP64 floor at the top of its interval, and with a failed Gerbicz check: either way the cheaper set
  // is gone, from the file and from what the search works from, and the set it kept out takes its place.
  for (const char* roe : {"roe   4 1K:8:1K:202 296960407 21 17.20 2150 0.4011 ok 1753471430\n",
                          "roe   4 1K:8:1K:202 296960407 21 25.10 2150 0.3021 fail 1753471430\n"}) {
    TuneDB const db = loaded(withSecondSet(roe));
    CHECK(published1K(db) == dearer);
    CHECK(published1K(db, Gating::Assumed) == dearer);
    CHECK(emitted(db, provenance()).find("3104.472") == std::string::npos);
    for (const OptionSet& s : optionSetsFor(db, 1, defaults())) {
      CHECK(s.entry.fft != "1K:8:1K:202" || s.entry.opts.contains("TAIL_KERNELS"));
    }
    CHECK(gatesOwed(db, 1, defaults()).empty());
  }
}

TEST(a_set_the_gate_has_not_read_waits_for_its_reading) {
  TuneDB const db = loaded(withSecondSet(""));

  // Not published, and nothing dearer of its identity stands in for it where the search expects it to pass.
  CHECK(published1K(db) == std::vector<std::string>{"INPLACE=1,PAD=256,TAIL_KERNELS=3"});
  CHECK(published1K(db, Gating::Assumed) == std::vector<std::string>{"INPLACE=1,PAD=256"});

  std::vector<OptionSet> const owed = gatesOwed(db, 1, defaults());
  CHECK_EQ(owed.size(), size_t{1});
  CHECK_EQ(configText(owed.at(0).entry.opts), std::string{"INPLACE=1,PAD=256"});
  CHECK_EQ(owed.at(0).gateExponent, u64(296'960'407));
  CHECK(!owed.at(0).gate.owesReference);

  // Read where it would not count -- below the top of the interval -- it is still owed.
  TuneDB const low = loaded(withSecondSet("roe   4 1K:8:1K:202 250000013 21 25.10 2150 0.3021 ok 1753471430\n"));
  CHECK_EQ(gatesOwed(low, 1, defaults()).size(), size_t{1});
}

TEST(a_reading_at_the_fitted_standard_publishes_its_reach_as_confirmed) {
  TuneDB const db = loaded(withRecord(ROE_1K, "roe   4 1K:8:1K:202 296960407 21 28.30 2150 0.2711 ok 1753471430\n"));
  bool seen = false;
  for (const SelectionEntry& e : entriesFor(db, 1, defaults())) {
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
  CHECK_EQ(owed.at(0).gateExponent, top);
  CHECK(entriesFor(db, id).empty());

  // Within ACCURACY_SLACK_Z of it, it is published; read again better, the defaults show what the set spent.
  read(accuracyReference(env, fft, moved), 12.4);
  CHECK_EQ(entriesFor(db, id).size(), size_t{1});
  read({}, 13);
  CHECK(entriesFor(db, id).empty());
  CHECK(gatesOwed(db, id).empty());
}
