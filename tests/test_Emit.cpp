// Copyright (C) Jason Lynch

// Tests that a hand-written database emits the selection file it should: that every published entry is a transcript of
// one row, that a costlier configuration reaching further is kept beside a cheaper one that does not reach as far,
// that one no exponent would choose is dropped, and that rows of another env, of a thinly measured call, or of a
// configuration that ever failed are not published at all.

#include "Emit.h"

#include "test.h"

#include <string>

using namespace tune;

namespace {

// Ten rows on env 1 and one on env 2, over three FFTs: one FP64 shape measured under three option sets and at two
// exponents of one regime, a second FP64 shape in another regime, and a pure NTT shape.
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
  "reach 4 512:15:512:212 prp short32 17 148000000 confirmed 1753471404\n";

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

  CHECK(!publishes(db, "1K:8:1K:202", TestKind::PRP));

  // With nothing to shadow it, the same row is published.
  CHECK_EQ(entriesFor(db, 1).size(), entriesFor(loaded(DB), 1, defaults()).size());
}
