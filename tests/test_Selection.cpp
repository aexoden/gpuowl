// Copyright (C) Jason Lynch

// Tests that a hand-written file round-trips, that an entry with no option set is refused, that an unknown record
// survives a rewrite, that an entry's id is stable, and that finalize() will not publish an interval nothing runs over.

#include "Selection.h"

#include "File.h"
#include "test.h"

#include <limits>
#include <string>

using namespace tune;

namespace {

const char* const FIXTURE =
  "# prpll selection v1\n"
  "# written 1753471500 by 0.15-430 from tunedb.txt env 1; T=1801.4 over workload 100000000-400000000\n"
  "use   INPLACE=1,PAD=256\n"
  "use ! 1 TAIL_KERNELS=3\n"
  "use ! prp:512:15:512 PAD=128\n"
  "entry adc5960c72b18307 1774.230 512:15:512:212 prp 100000000 143000000 confirmed\n"
  "opts  adc5960c72b18307 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "entry 25616810ba5a560c 1801.000 1K:8:1K:202 prp 100000000 160000000 n/a\n"
  "opts  25616810ba5a560c -\n"
  "budget 400\n"
  "# a note the user added by hand\n";

SelectionFile loaded(const std::string& text) {
  auto const file = parseSelection(text, "fixture");
  CHECK(file.has_value());
  return file.value_or(SelectionFile{});
}

std::string withRecord(const std::string& original, const std::string& replacement) {
  std::string text = FIXTURE;
  auto const at = text.find(original);
  CHECK(at != std::string::npos);
  return text.replace(at, original.size(), replacement);
}

void rejects(const std::string& original, const std::string& replacement) {
  CHECK(!parseSelection(withRecord(original, replacement), "fixture"));
}

}  // namespace

TEST(selection_round_trips) {
  SelectionFile const file = loaded(FIXTURE);
  CHECK_EQ(text(file), std::string{FIXTURE});
}

TEST(selection_records_are_read) {
  SelectionFile const file = loaded(FIXTURE);

  CHECK_EQ(file.global.size(), size_t{2});
  CHECK_EQ(file.global.at(0).first, std::string{"INPLACE"});
  CHECK_EQ(file.family.size(), size_t{2});
  CHECK(!file.family.at(0).selector.kind.has_value());
  CHECK(file.family.at(1).selector.kind.has_value());

  CHECK_EQ(file.entries.size(), size_t{2});
  SelectionEntry const& e = file.entries.at(0);
  CHECK_EQ(e.fft, std::string{"512:15:512:212"});
  CHECK(e.kind == TestKind::PRP);
  CHECK_EQ(e.cost, 1774.23);
  CHECK_EQ(e.emin, u64{100'000'000});
  CHECK_EQ(e.reach, u64{143'000'000});
  CHECK(e.evidence == Evidence::Confirmed);
  CHECK_EQ(e.opts.at("TAIL_KERNELS"), std::string{"3"});
  CHECK(file.entries.at(1).evidence == Evidence::NotApplicable);
  CHECK(file.entries.at(1).opts.empty());

  // The regime is not a column: it is recovered from the FFT and the bottom of the interval, which is what lets the
  // six-column entry line address an id that hashes the regime.
  CHECK_EQ(e.regime.label(), std::string{"short32"});

  // The layers this file contributes to production's precedence, weakest first.
  SelectionLayers const layers = file.layersFor(e);
  CHECK_EQ(layers.global.size(), size_t{2});
  CHECK_EQ(layers.family.size(), size_t{2});
  CHECK_EQ(layers.entry.size(), size_t{3});
}

TEST(unknown_selection_records_pass_through) {
  SelectionFile const file = loaded(FIXTURE);
  CHECK_EQ(file.unknown.size(), size_t{2});
  CHECK_EQ(file.unknown.at(0), std::string{"budget 400"});
  // A comment a user wrote survives a rewrite too, rather than being deleted by the next publish.
  CHECK_EQ(file.unknown.at(1), std::string{"# a note the user added by hand"});
}

TEST(malformed_selection_records_are_rejected) {
  rejects("# prpll selection v1", "# prpll selection v2");
  rejects("opts  adc5960c72b18307 INPLACE=1,PAD=256,TAIL_KERNELS=3\n", "");  // an entry with no option set
  rejects("entry 25616810ba5a560c 1801.000 1K:8:1K:202 prp 100000000 160000000 n/a\n",
          "");  // opts with no entry
  rejects("1774.230 512:15:512:212 prp", "cheap 512:15:512:212 prp");
  rejects("1774.230 512:15:512:212 prp", "1774.230 not-an-fft prp");
  rejects("1774.230 512:15:512:212 prp 100000000", "1774.230 512:15:512:212 cert 100000000");
  rejects("100000000 143000000 confirmed", "100000000 143000000 probably");
  rejects("100000000 143000000 confirmed", "100000000 confirmed");   // a column short
  rejects("1774.230 512:15:512:212 prp", "nan 512:15:512:212 prp");  // a cost that is not a number at all
  rejects("1774.230 512:15:512:212 prp", "-1.0 512:15:512:212 prp");
  rejects("use   INPLACE=1,PAD=256", "use   INPLACE=1,PAD=256\nuse   PAD=512");  // a second global line
  rejects("use   INPLACE=1,PAD=256", "use   INPLACE=1,");
  rejects("use ! 1 TAIL_KERNELS=3", "use ! 512:15 TAIL_KERNELS=3");
}

TEST(an_id_that_does_not_hash_to_its_entry_is_refused) {
  // Renamed on both lines, so the entry still finds its option set and only the hash disagrees: a hand-edited or
  // stale file that kept the id it used to have.
  std::string text = FIXTURE;
  for (auto at = text.find("adc5960c72b18307"); at != std::string::npos; at = text.find("adc5960c72b18307")) {
    text.replace(at, 16, "adc5960c72b18300");
  }

  CHECK(text.find("adc5960c72b18300") != std::string::npos);
  CHECK(!parseSelection(text, "fixture"));
}

TEST(entry_ids_are_stable) {
  UseConfig const opts{{"INPLACE", "1"}, {"PAD", "256"}, {"TAIL_KERNELS", "3"}};
  Regime const shortCarry{};

  // Pinned, because the id is what a published line is recognised by across rewrites and across machines: it may only
  // change when the schema version does.
  CHECK_EQ(entryId("512:15:512:212", TestKind::PRP, shortCarry, opts), std::string{"adc5960c72b18307"});

  // Every part of the key separates.
  CHECK(entryId("512:15:512:212", TestKind::LL, shortCarry, opts) !=
        entryId("512:15:512:212", TestKind::PRP, shortCarry, opts));
  CHECK(entryId("512:15:512:202", TestKind::PRP, shortCarry, opts) !=
        entryId("512:15:512:212", TestKind::PRP, shortCarry, opts));
  CHECK(entryId("512:15:512:212", TestKind::PRP, Regime{true, false}, opts) !=
        entryId("512:15:512:212", TestKind::PRP, shortCarry, opts));
  CHECK(entryId("512:15:512:212", TestKind::PRP, shortCarry, UseConfig{}) !=
        entryId("512:15:512:212", TestKind::PRP, shortCarry, opts));
}

TEST(finalize_fills_in_and_sorts) {
  SelectionFile file;

  SelectionEntry slow;
  slow.fft = "1024:8:1K:102";  // a spelling that folds (M1.4)
  slow.cost = 2000;
  slow.emin = 100'000'000;
  slow.reach = 160'000'000;

  SelectionEntry fast;
  fast.fft = "512:15:512:212";
  fast.cost = 1774.23;
  fast.emin = 100'000'000;
  fast.reach = 143'000'000;
  fast.opts = UseConfig{{"PAD", "256"}};

  file.entries = {slow, fast};
  CHECK(finalize(file));

  CHECK_EQ(file.entries.at(0).fft, std::string{"512:15:512:212"});  // cheapest first
  CHECK_EQ(file.entries.at(1).fft, std::string{"1K:8:1K:202"});     // canonical
  CHECK_EQ(file.entries.at(0).id, entryId("512:15:512:212", TestKind::PRP, Regime{}, file.entries.at(0).opts));
  CHECK_EQ(file.entries.at(0).regime.label(), std::string{"short32"});

  // And what finalize() wrote is what a reader reads back.
  SelectionFile const again = loaded(text(file));
  CHECK_EQ(again.entries.at(0).id, file.entries.at(0).id);
  CHECK_EQ(again.entries.at(0).opts.at("PAD"), std::string{"256"});
  CHECK_EQ(text(again), text(file));

  SelectionFile bad;
  SelectionEntry nonsense;
  nonsense.fft = "not-an-fft";
  bad.entries = {nonsense};
  CHECK(!finalize(bad));
}

TEST(a_long_carry_entry_recovers_its_regime) {
  // Below 10 bits per word the Gpu forces the long carry, so an entry whose interval sits there is a different set of
  // kernels and reads back as long32 without the file having to say so.
  FFTConfig const fft{"1K:8:1K:202"};
  u64 const low = u64(9.0 * fft.size());
  CHECK_EQ(regimeOf(fft, low).label(), std::string{"long32"});

  SelectionFile file;
  SelectionEntry e;
  e.fft = fft.spec();
  e.cost = 100;
  e.emin = low;
  e.reach = low + 1000;
  file.entries = {e};
  CHECK(finalize(file));
  CHECK_EQ(loaded(text(file)).entries.at(0).regime.label(), std::string{"long32"});
}

TEST(a_blank_looking_line_is_not_a_record) {
  // A line of spaces is not empty, but splits into no fields at all; a hand-edited file, or a writer that indents,
  // produces one.
  std::string const text = std::string{"# prpll selection v1\n"} + "   \n\t\nuse   -\n";
  auto const file = parseSelection(text, "fixture");
  CHECK(file.has_value());
  CHECK(file.value_or(SelectionFile{}).global.empty());
}

TEST(finalize_refuses_an_interval_nothing_runs_over) {
  auto const one = [](u64 emin, u64 reach, const char* fft = "1K:8:1K:202") {
    SelectionFile file;
    SelectionEntry e;
    e.fft = fft;
    e.cost = 100;
    e.emin = emin;
    e.reach = reach;
    file.entries = {e};
    return finalize(file);
  };

  CHECK(one(100'000'000, 160'000'000));     // one regime, as the fixture has it
  CHECK(!one(160'000'000, 100'000'000));    // empty: the interval runs backwards
  CHECK(!one(100'000'000, 3'005'470'400));  // far past what the bpw table covers
  CHECK(!one(150'994'944, 285'212'672));    // labelled long32 from emin, but short32 above 167772151
  CHECK(!one(1000, 2000));                  // below the smallest exponent this FFT accepts

  // Two entries for one configuration in one regime would share an id, and the file they write cannot be read back.
  SelectionFile twice;
  SelectionEntry e;
  e.fft = "1K:8:1K:202";
  e.cost = 100;
  e.emin = 100'000'000;
  e.reach = 120'000'000;
  SelectionEntry other = e;
  other.emin = 130'000'000;
  other.reach = 160'000'000;
  twice.entries = {e, other};
  CHECK(!finalize(twice));

  // An option value the row grammar cannot spell would come back as a different option.
  SelectionFile unspellable;
  e.opts = UseConfig{{"PAD", "2,5"}};
  unspellable.entries = {e};
  CHECK(!finalize(unspellable));
}

TEST(an_absent_selection_file_is_an_empty_one) {
  // Distinct from a file that is there and unreadable, which is what production has to refuse rather than ignore.
  auto const file = readSelection("no-such-selection-file.txt");
  CHECK(file.has_value());
  CHECK(file.value_or(SelectionFile{}).entries.empty());
}

TEST(an_empty_selection_file_is_empty_but_an_unreadable_one_is_refused) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-selection.txt";
  fs::remove(path);
  { File::openWrite(path); }

  auto const empty = readSelection(path);
  CHECK(empty.has_value());
  CHECK(empty.value_or(SelectionFile{}).entries.empty());

  fs::permissions(path, fs::perms::none);
  if (File::openRead(path)) {
    fs::remove(path);  // running as a user every file opens for, which this cannot tell apart
    return;
  }

  // Read as absent, this would drop every setting production was told to use, silently.
  CHECK(!readSelection(path));
  fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write);
  fs::remove(path);
}

TEST(the_reader_holds_an_entry_to_what_finalize_requires) {
  // Neither bound is hashed into the id, so an interval edited by hand reaches production unless the reader checks
  // what finalize() checked. 1K:8:1K:202 crosses from long32 to short32 at 167772151.
  rejects("100000000 160000000 n/a", "100000000 250000000 n/a");
  rejects("100000000 160000000 n/a", "160000000 100000000 n/a");  // runs backwards
  rejects("100000000 160000000 n/a", "1000 2000 n/a");            // below what the FFT accepts

  // An option value the row grammar cannot write back, on any of the three layers.
  rejects("opts  25616810ba5a560c -", "opts  25616810ba5a560c \"PAD=2 5\"");
  rejects("use   INPLACE=1,PAD=256", "use   \"PAD=2 5\"");
  rejects("use ! 1 TAIL_KERNELS=3", "use ! 1 \"PAD=2 5\"");
}

TEST(finalize_refuses_what_the_layers_cannot_spell) {
  // The global and family lines are published as they stand; checking only the entry leaves a file its own reader
  // rejects.
  SelectionFile global;
  global.global = {{"PAD", "2 5"}};
  CHECK(!finalize(global));

  SelectionFile family;
  family.family = {UseLine{.selector = parseUseLine("! 1 PAD=256").selector, .uses = {{"PAD", "2 5"}}}};
  CHECK(!finalize(family));

  SelectionFile cost;
  SelectionEntry e;
  e.fft = "1K:8:1K:202";
  e.cost = std::numeric_limits<double>::quiet_NaN();
  e.emin = 100'000'000;
  e.reach = 160'000'000;
  cost.entries = {e};
  CHECK(!finalize(cost));
}
