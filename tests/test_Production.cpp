// Copyright (C) Jason Lynch

// Tests that what a selection file publishes is what production resolves: every published option set comes back
// unchanged through production's own resolver, a user setting that overrides one of them costs the entry the reach its
// measurement bought and names the key that did it, a key that changes nothing here costs it nothing, an -fft spec
// holds the walk to the shape it names, and a file with no entry for the exponent leaves the shape scan to answer.

#include "Production.h"

#include "Args.h"
#include "Emit.h"
#include "File.h"
#include "log.h"
#include "Reach.h"

#include "test.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

using namespace tune;

namespace {

std::string joined(const UseConfig& config) { return configText(config); }

// A file built and finalized here rather than typed out, so that the ids are the ones the grammar computes, and then
// read back through the reader production reads it with.
SelectionFile published(std::vector<SelectionEntry> entries, const Defaults& defaults = {},
                        std::vector<Exclusion> excluded = {}) {
  SelectionFile file{.provenance = "written 1753471500 by test from tunedb.txt env 1",
                     .global = {defaults.global.begin(), defaults.global.end()},
                     .family = defaults.family,
                     .entries = std::move(entries),
                     .limits = {},
                     .excluded = std::move(excluded),
                     .unknown = {}};

  CHECK(finalize(file));

  auto const read = parseSelection(text(file), "fixture");
  CHECK(read.has_value());
  return read.value_or(SelectionFile{});
}

// Choice holds an FFTConfig, which has no empty value, so a failed choice is reported by spec rather than by a
// default-constructed Choice.
std::string specOf(const std::optional<Choice>& choice) { return choice ? choice->fft.spec() : std::string{"none"}; }

// Runs a test in a directory of its own, and puts the old one back however the test ends.
class WorkingDirectory {
public:
  explicit WorkingDirectory(const fs::path& to) : previous_{fs::current_path()} { fs::current_path(to); }
  ~WorkingDirectory() { fs::current_path(previous_); }

  WorkingDirectory(const WorkingDirectory&) = delete;
  WorkingDirectory& operator=(const WorkingDirectory&) = delete;

private:
  fs::path previous_;
};

Args configured(const std::vector<std::string>& configLines, const std::vector<std::string>& commandLine = {}) {
  Args args{true};
  for (const std::string& line : configLines) { args.parse(line, true); }
  for (const std::string& line : commandLine) { args.parse(line); }
  return args;
}

const UseConfig CHEAP_OPTS{{"INPLACE", "1"}, {"TAIL_KERNELS", "3"}};
const UseConfig DEAR_OPTS{{"INPLACE", "1"}, {"TAIL_KERNELS", "2"}};

// Two entries of one kind over one workload: the cheaper one stops where the fitted table stops for its shape plus a
// little, which a measured reach may do and which is what makes the clamp observable.
u64 table() { return maxExp(FFTConfig{"512:15:512:212"}); }

SelectionFile twoEntries() {
  SelectionEntry cheap{.id = {},
                       .cost = 1700,
                       .fft = "512:15:512:212",
                       .kind = TestKind::PRP,
                       .emin = 100'000'000,
                       .reach = table() + 30'000,
                       .regime = {},
                       .evidence = Evidence::Confirmed,
                       .opts = CHEAP_OPTS};

  SelectionEntry dear{.id = {},
                      .cost = 1900,
                      .fft = "1K:8:1K:202",
                      .kind = TestKind::PRP,
                      .emin = 100'000'000,
                      .reach = 160'000'000,
                      .regime = {},
                      .evidence = Evidence::Unvalidated,
                      .opts = DEAR_OPTS};

  return published({cheap, dear}, Defaults{.global = UseConfig{{"INPLACE", "1"}}, .family = {}});
}

}  // namespace

TEST(a_published_option_set_resolves_to_what_was_measured) {
  SelectionFile const file = twoEntries();
  Args const args = configured({});

  for (const SelectionEntry& e : file.entries) {
    Args pinned = args;
    pinned.fftSpec = e.fft;

    for (u64 const E : {e.emin, (e.emin + e.reach) / 2, e.reach}) {
      auto const choice = chooseFrom(file, pinned, Env{}, E, e.kind);
      CHECK(choice.has_value());
      if (!choice) { continue; }

      CHECK_EQ(choice->fft.spec(), e.fft);
      CHECK_EQ(joined(choice->options), joined(e.opts));
      CHECK(choice->shadowed.empty());
      CHECK_EQ(choice->reach, e.reach);
    }
  }
}

TEST(the_cheapest_entry_that_covers_the_exponent_wins) {
  SelectionFile const file = twoEntries();
  Args const args = configured({});

  // Both cover it, and the cheaper one is not the first line of the file after a sort by cost.
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, 120'000'000, TestKind::PRP)), std::string{"512:15:512:212"});

  // Only the dearer one reaches here.
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, 150'000'000, TestKind::PRP)), std::string{"1K:8:1K:202"});

  // Nothing is published for LL, and an entry of the other kind is not a substitute for one.
  CHECK(!chooseFrom(file, args, Env{}, 120'000'000, TestKind::LL));
}

// A file can be read by a device it was not published for: an FP64 entry on a card without FP64, or digit 0's broadcast
// where the compiler has no way to write it.  Either is passed over for the next entry this device can build.
TEST(an_entry_the_device_cannot_build_is_passed_over) {
  auto entry = [](const char* fft, double cost) {
    return SelectionEntry{.id = {},
                          .cost = cost,
                          .fft = fft,
                          .kind = TestKind::PRP,
                          .emin = 100'000'000,
                          .reach = 150'000'000,
                          .regime = {},
                          .evidence = Evidence::Unvalidated,
                          .opts = {}};
  };
  SelectionFile const file =
    published({entry("1K:8:512:000", 1500), entry("1K:8:512:202", 1600), entry("3:1K:8:512:202", 1700)});
  Args const args = configured({});
  u64 const E = 120'000'000;

  Env const cuda{.isNvidia = true, .cudaBackend = true, .computeCapability = 806};
  Env const nvidiaOpenCl{.isNvidia = true, .computeCapability = 806};
  Env const nvidiaNoAsm{.isNvidia = true, .noAsm = true, .computeCapability = 806};
  Env const amd{.isAmd = true};
  Env const amdWithoutBuiltins{.isAmd = true, .amdBuiltins = false};
  Env const noFp64{.isNvidia = true, .cudaBackend = true, .hasFP64 = false, .computeCapability = 806};

  CHECK_EQ(specOf(chooseFrom(file, args, cuda, E, TestKind::PRP)), std::string{"1K:8:512:000"});
  CHECK_EQ(specOf(chooseFrom(file, args, nvidiaOpenCl, E, TestKind::PRP)), std::string{"1K:8:512:000"});
  CHECK_EQ(specOf(chooseFrom(file, args, amd, E, TestKind::PRP)), std::string{"1K:8:512:000"});
  CHECK_EQ(specOf(chooseFrom(file, args, nvidiaNoAsm, E, TestKind::PRP)), std::string{"1K:8:512:202"});
  CHECK_EQ(specOf(chooseFrom(file, args, amdWithoutBuiltins, E, TestKind::PRP)), std::string{"1K:8:512:202"});
  CHECK_EQ(specOf(chooseFrom(file, args, noFp64, E, TestKind::PRP)), std::string{"3:1K:8:512:202"});

  // A pinned spec the device cannot build is not run from its entry either; the shape scan answers for it.
  CHECK(!chooseFrom(file, configured({}, {"-fft 1K:8:512:000"}), nvidiaNoAsm, E, TestKind::PRP));
}

TEST(a_shadowed_entry_clamps_and_names_the_keys) {
  SelectionFile const file = twoEntries();
  Args const args = configured({"-use TAIL_KERNELS=1"});
  u64 const clamped = table();

  // Below the fitted table's own limit the entry still runs, at the options the user asked for, saying what it lost.
  auto const inside = chooseFrom(file, args, Env{}, 120'000'000, TestKind::PRP);
  CHECK(specOf(inside) == std::string{"512:15:512:212"});
  if (inside) {
    CHECK_EQ(inside->options.at("TAIL_KERNELS"), std::string{"1"});
    CHECK_EQ(inside->shadowed.size(), size_t{1});
    CHECK_EQ(inside->shadowed.at(0), std::string{"TAIL_KERNELS"});
    CHECK_EQ(inside->reach, clamped);
  }

  // Between the clamped reach and the measured one, the entry is no longer eligible and the walk moves on, which is
  // the whole point of clamping before the interval is re-tested rather than after.
  u64 const between = clamped + 10'000;
  CHECK(between < file.entries.at(0).reach);
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, between, TestKind::PRP)), std::string{"1K:8:1K:202"});

  // Without the override it is eligible there, so it is the override that moved the answer and not the exponent.
  CHECK_EQ(specOf(chooseFrom(file, configured({}), Env{}, between, TestKind::PRP)), std::string{"512:15:512:212"});
}

TEST(a_key_that_changes_nothing_here_does_not_shadow) {
  SelectionFile const file = twoEntries();

  // ENABLE_BARSYNC needs PTX 200, which this env has not got, so setting it cannot make the entry run differently from
  // the way it was measured -- and taking its reach away for it would be a safety cost paid for nothing.
  auto const inert = chooseFrom(file, configured({"-use ENABLE_BARSYNC=1"}), Env{}, 120'000'000, TestKind::PRP);
  CHECK(inert.has_value());
  if (inert) {
    CHECK(inert->shadowed.empty());
    CHECK_EQ(inert->reach, file.entries.at(0).reach);
  }

  // On a card where it does apply, it does.
  Env nvidia;
  nvidia.isNvidia = true;
  nvidia.computeCapability = 806;
  auto const live = chooseFrom(file, configured({"-use ENABLE_BARSYNC=1"}), nvidia, 120'000'000, TestKind::PRP);
  CHECK(live.has_value());
  if (live) { CHECK_EQ(live->shadowed.size(), size_t{1}); }
}

TEST(a_command_line_use_shadows_as_a_config_file_one_does) {
  SelectionFile const file = twoEntries();

  auto const choice = chooseFrom(file, configured({}, {"-use TAIL_KERNELS=1"}), Env{}, 120'000'000, TestKind::PRP);
  CHECK(choice.has_value());
  if (choice) {
    CHECK_EQ(choice->shadowed.size(), size_t{1});
    CHECK_EQ(choice->reach, table());
  }
}

TEST(an_fft_spec_holds_the_walk_to_the_shape_it_names) {
  SelectionFile const file = twoEntries();

  Args args = configured({});
  args.fftSpec = "1K:8:1K:202";

  // The cheaper entry covers this exponent, but it is not the shape that was asked for.
  auto const choice = chooseFrom(file, args, Env{}, 120'000'000, TestKind::PRP);
  CHECK_EQ(specOf(choice), std::string{"1K:8:1K:202"});
  if (choice) { CHECK_EQ(joined(choice->options), joined(DEAR_OPTS)); }

  // A shape nothing is published for is the shape scan's to answer, not this walk's.
  args.fftSpec = "256:2:256:101";
  CHECK(!chooseFrom(file, args, Env{}, 20'000'000, TestKind::PRP));
}

TEST(an_exponent_no_entry_covers_is_left_to_the_shape_scan) {
  SelectionFile const file = twoEntries();
  Args const args = configured({});

  CHECK(!chooseFrom(file, args, Env{}, 90'000'000, TestKind::PRP));
  CHECK(!chooseFrom(file, args, Env{}, 400'000'000, TestKind::PRP));

  // -fftOverdrive is the user's standing override of every such limit, and means the same thing here as it does to the
  // shape scan.
  Args overdriven = configured({});
  overdriven.fftOverdrive = 1.5;
  CHECK(chooseFrom(file, overdriven, Env{}, 165'000'000, TestKind::PRP));

  // It does not carry an entry across a regime boundary, though: past 167772151 the 1K:8:1K entry's own kernels are
  // not the ones it was measured with, whatever the user is willing to risk on its accuracy.
  CHECK(!chooseFrom(file, overdriven, Env{}, 200'000'000, TestKind::PRP));
}

TEST(a_selection_file_is_optional_and_its_own_lines_survive_a_fallback) {
  // In a directory of its own, so that neither this checkout's tune.txt nor a selection file beside it answers for a
  // question about what happens when there is none.
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  Args const args = configured({});

  // No file at all: the shape scan answers, exactly as it does for a build that has never been tuned.
  Choice const untuned = choose(args, Env{}, 40'000'000, TestKind::PRP);
  CHECK(!untuned.entry.has_value());
  CHECK(untuned.options.empty());
  CHECK_EQ(untuned.reach, maxExp(untuned.fft));
  CHECK(isEligible(untuned.fft, 40'000'000));

  writeSelection("selection.txt", twoEntries());

  // Covered by an entry: that entry's own set, complete.
  Choice const covered = choose(args, Env{}, 120'000'000, TestKind::PRP);
  CHECK(covered.entry.has_value());
  CHECK_EQ(joined(covered.options), joined(CHEAP_OPTS));

  // Not covered by any entry: the shape scan answers again, and the file's global line still applies to what it
  // answers, since that line is what a configuration with no entry of its own runs at.
  Choice const uncovered = choose(args, Env{}, 40'000'000, TestKind::PRP);
  CHECK(!uncovered.entry.has_value());
  CHECK_EQ(joined(uncovered.options), std::string{"INPLACE=1"});

  fs::remove_all(dir);
}

TEST(a_file_that_covers_nothing_here_says_so_and_no_file_says_nothing_of_one) {
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production-uncovered";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  // Each is said once a process, so at an exponent no other test chooses for.
  u64 const E = 41'000'011;
  Args const args = configured({});
  std::vector<std::string> said;
  setStdoutSink([&](std::string_view s) { said.emplace_back(s); });

  auto const saysTuneTxt = [&] {
    return std::ranges::any_of(said, [](const std::string& s) { return s.find("in tune.txt") != std::string::npos; });
  };

  (void)choose(args, Env{}, E, TestKind::PRP);
  bool const quiet =
    std::ranges::none_of(said, [](const std::string& s) { return s.find("selection.txt") != std::string::npos; });
  bool const pointedAtTuneTxt = saysTuneTxt();

  // Once there is a file, its own note says why the shape scan answers; tune.txt, which it has replaced, goes
  // unmentioned.
  said.clear();
  writeSelection("selection.txt", twoEntries());
  (void)choose(args, Env{}, E, TestKind::PRP);
  setStdoutSink(nullptr);

  CHECK(quiet);
  CHECK(pointedAtTuneTxt);
  CHECK(!saysTuneTxt());
  std::string const reach = to_string(std::max(table() + 30'000, u64{160'000'000}));
  CHECK(std::ranges::any_of(said, [&](const std::string& s) {
    return s.find("Note: no entry in selection.txt covers 41000011 (its prp entries cover 100000000-" + reach +
                  "), so PRPLL's own choice of FFT runs it under the file's default lines") != std::string::npos &&
      s.find("-tune workload=41000011-" + reach) != std::string::npos;
  }));

  // With nothing of the kind published, and with an -fft nothing of was.
  CHECK_EQ(uncoveredNote(twoEntries(), args, E, TestKind::LL),
           std::string{"Note: selection.txt publishes no ll entry yet, so PRPLL's own choice of FFT runs it under the "
                       "file's default lines, which were not measured on it\n"});
  CHECK_EQ(uncoveredNote(twoEntries(), configured({}, {"-fft 256:13:512:101"}), E, TestKind::PRP),
           std::string{"Note: no prp entry of 256:13:512:101 in selection.txt covers 41000011; running it under the "
                       "file's default lines, which were not measured on it\n"});

  fs::remove_all(dir);
}

TEST(every_entry_emit_publishes_resolves_to_the_row_it_transcribes) {
  // The round trip end to end: rows in, a published file out, and every option set in it read back through the
  // resolver production runs -- so nothing can be published that production would not reproduce.
  const char* const DB =
    "# prpll tunedb v1\n"
    "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
    " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
    "cfg   1 -\n"
    "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
    "cfg   18 INPLACE=1,PAD=128,TAIL_KERNELS=3\n"
    "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@100000000\n"
    "run   4 512:15:512:212 prp 100000000 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
    "run   4 512:15:512:212 prp 100000000 short32 18 1750.000 3.000 16 4 1.0000 ok 1753471284\n"
    "run   4 1K:8:1K:202 prp 200000000 short32 1 3100.000 4.000 16 4 1.0000 ok 1753471354\n"
    "run   4 512:15:512:212 ll 100000000 short32 17 1800.000 3.000 16 4 1.0000 ok 1753471324\n"
    "roe   4 512:15:512:212 143413741 17 24.40 2150 0.3098 ok - 1753471330\n"
    "roe   4 1K:8:1K:202 296960407 1 25.10 2150 0.3021 ok - 1753471360\n";

  TuneDB db;
  CHECK(db.parse(DB, "fixture"));

  auto const emitted = emit(db, Defaults{}, Provenance{.ts = 1'753'471'500, .db = "tunedb.txt", .env = 1});
  CHECK(emitted.has_value());
  SelectionFile const file = emitted.value_or(SelectionFile{});
  CHECK(!file.entries.empty());

  Args const args = configured({});
  for (const SelectionEntry& e : file.entries) {
    auto const choice = chooseFrom(file, args, Env{}, e.emin, e.kind);
    CHECK(choice.has_value());
    if (!choice) { continue; }

    CHECK_EQ(joined(choice->options), joined(e.opts));
    CHECK(choice->shadowed.empty());
  }
}

TEST(shadowed_keys_names_the_key_that_differs) {
  SelectionFile const file = twoEntries();
  FFTConfig const fft{"512:15:512:212"};

  auto const keys = shadowedKeys(configured({"-use TAIL_KERNELS=1"}), Env{}, file, file.entries.at(0), fft);
  CHECK_EQ(keys.size(), size_t{1});
  CHECK_EQ(keys.at(0), std::string{"TAIL_KERNELS"});

  // A key nothing published names at all is a setting the measurement did not have, which is the same kind of
  // difference as one it had another value for.
  CHECK_EQ(shadowedKeys(configured({"-use WMUL=1"}), Env{}, file, file.entries.at(0), fft).size(), size_t{1});

  CHECK(shadowedKeys(configured({}), Env{}, file, file.entries.at(0), fft).empty());
}

TEST(a_size_only_fft_argument_still_finds_its_entry) {
  // -fft takes a bare size, a shape, or a full spec, and the shape scan resolves each of them to one FFT. The walk has
  // to resolve them the same way or a spec the scan would honour silently drops the options published for it.
  FFTConfig const named{"8M"};
  Interval const span = intervals(named, minExp(named), maxExp(named)).back();

  SelectionEntry entry{.id = {},
                       .cost = 1700,
                       .fft = named.spec(),
                       .kind = TestKind::PRP,
                       .emin = span.lo,
                       .reach = span.hi,
                       .regime = {},
                       .evidence = Evidence::Confirmed,
                       .opts = CHEAP_OPTS};

  SelectionFile const file = published({entry});
  u64 const E = (span.lo + span.hi) / 2;

  for (const std::string& spec : {std::string{"8M"}, named.spec()}) {
    Args args = configured({});
    args.fftSpec = spec;

    auto const choice = chooseFrom(file, args, Env{}, E, TestKind::PRP);
    CHECK_EQ(specOf(choice), named.spec());
    if (choice) { CHECK_EQ(joined(choice->options), joined(CHEAP_OPTS)); }
  }
}

TEST(carry_long_costs_an_entry_the_reach_it_measured) {
  SelectionFile const file = twoEntries();

  // The entry was measured in a short-carry regime; -carry long runs the expanded carry kernels instead, whatever the
  // bits per word say, so what runs is not what was measured.
  Args args = configured({});
  args.carry = CARRY_64;

  auto const choice = chooseFrom(file, args, Env{}, 120'000'000, TestKind::PRP);
  CHECK(choice.has_value());
  if (choice) {
    CHECK(choice->entry->regime.longCarry == false);
    CHECK_EQ(choice->shadowed.size(), size_t{1});
    CHECK_EQ(choice->shadowed.at(0), std::string{"-carry"});
    CHECK_EQ(choice->reach, table());
  }

  // -carry short does not reach the kernels at all -- Gpu takes the 32-bit carry from the spec, not from this -- so it
  // is not a difference and must not cost the entry anything.
  Args shortCarry = configured({});
  shortCarry.carry = CARRY_32;
  auto const unaffected = chooseFrom(file, shortCarry, Env{}, 120'000'000, TestKind::PRP);
  CHECK(unaffected.has_value());
  if (unaffected) { CHECK(unaffected->shadowed.empty()); }
}

TEST(the_fallback_does_not_re_select_a_configuration_published_as_reaching_less) {
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production-reach";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  // One entry, published as stopping short of what the fitted table allows its shape -- a configuration this machine
  // measured and found wanting, which is exactly what the fallback must not hand back.
  FFTConfig const small{"256:2:256:202"};
  Interval const span = intervals(small, minExp(small), maxExp(small)).back();
  u64 const restricted = span.lo + (span.hi - span.lo) / 2;

  SelectionEntry entry{.id = {},
                       .cost = 100,
                       .fft = small.spec(),
                       .kind = TestKind::PRP,
                       .emin = span.lo,
                       .reach = restricted,
                       .regime = {},
                       .evidence = Evidence::Rejected,
                       .opts = {}};

  writeSelection("selection.txt", published({entry}));

  // The shape scan reads tune.txt first, so this is what makes it offer the restricted shape.
  { File::openWrite("tune.txt").printf("100.0 %s # %llu\n", small.spec().c_str(), (unsigned long long)maxExp(small)); }

  u64 const past = restricted + 1000;
  CHECK(past < maxExp(small));

  Args const args = configured({});
  CHECK(!chooseFrom(*readSelection("selection.txt"), args, Env{}, past, TestKind::PRP));

  // Above the published reach the entry is gone, and the answer is something else entirely rather than the same
  // configuration at the limit the fitted table would have allowed it.
  Choice const above = choose(args, Env{}, past, TestKind::PRP);
  CHECK(above.fft.spec() != small.spec());
  CHECK(past <= above.reach);

  // Below it, the entry answers as usual.
  Choice const below = choose(args, Env{}, restricted - 1000, TestKind::PRP);
  CHECK_EQ(below.fft.spec(), small.spec());

  // An -fft naming that very shape is an explicit choice, and keeps upstream's warn-and-run.
  Args pinned = configured({});
  pinned.fftSpec = small.spec();
  CHECK_EQ(choose(pinned, Env{}, past, TestKind::PRP).fft.spec(), small.spec());

  fs::remove_all(dir);
}

// A measured reach is a limit, not a hint: no exponent above it selects the entry, whatever else is published beside
// it, and below it the entry still answers.
TEST(an_entry_with_reduced_reach_is_never_selected_above_it) {
  FFTConfig const small{"512:15:512:212"};
  Interval const band = intervals(small, minExp(small), maxExp(small)).back();
  u64 const reduced = band.lo + (band.hi - band.lo) / 2;

  SelectionFile const file = published({SelectionEntry{.id = {},
                                                       .cost = 1700,
                                                       .fft = small.spec(),
                                                       .kind = TestKind::PRP,
                                                       .emin = band.lo,
                                                       .reach = reduced,
                                                       .regime = {},
                                                       .evidence = Evidence::Confirmed,
                                                       .opts = CHEAP_OPTS},
                                        SelectionEntry{.id = {},
                                                       .cost = 1900,
                                                       .fft = "1K:8:1K:202",
                                                       .kind = TestKind::PRP,
                                                       .emin = 100'000'000,
                                                       .reach = 160'000'000,
                                                       .regime = {},
                                                       .evidence = Evidence::Unvalidated,
                                                       .opts = DEAR_OPTS}});
  Args const args = configured({});

  for (u64 E = band.lo; E <= band.hi; E += (band.hi - band.lo) / 64) {
    std::optional<Choice> const choice = chooseFrom(file, args, Env{}, E, TestKind::PRP);
    if (E <= reduced) {
      CHECK_EQ(specOf(choice), small.spec());
    } else {
      CHECK(!choice || choice->fft.spec() != small.spec());
      CHECK(!choice || E <= choice->reach);
    }
  }
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, reduced + 1, TestKind::PRP)), std::string{"1K:8:1K:202"});
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, band.hi, TestKind::PRP)), std::string{"1K:8:1K:202"});
}

// The shape scan resolves its own options, which need not be any entry's; where they round as a published entry's do,
// that entry's limit is theirs too, and where they round otherwise it says nothing about them.
TEST(the_fallback_holds_options_that_round_alike_to_the_reach_published_for_them) {
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production-rounding";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  FFTConfig const small{"256:2:256:202"};
  Interval const span = intervals(small, minExp(small), maxExp(small)).back();
  u64 const restricted = span.lo + (span.hi - span.lo) / 2;
  u64 const past = restricted + 1000;
  { File::openWrite("tune.txt").printf("100.0 %s # %llu\n", small.spec().c_str(), (unsigned long long)maxExp(small)); }

  auto const entryUnder = [&](const UseConfig& opts) {
    return SelectionEntry{.id = {},
                          .cost = 100,
                          .fft = small.spec(),
                          .kind = TestKind::PRP,
                          .emin = span.lo,
                          .reach = restricted,
                          .regime = {},
                          .evidence = Evidence::Confirmed,
                          .opts = opts};
  };
  Args const args = configured({});

  // TAIL_KERNELS=3 splits the default's arithmetic another way and rounds exactly as it does.
  writeSelection("selection.txt", published({entryUnder({{"TAIL_KERNELS", "3"}})}));
  CHECK(choose(args, Env{}, past, TestKind::PRP).fft.spec() != small.spec());

  // TAIL_KERNELS=1 rounds otherwise, so its shortfall is not the defaults'.
  writeSelection("selection.txt", published({entryUnder({{"TAIL_KERNELS", "1"}})}));
  Choice const other = choose(args, Env{}, past, TestKind::PRP);
  CHECK_EQ(other.fft.spec(), small.spec());
  CHECK_EQ(other.reach, maxExp(small));

  fs::remove_all(dir);
}

// Two entries that round alike are one arithmetic, and the lower of their reaches is its limit: the one measured
// further up does not carry it past where the other was held, overridden or not.
TEST(an_entry_is_held_to_the_limit_published_for_its_arithmetic) {
  FFTConfig const small{"512:15:512:212"};
  Interval const band = intervals(small, minExp(small), maxExp(small)).back();
  u64 const reduced = band.lo + (band.hi - band.lo) / 2;

  auto const entryOf = [&](double cost, u64 reach, const UseConfig& opts) {
    return SelectionEntry{.id = {},
                          .cost = cost,
                          .fft = small.spec(),
                          .kind = TestKind::PRP,
                          .emin = band.lo,
                          .reach = reach,
                          .regime = {},
                          .evidence = Evidence::Confirmed,
                          .opts = opts};
  };

  // TAIL_KERNELS=3 splits the defaults' arithmetic another way and rounds exactly as it does.
  SelectionFile const file = published({entryOf(1700, band.hi, {{"TAIL_KERNELS", "3"}}), entryOf(1800, reduced, {})});
  Args const args = configured({});

  std::optional<Choice> const below = chooseFrom(file, args, Env{}, reduced, TestKind::PRP);
  CHECK(below.has_value());
  if (below) {
    CHECK_EQ(below->options.at("TAIL_KERNELS"), std::string{"3"});
    CHECK(below->shadowed.empty());
    CHECK_EQ(below->reach, reduced);
  }
  CHECK(!chooseFrom(file, args, Env{}, reduced + 100, TestKind::PRP));
  CHECK(!chooseFrom(file, args, Env{}, band.hi, TestKind::PRP));

  // An entry that rounds otherwise is not held by it.
  SelectionFile const apart = published({entryOf(1700, band.hi, {{"TAIL_KERNELS", "1"}}), entryOf(1800, reduced, {})});
  std::optional<Choice> const other = chooseFrom(apart, args, Env{}, band.hi, TestKind::PRP);
  CHECK(other.has_value());
  if (other) { CHECK_EQ(other->reach, band.hi); }
}

TEST(a_limit_holds_every_entry_of_its_arithmetic_as_an_entry_would) {
  FFTConfig const small{"512:15:512:212"};
  Interval const band = intervals(small, minExp(small), maxExp(small)).back();
  u64 const reduced = band.lo + (band.hi - band.lo) / 2;
  Args const args = configured({});

  auto const withLimit = [&](const UseConfig& rounding) {
    SelectionFile file = published({SelectionEntry{.id = {},
                                                   .cost = 1700,
                                                   .fft = small.spec(),
                                                   .kind = TestKind::PRP,
                                                   .emin = band.lo,
                                                   .reach = band.hi,
                                                   .regime = {},
                                                   .evidence = Evidence::Confirmed,
                                                   .opts = {{"TAIL_KERNELS", "3"}}}});
    file.limits = {
      Limit{.fft = small.spec(), .kind = TestKind::PRP, .regime = band.regime, .reach = reduced, .rounding = rounding}};
    CHECK(finalize(file));
    return file;
  };

  // The defaults' arithmetic, written as roundingOf() gives it and spelled with a key at a value that rounds as the
  // default does: either way, the entry that rounds as they do is held to the limit.
  for (const UseConfig& rounding : {UseConfig{}, UseConfig{{"TAIL_KERNELS", "3"}}}) {
    SelectionFile const file = withLimit(rounding);
    std::optional<Choice> const below = chooseFrom(file, args, Env{}, reduced, TestKind::PRP);
    CHECK(below.has_value());
    if (below) { CHECK_EQ(below->reach, reduced); }
    CHECK(!chooseFrom(file, args, Env{}, reduced + 100, TestKind::PRP));
    CHECK_EQ(publishedReach(file, Env{}, small, TestKind::PRP, {}, band.lo), reduced);
  }

  // A limit on another arithmetic does not hold it, but does hold that arithmetic wherever production lands on it.
  SelectionFile const apart = withLimit({{"TAIL_KERNELS", "1"}});
  std::optional<Choice> const other = chooseFrom(apart, args, Env{}, band.hi, TestKind::PRP);
  CHECK(other.has_value());
  if (other) { CHECK_EQ(other->reach, band.hi); }
  CHECK_EQ(publishedReach(apart, Env{}, small, TestKind::PRP, {{"TAIL_KERNELS", "1"}}, band.lo), reduced);
  CHECK(!chooseFrom(apart, configured({"-use TAIL_KERNELS=1"}), Env{}, reduced + 100, TestKind::PRP));
}

TEST(a_raise_is_held_by_what_alike_entries_measured_and_not_by_the_tables_reach) {
  FFTConfig const small{"512:15:512:212"};
  Interval const band = intervals(small, minExp(small), maxExp(small)).back();
  CHECK_EQ(band.hi, maxExp(small));
  u64 const raised = raiseCeiling(small);
  CHECK(raised > band.hi);
  u64 const reduced = band.lo + (band.hi - band.lo) / 2;

  auto const entryOf = [&](double cost, u64 reach, const UseConfig& opts) {
    return SelectionEntry{.id = {},
                          .cost = cost,
                          .fft = small.spec(),
                          .kind = TestKind::PRP,
                          .emin = band.lo,
                          .reach = reach,
                          .regime = band.regime,
                          .evidence = Evidence::Confirmed,
                          .opts = opts};
  };
  Args const args = configured({});

  // A cheaper set that rounds alike was read only at the table's top, which says nothing past it.
  SelectionFile const file = published({entryOf(1700, band.hi, {{"TAIL_KERNELS", "3"}}), entryOf(1800, raised, {})});
  std::optional<Choice> const above = chooseFrom(file, args, Env{}, raised, TestKind::PRP);
  CHECK(above.has_value());
  if (above) {
    CHECK(above->options.find("TAIL_KERNELS") == above->options.end());
    CHECK_EQ(above->reach, raised);
  }
  std::optional<Choice> const below = chooseFrom(file, args, Env{}, band.hi, TestKind::PRP);
  CHECK(below.has_value());
  if (below) { CHECK_EQ(below->options.at("TAIL_KERNELS"), std::string{"3"}); }

  // One that measured a lower limit holds the raise to it, as it holds every entry that rounds alike.
  SelectionFile const held = published({entryOf(1700, band.hi, {{"TAIL_KERNELS", "3"}}), entryOf(1800, raised, {}),
                                        entryOf(1900, reduced, {{"TAIL_KERNELS", "3"}, {"INPLACE", "1"}})});
  CHECK(!chooseFrom(held, args, Env{}, raised, TestKind::PRP));
  CHECK(!chooseFrom(held, args, Env{}, reduced + 100, TestKind::PRP));

  // And the shape scan, landing on the raised arithmetic, is told how far it was measured to.
  CHECK_EQ(publishedReach(file, Env{}, small, TestKind::PRP, {}, band.hi), raised);
  CHECK_EQ(publishedReach(held, Env{}, small, TestKind::PRP, {}, band.hi), reduced);
}

// An exclusion is of what the kernels are built from, so it is matched as they see it, and only where it applies.
TEST(an_exclusion_matches_the_build_not_the_spelling) {
  FFTConfig const fft{"1K:8:1K:202"};
  Regime const regime = regimeOf(fft, 120'000'000);
  Regime const other{.longCarry = !regime.longCarry, .carry64 = regime.carry64};
  Env const env{.isNvidia = true};

  SelectionFile const file = published(
    {}, {},
    {Exclusion{
      .fft = fft.spec(), .kind = TestKind::PRP, .regime = regime, .opts = {{"TAIL_KERNELS", "3"}, {"WMUL", "2"}}}});
  UseConfig const condemned{{"TAIL_KERNELS", "3"}};

  CHECK(exclusionFor(file, env, fft, TestKind::PRP, regime, condemned));  // WMUL=2 is its default here
  CHECK(exclusionFor(file, env, fft, TestKind::PRP, regime, {{"TAIL_KERNELS", "3"}, {"DEBUG", "1"}}));
  CHECK(!exclusionFor(file, env, fft, TestKind::PRP, regime, {}));
  CHECK(!exclusionFor(file, env, fft, TestKind::PRP, regime, {{"TAIL_KERNELS", "3"}, {"WMUL", "1"}}));
  CHECK(!exclusionFor(file, env, fft, TestKind::LL, regime, condemned));
  CHECK(!exclusionFor(file, env, fft, TestKind::PRP, other, condemned));
  CHECK(!exclusionFor(file, env, FFTConfig{"1K:8:1K:212"}, TestKind::PRP, regime, condemned));
}

// Emission publishes no excluded entry, but a user's setting can resolve an entry it did publish into one; the walk
// then moves on as it would past an entry that does not cover the exponent.
TEST(an_entry_the_users_settings_turn_into_an_excluded_configuration_is_passed_over) {
  u64 const E = 120'000'000;
  FFTConfig const cheap{"512:15:512:212"};
  Args const args = configured({"-use TAIL_KERNELS=2"});

  SelectionFile const plain = twoEntries();
  CHECK_EQ(specOf(chooseFrom(plain, args, Env{}, E, TestKind::PRP)), cheap.spec());

  SelectionFile const file =
    published(plain.entries, Defaults{.global = UseConfig{{"INPLACE", "1"}}, .family = {}},
              {Exclusion{.fft = cheap.spec(), .kind = TestKind::PRP, .regime = regimeOf(cheap, E), .opts = DEAR_OPTS}});
  CHECK_EQ(specOf(chooseFrom(file, args, Env{}, E, TestKind::PRP)), std::string{"1K:8:1K:202"});

  // As measured, the entry is not the excluded configuration, and still answers.
  CHECK_EQ(specOf(chooseFrom(file, configured({}), Env{}, E, TestKind::PRP)), cheap.spec());
}

// The whole sequence: a configuration published from a clean row, then condemned by a wrong answer at another exponent
// of its regime.  Its entry goes, and the fallback -- an older tune.txt that still lists the FFT, and the file's lines
// that resolve to the very options it ran -- must not hand it back.
TEST(the_fallback_does_not_re_select_a_configuration_that_answered_wrongly) {
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production-excluded";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  FFTConfig const small{"3:256:2:256:202"};
  Interval const band = intervals(small, minExp(small), maxExp(small)).back();
  u64 const clean = band.lo + (band.hi - band.lo) / 4;
  u64 const wrong = band.hi - (band.hi - band.lo) / 4;

  auto runRow = [&](u64 exponent, const char* status) {
    return "run   4 " + small.spec() + " prp " + std::to_string(exponent) + ' ' + band.regime.label() +
      " 1 100.000 1.000 16 4 1.0000 " + status + " 1753471274\n";
  };
  std::string const db =
    "# prpll tunedb v1\n"
    "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
    " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304\n"
    "cfg   1 -\n"
    "sess  4 env=1 start=1753471200 gen=0 anchor=-\n" +
    runRow(clean, "ok");
  Provenance const from{.ts = 1'753'471'500, .db = "tunedb.txt", .env = 1, .T = 0, .workloadLo = 0, .workloadHi = 0};

  auto publishFrom = [&](const std::string& text) {
    TuneDB loaded;
    CHECK(loaded.parse(text, "fixture"));
    auto const file = emit(loaded, {}, from);
    CHECK(file.has_value());
    writeSelection("selection.txt", file.value_or(SelectionFile{}));
  };

  Args const args = configured({});

  publishFrom(db);
  Choice const before = choose(args, Env{}, clean, TestKind::PRP);
  CHECK_EQ(before.fft.spec(), small.spec());
  CHECK(before.entry.has_value());

  publishFrom(db + runRow(wrong, "err"));
  CHECK(readSelection("selection.txt").value_or(SelectionFile{}).entries.empty());
  { File::openWrite("tune.txt").printf("100.0 %s # %llu\n", small.spec().c_str(), (unsigned long long)maxExp(small)); }

  Choice const after = choose(args, Env{}, clean, TestKind::PRP);
  CHECK(after.fft.spec() != small.spec());
  CHECK(isEligible(after.fft, clean));

  // Another configuration of the same FFT is not what answered wrongly.
  CHECK_EQ(choose(configured({}, {"-use TAIL_KERNELS=3"}), Env{}, clean, TestKind::PRP).fft.spec(), small.spec());

  // And -fft naming it is the user's explicit choice, which runs with a warning.
  Args pinned = configured({});
  pinned.fftSpec = small.spec();
  CHECK_EQ(choose(pinned, Env{}, clean, TestKind::PRP).fft.spec(), small.spec());

  fs::remove_all(dir);
}

// Where the only thing the scan offers is excluded, the task fails rather than run it.
TEST(an_excluded_configuration_with_nothing_past_it_is_not_run) {
  fs::path const dir = fs::temp_directory_path() / "prpll-test-production-excluded-top";
  fs::remove_all(dir);
  fs::create_directories(dir);
  WorkingDirectory const here{dir};

  Args const args = configured({});
  FFTConfig top = FFTConfig::bestFit(args, 1'000'000'000, "", true);
  for (bool larger = true; larger;) {
    try {
      top = FFTConfig::bestFit(args, maxExp(top) + 1, "", true);
    } catch (...) { larger = false; }
  }
  u64 const E = maxExp(top) - 1000;

  writeSelection(
    "selection.txt",
    published({}, {}, {Exclusion{.fft = top.spec(), .kind = TestKind::PRP, .regime = regimeOf(top, E), .opts = {}}}));

  bool threw = false;
  try {
    (void)choose(args, Env{}, E, TestKind::PRP);
  } catch (...) { threw = true; }
  CHECK(threw);

  // The same exponent under another configuration runs.
  CHECK_EQ(choose(configured({}, {"-use TAIL_KERNELS=3"}), Env{}, E, TestKind::PRP).fft.spec(), top.spec());

  fs::remove_all(dir);
}
