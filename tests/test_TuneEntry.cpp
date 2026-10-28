// Copyright (C) Jason Lynch

// Production's reading of the tune files: the reach a line names, the kind, regime and range a tuner's line names,
// and the lines an LL test chooses from.

#include "test.h"

#include "Args.h"
#include "BuildId.h"
#include "TuneEntry.h"

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

// A directory of its own, made current for the test: the tune files are read from where the run is.
struct Here {
  fs::path path;
  fs::path before;

  explicit Here(const char* name) : path{fs::temp_directory_path() / name}, before{fs::current_path()} {
    fs::remove_all(path);
    fs::create_directories(path);
    fs::current_path(path);
  }

  ~Here() {
    fs::current_path(before);
    fs::remove_all(path);
  }

  Here(const Here&) = delete;
  Here& operator=(const Here&) = delete;

  void write(const char* name, const std::string& text) const {
    FILE* const f = fopen((path / name).string().c_str(), "wb");
    CHECK(f != nullptr);
    CHECK(fwrite(text.data(), 1, text.size(), f) == text.size());
    CHECK(fclose(f) == 0);
  }
};

}  // namespace

TEST(a_line_lowers_an_ffts_reach_but_never_raises_it) {
  Here const here{"prpll-test-tuneentry-reach"};
  u64 const table = FFTConfig{"2:512:4:512:212"}.maxExp();
  here.write("tune.txt",
             "151.1 2:512:4:512:212 # 67004008, WMUL=1\n"
             "  300 1K:13:256:202 # 999999999999\n");
  std::vector<TuneEntry> const entries = TuneEntry::readTuneFile(Args{true});
  CHECK_EQ(entries.size(), size_t{2});
  CHECK_EQ(entries[0].maxExp(), u64{67'004'008});
  CHECK(entries[0].maxExp() < table);
  CHECK_EQ(entries[0].uses.size(), size_t{1});
  CHECK_EQ(entries[1].maxExp(), FFTConfig{"1K:13:256:202"}.maxExp());
}

TEST(an_ll_test_reads_its_own_tune_file_where_there_is_one) {
  Here const here{"prpll-test-tuneentry-ll"};
  here.write("tune.txt", "100.0 1K:13:256:202 # 1\n");
  CHECK(TuneEntry::readTuneFile(Args{true}, true).front().fft.spec() == "1K:13:256:202");

  here.write("tune-ll.txt", "120.0 512:15:512:202 # 1\n");
  CHECK(TuneEntry::readTuneFile(Args{true}, true).front().fft.spec() == "512:15:512:202");
  CHECK(TuneEntry::readTuneFile(Args{true}, false).front().fft.spec() == "1K:13:256:202");
}

namespace {

std::string tunerFile(const std::string& lines) {
  char head[96];
  snprintf(head, sizeof(head), "# %s, build %016" PRIx64 "\n# written by a test\n", TUNE_FILE_MAGIC,
           tune::buildFingerprint());
  return head + lines;
}

}  // namespace

TEST(a_tuner_line_names_its_kind_regime_and_range) {
  Here const here{"prpll-test-tuneentry-v2"};
  here.write("tune.txt", tunerFile(" 246.0 2:512:4:512:202 ll short64 40265781-72729232 # LOADS=5502, MULTI_Q=1\n"
                                   " 260.0 2:512:4:512:202 prp short64 40265781-72000000\n"));
  CHECK(TuneEntry::tunerFile(Args{true}));
  std::vector<TuneEntry> const prp = TuneEntry::tunedLines(Args{true});
  CHECK_EQ(prp.size(), size_t{1});
  if (prp.empty()) { return; }
  CHECK_EQ(prp[0].kind, std::string{"prp"});
  CHECK_EQ(prp[0].regime, std::string{"short64"});
  CHECK_EQ(prp[0].minExp, u64{40'265'781});
  CHECK_EQ(prp[0].maxExp(), u64{72'000'000});
  CHECK(prp[0].uses.empty());

  std::vector<TuneEntry> const ll = TuneEntry::tunedLines(Args{true}, true);
  CHECK_EQ(ll.size(), size_t{2});
  if (ll.size() != 2) { return; }
  CHECK(!ll[0].fallback && ll[0].kind == "ll" && ll[0].uses.size() == 2);
  CHECK(ll[1].fallback && ll[1].kind == "prp");
}

TEST(production_takes_the_cheapest_line_holding_E_in_its_regime) {
  Here const here{"prpll-test-tuneentry-pick"};
  // 3:1K:8:256:202 runs short32 from 48234494 to 78434408, short64 above.
  here.write("tune.txt", tunerFile(" 900.0 3:1K:8:256:202 prp short32 48234494-78434408\n"
                                   " 950.0 3:1K:8:256:202 prp short64 78434409-99992208 # WMUL=1\n"
                                   " 920.0 3:1K:8:256:202 prp short64 60000000-99992208\n"));
  std::vector<TuneEntry> const lines = TuneEntry::tunedLines(Args{true});
  auto pick = [&](u64 E) { return TuneEntry::pick(lines, E, 1.0, true, false); };
  CHECK(pick(50'000'000) && pick(50'000'000)->cost == 900);
  CHECK(pick(80'000'000) && pick(80'000'000)->cost == 920);
  CHECK(pick(30'000'000) == nullptr);  // below every range: the untuned choice
  // The 920 line's range holds 70M, but 70M runs short32, not the short64 it names.
  CHECK(pick(70'000'000) && pick(70'000'000)->cost == 900);
}

TEST(an_ll_test_falls_back_to_prp_lines_and_ignores_tune_ll_beside_a_tuner_file) {
  Here const here{"prpll-test-tuneentry-fallback"};
  here.write("tune-ll.txt", "120.0 512:15:512:202 # 1\n");
  here.write("tune.txt", tunerFile(" 900.0 3:1K:8:256:202 prp short32 48234494-78434408\n"));
  std::vector<TuneEntry> const lines = TuneEntry::tunedLines(Args{true}, true);
  CHECK_EQ(lines.size(), size_t{1});
  if (lines.empty()) { return; }
  CHECK(lines[0].fallback);
  CHECK(TuneEntry::pick(lines, 50'000'000, 1.0, true, false) == &lines[0]);
}

TEST(a_tuner_line_with_words_it_cannot_read_is_ignored) {
  Here const here{"prpll-test-tuneentry-bad"};
  here.write("tune.txt", tunerFile(" 900.0 3:1K:8:256:202 prp sideways32 48234494-78434408\n"
                                   " 910.0 3:1K:8:256:202 prp short32 48234494-78434408\n"));
  std::vector<TuneEntry> const lines = TuneEntry::tunedLines(Args{true});
  CHECK_EQ(lines.size(), size_t{1});
}

TEST(the_settings_of_an_fft_given_with_fft_are_of_the_line_holding_E) {
  Here const here{"prpll-test-tuneentry-uses"};
  here.write("tune.txt", tunerFile(" 900.0 3:1K:8:256:202 prp short32 48234494-78434408 # WMUL=2\n"
                                   " 950.0 3:1K:8:256:202 prp short64 78434409-99992208 # WMUL=1\n"));
  FFTConfig const fft{"3:1K:8:256:202"};
  CHECK(TuneEntry::usesFor(Args{true}, fft, 90'000'000) == (vector<KeyVal>{{"WMUL", "1"}}));
  CHECK(TuneEntry::usesFor(Args{true}, fft, 50'000'000) == (vector<KeyVal>{{"WMUL", "2"}}));
  CHECK(TuneEntry::usesFor(Args{true}, fft, 10'000'000) == (vector<KeyVal>{{"WMUL", "2"}}));
}
