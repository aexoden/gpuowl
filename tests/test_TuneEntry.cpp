// Copyright (C) Jason Lynch

// Production's reading of the tune files the tuner writes: the reach a line names, and the file an LL test reads.

#include "test.h"

#include "Args.h"
#include "TuneEntry.h"

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
