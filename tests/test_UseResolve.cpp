// Copyright (C) Jason Lynch

// GPU-free tests of -use resolution (src/UseResolve.cpp): the '!' selectors, the precedence between sources, the merge
// into a Gpu's own Args, and the takeover for a tuning run.

#include "test.h"

#include "Args.h"
#include "UseResolve.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace tune;

namespace {

bool refused(auto fn) {
  try {
    fn();
  } catch (const char*) { return true; }
  return false;
}

Args configured(const std::vector<std::string>& configLines, const std::string& commandLine = "") {
  Args args{true};
  for (const std::string& line : configLines) { args.parse(line, true); }
  args.parse(commandLine);
  return args;
}

std::string valueOf(const UseConfig& config, const std::string& key) {
  auto it = config.find(key);
  return it == config.end() ? "-" : it->second;
}

std::string joined(const UseConfig& config) {
  std::string s;
  for (const auto& [k, v] : config) { s += (s.empty() ? "" : ",") + k + '=' + v; }
  return s;
}

std::string joined(const std::vector<std::string>& keys) {
  std::string s;
  for (const std::string& k : keys) { s += (s.empty() ? "" : ",") + k; }
  return s;
}

} // namespace

TEST(selector_parse) {
  struct Row {
    const char* text;
    const char* spec;
    u32 rank;
  };

  const Row rows[] = {
    {"ll", "ll", 5},
    {"prp", "prp", 5},
    {"0", "0", 1},
    {"1", "1", 1},
    {"53", "53", 1},
    {"512:15:512", "512:15:512", 2},
    {"0:512:15:512", "512:15:512", 2},
    {"1024:15:1024", "1K:15:1K", 2},
    {"1k:15:1K", "1K:15:1K", 2},
    {"1:512:8:512", "1:512:8:512", 2},
    {"512:15:512:101", "512:15:512:101", 3},
    {"512:15:512:012", "512:15:512:012", 3},
    {"256:2:256:101:0", "256:2:256:101:0", 4},
    {"ll:512:15:512", "ll:512:15:512", 7},
    {"prp:1", "prp:1", 6},
    {"ll:1:1K:8:256:202:1", "ll:1:1K:8:256:202", 9},  // a hybrid's carry reaches no kernel
    {"1:512:8:512:212", "1:512:8:512:202", 3},
    {"1K:8:1K:111", "1K:8:1K:212", 3},
    {"256:2:256:202:0", "256:2:256:202:0", 4},
  };

  for (const Row& row : rows) {
    FFTSelector const sel = FFTSelector::parse(row.text);
    CHECK_EQ(sel.spec(), std::string{row.spec});
    CHECK_EQ(sel.rank(), row.rank);
  }

  for (const char* text : {"",
                           ":512:15:512",
                           "prd:512:15:512",
                           "ll:",
                           "cert",
                           "512",
                           "7",
                           "5x",
                           "512:15",
                           "512:15:512:101:0:9",
                           "511:15:512",
                           "512:1:512",
                           "512:17:512",
                           "512:15:300",
                           "1:512:15:512",
                           "51x:15:512",
                           "512:15:512:301",
                           "512:15:512:12",
                           "512:15:512:1010",
                           "512:15:512:101:2",
                           "512:15:512:101:auto",
                           "ll:ll"}) {
    if (!refused([text] { (void)FFTSelector::parse(text); })) { testing::fail(__FILE__, __LINE__, text); }
  }
}

TEST(selector_matches) {
  struct Row {
    const char* selector;
    const char* fft;
    TestKind kind;
    bool matches;
  };

  const Row rows[] = {
    {"512:15:512", "512:15:512:101", TestKind::PRP, true},
    {"512:15:512", "512:15:512:202:0", TestKind::LL, true},
    {"512:15:512", "512:14:512:202", TestKind::PRP, false},
    {"512:15:512", "1:512:8:512:202", TestKind::PRP, false},
    {"1K:8:256", "1024:8:256:202", TestKind::PRP, true},
    {"4:1K:8:256", "4:1K:8:256:202", TestKind::PRP, true},
    {"0", "1K:15:1K:202", TestKind::PRP, true},
    {"0", "1:512:8:512:202", TestKind::PRP, false},
    {"1", "1:512:8:512:202", TestKind::LL, true},
    {"512:15:512:101", "512:15:512:101", TestKind::PRP, true},
    {"512:15:512:101", "512:15:512:202", TestKind::PRP, false},
    {"256:2:256:101:0", "256:2:256:101:0", TestKind::PRP, true},
    {"256:2:256:101:0", "256:2:256:101", TestKind::PRP, false},
    {"256:2:256:101:1", "256:2:256:101:0", TestKind::PRP, false},
    {"256:2:256:101:1", "256:2:256:101", TestKind::PRP, false},
    {"256:2:256:101:0", "256:2:256:101:1", TestKind::PRP, false},
    {"1:512:8:512:202:0", "1:512:8:512:202", TestKind::PRP, true},
    {"3:1K:8:256:202:0", "3:1K:8:256:202", TestKind::PRP, false},
    {"3:1K:8:256:202:0", "3:1K:8:256:202:0", TestKind::PRP, true},
    {"1K:8:1K:101", "1K:8:1K:202", TestKind::PRP, true},
    {"1K:8:1K:202", "1K:8:1K:101", TestKind::PRP, true},
    {"1K:8:1K:202", "1K:8:1K:212", TestKind::PRP, false},
    {"1:512:8:512:212", "1:512:8:512", TestKind::PRP, true},
    {"1K:5:256:202:0", "1K:5:256:202", TestKind::PRP, true},
    {"1K:5:256:202:0", "1K:5:256:202:1", TestKind::PRP, false},
    {"ll", "1:512:8:512:202", TestKind::LL, true},
    {"ll", "1:512:8:512:202", TestKind::PRP, false},
    {"prp:1:512:8:512", "1:512:8:512:202", TestKind::PRP, true},
    {"prp:1:512:8:512", "1:512:8:512:202", TestKind::LL, false},
  };

  for (const Row& row : rows) {
    if (FFTSelector::parse(row.selector).matches(FFTConfig{row.fft}, row.kind) != row.matches) {
      testing::fail(__FILE__, __LINE__,
                    std::string{row.selector} + " vs " + row.fft + " " + toString(row.kind) + ": expected " +
                      (row.matches ? "a match" : "no match"));
    }
  }
}

TEST(use_line_parse) {
  UseLine const line = parseUseLine("! 512:15:512 PAD=256,WMUL # 1234");
  CHECK_EQ(line.selector.spec(), std::string{"512:15:512"});
  CHECK_EQ(line.uses.size(), size_t{2});
  CHECK_EQ(line.uses[1].first, std::string{"WMUL"});
  CHECK_EQ(line.uses[1].second, std::string{"1"});
  CHECK_EQ(parseUseLine("!\tll:0\tSTATS=1\r").uses.size(), size_t{1});
  CHECK_EQ(parseUseLine("!0 PAD=0").selector.spec(), std::string{"0"});

  std::string uses;
  for (int i = 0; i < 40; ++i) { uses += "LONG_KEY_NUMBER_" + std::to_string(i) + "=140,"; }
  uses += "LOADS=140";
  UseLine const longLine = parseUseLine("! 512:15:512 " + uses);
  CHECK_EQ(longLine.uses.size(), size_t{41});
  CHECK_EQ(longLine.uses.back().second, std::string{"140"});

  for (const char* text : {"! 512:15:512", "!", "! PAD=1", "! 512:15:512 PAD=1 WMUL=2", "! 512:15:512 =1",
                           "! 512 PAD=1", "512:15:512 PAD=1", "! 512:15:512 PAD=1 junk # comment"}) {
    if (!refused([text] { (void)parseUseLine(text); })) { testing::fail(__FILE__, __LINE__, text); }
  }

  Args args{true};
  CHECK(refused([&args] { args.parse("! 512:15 PAD=1", true); }));
}

TEST(precedence) {
  struct Row {
    const char* name;
    std::vector<std::string> config;
    std::string commandLine;
    SelectionLayers selection;
    const char* fft;
    TestKind kind;
    const char* expected;  // X, or "-" when unset
  };
  UseLine const family = parseUseLine("! 0 X=family");
  UseLine const otherFamily = parseUseLine("! 1 X=other");
  std::vector<Row> const rows = {
    {"nothing set", {}, "", {}, "512:15:512:101", TestKind::PRP, "-"},
    {"selection global", {}, "", {{{"X", "global"}}, {}, {}}, "512:15:512:101", TestKind::PRP, "global"},
    {"family beats global", {}, "", {{{"X", "global"}}, {family}, {}}, "512:15:512:101", TestKind::PRP, "family"},
    {"family of another type",
     {},
     "",
     {{{"X", "global"}}, {otherFamily}, {}},
     "512:15:512:101",
     TestKind::PRP,
     "global"},
    {"entry beats family", {}, "", {{}, {family}, {{"X", "entry"}}}, "512:15:512:101", TestKind::PRP, "entry"},
    {"config -use beats entry",
     {"-use X=config"},
     "",
     {{}, {family}, {{"X", "entry"}}},
     "512:15:512:101",
     TestKind::PRP,
     "config"},
    {"config ! beats config -use", {"! 0 X=type", "-use X=config"}, "", {}, "512:15:512:101", TestKind::PRP, "type"},
    {"shape beats type, read later",
     {"! 512:15:512 X=shape", "! 0 X=type"},
     "",
     {},
     "512:15:512:101",
     TestKind::PRP,
     "shape"},
    {"variant beats shape",
     {"! 512:15:512:101 X=variant", "! 512:15:512 X=shape"},
     "",
     {},
     "512:15:512:101",
     TestKind::PRP,
     "variant"},
    {"carry selector needs that carry",
     {"! 256:2:256:101 X=variant", "! 256:2:256:101:0 X=carry"},
     "",
     {},
     "256:2:256:101",
     TestKind::PRP,
     "variant"},
    {"carry beats variant",
     {"! 256:2:256:101:0 X=carry", "! 256:2:256:101 X=variant"},
     "",
     {},
     "256:2:256:101:0",
     TestKind::PRP,
     "carry"},
    {"kind alone beats unqualified carry",
     {"! prp X=kind", "! 256:2:256:101:0 X=carry"},
     "",
     {},
     "256:2:256:101:0",
     TestKind::PRP,
     "kind"},
    {"kind shape beats kind alone",
     {"! prp:512:15:512 X=kindshape", "! prp X=kind"},
     "",
     {},
     "512:15:512:101",
     TestKind::PRP,
     "kindshape"},
    {"ll line under PRP", {"! 0 X=type", "! ll X=ll"}, "", {}, "512:15:512:101", TestKind::PRP, "type"},
    {"ll line under LL", {"! 0 X=type", "! ll X=ll"}, "", {}, "512:15:512:101", TestKind::LL, "ll"},
    {"same rank, last read wins", {"! 0 X=first", "! 0 X=second"}, "", {}, "512:15:512:101", TestKind::PRP, "second"},
    {"same rank, reversed", {"! 0 X=second", "! 0 X=first"}, "", {}, "512:15:512:101", TestKind::PRP, "first"},
    {"1024 and 1K are one shape",
     {"! 1K:15:1K X=shape", "! 1024:15:1024 X=spelled"},
     "",
     {},
     "1K:15:1K:202",
     TestKind::PRP,
     "spelled"},
    {"command line beats every ! line",
     {"! prp:256:2:256:101:0 X=best", "-use X=config"},
     "-use X=cli",
     {},
     "256:2:256:101:0",
     TestKind::PRP,
     "cli"},
    {"another command-line key leaves X alone",
     {"! 0 X=type", "-use X=config"},
     "-use Y=cli",
     {},
     "512:15:512:101",
     TestKind::PRP,
     "type"},
  };
  for (const Row& row : rows) {
    Args const args = configured(row.config, row.commandLine);
    std::string const got = valueOf(resolveConfig(args, FFTConfig{row.fft}, row.kind, row.selection), "X");
    if (got != row.expected) {
      testing::fail(__FILE__, __LINE__, std::string{row.name} + ": got " + got + ", expected " + row.expected);
    }
  }

  Args const merged = configured({"! 512:15:512 A=1", "! 512:15:512 B=2"});
  CHECK_EQ(joined(resolveConfig(merged, FFTConfig{"512:15:512:101"}, TestKind::PRP)), std::string{"A=1,B=2"});
}

TEST(read_config_is_not_command_line) {
  auto const path = std::filesystem::temp_directory_path() / "prpll-test-UseResolve-config.txt";
  {
    std::ofstream out{path};
    out << "-use X=config\n! 0 X=type\n";
  }
  FFTConfig const fft{"512:15:512:101"};

  Args fileOnly{true};
  fileOnly.readConfig(path);
  CHECK(fileOnly.cliKeys.empty());
  CHECK_EQ(valueOf(resolveConfig(fileOnly, fft, TestKind::PRP), "X"), std::string{"type"});

  Args withCli{true};
  withCli.readConfig(path);
  withCli.parse("-use X=cli");
  CHECK_EQ(valueOf(resolveConfig(withCli, fft, TestKind::PRP), "X"), std::string{"cli"});

  std::filesystem::remove(path);
}

TEST(resolve_into_gpu_args) {
  Args args =
    configured({"-use L2_STRIPING=0,PAD=0", "! 512:15:512 L2_STRIPING=8,INPLACE=1", "! 1 PAD=256"}, "-use MULTI_Q=1");
  FFTConfig const fft{"512:15:512:101"};

  resolveInto(args, fft, TestKind::PRP, {{"MULTI_Q", "0"}, {"WMUL", "1"}});
  CHECK_EQ(joined(args.flags), std::string{"INPLACE=1,L2_STRIPING=8,MULTI_Q=0,PAD=0,WMUL=1"});
  CHECK(args.perFftConfig.empty());
  CHECK_EQ(args.value("L2_STRIPING", 0), 8);

  // A second resolution, as a Gpu built from a Gpu's own Args would do, changes nothing.
  UseConfig const once = args.flags;
  resolveInto(args, fft, TestKind::LL, {});
  CHECK_EQ(joined(args.flags), joined(once));
}

TEST(takeover_lists_ignored_keys) {
  Args args = configured({"-use PAD=256,STATS=1", "! 512:15:512 WMUL=1", "! ll L2_STRIPING=8"},
                         "-use NO_ASM=1,INPLACE=1,NOT_A_KEY=2,DEBUG,BIGLIT=0");
  Takeover const takeover = takeOverConfig(args);

  CHECK_EQ(joined(takeover.configKeys), std::string{"L2_STRIPING,PAD,STATS,WMUL"});
  CHECK_EQ(joined(takeover.commandLineKeys), std::string{"INPLACE,NOT_A_KEY"});
  CHECK_EQ(describe(takeover),
           std::string{"Tuning ignores these -use settings; from the config files: L2_STRIPING, "
                       "PAD, STATS, WMUL; from the command line: INPLACE, NOT_A_KEY"});

  CHECK_EQ(joined(args.flags), std::string{"BIGLIT=0,DEBUG=1,NO_ASM=1"});
  CHECK(args.perFftConfig.empty());
  CHECK_EQ(joined(resolveConfig(args, FFTConfig{"1:512:8:512:202"}, TestKind::LL)),
           std::string{"BIGLIT=0,DEBUG=1,NO_ASM=1"});

  Args clean{true};
  CHECK(takeOverConfig(clean).configKeys.empty());
  CHECK_EQ(describe(takeOverConfig(clean)), std::string{"Tuning from built-in defaults; no -use settings to ignore"});
}
