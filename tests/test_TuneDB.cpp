// Copyright (C) Jason Lynch

// Tests that a hand-written file round-trips, that a malformed row of a known kind refuses the whole file, and that a
// row kind -- or an env field -- this build does not know survives a rewrite.

#include "TuneDB.h"

#include "fs.h"

#include "BuildId.h"
#include "FFTConfig.h"
#include "FFTVariants.h"
#include "File.h"
#include "test.h"

#include <atomic>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace tune;

namespace {

// A file with one of every row kind, in the order the writer emits them, plus a row kind and an env field from a build
// that does not exist yet.
const char* const FIXTURE =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 fp64=1 builtins=1 machine=01:00.0 build=9a3f21c0d1e2f304 fanspeed=42 \"note=two words\"\n"
  "cfg   1 -\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073 tune=workload=100000000-400000000,stop=0.1%\n"
  "sess  5 env=1 start=1753481200 gen=1 anchor=- alarmed=1\n"
  "work  4 prp 124647911 3\n"
  "work  4 ll 286472227 1\n"
  "run   4 512:15:512:212 prp 143400073 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 ll 143400073 short32 1 1801.000 3.000 8 2 0.9980 err 1753471300\n"
  "nogo  4 512:15:512:212 SHUFL_BYTES_W=16 1753471260\n"
  "roe   4 512:15:512:212 143400073 17 29.40 118 0.371094 ok - 1753471402\n"
  "anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410\n"
  "anchor 5 512:15:512:212 143400073 1 1792.000 1.0100 1753481410\n"
  "ref   5 512:15:512:212 1000151 2000 171f3662c332472f 1753471480\n"
  "jump  4 512:15:512:212 prp short32 17 3 1753471490\n"
  "combo 4 512:15:512:212 prp short32 17 2 1753471495\n"
  "boot  4 512:15:512:212 143400073 1753471497\n"
  "round 4 1 0 0 1753471498\n"
  "round 4 2 1 16 1753471499 512:15:512:212 prp short32 6 1K:8:1K:101 ll long32 0\n"
  "lines 4 1 2 1753471499 1 3 17\n"
  "try   4 512:15:512:212 prp 143400073 17 1753471250\n"
  "try   5 512:15:512:212 prp 143400073 1 1753481250\n"
  "done  4 1753471500\n"
  "hint  4 512:15:512:212 from-a-later-build 7\n"
  "milestone\n"
  "# a note the user added by hand\n";

TuneDB loaded(const std::string& text) {
  TuneDB db;
  CHECK(db.parse(text, "fixture"));
  return db;
}

// The fixture with one row replaced, for the rejection table below.
std::string withRow(const std::string& original, const std::string& replacement) {
  std::string text = FIXTURE;
  auto const at = text.find(original);
  CHECK(at != std::string::npos);
  return text.replace(at, original.size(), replacement);
}

void rejects(const std::string& original, const std::string& replacement) {
  TuneDB db;
  CHECK(!db.parse(withRow(original, replacement), "fixture"));
  // A refused load leaves nothing behind: a caller that ignores the return value cannot write a half-read file back.
  CHECK(db.envs().empty());
  CHECK(db.runs().empty());
}

}  // namespace

TEST(round_trips) {
  TuneDB const db = loaded(FIXTURE);
  CHECK_EQ(db.text(), std::string{FIXTURE});
}

TEST(rows_are_read) {
  TuneDB const db = loaded(FIXTURE);

  CHECK_EQ(db.envs().size(), size_t{1});
  DbEnv const& env = db.envs().at(0);
  CHECK_EQ(env.gpu, std::string{"NVIDIA RTX A4000"});
  CHECK(env.isNvidia && !env.isAmd && !env.cudaBackend && !env.noAsm && !env.pdlLaunch);
  CHECK(env.hasFP64 && env.amdBuiltins);
  CHECK_EQ(env.computeCapability, 806u);
  CHECK_EQ(env.machine, std::string{"01:00.0"});
  CHECK_EQ(env.build, u64{0x9a3f21c0d1e2f304});

  // What the emitter needs on a machine with no device.
  Env const resolved = env.toEnv();
  CHECK(resolved.isNvidia && !resolved.cudaBackend);
  CHECK_EQ(resolved.computeCapability, 806u);
  CHECK(resolved.hasPtx(500));

  CHECK_EQ(db.sessions().size(), size_t{2});
  CHECK(!db.sessions().at(0).alarmed);
  CHECK(db.sessions().at(1).alarmed);
  CHECK_EQ(db.sessions().at(1).gen, 1u);
  CHECK_EQ(db.sessions().at(1).anchor, std::string{});
  CHECK_EQ(db.sessions().at(0).tune, std::string{"workload=100000000-400000000,stop=0.1%"});
  CHECK_EQ(db.sessions().at(1).tune, std::string{});

  CHECK_EQ(db.works().size(), size_t{2});
  CHECK(db.works().at(0).sess == 4 && db.works().at(0).kind == TestKind::PRP);
  CHECK_EQ(db.works().at(0).exponent, u64(124'647'911));
  CHECK_EQ(db.works().at(0).count, 3u);
  CHECK(db.works().at(1).kind == TestKind::LL);

  CHECK_EQ(db.cfgs().size(), size_t{2});
  CHECK(db.findCfg(1)->empty());
  CHECK_EQ(db.findCfg(17)->at("TAIL_KERNELS"), std::string{"3"});

  CHECK_EQ(db.runs().size(), size_t{2});
  RunRow const& run = db.runs().at(0);
  CHECK_EQ(run.fft, std::string{"512:15:512:212"});
  CHECK(run.kind == TestKind::PRP);
  CHECK_EQ(run.exponent, u64{143'400'073});
  CHECK(!run.regime.longCarry && !run.regime.carry64);
  CHECK_EQ(run.m.mean, 1774.23);
  CHECK_EQ(run.m.blocks, 24u);
  CHECK_EQ(run.m.calls, 6u);
  CHECK(run.m.status == Status::Ok);
  CHECK(db.runs().at(1).kind == TestKind::LL);
  CHECK(db.runs().at(1).m.status == Status::Err);

  CHECK_EQ(db.tries().size(), size_t{2});
  // Session 4's attempt was answered by the rows that followed it; session 5's was not.
  std::vector<TryRow> const standing = db.diedHolding();
  CHECK_EQ(standing.size(), size_t{1});
  CHECK_EQ(standing.at(0).sess, 5u);
  CHECK(db.diedOn(1, 1, TestKind::PRP, "512:15:512:212", 143'400'073));
  CHECK(!db.diedOn(1, 17, TestKind::PRP, "512:15:512:212", 143'400'073));

  CHECK_EQ(db.nogos().size(), size_t{1});
  CHECK_EQ(db.nogos().at(0).key, std::string{"SHUFL_BYTES_W"});
  CHECK_EQ(db.nogos().at(0).val, std::string{"16"});

  CHECK_EQ(db.roes().size(), size_t{1});
  CHECK_EQ(db.roes().at(0).z, 29.40);
  CHECK_EQ(db.roes().at(0).n, 118u);
  CHECK(db.roes().at(0).checkOk);

  CHECK_EQ(db.anchors().size(), size_t{2});
  CHECK_EQ(db.anchors().at(0).mean, 1774.23);
  CHECK_EQ(db.anchors().at(0).ratio, 1.0);
  CHECK_EQ(db.anchors().at(1).ratio, 1.01);

  // The env is pinned to the anchor its earliest session named, and its baseline is the first reading of it.
  CHECK_EQ(db.envAnchor(1), std::string{"512:15:512:212@143400073"});
  CHECK_EQ(db.envAnchor(2), std::string{});
  CHECK(db.envBaseline(1) == &db.anchors().at(0));
  CHECK(db.envBaseline(2) == nullptr);

  CHECK_EQ(db.refs().size(), size_t{1});
  CHECK_EQ(db.refs().at(0).res64, u64{0x171f3662c332472f});
  CHECK_EQ(db.refs().at(0).iters, u64{2000});

  CHECK_EQ(db.jumps().size(), size_t{1});
  CHECK_EQ(db.jumps().at(0).cfg, 17u);
  CHECK_EQ(db.jumps().at(0).k, 3u);
  CHECK(db.jumps().at(0).regime.label() == "short32");

  CHECK_EQ(db.combos().size(), size_t{1});
  CHECK_EQ(db.combos().at(0).cfg, 17u);
  CHECK_EQ(db.combos().at(0).tier, 2u);
  CHECK(db.combos().at(0).regime.label() == "short32");

  CHECK_EQ(db.boots().size(), size_t{1});
  CHECK_EQ(db.boots().at(0).fft, std::string{"512:15:512:212"});
  CHECK_EQ(db.boots().at(0).probe, u64{143'400'073});
  CHECK_EQ(db.boots().at(0).sess, 4u);

  CHECK_EQ(db.rounds().size(), size_t{2});
  CHECK(db.rounds().at(0).members.empty());
  CHECK_EQ(db.rounds().at(0).round, 0u);
  const RoundRow& round = db.rounds().at(1);
  CHECK_EQ(round.n, 2u);
  CHECK_EQ(round.round, 1u);
  CHECK_EQ(round.calls, u64{16});
  CHECK(round.members ==
        (std::vector<RoundMember>{
          {.fft = "512:15:512:212", .kind = TestKind::PRP, .regime = *parseRegime("short32"), .from = 6},
          {.fft = "1K:8:1K:101", .kind = TestKind::LL, .regime = *parseRegime("long32"), .from = 0}}));

  CHECK_EQ(db.lines().size(), size_t{1});
  const LinesRow& lines = db.lines().at(0);
  CHECK_EQ(lines.sess, 4u);
  CHECK_EQ(lines.n, 1u);
  CHECK_EQ(lines.after, 2u);
  CHECK_EQ(lines.global, 1u);
  std::vector<std::pair<enum FFT_TYPES, u32>> const family{{FFT61, 17}};
  CHECK(lines.family == family);
}

TEST(unknown_rows_pass_through) {
  TuneDB const db = loaded(FIXTURE);
  CHECK_EQ(db.unknownRows().size(), size_t{3});
  CHECK_EQ(db.unknownRows().at(0), std::string{"hint  4 512:15:512:212 from-a-later-build 7"});
  // A row kind of one word: the tag decides, not the field count, or a later build's simplest row is refused.
  CHECK_EQ(db.unknownRows().at(1), std::string{"milestone"});
  // A comment a user wrote survives a rewrite too.
  CHECK_EQ(db.unknownRows().at(2), std::string{"# a note the user added by hand"});

  // And env fields from the same later build, including one that has to be re-quoted to survive.
  CHECK_EQ(db.envs().at(0).extra.size(), size_t{2});
  CHECK_EQ(db.envs().at(0).extra.at(0), std::string{"fanspeed=42"});
  CHECK_EQ(db.envs().at(0).extra.at(1), std::string{"note=two words"});
  CHECK(db.text().find("fanspeed=42") != std::string::npos);
}

TEST(malformed_rows_are_rejected) {
  // Each of these is a row of a kind this build knows, so misreading it would mean rewriting someone's measurements
  // wrongly; the whole load fails rather than the row being dropped.
  rejects("# prpll tunedb v1", "# prpll tunedb v2");
  rejects("run   4 512:15:512:212 prp 143400073 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274",
          "run   4 512:15:512:212 prp 143400073 short32 17 1774.230 2.100 24 6 1.0000 ok");  // a column short
  rejects("prp 143400073 short32 17 1774.230", "prp 143400073 sideways 17 1774.230");        // not a regime
  rejects("2.100 24 6 1.0000 ok 1753471274", "2.100 24 6 1.0000 fine 1753471274");           // not a status
  rejects("run   4 512:15:512:212 prp", "run   4 not-an-fft prp");                           // not an FFT
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   4 512:15:512:212 cert 143400073 short32 17");
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   9 512:15:512:212 prp 143400073 short32 17");
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   4 512:15:512:212 prp 143400073 short32 99");
  rejects("sess  4 env=1", "sess  4 env=9");
  rejects("work  4 prp 124647911 3", "work  4 prp 124647911 0");                // no assignments
  rejects("work  4 prp 124647911 3", "work  4 cert 124647911 3");               // not a test kind
  rejects("work  4 prp 124647911 3", "work  9 prp 124647911 3");                // an undeclared session
  rejects("work  4 prp 124647911 3", "work  4 prp 124647911");                  // a column short
  rejects("work  4 ll 286472227 1", "work  4 prp 124647911 1");                 // the same exponent twice
  rejects("sess  5 env=1 start=1753481200", "sess  4 env=1 start=1753481200");  // a second session 4
  rejects("cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3", "cfg   1 INPLACE=1,PAD=256,TAIL_KERNELS=3");
  rejects("cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3", "cfg   17 INPLACE=1,,PAD=256");
  rejects("env   1 gpu=", "env   1 cc=eight-oh-six gpu=");
  rejects("nogo  4 512:15:512:212 SHUFL_BYTES_W=16", "nogo  4 512:15:512:212 SHUFL_BYTES_W");
  rejects("ref   5 512:15:512:212 1000151 2000 171f3662c332472f", "ref   5 512:15:512:212 1000151 2000 nonsense");
  rejects("jump  4 512:15:512:212 prp short32 17 3", "jump  4 512:15:512:212 prp short32 17 third");
  rejects("jump  4 512:15:512:212 prp short32 17 3", "jump  4 512:15:512:212 prp short32 99 3");
  rejects("jump  4 512:15:512:212 prp short32 17 3", "jump  4 512:15:512:212 prp sideways 17 3");
  rejects("combo 4 512:15:512:212 prp short32 17 2", "combo 4 512:15:512:212 prp short32 17 1");  // not a combination
  rejects("combo 4 512:15:512:212 prp short32 17 2", "combo 4 512:15:512:212 prp short32 99 2");
  rejects("combo 4 512:15:512:212 prp short32 17 2", "combo 4 512:15:512:212 prp short32 17");
  rejects("boot  4 512:15:512:212 143400073", "boot  4 not-an-fft 143400073");
  rejects("boot  4 512:15:512:212 143400073", "boot  4 512:15:512:212 0");                     // no probe
  rejects("boot  4 512:15:512:212 143400073", "boot  9 512:15:512:212 143400073");             // an undeclared session
  rejects("boot  4 512:15:512:212 143400073 1753471497", "boot  4 512:15:512:212 143400073");  // a column short
  rejects("round 4 2 1 16 1753471499 512:15:512:212", "round 4 2 1 16 1753471499 not-an-fft");
  rejects("round 4 2 1 16 1753471499 512:15:512:212 prp short32 6",
          "round 4 2 1 16 1753471499 512:15:512:212 prp sideways 6");
  rejects("1K:8:1K:101 ll long32 0", "1K:8:1K:101 ll long32");        // an entry a column short
  rejects("round 4 2 1 16 1753471499", "round 4 0 1 16 1753471499");  // no round number
  rejects("round 4 2 1 16 1753471499", "round 4 2 0 16 1753471499");  // entries in a round of none
  rejects("round 4 1 0 0 1753471498", "round 4 1 1 0 1753471498");    // a round of none numbered as one
  rejects("round 4 1 0 0 1753471498", "round 9 1 0 0 1753471498");    // an undeclared session
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 0 2 1753471499 1 3 17");   // no sweep number
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 9 1 2 1753471499 1 3 17");   // an undeclared session
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 1 2 1753471499 99 3 17");  // an undeclared option set
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 1 2 1753471499 1 3 99");
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 1 2 1753471499 1 7 17");                // not an FFT type
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 1 2 1753471499 1 3");                   // a line a column short
  rejects("lines 4 1 2 1753471499 1 3 17", "lines 4 1 2 1753471499");                       // no global line
  rejects("29.40 118 0.371094 ok - 1753471402", "29.40 118 0.371094 ok 1753471402");        // no fingerprint field
  rejects("29.40 118 0.371094 ok - 1753471402", "29.40 118 0.371094 ok 12345 1753471402");  // not 16 digits
  rejects("29.40 118 0.371094 ok - 1753471402", "29.40 118 0.371094 ok 0123456789abcdeg 1753471402");
  rejects("sess  5 env=1 start=1753481200 gen=1 anchor=- alarmed=1",
          "sess  5 env=1 start=1753481200 gen=1 anchor=- alarmed=banana");

  // An unbalanced quote used to swallow the rest of the line, leaving every field after it at its default -- a row
  // that parses, and then reports the wrong hardware.
  rejects("gpu=\"NVIDIA RTX A4000\"", "gpu=\"NVIDIA RTX A4000");

  // "nan" and "inf" are what std::from_chars makes of those words, and %f writes them straight back.
  rejects("1774.230 2.100 24 6 1.0000 ok", "nan 2.100 24 6 1.0000 ok");
  rejects("1774.230 2.100 24 6 1.0000 ok", "1774.230 inf 24 6 1.0000 ok");
  rejects("1774.230 2.100 24 6 1.0000 ok", "-1.000 2.100 24 6 1.0000 ok");
  rejects("1774.230 2.100 24 6 1.0000 ok", "1774.230 2.100 24 6 0.0000 ok");
}

TEST(added_rows_are_canonical_too) {
  // parse() folds a spec on the way in; add() has to fold it the same way, or a caller writes a second key for one
  // configuration and it gets measured twice.
  TuneDB db;
  CHECK(db.parse("# prpll tunedb v1\n", "fixture"));

  DbEnv env;
  env.id = 1;
  CHECK(db.add(env));
  CHECK(db.addCfg(1, UseConfig{}));
  CHECK(db.add(SessRow{.id = 1, .env = 1, .anchor = ""}));

  CHECK(db.add(RunRow{.sess = 1, .fft = "1024:8:1024:102", .cfg = 1, .m = Measurement{}}));
  CHECK_EQ(db.runs().at(0).fft, std::string{"1K:8:1K:102"});
  CHECK(!db.add(RunRow{.sess = 1, .fft = "not-an-fft", .cfg = 1, .m = Measurement{}}));
}

TEST(values_the_format_cannot_write_back_are_refused) {
  // There is no escape in either grammar, so a row that cannot be written back is refused where it enters rather than
  // read back later as something else.
  TuneDB db;
  CHECK(db.parse("# prpll tunedb v1\n", "fixture"));

  DbEnv quoted;
  quoted.id = 1;
  quoted.gpu = "a \"quoted\" name";
  CHECK(!db.add(quoted));

  DbEnv plain;
  plain.id = 1;
  plain.gpu = "a name with spaces";  // spaces are fine: that is what the quoting is for
  CHECK(db.add(plain));

  CHECK(!db.addCfg(1, UseConfig{{"PAD", "2,5"}}));
  CHECK(!db.addCfg(1, UseConfig{{"PAD", "2 5"}}));
  CHECK(db.addCfg(1, UseConfig{{"PAD", "256"}}));

  CHECK(db.add(SessRow{.id = 1, .env = 1, .anchor = "512:15:512:212@143400073"}));
  CHECK(!db.add(SessRow{.id = 2, .env = 1, .anchor = "a \"quoted\" anchor"}));

  // A non-finite timing writes back as "nan" or "inf", which is also what from_chars reads: it would survive a
  // rewrite and reach the comparator a sort needs a strict weak ordering from.
  RunRow run{.sess = 1, .fft = "512:15:512:212", .cfg = 1, .m = Measurement{}};
  run.m.mean = std::numeric_limits<double>::quiet_NaN();
  CHECK(!db.add(run));
  run.m.mean = 1774.23;
  run.m.stddev = -1;
  CHECK(!db.add(run));
  run.m.stddev = 2.1;

  // The row carries four decimals of drift, so anything below half of the last of them writes back as zero -- which
  // the reader refuses, taking the whole database with it.
  run.m.drift = 0.00001;
  CHECK(!db.add(run));
  run.m.drift = 1;
  CHECK(db.add(run));

  CHECK(!db.add(RoeRow{.sess = 1, .fft = "512:15:512:212", .cfg = 1, .z = std::numeric_limits<double>::infinity()}));
}

TEST(a_malformed_spec_is_refused_rather_than_built) {
  // FFTConfig's own parser asserts on several of these, which no catch can see, and with assertions compiled out it
  // reads a variant digit past the end of the bits-per-word table, so a spec out of a file is checked before one is
  // built from it.
  CHECK(parseFft("512:15:512"));
  CHECK(parseFft("1K:8:1K:202"));
  CHECK(parseFft("3:256:2:256:1"));
  CHECK_EQ(parseFft("1024:8:1024:102")->spec(), std::string{"1K:8:1K:102"});
  CHECK_EQ(parseFft("2048:8:512:101")->spec(), std::string{"2K:8:512:201"});
  CHECK(parseFft("512:8:2K:202"));

  CHECK(!parseFft(""));
  CHECK(!parseFft("not-an-fft"));
  CHECK(!parseFft("512:15:512:999"));    // no such variant
  CHECK(!parseFft("512:15:512:212:2"));  // no such carry
  CHECK(!parseFft("512x:15:512"));       // garbage after the width, which strtod() would have read as 512
  CHECK(!parseFft("512:15"));
  CHECK(!parseFft("512:15:512:212:1:0"));
  CHECK(!parseFft("7:256:2:256"));   // no such FFT type
  CHECK(!parseFft("512:1:512"));     // a middle out of range
  CHECK(!parseFft("512:15:768"));    // no such height
  CHECK(!parseFft("512:8:4K"));      // nor this one, which is only a width
  CHECK(!parseFft("1:256:15:256"));  // an NTT middle that is not a power of two

  rejects("run   4 512:15:512:212 prp", "run   4 512:15:512:999 prp");
  rejects("run   4 512:15:512:212 prp", "run   4 512:15:512:212:2 prp");
}

TEST(an_environment_this_build_cannot_name_is_refused) {
  // A vendor or backend read as its default would be rewritten over the row it came from, so a measurement taken on
  // one machine would be filed under another.
  rejects("vendor=nvidia", "vendor=banana");
  rejects("be=ocl", "be=cdua");

  // '-' is what the writer emits for a card whose vendor it could not name.
  TuneDB db;
  CHECK(db.parse(withRow("vendor=nvidia", "vendor=-"), "fixture"));
  CHECK(!db.envs().at(0).isAmd && !db.envs().at(0).isNvidia);
}

TEST(an_env_row_says_what_the_card_can_build) {
  // Without them, a command that opens no device would offer a card FFTs it cannot build.
  rejects("fp64=1 builtins=1 machine=01:00.0", "builtins=1 machine=01:00.0");
  rejects("fp64=1 builtins=1 machine=01:00.0", "fp64=1 machine=01:00.0");
  rejects("fp64=1 builtins=1 machine=01:00.0", "fp64=yes builtins=1 machine=01:00.0");

  std::string text = withRow("vendor=nvidia", "vendor=amd");
  text.replace(text.find("fp64=1 builtins=1"), 17, "fp64=0 builtins=0");
  TuneDB const amd = loaded(text);
  CHECK_EQ(amd.text(), text);
  const DbEnv& row = amd.envs().at(0);
  CHECK(!row.hasFP64 && !row.amdBuiltins);

  Env const card = row.toEnv();
  CHECK(!card.hasFP64 && !card.amdBuiltins);
  CHECK(runnableVariants(card, FFTConfig{"1K:8:1K"}.shape).empty());
  std::vector<u32> const variants = runnableVariants(card, FFTConfig{"2:1K:8:1K"}.shape);
  CHECK(!variants.empty());
  for (u32 const v : variants) { CHECK(variant_W(v) != 0 && variant_H(v) != 0); }

  // The same card with and without the compiler's builtins is not the same card: it builds other kernels.
  DbEnv other = row;
  other.amdBuiltins = true;
  CHECK(!row.sameCard(other));
}

TEST(a_machine_or_an_anchor_holding_a_space_survives) {
  DbEnv env;
  env.id = 1;
  env.machine = "01 00";

  SessRow const sess{.id = 1, .env = 1, .start = 1, .gen = 0, .anchor = "512:15:512:212 @143400073"};
  std::string const text = "# prpll tunedb v1\n" + formatRow(env) + '\n' + formatRow(sess) + '\n';

  TuneDB const db = loaded(text);
  CHECK_EQ(db.envs().at(0).machine, std::string{"01 00"});
  CHECK_EQ(db.sessions().at(0).anchor, std::string{"512:15:512:212 @143400073"});
  CHECK_EQ(db.text(), text);
}

TEST(a_lock_is_read_off_the_kernels_table_by_the_file_it_is_on) {
  const char* const locks = "1: POSIX  ADVISORY  WRITE 2656 00:1c:155578 0 EOF\n"
                            "2: FLOCK  ADVISORY  WRITE 1177997 00:2c:499972 0 EOF\n"
                            "3: FLOCK  ADVISORY  WRITE 4242 103:02:77 0 EOF\n"
                            "3: -> FLOCK  ADVISORY  WRITE 4243 fd:01:88 0 EOF\n";
  CHECK(lockListed(locks, 0x00, 0x2c, 499'972));
  CHECK(lockListed(locks, 0x103, 0x02, 77));
  CHECK(lockListed(locks, 0x00, 0x1c, 155'578));
  CHECK(!lockListed(locks, 0x00, 0x2c, 499'973));
  CHECK(!lockListed(locks, 0x00, 0x2d, 499'972));

  // A process waiting for a lock does not hold it.
  CHECK(!lockListed(locks, 0xfd, 0x01, 88));
  CHECK(!lockListed("", 0, 0x2c, 499'972));
}

TEST(whether_another_process_is_writing_can_be_asked_without_keeping_it_out) {
#ifdef __linux__
  fs::path const path = fs::temp_directory_path() / "prpll-test-writer-holds.txt";
  fs::path const claim = path + ".lock";
  fs::remove(claim);

  // Asking creates nothing.
  CHECK(TuneDB::writerHolds(path) == false);
  CHECK(!fs::exists(claim));

  {
    TuneDB writer;
    CHECK(writer.lockForWriting(path));
    CHECK(TuneDB::writerHolds(path) == true);
  }
  CHECK(TuneDB::writerHolds(path) == false);

  // Asked over and over while writers come and go, it never once turns one away.
  std::atomic<bool> done{false};
  std::thread asker{[&] {
    while (!done) { (void)TuneDB::writerHolds(path); }
  }};
  u32 refused = 0;
  for (u32 i = 0; i < 2000; ++i) {
    TuneDB writer;
    if (!writer.lockForWriting(path)) { ++refused; }
  }
  done = true;
  asker.join();
  CHECK_EQ(refused, 0u);
  fs::remove(claim);
#endif
}

TEST(an_empty_file_is_an_empty_database_but_an_unreadable_one_is_not) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-tunedb.txt";
  fs::remove(path);
  { File::openWrite(path); }

  TuneDB db;
  CHECK(db.load(path));
  CHECK(db.envs().empty());

  // A file that is there and will not open must not read as empty: the save() after it would replace measurements
  // nobody read.
  fs::permissions(path, fs::perms::none);
  if (File::openRead(path)) {
    fs::remove(path);  // running as a user every file opens for, which this cannot tell apart
    return;
  }

  CHECK(!db.load(path));
  fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write);
  fs::remove(path);

  // And a path that is simply not there is an empty database.
  CHECK(db.load(path));
}

TEST(a_file_without_a_header_is_refused) {
  TuneDB db;
  CHECK(!db.parse("run   4 512:15:512:212 prp 143400073 short32 17 1.0 0.1 4 1 1.0 ok 9\n", "fixture"));

  // An empty file is an empty database, not a broken one.
  CHECK(db.parse("", "fixture"));
  CHECK(db.parse("# prpll tunedb v1\n", "fixture"));
  CHECK(db.text() == std::string{"# prpll tunedb v1\n"});
}

TEST(specs_are_canonical) {
  // 1K is 1024, so two spellings of one configuration land on one database key rather than being measured twice under
  // two names.
  std::string text = "# prpll tunedb v1\n"
                     "env   1 gpu=x name=x drv=1 vendor=amd be=ocl cc=0 noasm=0 pdl=0"
                     " fp64=1 builtins=1 machine=- build=0\n"
                     "cfg   1 -\n"
                     "sess  1 env=1 start=1 gen=0 anchor=-\n"
                     "run   1 1024:8:1024:102 prp 143400073 short32 1 1.000 0.100 4 1 1.0000 ok 9\n";
  TuneDB const db = loaded(text);
  CHECK_EQ(db.runs().at(0).fft, std::string{"1K:8:1K:102"});
}

TEST(option_sets_round_trip) {
  CHECK_EQ(configText({}), std::string{"-"});
  UseConfig const two{{"PAD", "256"}, {"INPLACE", "1"}};
  CHECK_EQ(configText(two), std::string{"INPLACE=1,PAD=256"});
  CHECK(parseConfigText("-")->empty());
  auto const pair = parseConfigText("INPLACE=1,PAD=256");
  CHECK_EQ(pair->at("PAD"), std::string{"256"});
  auto const bare = parseConfigText("DEBUG");  // the bare '-use KEY' spelling
  CHECK_EQ(bare->at("DEBUG"), std::string{"1"});
  CHECK(!parseConfigText(""));
  CHECK(!parseConfigText("INPLACE=1,"));
}

TEST(rows_are_refused_without_what_they_name) {
  TuneDB db;
  CHECK(db.parse("# prpll tunedb v1\n", "fixture"));

  DbEnv env;
  env.id = 1;
  CHECK(db.add(env));
  CHECK(!db.add(env));
  CHECK(db.addCfg(3, UseConfig{}));
  CHECK(!db.addCfg(3, UseConfig{}));

  CHECK(!db.add(SessRow{.id = 1, .env = 2, .anchor = ""}));  // no such env
  CHECK(db.add(SessRow{.id = 1, .env = 1, .anchor = ""}));

  RunRow run{.sess = 1, .fft = "512:15:512:212", .cfg = 3, .m = Measurement{}};
  CHECK(db.add(run));
  run.cfg = 4;
  CHECK(!db.add(run));  // no such cfg
  run.cfg = 3;
  run.sess = 2;
  CHECK(!db.add(run));  // no such session
}

TEST(device_names_with_spaces_survive) {
  DbEnv env;
  env.id = 1;
  env.gpu = "AMD Radeon Pro VII";
  env.name = "gfx906";
  env.driver = "3581.0";
  env.isAmd = true;
  std::string const text = "# prpll tunedb v1\n" + formatRow(env) + '\n';

  TuneDB const db = loaded(text);
  CHECK_EQ(db.envs().at(0).gpu, std::string{"AMD Radeon Pro VII"});
  CHECK_EQ(db.envs().at(0).driver, std::string{"3581.0"});
  CHECK(db.envs().at(0).isAmd);
  CHECK_EQ(db.text(), text);
}

namespace {

// A database with one env and one session, attached to a scratch file the way a measuring process attaches to its own.
struct Attached {
  fs::path path;
  TuneDB db;
  u32 sess = 0;

  explicit Attached(const char* name) : path{fs::temp_directory_path() / name} {
    fs::remove(path);
    CHECK(db.load(path));
    db.attach(path);

    u32 const env = db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"});
    CHECK(env);
    sess = db.beginSession(env, "-");
    CHECK(sess);
  }

  ~Attached() { fs::remove(path); }

  [[nodiscard]] std::string onDisk() const { return File::openRead(path).readAll(); }

  // What the next generation would see: the file as written, read by a process that wrote none of it.
  [[nodiscard]] TuneDB reread() const {
    TuneDB next;
    CHECK(next.load(path));
    return next;
  }

  [[nodiscard]] TryRow attempt(u32 cfg) const {
    return TryRow{.sess = sess,
                  .fft = "512:15:512:212",
                  .kind = TestKind::PRP,
                  .exponent = 143'400'073,
                  .cfg = cfg,
                  .ts = 1'753'471'250};
  }
};

}  // namespace

TEST(an_attempt_reaches_the_file_before_its_result_does) {
  Attached a{"prpll-test-attempt.txt"};
  u32 const cfg = a.db.internCfg(UseConfig{{"PAD", "256"}});
  CHECK(cfg);

  CHECK(a.db.add(a.attempt(cfg)));
  // The declaration is durable on its own: this is everything a process that died here would leave behind.
  CHECK(a.onDisk().find(formatRow(a.attempt(cfg)) + '\n') != std::string::npos);

  TuneDB const died = a.reread();
  CHECK_EQ(died.diedHolding().size(), size_t{1});
  CHECK(died.diedOn(1, cfg, TestKind::PRP, "512:15:512:212", 143'400'073));

  // And the configuration it names is the only one condemned.
  u32 const other = a.db.internCfg(UseConfig{{"PAD", "128"}});
  CHECK(!died.diedOn(1, other, TestKind::PRP, "512:15:512:212", 143'400'073));
}

TEST(a_result_answers_the_attempt_it_followed) {
  Attached a{"prpll-test-answered.txt"};
  u32 const cfg = a.db.internCfg(UseConfig{});
  CHECK(a.db.add(a.attempt(cfg)));
  CHECK(a.db.add(RunRow{.sess = a.sess,
                        .fft = "512:15:512:212",
                        .kind = TestKind::PRP,
                        .exponent = 143'400'073,
                        .regime = regimeOf(FFTConfig{"512:15:512:212"}, 143'400'073),
                        .cfg = cfg,
                        .m = {.mean = 1000, .stddev = 1, .blocks = 4, .calls = 1, .ts = 1'753'471'260}}));

  CHECK(a.reread().diedHolding().empty());
}

TEST(a_stop_answers_the_attempt_and_a_death_does_not) {
  Attached stopped{"prpll-test-stopped.txt"};
  u32 const cfg = stopped.db.internCfg(UseConfig{});
  CHECK(stopped.db.add(stopped.attempt(cfg)));
  stopped.db.closeTry(stopped.sess);
  // Ctrl-C between the declaration and the result is not a death: without the 'done' row the configuration being
  // measured at the time would be condemned for having been interrupted.
  CHECK(stopped.reread().diedHolding().empty());

  Attached lost{"prpll-test-lost.txt"};
  u32 const cfg2 = lost.db.internCfg(UseConfig{});
  CHECK(lost.db.add(lost.attempt(cfg2)));
  lost.db.sealSession(lost.sess);
  // A sealed session writes nothing further, so the round it was in cannot answer for the attempt that ended it.
  CHECK(!lost.db.add(RunRow{.sess = lost.sess,
                            .fft = "512:15:512:212",
                            .kind = TestKind::PRP,
                            .exponent = 143'400'073,
                            .regime = {},
                            .cfg = cfg2,
                            .m = {.mean = 1000, .blocks = 4, .calls = 1, .ts = 1'753'471'260}}));
  lost.db.closeTry(lost.sess);
  CHECK_EQ(lost.reread().diedHolding().size(), size_t{1});
}

TEST(this_processs_own_attempt_is_in_flight_rather_than_fatal) {
  Attached a{"prpll-test-inflight.txt"};
  u32 const cfg = a.db.internCfg(UseConfig{});
  CHECK(a.db.add(a.attempt(cfg)));

  // Held open by a session this process opened: reading it as a death would condemn every configuration the moment it
  // was declared.
  CHECK(a.db.diedHolding().empty());
  CHECK(!a.db.diedOn(1, cfg, TestKind::PRP, "512:15:512:212", 143'400'073));

  // Until the device goes away under it.
  a.db.sealSession(a.sess);
  CHECK_EQ(a.db.diedHolding().size(), size_t{1});
  CHECK(a.db.diedOn(1, cfg, TestKind::PRP, "512:15:512:212", 143'400'073));
}

TEST(a_value_a_build_failed_with_is_remembered_for_its_fft_whatever_else_is_set) {
  Attached a{"prpll-test-nogo.txt"};
  CHECK(a.db.add(
    NogoRow{.sess = a.sess, .fft = "512:15:512:212", .key = "SHUFL_BYTES_W", .val = "16", .ts = 1'753'471'260}));

  TuneDB const next = a.reread();
  CHECK(next.failedWith(1, "512:15:512:212", UseConfig{{"PAD", "256"}, {"SHUFL_BYTES_W", "16"}}));
  CHECK(!next.failedWith(1, "512:15:512:212", UseConfig{{"SHUFL_BYTES_W", "8"}}));
  // One FFT only: the budget that will not fit here may fit at another shape.
  CHECK(!next.failedWith(1, "256:2:256:212", UseConfig{{"SHUFL_BYTES_W", "16"}}));
}

TEST(ids_are_allocated_from_what_was_read) {
  Attached a{"prpll-test-ids.txt"};
  u32 const first = a.db.internCfg(UseConfig{{"PAD", "256"}});
  CHECK_EQ(a.db.internCfg(UseConfig{{"PAD", "256"}}), first);
  CHECK(a.db.internCfg(UseConfig{{"PAD", "128"}}) != first);

  // The same machine is the same env, and a second session of it does not declare a second one.
  CHECK_EQ(a.db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"}), 1u);
  CHECK_EQ(a.db.envs().size(), size_t{1});

  TuneDB const next = a.reread();
  CHECK_EQ(next.findCfgId(UseConfig{{"PAD", "256"}}), first);
  CHECK_EQ(next.findCfgId(UseConfig{{"NOT", "HERE"}}), 0u);
}

TEST(a_row_that_cannot_be_written_is_not_kept) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-unwritable.txt";
  fs::remove(path);
  // A file every write fails against, which is what a full disk looks like from here.
  fs::create_symlink("/dev/full", path);
  if (!fs::exists("/dev/full")) {
    fs::remove(path);
    return;
  }

  TuneDB db;
  CHECK(db.load(path));
  db.attach(path);

  // The whole crash layer rests on the row reaching the disk, so a database that says it wrote one when it did not is
  // worse than one that refuses: the caller would measure a configuration nothing is recorded against.
  CHECK(!db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"}));
  CHECK(db.envs().empty());
  CHECK(!db.beginSession(1, "-"));
  CHECK(db.sessions().empty());
  CHECK(!db.internCfg(UseConfig{{"PAD", "256"}}));
  CHECK(db.cfgs().empty());

  fs::remove(path);
}

TEST(only_one_process_may_write_a_database) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-lock.txt";
  fs::remove(path);
  fs::remove(path + ".lock");

  TuneDB first;
  CHECK(first.lockForWriting(path));

  // Ids are allocated from what was read, so a second writer would hand out the same ones and the next load would
  // refuse the file for declaring an id twice -- permanently, since a file that did not parse is never rewritten.
  TuneDB second;
  CHECK(!second.lockForWriting(path));

  // And the claim goes with the object, so the next one in gets it.
  {
    TuneDB third;
    CHECK(!third.lockForWriting(path));
  }
  first = TuneDB{};
  TuneDB fourth;
  CHECK(fourth.lockForWriting(path));

  fs::remove(path);
  fs::remove(path + ".lock");
}

TEST(a_rewrite_does_not_give_the_database_away) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-lock-rewrite.txt";
  fs::remove(path);
  fs::remove(path + ".lock");

  TuneDB holder;
  CHECK(holder.lockForWriting(path));
  CHECK(holder.load(path));

  // `save` replaces the database through a rename, which unlinks the inode it was read from.  A claim taken on that
  // inode would go with it, and the next writer in would be told the database is free while this one still has it.
  holder.save(path);

  TuneDB other;
  CHECK(!other.lockForWriting(path));

  fs::remove(path);
  fs::remove(path + ".lock");
}

TEST(an_alarm_row_flags_the_session_it_names) {
  // A session's row is written before its anchor has ever been timed, so the flag arrives later as a row of its own
  // and a rewrite puts it back where the grammar keeps it.
  std::string const text = withRow("sess  5 env=1 start=1753481200 gen=1 anchor=- alarmed=1",
                                   "sess  5 env=1 start=1753481200 gen=1 anchor=-") +
    "alarm 5 1753481500\n";

  TuneDB const db = loaded(text);
  CHECK(!db.sessions().at(0).alarmed);
  CHECK(db.sessions().at(1).alarmed);
  CHECK(db.text().find("alarm 5") == std::string::npos);
  CHECK(db.text().find("anchor=- alarmed=1") != std::string::npos);
}

TEST(an_alarm_is_recorded_once) {
  TuneDB db = loaded(FIXTURE);
  CHECK(db.add(AlarmRow{.sess = 4, .ts = 1'753'471'600}));
  CHECK(db.sessions().at(0).alarmed);

  // Already flagged: nothing more to say, and nothing more to write.
  CHECK(db.add(AlarmRow{.sess = 4, .ts = 1'753'471'700}));
  CHECK(!db.add(AlarmRow{.sess = 99, .ts = 1'753'471'700}));
}

TEST(a_malformed_anchor_row_is_refused) {
  rejects("anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410",
          "anchor 4 512:15:512:212 143400073 1 1774.230 1.0000");
  rejects("anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410",
          "anchor 4 512:15:512:212 143400073 1 1774.230 0.0000 1753471410");
  rejects("anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410",
          "anchor 4 512:15:512:212 143400073 1 -1.000 1.0000 1753471410");
  rejects("anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410",
          "anchor 4 512:15:512:212 143400073 99 1774.230 1.0000 1753471410");

  TuneDB db;
  CHECK(!db.parse(std::string{FIXTURE} + "alarm 5 later\n", "fixture"));
  CHECK(!db.parse(std::string{FIXTURE} + "alarm 99 1753481500\n", "fixture"));
}

TEST(an_anchor_row_answers_the_attempt_it_followed) {
  TuneDB db = loaded(FIXTURE);
  CHECK_EQ(db.diedHolding().size(), size_t{1});
  CHECK(db.add(AnchorRow{.sess = 5,
                         .fft = "512:15:512:212",
                         .exponent = 143'400'073,
                         .cfg = 1,
                         .mean = 1800,
                         .ratio = 1.014,
                         .ts = 1'753'481'600}));
  CHECK(db.diedHolding().empty());
}

TEST(an_env_row_carries_this_builds_kernels) {
  DbEnv const fresh = dbEnvOf(Env{.isNvidia = true, .computeCapability = 806});
  CHECK_EQ(fresh.build, buildFingerprint());
  CHECK(fresh.build != 0);

  // A row measured against other kernels is another env, however much of the machine it shares.
  DbEnv other = fresh;
  other.build = fresh.build + 1;
  CHECK(!fresh.sameMachine(other));
  CHECK(fresh.sameMachine(fresh));
}

TEST(a_baseline_is_found_however_its_anchor_was_spelled) {
  // An explicit FP64 type digit and an unstated variant both name the configuration the rows were measured under.
  std::string const text = withRow("sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073",
                                   "sess  4 env=1 start=1753471200 gen=0 anchor=0:512:15:512@143400073");
  TuneDB const db = loaded(text);
  CHECK(db.envBaseline(1) == &db.anchors().at(0));
}

TEST(a_baseline_belongs_to_the_anchor_its_env_is_pinned_to) {
  // A reading of something else -- an env re-pinned by hand, or a row from a build that chose differently -- is not
  // this env's baseline.
  std::string const text = withRow("anchor 4 512:15:512:212 143400073 1 1774.230 1.0000 1753471410",
                                   "anchor 4 512:15:512:212 143400071 1 1774.230 1.0000 1753471410");
  TuneDB const db = loaded(text);
  CHECK(db.envBaseline(1) == &db.anchors().at(1));
}

namespace {

// Two envs that are the same card under two kernel builds, each with readings of its own, plus a pair of duplicate
// readings inside one env and a second spelling of one option set.
const char* const FOLDING = "# prpll tunedb v1\n"
                            "env   1 gpu=\"a card\" name=\"a card\" drv=1.0 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                            " fp64=1 builtins=1 machine=01:00.0"
                            " build=1111111111111111\n"
                            "env   2 gpu=\"a card\" name=\"a card\" drv=1.0 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                            " fp64=1 builtins=1 machine=01:00.0"
                            " build=2222222222222222\n"
                            "cfg   1 PAD=256\n"
                            "cfg   2 PAD=256\n"
                            "cfg   3 PAD=128\n"
                            "sess  1 env=1 start=1000 gen=0 anchor=512:15:512:212@143400073\n"
                            "sess  2 env=1 start=2000 gen=0 anchor=512:15:512:212@143400073\n"
                            "sess  3 env=2 start=3000 gen=0 anchor=512:15:512:212@143400073\n"
                            "run   1 512:15:512:212 prp 143400073 short32 1 1000.000 0.000 4 1 1.0000 ok 1100\n"
                            "run   2 512:15:512:212 prp 143400073 short32 2 1010.000 0.000 4 1 1.0000 ok 2100\n"
                            "run   1 512:15:512:212 prp 143400073 short32 3 1500.000 0.000 4 1 1.0000 ok 1200\n"
                            "run   3 512:15:512:212 prp 143400073 short32 1 1400.000 0.000 4 1 1.0000 ok 3100\n"
                            "nogo  1 256:2:256:212 SHUFL_BYTES_W=16 1150\n"
                            "roe   1 512:15:512:212 143400073 1 29.40 118 0.371094 ok - 1300\n"
                            "roe   2 512:15:512:212 143400073 1 31.00 200 0.400000 fail - 2300\n"
                            "ref   1 512:15:512:212 1000151 2000 171f3662c332472f 1500\n";

// The fixture above with one row replaced.
std::string withRow(const std::string& original, const std::string& replacement, const std::string& text) {
  auto const at = text.find(original);
  CHECK(at != std::string::npos);
  std::string out = text;
  return out.replace(at, original.size(), replacement);
}

const RunRow* runAt(const std::vector<RunRow>& rows, u64 exponent, u32 cfg) {
  for (const RunRow& r : rows) {
    if (r.exponent == exponent && r.cfg == cfg) { return &r; }
  }
  return nullptr;
}

}  // namespace

TEST(duplicate_readings_of_one_configuration_pool) {
  TuneDB const db = loaded(FOLDING);
  CHECK_EQ(db.runs().size(), size_t{4});

  // Three keys: the same options at 1 and 2 in env 1, PAD=128 in env 1, and env 2's own reading, which is a different
  // kernel build and so never enters the comparison.
  auto const merged = db.mergedRuns();
  CHECK_EQ(merged.size(), size_t{3});

  const RunRow* const pooled = runAt(merged, 143'400'073, 1);
  CHECK(pooled);
  CHECK_EQ(pooled->m.mean, 1005.0);
  CHECK_EQ(pooled->m.blocks, 8u);
  CHECK_EQ(pooled->m.calls, 2u);
  // The spread the two readings really showed, which neither row declared on its own.
  CHECK(pooled->m.stddev > 5.34 && pooled->m.stddev < 5.35);
  CHECK_EQ(pooled->m.ts, u64{2100});

  CHECK_EQ(runAt(merged, 143'400'073, 3)->m.mean, 1500.0);
}

// A measurement taken in a mode the table later withdrew -- the carry shuttle through ld.global.nc, which raced -- stays
// in the file, but is read as if it were never taken: nothing may choose, publish or pool it.
TEST(a_reading_in_a_withdrawn_mode_is_not_read) {
  TuneDB const withdrawn = loaded(withRow("cfg   3 PAD=128", "cfg   3 LOADS=35554", FOLDING));
  CHECK_EQ(withdrawn.runs().size(), size_t{4});
  CHECK_EQ(withdrawn.mergedRuns().size(), size_t{2});
  CHECK(runAt(withdrawn.mergedRuns(), 143'400'073, 3) == nullptr);

  TuneDB const kept = loaded(withRow("cfg   3 PAD=128", "cfg   3 LOADS=35504", FOLDING));
  CHECK(runAt(kept.mergedRuns(), 143'400'073, 3) != nullptr);
}

TEST(a_failure_does_not_average_with_a_reading) {
  std::string const okThenErr =
    withRow("run   2 512:15:512:212 prp 143400073 short32 2 1010.000 0.000 4 1 1.0000 ok 2100",
            "run   2 512:15:512:212 prp 143400073 short32 2 1010.000 0.000 4 1 1.0000 err 2100", FOLDING);
  auto const afterErr = loaded(okThenErr).mergedRuns();
  const RunRow* const failed = runAt(afterErr, 143'400'073, 1);
  CHECK(failed->m.status == Status::Err);
  CHECK_EQ(failed->m.mean, 1010.0);

  // And the other way round: a configuration that failed once is not redeemed by a later reading that happened to run.
  std::string const errThenOk =
    withRow("run   1 512:15:512:212 prp 143400073 short32 1 1000.000 0.000 4 1 1.0000 ok 1100",
            "run   1 512:15:512:212 prp 143400073 short32 1 1000.000 0.000 4 1 1.0000 err 1100", FOLDING);
  auto const afterOk = loaded(errThenOk).mergedRuns();
  const RunRow* const sticky = runAt(afterOk, 143'400'073, 1);
  CHECK(sticky->m.status == Status::Err);
  CHECK_EQ(sticky->m.mean, 1000.0);
}

TEST(one_option_set_spelled_twice_is_one_measurement) {
  // cfg 1 and cfg 2 are the same options, which is what makes the two readings above duplicates at all.
  TuneDB const db = loaded(FOLDING);
  CHECK(*db.findCfg(1) == *db.findCfg(2));

  std::string const apart = withRow("cfg   2 PAD=256", "cfg   2 PAD=64", FOLDING);
  CHECK_EQ(loaded(apart).mergedRuns().size(), size_t{4});
}

TEST(evidence_is_replaced_rather_than_pooled) {
  TuneDB const db = loaded(FOLDING);

  auto const roes = db.latestRoes();
  CHECK_EQ(roes.size(), size_t{1});
  CHECK_EQ(roes.at(0).z, 31.0);
  CHECK(!roes.at(0).checkOk);
}

TEST(a_compacted_database_says_the_same_thing_and_says_it_once) {
  TuneDB db = loaded(FOLDING);
  CHECK(db.compact());

  CHECK_EQ(db.runs().size(), size_t{3});
  CHECK_EQ(db.roes().size(), size_t{1});

  // PAD=128 is named by a surviving row; the second spelling of PAD=256 is not, since the readings under it folded
  // into the row that names the first.
  CHECK(db.findCfg(1) && db.findCfg(3));
  CHECK(!db.findCfg(2));

  // An option set a lines row alone names is kept as well.
  TuneDB swept = loaded(std::string{FOLDING} + "lines 1 1 0 1600 3 3 2\n");
  CHECK(swept.compact());
  CHECK(swept.findCfg(2) && swept.findCfg(3));

  // A rewrite is a file another build has to read back, and compacting one twice changes nothing further.
  TuneDB again = loaded(db.text());
  CHECK(again.compact());
  CHECK_EQ(again.text(), db.text());
  CHECK_EQ(again.mergedRuns().size(), db.mergedRuns().size());
}

TEST(a_reset_drops_what_was_measured_and_keeps_the_env_that_measured_it) {
  TuneDB db = loaded(FOLDING);
  CHECK(db.reset(1));

  CHECK_EQ(db.runs().size(), size_t{1});
  CHECK_EQ(db.runs().at(0).sess, 3u);
  CHECK(db.roes().empty() && db.refs().empty() && db.nogos().empty());

  // The env and its sessions stay: the env is still this card under these kernels, and it is the earliest session
  // that pins the anchor every later reading is divided by.
  CHECK_EQ(db.envs().size(), size_t{2});
  CHECK_EQ(db.sessions().size(), size_t{3});
  CHECK_EQ(db.envAnchor(1), std::string{"512:15:512:212@143400073"});

  CHECK(!db.reset(9));
  CHECK(!db.reset(1, "not-an-fft"));
}

TEST(a_reset_of_one_shape_leaves_the_others) {
  std::string const twoShapes =
    withRow("run   1 512:15:512:212 prp 143400073 short32 3 1500.000 0.000 4 1 1.0000 ok 1200",
            "run   1 256:2:256:212 prp 143400073 short32 3 1500.000 0.000 4 1 1.0000 ok 1200", FOLDING);

  TuneDB db = loaded(twoShapes);
  CHECK(db.reset(1, "512:15:512:212"));

  CHECK_EQ(db.runs().size(), size_t{2});
  CHECK(runAt(db.runs(), 143'400'073, 3));
  CHECK_EQ(db.nogos().size(), size_t{1});
  CHECK(db.roes().empty() && db.refs().empty());
}

TEST(a_reset_drops_the_attempts_standing_against_an_env) {
  std::string text = FOLDING;
  text += "try   1 512:15:512:212 prp 143400073 1 1600\n";
  text += "try   3 512:15:512:212 prp 143400073 1 3600\n";

  TuneDB db = loaded(text);
  CHECK_EQ(db.diedHolding().size(), size_t{2});

  // A kernel change can undo a configuration that took the card down, which is the whole reason to ask for this.
  CHECK(db.reset(1));
  CHECK_EQ(db.diedHolding().size(), size_t{1});
  CHECK(!db.diedOn(1, 1, TestKind::PRP, "512:15:512:212", 143'400'073));
  CHECK(db.diedOn(2, 1, TestKind::PRP, "512:15:512:212", 143'400'073));

  CHECK_EQ(loaded(db.text()).diedHolding().size(), size_t{1});
}

TEST(a_reset_keeps_saying_that_an_attempt_it_orphans_was_answered) {
  std::string text = FOLDING;
  text += "try   1 512:15:512:212 prp 143400073 1 1600\n";
  text += "run   1 512:15:512:212 prp 143400073 short32 1 1000.000 0.000 4 1 1.0000 ok 1700\n";
  text += "try   1 256:2:256:212 prp 143400073 1 1800\n";

  TuneDB db = loaded(text);
  CHECK_EQ(db.diedHolding().size(), size_t{1});

  // Dropping the attempt that was in flight leaves the earlier one standing, and an attempt left standing condemns
  // its configuration for good -- so the `done` that resolved it has to survive the rewrite.
  CHECK(db.reset(1, "256:2:256:212"));
  CHECK(db.diedHolding().empty());
  CHECK(loaded(db.text()).diedHolding().empty());
}

TEST(a_reset_will_not_take_the_baseline_out_from_under_what_survives) {
  std::string text = FOLDING;
  text += "anchor 1 512:15:512:212 143400073 1 1000.000 1.0000 1900\n";
  text += "run   1 256:2:256:212 prp 143400073 short32 1 900.000 0.000 4 1 1.0000 ok 1950\n";

  TuneDB db = loaded(text);

  // Every other shape's cost is expressed against these readings, so dropping them alone would leave the survivors
  // normalised against a reference that is gone.
  CHECK(!db.reset(1, "512:15:512:212"));
  CHECK_EQ(db.anchors().size(), size_t{1});

  // The shape that is not the anchor is free to go, and so is the whole env, which takes the dependent rows with it.
  CHECK(db.reset(1, "256:2:256:212"));
  CHECK(db.reset(1));
  CHECK(db.anchors().empty());
}

TEST(adopt_folds_two_envs_and_merges_their_common_rows) {
  TuneDB db = loaded(FOLDING);
  CHECK_EQ(db.adoptCandidate(1), 2u);
  CHECK(db.adopt(2, 1));

  CHECK_EQ(db.envs().size(), size_t{1});
  CHECK_EQ(db.envs().at(0).id, 1u);
  CHECK_EQ(db.envOf(3), 1u);

  // env 2's reading was of the same configuration as env 1's pair, and now that the user has said the kernels that
  // moved were not these, the three are one measurement.
  auto const merged = db.mergedRuns();
  CHECK_EQ(merged.size(), size_t{2});
  const RunRow* const pooled = runAt(merged, 143'400'073, 1);
  CHECK_EQ(pooled->m.calls, 3u);
  CHECK_EQ(pooled->m.blocks, 12u);
  CHECK(pooled->m.mean > 1136.66 && pooled->m.mean < 1136.67);

  CHECK(loaded(db.text()).mergedRuns().size() == merged.size());
}

TEST(adopt_leaves_the_target_env_the_reference_it_had) {
  std::string text = FOLDING;
  text += "anchor 1 512:15:512:212 143400073 1 1100.000 1.0000 1900\n";
  text += "anchor 3 512:15:512:212 143400073 1 1000.000 1.0000 3900\n";

  TuneDB db = loaded(text);
  CHECK_EQ(db.envBaseline(2)->mean, 1000.0);

  // env 1's sessions are the older ones, so without care the fold would hand env 2 a baseline taken under the very
  // kernels being adopted -- and every later session divides by it.
  CHECK(db.adopt(1, 2));
  CHECK_EQ(db.envAnchor(2), std::string{"512:15:512:212@143400073"});
  CHECK_EQ(db.envBaseline(2)->mean, 1000.0);
  CHECK_EQ(db.anchors().size(), size_t{1});

  // The adopted readings keep the correction they were measured under: what the user is saying is that it still
  // applies, not that it should be recomputed against another build's clock.
  CHECK_EQ(db.runs().size(), size_t{4});
}

TEST(adopt_refuses_two_envs_that_measure_against_different_anchors) {
  std::string const elsewhere = withRow("sess  3 env=2 start=3000 gen=0 anchor=512:15:512:212@143400073",
                                        "sess  3 env=2 start=3000 gen=0 anchor=256:2:256:212@143400073", FOLDING);

  TuneDB db = loaded(elsewhere);
  CHECK(!db.adopt(1, 2));
  CHECK(!db.adopt(2, 1));
  CHECK_EQ(db.envs().size(), size_t{2});

  // An env that has never named an anchor has no reference to disagree with.
  TuneDB unpinned = loaded(withRow("sess  3 env=2 start=3000 gen=0 anchor=256:2:256:212@143400073",
                                   "sess  3 env=2 start=3000 gen=0 anchor=-", elsewhere));
  CHECK(unpinned.adopt(1, 2));
}

TEST(adopt_refuses_an_env_that_is_not_the_card) {
  std::string const otherCard = withRow("env   2 gpu=\"a card\" name=\"a card\" drv=1.0 vendor=nvidia be=ocl cc=806"
                                        " noasm=0 pdl=0 fp64=1 builtins=1 machine=01:00.0 build=2222222222222222",
                                        "env   2 gpu=\"a card\" name=\"a card\" drv=1.0 vendor=nvidia be=ocl cc=806"
                                        " noasm=0 pdl=0 fp64=1 builtins=1 machine=02:00.0 build=2222222222222222",
                                        FOLDING);

  TuneDB db = loaded(otherCard);
  CHECK_EQ(db.adoptCandidate(1), 0u);
  CHECK(!db.adopt(2, 1));
  CHECK(!db.adopt(9, 1));
  CHECK(!db.adopt(1, 9));
  CHECK_EQ(db.envs().size(), size_t{2});
}

TEST(a_rewrite_is_refused_while_a_session_is_appending) {
  fs::path const path = fs::temp_directory_path() / "prpll-test-rewrite.txt";
  fs::remove(path);

  TuneDB db = loaded(FOLDING);
  db.attach(path);

  // These three are the only operations that do not append, and a file being appended to cannot be rewritten under
  // the rows still arriving in it.
  CHECK(!db.compact());
  CHECK(!db.reset(1));
  CHECK(!db.adopt(2, 1));
  CHECK_EQ(db.runs().size(), size_t{4});

  fs::remove(path);
}

TEST(an_env_whose_sessions_raced_is_pinned_by_its_first_anchor_reading) {
  // A session that races for its anchor writes its own row before it knows the winner, so the env's anchor is the one
  // its first reading names -- and a later session, which does name it, agrees.
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} +
                   "\n"
                   "env   1 gpu=\"a card\" name=\"a card\" drv=1 vendor=nvidia be=ocl cc=806 noasm=0 pdl=0"
                   " fp64=1 builtins=1 machine=-"
                   " build=0000000000000001\n"
                   "cfg   1 -\n"
                   "sess  1 env=1 start=1753471200 gen=0 anchor=-\n"
                   "sess  2 env=1 start=1753471300 gen=0 anchor=-\n"
                   "anchor 2 51:1K:8:256:202 118063003 1 5173.448 1.0000 1753471350\n"
                   "anchor 1 3:1K:8:512:202 118063003 1 2250.055 1.0000 1753471250\n",
                 "fixture"));

  CHECK_EQ(db.envAnchor(1), std::string{"3:1K:8:512:202@118063003"});
  CHECK(db.envBaseline(1) != nullptr);
  CHECK_EQ(db.envBaseline(1)->mean, 2250.055);

  // A session row that names one still decides, as before.
  CHECK(db.add(SessRow{.id = 3, .env = 1, .start = 1'753'471'400, .gen = 0, .anchor = "1K:13:256:212@118063003"}));
  CHECK_EQ(db.envAnchor(1), std::string{"1K:13:256:212@118063003"});
}

TEST(a_roe_row_is_held_as_its_line_reads_back) {
  // The reach derivation works out where to read next from z, so a process and a later one reading its file have to
  // see the same z.
  TuneDB db;
  u32 const env = db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"});
  u32 const sess = db.beginSession(env, "-");
  CHECK(db.add(RoeRow{.sess = sess,
                      .fft = "512:15:512:212",
                      .exponent = 143'400'073,
                      .cfg = db.internCfg({}),
                      .z = 20.834567891,
                      .n = 2150,
                      .maxRoe = 0.30571234567,
                      .checkOk = true,
                      .ts = 1}));

  TuneDB const again = loaded(db.text());
  CHECK_EQ(db.roes().size(), size_t{1});
  CHECK_EQ(again.roes().size(), size_t{1});
  if (db.roes().empty() || again.roes().empty()) { return; }
  CHECK_EQ(db.roes().at(0).z, again.roes().at(0).z);
  CHECK_EQ(db.roes().at(0).maxRoe, again.roes().at(0).maxRoe);
  CHECK_EQ(db.roes().at(0).z, 20.83);
}

TEST(a_run_and_an_anchor_row_are_held_as_their_lines_read_back) {
  // A race decided by margin on the readings this process holds has to stay decided in the one that reloads them: a
  // rival 1.2190 us/it behind a leader at 486.917, against a margin of 1.2173, was decided live and pending on reload
  // when the process kept the digits the file does not.
  TuneDB db;
  u32 const env = db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"});
  u32 const sess = db.beginSession(env, "-");
  CHECK(db.add(RunRow{.sess = sess,
                      .fft = "1K:7:256:212",
                      .kind = TestKind::PRP,
                      .exponent = 67'513'549,
                      .regime = regimeOf(FFTConfig{"1K:7:256:212"}, 67'513'549),
                      .cfg = db.internCfg({}),
                      .m = {.mean = 488.13649876,
                            .stddev = 0.40412345,
                            .blocks = 4,
                            .calls = 7,
                            .drift = 1.00304567,
                            .status = Status::Ok,
                            .ts = 1}}));
  CHECK(db.add(AnchorRow{.sess = sess,
                         .fft = "1K:7:256:212",
                         .exponent = 67'513'549,
                         .cfg = db.internCfg({}),
                         .mean = 520.90749,
                         .ratio = 1.00304567,
                         .ts = 2}));

  TuneDB const again = loaded(db.text());
  CHECK_EQ(db.runs().size(), size_t{1});
  CHECK_EQ(again.runs().size(), size_t{1});
  CHECK_EQ(again.anchors().size(), size_t{1});
  if (db.runs().empty() || again.runs().empty() || db.anchors().empty() || again.anchors().empty()) { return; }
  const Measurement& held = db.runs().at(0).m;
  const Measurement& read = again.runs().at(0).m;
  CHECK_EQ(held.mean, read.mean);
  CHECK_EQ(held.stddev, read.stddev);
  CHECK_EQ(held.drift, read.drift);
  CHECK_EQ(held.cost(), read.cost());
  CHECK_EQ(held.mean, 488.136);
  CHECK_EQ(db.anchors().at(0).mean, again.anchors().at(0).mean);
  CHECK_EQ(db.anchors().at(0).ratio, again.anchors().at(0).ratio);
  CHECK_EQ(db.anchors().at(0).ratio, 1.0030);
}

TEST(a_roe_rows_fingerprint_survives_the_file_exactly) {
  TuneDB db;
  u32 const env = db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"});
  u32 const sess = db.beginSession(env, "-");
  RoeRow row{.sess = sess,
             .fft = "512:15:512:212",
             .exponent = 143'400'073,
             .cfg = db.internCfg({}),
             .z = 20.83,
             .n = 2150,
             .maxRoe = 0.3,
             .checkOk = true,
             .fp = 0xfedcba9876543210ull,
             .ts = 1};
  CHECK(db.add(row));
  row.fp = 0;
  row.ts = 2;
  CHECK(db.add(row));

  std::string const text = db.text();
  CHECK(text.find(" ok fedcba9876543210 1\n") != std::string::npos);
  CHECK(text.find(" ok - 2\n") != std::string::npos);
  TuneDB const again = loaded(text);
  CHECK_EQ(again.roes().size(), size_t{2});
  if (again.roes().size() < 2) { return; }
  CHECK_EQ(again.roes().at(0).fp, 0xfedcba9876543210ull);
  CHECK_EQ(again.roes().at(1).fp, u64{0});
}
