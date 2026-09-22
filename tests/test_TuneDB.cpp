// Copyright (C) Jason Lynch

// Tests that a hand-written file round-trips, that a malformed row of a known kind refuses the whole file, and that a
// row kind -- or an env field -- this build does not know survives a rewrite.

#include "TuneDB.h"

#include "File.h"
#include "test.h"

#include <limits>
#include <string>
#include <vector>

using namespace tune;

namespace {

// A file with one of every row kind, in the order the writer emits them, plus a row kind and an env field from a build
// that does not exist yet.
const char* const FIXTURE =
  "# prpll tunedb v1\n"
  "env   1 gpu=\"NVIDIA RTX A4000\" name=\"NVIDIA RTX A4000\" drv=550.163.01 vendor=nvidia be=ocl cc=806 noasm=0"
  " pdl=0 machine=01:00.0 build=9a3f21c0d1e2f304 fanspeed=42 \"note=two words\"\n"
  "cfg   1 -\n"
  "cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3\n"
  "sess  4 env=1 start=1753471200 gen=0 anchor=512:15:512:212@143400073\n"
  "sess  5 env=1 start=1753481200 gen=1 anchor=- alarmed=1\n"
  "run   4 512:15:512:212 prp 143400073 short32 17 1774.230 2.100 24 6 1.0000 ok 1753471274\n"
  "run   4 512:15:512:212 ll 143400073 short32 1 1801.000 3.000 8 2 0.9980 err 1753471300\n"
  "nogo  4 512:15:512:212 SHUFL_BYTES_W=16 1753471260\n"
  "roe   4 512:15:512:212 143400073 17 29.40 118 0.371094 ok 1753471402\n"
  "reach 4 512:15:512:212 prp short32 17 148000000 confirmed 1753471460\n"
  "ref   5 512:15:512:212 1000151 2000 171f3662c332472f 1753471480\n"
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

  CHECK_EQ(db.reaches().size(), size_t{1});
  CHECK_EQ(db.reaches().at(0).reach, u64{148'000'000});
  CHECK(db.reaches().at(0).evidence == Evidence::Confirmed);

  CHECK_EQ(db.refs().size(), size_t{1});
  CHECK_EQ(db.refs().at(0).res64, u64{0x171f3662c332472f});
  CHECK_EQ(db.refs().at(0).iters, u64{2000});
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
  rejects("prp short32 17 148000000 confirmed", "prp short32 17 148000000 probably");        // not an evidence state
  rejects("run   4 512:15:512:212 prp", "run   4 not-an-fft prp");                           // not an FFT
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   4 512:15:512:212 cert 143400073 short32 17");
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   9 512:15:512:212 prp 143400073 short32 17");
  rejects("run   4 512:15:512:212 prp 143400073 short32 17", "run   4 512:15:512:212 prp 143400073 short32 99");
  rejects("sess  4 env=1", "sess  4 env=9");
  rejects("sess  5 env=1 start=1753481200", "sess  4 env=1 start=1753481200");  // a second session 4
  rejects("cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3", "cfg   1 INPLACE=1,PAD=256,TAIL_KERNELS=3");
  rejects("cfg   17 INPLACE=1,PAD=256,TAIL_KERNELS=3", "cfg   17 INPLACE=1,,PAD=256");
  rejects("env   1 gpu=", "env   1 cc=eight-oh-six gpu=");
  rejects("nogo  4 512:15:512:212 SHUFL_BYTES_W=16", "nogo  4 512:15:512:212 SHUFL_BYTES_W");
  rejects("ref   5 512:15:512:212 1000151 2000 171f3662c332472f", "ref   5 512:15:512:212 1000151 2000 nonsense");
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

  CHECK(db.add(RunRow{.sess = 1, .fft = "1024:8:1K:102", .cfg = 1, .m = Measurement{}}));
  CHECK_EQ(db.runs().at(0).fft, std::string{"1K:8:1K:202"});
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
  CHECK_EQ(parseFft("1024:8:1024:102")->spec(), std::string{"1K:8:1K:202"});

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
  // 1K is 1024, and at width 1024 variant digit 1 folds onto 2 (M1.4), so two spellings of one configuration land on
  // one database key rather than being measured twice under two names.
  std::string text = "# prpll tunedb v1\n"
                     "env   1 gpu=x name=x drv=1 vendor=amd be=ocl cc=0 noasm=0 pdl=0 machine=- build=0\n"
                     "cfg   1 -\n"
                     "sess  1 env=1 start=1 gen=0 anchor=-\n"
                     "run   1 1024:8:1K:102 prp 143400073 short32 1 1.000 0.100 4 1 1.0000 ok 9\n";
  TuneDB const db = loaded(text);
  CHECK_EQ(db.runs().at(0).fft, std::string{"1K:8:1K:202"});
}

TEST(option_sets_round_trip) {
  CHECK_EQ(configText({}), std::string{"-"});
  UseConfig const two{{"PAD", "256"}, {"INPLACE", "1"}};
  CHECK_EQ(configText(two), std::string{"INPLACE=1,PAD=256"});
  CHECK(parseConfigText("-")->empty());
  CHECK_EQ(parseConfigText("INPLACE=1,PAD=256")->at("PAD"), std::string{"256"});
  CHECK_EQ(parseConfigText("DEBUG")->at("DEBUG"), std::string{"1"});  // the bare '-use KEY' spelling
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
    return TryRow{.sess = sess, .fft = "512:15:512:212", .kind = TestKind::PRP, .exponent = 143'400'073, .cfg = cfg,
                  .ts = 1753471250};
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
                        .m = {.mean = 1000, .stddev = 1, .blocks = 4, .calls = 1, .ts = 1753471260}}));

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
                            .m = {.mean = 1000, .blocks = 4, .calls = 1, .ts = 1753471260}}));
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

TEST(a_key_that_will_not_build_is_excluded_whatever_else_is_set) {
  Attached a{"prpll-test-nogo.txt"};
  CHECK(a.db.add(NogoRow{.sess = a.sess, .fft = "512:15:512:212", .key = "SHUFL_BYTES_W", .val = "16",
                         .ts = 1753471260}));

  TuneDB const next = a.reread();
  CHECK(next.isNogo(1, "512:15:512:212", UseConfig{{"PAD", "256"}, {"SHUFL_BYTES_W", "16"}}));
  CHECK(!next.isNogo(1, "512:15:512:212", UseConfig{{"SHUFL_BYTES_W", "8"}}));
  // One FFT only: the budget that will not fit here may fit at another shape.
  CHECK(!next.isNogo(1, "256:2:256:212", UseConfig{{"SHUFL_BYTES_W", "16"}}));
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
}
