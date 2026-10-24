// Copyright (C) Jason Lynch

#include "test.h"

#include "TuneDB.h"

#include <string>

using namespace tune;

namespace {

DbEnv card() {
  return DbEnv{.gpu = "NVIDIA RTX A4000",
               .driver = "570.133",
               .isNvidia = true,
               .cudaBackend = true,
               .computeCapability = 860,
               .machine = "0000:01:00.0",
               .build = 0x0123456789abcdefull};
}

// A database with one env, one session and one option set, unattached.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  u32 cfg = 0;

  Fixture() {
    CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "fixture"));
    env = db.internEnv(card());
    sess = db.beginSession(env, "1K:13:256:202@67513549", 0, "workload=67M-72M", 1000);
    cfg = db.internCfg({{"INPLACE", "0"}, {"PAD", "64"}});
  }

  CallRow call(double mean) const {
    return CallRow{.sess = sess,
                   .fft = "512:15:512:202",
                   .kind = TestKind::LL,
                   .exponent = 67'513'549,
                   .cfg = cfg,
                   .mean = mean,
                   .sd = 0.25,
                   .blocks = 4,
                   .drift = 1.0125,
                   .buildSec = 4.5,
                   .timedSec = 1.25,
                   .ts = 1001};
  }
};

}  // namespace

TEST(db_round_trips_every_record_in_order) {
  Fixture f;
  CHECK(f.db.add(TryRow{
    .sess = f.sess, .fft = "512:15:512:202", .kind = TestKind::LL, .exponent = 67'513'549, .cfg = f.cfg, .ts = 1001}));
  CHECK(f.db.add(f.call(812.5)));
  CHECK(f.db.add(RoeRow{.sess = f.sess,
                        .fft = "512:15:512:202",
                        .exponent = 72'000'000,
                        .cfg = f.cfg,
                        .checkOk = true,
                        .z = 31.25,
                        .n = 9000,
                        .maxRoe = 0.25,
                        .ts = 1002}));
  CHECK(f.db.add(AnchorRow{.sess = f.sess,
                           .fft = "1K:13:256:202",
                           .exponent = 67'513'549,
                           .cfg = f.cfg,
                           .mean = 967.5,
                           .ratio = 1,
                           .ts = 1003}));
  CHECK(f.db.add(RegsRow{.sess = f.sess,
                         .fft = "512:15:512:202",
                         .kind = TestKind::LL,
                         .key = "REGCF64",
                         .resources = {.regs = 96, .localBytes = 0, .sharedBytes = 8192, .threads = 256},
                         .plan = {.k0 = 2, .occupancies = {{2, 128}, {3, 80}}, .lift = true},
                         .ts = 1004}));
  CHECK(f.db.add(RefRow{.sess = f.sess,
                        .fft = "512:15:512:202",
                        .exponent = 67'513'549,
                        .iters = 5000,
                        .res64 = 0x4eeadb0011223344ull,
                        .ts = 1005}));
  CHECK(f.db.add(KresRow{.sess = f.sess,
                         .fft = "512:15:512:202",
                         .kind = TestKind::LL,
                         .key = "REGCF64",
                         .resources = {.regs = 128, .localBytes = 16, .sharedBytes = 8192, .threads = 256},
                         .cfg = f.cfg,
                         .ts = 1006}));

  std::string const text = f.db.text();
  TuneDB back;
  CHECK(back.parse(text, "round trip"));
  CHECK_EQ(back.text(), text);
  CHECK_EQ(back.calls().size(), size_t{1});
  CHECK(back.calls()[0].kind == TestKind::LL);
  CHECK(back.calls()[0].mean == 812.5);
  CHECK(back.calls()[0].cost() == 812.5 / 1.0125);
  CHECK_EQ(back.roes().size(), size_t{1});
  CHECK(back.regs()[0].plan == (RegisterPlan{.k0 = 2, .occupancies = {{2, 128}, {3, 80}}, .lift = true}));
  CHECK_EQ(back.refs()[0].res64, 0x4eeadb0011223344ull);
  CHECK_EQ(back.kres().size(), size_t{1});
  CHECK_EQ(back.kres()[0].resources.localBytes, 16);
  CHECK_EQ(back.kres()[0].cfg, f.cfg);
  CHECK_EQ(back.findSession(f.sess)->settings, std::string{"workload=67M-72M"});
  CHECK_EQ(back.envAnchor(f.env), std::string{"1K:13:256:202@67513549"});
  CHECK(back.envBaseline(f.env) != nullptr);
  CHECK(back.findEnv(f.env)->sameMachine(card()));
}

TEST(db_names_configurations_canonically) {
  Fixture f;
  CallRow row = f.call(800);
  row.fft = "1:512:8:512:010";  // a pure NTT: the variant folds onto 202
  CHECK(f.db.add(row));
  CHECK_EQ(f.db.calls().back().fft, std::string{"1:512:8:512:202"});

  row.fft = "1:512:8";
  CHECK(!f.db.add(row));
}

TEST(db_interns_equal_option_sets_and_envs_once) {
  Fixture f;
  CHECK_EQ(f.db.internCfg({{"PAD", "64"}, {"INPLACE", "0"}}), f.cfg);
  CHECK(f.db.internCfg({}) != f.cfg);
  CHECK_EQ(f.db.internEnv(card()), f.env);

  DbEnv rebuilt = card();
  rebuilt.build = 1;
  CHECK(f.db.internEnv(rebuilt) != f.env);
  CHECK(f.db.findEnv(f.env)->sameCard(rebuilt));
}

TEST(db_names_what_a_process_died_holding) {
  Fixture f;
  TryRow const attempt{
    .sess = f.sess, .fft = "512:15:512:202", .kind = TestKind::PRP, .exponent = 67'513'549, .cfg = f.cfg, .ts = 1001};
  CHECK(f.db.add(attempt));
  CHECK(f.db.add(f.call(800)));
  CHECK(f.db.diedHolding().empty());

  CHECK(f.db.add(attempt));
  CHECK_EQ(f.db.diedHolding().size(), size_t{1});
  CHECK(f.db.diedOn(f.env, "512:15:512:202", TestKind::PRP, 67'513'549, f.cfg));
  CHECK(!f.db.diedOn(f.env, "512:15:512:202", TestKind::LL, 67'513'549, f.cfg));

  // Read back, the open attempt is still open.
  TuneDB back;
  CHECK(back.parse(f.db.text(), "died"));
  CHECK_EQ(back.diedHolding().size(), size_t{1});

  CHECK(f.db.add(DoneRow{.sess = f.sess, .ts = 1002}));
  CHECK(f.db.diedHolding().empty());
}

TEST(db_keeps_records_a_later_build_writes) {
  Fixture f;
  std::string const text = f.db.text() + "future 1 something new\n";
  TuneDB back;
  CHECK(back.parse(text, "future"));
  CHECK_EQ(back.unknownRows().size(), size_t{1});
  CHECK_EQ(back.text(), text);
}

TEST(db_refuses_what_it_cannot_trust) {
  Fixture f;
  std::string const good = f.db.text();
  TuneDB back;

  CHECK(!back.parse("# prpll tunedb v1\n", "old"));
  CHECK(!back.parse(good + "call 1 512:15:512:202 ll 67513549 1 ok 800.000\n", "short call"));
  CHECK(!back.parse(good + "call 9 512:15:512:202 ll 67513549 1 ok 800.000 0.250 4 0 1.0000 4.50 1.25 1001\n",
                    "unknown session"));
  CHECK(!back.parse(good + "call 1 512:15:512:202 ll 67513549 7 ok 800.000 0.250 4 0 1.0000 4.50 1.25 1001\n",
                    "unknown cfg"));
  CHECK(!back.parse(good + "cfg 1 PAD=64\n", "duplicate cfg"));
}

TEST(db_drops_an_unfinished_last_record) {
  Fixture f;
  CHECK(f.db.add(f.call(800)));
  std::string text = f.db.text();
  text += "call 1 512:15:512:202 ll 675";
  TuneDB back;
  CHECK(back.parse(text, "torn"));
  CHECK_EQ(back.calls().size(), size_t{1});
}

TEST(db_parses_configs_and_specs) {
  CHECK_EQ(configText({}), std::string{"-"});
  CHECK_EQ(configText({{"B", "2"}, {"A", "1"}}), std::string{"A=1,B=2"});
  CHECK(parseConfigText("A=1,B=2") == (UseConfig{{"A", "1"}, {"B", "2"}}));
  CHECK(!parseConfigText("A=1,,B=2"));
  CHECK(!parseConfigText("A"));
  CHECK(!parseConfigText("A=1,A=2"));

  CHECK(parseFft("1K:13:256:202").has_value());
  CHECK(parseFft("2:512:4:512:212").has_value());
  CHECK(!parseFft("7:512:4:512:212"));
  CHECK(!parseFft("1K:13:256:999"));
  CHECK(!parseFft("1K:13"));
}

TEST(lock_listing_names_holders_not_waiters) {
  std::string_view const locks = "1: FLOCK  ADVISORY  WRITE 1234 08:02:5551212 0 EOF\n"
                                 "2: -> FLOCK  ADVISORY  WRITE 1235 08:02:777 0 EOF\n";
  CHECK(lockListed(locks, 8, 2, 5'551'212));
  CHECK(!lockListed(locks, 8, 2, 777));
  CHECK(!lockListed(locks, 8, 3, 5'551'212));
}
