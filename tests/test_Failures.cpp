// Copyright (C) Jason Lynch

// Failures worth a person's look: the log they go to, what it says of each, and what a death holds out.

#include "test.h"

#include "Failures.h"
#include "Readings.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

// A database written by a process that ended with an attempt open -- as another process reads it -- and a log
// directory of its own.
struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;
  u32 cfg = 0;
  std::filesystem::path dir;

  Fixture() {
    TuneDB writer;
    CHECK(writer.parse(std::string{TuneDB::HEADER} + '\n', "writer"));
    env = writer.internEnv(DbEnv{.gpu = "card", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
    sess = writer.beginSession(env, "", 0, "");
    cfg = writer.internCfg({});
    auto call = [&](const char* fft, u64 exponent, Status status, u64 ts) {
      CHECK(writer.add(CallRow{.sess = sess,
                               .fft = fft,
                               .kind = TestKind::PRP,
                               .exponent = exponent,
                               .cfg = cfg,
                               .status = status,
                               .mean = status == Status::Ok ? 800.0 : 0.0,
                               .sd = 1,
                               .blocks = status == Status::Ok ? 4u : 0u,
                               .ts = ts}));
    };
    call("512:15:512:202", 67'000'000, Status::Ok, 10);
    call("512:15:512:202", 67'000'000, Status::Ok, 11);
    call("512:15:512:212", 67'000'000, Status::Err, 12);
    call("512:15:512:200", 67'000'000, Status::NoCompile, 13);
    // Last: what the process held when it died, an LL call at another exponent.
    CHECK(writer.add(TryRow{
      .sess = sess, .fft = "512:15:512:202", .kind = TestKind::LL, .exponent = 69'000'000, .cfg = cfg, .ts = 14}));
    CHECK(db.parse(writer.text(), "reader"));

    dir = std::filesystem::temp_directory_path() / ("prpll-test-failures-" + std::to_string(getpid()));
    std::filesystem::remove_all(dir);
  }

  ~Fixture() { std::filesystem::remove_all(dir); }

  [[nodiscard]] std::string log() const {
    std::ifstream in{dir / FAILURES_LOG};
    return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
  }
};

bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

TEST(a_failure_entry_says_what_reproduces_it) {
  std::string const text = failureText({.kind = FailureKind::Wrong,
                                        .fft = "1:512:4:512:212",
                                        .test = TestKind::LL,
                                        .exponent = 57'885'161,
                                        .options = "WMUL=4",
                                        .device = "card",
                                        .detail = "LL residue twice",
                                        .at = 1,
                                        .id = "call 1 2 1:512:4:512:212 3"});
  CHECK(has(text, "  computed a wrong answer\n"));
  CHECK(has(text, "\n  config  -fft 1:512:4:512:212 -use WMUL=4   (ll at 57885161)\n"));
  CHECK(has(text, "\n  repro   prpll -fft 1:512:4:512:212 -use WMUL=4 -ll 57885161 -iters 20000\n"));
  CHECK(has(text, "\n  device  card\n"));
  CHECK(has(text, "\n  detail  LL residue twice\n"));
  CHECK(has(text, "\n  id      call 1 2 1:512:4:512:212 3\n"));
  // The built-in defaults name no -use.
  CHECK(has(failureText({.fft = "512:15:512:202", .exponent = 7}), "-fft 512:15:512:202 -prp 7 -iters 20000\n"));
}

TEST(the_failure_log_holds_each_failure_once) {
  Fixture f;
  CHECK_EQ(recordMissingFailures(f.db, f.env, f.dir), 3u);
  std::string const text = f.log();
  CHECK(has(text, "took the device down\n  config  -fft 512:15:512:202   (ll at 69000000)\n"));
  CHECK(has(text, "computed a wrong answer\n  config  -fft 512:15:512:212   (prp at 67000000)\n"));
  CHECK(has(text, "would not build\n  config  -fft 512:15:512:200   (prp at 67000000)\n"));
  CHECK_EQ(recordedFailureIds(f.dir).size(), size_t{3});
  CHECK(recordedFailureIds(f.dir).contains("attempt " + std::to_string(f.sess) + " 14"));

  // Written once: a later run, or one that wrote an entry as the failure happened, adds nothing.
  CHECK_EQ(recordMissingFailures(f.db, f.env, f.dir), 0u);
  CHECK_EQ(f.log(), text);
}

TEST(a_run_says_how_many_configurations_are_held_out) {
  Fixture f;
  FailureTally const tally = failureTally(f.db, f.env);
  CHECK_EQ(tally.deviceLost, 1u);
  CHECK_EQ(tally.wrong, 1u);
  CHECK_EQ(tally.noBuild, 1u);
  CHECK_EQ(toString(tally), std::string{"3 configurations held out: 1 took the device down, 1 computed a wrong answer, "
                                        "1 would not build; each is in failures/failures.log"});
  CHECK(toString(FailureTally{}).empty());
}

TEST(what_a_process_died_holding_is_failed_at_every_exponent_and_kind) {
  Fixture f;
  Readings const readings{f.db, f.env, nvidia()};
  // It read a cost as PRP at 67M before it took the device down as LL at 69M: that reading is failed too.
  bool seen = false;
  for (const auto& [key, r] : readings.all()) {
    if (key.fft != "512:15:512:202") { continue; }
    seen = true;
    CHECK(r.failed);
  }
  CHECK(seen);
  CHECK(readings.died(FFTConfig{"512:15:512:202"}, {}));
  CHECK(!readings.died(FFTConfig{"512:15:512:212"}, {}));
}
