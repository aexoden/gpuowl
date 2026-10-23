// Copyright (C) Jason Lynch

#include "test.h"

#include "Lanes.h"
#include "TuneDB.h"

#include <cmath>
#include <string>

using namespace tune;

namespace {

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

}  // namespace

TEST(a_timetable_reads_and_writes_its_shares) {
  std::optional<Timetable> const t = parseTimetable("70/20/10/0~30/30/20/20@4h");
  CHECK(t.has_value());
  CHECK(*t == Timetable{});
  CHECK_EQ(toString(*t), std::string{"70/20/10/0~30/30/20/20@4h"});
  CHECK_EQ(toString(*parseTimetable("50/25/15/10")), std::string{"50/25/15/10"});
  CHECK_EQ(toString(*parseTimetable("model")), std::string{"model"});
  CHECK_EQ(toString(*parseTimetable("70/20/10/0~30/30/20/20@90m")), std::string{"70/20/10/0~30/30/20/20@90m"});
  for (const char* bad : {"", "70/20/10", "70/20/10/1", "70/20/10/0~30/30/20/20", "70/20/10/0~30/30/20/20@4",
                          "70/20/10/0~30/30/20/20@0h", "-10/90/10/10", "x/20/10/70"}) {
    CHECK(!parseTimetable(bad).has_value());
  }
}

TEST(a_lane_is_owed_the_integral_of_its_share) {
  Timetable const t{};
  CHECK(near(t.at(0)[0], 0.7));
  CHECK(near(t.at(2 * 3600)[0], 0.5));
  CHECK(near(t.at(10 * 3600)[0], 0.3));
  // Over the 4-hour ramp the model averages 50%, then takes 30%.
  CHECK(near(t.owed(4 * 3600)[0], 0.5 * 4 * 3600));
  CHECK(near(t.owed(6 * 3600)[0], 0.5 * 4 * 3600 + 0.3 * 2 * 3600));
  // In the first half hour, nearly the starting split.
  CHECK(near(t.owed(1800)[1], 1800 * (0.2 + (0.2 + 0.1 * 1800 / 14'400)) / 2));
  LaneShares total = t.owed(5 * 3600);
  CHECK(near(total[0] + total[1] + total[2] + total[3], 5 * 3600));
}

TEST(the_lane_furthest_behind_goes_first) {
  Timetable const t = *parseTimetable("70/20/10/0");
  LaneTime time;
  time.seconds = {70, 10, 20, 0};
  auto const order = laneOrder(t, time);
  CHECK(order[0] == Lane::Coverage);
  CHECK(order[1] == Lane::Model);
  CHECK(order[2] == Lane::Restarts);
  CHECK(order[3] == Lane::Interactions);
  CHECK(!t.serves(Lane::Interactions));
}

TEST(lane_time_runs_from_each_item_to_the_next) {
  TuneDB db;
  CHECK(db.parse(std::string{TuneDB::HEADER} + '\n', "lanes"));
  u32 const env = db.internEnv(DbEnv{.gpu = "sim", .isNvidia = true, .cudaBackend = true, .computeCapability = 860});
  u32 const sess = db.beginSession(env, "", 0, "", 90);
  u32 const cfg = db.internCfg({});
  auto item = [&](const char* lane, u64 ts) {
    CHECK(db.add(ItemRow{.sess = sess,
                         .lane = lane,
                         .what = "x",
                         .packet = 0,
                         .ordinal = 1,
                         .fft = "1K:13:256:202",
                         .kind = TestKind::LL,
                         .cfg = cfg,
                         .ts = ts}));
  };
  auto call = [&](u64 ts) {
    CHECK(db.add(CallRow{.sess = sess,
                         .fft = "1K:13:256:202",
                         .kind = TestKind::LL,
                         .exponent = 67'000'000,
                         .cfg = cfg,
                         .mean = 500,
                         .blocks = 4,
                         .ts = ts}));
  };
  item("model", 100);
  call(108);
  item("coverage", 110);
  call(112);
  item("restarts", 112);
  call(130);
  item(DUTY, 130);
  call(130);

  LaneTime const t = laneTime(db, env);
  CHECK(near(t.seconds[0], 10));
  CHECK(near(t.seconds[1], 2));
  CHECK(near(t.seconds[2], 18));
  CHECK(near(t.duty, 1));  // at least a second

  // Read back from the file's text, the same.
  TuneDB again;
  CHECK(again.parse(db.text(), "again"));
  CHECK_EQ(again.items().size(), size_t{4});
  CHECK_EQ(formatRow(again.items()[1]), formatRow(db.items()[1]));
  CHECK(near(laneTime(again, env).seconds[2], 18));
  // An item answers no attempt: one open before it stays open.
  CHECK(db.add(
    TryRow{.sess = sess, .fft = "1K:13:256:202", .kind = TestKind::LL, .exponent = 67'000'000, .cfg = cfg, .ts = 131}));
  item("model", 132);
  CHECK(db.attemptOpen(sess));
}
