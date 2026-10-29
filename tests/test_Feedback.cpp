// Copyright (C) Jason Lynch

// What production sustains: the lines it writes, the stretch they summarise, and what the tuner makes of them.

#include "test.h"

#include "BuildId.h"
#include "Eligibility.h"
#include "Feedback.h"
#include "OptionSpace.h"
#include "Report.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace tune;

namespace {

Env nvidia() { return {.isNvidia = true, .cudaBackend = true, .computeCapability = 860}; }

bool near(double a, double b, double tolerance = 1e-9) { return std::abs(a - b) < tolerance; }

constexpr u64 E = 67'249'139;
constexpr u32 STEP = 1'000'000;

// A working directory of its own, so what a ProductionRate appends lands there.
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
};

FFTConfig fft() { return FFTConfig{"2:512:4:512:202"}; }

FeedbackRow row(double us, const std::map<std::string, std::string>& settings, u64 iters = 60'000'000) {
  return {.usPerIt = us,
          .fft = fft().spec(),
          .kind = TestKind::LL,
          .regime = regimeOf(fft(), E).label(),
          .exponent = E,
          .steps = 60,
          .iters = iters,
          .build = buildFingerprint(),
          .ts = 1'761'674'436,
          .settings = settings};
}

}  // namespace

TEST(a_feedback_line_reads_back_as_written) {
  FeedbackRow const out = row(181.62, {{"LOADS", "5502"}, {"HOIST_W", "1"}});
  std::string const text = feedbackLine(out);
  std::optional<FeedbackRow> const back = parseFeedbackLine(text);
  CHECK(back.has_value());
  CHECK(near(back->usPerIt, 181.62));
  CHECK_EQ(back->fft, out.fft);
  CHECK(back->kind == TestKind::LL);
  CHECK_EQ(back->regime, out.regime);
  CHECK_EQ(back->exponent, E);
  CHECK_EQ(back->steps, 60u);
  CHECK_EQ(back->iters, u64{60'000'000});
  CHECK_EQ(back->build, buildFingerprint());
  CHECK_EQ(back->ts, u64{1'761'674'436});
  CHECK(back->settings == out.settings);
  CHECK_EQ(feedbackLine(*back), text);
  CHECK(parseFeedbackLine(text + "\n").has_value());  // as a file's lines are read

  // No settings at all is "-", as configText writes it.
  CHECK(parseFeedbackLine(feedbackLine(row(200, {})))->settings.empty());
  CHECK(!parseFeedbackLine("181.6 2:512:4:512:202 ll short64 67249139"));
  CHECK(!parseFeedbackLine("# prpll feedback"));
}

TEST(a_stretch_leaves_out_its_first_and_short_steps_and_takes_the_median) {
  Here const here{"prpll-test-feedback-median"};
  ProductionRate rate{E, fft(), TestKind::LL, {{"LOADS", "5502"}}, STEP};
  rate.restart(0);
  rate.step(STEP, 186.6e-6);  // the load and the warm-up
  CHECK(!rate.row());
  rate.step(2 * STEP, 181.6e-6);
  rate.step(3 * STEP, 182.8e-6);
  CHECK(!rate.row());  // below FEEDBACK_MIN_STEPS
  rate.step(4 * STEP, 199.0e-6);  // a disturbed step: the median passes it over
  rate.step(4 * STEP + 1000, 300e-6);  // the short last step of a test
  std::optional<FeedbackRow> const out = rate.row();
  CHECK(out.has_value());
  CHECK_EQ(out->steps, 3u);
  CHECK_EQ(out->iters, u64{3} * STEP);
  CHECK(near(out->usPerIt, 182.8, 1e-6));
  CHECK_EQ(out->settings.at("LOADS"), std::string{"5502"});
}

TEST(a_stretch_is_appended_when_it_ends_or_the_test_reloads) {
  Here const here{"prpll-test-feedback-append"};
  {
    ProductionRate rate{E, fft(), TestKind::LL, {}, STEP};
    rate.restart(0);
    for (u32 i = 1; i <= 5; ++i) { rate.step(u64(i) * STEP, 180e-6); }
    rate.restart(5 * STEP);  // a reload: the stretch so far is written, and the next step warms up again
    CHECK_EQ(readFeedback(FEEDBACK_FILE).size(), size_t{1});
    rate.step(6 * STEP, 500e-6);
    for (u32 i = 7; i <= 9; ++i) { rate.step(u64(i) * STEP, 190e-6); }
  }
  std::vector<FeedbackRow> const rows = readFeedback(FEEDBACK_FILE);
  CHECK_EQ(rows.size(), size_t{2});
  if (rows.size() != 2) { return; }
  CHECK(near(rows[0].usPerIt, 180, 1e-6));
  CHECK_EQ(rows[0].steps, 4u);
  CHECK(near(rows[1].usPerIt, 190, 1e-6));
  CHECK_EQ(rows[1].steps, 3u);

  // A stretch too short to say anything writes nothing.
  { ProductionRate rate{E, fft(), TestKind::LL, {}, STEP}; rate.step(STEP, 180e-6); }
  CHECK_EQ(readFeedback(FEEDBACK_FILE).size(), size_t{2});
}

TEST(production_is_matched_to_a_configuration_by_its_canonical_settings) {
  // A setting at its default, or unknown to the option table, does not make another configuration.
  std::map<std::string, std::string> const tuned{{"LOADS", "5502"}};
  std::map<std::string, std::string> withExtra = tuned;
  withExtra["NOT_AN_OPTION"] = "1";
  std::vector<FeedbackRow> const rows{row(180, tuned, 30'000'000), row(186, withExtra, 10'000'000),
                                      row(250, {}, 60'000'000)};
  auto other = row(170, tuned);
  other.build ^= 1;  // another build's kernels say nothing of this one's
  std::vector<FeedbackRow> all = rows;
  all.push_back(other);

  std::map<ConfigKey, Sustained> const got = sustainedOf(all, buildFingerprint(), nvidia());
  Subject const subject{.fft = fft(), .kind = TestKind::LL, .regime = regimeOf(fft(), E)};
  auto const it = got.find(Readings::keyOf(subject, canonicalConfig(nvidia(), fft(), tuned)));
  CHECK(it != got.end());
  CHECK_EQ(it->second.stretches, 2u);
  CHECK_EQ(it->second.iters, u64{40'000'000});
  CHECK(near(it->second.usPerIt, 181.5));  // weighted by iterations
  CHECK_EQ(got.size(), size_t{2});
}

TEST(the_gap_is_what_production_sustained_over_the_published_cost) {
  Subject const subject{.fft = fft(), .kind = TestKind::LL, .regime = regimeOf(fft(), E)};
  UseConfig const tuned = canonicalConfig(nvidia(), fft(), {{"LOADS", "5502"}});
  Reading const reading{.fft = fft(), .kind = TestKind::LL, .regime = subject.regime, .options = tuned, .calls = 4,
                        .mean = 166.9, .se = 0};
  Reading const unrun{.fft = fft(), .kind = TestKind::LL, .regime = subject.regime, .options = {}, .calls = 4,
                      .mean = 190, .se = 0};
  Publication publication;
  publication.kinds.push_back({.kind = TestKind::LL,
                               .lines = {},
                               .written = {Line{.subject = subject, .reading = &reading},
                                           Line{.subject = subject, .reading = &unrun}}});

  CHECK_EQ(productionGap(publication, {}).lines, 0u);
  CHECK(std::isnan(productionGap(publication, {}).ratio));

  std::map<ConfigKey, Sustained> const production{
    {Readings::keyOf(subject, tuned), Sustained{.usPerIt = 181.9, .stretches = 3, .iters = 180'000'000}}};
  ProductionGap const gap = productionGap(publication, production);
  CHECK_EQ(gap.lines, 1u);
  CHECK(near(gap.ratio, 181.9 / 166.9));
  CHECK(sustainedOn(publication.kinds[0].written[1], production) == nullptr);

  RunReport const report{.publication = &publication, .production = &production};
  std::string const json = reportJson(report);
  CHECK(json.find("\"production\": {\"lines\": 1, \"ratio\": 1.08987}") != std::string::npos);
  CHECK(json.find("\"production\": {\"usPerIt\": 181.9, \"ratio\": 1.08987, \"stretches\": 3") != std::string::npos);
  CHECK(json.find("\"production\": null}") != std::string::npos);
}
