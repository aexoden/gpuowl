// Copyright (C) Jason Lynch

// GPU-free tests of the LL reference residue (src/LLCheck.cpp): the vote, the witnesses it asks, and settling it.

#include "test.h"

#include "Eligibility.h"
#include "FFTVariants.h"
#include "LLCheck.h"

#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>

using namespace tune;

namespace {

// For CHECK_EQ's message.
std::ostream& operator<<(std::ostream& os, const std::optional<u64>& r) {
  return r ? os << std::hex << *r << std::dec : os << "none";
}

constexpr u64 E = 118'063'003;
constexpr u64 ITERS = 5000;
constexpr u64 RIGHT = 0x1111'2222'3333'4444;
constexpr u64 WRONG = 0x5555'6666'7777'8888;

RefRow reading(const std::string& fft, u64 res64, u64 exponent = E, u64 iters = ITERS, u32 sess = 1) {
  return {.sess = sess, .fft = FFTConfig{fft}.spec(), .exponent = exponent, .iters = iters, .res64 = res64, .ts = 1};
}

struct Fixture {
  TuneDB db;
  u32 env = 0;
  u32 sess = 0;

  Fixture() {
    env = db.internEnv(DbEnv{.gpu = "a card", .name = "a card", .driver = "1.0"});
    sess = db.beginSession(env, "");
  }
};

// Answers each witness from `residues`, and records which were asked.
struct Witnesses {
  std::map<std::string, std::optional<u64>> residues;
  std::vector<std::string> asked{};

  [[nodiscard]] ReadWitness reader() {
    return [this](const FFTConfig& fft) -> std::optional<u64> {
      asked.push_back(fft.spec());
      auto const at = residues.find(fft.spec());
      return at == residues.end() ? std::optional<u64>{} : at->second;
    };
  }
};

std::vector<FFTConfig> ffts(const std::vector<std::string>& specs) {
  std::vector<FFTConfig> out;
  for (const std::string& spec : specs) { out.emplace_back(spec); }
  return out;
}

std::string spec(const std::string& text) { return FFTConfig{text}.spec(); }

}  // namespace

TEST(no_single_reading_is_a_reference) {
  CHECK(!agreedResidue({}));
  CHECK(!agreedResidue({reading("1K:8:1K:202", RIGHT)}));

  // Nor is one FFT read twice: a bug of that FFT reproduces itself.
  CHECK(!agreedResidue({reading("1K:8:1K:202", RIGHT), reading("1K:8:1K:202", RIGHT)}));
}

TEST(two_ffts_that_agree_make_the_reference) {
  CHECK_EQ(agreedResidue({reading("1K:8:1K:202", RIGHT), reading("2:1K:8:512:202", RIGHT)}), std::optional{RIGHT});
}

TEST(a_disagreement_is_settled_by_a_third_fft) {
  CHECK(!agreedResidue({reading("1K:8:1K:202", RIGHT), reading("2:1K:8:512:202", WRONG)}));
  CHECK_EQ(
    agreedResidue({reading("1K:8:1K:202", RIGHT), reading("2:1K:8:512:202", WRONG), reading("3:1K:8:512", RIGHT)}),
    std::optional{RIGHT});
  CHECK_EQ(
    agreedResidue({reading("1K:8:1K:202", WRONG), reading("2:1K:8:512:202", RIGHT), reading("3:1K:8:512", RIGHT)}),
    std::optional{RIGHT});
}

TEST(an_agreement_is_never_revised) {
  // Whatever comes later, the first two to agree decided it.
  CHECK_EQ(agreedResidue({reading("1K:8:1K:202", RIGHT), reading("2:1K:8:512:202", RIGHT), reading("3:1K:8:512", WRONG),
                          reading("51:1K:8:512", WRONG), reading("1:1K:8:512", WRONG)}),
           std::optional{RIGHT});
}

TEST(readings_count_only_at_their_exponent_iterations_and_env) {
  Fixture f;
  u32 const otherEnv = f.db.internEnv(DbEnv{.gpu = "another card", .name = "another card", .driver = "1.0"});
  u32 const otherSess = f.db.beginSession(otherEnv, "");

  CHECK(f.db.add(reading("1K:8:1K:202", RIGHT, E, ITERS, f.sess)));
  CHECK(f.db.add(reading("2:1K:8:512:202", RIGHT, E, ITERS, otherSess)));      // another card
  CHECK(f.db.add(reading("3:1K:8:512", RIGHT, E, 2 * ITERS, f.sess)));         // another length
  CHECK(f.db.add(reading("51:1K:8:512", RIGHT, 118'062'997, ITERS, f.sess)));  // another exponent

  std::vector<RefRow> const mine = referenceReadings(f.db, f.env, E, ITERS);
  CHECK_EQ(mine.size(), size_t{1});
  CHECK(!agreedResidue(mine));

  CHECK(f.db.add(reading("2:1K:8:512:202", RIGHT, E, ITERS, f.sess)));
  CHECK_EQ(agreedResidue(referenceReadings(f.db, f.env, E, ITERS)), std::optional{RIGHT});
}

TEST(only_a_set_that_moves_no_tunable_key_is_at_built_in_defaults) {
  CHECK(atBuiltInDefaults({}));
  CHECK(atBuiltInDefaults({{"DEBUG", "1"}}));  // recognised, never searched
  CHECK(!atBuiltInDefaults({{"TAIL_KERNELS", "3"}}));
  CHECK(!atBuiltInDefaults({{"DEBUG", "1"}, {"IN_WG", "128"}}));
}

TEST(witnesses_start_with_the_fft_timed_then_share_as_little_with_it_as_the_device_allows) {
  FFTConfig const own{"1K:8:1K:202"};
  std::vector<FFTConfig> const order = witnessOrder(Env{.isNvidia = true}, own, E);

  CHECK(order.size() >= 3);
  CHECK_EQ(order.front().spec(), own.spec());

  // Next, the FFTs with no arithmetic in common with FP64, then those sharing some, then another FP64 shape.
  auto tier = [&own](const FFTConfig& fft) {
    FFTParts const a = fftParts(own.shape.fft_type);
    FFTParts const b = fftParts(fft.shape.fft_type);
    bool const shares = (a.fp64 && b.fp64) || (a.fp32 && b.fp32) || (a.gf31 && b.gf31) || (a.gf61 && b.gf61);
    return !shares ? 0 : fft.shape.fft_type != own.shape.fft_type ? 1 : 2;
  };
  CHECK_EQ(tier(order[1]), 0);
  std::set<std::string> seen{own.spec()};
  for (size_t i = 1; i < order.size(); ++i) {
    CHECK(seen.insert(order[i].spec()).second);
    CHECK(!interval(order[i], E).empty());
    if (i > 1) { CHECK(tier(order[i - 1]) <= tier(order[i])); }
  }
}

TEST(witnesses_are_only_what_the_device_can_compile) {
  FFTConfig const own{"1:1K:8:512:202"};
  std::vector<FFTConfig> const order = witnessOrder(Env{.isAmd = true, .hasFP64 = false}, own, E);

  CHECK(order.size() >= 2);
  for (const FFTConfig& fft : order) { CHECK(!fftParts(fft.shape.fft_type).fp64); }
}

TEST(two_witnesses_that_agree_settle_the_reference_and_are_recorded) {
  Fixture f;
  Witnesses w{.residues = {{spec("1K:8:1K:202"), RIGHT}, {spec("2:1K:8:512:202"), RIGHT}, {spec("3:1K:8:512"), RIGHT}}};

  CHECK_EQ(settleReference(f.db, f.sess, E, ITERS, ffts({"1K:8:1K:202", "2:1K:8:512:202", "3:1K:8:512"}), w.reader()),
           std::optional{RIGHT});
  CHECK_EQ(w.asked.size(), size_t{2});
  CHECK_EQ(f.db.refs().size(), size_t{2});
  CHECK_EQ(f.db.refs().back().res64, RIGHT);

  // Settled, it is read off the database and nothing is asked again.
  Witnesses again;
  CHECK_EQ(settleReference(f.db, f.sess, E, ITERS, ffts({"1K:8:1K:202"}), again.reader()), std::optional{RIGHT});
  CHECK(again.asked.empty());
}

TEST(a_witness_that_disagrees_is_outvoted_by_the_next) {
  Fixture f;
  Witnesses w{.residues = {{spec("1K:8:1K:202"), RIGHT}, {spec("2:1K:8:512:202"), WRONG}, {spec("3:1K:8:512"), RIGHT}}};

  CHECK_EQ(settleReference(f.db, f.sess, E, ITERS, ffts({"1K:8:1K:202", "2:1K:8:512:202", "3:1K:8:512"}), w.reader()),
           std::optional{RIGHT});
  CHECK_EQ(w.asked.size(), size_t{3});
  CHECK_EQ(f.db.refs().size(), size_t{3});
}

TEST(a_witness_that_gives_no_reading_is_passed_over_and_not_counted) {
  Fixture f;
  Witnesses w{.residues = {{spec("1K:8:1K:202"), RIGHT}, {spec("3:1K:8:512"), RIGHT}}};

  CHECK_EQ(settleReference(f.db, f.sess, E, ITERS, ffts({"1K:8:1K:202", "2:1K:8:512:202", "3:1K:8:512"}), w.reader()),
           std::optional{RIGHT});
  CHECK_EQ(w.asked.size(), size_t{3});
  CHECK_EQ(f.db.refs().size(), size_t{2});
}

TEST(no_reference_stands_where_the_witnesses_run_out_without_two_agreeing) {
  Fixture f;
  std::vector<std::string> const specs{"1K:8:1K:202", "2:1K:8:512:202", "3:1K:8:512", "51:1K:8:512", "1:1K:8:512"};
  Witnesses w;
  u64 residue = 1;
  for (const std::string& s : specs) { w.residues[spec(s)] = residue++; }

  CHECK(!settleReference(f.db, f.sess, E, ITERS, ffts(specs), w.reader()));
  CHECK_EQ(w.asked.size(), size_t{LL_MAX_WITNESSES});

  // A later session does not read them again, nor more of them.
  Witnesses later{.residues = w.residues};
  CHECK(!settleReference(f.db, f.sess, E, ITERS, ffts(specs), later.reader()));
  CHECK(later.asked.empty());
}

TEST(a_later_session_reads_only_the_witnesses_not_yet_read) {
  Fixture f;
  CHECK(f.db.add(reading("1K:8:1K:202", RIGHT, E, ITERS, f.sess)));

  u32 const later = f.db.beginSession(f.env, "");
  Witnesses w{.residues = {{spec("1K:8:1K:202"), WRONG}, {spec("2:1K:8:512:202"), RIGHT}}};
  CHECK_EQ(settleReference(f.db, later, E, ITERS, ffts({"1K:8:1K:202", "2:1K:8:512:202"}), w.reader()),
           std::optional{RIGHT});
  CHECK(w.asked == std::vector<std::string>{spec("2:1K:8:512:202")});
}
