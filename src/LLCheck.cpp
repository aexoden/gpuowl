// Copyright (C) Jason Lynch

#include "LLCheck.h"

#include "Anchor.h"
#include "Bootstrap.h"
#include "FFTVariants.h"
#include "Objective.h"

#include <algorithm>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace tune {

bool atBuiltInDefaults(const Env& env, const FFTConfig& fft, const UseConfig& ran) {
  return canonicalConfig(env, fft, ran).empty();
}

std::vector<RefRow> referenceReadings(const TuneDB& db, u32 env, u64 exponent, u64 iters) {
  std::vector<RefRow> out;
  for (const RefRow& row : db.refs()) {
    if (row.exponent == exponent && row.iters == iters && db.envOf(row.sess) == env) { out.push_back(row); }
  }
  return out;
}

std::optional<u64> agreedResidue(const std::vector<RefRow>& readings) {
  std::map<u64, std::set<std::string>> readBy;
  for (const RefRow& row : readings) {
    std::set<std::string>& ffts = readBy[row.res64];
    ffts.insert(row.fft);
    if (ffts.size() >= 2) { return row.res64; }
  }
  return {};
}

std::vector<FFTConfig> witnessOrder(const Env& env, const FFTConfig& own, u64 exponent) {
  FFTParts const mine = fftParts(own.shape.fft_type);
  auto shares = [&mine](const FFTParts& p) {
    return (mine.fp64 && p.fp64) || (mine.fp32 && p.fp32) || (mine.gf31 && p.gf31) || (mine.gf61 && p.gf61);
  };

  using Ranked = std::tuple<int, double, FFTConfig>;
  std::vector<Ranked> others;
  for (const AnchorSpec& candidate : anchorCandidates(exponent)) {
    FFTConfig const fft{candidate.fft};
    std::vector<u32> const runnable = runnableVariants(env, fft.shape);
    if (fft.spec() == own.spec() || std::ranges::find(runnable, fft.variant) == runnable.end()) { continue; }

    enum FFT_TYPES const type = fft.shape.fft_type;
    int const tier = !shares(fftParts(type)) ? 0 : type != own.shape.fft_type ? 1 : 2;
    others.emplace_back(tier, statedPriorK(type) * priorWork(fft.shape.size()), fft);
  }
  std::ranges::stable_sort(others, [](const Ranked& a, const Ranked& b) {
    return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
  });

  std::vector<FFTConfig> out{own};
  for (const auto& [tier, cost, fft] : others) { out.push_back(fft); }
  return out;
}

std::optional<u64> settleReference(TuneDB& db, u32 sess, u64 exponent, u64 iters, const std::vector<FFTConfig>& order,
                                   const ReadWitness& read) {
  std::vector<RefRow> readings = referenceReadings(db, db.envOf(sess), exponent, iters);
  if (std::optional<u64> const agreed = agreedResidue(readings)) { return agreed; }

  std::set<std::string> witnessed;
  for (const RefRow& row : readings) { witnessed.insert(row.fft); }

  for (const FFTConfig& fft : order) {
    if (witnessed.size() >= LL_MAX_WITNESSES) { break; }
    std::string const spec = fft.spec();
    if (witnessed.contains(spec)) { continue; }

    std::optional<u64> const res64 = read(fft);
    if (!res64) { continue; }

    RefRow const row{
      .sess = sess, .fft = spec, .exponent = exponent, .iters = iters, .res64 = *res64, .ts = u64(std::time(nullptr))};
    // A reading that cannot be recorded is not a vote a later process would see.
    if (!db.add(row)) { break; }
    witnessed.insert(spec);
    readings.push_back(row);

    if (std::optional<u64> const agreed = agreedResidue(readings)) { return agreed; }
  }
  return {};
}

}  // namespace tune
