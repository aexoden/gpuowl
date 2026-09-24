// Copyright (C) Jason Lynch

#include "Gate.h"

#include "Bootstrap.h"
#include "Primes.h"

#include <cstdio>

namespace tune {

namespace {

[[nodiscard]] bool changesRounding(const Option& option) { return option.accuracyImpact != AccuracyImpact::None; }

[[nodiscard]] std::string format(const char* fmt, double a, double b) {
  char buf[128];
  snprintf(buf, sizeof(buf), fmt, a, b);
  return buf;
}

// The set as the gate tells builds apart: canonical where the table knows a key well enough to say it changes nothing,
// and as written everywhere else.
[[nodiscard]] std::string identity(const Env& env, const FFTConfig& fft, const UseConfig& opts) {
  UseConfig out = canonicalConfig(env, fft, opts);
  for (const auto& [key, value] : opts) {
    const Option* const option = findOption(key);
    if (!option || option->kind != Kind::Tunable) { out.emplace(key, value); }
  }
  return configText(out);
}

}  // namespace

double minSafeZ(enum FFT_TYPES type) { return type == FFT64 ? 20 : 6; }

bool movesAccuracy(const Env& env, const FFTConfig& fft, const UseConfig& opts) {
  for (const auto& [key, value] : canonicalConfig(env, fft, opts)) {
    if (changesRounding(*findOption(key))) { return true; }
  }
  return false;
}

UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& opts) {
  UseConfig out = opts;

  // Until nothing moves, since one key's default can turn on another's value.
  for (bool changed = true; changed;) {
    changed = false;
    for (const Option& option : allOptions()) {
      if (option.kind != Kind::Tunable || !changesRounding(option) || !option.appliesTo(env, fft, out)) { continue; }
      std::string const own = std::to_string(option.defaultFor(env, fft, out));
      if (auto const at = out.find(option.key); at == out.end() || at->second != own) {
        out[option.key] = own;
        changed = true;
      }
    }
  }
  return out;
}

u64 gateExponent(const Interval& span) {
  if (span.empty() || span.hi < 2) { return 0; }

  static const Primes primes;
  u64 const top = primes.isPrime(span.hi) ? span.hi : primes.prevPrime(span.hi);
  return top >= span.lo ? top : 0;
}

GateVerdict judge(enum FFT_TYPES type, const std::optional<RoeRow>& own, bool movesAccuracy,
                  const std::optional<RoeRow>& reference) {
  if (!own) { return {}; }

  auto rejected = [](std::string why) {
    return GateVerdict{.state = GateState::Rejected, .evidence = Evidence::Rejected, .why = std::move(why)};
  };

  if (!own->checkOk) { return rejected("its Gerbicz check failed"); }

  double const floor = minSafeZ(type);

  // Too few rounding errors to fit z to: the errors are small, which the floor has nothing to say against, but there
  // is no reading to hold a set that spends accuracy to.
  if (own->n <= 2) {
    if (movesAccuracy) { return rejected("too few rounding errors to compare with its defaults'"); }
    return {.state = GateState::Passed, .evidence = Evidence::Unavailable};
  }

  if (own->z < floor) { return rejected(format("z %.2f is below the floor of %.0f", own->z, floor)); }

  if (movesAccuracy) {
    if (!reference) { return {.owesReference = true}; }

    // A reference that failed its own check says the defaults cannot run here at all, which no set can read worse
    // than; one with too few errors to fit is more accurate than anything a z can be compared with.
    if (reference->checkOk) {
      if (reference->n <= 2) { return rejected("its defaults are too accurate to fit z to"); }
      if (own->z < reference->z - ACCURACY_SLACK_Z) {
        return rejected(format("z %.2f is below the %.2f its defaults read", own->z, reference->z));
      }
    }
  }

  return {.state = GateState::Passed, .evidence = own->z >= TARGET_Z ? Evidence::Confirmed : Evidence::Unvalidated};
}

Gates::Gates(const TuneDB& db, u32 env, const Env& device) : device_{device} {
  for (const RoeRow& row : db.latestRoes()) {
    if (db.envOf(row.sess) != env) { continue; }
    const UseConfig* const opts = db.findCfg(row.cfg);
    auto const fft = parseFft(row.fft);
    if (!opts || !fft) { continue; }
    rows_[{row.fft, identity(device, *fft, *opts)}].push_back(row);
  }
}

std::optional<RoeRow> Gates::reading(const FFTConfig& fft, u64 exponent, const UseConfig& opts, bool above) const {
  auto const at = rows_.find({fft.spec(), identity(device_, fft, opts)});
  if (at == rows_.end()) { return {}; }

  Regime const regime = regimeOf(fft, exponent);
  std::optional<RoeRow> out;
  for (const RoeRow& row : at->second) {
    bool const counts = above ? row.exponent >= exponent : row.exponent == exponent;
    if (!counts || regimeOf(fft, row.exponent) != regime) { continue; }
    if (!out || row.ts > out->ts) { out = row; }
  }
  return out;
}

GateVerdict Gates::operator()(const FFTConfig& fft, u64 exponent, const UseConfig& opts) const {
  if (exactArithmetic(fft)) { return {.state = GateState::Passed, .evidence = Evidence::NotApplicable}; }

  bool const moves = movesAccuracy(device_, fft, opts);
  std::optional<RoeRow> const own = reading(fft, exponent, opts, true);
  std::optional<RoeRow> const reference =
    moves && own ? reading(fft, own->exponent, accuracyReference(device_, fft, opts), false) : std::nullopt;

  GateVerdict out = judge(fft.shape.fft_type, own, moves, reference);
  if (out.owesReference) { out.referenceAt = own->exponent; }
  return out;
}

}  // namespace tune
