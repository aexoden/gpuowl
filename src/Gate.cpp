// Copyright (C) Jason Lynch

#include "Gate.h"

#include "Bootstrap.h"
#include "Primes.h"

#include <algorithm>
#include <cstdio>

namespace tune {

namespace {

// A value that does not parse is taken to change the rounding, which is the safe error.
[[nodiscard]] bool changesRounding(const Option& option, const std::string& value) {
  std::optional<int> const v = parseInt<int>(value);
  return option.accuracyImpact != AccuracyImpact::None && (!v || option.changesRounding(*v));
}

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
    if (changesRounding(*findOption(key), value)) { return true; }
  }
  return false;
}

UseConfig accuracyReference(const Env& env, const FFTConfig& fft, const UseConfig& opts) {
  UseConfig out = opts;

  // Until nothing moves, since one key's default can turn on another's value.
  for (bool changed = true; changed;) {
    changed = false;
    for (const Option& option : allOptions()) {
      if (option.kind != Kind::Tunable || option.accuracyImpact == AccuracyImpact::None ||
          !option.appliesTo(env, fft, out)) {
        continue;
      }
      std::string const own = std::to_string(option.defaultFor(env, fft, out));
      auto const at = out.find(option.key);
      if (at != out.end() && at->second != own && !changesRounding(option, at->second)) { continue; }
      if (at == out.end() || at->second != own) {
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

Standard standardFor(enum FFT_TYPES type, bool movesAccuracy, const std::optional<RoeRow>& reference) {
  double const floor = minSafeZ(type);
  if (movesAccuracy && reference && reference->checkOk && reference->n > 2 && reference->z >= floor) {
    return {.aim = reference->z, .bar = std::max(reference->z - ACCURACY_SLACK_Z, floor)};
  }
  return {.aim = TARGET_Z, .bar = TARGET_Z};
}

GateVerdict judge(enum FFT_TYPES type, const std::optional<RoeRow>& own, bool movesAccuracy,
                  const std::optional<RoeRow>& reference) {
  if (!own) { return {}; }

  auto rejected = [](std::string why, bool derivable) {
    return GateVerdict{
      .state = GateState::Rejected, .evidence = Evidence::Rejected, .derivable = derivable, .why = std::move(why)};
  };

  // A set that spends accuracy is held to its reference, so there is nothing to judge it by without the reference's
  // reading, and nothing to compare where either has too few rounding errors to fit z to -- however small they are.
  if (movesAccuracy) {
    if (own->checkOk && own->n <= 2) {
      return rejected("too few rounding errors to compare with its defaults'", false);
    }
    if (!reference) { return {.owesReference = true}; }
    if (reference->checkOk && reference->n <= 2) {
      return rejected("its defaults are too accurate to fit z to", false);
    }
  }

  if (!own->checkOk) { return rejected("its Gerbicz check failed", true); }

  // Too few rounding errors to fit z to: the errors are small, which the floor has nothing to say against.
  if (own->n <= 2) { return {.state = GateState::Passed, .evidence = Evidence::Unavailable}; }

  double const floor = minSafeZ(type);
  if (own->z < floor) { return rejected(format("z %.2f is below the floor of %.0f", own->z, floor), true); }

  // Defaults that fall short here themselves are held to the fitted standard, and so is a set that spends accuracy
  // beside them.
  if (movesAccuracy) {
    Standard const standard = standardFor(type, true, reference);
    bool const relative = reference->checkOk && reference->z >= floor;
    if (own->z < standard.bar) {
      return rejected(relative ? format("z %.2f is below the %.2f its defaults read", own->z, reference->z)
                               : format("z %.2f is below the %.0f its defaults are held to, which fall short here",
                                        own->z, standard.aim),
                      true);
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

GateVerdict Gates::operator()(const FFTConfig& fft, const Interval& span, const UseConfig& opts) const {
  if (exactArithmetic(fft)) {
    return {.state = GateState::Passed, .evidence = Evidence::NotApplicable, .reach = span.hi};
  }

  u64 const top = gateExponent(span);
  if (!top) { return {}; }

  bool const moves = movesAccuracy(device_, fft, opts);
  std::optional<RoeRow> const own = reading(fft, top, opts, true);
  if (!own) { return {.owedAt = top}; }

  std::optional<RoeRow> const reference =
    moves ? reading(fft, own->exponent, accuracyReference(device_, fft, opts), false) : std::nullopt;

  GateVerdict out = judge(fft.shape.fft_type, own, moves, reference);
  if (out.state == GateState::Owed) {
    out.owedAt = own->exponent;
    return out;
  }
  if (out.state == GateState::Passed) {
    out.reach = span.hi;
    return out;
  }
  if (!out.derivable) { return out; }

  auto const toZ = [](const RoeRow& row) {
    return ZReading{.exponent = row.exponent, .z = row.z, .n = row.n, .checkOk = row.checkOk};
  };
  ReachOutcome const derived = deriveReach({.words = fft.size(),
                                            .lo = span.lo,
                                            .hi = std::min(top, carryCeiling(fft)),
                                            .standard = standardFor(fft.shape.fft_type, moves, reference)},
                                           toZ(*own), [&](u64 E) -> std::optional<ZReading> {
                                             std::optional<RoeRow> const row = reading(fft, E, opts, false);
                                             return row ? std::optional{toZ(*row)} : std::nullopt;
                                           });

  switch (derived.state) {
  case ReachState::Owed: return {.owedAt = derived.exponent};
  case ReachState::Confirmed:
    return {.state = GateState::Passed, .evidence = Evidence::Confirmed, .reach = derived.exponent, .derived = true};
  case ReachState::Rejected:
    out.why += ", and " + derived.why;
    out.derivable = false;
    return out;
  }
  return out;
}

}  // namespace tune
