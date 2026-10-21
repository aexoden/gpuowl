// Copyright (C) Jason Lynch

#include "Stats.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tune {

double Stats::sem() const { return n > 1 ? sd / std::sqrt(double(n)) : 0; }
double Stats::relSem() const { return mean > 0 ? sem() / mean : 0; }

const char* toString(Status status) {
  switch (status) {
  case Status::Ok: return "ok";
  case Status::Err: return "err";
  case Status::NoCompile: return "nocompile";
  case Status::Unsupported: return "unsupported";
  case Status::Lost: return "lost";
  }

  return "err";
}

std::optional<Status> parseStatus(std::string_view text) {
  for (Status s : {Status::Ok, Status::Err, Status::NoCompile, Status::Unsupported, Status::Lost}) {
    if (text == toString(s)) { return s; }
  }

  return {};
}

Stats statsOf(std::span<const double> samples) {
  Stats s;
  s.n = u32(samples.size());
  if (!s.n) { return s; }

  double sum = 0;
  for (double const x : samples) { sum += x; }
  s.mean = sum / s.n;

  if (s.n > 1) {
    double sq = 0;
    for (double const x : samples) { sq += (x - s.mean) * (x - s.mean); }
    s.sd = std::sqrt(sq / (s.n - 1));
  }

  return s;
}

CoreStats coreStats(std::span<const double> samples, double spikeCut, double maxFraction) {
  if (samples.size() < 3) { return {.stats = statsOf(samples)}; }

  std::vector<double> sorted{samples.begin(), samples.end()};
  std::ranges::sort(sorted);
  size_t const n = sorted.size();
  double const median = n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  double const cut = median * spikeCut;

  std::vector<double> kept;
  kept.reserve(n);
  for (double const x : samples) {
    if (x <= cut) { kept.push_back(x); }
  }

  size_t const nDropped = n - kept.size();
  if (double(nDropped) > maxFraction * double(n)) { return {.stats = statsOf(samples), .declined = true}; }

  return {.stats = statsOf(kept), .dropped = u32(nDropped)};
}

double Measurement::cost() const { return drift > 0 ? mean / drift : mean; }
double Measurement::costStddev() const { return drift > 0 ? stddev / drift : stddev; }

Measurement measurementOf(const CoreStats& core, double drift, u64 ts) {
  return {.mean = core.stats.mean,
          .stddev = core.stats.sd,
          .blocks = core.stats.n,
          .calls = core.stats.n ? 1u : 0u,
          .drift = drift,
          .status = Status::Ok,
          .ts = ts};
}

double standardError(const Measurement& m) {
  if (m.calls < 2 || m.blocks < 2) { return m.costStddev(); }

  double const blocksPerCall = double(m.blocks) / m.calls;
  double const betweenSd = m.costStddev() * std::sqrt((m.blocks - 1) / (blocksPerCall * (m.calls - 1)));
  return betweenSd / std::sqrt(double(m.calls));
}

double relStandardError(const Measurement& m) { return m.cost() > 0 ? standardError(m) / m.cost() : 0; }

double pessimisticCost(const Measurement& m) { return m.cost() * (1 + PESSIMISM_SIGMA * relStandardError(m)); }

bool concluded(const Measurement& m) { return m.ok() && m.calls >= MIN_CALLS; }

const char* toString(NoiseVerdict verdict) {
  switch (verdict) {
  case NoiseVerdict::TooFew: return "too few readings";
  case NoiseVerdict::Matches: return "the error bars match what the readings do";
  case NoiseVerdict::Conservative: return "the error bars are wider than they need to be";
  case NoiseVerdict::Drifting: return "the device is drifting between readings";
  case NoiseVerdict::Disturbed: return "the readings disagree by more than their error bars allow";
  }

  return "too few readings";
}

namespace {

// `predicted` pools the individual error bars in quadrature, so one wide reading cannot be averaged away by narrow
// ones; `bars` are those error bars, `values` the readings themselves, in the order taken.
Spread spreadOf(std::span<const double> values, std::span<const double> bars) {
  Spread out;
  Stats const st = statsOf(values);
  out.n = st.n;
  out.mean = st.mean;
  out.observed = st.sd;

  if (st.n < 2) { return out; }

  double sq = 0;
  for (double const bar : bars) { sq += bar * bar; }
  out.predicted = std::sqrt(sq / bars.size());

  // Von Neumann's successive-difference estimator: blind to any smooth trend, so a device warming up leaves this
  // alone while `observed` grows, and noise that is genuinely correlated moves both together.
  double diffSq = 0;
  for (size_t i = 1; i < values.size(); ++i) {
    double const d = values[i] - values[i - 1];
    diffSq += d * d;
  }
  out.neighbour = std::sqrt(diffSq / (2 * double(values.size() - 1)));
  out.trend = values.back() - values.front();

  return out;
}

Measurement asMeasurement(const CallSummary& c) {
  return {.mean = c.mean, .stddev = c.sd, .blocks = c.blocks, .calls = 1, .drift = c.drift, .status = Status::Ok};
}

}  // namespace

NoiseReport noiseOf(std::span<const CallSummary> calls, u32 callsPerRow) {
  NoiseReport out;
  out.callsPerRow = std::max(callsPerRow, 1u);

  std::vector<double> callMeans;
  std::vector<double> callBars;
  for (const CallSummary& c : calls) {
    Measurement const m = asMeasurement(c);
    callMeans.push_back(m.cost());
    callBars.push_back(c.blocks > 1 ? m.costStddev() / std::sqrt(double(c.blocks)) : 0);
  }
  out.call = spreadOf(callMeans, callBars);

  std::vector<double> rowMeans;
  std::vector<double> rowBars;
  for (size_t i = 0; i + out.callsPerRow <= calls.size(); i += out.callsPerRow) {
    Measurement row = asMeasurement(calls[i]);
    for (u32 k = 1; k < out.callsPerRow; ++k) { mergeInto(row, asMeasurement(calls[i + k])); }
    rowMeans.push_back(row.cost());
    rowBars.push_back(standardError(row));
  }
  out.row = spreadOf(rowMeans, rowBars);

  if (out.row.n < 2) { return out; }

  double blocks = 0;
  for (const CallSummary& c : calls) { blocks += c.blocks; }
  blocks /= double(calls.size());

  bool const trendShaped = out.call.neighbour > 0 && out.call.observed > NOISE_TREND_SHAPE * out.call.neighbour;

  if (trendShaped && out.row.ratio() > NOISE_DRIFTING) {
    out.verdict = NoiseVerdict::Drifting;
  } else if (out.call.detrended() > NOISE_BLOCK_SIGMAS * std::sqrt(std::max(1.0, blocks))) {
    out.verdict = NoiseVerdict::Disturbed;
  } else if (out.row.detrended() > NOISE_DISTURBED) {
    out.verdict = NoiseVerdict::Disturbed;
  } else if (out.row.detrended() * NOISE_CONSERVATIVE * std::sqrt(std::max(1.0, blocks)) < 1) {
    out.verdict = NoiseVerdict::Conservative;
  } else {
    out.verdict = NoiseVerdict::Matches;
  }

  return out;
}

void mergeInto(Measurement& into, const Measurement& add) {
  if (add.status != Status::Ok) {
    // A wrong answer is evidence about the kernels; a later refusal to build or run them says nothing against it.
    if (into.status != Status::Err) { into = add; }
    return;
  }

  if (into.status != Status::Ok) { return; }

  if (!into.blocks) {
    into = add;
    return;
  }

  if (!add.blocks) { return; }

  double const m1 = into.cost();
  double const m2 = add.cost();
  double const s1 = into.costStddev();
  double const s2 = add.costStddev();
  double const n1 = into.blocks;
  double const n2 = add.blocks;
  double const n = n1 + n2;
  double const delta = m2 - m1;

  double const sq1 = s1 * s1 * (n1 > 1 ? n1 - 1 : 0);
  double const sq2 = s2 * s2 * (n2 > 1 ? n2 - 1 : 0);
  double const sq = sq1 + sq2 + delta * delta * n1 * n2 / n;

  into.mean = m1 + delta * n2 / n;
  into.stddev = n > 1 ? std::sqrt(sq / (n - 1)) : 0;
  into.blocks = u32(n);

  into.calls += add.calls;
  into.drift = 1;
  into.ts = std::max(into.ts, add.ts);
}

Measurement merged(Measurement a, const Measurement& b) {
  mergeInto(a, b);
  return a;
}

}  // namespace tune
