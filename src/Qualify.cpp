// Copyright (C) Jason Lynch

#include "Qualify.h"

#include "Args.h"
#include "clwrap.h"
#include "Gpu.h"
#include "gpuid.h"
#include "Lanes.h"
#include "log.h"
#include "Measure.h"
#include "Primes.h"
#include "Signal.h"
#include "timeutil.h"
#include "TuneEntry.h"
#include "version.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <thread>

namespace tune {

namespace {

// The length of one chunk of the soak: long enough that the warm-up block each starts with costs little, short enough
// that a minute holds several.
constexpr double CHUNK_SEC = 10;

// The length of one chunk of a heater's squaring, which is how long a heater takes to stop once asked.
constexpr double HEATER_SEC = 0.5;

std::vector<std::string_view> splitComma(std::string_view text) {
  std::vector<std::string_view> out;
  while (true) {
    size_t const comma = text.find(',');
    out.push_back(text.substr(0, comma));
    if (comma == std::string_view::npos) { return out; }
    text.remove_prefix(comma + 1);
  }
}

std::optional<double> parseNumber(std::string_view text) {
  double v = 0;
  auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
  if (ec != std::errc{} || end != text.data() + text.size()) { return {}; }
  return v;
}

std::optional<double> parsePercent(std::string_view text) {
  if (!text.ends_with('%')) { return {}; }
  std::optional<double> const v = parseNumber(text.substr(0, text.size() - 1));
  if (!v || *v < 0) { return {}; }
  return *v / 100;
}

std::string pct(double fraction) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.2f%%", fraction * 100);
  return buf;
}

std::string json(const std::string& s) {
  std::string out = "\"";
  for (char const c : s) {
    if (c == '"' || c == '\\') { out += '\\'; }
    out += c;
  }
  return out + '"';
}

std::string number(double v) {
  if (!std::isfinite(v)) { return "null"; }
  char buf[32];
  snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

std::string numbers(const std::vector<double>& vs) {
  std::string out = "[";
  for (size_t i = 0; i < vs.size(); ++i) { out += (i ? ", " : "") + number(vs[i]); }
  return out + "]";
}

std::string cardJson(const CardSummary& s) {
  std::string held = "{";
  for (size_t i = 0; i < s.held.size(); ++i) {
    held += (i ? ", " : "") + json(s.held[i].first) + ": " + number(s.held[i].second);
  }
  held += "}";
  return "{\"samples\": " + std::to_string(s.n) + ", \"smMHz\": " + number(s.smMedian) + ", \"smMinMHz\": " +
    number(s.smMin) + ", \"smMaxMHz\": " + number(s.smMax) + ", \"memMHz\": " + number(s.memMedian) +
    ", \"tempC\": " + number(s.tempMedian) + ", \"tempMaxC\": " + number(s.tempMax) + ", \"powerW\": " +
    number(s.powerMedian) + ", \"capW\": " + number(s.capW) + ", \"heldShare\": " + held + "}";
}


}  // namespace

double median(std::vector<double> v) {
  if (v.empty()) { return NAN; }
  std::ranges::sort(v);
  size_t const n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

MeasureArgs parseMeasureArgs(std::string_view text) {
  std::vector<std::string_view> const tokens = splitComma(text);
  MeasureArgs out;
  out.fft = std::string{tokens.front()};
  if (out.fft.empty()) { throw std::string{"-measure needs an FFT spec, such as -measure 512:7:512:000"}; }
  try {
    (void)FFTConfig{out.fft};
  } catch (const char*) { throw "-measure: '" + out.fft + "' is not an FFT spec"; }

  for (size_t i = 1; i < tokens.size(); ++i) {
    std::string_view const token = tokens[i];
    size_t const eq = token.find('=');
    std::string_view const key = token.substr(0, eq);
    std::string_view const value = eq == std::string_view::npos ? std::string_view{} : token.substr(eq + 1);
    auto fail = [&](const char* takes) { return "-measure: " + std::string{key} + "= takes " + takes; };
    if (eq == std::string_view::npos || value.empty()) {
      throw "-measure: '" + std::string{token} + "' is not a setting";
    } else if (key == "kind") {
      std::optional<TestKind> const kind = parseTestKind(value);
      if (!kind) { throw fail("prp or ll"); }
      out.kind = *kind;
    } else if (key == "exp") {
      std::optional<double> const e = parseNumber(value);
      if (!e || *e < 1 || *e != std::floor(*e)) { throw fail("an exponent"); }
      out.exponent = u64(*e);
    } else if (key == "bursts" || key == "minutes") {
      std::optional<double> const n = parseNumber(value);
      if (!n || *n < 1 || *n > 1000 || *n != std::floor(*n)) { throw fail("a count from 1 to 1000"); }
      (key == "bursts" ? out.bursts : out.minutes) = u32(*n);
    } else if (key == "idle" || key == "soak" || key == "heat") {
      std::optional<double> const t = value == "0" ? std::optional<double>{0} : parseDuration(value);
      if (!t) { throw fail("a span of time, such as 5s, 15m or 1h"); }
      (key == "idle" ? out.idle : key == "soak" ? out.soak : out.heat) = *t;
    } else if (key == "settle" || key == "gap" || key == "spread") {
      std::optional<double> const p = parsePercent(value);
      if (!p) { throw fail("a percentage, such as 0.5%"); }
      (key == "settle" ? out.settle : key == "gap" ? out.gap : out.spread) = *p;
    } else {
      throw "-measure: '" + std::string{key} + "' is not a setting it takes";
    }
  }
  return out;
}

std::vector<double> minuteMeans(std::span<const Chunk> chunks, double from) {
  if (chunks.empty()) { return {}; }
  double const last = chunks.back().end;
  std::vector<double> out;
  for (u32 m = 0; from + 60.0 * (m + 1) <= last; ++m) {
    double const lo = from + 60.0 * m;
    double const hi = lo + 60;
    double sum = 0;
    u32 blocks = 0;
    for (const Chunk& c : chunks) {
      if (c.end >= lo && c.end < hi) {
        sum += c.us * c.blocks;
        blocks += c.blocks;
      }
    }
    if (blocks) { out.push_back(sum / blocks); }
  }
  return out;
}

std::optional<size_t> settledAt(std::span<const double> minutes, double settle) {
  size_t const w = SETTLE_MINUTES;
  auto mean = [&](size_t from) {
    double sum = 0;
    for (size_t i = from; i < from + w; ++i) { sum += minutes[i]; }
    return sum / double(w);
  };
  for (size_t m = 2 * w - 1; m < minutes.size(); ++m) {
    if (std::abs(mean(m + 1 - w) / mean(m + 1 - 2 * w) - 1) <= settle) { return m; }
  }
  return {};
}

MeasureVerdict judge(const MeasureResult& r, const MeasureArgs& args, std::string_view vendor, const std::string& bdf) {
  MeasureVerdict v;
  v.burst = median(r.bursts);
  v.hot = median(r.hot);
  v.sustained = median(r.sustained);
  // Over the means of SETTLE_MINUTES running minutes where there are three of them: a card at its power limit swings
  // 0.5-1% from one minute to the next as the limit moves its clocks, which is not wandering.
  std::vector<double> spans = r.sustained;
  if (r.sustained.size() >= SETTLE_MINUTES + 2) {
    spans.clear();
    for (size_t m = 0; m + SETTLE_MINUTES <= r.sustained.size(); ++m) {
      double sum = 0;
      for (size_t i = m; i < m + SETTLE_MINUTES; ++i) { sum += r.sustained[i]; }
      spans.push_back(sum / double(SETTLE_MINUTES));
    }
  }
  if (!spans.empty()) { v.spread = (std::ranges::max(spans) - std::ranges::min(spans)) / v.sustained; }
  v.burstGap = v.sustained / v.burst - 1;
  v.hotGap = v.sustained / v.hot - 1;
  v.heated = median(r.heated);
  v.heatedGap = v.sustained / v.heated - 1;
  v.heatedHolds = !r.heated.empty() && std::abs(v.heatedGap) <= args.gap;
  v.optimistic = v.burstGap > args.gap;
  v.wandering = v.spread > args.spread;

  char buf[256];
  snprintf(buf, sizeof(buf), "a call as the tuner times one reads %.2f us/it; the card sustains %.2f, %s %s", v.burst,
           v.sustained, pct(std::abs(v.burstGap)).c_str(), v.burstGap >= 0 ? "slower" : "faster");
  v.lines.push_back(buf);
  snprintf(buf, sizeof(buf), "with no idle before them, calls read %.2f us/it, %s from sustained", v.hot,
           pct(v.hotGap).c_str());
  v.lines.push_back(buf);
  if (!r.heated.empty()) {
    snprintf(buf, sizeof(buf),
             "with the card kept loaded between them and %.0f s of their own load first, calls read %.2f us/it, %s "
             "from sustained",
             args.heat, v.heated, pct(v.heatedGap).c_str());
    v.lines.push_back(buf);
  }
  snprintf(buf, sizeof(buf), "the sustained %s lie within %s of each other",
           r.sustained.size() >= SETTLE_MINUTES + 2 ? "3-minute means" : "minutes", pct(v.spread).c_str());
  v.lines.push_back(buf);

  // Watts times microseconds per iteration is microjoules per iteration.
  auto said = [](const std::vector<double>& watts) {
    std::vector<double> out;
    for (double const w : watts) {
      if (!std::isnan(w)) { out.push_back(w); }
    }
    return median(std::move(out));
  };
  v.burstWatts = said(r.burstWatts.taken);
  v.burstMJ = v.burstWatts * v.burst / 1000;
  v.sustainedWatts = r.sustainedWatts;
  v.sustainedMJ = v.sustainedWatts * v.sustained / 1000;
  v.capW = r.sustainedCard.capW;
  v.held = v.sustainedWatts / v.capW;
  v.estimate = v.burst;
  if (!std::isnan(v.burstMJ) && v.capW > 0) {
    v.estimate = std::max(v.burst, v.burstMJ * 1000 / (HOLD_SHARE * v.capW));
  }
  v.estimateGap = v.sustained / v.estimate - 1;
  if (!std::isnan(v.burstWatts)) {
    snprintf(buf, sizeof(buf), "a call drew %.0f W, %.2f mJ an iteration; sustained, %.0f W and %.2f mJ", v.burstWatts,
             v.burstMJ, v.sustainedWatts, v.sustainedMJ);
    v.lines.push_back(buf);
    auto w = [](double watts) {
      char text[32] = "none";
      if (!std::isnan(watts)) { snprintf(text, sizeof(text), "%.0f W", watts); }
      return std::string{text};
    };
    v.lines.push_back("a call's power by the energy counter: " + w(said(r.burstWatts.counter)) +
                      ", by the power readings: " + w(said(r.burstWatts.readings)) + "; with no idle before it, " +
                      w(said(r.hotWatts.counter)) + " and " + w(said(r.hotWatts.readings)));
  }
  if (!std::isnan(v.held)) {
    snprintf(buf, sizeof(buf), "sustained, the card held %.0f W of its %.0f W limit: %.3f of it", v.sustainedWatts,
             v.capW, v.held);
    v.lines.push_back(buf);
    snprintf(buf, sizeof(buf),
             "costed by its energy over %.2f of the limit (-tune powerHold=%.2f), a call here reads %.2f us/it (%s); "
             "the card sustains %s %s",
             HOLD_SHARE, HOLD_SHARE, v.estimate, v.estimate > v.burst ? "its energy sets it" : "its timing sets it",
             pct(std::abs(v.estimateGap)).c_str(), v.estimateGap >= 0 ? "more" : "less");
    v.lines.push_back(buf);
  }
  // At higher clocks and voltage a call's iteration cannot take less energy than a sustained one: where it reads so,
  // the card's power readings lag behind a call this short, and a cost built on them would be wrong.
  if (v.burstMJ < v.sustainedMJ * (1 - args.gap)) {
    snprintf(buf, sizeof(buf),
             "a call's energy reads %s below a sustained iteration's, which it cannot be: the card's power readings "
             "lag behind a call this short, and powerHold= would not rank configurations by them here",
             pct(1 - v.burstMJ / v.sustainedMJ).c_str());
    v.lines.push_back(buf);
  }
  if (!r.settled) {
    v.lines.push_back("the soak ran out of time before two minutes agreed: the card may not have settled; a longer "
                      "soak= says");
  }

  if (!v.optimistic && !v.wandering) {
    v.lines.push_back("verdict: stable -- what the tuner reads here is what production runs at, with the card set as "
                      "it is");
    return v;
  }
  if (v.optimistic && v.heatedHolds) {
    v.lines.push_back("keeping the card loaded between calls is enough here: calls timed that way read what it "
                      "sustains, with no clock locked");
  }
  v.lines.push_back(std::string{"verdict: "} + (v.optimistic ? "optimistic" : "") +
                    (v.optimistic && v.wandering ? " and " : "") + (v.wandering ? "wandering" : "") +
                    (v.optimistic ? " -- the tuner's short calls run faster than this card sustains, and can rank "
                                    "configurations as production would not"
                                  : " -- one minute's speed is not the next's, which the tuner's readings carry"));
  if (v.optimistic && !v.wandering && v.estimate > v.burst && std::abs(v.estimateGap) <= args.gap) {
    snprintf(buf, sizeof(buf),
             "the power-aware cost comes within the gap of what the card sustains here: tuning with powerHold=%.2f "
             "ranks this configuration as production would; locking the clocks would too, at some speed",
             HOLD_SHARE);
    v.lines.push_back(buf);
  }
  double const sm = r.sustainedCard.smMedian;
  if (!std::isnan(sm) && vendor == "nvidia") {
    u32 const lock = u32(sm) / 15 * 15;
    snprintf(buf, sizeof(buf),
             "lock the clocks at what the card sustains, for tuning and production alike: sudo nvidia-smi -i %s -lgc "
             "%u,%u; then measure again, and tune with tag=%umhz",
             bdf.empty() ? "<card>" : ("0000:" + bdf).c_str(), lock, lock, lock);
    v.lines.push_back(buf);
  } else if (!std::isnan(sm) && vendor == "amd") {
    snprintf(buf, sizeof(buf),
             "lock the clocks at what the card sustains, for tuning and production alike: sudo rocm-smi "
             "--setperfdeterminism %.0f; then measure again, and tune with tag=%.0fmhz",
             sm, sm);
    v.lines.push_back(buf);
  } else {
    v.lines.push_back("lock the card's clocks at or below what it sustains, for tuning and production alike, then "
                      "measure again");
  }
  return v;
}

std::string measureJson(const MeasureArgs& args, const std::string& options, u64 exponent,
                        const std::string& telemetry, const MeasureResult& r, const MeasureVerdict& v) {
  std::string out = "{\n";
  out += "  \"version\": " + json(VERSION) + ",\n";
  out += "  \"fft\": " + json(args.fft) + ", \"kind\": " + json(toString(args.kind)) +
    ", \"exponent\": " + std::to_string(exponent) + ", \"options\": " + json(options) + ",\n";
  out += "  \"telemetry\": " + json(telemetry) + ",\n";
  out += "  \"burstsUsPerIt\": " + numbers(r.bursts) + ",\n";
  out += "  \"soakMinutesUsPerIt\": " + numbers(r.soak) + ", \"settledAtMinute\": " +
    (r.settled ? std::to_string(*r.settled) : std::string{"null"}) + ",\n";
  out += "  \"sustainedMinutesUsPerIt\": " + numbers(r.sustained) + ",\n";
  out += "  \"hotBurstsUsPerIt\": " + numbers(r.hot) + ",\n";
  out += "  \"heatedUsPerIt\": " + numbers(r.heated) + ", \"heatSec\": " + number(args.heat) + ",\n";
  out += "  \"burstWatts\": " + numbers(r.burstWatts.taken) + ", \"burstCounterWatts\": " +
    numbers(r.burstWatts.counter) + ", \"burstReadingWatts\": " + numbers(r.burstWatts.readings) + ",\n";
  out += "  \"hotWatts\": " + numbers(r.hotWatts.taken) + ", \"hotCounterWatts\": " + numbers(r.hotWatts.counter) +
    ", \"hotReadingWatts\": " + numbers(r.hotWatts.readings) + ",\n";
  out += "  \"heatedWatts\": " + numbers(r.heatedWatts.taken) + ", \"heatedCounterWatts\": " +
    numbers(r.heatedWatts.counter) + ", \"heatedReadingWatts\": " + numbers(r.heatedWatts.readings) + ",\n";
  out += "  \"burstCard\": " + cardJson(r.burstCard) + ",\n";
  out += "  \"sustainedCard\": " + cardJson(r.sustainedCard) + ",\n";
  out += "  \"heatedCard\": " + cardJson(r.heatedCard) + ",\n";
  out += "  \"verdict\": {\"burst\": " + number(v.burst) + ", \"hot\": " + number(v.hot) + ", \"heated\": " +
    number(v.heated) + ", \"sustained\": " + number(v.sustained) + ", \"burstGap\": " + number(v.burstGap) +
    ", \"hotGap\": " + number(v.hotGap) + ", \"heatedGap\": " + number(v.heatedGap) +
    ", \"spread\": " + number(v.spread) + ", \"burstWatts\": " + number(v.burstWatts) + ", \"burstMJPerIt\": " +
    number(v.burstMJ) + ", \"sustainedWatts\": " + number(v.sustainedWatts) + ", \"sustainedMJPerIt\": " +
    number(v.sustainedMJ) + ", \"capW\": " + number(v.capW) + ", \"heldShare\": " + number(v.held) +
    ", \"estimate\": " + number(v.estimate) + ", \"estimateGap\": " + number(v.estimateGap) +
    ", \"optimistic\": " + (v.optimistic ? "true" : "false") +
    ", \"wandering\": " + (v.wandering ? "true" : "false") + "}\n";
  return out + "}\n";
}

bool runMeasure(GpuCommon shared, const MeasureArgs& args) {
  const Args& a = *shared.args;
  FFTConfig const fft{args.fft};
  u64 const E = args.exponent ? args.exponent : Primes{}.prevPrime(fft.maxExp());
  bool const ll = args.kind == TestKind::LL;
  vector<KeyVal> const uses = TuneEntry::usesFor(a, fft, E, ll);
  std::string options;
  for (const auto& [k, v] : uses) { options += (options.empty() ? "" : ",") + k + '=' + v; }
  u32 const blockSize = a.blockSize;

  cl_device_id const id = shared.context->deviceId();
  bool const nvidia = isNvidiaGpu(id);
  bool const amd = isAmdGpu(id);
  std::string const bdf = pciAddressOf(id);
  std::string why;
  std::unique_ptr<Telemetry> const telemetry = openTelemetry(bdf, nvidia, amd, why);
  std::string const source = telemetry ? telemetry->source() : "none: " + why;
  bool const saysWhy = telemetry && telemetry->saysWhy();
  Sampler const sampler{telemetry.get(), SAMPLE_PERIOD};

  log("measure: %s %s at %" PRIu64 ", block %u, with %s (config.txt, the command line and its tune.txt line)\n",
      fft.spec().c_str(), toString(args.kind), E, blockSize, options.empty() ? "no -use settings" : options.c_str());
  log("measure: card readings: %s\n", source.c_str());

  auto timed = [&](Gpu& gpu, u32 nBlocks, double heatSec) {
    return summarize(ll ? gpu.timeItersLL(nBlocks, blockSize, CALL_WARMUP_BLOCKS, heatSec)
                        : gpu.timeIters(nBlocks, blockSize, CALL_WARMUP_BLOCKS, heatSec));
  };
  auto checked = [&](const Call& call) {
    if (call.status == Status::Err) { throw std::string{"its Gerbicz check failed: it computes wrongly"}; }
    return call.mean();
  };
  auto cardLine = [&](double from, double to) {
    std::string const s = toString(summarizeCard(sampler.between(from, to)), saysWhy);
    return s.empty() ? s : "; " + s;
  };

  MeasureResult r;
  try {
    // As the tuner calls: a Gpu built afresh, a second of heat, a warm-up block and four timed ones.  What the card
    // read is taken over the timed blocks alone, as the tuner takes it, not over the idle and the build around them.
    // A call's reading, what the card drew over its timed blocks and what it read there, logged; the card's readings
    // over those blocks are added to `seen`.
    auto record = [&](const Call& call, std::vector<double>& into, CallWatts& watts, std::vector<CardSample>& seen,
                      const char* what, u32 i) {
      into.push_back(checked(call));
      double const from = sampler.at(call.timedFrom);
      double const to = sampler.at(call.timedTo);
      (void)sampler.waitPast(to, 5 * SAMPLE_PERIOD);
      std::vector<CardSample> const window = sampler.between(from, to);
      seen.insert(seen.end(), window.begin(), window.end());
      std::optional<WindowPower> const power = powerOver(sampler, call.timedFrom, call.timedTo);
      watts.taken.push_back(power ? power->watts : NAN);
      watts.counter.push_back(power ? power->counterWatts : NAN);
      watts.readings.push_back(power ? power->readingWatts : NAN);
      char drew[128] = "";
      if (power) {
        snprintf(drew, sizeof(drew), ", %.0f W by the %s (counter %.0f, %u readings %.0f), %.2f mJ/it", power->watts,
                 power->counted ? "counter" : "readings", power->counterWatts, power->readings, power->readingWatts,
                 power->watts * into.back() / 1000);
      }
      std::string const card = toString(summarizeCard(window), saysWhy);
      log("measure: %s %u: %.2f us/it%s%s\n", what, i + 1, into.back(), drew,
          card.empty() ? "" : ("; " + card).c_str());
    };

    auto bursts = [&](double idle, std::vector<double>& into, CallWatts& watts, const char* what) {
      std::vector<CardSample> seen;
      for (u32 i = 0; i < args.bursts; ++i) {
        for (Timer t; t.at() < idle;) {
          if (Signal::stopRequested()) { throw "stop requested"; }
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        auto gpu = Gpu::make(E, shared, fft, {}, false, uses);
        record(timed(*gpu, BLOCKS_PER_CALL, CALL_HEAT_SEC), into, watts, seen, what, i);
      }
      return seen;
    };

    r.burstCard = summarizeCard(bursts(args.idle, r.bursts, r.burstWatts, "burst"));

    auto gpu = Gpu::make(E, shared, fft, {}, false, uses);
    u32 const chunkBlocks = std::max(2u, u32(std::lround(CHUNK_SEC / (median(r.bursts) * blockSize * 1e-6))));
    // Squares in chunks from `from` until `done` says so, logging each whole minute as it ends.
    auto squareOn = [&](double from, const char* what, auto done) {
      std::vector<Chunk> chunks;
      std::vector<double> minutes;
      while (!done(minutes, sampler.now() - from)) {
        chunks.push_back({.us = checked(timed(*gpu, chunkBlocks, 0)), .blocks = chunkBlocks});
        chunks.back().end = sampler.now();
        std::vector<double> now = minuteMeans(chunks, from);
        for (size_t m = minutes.size(); m < now.size(); ++m) {
          log("measure: %s minute %zu: %.2f us/it%s\n", what, m + 1, now[m],
              cardLine(from + 60.0 * m, from + 60.0 * (m + 1)).c_str());
        }
        minutes = std::move(now);
      }
      return minutes;
    };

    r.soak = squareOn(sampler.now(), "soak", [&](const std::vector<double>& minutes, double elapsed) {
      r.settled = settledAt(minutes, args.settle);
      return r.settled.has_value() || elapsed >= args.soak;
    });
    if (r.settled) {
      log("measure: settled in minute %zu, at %.2f us/it\n", *r.settled + 1, r.soak[*r.settled]);
    } else {
      log("measure: no two minutes agreed within %s in %.0f minutes of soak\n", pct(args.settle).c_str(),
          args.soak / 60);
    }

    double const sustainedFrom = sampler.now();
    r.sustained = squareOn(sustainedFrom, "sustained", [&](const std::vector<double>& minutes, double) {
      return minutes.size() >= args.minutes;
    });
    double const sustainedTo = sampler.now();
    r.sustainedCard = summarizeCard(sampler.between(sustainedFrom, sustainedTo));
    std::vector<CardSample> const around = sampler.between(sustainedFrom - 1, sustainedTo + 1);
    bool const instant = sampler.instantPower();
    if (std::optional<WindowPower> const power = powerOver(around, sustainedFrom, sustainedTo, instant)) {
      r.sustainedWatts = power->watts;
    }

    // Heated: while the next call's Gpu is built, the last one goes on squaring on a thread of its own, so the card
    // never cools between calls, as a tuner holding its last configuration could keep it; then `heat` seconds of the
    // call's own load before its timed blocks.
    {
      u32 const heaterBlocks = std::max(1u, u32(std::lround(HEATER_SEC / (median(r.bursts) * blockSize * 1e-6))));
      std::vector<CardSample> seen;
      for (u32 i = 0; i < args.bursts; ++i) {
        std::atomic<bool> stop{false};
        std::exception_ptr failed;
        std::thread warm{[&] {
          try {
            while (!stop) { (void)timed(*gpu, heaterBlocks, 0); }
          } catch (...) { failed = std::current_exception(); }
        }};
        std::unique_ptr<Gpu> next;
        try {
          next = Gpu::make(E, shared, fft, {}, false, uses);
        } catch (...) {
          stop = true;
          warm.join();
          throw;
        }
        stop = true;
        warm.join();
        if (failed) { std::rethrow_exception(failed); }
        record(timed(*next, BLOCKS_PER_CALL, args.heat), r.heated, r.heatedWatts, seen, "heated call", i);
        gpu = std::move(next);
      }
      r.heatedCard = summarizeCard(seen);
    }
    gpu.reset();

    (void)bursts(0, r.hot, r.hotWatts, "hot burst");
  } catch (const char* what) {
    log("measure: stopped: %s\n", what);
    return false;
  } catch (const std::string& what) {
    log("measure: stopped: %s\n", what.c_str());
    return false;
  } catch (const std::exception& e) {
    log("measure: stopped: %s\n", e.what());
    return false;
  }

  MeasureVerdict const v = judge(r, args, nvidia ? "nvidia" : amd ? "amd" : "", bdf);
  if (std::string const card = toString(r.burstCard, saysWhy); !card.empty()) {
    log("measure: card during the bursts: %s\n", card.c_str());
  }
  if (std::string const card = toString(r.sustainedCard, saysWhy); !card.empty()) {
    log("measure: card while sustained: %s\n", card.c_str());
  }
  for (const std::string& line : v.lines) { log("measure: %s\n", line.c_str()); }

  std::ofstream{MEASURE_REPORT} << measureJson(args, options, E, source, r, v);
  log("measure: wrote %s\n", MEASURE_REPORT);
  return true;
}

}  // namespace tune
