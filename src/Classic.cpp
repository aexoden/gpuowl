// Copyright (C) Jason Lynch

#include "Classic.h"

#include "Args.h"
#include "Bootstrap.h"
#include "Eligibility.h"
#include "FFTVariants.h"
#include "Gate.h"
#include "Primes.h"
#include "Scheduler.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <utility>

namespace tune {

namespace {

std::string percent(double fraction) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%g%%", fraction * 100);
  return buf;
}

std::optional<double> parsePercent(std::string_view text) {
  if (!text.ends_with('%')) { return {}; }
  std::string const digits{text.substr(0, text.size() - 1)};
  char* end = nullptr;
  double const v = std::strtod(digits.c_str(), &end);
  if (digits.empty() || end != digits.c_str() + digits.size() || !(v >= 0) || v >= 100) { return {}; }
  return v / 100;
}

std::optional<u32> parseCount(std::string_view text) {
  u32 v = 0;
  auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
  if (ec != std::errc{} || end != text.data() + text.size()) { return {}; }
  return v;
}

const char* groupText(Group3 g) { return g == Group3::Off ? "0" : g == Group3::On ? "1" : "2"; }

}  // namespace

std::vector<std::string> classicWords(const ClassicSettings& s) {
  std::vector<std::string> out;
  out.emplace_back(std::string{"types="} +
                   (s.types == ClassicTypes::Auto     ? "auto"
                      : s.types == ClassicTypes::Fp64 ? "fp64"
                      : s.types == ClassicTypes::Ntt  ? "ntt"
                                                      : "both"));
  out.emplace_back(std::string{"sweeps="} + (s.sweeps ? "1" : "0"));
  out.emplace_back(std::string{"inplace="} + (!s.inplaceOnly ? "auto" : *s.inplaceOnly ? "1" : "0"));
  out.emplace_back(std::string{"fp32="} + groupText(s.fp32));
  out.emplace_back(std::string{"fp6431="} + groupText(s.fp6431));
  out.emplace_back(std::string{"1k256="} + groupText(s.k1x256));
  out.emplace_back(std::string{"m61="} + groupText(s.m61));
  out.emplace_back(std::string{"pfa="} + groupText(s.pfa));
  if (s.regsOnly) { out.push_back("regs"); }
  if (s.variantsOnly) { out.push_back("variants"); }
  out.emplace_back("fine=" + percent(s.fine));
  out.emplace_back("tk=" + (s.tk < 0 ? std::string{"auto"} : std::to_string(s.tk)));
  out.emplace_back("keepMargin=" + (s.keepMargin ? percent(*s.keepMargin) : std::string{"upstream"}));
  out.emplace_back("typesMargin=" + percent(s.typeMargin));
  out.emplace_back(std::string{"whVariants="} +
                   (s.whVariants == WhVariants::Fastest      ? "fastest"
                      : s.whVariants == WhVariants::Accurate ? "accurate"
                                                             : "all"));
  out.emplace_back("maxCalls=" + std::to_string(s.maxCalls));
  out.emplace_back("tkLosses=" + std::to_string(s.tkLosses));
  out.emplace_back("tkBad=" + percent(s.tkBad));
  return out;
}

bool parseClassicWord(std::string_view word, ClassicSettings& s) {
  size_t const eq = word.find('=');
  std::string_view const key = word.substr(0, eq);
  std::optional<std::string_view> const value =
    eq == std::string_view::npos ? std::nullopt : std::optional<std::string_view>{word.substr(eq + 1)};
  auto const fail = [&](const char* takes) {
    throw "-tune: " + std::string{key} + "= takes " + takes + " (found '" + std::string{word} + "')";
  };
  auto const group = [&](Group3& g) {
    if (!value) {
      g = Group3::On;
    } else if (*value == "0" || *value == "1" || *value == "2") {
      g = Group3((*value)[0] - '0');
    } else {
      fail("0, 1 or 2");
    }
  };

  if (key == "noconfig" && !value) {
    s.sweeps = false;
  } else if (key == "sweeps") {
    if (value != "0" && value != "1") { fail("0 or 1"); }
    s.sweeps = value == "1";
  } else if (key == "fp64" && !value) {
    s.types = s.types == ClassicTypes::Ntt || s.types == ClassicTypes::Both ? ClassicTypes::Both : ClassicTypes::Fp64;
  } else if (key == "ntt" && !value) {
    s.types = s.types == ClassicTypes::Fp64 || s.types == ClassicTypes::Both ? ClassicTypes::Both : ClassicTypes::Ntt;
  } else if (key == "types") {
    if (value == "auto") {
      s.types = ClassicTypes::Auto;
    } else if (value == "fp64") {
      s.types = ClassicTypes::Fp64;
    } else if (value == "ntt") {
      s.types = ClassicTypes::Ntt;
    } else if (value == "both") {
      s.types = ClassicTypes::Both;
    } else {
      fail("auto, fp64, ntt or both");
    }
  } else if (key == "inplace") {
    if (!value || value == "1") {
      s.inplaceOnly = true;
    } else if (value == "0") {
      s.inplaceOnly = false;
    } else if (value == "auto") {
      s.inplaceOnly.reset();
    } else {
      fail("auto, 0 or 1");
    }
  } else if (key == "nofp32" && !value) {
    s.fp32 = Group3::Off;
  } else if (key == "fp32") {
    group(s.fp32);
  } else if (key == "fp6431") {
    group(s.fp6431);
  } else if (key == "1k256") {
    group(s.k1x256);
  } else if (key == "m61") {
    group(s.m61);
  } else if (key == "pfa") {
    group(s.pfa);
  } else if (key == "regs" && !value) {
    s.regsOnly = true;
  } else if (key == "variants" && !value) {
    s.variantsOnly = true;
  } else if (key == "fine") {
    if (!value) {
      s.fine = 0.02;
    } else if (auto const p = parsePercent(*value); p && *p <= 0.5) {
      s.fine = *p;
    } else if (auto const n = parseCount(*value); n && *n <= 50) {
      s.fine = *n / 100.0;  // upstream's fine=<pct>
    } else {
      fail("a percentage up to 50%");
    }
  } else if (key == "tk") {
    if (value == "auto") {
      s.tk = -1;
    } else if (value == "0" || value == "1" || value == "2" || value == "3" || value == "11" || value == "13") {
      s.tk = std::stoi(std::string{*value});
    } else {
      fail("auto, 0 or a TAIL_KERNELS value: 1, 2, 3, 11 or 13");
    }
  } else if (key == "keepMargin") {
    if (value == "upstream") {
      s.keepMargin.reset();
    } else if (auto const p = value ? parsePercent(*value) : std::nullopt) {
      s.keepMargin = *p;
    } else {
      fail("upstream or a percentage");
    }
  } else if (key == "typesMargin") {
    if (auto const p = value ? parsePercent(*value) : std::nullopt) {
      s.typeMargin = *p;
    } else {
      fail("a percentage");
    }
  } else if (key == "whVariants") {
    if (value == "fastest") {
      s.whVariants = WhVariants::Fastest;
    } else if (value == "accurate") {
      s.whVariants = WhVariants::Accurate;
    } else if (value == "all") {
      s.whVariants = WhVariants::All;
    } else {
      fail("fastest, accurate or all");
    }
  } else if (key == "maxCalls") {
    auto const n = value ? parseCount(*value) : std::nullopt;
    if (!n || *n < MIN_CALLS) { fail("a count of 2 or more"); }
    s.maxCalls = *n;
  } else if (key == "tkLosses") {
    auto const n = value ? parseCount(*value) : std::nullopt;
    if (!n || *n < 1) { fail("a count of 1 or more"); }
    s.tkLosses = *n;
  } else if (key == "tkBad") {
    if (auto const p = value ? parsePercent(*value) : std::nullopt) {
      s.tkBad = *p;
    } else {
      fail("a percentage");
    }
  } else {
    return false;
  }
  return true;
}

// The readings of an env, each under its canonical option set, looked up by FFT, kind and exponent; rebuilt whenever
// a row has been added since.
class ReadingIndex {
public:
  ReadingIndex(const TuneDB& db, u32 env, const Env& device) : db_{db}, env_{env}, device_{device} {}

  [[nodiscard]] std::optional<Measurement> find(const FFTConfig& fft, TestKind kind, u64 exponent,
                                                const UseConfig& options) {
    refresh();
    auto const at = rows_.find({fft.spec(), kind, exponent});
    if (at == rows_.end()) { return {}; }
    std::string const want = configText(canonicalConfig(device_, fft, options));
    auto const hit = at->second.find(want);
    if (hit == at->second.end()) { return {}; }
    return hit->second;
  }

private:
  void refresh() {
    if (seen_ == db_.runs().size()) { return; }
    seen_ = db_.runs().size();
    rows_.clear();
    for (const RunRow& row : db_.mergedRuns()) {
      if (db_.envOf(row.sess) != env_ || !row.m.ok()) { continue; }
      const UseConfig* const opts = db_.findCfg(row.cfg);
      auto const fft = parseFft(row.fft);
      if (!opts || !fft) { continue; }
      rows_[{row.fft, row.kind, row.exponent}][configText(canonicalConfig(device_, *fft, *opts))] = row.m;
    }
  }

  const TuneDB& db_;
  u32 env_;
  const Env& device_;
  size_t seen_ = size_t(-1);
  std::map<std::tuple<std::string, TestKind, u64>, std::map<std::string, Measurement>> rows_;
};

std::optional<Measurement> classicReading(const TuneDB& db, u32 env, const Env& device, const FFTConfig& fft,
                                          TestKind kind, u64 exponent, const UseConfig& options) {
  return ReadingIndex{db, env, device}.find(fft, kind, exponent, options);
}

ReplayBench::ReplayBench(const TuneDB& db, u32 env, const Env& device) :
  device_{device}, index_{std::make_unique<ReadingIndex>(db, env, device)} {}

ReplayBench::~ReplayBench() = default;

std::optional<Measurement> ReplayBench::read(const FFTConfig& fft, TestKind kind, u64 exponent,
                                             const UseConfig& options, u32 calls) {
  if (!next_.empty()) { return {}; }
  std::optional<Measurement> const m = index_->find(fft, kind, exponent, options);
  if (m && m->calls >= calls) { return m; }
  UseConfig const canonical = canonicalConfig(device_, fft, options);
  next_ = fft.spec() + " " + toString(kind) + " " + (canonical.empty() ? "at its defaults" : configText(canonical)) +
    " at " + std::to_string(exponent) + " (call " + std::to_string(m ? m->calls + 1 : 1) + ")";
  return {};
}

bool ReplayBench::gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) {
  if (next_.empty()) {
    UseConfig const canonical = canonicalConfig(device_, fft, options);
    next_ = "gate " + fft.spec() + " " + (canonical.empty() ? "at its defaults" : configText(canonical)) + " at " +
      std::to_string(exponent);
  }
  return false;
}


namespace {

// Upstream's threshold of each option sweep (configsUpdate): how much faster a value had to time than the value in
// force before -tune wrote it to config.txt rather than suggesting it.
double upstreamMargin(const std::string& key) {
  static const std::map<std::string, double> margins{
    {"IN_WG", 0.003},          {"OUT_WG", 0.003},         {"INPLACE", 0.002},        {"TAIL_TRIGS", 0.003},
    {"TAIL_TRIGS31", 0.003},   {"TAIL_TRIGS32", 0.003},   {"TAIL_TRIGS61", 0.003},   {"TABMUL_CHAIN", 0.003},
    {"TABMUL_CHAIN31", 0.003}, {"TABMUL_CHAIN32", 0.003}, {"TABMUL_CHAIN61", 0.003}, {"UNROLL_W", 0.003},
    {"UNROLL_H", 0.003},       {"ZEROHACK_W", 0.003},     {"ZEROHACK_H", 0.003},     {"GRAPHS", 0.003},
  };
  auto const at = margins.find(key);
  return at == margins.end() ? 0 : at->second;
}

// One FFT with what the run has made of it: upstream's tune.txt entry, over one regime band of this tuner's.
struct Entry {
  FFTConfig fft;
  Interval band{};
  u64 exponent = 0;
  UseConfig config{};
  double cost = 0;
  bool added = false;

  [[nodiscard]] u64 reach() const { return band.hi; }
  [[nodiscard]] std::string label() const { return fft.spec() + (config.empty() ? "" : " " + configText(config)); }
};

// Upstream's TuneEntry::update: the frontier of cost against reach, cheapest first.  Whether `e` joined it.
bool joinFrontier(std::vector<Entry>& frontier, const Entry& e) {
  if (!std::isfinite(e.cost)) { return false; }
  int i = int(frontier.size()) - 1;
  for (; i >= 0 && frontier[size_t(i)].cost > e.cost; --i) {
    if (frontier[size_t(i)].reach() <= e.reach()) { frontier.erase(frontier.begin() + i); }
  }
  if (i >= 0 && frontier[size_t(i)].reach() >= e.reach()) { return false; }
  frontier.insert(frontier.begin() + (i + 1), e);
  return true;
}

// Upstream's TuneEntry::willUpdate.
bool wouldJoin(const std::vector<Entry>& frontier, double cost, u64 reach) {
  for (const Entry& f : frontier) {
    if (f.cost > cost) { break; }
    if (f.reach() >= reach) { return false; }
  }
  return true;
}

// The value of `digit` (0 = units) of a LOADS or STORES value replaced by `d`.
int withDigit(int value, int digit, int d) {
  int scale = 1;
  for (int i = 0; i < digit; ++i) { scale *= 10; }
  return value - value / scale % 10 * scale + d * scale;
}

class KindRun {
public:
  KindRun(const ClassicContext& c, ClassicBench& bench, TestKind kind, std::vector<Baseline> baselines,
          std::vector<Family> families) :
    c_{c}, bench_{bench}, kind_{kind}, baselines_{std::move(baselines)}, families_{std::move(families)} {}

  void run() {
    const ClassicSettings& s = c_.settings;
    say(std::string{"classic mode, "} + toString(kind_) + ": " +
        (s.regsOnly         ? "the register limits of the frontier"
           : s.variantsOnly ? "the variants of the frontier"
                            : "option sweeps, then every shape"));
    if (s.regsOnly) { return regsOnly(); }
    if (s.variantsOnly) { return variantsOnly(); }

    chooseTypes();
    if (stopped()) { return; }
    if (sweeps_) { optionSweeps(); }
    if (stopped()) { return; }
    shapeSweep();
  }

  [[nodiscard]] const UseConfig& lines() const { return lines_; }
  [[nodiscard]] const std::vector<Entry>& frontier() const { return frontier_; }
  [[nodiscard]] const std::string& stage() const { return stage_; }
  [[nodiscard]] TestKind kind() const { return kind_; }

private:
  [[nodiscard]] bool stopped() const { return bench_.stopped(); }
  void say(const std::string& text) { bench_.say("classic: " + text); }

  [[nodiscard]] const Family* family(enum FFT_TYPES type) const {
    auto const at = std::ranges::find(families_, type, &Family::type);
    return at == families_.end() ? nullptr : &*at;
  }

  // `config` as `fft` takes it: canonical, and without any value the table does not offer it beside the rest, which
  // production would fit away too.
  [[nodiscard]] UseConfig fitted(const FFTConfig& fft, UseConfig config) const {
    for (bool changed = true; changed;) {
      changed = false;
      config = canonicalConfig(c_.device, fft, config);
      for (auto it = config.begin(); it != config.end(); ++it) {
        const Option* const option = findOption(it->first);
        if (!option || option->compound) { continue; }
        std::vector<int> const values = option->valuesFor(c_.device, fft, config);
        if (std::ranges::find(values, std::atoi(it->second.c_str())) == values.end()) {
          config.erase(it);
          changed = true;
          break;
        }
      }
    }
    return config;
  }

  // Whether `fft` can run `config` as written: every key at a value the table offers it, and every access class at a
  // mode this device can emit.
  [[nodiscard]] bool offers(const FFTConfig& fft, const UseConfig& config) const {
    UseConfig const canonical = canonicalConfig(c_.device, fft, config);
    for (const auto& [key, value] : canonical) {
      const Option* const option = findOption(key);
      if (!option->compound) {
        std::vector<int> const values = option->valuesFor(c_.device, fft, canonical);
        if (std::ranges::find(values, std::atoi(value.c_str())) == values.end()) { return false; }
      }
    }
    for (const Axis& axis : axesOf(c_.device, fft, {})) {
      if (!axis.option->compound) { continue; }
      if (std::ranges::find(axis.values, positionOf(c_.device, fft, canonical, axis)) == axis.values.end()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::optional<Measurement> read(const FFTConfig& fft, u64 exponent, const UseConfig& config,
                                                u32 calls = MIN_CALLS) {
    return bench_.read(fft, kind_, exponent, canonicalConfig(c_.device, fft, config), calls);
  }

  // Whether `a` reads faster than `b` by more than `margin`, beyond both error bars: after MIN_CALLS calls each, and
  // while that is undecided, one more call of each up to maxCalls.
  [[nodiscard]] bool beats(const FFTConfig& fft, u64 exponent, const UseConfig& a, const UseConfig& b, double margin) {
    for (u32 calls = MIN_CALLS;; ++calls) {
      std::optional<Measurement> const ma = read(fft, exponent, a, calls);
      std::optional<Measurement> const mb = read(fft, exponent, b, calls);
      if (!ma) { return false; }
      if (!mb) { return true; }
      double const gap = mb->cost() * (1 - margin) - ma->cost();
      if (gap <= 0) { return false; }
      double const sigma = std::hypot(standardError(*ma), standardError(*mb));
      if (gap > PESSIMISM_SIGMA * sigma) { return true; }
      if (calls >= c_.settings.maxCalls || stopped()) { return false; }
    }
  }

  void chooseTypes() {
    stage_ = "type check";
    const ClassicSettings& s = c_.settings;
    const Family* const fp64 = c_.device.hasFP64 ? family(FFT64) : nullptr;
    const Family* const ntt = family(FFT3161);
    timeFP64_ = s.types == ClassicTypes::Fp64 || s.types == ClassicTypes::Both;
    timeNTT_ = s.types == ClassicTypes::Ntt || s.types == ClassicTypes::Both;
    if (s.types == ClassicTypes::Auto) {
      if (!fp64 || !ntt) {
        timeFP64_ = fp64 != nullptr;
        timeNTT_ = ntt != nullptr;
      } else {
        std::optional<Measurement> const f = read(fp64->fft, c_.scope.probe, {});
        std::optional<Measurement> const n = read(ntt->fft, c_.scope.probe, {});
        if (!f || !n) {
          timeFP64_ = f.has_value();
          timeNTT_ = n.has_value();
        } else if (f->cost() < n->cost()) {
          timeFP64_ = true;
          timeNTT_ = f->cost() >= (1 - s.typeMargin) * n->cost();
        } else {
          timeNTT_ = true;
          timeFP64_ = f->cost() <= (1 + s.typeMargin) * n->cost();
        }
        if (f && n) {
          char buf[192];
          snprintf(buf, sizeof(buf), "FP64 %s %.3f us/it, M31*M61 %s %.3f: tuning %s", fp64->fft.spec().c_str(),
                   f->cost(), ntt->fft.spec().c_str(), n->cost(),
                   timeFP64_ && timeNTT_ ? "both"
                     : timeFP64_         ? "FP64 FFTs"
                                         : "NTTs");
          say(buf);
        }
      }
    }
    if (!c_.device.hasFP64) {
      timeFP64_ = false;
      timeNTT_ = true;
    }

    // A group set to 2 times only its own FFTs, and the option sweeps are left alone.
    onlyGroups_ = s.fp32 == Group3::Only || s.fp6431 == Group3::Only || s.k1x256 == Group3::Only ||
      s.m61 == Group3::Only || s.pfa == Group3::Only;
    if (s.fp32 == Group3::Only || s.pfa == Group3::Only) { timeNTT_ = true; }
    timeFP6431_ = s.fp6431 != Group3::Off || (s.pfa == Group3::Only && c_.device.hasFP64);
    sweeps_ = s.sweeps && !onlyGroups_;
  }

  // One option sweep: `deltas` each laid over the lines on `fft`, the cheapest kept where it beats what the lines run.
  void sweep(const char* title, const FFTConfig& fft, const std::vector<UseConfig>& deltas, const std::string& key) {
    if (stopped()) { return; }
    u64 const exponent = c_.scope.probe;
    UseConfig const base = fitted(fft, lines_);
    std::vector<std::pair<UseConfig, UseConfig>> candidates;  // (delta, fitted set)
    std::set<std::string> seen;
    for (const UseConfig& delta : deltas) {
      UseConfig set = lines_;
      for (const auto& [k, v] : delta) { set[k] = v; }
      if (!offers(fft, set)) { continue; }
      UseConfig const canonical = canonicalConfig(c_.device, fft, set);
      if (seen.insert(configText(canonical)).second) { candidates.emplace_back(delta, canonical); }
    }
    if (candidates.size() < 2 && seen.contains(configText(base))) { return; }

    stage_ = std::string{"option sweep: "} + title;
    std::optional<std::pair<UseConfig, double>> best;
    std::string readings;
    for (const auto& [delta, set] : candidates) {
      std::optional<Measurement> const m = read(fft, exponent, set);
      if (stopped()) { return; }
      if (!m) { continue; }
      readings += (readings.empty() ? "" : ", ") + (delta.empty() ? std::string{"-"} : configText(delta)) + " " +
        std::to_string(m->cost()).substr(0, 7);
      if (!best || m->cost() < best->second) { best = {delta, m->cost()}; }
    }
    if (!best) { return; }

    UseConfig chosen = lines_;
    for (const auto& [k, v] : best->first) { chosen[k] = v; }
    double const margin = c_.settings.keepMargin.value_or(upstreamMargin(key));
    bool const kept = configText(canonicalConfig(c_.device, fft, chosen)) != configText(base) &&
      beats(fft, exponent, chosen, base, margin);
    if (stopped()) { return; }
    if (kept) { lines_ = std::move(chosen); }
    say(std::string{title} + " on " + fft.spec() + ": " + readings + " -> " +
        (kept ? configText(best->first) + " kept" : "the lines' own kept"));
  }

  void sweepValues(const char* key, const FFTConfig& fft, std::initializer_list<int> values) {
    std::vector<UseConfig> deltas;
    for (int const v : values) { deltas.push_back({{key, std::to_string(v)}}); }
    sweep(key, fft, deltas, key);
  }

  void optionSweeps() {
    const Family* const fp64 = timeFP64_ ? family(FFT64) : nullptr;
    const Family* const ntt = timeNTT_ ? family(FFT3161) : nullptr;
    const Family* const fp32 = timeNTT_ && c_.settings.fp32 != Group3::Off ? family(FFT3261) : nullptr;
    const Family* const def = ntt ? ntt : fp64;
    if (!def) { return; }
    for (const Family* f : {fp64, ntt, fp32}) {
      if (f) { bench_.declareBootstrap(f->fft, c_.scope.probe); }
    }

    bool const inplaceOnly = c_.settings.inplaceOnly.value_or(c_.device.isNvidia);
    const FFTConfig& d = def->fft;

    if (!inplaceOnly) {
      lines_["INPLACE"] = "0";
      std::vector<UseConfig> in, out;
      for (int const wg : {64, 128, 256}) {
        for (int const x : {8, 16, 32}) {
          in.push_back({{"IN_WG", std::to_string(wg)}, {"IN_SIZEX", std::to_string(x)}});
          out.push_back({{"OUT_WG", std::to_string(wg)}, {"OUT_SIZEX", std::to_string(x)}});
        }
      }
      sweep("IN_WG, IN_SIZEX", d, in, "IN_WG");
      sweep("OUT_WG, OUT_SIZEX", d, out, "OUT_WG");
      sweepValues("PAD", d, {0, 64, 128, 256, 512});
      sweepValues("MIDDLE_IN_LDS_TRANSPOSE", d, {0, 1});
      sweepValues("MIDDLE_OUT_LDS_TRANSPOSE", d, {0, 1});
      sweepValues("INPLACE", d, {0, 1});
    } else {
      lines_["INPLACE"] = "1";
    }

    // LOADS and STORES a digit at a time: FFT data, the carry shuttle's load/store pairs, then the three trig classes.
    auto const digits = [&](const char* title, const char* key, int digit, std::initializer_list<int> modes) {
      int const current = std::atoi((lines_.contains(key) ? lines_[key] : "0").c_str());
      std::vector<UseConfig> deltas;
      for (int const m : modes) { deltas.push_back({{key, std::to_string(withDigit(current, digit, m))}}); }
      sweep(title, d, deltas, key);
    };
    digits("FFT data loads", "LOADS", 0, {0, 1, 2, 3, 4});
    digits("FFT data stores", "STORES", 0, {0, 1, 2, 3});
    {
      int const loads = std::atoi((lines_.contains("LOADS") ? lines_["LOADS"] : "0").c_str());
      int const stores = std::atoi((lines_.contains("STORES") ? lines_["STORES"] : "0").c_str());
      std::vector<UseConfig> deltas;
      for (auto const& [l, st] : {std::pair{0, 0}, std::pair{1, 1}, std::pair{4, 2}}) {
        deltas.push_back(
          {{"LOADS", std::to_string(withDigit(loads, 1, l))}, {"STORES", std::to_string(withDigit(stores, 1, st))}});
      }
      sweep("carry shuttle loads/stores", d, deltas, "LOADS");
    }
    digits("trig loads, frequently used", "LOADS", 2, {0, 5});
    digits("trig loads, several uses", "LOADS", 3, {0, 1, 2, 3, 4, 5});
    digits("trig loads, used once", "LOADS", 4, {0, 1, 2, 3, 4, 5});

    if (!c_.device.cudaBackend) { sweepValues("FAST_BARRIER", d, {0, 1}); }

    const Family* const gf31 = ntt && ntt->fft.NTT_GF31 ? ntt : timeNTT_ ? family(FFT3161) : nullptr;
    const Family* const gf61 = ntt && ntt->fft.NTT_GF61 ? ntt : timeNTT_ ? family(FFT3161) : nullptr;
    const Family* const f32 = ntt && ntt->fft.FFT_FP32 ? ntt : fp32;
    if (fp64) { sweepValues("TAIL_TRIGS", fp64->fft, {0, 1, 2}); }
    if (gf31) { sweepValues("TAIL_TRIGS31", gf31->fft, {0, 1}); }
    if (f32) { sweepValues("TAIL_TRIGS32", f32->fft, {0, 1, 2}); }
    if (gf61) { sweepValues("TAIL_TRIGS61", gf61->fft, {0, 1}); }
    if (fp64) { sweepValues("TABMUL_CHAIN", fp64->fft, {0, 1}); }
    if (gf31) { sweepValues("TABMUL_CHAIN31", gf31->fft, {0, 1}); }
    if (f32) { sweepValues("TABMUL_CHAIN32", f32->fft, {0, 1}); }
    if (gf61) { sweepValues("TABMUL_CHAIN61", gf61->fft, {0, 1}); }
    if (gf31) { sweepValues("MODM31", gf31->fft, {0, 1, 2}); }
    sweepValues("UNROLL_W", d, {0, 1});
    sweepValues("UNROLL_H", d, {0, 1});
    sweepValues("ZEROHACK_W", d, {0, 1});
    sweepValues("ZEROHACK_H", d, {0, 1});
    if (d.shape.width != 4096) { sweepValues("WMUL", d, {1, 2, 4}); }
    if (gf61) { sweepValues("MULTI_Q", gf61->fft, {0, 1}); }
    if (c_.device.cudaBackend) {
      sweepValues("L1CUDA", d, {0, 1, 2, 3});
      sweepValues("GRAPHS", d, {0, 1});
    }
    // After MULTI_Q and GRAPHS: 11 and 13 add queues, whose cost depends on those.
    {
      std::vector<UseConfig> deltas;
      for (int const tk : {0, 1, 11, 2, 3, 13}) {
        if (tk >= 10 && !queuesApart(d)) { continue; }
        deltas.push_back({{"TAIL_KERNELS", std::to_string(tk)}});
      }
      sweep("TAIL_KERNELS", d, deltas, "TAIL_KERNELS");
    }
    say("option sweeps done: the lines are " + (lines_.empty() ? std::string{"empty"} : configText(lines_)));
  }

  // Whether TAIL_KERNELS 11 and 13 are tried on `fft`, as the search's tailQueues= says.
  [[nodiscard]] bool queuesApart(const FFTConfig& fft) const {
    return c_.tailQueues == TailQueues::All ||
      (c_.tailQueues == TailQueues::Ntt && c_.device.isNvidia && (fft.NTT_GF31 || fft.NTT_GF61));
  }

  // Whether the shape sweep times this FFT type and shape at all, by upstream's types and groups.
  [[nodiscard]] bool wanted(const FFTConfig& fft) const {
    enum FFT_TYPES const t = fft.shape.fft_type;
    bool const fp32Type = fft.FFT_FP32;
    if (t == FFT64 && !timeFP64_) { return false; }
    if (t == FFT6431 && !timeFP6431_) { return false; }
    if (t != FFT64 && t != FFT6431 && !timeNTT_) { return false; }
    if (fp32Type && c_.settings.fp32 == Group3::Off) { return false; }

    bool skip = false, inOnly = false;
    const FFTShape& sh = fft.shape;
    for (auto [in, g] :
         {std::pair{(sh.width == 256 && sh.height == 1024) || (sh.width == 1024 && sh.height == 256),
                    c_.settings.k1x256},
          std::pair{t == FFT61, c_.settings.m61}, std::pair{t == FFT6431, timeFP6431_ ? Group3::On : Group3::Off},
          std::pair{fp32Type, c_.settings.fp32},
          std::pair{sh.isPfa() && (fft.FFT_FP64 || fft.FFT_FP32), c_.settings.pfa}}) {
      if (!in) { continue; }
      if (g == Group3::Off) { skip = true; }
      if (g == Group3::Only) { inOnly = true; }
    }
    if (t == FFT6431 && c_.settings.fp6431 == Group3::Only) { inOnly = true; }
    return !skip && (!onlyGroups_ || inOnly);
  }

  // The width (height) variant of FP64 that times fastest at this width (height), on upstream's test shapes.
  [[nodiscard]] std::optional<u32> fastestDigit(u32 size, bool height) {
    auto& memo = height ? fastestH_ : fastestW_;
    if (auto const at = memo.find(size); at != memo.end()) { return at->second; }
    FFTShape const test = height ? FFTShape{FFT64, size, 12, size} : FFTShape{FFT64, size, 12, 256};
    std::vector<u32> const runnable = runnableVariants(c_.device, test);
    std::optional<std::pair<u32, double>> best;
    for (u32 digit = 0; digit < (height ? N_VARIANT_H : N_VARIANT_W); ++digit) {
      u32 const variant = height ? variant_WMH(1, 0, digit) : variant_WMH(digit, 0, 1);
      if (std::ranges::find(runnable, variant) == runnable.end()) { continue; }
      FFTConfig const fft{test, variant, CARRY_32};
      if (variant_W(fft.variant) != variant_W(variant) || variant_H(fft.variant) != variant_H(variant)) { continue; }
      std::optional<Measurement> const m = read(fft, primes_.prevPrime(maxExp(fft)), fitted(fft, lines_));
      if (stopped()) { return {}; }
      if (m && (!best || m->cost() < best->second)) { best = {digit, m->cost()}; }
    }
    if (!best) { return {}; }
    memo[size] = best->first;
    say(std::string{height ? "fastest height variant at " : "fastest width variant at "} + numberK(size) + ": " +
        std::to_string(best->first));
    return best->first;
  }

  [[nodiscard]] bool variantPruned(const FFTConfig& fft) {
    if (!fft.FFT_FP64 || c_.settings.whVariants == WhVariants::All) { return false; }
    std::optional<u32> const w = fastestDigit(fft.shape.width, false);
    std::optional<u32> const h = fastestDigit(fft.shape.height, true);
    if (stopped()) { return true; }
    if (c_.settings.whVariants == WhVariants::Fastest) {
      return (w && variant_W(fft.variant) != *w) || (h && variant_H(fft.variant) != *h);
    }
    auto const atLeastAsAccurate = [&](u32 variant) {
      return FFTConfig{fft.shape, fft.variant, CARRY_32}.maxBpw() >= FFTConfig{fft.shape, variant, CARRY_32}.maxBpw();
    };
    return (w && !atLeastAsAccurate(variant_WMH(*w, variant_M(fft.variant), variant_H(fft.variant)))) ||
      (h && !atLeastAsAccurate(variant_WMH(variant_W(fft.variant), variant_M(fft.variant), *h)));
  }

  // Times `b` under the lines; the entry it makes, finite where it was read.
  [[nodiscard]] Entry entryOf(const Baseline& b) {
    UseConfig const config = fitted(b.fft, lines_);
    std::optional<Measurement> const m = read(b.fft, b.exponent, config);
    return Entry{.fft = b.fft,
                 .band = b.band,
                 .exponent = b.exponent,
                 .config = config,
                 .cost = m ? m->cost() : INFINITY,
                 .added = true};
  }

  void consider(Entry e) {
    if (c_.settings.fine > 0 && std::isfinite(e.cost) &&
        wouldJoin(frontier_, e.cost * (1 - c_.settings.fine), e.reach())) {
      say(e.fft.spec() + " is within " + percent(c_.settings.fine) + " of the frontier: tuning its register limits");
      registerPass(e);
      if (stopped()) { return; }
    }
    bool const joined = joinFrontier(frontier_, e);
    char buf[160];
    snprintf(buf, sizeof(buf), "%c %8.3f %-22s %llu", joined ? '*' : ' ', e.cost, e.label().c_str(),
             (unsigned long long)e.reach());
    say(buf);
  }

  void shapeSweep() {
    stage_ = "shape sweep";
    u64 const lo = c_.scope.lo;
    u64 const hi = c_.scope.hi;
    std::vector<const Baseline*> order;
    for (const Baseline& b : baselines_) {
      const FFTShape& sh = b.fft.shape;
      if (!wanted(b.fft) || !sh.isFavoredShape()) { continue; }
      if ((sh.width == 256 && sh.height == 2048) || (sh.width == 2048 && sh.height == 256)) { continue; }
      if ((sh.width == 512 && sh.height == 2048) || (sh.width == 2048 && sh.height == 512)) { continue; }
      if (b.fft.FFT_FP64 && variant_M(b.fft.variant) == 1) { continue; }
      u64 const top = maxExp(b.fft);
      if (top < lo || double(top) > (b.fft.shape.fft_type == FFT64 ? 1.2 : 1.3) * double(hi)) { continue; }
      order.push_back(&b);
    }
    std::ranges::stable_sort(order, {}, [](const Baseline* b) { return b->band.hi; });

    for (const Baseline* b : order) {
      if (variantPruned(b->fft)) {
        if (stopped()) { return; }
        continue;
      }
      Entry e = entryOf(*b);
      if (stopped()) { return; }
      consider(std::move(e));
      if (stopped()) { return; }
    }
    pruneAbove();

    // Middle variant 1 holds a slightly higher exponent, usually at a small cost: tried for the middle variant 0
    // entries added below the workload's top, which it may take the next entry from.
    stage_ = "middle variant 1";
    std::vector<Entry> const added = frontier_;
    for (const Entry& e : added) {
      if (!e.added || !e.fft.FFT_FP64 || variant_M(e.fft.variant) != 0 || e.reach() >= hi) { continue; }
      FFTConfig const next{e.fft.shape, e.fft.variant + 10, e.fft.carry};
      if (next.spec() == e.fft.spec() || maxExp(next) <= maxExp(e.fft)) { continue; }
      for (const Baseline& b : baselines_) {
        if (b.fft.spec() != next.spec()) { continue; }
        Entry m = entryOf(b);
        if (stopped()) { return; }
        consider(std::move(m));
      }
    }
    pruneAbove();

    if (c_.settings.fine == 0) {
      stage_ = "register limits";
      std::vector<Entry> frontier = frontier_;
      for (Entry& e : frontier) {
        if (!e.added) { continue; }
        registerPass(e);
        if (stopped()) { return; }
        tailChoices(e);
        if (stopped()) { return; }
      }
      frontier_ = std::move(frontier);
    }
  }

  // Of the entries above the workload's top only the cheapest is kept: it is the one that covers the top.
  void pruneAbove() {
    std::vector<Entry> kept;
    bool above = false;
    for (const Entry& e : frontier_) {
      bool const isAbove = e.reach() >= c_.scope.hi;
      if (!isAbove || !above) { kept.push_back(e); }
      above = above || isAbove;
    }
    frontier_ = std::move(kept);
  }

  // The REG* keys of `e`'s kernels, in the table's order, each with its plan.
  [[nodiscard]] std::vector<std::pair<std::string, RegisterPlan>> plansOf(const Entry& e) const {
    std::vector<std::pair<std::string, RegisterPlan>> out;
    for (const Option& o : allOptions()) {
      if (o.key == "NOREG" || !Args::isRegisterKey(o.key) || !o.appliesTo(c_.device, e.fft, e.config)) { continue; }
      if (const RegsRow* const row = c_.db.findRegs(c_.env, e.fft.spec(), kind_, o.key)) {
        out.emplace_back(o.key, row->plan);
      }
    }
    return out;
  }

  [[nodiscard]] UseConfig with(UseConfig config, const std::string& key, int value) const {
    config[key] = std::to_string(value);
    return config;
  }

  // Upstream's register pass (Tune::regTuneEntry): each kernel's limits in its order, each kept where it beats the
  // compiler's registers, then all of them together, or one at a time where the queues overlap.
  void registerPass(Entry& e) {
    UseConfig base = e.config;
    std::erase_if(base, [](const auto& kv) { return Args::isRegisterKey(kv.first); });
    std::optional<Measurement> const mb = read(e.fft, e.exponent, base);
    if (!mb || stopped()) { return; }

    struct Win {
      std::string key;
      int value;
      double saving;
    };
    std::vector<Win> wins;
    for (const auto& [key, plan] : plansOf(e)) {
      std::map<int, double> tried;
      auto const time = [&](int v) {
        if (tried.contains(v)) { return; }
        std::optional<Measurement> const m = read(e.fft, e.exponent, with(base, key, v));
        tried[v] = m ? m->cost() : INFINITY;
      };
      auto const has = [&](int blocks) {
        return std::ranges::any_of(plan.occupancies, [&](auto const& o) { return o.first == blocks; });
      };
      auto const at = [&](int v) { return tried.contains(v) ? tried[v] : INFINITY; };
      int const k0 = plan.k0;
      if (!c_.device.cudaBackend && plan.lift) { time(-1); }
      if (c_.device.cudaBackend && has(k0)) { time(k0); }
      if (c_.device.cudaBackend && has(k0 - 1)) { time(k0 - 1); }
      if (has(k0 + 1)) { time(k0 + 1); }
      if (has(k0 + 2) && at(k0 + 1) < std::min(at(k0), mb->cost())) { time(k0 + 2); }
      if (stopped()) { return; }
      // The register count of the fastest occupancy, or of k0 where none beat the compiler.
      int bestBlocks = k0;
      double bestUs = mb->cost();
      for (auto const& [v, us] : tried) {
        if (v > 0 && v <= 20 && us < bestUs) {
          bestUs = us;
          bestBlocks = v;
        }
      }
      for (auto const& [blocks, regs] : plan.occupancies) {
        if (blocks == bestBlocks && regs > 0) { time(regs); }
      }
      if (stopped()) { return; }

      std::optional<std::pair<int, double>> best;
      for (auto const& [v, us] : tried) {
        if (!best || us < best->second) { best = {v, us}; }
      }
      if (best && beats(e.fft, e.exponent, with(base, key, best->first), base, 0)) {
        wins.push_back({key, best->first, mb->cost() - best->second});
        say(e.fft.spec() + " " + key + "=" + std::to_string(best->first) + " beats the compiler's registers");
      }
      if (stopped()) { return; }
    }
    if (wins.empty()) { return; }

    bool const overlap = useValue(base, "MULTI_Q", 0) &&
      int(e.fft.FFT_FP64) + int(e.fft.FFT_FP32) + int(e.fft.NTT_GF31) + int(e.fft.NTT_GF61) >= 2;
    UseConfig tuned = base;
    if (overlap && wins.size() >= 2) {
      std::ranges::sort(wins, [](const Win& a, const Win& b) { return a.saving > b.saving; });
      for (const Win& w : wins) {
        UseConfig const trial = with(tuned, w.key, w.value);
        if (beats(e.fft, e.exponent, trial, tuned, 0)) { tuned = trial; }
        if (stopped()) { return; }
      }
    } else {
      for (const Win& w : wins) { tuned[w.key] = std::to_string(w.value); }
      if (!beats(e.fft, e.exponent, tuned, base, 0)) { tuned = base; }
      if (stopped()) { return; }
    }
    std::optional<Measurement> const mt = read(e.fft, e.exponent, tuned);
    if (!mt) { return; }
    e.config = canonicalConfig(c_.device, e.fft, tuned);
    e.cost = mt->cost();
    say(e.fft.spec() + " register limits: " + (tuned == base ? std::string{"the compiler's"} : configText(e.config)));
  }

  // tk=: the other TAIL_KERNELS choices of the entry's tail width, with its register limits and with the tail's left to
  // the compiler, each kept where it beats the entry.  A choice that keeps losing for an FFT type is no longer tried.
  void tailChoices(Entry& e) {
    int const tk = c_.settings.tk;
    if (tk == 0) { return; }
    int const tk0 = useValue(e.config, "TAIL_KERNELS", 2);
    std::vector<int> alts;
    for (int const alt : tk0 % 10 < 2 ? std::initializer_list<int>{0, 1, 11} : std::initializer_list<int>{2, 3, 13}) {
      if (alt == tk0 || (tk > 0 && alt != tk) || (alt >= 10 && !queuesApart(e.fft))) { continue; }
      alts.push_back(alt);
    }
    if (tk > 0 && tk != tk0 && std::ranges::find(alts, tk) == alts.end() && (tk < 10 || queuesApart(e.fft))) {
      alts.push_back(tk);
    }
    for (int const alt : alts) {
      Record& r = records_[{e.fft.shape.fft_type, alt}];
      if (r.wins == 0 && (r.badLosses * 2 >= c_.settings.tkLosses || r.losses >= c_.settings.tkLosses)) { continue; }
      UseConfig const same = with(e.config, "TAIL_KERNELS", alt);
      UseConfig bare = same;
      std::erase_if(bare, [](const auto& kv) { return kv.first.starts_with("REGTS"); });
      double withLimits = INFINITY;
      double without = INFINITY;
      if (offers(e.fft, same)) {
        if (std::optional<Measurement> const m = read(e.fft, e.exponent, same)) { withLimits = m->cost(); }
      }
      if (bare != same && offers(e.fft, bare)) {
        if (std::optional<Measurement> const m = read(e.fft, e.exponent, bare)) { without = m->cost(); }
      }
      if (stopped()) { return; }
      UseConfig const& pick = without < withLimits ? bare : same;
      double const cost = std::min(withLimits, without);
      if (!std::isfinite(cost)) { continue; }
      if (beats(e.fft, e.exponent, pick, e.config, 0)) {
        ++r.wins;
        e.config = canonicalConfig(c_.device, e.fft, pick);
        e.cost = cost;
        say(e.fft.spec() + " TAIL_KERNELS=" + std::to_string(alt) + " beats its own");
      } else {
        ++r.losses;
        if (cost > (1 + c_.settings.tkBad) * e.cost) { ++r.badLosses; }
      }
      if (stopped()) { return; }
    }
  }

  // The frontier of what the env has measured of this kind, each FFT at its cheapest reading.
  [[nodiscard]] std::vector<Entry> measuredFrontier() const {
    std::vector<Entry> out;
    for (const RunRow& row : c_.db.mergedRuns()) {
      if (c_.db.envOf(row.sess) != c_.env || row.kind != kind_ || !row.m.ok() || !concluded(row.m)) { continue; }
      auto const fft = parseFft(row.fft);
      const UseConfig* const opts = c_.db.findCfg(row.cfg);
      if (!fft || !opts) { continue; }
      (void)joinFrontier(out,
                         Entry{.fft = *fft,
                               .band = interval(*fft, row.exponent),
                               .exponent = row.exponent,
                               .config = canonicalConfig(c_.device, *fft, *opts),
                               .cost = row.m.cost(),
                               .added = false});
    }
    return out;
  }

  // -tune regs: the register limits of every frontier entry used in the workload, again from the compiler's.
  void regsOnly() {
    stage_ = "register limits";
    frontier_ = measuredFrontier();
    u64 below = 0;
    for (Entry& e : frontier_) {
      bool const used = e.reach() >= c_.scope.lo && below <= c_.scope.hi;
      below = e.reach() + 1;
      if (!used) { continue; }
      registerPass(e);
      if (stopped()) { return; }
      tailChoices(e);
      if (stopped()) { return; }
    }
  }

  // -tune variants: every other variant of the frontier's FP64 shapes inside the workload, then the register limits
  // of those that join it.
  void variantsOnly() {
    stage_ = "variants";
    frontier_ = measuredFrontier();
    std::set<std::string> shapes, have;
    for (const Entry& e : frontier_) {
      have.insert(e.fft.spec());
      if (e.fft.FFT_FP64) { shapes.insert(e.fft.shape.spec()); }
    }
    for (const Baseline& b : baselines_) {
      if (!shapes.contains(b.fft.shape.spec()) || have.contains(b.fft.spec())) { continue; }
      if (maxExp(b.fft) < c_.scope.lo || maxExp(b.fft) > c_.scope.hi) { continue; }
      Entry e = entryOf(b);
      if (stopped()) { return; }
      consider(std::move(e));
    }
    stage_ = "register limits";
    for (Entry& e : frontier_) {
      if (!e.added) { continue; }
      registerPass(e);
      if (stopped()) { return; }
    }
  }

  struct Record {
    u32 wins = 0;
    u32 losses = 0;
    u32 badLosses = 0;
  };

  const ClassicContext& c_;
  ClassicBench& bench_;
  TestKind kind_;
  std::vector<Baseline> baselines_;
  std::vector<Family> families_;
  Primes primes_;

  bool timeFP64_ = false;
  bool timeNTT_ = false;
  bool timeFP6431_ = false;
  bool onlyGroups_ = false;
  bool sweeps_ = true;

  UseConfig lines_;
  std::vector<Entry> frontier_;
  std::map<u32, u32> fastestW_, fastestH_;
  std::map<std::pair<enum FFT_TYPES, int>, Record> records_;
  std::string stage_;
};

// The FFTs and exponents of the frontier, which the gate is asked to read for before anything is published.
void gateFrontier(const ClassicContext& c, ClassicBench& bench,
                  const std::vector<std::pair<TestKind, std::string>>& keep) {
  std::map<std::string, u32> attempts;
  for (bool progress = true; progress && !bench.stopped();) {
    progress = false;
    for (const OptionSet& s : gatesOwed(c.db, c.env)) {
      const SelectionEntry& e = s.entry;
      if (!s.gate.owedAt || std::ranges::find(keep, std::pair{e.kind, e.fft}) == keep.end()) { continue; }
      auto const fft = parseFft(e.fft);
      if (!fft) { continue; }
      UseConfig const options = s.gate.owesReference ? accuracyReference(c.device, *fft, e.opts) : e.opts;
      std::string const key =
        e.fft + " " + configText(canonicalConfig(c.device, *fft, options)) + " " + std::to_string(s.gate.owedAt);
      if (attempts[key]++ >= 2) { continue; }
      if (bench.gate(*fft, s.gate.owedAt, options)) { progress = true; }
      if (bench.stopped()) { return; }
    }
  }
}

}  // namespace

ClassicOutcome runClassic(const ClassicContext& c, ClassicBench& bench) {
  ClassicOutcome out;

  // Upstream times the PFA hybrids by default, so the scope's own pfaShapes= does not narrow what is considered.
  RunScope everything = c.scope;
  everything.pfaShapes = PfaShapes::All;
  std::vector<Baseline> const all = baselines(c.device, everything);
  std::vector<FFTConfig> inScope;
  for (const Baseline& b : all) { inScope.push_back(b.fft); }
  std::vector<Family> const families = bootstrapFamilies(c.device, c.scope.probe, inScope);

  std::vector<std::pair<TestKind, std::string>> published;
  std::vector<std::pair<TestKind, UseConfig>> lines;
  for (TestKind const kind : c.kinds) {
    std::vector<Baseline> mine;
    for (const Baseline& b : all) {
      if (b.kind == kind) { mine.push_back(b); }
    }
    KindRun run{c, bench, kind, std::move(mine), families};
    run.run();
    lines.emplace_back(kind, run.lines());
    for (const Entry& e : run.frontier()) { published.emplace_back(kind, e.fft.spec()); }
    if (bench.stopped()) {
      out.stage = std::string{toString(kind)} + ": " + run.stage();
      break;
    }
  }

  if (!lines.empty()) {
    out.lines.global = lines.front().second;
    for (size_t i = 1; i < lines.size(); ++i) {
      UseConfig own = lines[i].second;
      // Every key the global line sets, at this kind's own value or the built-in one, so that it is this kind's alone.
      for (const auto& [key, value] : out.lines.global) {
        if (own.contains(key) || families.empty()) { continue; }
        if (const Option* const option = findOption(key)) {
          own[key] = std::to_string(option->defaultFor(c.device, families.front().fft, own));
        }
      }
      if (!own.empty()) {
        FFTSelector selector;
        selector.kind = lines[i].first;
        out.lines.family.push_back(UseLine{.selector = selector, .uses = {own.begin(), own.end()}});
      }
    }
  }

  if (!bench.stopped()) {
    out.stage = "accuracy gate";
    gateFrontier(c, bench, published);
  }
  out.complete = !bench.stopped();
  if (out.complete) { out.stage = "done"; }
  return out;
}

namespace {

std::string setText(const FFTConfig& fft, const Env& device, const UseConfig& options) {
  UseConfig const canonical = canonicalConfig(device, fft, options);
  return canonical.empty() ? "the built-in defaults" : configText(canonical);
}

}  // namespace

SessionClassicBench::SessionClassicBench(Bench& bench, const TuneDB& db, u32 env, const Env& device) :
  bench_{bench}, db_{db}, env_{env}, device_{device}, index_{std::make_unique<ReadingIndex>(db, env, device)} {}

SessionClassicBench::~SessionClassicBench() = default;

std::optional<Measurement> SessionClassicBench::read(const FFTConfig& fft, TestKind kind, u64 exponent,
                                                     const UseConfig& options, u32 calls) {
  UseConfig const canonical = canonicalConfig(device_, fft, options);
  std::string const label =
    fft.spec() + " " + toString(kind) + " " + setText(fft, device_, canonical) + " at " + std::to_string(exponent);
  if (failed_.contains(label)) { return {}; }

  for (;;) {
    std::optional<Measurement> const m = index_->find(fft, kind, exponent, canonical);
    if (m && m->calls >= calls) { return m; }
    if (bench_.stopped()) { return {}; }
    if (bench_.anchorDue()) {
      bench_.timeAnchor();
      continue;
    }
    Bench::Result const r = bench_.run(fft, kind, exponent, canonical, "");
    ++calls_;
    seconds_ += r.seconds;
    if (!r.completed) {
      if (!bench_.stopped()) {
        log("tune: classic: %s gave no reading\n", label.c_str());
        failed_.insert(label);
      }
      return {};
    }
    if (canonicalConfig(device_, fft, r.ran) != canonical) {
      log("tune: classic: %s was built as %s, so it cannot be read\n", label.c_str(),
          setText(fft, device_, r.ran).c_str());
      failed_.insert(label);
      return {};
    }
    log("tune: classic: %u. %s (call %u): %.3f us/it, %.1f s\n", calls_, label.c_str(), m ? m->calls + 1 : 1, r.usPerIt,
        r.seconds);
  }
}

bool SessionClassicBench::gate(const FFTConfig& fft, u64 exponent, const UseConfig& options) {
  if (bench_.stopped()) { return false; }
  Bench::Reading const r = bench_.gate(fft, exponent, options);
  log("tune: classic: gate %s %s at %" PRIu64 ": z %.2f over %u rounding errors, check %s\n", fft.spec().c_str(),
      setText(fft, device_, options).c_str(), exponent, r.z, r.n, r.checkOk ? "OK" : "failed");
  return r.completed;
}

void SessionClassicBench::declareBootstrap(const FFTConfig& fft, u64 probe) {
  for (const BootRow& row : db_.boots()) {
    if (db_.envOf(row.sess) == env_ && row.fft == fft.spec() && row.probe == probe) { return; }
  }
  bench_.declareBootstrap(fft, probe);
}

bool SessionClassicBench::stopped() const { return bench_.stopped(); }

void SessionClassicBench::say(const std::string& text) { log("tune: %s\n", text.c_str()); }

}  // namespace tune
