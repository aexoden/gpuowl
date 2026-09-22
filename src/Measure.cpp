// Copyright (C) Jason Lynch

#include "Measure.h"

#include "Args.h"
#include "clwrap.h"
#include "Context.h"
#include "Gpu.h"
#include "GpuFault.h"
#include "log.h"
#include "Primes.h"
#include "Restart.h"
#include "timeutil.h"

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <optional>
#include <vector>

namespace tune {

namespace {

std::vector<KeyVal> asExtraConf(const UseConfig& options) { return {options.begin(), options.end()}; }

u64 now() { return u64(std::time(nullptr)); }

std::string attemptText(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options) {
  return "-fft " + fft.spec() + " -use " + configText(options) + " (" + toString(kind) + ", E=" + to_string(exponent) +
    ")";
}

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

class Attempt {
public:
  Attempt(TuneDB& db, u32 sess, const bool& deviceLost, const TryRow& row, std::string what) :
    db_{db}, sess_{sess}, deviceLost_{deviceLost} {
    declared_ = db_.add(row);
    outer_ = gpufault::attempting(std::move(what));
  }

  [[nodiscard]] bool declared() const { return declared_; }

  Attempt(const Attempt&) = delete;
  Attempt& operator=(const Attempt&) = delete;

  ~Attempt() {
    if (!deviceLost_) { db_.closeTry(sess_); }
    (void)gpufault::attempting(std::move(outer_));
  }

private:
  TuneDB& db_;
  u32 sess_;
  const bool& deviceLost_;
  bool declared_ = false;
  std::string outer_;
};

}  // namespace

Failure classify(std::string_view message) {
  if (contains(message, "stop requested")) { return {.stop = true, .what = std::string{message}}; }

  if (contains(message, "DEVICE_NOT_AVAILABLE") || contains(message, "DEVICE_NOT_FOUND")) {
    return {.stop = true, .fatal = true, .what = std::string{message}};
  }

  if (contains(message, "Can't compile") || contains(message, "Can't find")) {
    return {.status = Status::NoCompile, .what = std::string{message}};
  }

  return {.status = Status::Unsupported, .what = std::string{message}};
}

double minSafeZ(enum FFT_TYPES type) { return type == FFT64 ? 20 : 6; }

Call summarize(const IterSamples& samples) {
  CoreStats const core = coreStats(samples.usPerIt);

  Call out{.measurement = measurementOf(core, 1, u64(std::time(nullptr))),
           .usPerIt = samples.usPerIt,
           .dropped = core.dropped,
           .declined = core.declined,
           .res64 = samples.res64,
           .checkOk = samples.checkOk,
           .ran = {}};

  if (!samples.checkOk) { out.measurement.status = Status::Err; }
  return out;
}

Call timeCall(GpuCommon shared, const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
              u32 nBlocks, u32 blockSize) {
  if (kind != TestKind::PRP) { throw "LL timing is not implemented"; }

  Timer t;
  auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(options), false, kind);
  double const buildSec = t.reset();

  Call out = summarize(gpu->timeIters(nBlocks, blockSize));
  out.timedSec = t.at();
  out.buildSec = buildSec;
  out.ran = gpu->args.flags;
  return out;
}

RoeCheck roeCheck(GpuCommon shared, const FFTConfig& fft, const UseConfig& options, u64 exponent) {
  RoeCheck out{.minZ = minSafeZ(fft.shape.fft_type), .exponent = exponent};

  if (!fft.FFT_FP64 && !fft.FFT_FP32) { return out; }
  out.applicable = true;

  auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(options), false, TestKind::PRP);
  auto [checkOk, res, roeSq, roeMul] = gpu->measureROE(false);

  out.checkOk = checkOk;
  out.z = roeSq.z();
  out.n = roeSq.N;
  out.maxRoe = roeSq.max;
  return out;
}

Session::Session(GpuCommon shared, TuneDB& db, const Env& env) : shared_{shared}, db_{db}, env_{env} {}

bool Session::begin(u64 probe) {
  envId_ = db_.internEnv(dbEnvOf(env_));
  if (!envId_) { return false; }

  if (probe) {
    // The env's first session fixes the anchor for the rest of the env's life: rows are comparable only against
    // readings of one configuration at one exponent, and a later session probing elsewhere still has to divide by the
    // movement of the same thing.
    if (auto const pinned = parseAnchorSpec(db_.envAnchor(envId_))) {
      anchor_ = *pinned;
    } else if (auto const chosen = chooseAnchor(probe)) {
      anchor_ = *chosen;
    } else {
      log("measure: no FP64 configuration is eligible at %" PRIu64 ", so this session has no drift anchor and its\n"
          "measure:   rows are recorded as measured\n",
          probe);
    }

    if (const AnchorRow* const baseline = db_.envBaseline(envId_)) {
      anchorState_.baseline = baseline->mean;
      baselineCfg_ = baseline->cfg;
    }
  }

  session_ = db_.beginSession(envId_, anchor_.valid() ? anchor_.text() : "", restart::generation());
  return session_ != 0;
}

void Session::end() {
  if (session_) { db_.closeTry(session_); }
}

std::string Session::held(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options) const {
  std::string const spec = fft.spec();
  if (db_.isNogo(envId_, spec, options)) { return "one of its options is recorded as unbuildable on this FFT"; }

  u32 const cfg = db_.findCfgId(options);
  if (cfg && db_.diedOn(envId_, cfg, kind, spec, exponent)) {
    return "an earlier generation was holding it when the device went away";
  }
  return {};
}

bool Session::deviceUsable() {
  if (isContextLost()) { return false; }
  try {
    cl_context const context = shared_.context->get();
    QueueHolder const queue{makeQueue(shared_.context->deviceId(), context, false)};
    Holder<cl_mem> const buf{makeBuf_(context, CL_MEM_READ_WRITE, sizeof(u32))};
    u32 const probe = 0x50524F42;
    write(queue.get(), {}, true, buf.get(), sizeof(probe), &probe, false);
    finish(queue.get());
    return true;
  } catch (...) { return false; }
}

void Session::lost(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, const std::string& what,
                   const char* during) {
  stopped_ = true;
  // Only the first one says anything: once the device is gone every later call fails for that reason.
  if (deviceLost_) { return; }
  deviceLost_ = true;

  markContextLost("it did not survive what was being measured");
  db_.sealSession(session_);

  log("measure: the device is no longer usable (%s) -- stopping. Nothing further can be measured on it, and the\n"
      "measure:   measurements already recorded are unaffected. It was lost during the %s of:\n"
      "measure:     -fft %s -use %s\n"
      "measure:     (%s, E=%" PRIu64 ")\n"
      "measure:   that configuration is recorded and will not be built again.\n",
      what.c_str(), during, fft.spec().c_str(), configText(options).c_str(), toString(kind), exponent);
}

void Session::noteNogo(const FFTConfig& fft, const UseConfig& options) {
  // Sound only when one thing moved. Two keys changed together and the failure is a failure of the pair; blaming
  // either would exclude configurations that build perfectly well.
  if (varying_.size() != 1) { return; }
  auto const it = options.find(varying_.front());
  if (it == options.end()) { return; }

  (void)db_.add(NogoRow{.sess = session_, .fft = fft.spec(), .key = it->first, .val = it->second, .ts = now()});
  log("measure: %s: %s=%s will not build here; it is excluded on this FFT whatever else is set\n", fft.spec().c_str(),
      it->first.c_str(), it->second.c_str());
}

Status Session::failed(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                       const std::string& message, const char* during) {
  Failure const f = classify(message);
  if (f.stop) {
    if (f.fatal) {
      lost(fft, kind, exponent, options, f.what, during);
    } else {
      log("measure: %s\n", f.what.c_str());
      stopped_ = true;
    }
    return Status::Lost;
  }

  log("measure: %s during the %s of %s: %s\n", toString(f.status), during, fft.spec().c_str(), f.what.c_str());
  if (f.status == Status::NoCompile) {
    if (!deviceUsable()) {
      lost(fft, kind, exponent, options, "the context did not survive the failed build", during);
      return Status::Lost;
    }
    noteNogo(fft, options);
  }
  return f.status;
}

Status Session::caught(const std::function<void()>& work, const FFTConfig& fft, TestKind kind, u64 exponent,
                       const UseConfig& options, const char* during) {
  try {
    work();
    return Status::Ok;
  } catch (const char* mes) {
    return failed(fft, kind, exponent, options, mes, during);
  } catch (const std::string& mes) {
    return failed(fft, kind, exponent, options, mes, during);
  } catch (const std::exception& e) { return failed(fft, kind, exponent, options, e.what(), during); }
}

void Session::cannotDeclare(const FFTConfig& fft, const UseConfig& options) {
  cannotRecord_ = true;
  stopped_ = true;
  log("measure: the attempt could not be recorded, so %s -use %s was not built. Without that row on disk a fault here\n"
      "measure:   would leave nothing to stop the next run repeating it.\n",
      fft.spec().c_str(), configText(options).c_str());
}

void Session::keepAnchor() {
  if (!anchor_.valid() || inAnchor_ || stopped_) { return; }
  if (anchorState_.readings && sinceAnchor_.at() < ANCHOR_EVERY_SEC) { return; }

  FFTConfig const fft{anchor_.fft};

  // The anchor is built like anything else, so it is held back like anything else. Without this a configuration that
  // took the device down would be rebuilt by every generation the restart limit allows, since the env names it and
  // every session re-times it.
  if (std::string const why = held(fft, TestKind::PRP, anchor_.exponent, {}); !why.empty()) {
    log("measure: the drift anchor %s is not built again here: %s. This session is unanchored and its rows are\n"
        "measure:   recorded as measured.\n",
        anchor_.text().c_str(), why.c_str());
    unanchor();
    return;
  }

  // Always the same shape of call, whatever the caller is timing its own configurations with: a reading taken over a
  // different number of iterations is not comparable with the baseline it is divided by.
  auto time = [&] {
    inAnchor_ = true;
    Call const c = runCall(fft, TestKind::PRP, anchor_.exponent, {}, BLOCKS_PER_CALL, ANCHOR_BLOCK_SIZE, false);
    inAnchor_ = false;
    sinceAnchor_ = Timer{};
    return c;
  };

  Call c = time();
  if (stopped_ || !c.measurement.ok()) {
    // The anchor is one fixed configuration, so what stopped it once will stop it every time; asking again at each
    // call would spend a build on it and say the same thing.
    log("measure: the drift anchor %s could not be timed, so this session is unanchored from here on and its rows\n"
        "measure:   carry the last ratio it gave (%.4f)\n",
        anchor_.text().c_str(), anchorState_.ratio);
    unanchor();
    return;
  }

  u32 ranCfg = db_.internCfg(c.ran);

  // The ratio only means anything against a reading of the same thing. An env holds one reference, so a session that
  // cannot reproduce the options the baseline was taken under leaves its rows as measured rather than re-basing them
  // on something else and putting two references in one env. Nothing is recorded either: this is not a reading of the
  // configuration the env is anchored to.
  if (baselineCfg_ && ranCfg != baselineCfg_) {
    log("measure: this env's anchor baseline was taken under -use %s and this session times it under %s, so the two\n"
        "measure:   are not comparable; this session is unanchored and its rows are recorded as measured.\n",
        configText(*db_.findCfg(baselineCfg_)).c_str(), configText(c.ran).c_str());
    unanchor();
    return;
  }

  bool const inherited = anchorState_.baseline > 0;
  DriftLevel level = anchorState_.observe(c.measurement.mean);
  bool const first = anchorState_.readings == 1;

  if (first && inherited) {
    log("measure: anchor %s: %.3f us/it, against a baseline of %.3f (%+.1f%%)\n", anchor_.text().c_str(),
        anchorState_.latest, anchorState_.baseline, (anchorState_.ratio - 1) * 100);
  } else if (first) {
    log("measure: anchor %s: %.3f us/it, which is the baseline for this env\n", anchor_.text().c_str(),
        anchorState_.latest);
  }

  if (level == DriftLevel::Alarm) {
    log("measure: the drift anchor %s reads %.3f us/it against a baseline of %.3f (%+.1f%%) -- pausing %.0f s and\n"
        "measure:   asking again\n",
        anchor_.text().c_str(), anchorState_.latest, anchorState_.baseline, (anchorState_.ratio - 1) * 100,
        ALARM_COOLDOWN_SEC);
    Timer::usleep(u32(ALARM_COOLDOWN_SEC * 1'000'000));

    c = time();
    if (!stopped_ && c.measurement.ok()) {
      level = anchorState_.observe(c.measurement.mean);
      ranCfg = db_.internCfg(c.ran);
    }

    if (level == DriftLevel::Alarm && !anchorState_.alarmed) {
      anchorState_.alarmed = true;
      (void)db_.add(AlarmRow{.sess = session_, .ts = now()});
      log("measure: it is still there, so this device has moved by more than the correction should be trusted to\n"
          "measure:   absorb. The ratio is still applied and the session is flagged.\n");
    }
  } else if (level == DriftLevel::Warn && !first) {
    log("measure: drift: the anchor %s has moved %+.1f%% since its baseline of %.3f us/it\n", anchor_.text().c_str(),
        (anchorState_.ratio - 1) * 100, anchorState_.baseline);
  }

  (void)db_.add(AnchorRow{.sess = session_,
                          .fft = anchor_.fft,
                          .exponent = anchor_.exponent,
                          .cfg = ranCfg,
                          .mean = anchorState_.latest,
                          .ratio = anchorState_.ratio,
                          .ts = now()});
}

Call Session::warmUp(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                     u32 blockSize) {
  return runCall(fft, kind, exponent, options, nBlocks, blockSize, false);
}

Call Session::run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                  u32 blockSize) {
  return runCall(fft, kind, exponent, options, nBlocks, blockSize, true);
}

Call Session::runCall(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                      u32 blockSize, bool record) {
  Call out;
  if (record) { keepAnchor(); }
  if (stopped_) {
    out.measurement.status = Status::Lost;
    return out;
  }

  u32 const cfg = db_.internCfg(options);

  Attempt const attempt{
    db_, session_, deviceLost_,
    TryRow{.sess = session_, .fft = fft.spec(), .kind = kind, .exponent = exponent, .cfg = cfg, .ts = now()},
    attemptText(fft, kind, exponent, options)};
  if (!attempt.declared()) {
    cannotDeclare(fft, options);
    return out;
  }

  if (Status const status = caught([&] { out = timeCall(shared_, fft, kind, exponent, options, nBlocks, blockSize); },
                                   fft, kind, exponent, options, "timing");
      status != Status::Ok) {
    out.measurement.status = status;
  }

  // Nothing is recorded for a stop, nor for a reading taken only to warm the device.
  if (stopped_ || !record) { return out; }

  out.measurement.drift = anchorState_.ratio;

  // Keyed on what the kernels were built with.
  u32 const ran = out.ran.empty() ? cfg : db_.internCfg(out.ran);
  (void)db_.add(RunRow{.sess = session_,
                       .fft = fft.spec(),
                       .kind = kind,
                       .exponent = exponent,
                       .regime = regimeOf(fft, exponent),
                       .cfg = ran,
                       .m = out.measurement});
  return out;
}

RoeCheck Session::checkRoe(const FFTConfig& fft, const UseConfig& options, u64 exponent) {
  RoeCheck out{.minZ = minSafeZ(fft.shape.fft_type), .exponent = exponent};
  if (stopped_) { return out; }

  u32 const cfg = db_.internCfg(options);
  Attempt const attempt{
    db_, session_, deviceLost_,
    TryRow{.sess = session_, .fft = fft.spec(), .kind = TestKind::PRP, .exponent = exponent, .cfg = cfg, .ts = now()},
    attemptText(fft, TestKind::PRP, exponent, options)};
  if (!attempt.declared()) {
    cannotDeclare(fft, options);
    out.status = Status::Lost;
    return out;
  }

  if (Status const status = caught([&] { out = roeCheck(shared_, fft, options, exponent); }, fft, TestKind::PRP,
                                   exponent, options, "accuracy check");
      status != Status::Ok) {
    // The reading was never taken, so nothing about it may be reported.
    out = RoeCheck{.minZ = minSafeZ(fft.shape.fft_type), .exponent = exponent, .status = status};
    return out;
  }

  if (out.applicable && !stopped_) {
    (void)db_.add(RoeRow{.sess = session_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = cfg,
                         .z = out.z,
                         .n = out.n,
                         .maxRoe = out.maxRoe,
                         .checkOk = out.checkOk,
                         .ts = now()});
  }
  return out;
}

bool Session::underAttempt(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                           const char* during, const std::function<void()>& work) {
  if (stopped_) { return false; }

  u32 const cfg = db_.internCfg(options);
  Attempt const attempt{
    db_, session_, deviceLost_,
    TryRow{.sess = session_, .fft = fft.spec(), .kind = kind, .exponent = exponent, .cfg = cfg, .ts = now()},
    attemptText(fft, kind, exponent, options)};
  if (!attempt.declared()) {
    cannotDeclare(fft, options);
    return false;
  }

  return caught(work, fft, kind, exponent, options, during) == Status::Ok;
}

MeasureArgs parseMeasureArgs(std::string_view text) {
  MeasureArgs out;

  auto number = [](std::string_view v, const char* what) -> u64 {
    if (v.empty() || v.find_first_not_of("0123456789") != std::string_view::npos) {
      throw "-measure: " + std::string{what} + " takes a number";
    }
    return strtoull(std::string{v}.c_str(), nullptr, 10);
  };

  bool first = true;
  for (size_t at = 0; at <= text.size();) {
    size_t const comma = text.find(',', at);
    std::string_view const token = text.substr(at, comma == std::string_view::npos ? comma : comma - at);
    at = comma == std::string_view::npos ? text.size() + 1 : comma + 1;

    size_t const eq = token.find('=');
    if (eq == std::string_view::npos) {
      // Only the leading token may be a bare FFT spec; a later one is a mistyped setting, not a second spec.
      if (!first || token.empty()) { throw "-measure: '" + std::string{token} + "' is not a <key>=<value> setting"; }
      out.fft = std::string{token};
      first = false;
      continue;
    }
    first = false;

    std::string_view const key = token.substr(0, eq);
    std::string_view const val = token.substr(eq + 1);

    if (key == "fft") {
      out.fft = std::string{val};
    } else if (key == "anchor") {
      out.anchorFft = std::string{val};
    } else if (key == "exp") {
      out.exponent = number(val, "exp=");
    } else if (key == "n") {
      out.calls = u32(number(val, "n="));
    } else if (key == "blocks") {
      out.blocks = u32(number(val, "blocks="));
    } else if (key == "block") {
      out.blockSize = u32(number(val, "block="));
    } else if (key == "roe") {
      out.roe = number(val, "roe=") != 0;
    } else if (key == "drain") {
      out.drain = number(val, "drain=") != 0;
    } else if (key == "drift") {
      out.drift = number(val, "drift=") != 0;
    } else {
      throw "-measure: '" + std::string{key} +
        "=' is not understood. Accepted: fft=<spec>, exp=<E>, n=<calls>, blocks=<per call>, block=<iterations>,"
        " anchor=<spec>, roe=0|1, drain=0|1, drift=0|1";
    }
  }

  if (out.fft.empty()) { throw std::string{"-measure needs an FFT spec"}; }
  // A verdict is about how far independent rows move, so it takes two whole rows; fewer calls than that can time a
  // configuration but cannot say anything about the error bar, which is what this command is for.
  if (out.calls < 2 * MIN_CALLS) {
    throw "-measure: n= must be at least " + to_string(2 * MIN_CALLS) + ": a verdict needs two rows of " +
      to_string(MIN_CALLS) + " calls";
  }
  if (out.blocks < 2) { throw std::string{"-measure: blocks= must be at least 2"}; }
  if (out.blockSize == 1) { throw std::string{"-measure: block= must be at least 2 iterations"}; }

  // The scheduled anchor and the alternating one correct for the same thing, and applying both describes neither.
  if (!out.anchorFft.empty()) { out.drift = false; }

  // Validated here so that a bad spec is a usage error rather than a failure once a device is open.
  (void)FFTConfig{out.fft};
  if (!out.anchorFft.empty()) { (void)FFTConfig{out.anchorFft}; }

  return out;
}

namespace {

void logBlocks(const char* what, const Call& c) {
  string blocks;
  for (double const us : c.usPerIt) {
    char buf[32];
    snprintf(buf, sizeof(buf), " %.3f", us);
    blocks += buf;
  }
  log("measure: %s:%s us/it | mean %.3f sd %.3f (%.3f%%) dropped %u%s, %s %016" PRIx64 "\n", what, blocks.c_str(),
      c.measurement.mean, c.measurement.stddev,
      c.measurement.mean > 0 ? c.measurement.stddev / c.measurement.mean * 100 : 0, c.dropped,
      c.declined ? " (declined)" : "", c.checkOk ? "OK" : "EE", c.res64);
}

void logSpread(const char* what, const Spread& s) {
  auto pct = [&](double v) { return s.mean > 0 ? v / s.mean * 100 : 0; };
  log("measure:   %s: observed %.3f us (%.3f%%) against predicted %.3f us (%.3f%%) over %u -- ratio %.2f, %.2f with\n"
      "measure:     any trend removed; first to last %+.3f us (%+.3f%%)\n",
      what, s.observed, pct(s.observed), s.predicted, pct(s.predicted), s.n, s.ratio(), s.detrended(), s.trend,
      pct(s.trend));
}

void logNoise(const char* what, const NoiseReport& r) {
  log("measure: %s, mean %.3f us/it:\n", what, r.row.mean > 0 ? r.row.mean : r.call.mean);
  logSpread("between calls, against the blocks inside one", r.call);
  if (r.row.n >= 2) {
    char label[80];
    snprintf(label, sizeof(label), "between rows of %u calls, against the bar such a row declares", r.callsPerRow);
    logSpread(label, r.row);
  }
  log("measure:   verdict: %s\n", toString(r.verdict));
}

}  // namespace

MeasureOutcome runMeasure(GpuCommon shared, const MeasureArgs& want) {
  static const Primes primes;

  FFTConfig const fft{want.fft};
  u64 exponent = want.exponent ? want.exponent : shared.args->prpExp;
  if (!exponent) { exponent = primes.prevPrime(fft.maxExp()); }
  if (!primes.isPrime(exponent)) { log("measure: warning: %" PRIu64 " is not prime\n", exponent); }

  u32 const blockSize = want.blockSize ? want.blockSize : shared.args->blockSize;
  u32 const nBlocks = want.blocks;

  std::optional<FFTConfig> anchor;
  u64 anchorExp = exponent;
  if (!want.anchorFft.empty()) {
    anchor.emplace(want.anchorFft);
    if (anchor->spec() == fft.spec()) {
      // An anchor that is the configuration under test divides out exactly what this command measures.
      log("measure: the anchor must be a different configuration from the one being measured\n");
      return MeasureOutcome::Failed;
    }
    if (exponent > anchor->maxExp()) {
      anchorExp = primes.prevPrime(anchor->maxExp());
      log("measure: the anchor %s cannot hold E=%" PRIu64 "; timing it at %" PRIu64 " instead\n",
          anchor->spec().c_str(), exponent, anchorExp);
    }
  }

  std::string const anchoredOn = anchor ? ", anchored on " + anchor->spec() : std::string{};
  log("measure: %s at exponent %" PRIu64 " (%.2f bpw), %u calls of %u blocks of %u%s\n", fft.spec().c_str(), exponent,
      double(exponent) / fft.shape.size(), want.calls, nBlocks, blockSize, anchoredOn.c_str());

  fs::path const dbPath = TuneDB::DEFAULT_NAME;
  TuneDB db;
  // Before the load, and held for the rest of the run: every id this writes is allocated from what it read.
  if (!db.lockForWriting(dbPath)) { return MeasureOutcome::Failed; }
  if (!db.load(dbPath)) { return MeasureOutcome::Failed; }
  db.attach(dbPath);

  Session session{shared, db, detectEnv(*shared.context, *shared.args)};
  if (!session.begin(want.drift ? exponent : 0)) {
    log("measure: '%s' would not take a session\n", dbPath.string().c_str());
    return MeasureOutcome::Failed;
  }
  if (u32 const gen = restart::generation()) { log("measure: generation %u\n", gen); }

  UseConfig const options = resolveConfig(*shared.args, fft, TestKind::PRP);
  UseConfig const anchorOptions = anchor ? resolveConfig(*shared.args, *anchor, TestKind::PRP) : UseConfig{};

  std::vector<std::string> varied;
  for (const auto& [key, value] : shared.args->flags) { varied.push_back(key); }
  session.varying(varied);

  auto skip = [&](const FFTConfig& what, const UseConfig& with, u64 at) {
    std::string const why = session.held(what, TestKind::PRP, at, with);
    if (why.empty()) { return false; }
    log("measure: skipping %s -use %s: %s.\n"
        "measure:   It will not be built again on this device.\n",
        what.spec().c_str(), configText(with).c_str(), why.c_str());
    return true;
  };

  if (skip(fft, options, exponent) || (anchor && skip(*anchor, anchorOptions, anchorExp))) {
    session.end();
    return MeasureOutcome::Failed;
  }

  bool ok = true;
  double warmUpOverhead = 0;

  // Only the configuration under test: the anchor is a different shape, and averaging the two would describe neither.
  std::vector<double> buildSecs;
  std::vector<double> timedSecs;

  for (u32 w = 0; w < SESSION_WARM_CALLS && !session.stopped(); ++w) {
    Call const c = session.warmUp(fft, TestKind::PRP, exponent, options, nBlocks, blockSize);
    if (session.stopped() || !c.measurement.ok()) {
      ok = false;
      break;
    }
    // A cold kernel cache, if this process has one, is paid here, which is what makes it visible on its own.
    warmUpOverhead = c.buildSec + c.timedSec - double(nBlocks) * blockSize * c.measurement.mean * 1e-6;
    logBlocks("warm-up (discarded)", c);
  }

  std::vector<CallSummary> raw;
  std::vector<CallSummary> corrected;
  double firstAnchor = 0;
  double drift = 1;

  for (u32 call = 0; call < want.calls && !session.stopped(); ++call) {
    if (anchor) {
      Call const a = session.run(*anchor, TestKind::PRP, anchorExp, anchorOptions, nBlocks, blockSize);
      if (session.stopped() || !a.measurement.ok()) {
        ok = false;
        break;
      }
      if (!firstAnchor) { firstAnchor = a.measurement.mean; }
      drift = a.measurement.mean / firstAnchor;
      log("measure: anchor %u: %.3f us/it, drift %.4f\n", call, a.measurement.mean, drift);
    }

    Call const c = session.run(fft, TestKind::PRP, exponent, options, nBlocks, blockSize);
    if (session.stopped() || !c.measurement.ok()) {
      ok = false;
      break;
    }

    char label[32];
    snprintf(label, sizeof(label), "call %u", call);
    logBlocks(label, c);
    raw.push_back(c.summary());
    corrected.push_back(c.summary(drift));
    buildSecs.push_back(c.buildSec);
    timedSecs.push_back(c.timedSec);
    ok = ok && c.checkOk;
  }

  NoiseReport const rawReport = noiseOf(raw);
  NoiseReport const finalReport = anchor ? noiseOf(corrected) : rawReport;

  if (!raw.empty()) {
    logNoise("as measured", rawReport);
    if (anchor) { logNoise("with the anchor correction applied", finalReport); }
  }

  // A stop is not a verdict about anything.
  if (session.stopped()) {
    // Nothing to conclude.
  } else if (finalReport.verdict == NoiseVerdict::TooFew) {
    // Never the device's fault: `n` cannot be set low enough to reach this, so a run that lands here was cut short.
    log("measure: only %zu of the %u calls asked for completed, which is too few to judge the error bar a row\n"
        "measure:   declares -- that takes two rows of %u. Nothing here says the device is bad, only that it was\n"
        "measure:   not measured.\n",
        raw.size(), want.calls, MIN_CALLS);
    ok = false;
  } else if (!finalReport.trustworthy()) {
    log("measure: the error bar a row declares does not describe how far its readings move, so a race here would\n"
        "measure:   eliminate candidates on differences that are not real. Nothing measured on this device can be\n"
        "measure:   trusted to rank configurations until that is dealt with.\n");
    ok = false;
  }

  // What a call spends on something other than the samples it yields: an item is one call, so this is the overhead the
  // queue pays per item.  Gpu::make loads kernels and allocates buffers lazily, so most of the setup falls inside the
  // first block rather than in the constructor; what separates the two here is the iterations the samples account for.
  if (!buildSecs.empty()) {
    Stats const build = statsOf(buildSecs);
    Stats const timed = statsOf(timedSecs);
    double const total = build.mean + timed.mean;
    double const sampled = double(nBlocks) * blockSize * rawReport.call.mean * 1e-6;
    double const overhead = std::max(0.0, total - sampled);
    log("measure: one call: %.2f s, of which %.2f s is the blocks it timed and %.2f s (%.1f%%) is everything else --\n"
        "measure:   the Gpu, its buffers, its warm-up block and the kernel loads the first block triggers (%.2f s of\n"
        "measure:   that is the constructor itself)\n",
        total, sampled, overhead, total > 0 ? overhead / total * 100 : 0, build.mean);
    // The warm-up call pays whatever the kernel cache could not answer, so the gap between it and the rest is what a
    // compile costs on this machine.
    if (warmUpOverhead > overhead + 0.05) {
      log("measure:   the warm-up call spent %.2f s there instead, so a compile costs about %.2f s here\n",
          warmUpOverhead, warmUpOverhead - overhead);
    }
  }

  // What draining at every block boundary costs.
  if (want.drain && !session.stopped()) {
    auto time = [&](u32 blocks, u32 size) {
      auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(options), false, TestKind::PRP);
      return statsOf(gpu->timeIters(blocks, size, 5000 / size).usPerIt).mean;
    };

    double drained = 0;
    double whole = 0;
    if (session.underAttempt(fft, TestKind::PRP, exponent, options, "drain control", [&] {
          drained = time(5, 1000);
          whole = time(1, 5000);
        })) {
      log("measure: 5x1000 per-block %.3f vs 1x5000 %.3f us/it, drain costs at most %+.3f%%\n", drained, whole,
          (drained / whole - 1) * 100);
    } else {
      ok = false;
    }
  }

  if (want.roe && !session.stopped()) {
    RoeCheck const roe = session.checkRoe(fft, options, exponent);
    if (roe.status != Status::Ok) {
      log("measure: ROE: could not be checked (%s)\n", toString(roe.status));
      ok = false;
    } else if (!roe.applicable) {
      log("measure: ROE: not applicable (exact arithmetic)\n");
    } else {
      log("measure: ROE at %" PRIu64 ": z %.4g (floor %.0f) n %u max %.4g, %s%s\n", roe.exponent, roe.z, roe.minZ,
          roe.n, roe.maxRoe, roe.checkOk ? "OK" : "EE", roe.conclusive() ? "" : ", inconclusive");
      ok = ok && roe.passed();
    }
  }

  if (want.drift && session.anchor().valid() && !raw.empty()) {
    log("measure: the rows this session recorded carry a drift of %.4f against %s\n", session.drift(),
        session.anchor().text().c_str());
  }

  if (session.deviceLost()) { return MeasureOutcome::DeviceLost; }
  session.end();
  if (session.cannotRecord()) { return MeasureOutcome::Failed; }
  if (session.stopped()) { return MeasureOutcome::Ok; }
  return ok ? MeasureOutcome::Ok : MeasureOutcome::Failed;
}

}  // namespace tune
