// Copyright (C) Jason Lynch

#include "Measure.h"

#include "Failures.h"

#include "clwrap.h"
#include "Context.h"
#include "Gpu.h"
#include "GpuFault.h"
#include "LLCheck.h"
#include "log.h"
#include "Restart.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <ctime>
#include <thread>
#include <utility>

namespace tune {

namespace {

u64 now() { return u64(std::time(nullptr)); }

std::vector<KeyVal> asExtraConf(const Env& env, const FFTConfig& fft, const UseConfig& options) {
  UseConfig const fitted = withHostFit(env, fft, options);
  return {fitted.begin(), fitted.end()};
}

std::string attemptText(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options) {
  return "-fft " + fft.spec() + " -use " + configText(options) + " (" + toString(kind) +
    ", E=" + std::to_string(exponent) + ")";
}

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

// The attempt record around one build: written before it, answered by whatever the session records next, and left
// open only where the process does not survive.
class Attempt {
public:
  Attempt(TuneDB& db, const TryRow& row, std::string what) {
    declared_ = db.add(row);
    outer_ = gpufault::attempting(std::move(what));
  }

  Attempt(const Attempt&) = delete;
  Attempt& operator=(const Attempt&) = delete;

  ~Attempt() { (void)gpufault::attempting(std::move(outer_)); }

  [[nodiscard]] bool declared() const { return declared_; }

private:
  bool declared_ = false;
  std::string outer_;
};

}  // namespace

Call summarize(const IterSamples& samples) {
  Call out{.core = coreStats(samples.usPerIt),
           .usPerIt = samples.usPerIt,
           .res64 = samples.res64,
           .checkOk = samples.checkOk,
           .iters = samples.iters};
  out.timedFrom = samples.timedFrom;
  out.timedTo = samples.timedTo;
  if (!samples.checkOk) { out.status = Status::Err; }
  return out;
}

Takeover takeOver(const Args& shared) {
  Takeover out{.args = shared};
  std::erase_if(out.args.flags, [&](const auto& kv) {
    const Option* const option = findOption(kv.first);
    bool const tunable = !option || option->kind == Kind::Tunable;
    if (tunable) { out.setAside.insert(kv.first); }
    return tunable;
  });
  for (const auto& [fft, uses] : out.args.perFftConfig) {
    for (const auto& [key, value] : uses) { out.setAside.insert(key); }
  }
  out.args.perFftConfig.clear();
  out.carry = out.args.carry != CARRY_AUTO;
  out.args.carry = CARRY_AUTO;
  return out;
}

Call timeCall(GpuCommon shared, const Env& env, const FFTConfig& fft, TestKind kind, u64 exponent,
              const UseConfig& options, u32 nBlocks, u32 blockSize, double heatSec, std::unique_ptr<Gpu>* keep) {
  Timer t;
  auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(env, fft, options), false);
  double const buildSec = t.reset();

  Call out = summarize(kind == TestKind::LL ? gpu->timeItersLL(nBlocks, blockSize, CALL_WARMUP_BLOCKS, heatSec)
                                            : gpu->timeIters(nBlocks, blockSize, CALL_WARMUP_BLOCKS, heatSec));
  out.timedSec = t.at();
  out.buildSec = buildSec;
  out.ran = gpu->args.flags;
  out.loaded = gpu->loadedKernels();
  if (std::optional<OccupancyLimits> const limits = deviceLimits(shared.context->deviceId())) {
    for (const Gpu::RegTunable& k : gpu->regTunables(false)) {
      out.kernels.push_back({.key = k.key, .resources = k.res, .plan = registerPlan(k.res, *limits, k.defaultCap)});
    }
  }
  if (keep) { *keep = std::move(gpu); }
  return out;
}

// The last call's Gpu, squaring on a thread of its own while the session is idle -- choosing what to measure next,
// recording what it read -- so the card does not cool between calls.  Idle stood at a quarter to a half of a tuning run's
// wall time, 5-25 s between calls, and a card that cools between calls reads them faster than production, which never
// lets it: on a 4070 SUPER, calls with the card kept loaded read 2.5% closer to what it sustained, on an A4000 2.6%.
// The squaring stops before anything else is built, so two Gpus are never held at once and a build or a fault is the
// next configuration's alone.
//
// What it left idle was the next configuration's first run, where its kernels compiled: 2-3 s for one not read before.
// Kept squaring through that too, the card idled in 1.5-5% of a run rather than 34-40%, but a call's build took 9-16 s
// rather than 3, the other Gpu holding up every step of it, and a run made half the calls.  So the kernels compile before
// it stops instead, with no Gpu and nothing on the device (Bench::compileAhead), and the first run only loads them.
class Heater {
public:
  Heater(std::unique_ptr<Gpu> gpu, TestKind kind, u32 blocks, u32 blockSize, const std::string& what)
    : gpu_{std::move(gpu)}, what_{"the load between calls, on " + what} {
    // Named as an attempt is, so a fault while it alone runs restarts the process rather than ending it (GpuFault).
    (void)gpufault::attempting(what_);
    thread_ = std::thread{[this, kind, blocks, blockSize] {
      try {
        while (!stop_) {
          (void)(kind == TestKind::LL ? gpu_->timeItersLL(blocks, blockSize, CALL_WARMUP_BLOCKS)
                                      : gpu_->timeIters(blocks, blockSize, CALL_WARMUP_BLOCKS));
        }
      } catch (const char* mes) {
        failed_ = mes;
      } catch (const std::string& mes) {
        failed_ = mes;
      } catch (const std::exception& e) {
        failed_ = e.what();
      } catch (...) { failed_ = "an unknown error"; }
    }};
  }

  ~Heater() { (void)stop(); }

  Heater(const Heater&) = delete;
  Heater& operator=(const Heater&) = delete;

  // Stops the squaring, waiting for the chunk under way, and lets the Gpu go.  Why it had stopped already, if it had.
  std::optional<std::string> stop() {
    if (!thread_.joinable()) { return {}; }
    stop_ = true;
    thread_.join();
    gpu_.reset();
    (void)gpufault::attempting({});
    return failed_;
  }

private:
  std::unique_ptr<Gpu> gpu_;
  std::string what_;
  std::atomic<bool> stop_{false};
  std::optional<std::string> failed_;  // written by the thread, read once it is joined
  std::thread thread_;
};

namespace {

// The squaring a heater does between looks at whether to stop, which is how long it can hold up the next build.
constexpr double HEATER_CHUNK_SEC = 0.25;

// Calls that run the same kernels, near enough (Bench::compileAhead): of one kind, on FFTs of one type.  Options choose among
// variants of a few (the tail's, the carry's), and every variant a call of the two has run is compiled ahead.
std::string kernelsRunBy(const FFTConfig& fft, TestKind kind) {
  return std::string{toString(kind)} + ' ' + std::to_string(int(fft.shape.fft_type));
}

std::string tallyText(const KernelCompiler::Tally& t) {
  char buf[256];
  snprintf(buf, sizeof(buf),
           "%u compiled ahead in %.1f s; at the loads %u compiled in %.1f s, %u loaded from memory in %.2f s, %u register "
           "limits applied in %.1f s",
           t.ahead, t.aheadSec, t.compiled, t.compiledSec, t.fromCache, t.fromCacheSec, t.limited, t.limitedSec);
  return buf;
}

}  // namespace

Call checkedAgainst(u64 reference, Call first, const std::function<Call()>& again) {
  if (!first.ok() || first.res64 == reference) { return first; }
  Call second = again();
  if (second.ok() && second.res64 != reference) { second.status = Status::Err; }
  return second;
}

Call checkedTwice(Call first, const std::function<Call()>& again) {
  if (first.status != Status::Err) { return first; }
  return again();
}

RoeCheck roeCheck(GpuCommon shared, const Env& env, const FFTConfig& fft, const UseConfig& options, u64 exponent) {
  RoeCheck out{.exponent = exponent};
  if (exactArithmetic(fft)) { return out; }
  out.applicable = true;

  auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(env, fft, options), false);
  auto [checkOk, res, roeSq, roeMul] = gpu->measureROE(false);
  out.checkOk = checkOk;
  out.z = roeSq.z();
  out.n = roeSq.N;
  out.maxRoe = roeSq.max;
  out.ran = gpu->args.flags;
  return out;
}

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

Bench::Bench(GpuCommon shared, TuneDB& db, const Env& env) :
  shared_{shared},
  args_{shared.args ? takeOver(*shared.args).args : Args{true}},
  db_{db},
  env_{env},
  blockSize_{args_.blockSize} {
  shared_.args = &args_;
  if (shared_.context) {
    std::string why;
    telemetry_ = openTelemetryFor(shared_.context->deviceId(), why);
    log("measure: card readings: %s\n", telemetry_ ? telemetry_->source().c_str() : ("none, " + why).c_str());
    // A call's timed blocks last about a second, so the card is read often enough to bracket them closely.
    sampler_ = std::make_unique<Sampler>(telemetry_.get(), SAMPLE_PERIOD, SAMPLE_KEEP);
  }
}

bool Bench::begin(u64 probe, const std::string& settings, std::string machine, std::string tag) {
  envId_ = db_.internEnv(dbEnvOf(env_, std::move(machine), std::move(tag)));
  if (!envId_) { return false; }

  if (probe) {
    // The env's first session fixes the anchor for the env's life: readings are comparable only against readings of
    // one configuration at one exponent.
    if (auto const pinned = parseAnchorSpec(db_.envAnchor(envId_))) {
      anchor_ = *pinned;
    } else {
      race_ = anchorCandidates(probe);
      if (race_.empty()) {
        log("measure: nothing is eligible at %" PRIu64 " to anchor on, so this session's readings are recorded as "
            "measured\n",
            probe);
      }
    }
    if (const AnchorRow* const baseline = db_.envBaseline(envId_)) {
      anchorState_.baseline = baseline->mean;
      baselineCfg_ = baseline->cfg;
    }
  }

  tallyAtBegin_ = KernelCompiler::tally();
  session_ = db_.beginSession(envId_, anchor_.valid() ? anchor_.text() : "", restart::generation(), settings);
  return session_ != 0;
}

void Bench::end() {
  coolDown();
  if (session_ && (aheadCalls_ || !kernelsRun_.empty())) {
    log("measure: kernels this session: %s; %u calls had theirs compiled ahead, %u kernels\n",
        tallyText(KernelCompiler::tally() - tallyAtBegin_).c_str(), aheadCalls_, aheadKernels_);
    aheadCalls_ = 0;
    kernelsRun_.clear();
  }
  if (session_ && !deviceLost_ && db_.attemptOpen(session_)) { (void)db_.add(DoneRow{.sess = session_, .ts = now()}); }
}

Bench::~Bench() { end(); }

void Bench::coolDown() {
  if (!heater_) { return; }
  std::optional<std::string> const failed = heater_->stop();
  heater_.reset();
  if (!failed) { return; }
  Failure const f = classify(*failed);
  if (f.stop && !f.fatal) {
    stopped_ = true;  // a stop asked for while it squared
    return;
  }
  log("measure: the card's load between calls stopped: %s\n", failed->c_str());
  if (f.fatal || !deviceUsable()) {
    stopped_ = true;
    deviceLost_ = true;
    markContextLost("it did not survive the load between calls");
    log("measure: the device is no longer usable after the load between calls -- stopping.  Nothing is recorded "
        "against what would have been built next.\n");
  }
}

std::string Bench::deviceName() const {
  const DbEnv* const env = db_.findEnv(envId_);
  return env ? env->gpu : std::string{};
}

std::string Bench::held(const FFTConfig& fft, const UseConfig& options) const {
  u32 const cfg = db_.findCfgId(builtAs(env_, fft, options));
  if (cfg && db_.diedOn(envId_, fft.spec(), cfg)) {
    return "an earlier process was holding it when the device went away (" + std::string{FAILURES_DIR} + '/' +
           FAILURES_LOG + ")";
  }
  return {};
}

bool Bench::deviceUsable() {
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

void Bench::lost(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, const std::string& what,
                 const char* during) {
  stopped_ = true;
  if (deviceLost_) { return; }
  deviceLost_ = true;
  markContextLost("it did not survive what was being measured");
  if (const TryRow* const t = db_.openAttempt(session_)) {
    (void)recordFailure({.kind = FailureKind::DeviceLost,
                         .fft = t->fft,
                         .test = t->kind,
                         .exponent = t->exponent,
                         .options = db_.findCfg(t->cfg) ? configText(*db_.findCfg(t->cfg)) : configText(options),
                         .device = deviceName(),
                         .detail = what + " (during the " + during + ")",
                         .at = t->ts,
                         .id = failureId(*t)});
  }
  log("measure: the device is no longer usable (%s) -- stopping.  It was lost during the %s of\n"
      "measure:   -fft %s -use %s (%s, E=%" PRIu64 ")\n"
      "measure:   which is recorded and will not be built again.\n",
      what.c_str(), during, fft.spec().c_str(), configText(options).c_str(), toString(kind), exponent);
}

Status Bench::failed(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
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

  log("measure: %s during the %s of %s -use %s: %s\n", toString(f.status), during, fft.spec().c_str(),
      configText(options).c_str(), f.what.c_str());
  lastFailure_ = f.what;

  // A kernel that faults can leave the context unusable while reporting something ordinary.  Asked now, the loss is
  // pinned on what caused it rather than on whatever the next call builds.
  if (!deviceUsable()) {
    lost(fft, kind, exponent, options,
         f.status == Status::NoCompile ? "the context did not survive the failed build"
                                       : "the context did not survive it: " + f.what,
         during);
    return Status::Lost;
  }
  return f.status;
}

Status Bench::caught(const std::function<void()>& work, const FFTConfig& fft, TestKind kind, u64 exponent,
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

void Bench::cannotDeclare(const FFTConfig& fft, const UseConfig& options) {
  cannotRecord_ = true;
  stopped_ = true;
  log(
    "measure: the attempt could not be recorded, so %s -use %s was not built: without that record a fault here would\n"
    "measure:   leave nothing to stop the next process repeating it.\n",
    fft.spec().c_str(), configText(options).c_str());
}

bool Bench::anchorDue() const {
  if ((!anchor_.valid() && race_.empty()) || inAnchor_ || stopped_) { return false; }
  return !anchorState_.readings || sinceAnchor_.at() >= ANCHOR_EVERY_SEC;
}

void Bench::raceAnchor() {
  if (race_.empty() || inAnchor_ || stopped_) { return; }
  std::vector<AnchorSpec> const candidates = std::exchange(race_, {});

  // What the env already has of each at the built-in defaults: a race cut short picks up where it was.
  auto recorded = [&](const AnchorSpec& c) {
    double best = 0;
    for (const CallRow& row : db_.calls()) {
      if (row.fft != c.fft || row.kind != TestKind::PRP || row.exponent != c.exponent || !row.ok() ||
          db_.envOf(row.sess) != envId_) {
        continue;
      }
      const UseConfig* const opts = db_.findCfg(row.cfg);
      if (opts && atBuiltInDefaults(env_, FFTConfig{c.fft}, *opts) && (!best || row.cost() < best)) {
        best = row.cost();
      }
    }
    return best;
  };

  // The race's own calls are recorded, and a recorded call would otherwise ask for the anchor it is choosing.
  inAnchor_ = true;
  std::vector<AnchorReading> readings;
  for (const AnchorSpec& c : candidates) {
    FFTConfig const fft{c.fft};
    if (!held(fft, {}).empty()) { continue; }

    double us = recorded(c);
    if (!us) {
      Call const call = runCall(fft, TestKind::PRP, c.exponent, {}, BLOCKS_PER_CALL, blockSize_, true);
      if (stopped_) { break; }
      if (call.ok()) { us = call.mean(); }
    }
    if (us > 0) {
      readings.push_back({.anchor = c, .us = us});
      log("measure: anchor race at %" PRIu64 ": %s %.3f us/it\n", c.exponent, c.fft.c_str(), us);
    }
  }
  inAnchor_ = false;

  // Nothing is pinned, so the next session races again from what this one recorded.
  if (stopped_) { return; }

  if (auto const winner = raceWinner(readings)) {
    anchor_ = *winner;
    log("measure: anchoring this env on %s, the cheapest of the %zu raced\n", anchor_.text().c_str(), readings.size());
  } else {
    log("measure: no anchor candidate gave a reading, so this session's readings are recorded as measured\n");
  }
}

void Bench::keepAnchor() {
  if (!anchorDue()) { return; }
  raceAnchor();
  if (!anchorDue()) { return; }

  FFTConfig const fft{anchor_.fft};

  // Held back like anything else: otherwise every generation the restart limit allows would rebuild an anchor that
  // takes the device down.
  if (std::string const why = held(fft, {}); !why.empty()) {
    log("measure: the drift anchor %s is not built again here: %s.  This session is unanchored.\n",
        anchor_.text().c_str(), why.c_str());
    unanchor();
    return;
  }

  // Always the same shape of call: a reading over a different amount of work is not comparable with the baseline.
  auto time = [&] {
    inAnchor_ = true;
    Call const c = runCall(fft, TestKind::PRP, anchor_.exponent, {}, BLOCKS_PER_CALL, ANCHOR_BLOCK_SIZE, false);
    inAnchor_ = false;
    sinceAnchor_ = Timer{};
    return c;
  };

  Call c = time();
  if (stopped_ || !c.ok()) {
    log(
      "measure: the drift anchor %s could not be timed, so this session is unanchored from here on and keeps the last\n"
      "measure:   ratio it gave (%.4f)\n",
      anchor_.text().c_str(), anchorState_.ratio);
    unanchor();
    return;
  }

  u32 ranCfg = db_.internCfg(builtAs(env_, fft, c.ran));

  // An env holds one reference: a session that cannot reproduce the options its baseline was taken under leaves its
  // readings as measured rather than putting a second reference in the env.
  if (baselineCfg_ && ranCfg != baselineCfg_) {
    log("measure: this env's anchor baseline was taken under -use %s and this session builds it as %s, so the two are\n"
        "measure:   not comparable; this session is unanchored.\n",
        configText(*db_.findCfg(baselineCfg_)).c_str(), configText(builtAs(env_, fft, c.ran)).c_str());
    unanchor();
    return;
  }

  bool const inherited = anchorState_.baseline > 0;
  DriftLevel level = anchorState_.observe(c.mean());
  bool const first = anchorState_.readings == 1;

  if (first && inherited) {
    log("measure: anchor %s: %.3f us/it against a baseline of %.3f (%+.1f%%)\n", anchor_.text().c_str(),
        anchorState_.latest, anchorState_.baseline, (anchorState_.ratio - 1) * 100);
  } else if (first) {
    log("measure: anchor %s: %.3f us/it, the baseline for this env\n", anchor_.text().c_str(), anchorState_.latest);
  }

  if (level == DriftLevel::Alarm) {
    log("measure: the drift anchor reads %+.1f%% from its baseline -- pausing %.0f s and asking again\n",
        (anchorState_.ratio - 1) * 100, ALARM_COOLDOWN_SEC);
    Timer::usleep(u32(ALARM_COOLDOWN_SEC * 1'000'000));
    c = time();
    if (!stopped_ && c.ok()) { level = anchorState_.observe(c.mean()); }
    if (level == DriftLevel::Alarm && !anchorState_.alarmed) {
      anchorState_.alarmed = true;
      log("measure: it is still there, so this device has moved by more than the correction should be trusted to\n"
          "measure:   absorb.  The ratio is still applied.\n");
    }
  } else if (level == DriftLevel::Warn && !first) {
    log("measure: drift: the anchor has moved %+.1f%% since its baseline\n", (anchorState_.ratio - 1) * 100);
  }

  (void)db_.add(AnchorRow{.sess = session_,
                          .fft = anchor_.fft,
                          .exponent = anchor_.exponent,
                          .cfg = ranCfg,
                          .mean = anchorState_.latest,
                          .ratio = anchorState_.ratio,
                          .ts = now()});
}

Call Bench::call(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks) {
  return runCall(fft, kind, exponent, options, nBlocks ? nBlocks : blocks_, blockSize_, true, heatSec_);
}

Call Bench::runCall(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                    u32 blockSize, bool record, double heatSec) {
  Call out;
  if (record) { keepAnchor(); }
  if (stopped_) {
    out.status = Status::Lost;
    return out;
  }

  if (std::string const why = held(fft, options); !why.empty()) {
    log("measure: %s -use %s is not built again here: %s\n", fft.spec().c_str(), configText(options).c_str(),
        why.c_str());
    out.status = Status::Unsupported;
    return out;
  }

  std::optional<u64> reference;
  if (record && kind == TestKind::LL) {
    reference = llReference(fft, exponent, nBlocks, blockSize);
    if (stopped_ || !reference) {
      // A missing reference says nothing of the configuration, and the `ref` records already say it.
      out.status = stopped_ ? Status::Lost : Status::Unsupported;
      return out;
    }
  }

  // The first Gpu of a process is slow everywhere it has been measured; a session that does not absorb that charges it
  // to whatever it happens to run first.
  if (!warmed_) {
    warmed_ = true;
    for (u32 w = 0; w < SESSION_WARM_CALLS && !stopped_; ++w) {
      (void)runCall(fft, kind, exponent, options, nBlocks, blockSize, false, heatSec);
    }
    if (stopped_) {
      out.status = Status::Lost;
      return out;
    }
  }

  out = attempt(shared_, fft, kind, exponent, options, nBlocks, blockSize, heatSec);
  if (stopped_ || !record) { return out; }

  auto again = [&] { return attempt(shared_, fft, kind, exponent, options, nBlocks, blockSize, heatSec); };
  std::string wrongBy = "its Gerbicz check failed twice in a row";
  if (out.status == Status::Err) {
    log("measure: %s -use %s failed its check at %" PRIu64 "; reading it again, since one failure decides nothing\n",
        fft.spec().c_str(), configText(options).c_str(), exponent);
    out = checkedTwice(std::move(out), again);
    if (stopped_) { return out; }
    if (out.status == Status::Err) {
      log("measure: it failed again: it computes wrongly, and is recorded as an error\n");
    }
  }

  if (reference && out.ok() && out.res64 != *reference) {
    log("measure: %s -use %s read LL residue %016" PRIx64 " after %" PRIu64
        " iterations where the reference is %016" PRIx64 "; reading it again, since one reading decides nothing\n",
        fft.spec().c_str(), configText(options).c_str(), out.res64, out.iters, *reference);
    out = checkedAgainst(*reference, std::move(out), again);
    if (stopped_) { return out; }
    if (out.status == Status::Err) {
      log("measure: it read %016" PRIx64 " again: it computes LL wrongly, and is recorded as an error\n", out.res64);
      char buf[160];
      snprintf(buf, sizeof(buf), "LL residue %016" PRIx64 " twice after %" PRIu64 " iterations, where the reference is %016" PRIx64,
               out.res64, out.iters, *reference);
      wrongBy = buf;
    }
  }

  // What the card drew over the timed blocks, once it has been read past their end.
  if (out.ok() && sampler_ && sampler_->waitPast(sampler_->at(out.timedTo), 5 * SAMPLE_PERIOD)) {
    out.power = powerOver(*sampler_, out.timedFrom, out.timedTo);
  }

  // Keyed on what the kernels were built with.
  u32 const cfg = db_.internCfg(builtAs(env_, fft, out.ran.empty() ? options : out.ran));
  u64 const ts = now();
  CallRow const row{.sess = session_,
                    .fft = fft.spec(),
                    .kind = kind,
                    .exponent = exponent,
                    .cfg = cfg,
                    .status = out.status,
                    .mean = out.core.stats.mean,
                    .sd = out.core.stats.sd,
                    .blocks = out.core.stats.n,
                    .dropped = out.core.dropped,
                    .drift = anchorState_.ratio,
                    .buildSec = out.buildSec,
                    .timedSec = out.timedSec,
                    .ts = ts};
  if (db_.add(row) && (out.status == Status::Err || out.status == Status::NoCompile)) {
    bool const wrong = out.status == Status::Err;
    (void)recordFailure({.kind = wrong ? FailureKind::Wrong : FailureKind::NoBuild,
                         .fft = row.fft,
                         .test = kind,
                         .exponent = exponent,
                         .options = configText(out.ran.empty() ? options : out.ran),
                         .device = deviceName(),
                         .detail = wrong ? wrongBy : lastFailure_ + "; any build log is in " + FAILURES_DIR + '/',
                         .at = ts,
                         .id = failureId(row)});
  }
  if (out.power && !out.usPerIt.empty()) {
    double const iterations = double(out.usPerIt.size()) * blockSize;
    (void)db_.add(PowerRow{.sess = session_,
                           .fft = fft.spec(),
                           .kind = kind,
                           .exponent = exponent,
                           .cfg = cfg,
                           .mjPerIt = out.power->joules * 1000 / iterations,
                           .watts = out.power->watts,
                           .capW = std::isnan(out.power->capW) ? 0 : out.power->capW,
                           .smMHz = std::isnan(out.power->smMHz) ? 0 : out.power->smMHz,
                           .source = out.power->counted ? PowerSource::Counter
                                     : out.power->instant ? PowerSource::Instant
                                                          : PowerSource::Readings,
                           .ts = ts});
  }
  recordRegisters(fft, kind, out, cfg);
  std::vector<std::string> spilled;
  for (const KernelUse& k : out.kernels) {
    if (k.resources.localBytes > 0 && std::ranges::find(spilled, k.key) == spilled.end()) { spilled.push_back(k.key); }
  }
  if (out.ok() && !spilled.empty()) {
    (void)db_.add(
      SpillRow{.sess = session_, .fft = fft.spec(), .kind = kind, .cfg = cfg, .keys = spilled, .ts = now()});
  }
  return out;
}

void Bench::recordRegisters(const FFTConfig& fft, TestKind kind, const Call& call, u32 cfg) {
  auto const same = [](const KernelResources& a, const KernelResources& b) {
    return a.regs == b.regs && a.localBytes == b.localBytes && a.sharedBytes == b.sharedBytes && a.threads == b.threads;
  };
  for (const KernelUse& k : call.kernels) {
    // Only a build that left the kernel to the compiler says what the occupancy model should start from.
    if (useValue(call.ran, k.key, 0) != 0) { continue; }
    const RegsRow* const first = db_.findRegs(envId_, fft.spec(), kind, k.key);
    bool const known =
      (first && same(first->resources, k.resources)) || std::ranges::any_of(db_.kres(), [&](const KresRow& r) {
        return db_.envOf(r.sess) == envId_ && r.fft == fft.spec() && r.kind == kind && r.key == k.key &&
          same(r.resources, k.resources);
      });
    if (first && !known) {
      (void)db_.add(KresRow{.sess = session_,
                            .fft = fft.spec(),
                            .kind = kind,
                            .key = k.key,
                            .resources = k.resources,
                            .plan = k.plan,
                            .cfg = cfg,
                            .ts = now()});
    }
    if (first) { continue; }
    (void)db_.add(RegsRow{.sess = session_,
                          .fft = fft.spec(),
                          .kind = kind,
                          .key = k.key,
                          .resources = k.resources,
                          .plan = k.plan,
                          .ts = now()});
  }
}

Call Bench::attempt(GpuCommon shared, const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options,
                    u32 nBlocks, u32 blockSize, double heatSec) {
  Call out;
  KernelCompiler::Tally const tallyBefore = KernelCompiler::tally();
  if (aheadOn_ && !stopped_) { compileAhead(fft, kind, exponent, options); }
  coolDown();
  if (stopped_) {
    out.status = Status::Lost;
    return out;
  }
  std::unique_ptr<Gpu> gpu;
  {
    u32 const cfg = db_.internCfg(builtAs(env_, fft, options));
    Attempt const attempt{
      db_, TryRow{.sess = session_, .fft = fft.spec(), .kind = kind, .exponent = exponent, .cfg = cfg, .ts = now()},
      attemptText(fft, kind, exponent, options)};
    if (!cfg || !attempt.declared()) {
      cannotDeclare(fft, options);
      out.status = Status::Lost;
      return out;
    }

    if (Status const status = caught(
          [&] {
            out = timeCall(shared, env_, fft, kind, exponent, options, nBlocks, blockSize, heatSec,
                           heaterOn_ ? &gpu : nullptr);
          },
          fft, kind, exponent, options, "timing");
        status != Status::Ok) {
      out.status = status;
    }
  }
  if (!out.loaded.empty()) { kernelsRun_[kernelsRunBy(fft, kind)].insert(out.loaded.begin(), out.loaded.end()); }
  if (args_.verbose) {
    log("measure: kernels of %s -use %s: %s\n", fft.spec().c_str(), configText(options).c_str(),
        tallyText(KernelCompiler::tally() - tallyBefore).c_str());
  }
  // Once the attempt has put back what it found, so the heater's name for what runs is the one left standing.
  if (gpu && out.ok() && !stopped_) {
    double const chunkIters = HEATER_CHUNK_SEC / (out.mean() * 1e-6);
    u32 const blocks = u32(std::clamp(std::lround(chunkIters / blockSize), 1L, 1000L));
    heater_ =
      std::make_unique<Heater>(std::move(gpu), kind, blocks, blockSize, attemptText(fft, kind, exponent, options));
  }
  return out;
}

void Bench::compileAhead(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options) {
  auto const it = kernelsRun_.find(kernelsRunBy(fft, kind));
  if (it == kernelsRun_.end()) { return; }
  aheadKernels_ += Gpu::compileAhead(exponent, shared_, fft, asExtraConf(env_, fft, options), it->second);
  ++aheadCalls_;
}

std::optional<u64> Bench::llReference(const FFTConfig& fft, u64 exponent, u32 nBlocks, u32 blockSize) {
  u64 const iters = callIterations(nBlocks, blockSize);
  if (auto const agreed = agreedResidue(referenceReadings(db_, envId_, exponent, iters))) { return agreed; }
  if (unreferenced_.contains({exponent, iters})) { return {}; }

  auto read = [&](const FFTConfig& witness) -> std::optional<u64> {
    if (stopped_ || !held(witness, {}).empty()) { return {}; }
    Call const c = attempt(shared_, witness, TestKind::LL, exponent, {}, nBlocks, blockSize);
    if (!c.ok()) { return {}; }
    if (!atBuiltInDefaults(env_, witness, c.ran)) {
      log("measure: %s was built as -use %s rather than at the built-in defaults, so it cannot vote on the LL "
          "reference\n",
          witness.spec().c_str(), configText(c.ran).c_str());
      return {};
    }
    log("measure: LL reference at %" PRIu64 " after %" PRIu64 " iterations: %s reads %016" PRIx64 "\n", exponent, iters,
        witness.spec().c_str(), c.res64);
    return c.res64;
  };
  std::optional<u64> const agreed =
    settleReference(db_, session_, exponent, iters, witnessOrder(env_, fft, exponent), read);

  if (stopped_) { return {}; }
  if (agreed) { return agreed; }

  log("measure: no LL reference could be agreed at %" PRIu64 " after %" PRIu64 " iterations, so no LL configuration is "
      "timed there by this session\n",
      exponent, iters);
  unreferenced_.insert({exponent, iters});
  return {};
}

std::vector<double> Bench::soak(const std::vector<SoakEntry>& entries, u64 exponent, double warm, double slice,
                                u32 rounds) {
  std::vector<double> out(entries.size(), NAN);
  if (entries.empty() || stopped_) { return out; }

  // An LL slice has no check of its own, so its chunks are as long as a call and end where the LL reference does.  Settled
  // before any soak Gpu is built: its witnesses build Gpus of their own.
  std::vector<std::optional<u64>> references(entries.size());
  for (size_t i = 0; i < entries.size() && !stopped_; ++i) {
    if (entries[i].kind == TestKind::LL) { references[i] = llReference(entries[i].fft, exponent, blocks_, blockSize_); }
  }
  if (stopped_) { return out; }

  keepAnchor();
  coolDown();
  if (stopped_) { return out; }
  u32 const group = db_.nextSoakGroup();

  // Every one built first, each under an attempt of its own, so that a build that takes the device down is its own.
  std::vector<std::unique_ptr<Gpu>> gpus(entries.size());
  std::vector<u32> cfgs(entries.size());
  auto tryOf = [&](size_t i) {
    return TryRow{.sess = session_,
                  .fft = entries[i].fft.spec(),
                  .kind = entries[i].kind,
                  .exponent = exponent,
                  .cfg = cfgs[i],
                  .ts = now()};
  };
  for (size_t i = 0; i < entries.size(); ++i) {
    const SoakEntry& e = entries[i];
    cfgs[i] = db_.internCfg(builtAs(env_, e.fft, e.options));
    if (std::string const why = held(e.fft, e.options); !why.empty()) { continue; }
    if (e.kind == TestKind::LL && !references[i]) {
      log("measure: %s -use %s is not soaked: no LL reference at %" PRIu64 " to check it against\n", e.fft.spec().c_str(),
          configText(e.options).c_str(), exponent);
      continue;
    }
    Attempt const attempt{db_, tryOf(i), attemptText(e.fft, e.kind, exponent, e.options)};
    if (!cfgs[i] || !attempt.declared()) {
      cannotDeclare(e.fft, e.options);
      return out;
    }
    (void)caught([&] { gpus[i] = Gpu::make(exponent, shared_, e.fft, asExtraConf(env_, e.fft, e.options), false); },
                 e.fft, e.kind, exponent, e.options, "build for a soak");
    if (stopped_) { return out; }
  }

  // One slice: squaring in chunks of a few seconds until `seconds` have passed, recorded under an attempt of its own,
  // which the slice's record answers.  A chunk is judged as a call is: one that fails its Gerbicz check, or for LL reads
  // other than the reference, is read again at once, and only a second failure is believed -- which is recorded as the
  // call that failed would be, so the configuration is never published, and the slice is not.
  constexpr double CHUNK_SEC = 5;
  auto runSlice = [&](size_t i, u32 round, double seconds) -> std::optional<double> {
    const SoakEntry& e = entries[i];
    bool const ll = e.kind == TestKind::LL;
    Attempt const attempt{db_, tryOf(i), attemptText(e.fft, e.kind, exponent, e.options)};
    if (!attempt.declared()) {
      cannotDeclare(e.fft, e.options);
      return {};
    }
    std::vector<double> blocks;
    bool wrong = false;
    auto const from = std::chrono::steady_clock::now();
    Timer t;
    Status const status = caught(
      [&] {
        u32 n = ll ? blocks_ : 2;
        auto chunk = [&] {
          return summarize(ll ? gpus[i]->timeItersLL(n, blockSize_, CALL_WARMUP_BLOCKS, 0)
                              : gpus[i]->timeIters(n, blockSize_, CALL_WARMUP_BLOCKS, 0));
        };
        while (t.at() < seconds) {
          Call c = chunk();
          if (c.status == Status::Err || (ll && c.ok() && c.res64 != *references[i])) {
            log("measure: %s -use %s failed its check during soak %u; reading it again, since one failure decides "
                "nothing\n", e.fft.spec().c_str(), configText(e.options).c_str(), group);
            c = ll ? checkedAgainst(*references[i], std::move(c), chunk) : checkedTwice(std::move(c), chunk);
          }
          if (c.status == Status::Err) {
            wrong = true;
            return;
          }
          blocks.insert(blocks.end(), c.usPerIt.begin(), c.usPerIt.end());
          if (!ll) { n = std::max(1u, u32(std::lround(CHUNK_SEC / (statsOf(c.usPerIt).mean * blockSize_ * 1e-6)))); }
        }
      },
      e.fft, e.kind, exponent, e.options, "soak");
    if (wrong) {
      log("measure: it failed again: it computes wrongly, and is recorded as an error\n");
      CallRow const row{.sess = session_,
                        .fft = e.fft.spec(),
                        .kind = e.kind,
                        .exponent = exponent,
                        .cfg = cfgs[i],
                        .status = Status::Err,
                        .drift = anchorState_.ratio,
                        .ts = now()};
      if (db_.add(row)) {
        (void)recordFailure({.kind = FailureKind::Wrong,
                             .fft = row.fft,
                             .test = e.kind,
                             .exponent = exponent,
                             .options = configText(e.options),
                             .device = deviceName(),
                             .detail = ll ? "its LL residue differed from the reference twice in a row during a soak"
                                          : "its Gerbicz check failed twice in a row during a soak",
                             .at = row.ts,
                             .id = failureId(row)});
      }
      gpus[i].reset();
      return {};
    }
    if (status != Status::Ok || blocks.empty()) {
      gpus[i].reset();
      return {};
    }
    auto const to = std::chrono::steady_clock::now();
    double watts = 0;
    if (sampler_ && sampler_->waitPast(sampler_->at(to), 5 * SAMPLE_PERIOD)) {
      if (std::optional<WindowPower> const p = powerOver(*sampler_, from, to)) { watts = p->watts; }
    }
    Stats const s = statsOf(blocks);
    (void)db_.add(SoakRow{.sess = session_,
                          .fft = e.fft.spec(),
                          .kind = e.kind,
                          .exponent = exponent,
                          .cfg = cfgs[i],
                          .group = group,
                          .round = round,
                          .mean = s.mean,
                          .sd = s.sd,
                          .blocks = s.n,
                          .watts = std::isnan(watts) ? 0 : watts,
                          .sec = std::max(t.at(), 0.1),
                          .ts = now()});
    log("measure: soak %u round %u: %s -use %s at %" PRIu64 ": %.2f us/it over %.0f s%s\n", group, round,
        e.fft.spec().c_str(), configText(e.options).c_str(), exponent, s.mean, t.at(),
        watts > 0 ? (", " + std::to_string(int(std::lround(watts))) + " W").c_str() : "");
    return s.mean;
  };

  auto const first = std::ranges::find_if(gpus, [](const auto& g) { return g != nullptr; });
  if (first == gpus.end()) { return out; }
  (void)runSlice(size_t(first - gpus.begin()), 0, warm);
  std::vector<std::vector<double>> sustained(entries.size());
  for (u32 round = 1; round <= rounds && !stopped_; ++round) {
    for (size_t i = 0; i < entries.size() && !stopped_; ++i) {
      if (!gpus[i]) { continue; }
      if (std::optional<double> const us = runSlice(i, round, slice)) { sustained[i].push_back(*us); }
    }
  }
  for (size_t i = 0; i < entries.size(); ++i) {
    if (!sustained[i].empty()) { out[i] = statsOf(sustained[i]).mean; }
  }
  return out;
}

RoeCheck Bench::roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) {
  RoeCheck out{.exponent = exponent};
  coolDown();
  if (stopped_) { return out; }

  if (std::string const why = held(fft, options); !why.empty()) {
    log("measure: %s -use %s is not built again here: %s\n", fft.spec().c_str(), configText(options).c_str(),
        why.c_str());
    out.status = Status::Unsupported;
    return out;
  }

  u32 const cfg = db_.internCfg(builtAs(env_, fft, options));
  Attempt const attempt{
    db_,
    TryRow{.sess = session_, .fft = fft.spec(), .kind = TestKind::PRP, .exponent = exponent, .cfg = cfg, .ts = now()},
    attemptText(fft, TestKind::PRP, exponent, options)};
  if (!cfg || !attempt.declared()) {
    cannotDeclare(fft, options);
    out.status = Status::Lost;
    return out;
  }

  if (Status const status = caught([&] { out = roeCheck(shared_, env_, fft, options, exponent); }, fft, TestKind::PRP,
                                   exponent, options, "accuracy check");
      status != Status::Ok) {
    return RoeCheck{.exponent = exponent, .status = status};
  }

  if (out.applicable && !stopped_) {
    (void)db_.add(RoeRow{.sess = session_,
                         .fft = fft.spec(),
                         .exponent = exponent,
                         .cfg = db_.internCfg(builtAs(env_, fft, out.ran.empty() ? options : out.ran)),
                         .checkOk = out.checkOk,
                         .z = out.z,
                         .n = out.n,
                         .maxRoe = out.maxRoe,
                         .ts = now()});
  }
  return out;
}

}  // namespace tune
