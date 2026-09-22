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

#include <cinttypes>
#include <cmath>
#include <ctime>
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

  auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(options), false, kind);
  Call out = summarize(gpu->timeIters(nBlocks, blockSize));
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

bool Session::begin() {
  envId_ = db_.internEnv(dbEnvOf(env_));
  if (!envId_) { return false; }

  session_ = db_.beginSession(envId_, "-", restart::generation());
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

Call Session::run(const FFTConfig& fft, TestKind kind, u64 exponent, const UseConfig& options, u32 nBlocks,
                  u32 blockSize) {
  Call out;
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

  // Nothing is recorded for a stop.
  if (stopped_) { return out; }

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

MeasureOutcome runMeasure(GpuCommon shared, const std::string& fftSpec, u64 exponent) {
  static const Primes primes;

  FFTConfig const fft{fftSpec};
  if (!exponent) { exponent = primes.prevPrime(fft.maxExp()); }
  if (!primes.isPrime(exponent)) { log("measure: warning: %" PRIu64 " is not prime\n", exponent); }

  u32 const blockSize = shared.args->blockSize;
  u32 const nBlocks = BLOCKS_PER_CALL;

  log("measure: %s at exponent %" PRIu64 " (%.2f bpw), %u blocks of %u\n", fft.spec().c_str(), exponent,
      double(exponent) / fft.shape.size(), nBlocks, blockSize);

  fs::path const dbPath = TuneDB::DEFAULT_NAME;
  TuneDB db;
  // Before the load, and held for the rest of the run: every id this writes is allocated from what it read.
  if (!db.lockForWriting(dbPath)) { return MeasureOutcome::Failed; }
  if (!db.load(dbPath)) { return MeasureOutcome::Failed; }
  db.attach(dbPath);

  Session session{shared, db, detectEnv(*shared.context, *shared.args)};
  if (!session.begin()) {
    log("measure: '%s' would not take a session\n", dbPath.string().c_str());
    return MeasureOutcome::Failed;
  }
  if (u32 const gen = restart::generation()) { log("measure: generation %u\n", gen); }

  UseConfig const options = resolveConfig(*shared.args, fft, TestKind::PRP);

  std::vector<std::string> varied;
  for (const auto& [key, value] : shared.args->flags) { varied.push_back(key); }
  session.varying(varied);

  if (std::string const why = session.held(fft, TestKind::PRP, exponent, options); !why.empty()) {
    log("measure: skipping %s -use %s: %s.\n"
        "measure:   It will not be built again on this device.\n",
        fft.spec().c_str(), configText(options).c_str(), why.c_str());
    session.end();
    return MeasureOutcome::Failed;
  }

  bool ok = true;
  for (u32 call = 0; call < MIN_CALLS && !session.stopped(); ++call) {
    Call const c = session.run(fft, TestKind::PRP, exponent, options, nBlocks, blockSize);
    if (session.stopped() || !c.measurement.ok()) {
      ok = false;
      continue;
    }

    string blocks;
    for (double const us : c.usPerIt) {
      char buf[32];
      snprintf(buf, sizeof(buf), " %.3f", us);
      blocks += buf;
    }
    log("measure: call %u:%s us/it\n", call, blocks.c_str());
    log("measure: call %u: mean %.3f sd %.3f (%.3f%%) blocks %u dropped %u%s, %s %016" PRIx64 "\n", call,
        c.measurement.mean, c.measurement.stddev, c.measurement.stddev / c.measurement.mean * 100, c.measurement.blocks,
        c.dropped, c.declined ? " (declined)" : "", c.checkOk ? "OK" : "EE", c.res64);
    ok = ok && c.checkOk;
  }

  // What draining at every block boundary costs.
  if (!session.stopped()) {
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

  if (!session.stopped()) {
    RoeCheck const roe = session.checkRoe(fft, options, exponent);
    if (roe.status != Status::Ok) {
      log("measure: ROE: could not be checked (%s)\n", toString(roe.status));
      ok = false;
    } else if (!roe.applicable) {
      log("measure: ROE: not applicable (exact arithmetic)\n");
    } else {
      log("measure: ROE at %" PRIu64 ": z %.2f (floor %.0f) n %u max %f, %s%s\n", roe.exponent, roe.z, roe.minZ, roe.n,
          roe.maxRoe, roe.checkOk ? "OK" : "EE", roe.conclusive() ? "" : ", inconclusive");
      ok = ok && roe.passed();
    }
  }

  if (session.deviceLost()) { return MeasureOutcome::DeviceLost; }
  session.end();
  if (session.cannotRecord()) { return MeasureOutcome::Failed; }
  if (session.stopped()) { return MeasureOutcome::Ok; }
  return ok ? MeasureOutcome::Ok : MeasureOutcome::Failed;
}

}  // namespace tune
