// Copyright (C) Jason Lynch

#include "Measure.h"

#include "Args.h"
#include "Gpu.h"
#include "log.h"
#include "Primes.h"

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <vector>

namespace tune {

namespace {

std::vector<KeyVal> asExtraConf(const UseConfig& options) { return {options.begin(), options.end()}; }

}  // namespace

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

bool runMeasure(GpuCommon shared, const std::string& fftSpec, u64 exponent) {
  static const Primes primes;

  FFTConfig const fft{fftSpec};
  if (!exponent) { exponent = primes.prevPrime(fft.maxExp()); }
  if (!primes.isPrime(exponent)) { log("measure: warning: %" PRIu64 " is not prime\n", exponent); }

  u32 const blockSize = shared.args->blockSize;
  u32 const nBlocks = BLOCKS_PER_CALL;

  log("measure: %s at exponent %" PRIu64 " (%.2f bpw), %u blocks of %u\n", fft.spec().c_str(), exponent,
      double(exponent) / fft.shape.size(), nBlocks, blockSize);

  UseConfig const options = resolveConfig(*shared.args, fft, TestKind::PRP);

  bool ok = true;
  for (u32 call = 0; call < MIN_CALLS; ++call) {
    Call const c = timeCall(shared, fft, TestKind::PRP, exponent, options, nBlocks, blockSize);

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
  {
    auto time = [&](u32 blocks, u32 size) {
      auto gpu = Gpu::make(exponent, shared, fft, asExtraConf(options), false, TestKind::PRP);
      return statsOf(gpu->timeIters(blocks, size, 5000 / size).usPerIt).mean;
    };

    double const drained = time(5, 1000);
    double const whole = time(1, 5000);
    log("measure: 5x1000 per-block %.3f vs 1x5000 %.3f us/it, drain costs at most %+.3f%%\n", drained, whole,
        (drained / whole - 1) * 100);
  }

  RoeCheck const roe = roeCheck(shared, fft, options, exponent);
  if (!roe.applicable) {
    log("measure: ROE: not applicable (exact arithmetic)\n");
  } else {
    log("measure: ROE at %" PRIu64 ": z %.2f (floor %.0f) n %u max %f, %s%s\n", roe.exponent, roe.z, roe.minZ, roe.n,
        roe.maxRoe, roe.checkOk ? "OK" : "EE", roe.conclusive() ? "" : ", inconclusive");
    ok = ok && roe.passed();
  }

  return ok;
}

}  // namespace tune
