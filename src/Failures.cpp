// Copyright (C) Jason Lynch

#include "Failures.h"

#include "BuildId.h"
#include "File.h"
#include "version.h"

#include <cinttypes>
#include <ctime>
#include <fstream>
#include <map>
#include <tuple>

namespace tune {

namespace {

std::string localTime(u64 seconds) {
  time_t const t = time_t(seconds);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

}  // namespace

std::string failureId(const CallRow& c) {
  return "call " + std::to_string(c.sess) + ' ' + std::to_string(c.ts) + ' ' + c.fft + ' ' + std::to_string(c.cfg);
}

std::string failureId(const TryRow& t) { return "attempt " + std::to_string(t.sess) + ' ' + std::to_string(t.ts); }

const char* toString(FailureKind kind) {
  switch (kind) {
  case FailureKind::DeviceLost: return "took the device down";
  case FailureKind::Wrong: return "computed a wrong answer";
  case FailureKind::NoBuild: return "would not build";
  }
  return "?";
}

std::string failureText(const FailureEntry& e) {
  std::string const use = e.options.empty() || e.options == "-" ? std::string{} : " -use " + e.options;
  char exponent[32];
  snprintf(exponent, sizeof(exponent), "%" PRIu64, e.exponent);
  char build[64];
  snprintf(build, sizeof(build), "%s, kernels %016" PRIx64, VERSION, buildFingerprint());

  std::string out = localTime(e.at ? e.at : u64(std::time(nullptr))) + "  " + toString(e.kind) + '\n';
  out += "  config  -fft " + e.fft + use + "   (" + toString(e.test) + " at " + exponent + ")\n";
  out += "  repro   prpll -fft " + e.fft + use + (e.test == TestKind::LL ? " -ll " : " -prp ") + exponent +
         " -iters 20000\n";
  if (!e.device.empty()) { out += "  device  " + e.device + '\n'; }
  out += "  build   " + std::string{build} + '\n';
  if (!e.detail.empty()) { out += "  detail  " + e.detail + '\n'; }
  if (!e.id.empty()) { out += "  id      " + e.id + '\n'; }
  return out + '\n';
}

bool recordFailure(const FailureEntry& entry, const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream out{dir / FAILURES_LOG, std::ios::app};
  out << failureText(entry);
  return bool(out);
}

std::set<std::string> recordedFailureIds(const std::filesystem::path& dir) {
  std::set<std::string> out;
  std::ifstream in{dir / FAILURES_LOG};
  std::string const marker = "  id      ";
  for (std::string line; std::getline(in, line);) {
    if (line.starts_with(marker)) { out.insert(line.substr(marker.size())); }
  }
  return out;
}

u32 recordMissingFailures(const TuneDB& db, u32 env, const std::filesystem::path& dir) {
  std::set<std::string> const known = recordedFailureIds(dir);
  const DbEnv* const device = db.findEnv(env);
  auto optionsOf = [&](u32 cfg) {
    const UseConfig* const options = db.findCfg(cfg);
    return options ? configText(*options) : std::string{"?"};
  };
  u32 written = 0;
  for (const TryRow& t : db.deaths()) {
    if (db.envOf(t.sess) != env || known.contains(failureId(t))) { continue; }
    written += recordFailure({.kind = FailureKind::DeviceLost,
                              .fft = t.fft,
                              .test = t.kind,
                              .exponent = t.exponent,
                              .options = optionsOf(t.cfg),
                              .device = device ? device->gpu : "",
                              .detail = "a process was holding it when it died; it is not built on this card again",
                              .at = t.ts,
                              .id = failureId(t)},
                             dir);
  }
  for (const CallRow& c : db.calls()) {
    bool const wrong = c.status == Status::Err;
    if ((!wrong && c.status != Status::NoCompile) || db.envOf(c.sess) != env || known.contains(failureId(c))) {
      continue;
    }
    written += recordFailure({.kind = wrong ? FailureKind::Wrong : FailureKind::NoBuild,
                              .fft = c.fft,
                              .test = c.kind,
                              .exponent = c.exponent,
                              .options = optionsOf(c.cfg),
                              .device = device ? device->gpu : "",
                              .detail = "recorded in tunedb.txt before this log kept it, so with no more detail",
                              .at = c.ts,
                              .id = failureId(c)},
                             dir);
  }
  return written;
}

FailureTally failureTally(const TuneDB& db, u32 env) {
  std::set<std::pair<std::string, u32>> lost;
  for (const TryRow& t : db.deaths()) {
    if (db.envOf(t.sess) == env) { lost.insert({t.fft, t.cfg}); }
  }
  std::set<std::tuple<std::string, TestKind, u32>> wrong, noBuild;
  for (const CallRow& c : db.calls()) {
    if (db.envOf(c.sess) != env) { continue; }
    if (c.status == Status::Err) { wrong.insert({c.fft, c.kind, c.cfg}); }
    if (c.status == Status::NoCompile) { noBuild.insert({c.fft, c.kind, c.cfg}); }
  }
  return {.deviceLost = u32(lost.size()), .wrong = u32(wrong.size()), .noBuild = u32(noBuild.size())};
}

std::string toString(const FailureTally& t) {
  if (!t.total()) { return {}; }
  std::string parts;
  auto add = [&](u32 n, FailureKind kind) {
    if (n) { parts += (parts.empty() ? "" : ", ") + std::to_string(n) + ' ' + toString(kind); }
  };
  add(t.deviceLost, FailureKind::DeviceLost);
  add(t.wrong, FailureKind::Wrong);
  add(t.noBuild, FailureKind::NoBuild);
  return std::to_string(t.total()) + (t.total() == 1 ? " configuration" : " configurations") + " held out: " + parts +
         "; each is in " + FAILURES_DIR + '/' + FAILURES_LOG;
}

}  // namespace tune
