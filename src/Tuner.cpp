// Copyright (C) Jason Lynch

#include "Tuner.h"

#include "Args.h"
#include "BuildId.h"
#include "File.h"
#include "fs.h"
#include "GpuCommon.h"
#include "log.h"
#include "Signal.h"
#include "version.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <map>

namespace tune {

namespace {

std::string percent(double fraction) {
  if (!std::isfinite(fraction)) { return "unknown"; }
  char buf[32];
  snprintf(buf, sizeof(buf), "%.2f%%", fraction * 100);
  return buf;
}

std::string optionsText(const UseConfig& options) { return options.empty() ? "the defaults" : configText(options); }

// The tune files as last written, so a file is rewritten only when what it says changes.
class Publisher {
public:
  Publisher(fs::path dir, std::string header) : dir_{std::move(dir)}, header_{std::move(header)} {}

  void write(const Publication& publication) {
    if (dir_.empty()) { return; }
    for (const KindPublication& kind : publication.kinds) {
      if (kind.lines.empty()) { continue; }
      std::string const text = tuneFileText(kind, header_);
      std::string& last = written_[kind.kind];
      if (text == last) { continue; }
      keepUsersFile(kind.kind);
      writeTuneFile(dir_, kind, header_);
      last = text;
      log("tune: wrote %s: %zu lines, T %.2f us/it over %s of the workload\n", tuneFileName(kind.kind),
          kind.lines.size(), kind.covered > 0 ? kind.T / kind.covered : 0.0, percent(kind.covered).c_str());
    }
  }

private:
  // A tune file this tuner did not write is copied aside once before it is first replaced.
  void keepUsersFile(TestKind kind) {
    fs::path const path = dir_ / tuneFileName(kind);
    if (written_.contains(kind) && !written_[kind].empty()) { return; }
    File in = File::openRead(path);
    if (!in) { return; }
    std::string const text = in.readAll();
    if (text.starts_with("# " + header_.substr(0, header_.find('\n')))) { return; }
    fs::path const aside = path + std::string{".before-tune"};
    std::error_code ec;
    if (!fs::exists(aside, ec)) {
      fs::copy_file(path, aside, ec);
      log("tune: %s was not written by this tuner; kept as %s\n", path.filename().string().c_str(),
          aside.filename().string().c_str());
    }
  }

  fs::path dir_;
  std::string header_;
  std::map<TestKind, std::string> written_;
};

}  // namespace

LoopResult runLoop(Device& device, TuneDB& db, u32 env, const Env& card, const RunScope& scope,
                   const LoopSettings& settings) {
  std::vector<std::vector<Candidate>> candidates;
  for (const Grid& g : scope.grids) { candidates.push_back(candidatesOf(card, scope, g.kind)); }

  Model model{card};
  std::mt19937_64 rng{settings.seed};
  Publisher publisher{settings.publishTo, settings.header};
  LoopResult out;
  u32 sinceRegret = settings.regretEvery;
  u32 items = 0;

  while (!device.stopped() && !Signal::stopRequested() && (!settings.maxItems || items < settings.maxItems)) {
    device.keepAnchor();
    if (device.stopped()) { break; }

    Readings const readings{db, env, card};
    model.fit(readings.observations());
    Publication const publication = publish(scope, candidates, readings);
    out.T = publication.T();
    publisher.write(publication);

    // What the occupancy model offers each FFT's register limits, from its `regs` records; built before the search,
    // which asks from many threads.
    std::map<std::string, RegisterOffer> offers;
    for (const RegsRow& r : db.regs()) {
      if (db.envOf(r.sess) == env) { offers[r.fft + '|' + toString(r.kind)].try_emplace(r.key, r.plan.values()); }
    }
    RegisterOffer const none{};
    auto filter = [&](const Candidate& c) {
      auto const it = offers.find(c.subject.fft.spec() + '|' + toString(c.subject.kind));
      return AxisFilter{.tailQueues = settings.tailQueues, .registers = it == offers.end() ? &none : &it->second};
    };
    auto blocked = [&](const Candidate& c, const UseConfig& options) {
      const Reading* const r = readings.find(c.subject, options);
      return (r && r->failed) || !device.held(c, options).empty();
    };
    MeanOptima means;
    AcquireContext const context{.model = model,
                                 .scope = scope,
                                 .candidates = candidates,
                                 .readings = readings,
                                 .filter = filter,
                                 .blocked = blocked,
                                 .means = &means};
    ++items;

    if (!publication.owedGates.empty()) {
      const GateItem& g = publication.owedGates.front();
      RoeCheck const r = device.roe(g.fft, g.options, g.exponent);
      ++out.gateReadings;
      log("tune: %u. gate %s -use %s at %" PRIu64 ": %s\n", items, g.fft.spec().c_str(), optionsText(g.options).c_str(),
          g.exponent,
          r.status != Status::Ok ? toString(r.status)
            : r.applicable       ? ("z " + std::to_string(r.z).substr(0, 5)).c_str()
                                 : "exact");
      continue;
    }

    if (!publication.owedCalls.empty()) {
      const ConcludeItem& c = publication.owedCalls.front();
      Call const call = device.call(*c.candidate, c.options);
      ++out.calls;
      ++sinceRegret;
      log("tune: %u. %s %s -use %s at %" PRIu64 ": %.1f us/it, its second call\n", items,
          toString(c.candidate->subject.kind), c.candidate->subject.fft.spec().c_str(), optionsText(c.options).c_str(),
          c.candidate->timing, call.mean());
      continue;
    }

    if (sinceRegret >= settings.regretEvery) {
      sinceRegret = 0;
      out.regret = expectedRegret(context, publication, rng, settings.regretDraws);
      log("tune: T %.2f us/it over %s of the workload; expected regret %s\n", out.T,
          percent(publication.coverage()).c_str(), percent(out.regret).c_str());
      if (settings.stop > 0 && out.regret <= settings.stop) {
        out.converged = true;
        break;
      }
    }

    std::optional<Choice> const choice = choose(context, rng);
    if (!choice) {
      out.regret = expectedRegret(context, publication, rng, settings.regretDraws);
      out.converged = out.regret <= settings.stop;
      log("tune: no draw names anything not already well measured; expected regret %s\n", percent(out.regret).c_str());
      break;
    }

    Call const call = device.call(*choice->candidate, choice->options);
    ++out.calls;
    ++sinceRegret;
    log("tune: %u. %s %s -use %s at %" PRIu64 ": %s (predicted %.1f; %s)\n", items,
        toString(choice->candidate->subject.kind), choice->candidate->subject.fft.spec().c_str(),
        optionsText(choice->options).c_str(), choice->candidate->timing,
        call.ok() ? (std::to_string(call.mean()).substr(0, 7) + " us/it").c_str() : toString(call.status),
        choice->predictedUs, choice->why.c_str());
  }
  return out;
}

namespace {

// The device, through a measuring session.
class BenchDevice final : public Device {
public:
  explicit BenchDevice(Bench& bench) : bench_{bench} {}

  Call call(const Candidate& candidate, const UseConfig& options) override {
    return bench_.call(candidate.subject.fft, candidate.subject.kind, candidate.timing, options);
  }

  RoeCheck roe(const FFTConfig& fft, const UseConfig& options, u64 exponent) override {
    return bench_.roe(fft, options, exponent);
  }

  [[nodiscard]] std::string held(const Candidate& candidate, const UseConfig& options) const override {
    return bench_.held(candidate.subject.fft, candidate.subject.kind, candidate.timing, options);
  }

  void keepAnchor() override { bench_.keepAnchor(); }

  [[nodiscard]] bool stopped() const override { return bench_.stopped(); }

private:
  Bench& bench_;
};

std::string headerFor(const RunScope& scope, const TuneCommand& command) {
  return std::string{"written by prpll "} + VERSION + " -tune " + settingsWord(scope, command) +
    "\nfrom tunedb.txt; each line's maximum exponent is how far its settings were found to be accurate";
}

void reportScope(const RunScope& scope) {
  log("tune: workload %" PRIu64 "-%" PRIu64 " (%s)\n", scope.lo, scope.hi, scope.rangeSource.c_str());
  log("tune: probe %" PRIu64 " (%s), carrying %.0f%% of the weight\n", scope.probe, scope.probeSource.c_str(),
      scope.probeWeight * 100);
  for (const Grid& g : scope.grids) { log("tune: %s grid: %zu points\n", toString(g.kind), g.points.size()); }
}

std::optional<RunScope> scopeFor(const TuneCommand& command, const Args& args) {
  try {
    RunScope scope = makeScope(command.scope, scanWorktodo(worktodoFiles(args, fs::current_path())));
    return scope;
  } catch (const std::string& mes) { log("%s\n", mes.c_str()); } catch (const char* mes) {
    log("%s\n", mes);
  }
  return {};
}

}  // namespace

bool runTune(GpuCommon shared, const TuneCommand& command) {
  const Args& args = *shared.args;
  Takeover const takeover = takeOver(args);
  if (!takeover.setAside.empty()) {
    std::string keys;
    for (const std::string& k : takeover.setAside) { keys += (keys.empty() ? "" : ", ") + k; }
    log("tune: these -use keys are set aside while tuning, and reach production under every line that does not set "
        "them: %s\n",
        keys.c_str());
  }
  if (takeover.carry) { log("tune: -carry is set aside: a reading's carry is the one its exponent takes\n"); }

  std::optional<RunScope> const scope = scopeFor(command, args);
  if (!scope) { return false; }
  reportScope(*scope);

  fs::path const path = fs::current_path() / TuneDB::DEFAULT_NAME;
  TuneDB db;
  if (!db.lockForWriting(path) || !db.load(path)) { return false; }
  db.attach(path);

  Env const card = detectEnv(*shared.context, args);
  Bench bench{shared, db, card};
  if (!bench.begin(scope->probe, settingsWord(*scope, command))) {
    log("tune: could not open a session in %s\n", path.string().c_str());
    return false;
  }

  u64 const seed = command.seed ? command.seed : u64(std::chrono::steady_clock::now().time_since_epoch().count());
  BenchDevice device{bench};
  LoopResult const result = runLoop(device, db, bench.envId(), card, *scope,
                                    {.stop = command.stop,
                                     .seed = seed,
                                     .tailQueues = command.tailQueues,
                                     .publishTo = fs::current_path(),
                                     .header = headerFor(*scope, command)});

  log("tune: %u calls and %u accuracy readings; T %.2f us/it; expected regret %s%s\n", result.calls,
      result.gateReadings, result.T, percent(result.regret).c_str(),
      result.converged       ? ", below the stop"
        : bench.deviceLost() ? ", stopped by a lost device"
                             : "");
  return !bench.deviceLost();
}

bool runOffline(const TuneCommand& command, const Args& args) {
  std::optional<RunScope> const scope = scopeFor(command, args);
  if (!scope) { return false; }
  reportScope(*scope);
  if (command.verb == TuneVerb::Scope) { return true; }

  fs::path const path = fs::current_path() / TuneDB::DEFAULT_NAME;
  TuneDB db;
  if (!db.load(path)) { return false; }

  // The env this binary's kernels were measured on; with several cards, the one most recently measured.
  const DbEnv* env = nullptr;
  for (const DbEnv& e : db.envs()) {
    if (e.build == buildFingerprint()) { env = &e; }
  }
  if (!env) {
    log("tune: %s has no readings of this build's kernels\n", path.string().c_str());
    return false;
  }
  Env const card = env->toEnv();

  std::vector<std::vector<Candidate>> candidates;
  for (const Grid& g : scope->grids) { candidates.push_back(candidatesOf(card, *scope, g.kind)); }
  Readings const readings{db, env->id, card};
  Publication const publication = publish(*scope, candidates, readings);

  for (const KindPublication& kind : publication.kinds) {
    log("tune: %s: %zu lines, T %.2f us/it over %s of the workload\n", toString(kind.kind), kind.lines.size(),
        kind.covered > 0 ? kind.T / kind.covered : 0.0, percent(kind.covered).c_str());
    for (const Line& line : kind.lines) {
      log("tune:   %8.1f %s -use %s up to %" PRIu64 " (%s)\n", line.cost(), line.candidate->subject.fft.spec().c_str(),
          optionsText(line.reading->options).c_str(), line.reach(), toString(line.accuracy.state));
    }
  }
  log("tune: %zu accuracy readings and %zu calls owed before more lines can be published\n",
      publication.owedGates.size(), publication.owedCalls.size());

  if (command.verb == TuneVerb::Emit) {
    Publisher publisher{fs::current_path(), headerFor(*scope, command)};
    publisher.write(publication);
  }
  return true;
}

}  // namespace tune
