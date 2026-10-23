// Copyright (C) Jason Lynch

#include "Tuner.h"

#include "Args.h"
#include "BuildId.h"
#include "Coverage.h"
#include "Dashboard.h"
#include "Explore.h"
#include "File.h"
#include "fs.h"
#include "GpuCommon.h"
#include "log.h"
#include "Restart.h"
#include "Signal.h"
#include "version.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <ctime>
#include <limits>
#include <map>

namespace tune {

namespace {

std::string percent(double fraction) {
  if (!std::isfinite(fraction)) { return "unknown"; }
  char buf[32];
  snprintf(buf, sizeof(buf), "%.2f%%", fraction * 100);
  return buf;
}

std::string usText(double us) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.2f", us);
  return buf;
}

std::string optionsText(const UseConfig& options) { return options.empty() ? "the defaults" : configText(options); }

// The tune files as last written, so a file is rewritten only when what it says changes.
class Publisher {
public:
  Publisher(fs::path dir, std::string header) : dir_{std::move(dir)}, header_{std::move(header)} {}

  // `regret` is the run's latest estimate, over every kind; infinite where there is none.
  void write(const Publication& publication, double regret) {
    if (dir_.empty()) { return; }
    for (const KindPublication& kind : publication.kinds) {
      if (kind.lines.empty()) { continue; }
      // Where the run stands, for whoever is watching the file: only what changes as the run learns, so the file is
      // not rewritten on every call.
      std::string const status = "T " + usText(kind.covered > 0 ? kind.T / kind.covered : 0.0) + " us/it over " +
        percent(kind.covered) + " of the workload; expected regret " +
        (std::isfinite(regret) ? percent(regret) + " (all kinds)" : std::string{"unknown"});
      std::string const header = header_ + '\n' + status;
      std::string const text = tuneFileText(kind, header);
      std::string& last = written_[kind.kind];
      if (text == last) { continue; }
      keepUsersFile(kind.kind);
      writeTuneFile(dir_, kind, header);
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

void report(const ItemReport& item, Watch* watch) {
  std::string const predicted = item.predicted > 0 ? "predicted " + std::to_string(item.predicted).substr(0, 6) : "";
  std::string const note = predicted.empty() ? item.why : item.why.empty() ? predicted : predicted + "; " + item.why;
  log("tune: %u. %s%s -use %s at %" PRIu64 ": %s%s\n", item.n, item.kind == "gate" ? "gate " : "", item.what.c_str(),
      item.options.c_str(), item.exponent, item.result.c_str(), note.empty() ? "" : (" (" + note + ")").c_str());
  if (watch) { watch->finished(item); }
}

StateReport stateOf(const Publication& publication, const Readings& readings, const Model& model, const TuneDB& db,
                    const LoopResult& sofar) {
  StateReport out{.T = publication.T(),
                  .coverage = publication.coverage(),
                  .regret = sofar.regret,
                  .calls = sofar.calls,
                  .gates = sofar.gateReadings,
                  .observations = u32(readings.observations().size()),
                  .spreads = model.spreads()};
  for (const KindPublication& kind : publication.kinds) {
    for (const Line& line : kind.lines) {
      char buf[96];
      snprintf(buf, sizeof(buf), "%s %7.1f %-16s to %" PRIu64 " ", toString(kind.kind), line.cost(),
               line.candidate->subject.fft.spec().c_str(), line.reach());
      out.lines.push_back(buf + optionsText(line.reading->options));
    }
  }
  for (const auto& [key, r] : readings.all()) {
    if (r.failed) { out.faults.push_back("failed: " + key.fft + " " + toString(key.kind) + " -use " + key.options); }
  }
  for (const TryRow& t : db.diedHolding()) {
    const UseConfig* const options = db.findCfg(t.cfg);
    out.faults.push_back("died holding: " + t.fft + " " + toString(t.kind) + " -use " +
                         (options ? configText(*options) : std::string{"?"}));
  }
  return out;
}

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
    publisher.write(publication, out.regret);
    if (settings.watch) { settings.watch->state(stateOf(publication, readings, model, db, out)); }

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

    // What each item is for, written before it is measured, from when its choosing began: its lane's time.
    auto recordItem = [&](const std::string& lane, const std::string& what, u32 packet, u32 ordinal,
                          const FFTConfig& fft, TestKind kind, const UseConfig& options, u64 began) {
      (void)db.add(ItemRow{.sess = device.session(),
                           .lane = lane,
                           .what = what,
                           .packet = packet,
                           .ordinal = ordinal,
                           .fft = fft.spec(),
                           .kind = kind,
                           .cfg = db.internCfg(options),
                           .ts = began});
    };
    u64 const began = u64(std::time(nullptr));

    if (!publication.owedGates.empty()) {
      const GateItem& g = publication.owedGates.front();
      recordItem(DUTY, "gate", 0, 0, g.fft, TestKind::PRP, g.options, began);
      ItemReport item{
        .n = items, .kind = "gate", .what = g.fft.spec(), .options = optionsText(g.options), .exponent = g.exponent};
      if (settings.watch) { settings.watch->measuring("gate " + item.what + " at " + std::to_string(g.exponent)); }
      RoeCheck const r = device.roe(g.fft, g.options, g.exponent);
      ++out.gateReadings;
      char z[32];
      snprintf(z, sizeof(z), "z %.2f", r.z);
      item.result = r.status != Status::Ok ? toString(r.status) : r.applicable ? z : "exact";
      report(item, settings.watch);
      continue;
    }

    auto callOf = [&](const Candidate& c, const UseConfig& options, ItemReport item) {
      item.what = std::string{toString(c.subject.kind)} + ' ' + c.subject.fft.spec();
      item.options = optionsText(options);
      item.exponent = c.timing;
      if (settings.watch) { settings.watch->measuring(item.what + " -use " + item.options); }
      Call const call = device.call(c, options);
      ++out.calls;
      ++sinceRegret;
      char us[32];
      snprintf(us, sizeof(us), "%.1f us/it", call.mean());
      item.result = call.ok() ? us : toString(call.status);
      item.measured = call.ok() ? call.mean() : 0;
      report(item, settings.watch);
    };

    if (!publication.owedCalls.empty()) {
      const ConcludeItem& c = publication.owedCalls.front();
      recordItem(DUTY, "second-call", 0, 0, c.candidate->subject.fft, c.candidate->subject.kind, c.options, began);
      callOf(*c.candidate, c.options, {.n = items, .kind = "second call", .why = "its second call"});
      continue;
    }

    if (sinceRegret >= settings.regretEvery) {
      sinceRegret = 0;
      out.regret = expectedRegret(context, publication, rng, settings.regretDraws);
      OptimizerCounts const counts = optimizerCounts();
      std::string const inexact = counts.inexact == 0 ? std::string{}
                                                      : "; optimiser inexact in " + std::to_string(counts.inexact) +
          " of " + std::to_string(counts.searches) + " searches";
      log("tune: T %.2f us/it over %s of the workload; expected regret %s%s\n", out.T,
          percent(publication.coverage()).c_str(), percent(out.regret).c_str(), inexact.c_str());
      log("tune: coverage: %s\n", toString(summarize(card, candidates, readings, filter)).c_str());
      log("tune: lanes: %s\n", toString(laneTime(db, env)).c_str());
      if (settings.watch) { settings.watch->state(stateOf(publication, readings, model, db, out)); }
      if (settings.stop > 0 && out.regret <= settings.stop) {
        out.converged = true;
        break;
      }
    }

    auto modelStep = [&]() -> std::optional<Proposal> {
      std::optional<Choice> choice;
      if (settings.sweep > 0 && std::uniform_real_distribution<double>{0, 1}(rng) < settings.sweep) {
        choice = untriedStep(context, rng, true);
      }
      if (!choice) { choice = choose(context, rng); }
      if (!choice && settings.stop == 0) { choice = untriedStep(context, rng); }
      if (!choice) { return {}; }
      return Proposal{.candidate = choice->candidate,
                      .options = choice->options,
                      .lane = Lane::Model,
                      .what = "model",
                      .predictedUs = choice->predictedUs,
                      .why = choice->why};
    };
    std::optional<Proposal> proposal;
    for (Lane const lane : laneOrder(settings.lanes, laneTime(db, env))) {
      if (!settings.lanes.serves(lane)) { continue; }
      switch (lane) {
      case Lane::Model: proposal = modelStep(); break;
      case Lane::Coverage: proposal = coverageStep(context, rng); break;
      case Lane::Restarts: proposal = restartStep(context, db, env, rng); break;
      case Lane::Interactions: break;
      }
      if (proposal) { break; }
    }
    if (!proposal) {
      out.regret = expectedRegret(context, publication, rng, settings.regretDraws);
      out.converged = out.regret <= settings.stop;
      log("tune: no draw names anything not already well measured%s; expected regret %s\n",
          settings.stop == 0 ? ", and no untried step is left" : "", percent(out.regret).c_str());
      break;
    }

    recordItem(toString(proposal->lane), proposal->what, proposal->packet, proposal->ordinal,
               proposal->candidate->subject.fft, proposal->candidate->subject.kind, proposal->options, began);
    callOf(*proposal->candidate, proposal->options,
           {.n = items, .kind = "call", .predicted = proposal->predictedUs, .why = proposal->why});
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

  [[nodiscard]] u32 session() const override { return bench_.session(); }

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

RunOutcome runTune(GpuCommon shared, const TuneCommand& command) {
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
  if (!scope) { return RunOutcome::Failed; }
  reportScope(*scope);

  fs::path const path = fs::current_path() / TuneDB::DEFAULT_NAME;
  TuneDB db;
  if (!db.lockForWriting(path) || !db.load(path)) { return RunOutcome::Failed; }
  db.attach(path);

  Env const card = detectEnv(*shared.context, args);
  Bench bench{shared, db, card};
  if (!bench.begin(scope->probe, settingsWord(*scope, command))) {
    log("tune: could not open a session in %s\n", path.string().c_str());
    return RunOutcome::Failed;
  }

  u64 const seed = command.seed ? command.seed : u64(std::chrono::steady_clock::now().time_since_epoch().count());
  BenchDevice device{bench};

  std::optional<DashboardView> dashboard;
  if (command.dashboard && canTakeTerminal()) {
    std::string kinds;
    for (const Grid& g : scope->grids) { kinds += (kinds.empty() ? "" : "+") + std::string{toString(g.kind)}; }
    dashboard.emplace(card.deviceName,
                      kinds + " " + std::to_string(scope->lo) + "-" + std::to_string(scope->hi) + ", probe " +
                        std::to_string(scope->probe),
                      restart::generation());
  } else if (command.dashboard) {
    log("tune: dashboard=1 needs stdout to be a terminal; running without it\n");
  }
  LoopResult const result = runLoop(device, db, bench.envId(), card, *scope,
                                    {.stop = command.stop,
                                     .sweep = command.sweep,
                                     .lanes = command.lanes,
                                     .seed = seed,
                                     .tailQueues = command.tailQueues,
                                     .publishTo = fs::current_path(),
                                     .header = headerFor(*scope, command),
                                     .watch = dashboard ? &*dashboard : nullptr});
  dashboard.reset();

  log("tune: %u calls and %u accuracy readings; T %.2f us/it; expected regret %s%s\n", result.calls,
      result.gateReadings, result.T, percent(result.regret).c_str(),
      result.converged       ? ", below the stop"
        : bench.deviceLost() ? ", stopped by a lost device"
                             : "");
  return bench.deviceLost() ? RunOutcome::DeviceLost : RunOutcome::Ok;
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
    publisher.write(publication, std::numeric_limits<double>::infinity());
  }
  return true;
}

}  // namespace tune
