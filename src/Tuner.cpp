// Copyright (C) Jason Lynch

#include "Tuner.h"

#include "Args.h"
#include "Audit.h"
#include "BuildId.h"
#include "Coverage.h"
#include "Dashboard.h"
#include "Explore.h"
#include "File.h"
#include "fs.h"
#include "GpuCommon.h"
#include "log.h"
#include "Report.h"
#include "Restart.h"
#include "Signal.h"
#include "version.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
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
  std::string predicted = item.predicted > 0 ? "predicted " + std::to_string(item.predicted).substr(0, 6) : "";
  if (item.mean > 0 && std::abs(item.mean / item.predicted - 1) > 0.0005) {
    predicted += ", mean " + std::to_string(item.mean).substr(0, 6);
  }
  std::string const note = predicted.empty() ? item.why : item.why.empty() ? predicted : predicted + "; " + item.why;
  log("tune: %u. %s%s -use %s at %" PRIu64 ": %s%s\n", item.n, item.kind == "gate" ? "gate " : "", item.what.c_str(),
      item.options.c_str(), item.exponent, item.result.c_str(), note.empty() ? "" : (" (" + note + ")").c_str());
  if (watch) { watch->finished(item); }
}

StateReport stateOf(const Publication& publication, const Readings& readings, const Model& model, const TuneDB& db,
                    u32 env, const std::string& coverage, const LoopResult& sofar) {
  StateReport out{.T = publication.T(),
                  .coverage = publication.coverage(),
                  .regret = sofar.regret,
                  .calls = sofar.calls,
                  .gates = sofar.gateReadings,
                  .observations = u32(readings.observations().size()),
                  .spreads = model.spreads(),
                  .lanes = toString(laneTime(db, env)),
                  .explored = coverage};
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

std::map<const Candidate*, RegisterOffer> registerOffers(const TuneDB& db, u32 env, const Env& card,
                                                         const std::vector<std::vector<Candidate>>& candidates,
                                                         const Readings& readings) {
  struct Read {
    UseConfig options;  // canonical
    UseConfig branch;
    const RegisterPlan* plan;
  };
  std::map<std::string, std::map<std::string, std::vector<Read>>> plans;
  auto read = [&](const std::string& fft, TestKind kind, const std::string& key, const UseConfig& options,
                  const RegisterPlan* plan) {
    FFTConfig const f{fft};
    UseConfig const canonical = canonicalConfig(card, f, options);
    plans[fft + '|' + toString(kind)][key].push_back(
      {.options = canonical, .branch = branchOf(card, f, canonical), .plan = plan});
  };
  for (const RegsRow& r : db.regs()) {
    if (db.envOf(r.sess) == env) { read(r.fft, r.kind, r.key, {}, &r.plan); }
  }
  for (const KresRow& r : db.kres()) {
    const UseConfig* const options = db.findCfg(r.cfg);
    if (db.envOf(r.sess) == env && r.plan && options) { read(r.fft, r.kind, r.key, *options, &*r.plan); }
  }
  std::map<const Candidate*, RegisterOffer> offers;
  for (const std::vector<Candidate>& kind : candidates) {
    for (const Candidate& c : kind) {
      auto const it = plans.find(c.subject.fft.spec() + '|' + toString(c.subject.kind));
      if (it == plans.end()) { continue; }
      const Reading* best = nullptr;
      for (const Reading* r : readings.of(c.subject)) {
        if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
      }
      UseConfig const around = best ? best->options : UseConfig{};
      UseConfig const branch = branchOf(card, c.subject.fft, around);
      RegisterOffer& offer = offers[&c];
      for (const auto& [key, reads] : it->second) {
        auto const rank = [&](const Read& r) { return std::pair{r.branch != branch, axesApart(r.options, around)}; };
        offer[key] = std::ranges::min(reads, {}, rank).plan->values();
      }
    }
  }
  return offers;
}

LoopResult runLoop(Device& device, TuneDB& db, u32 env, const Env& card, const RunScope& scope,
                   const LoopSettings& settings) {
  std::vector<std::vector<Candidate>> candidates;
  for (const Grid& g : scope.grids) { candidates.push_back(candidatesOf(card, scope, g.kind)); }

  Model model{card};
  std::mt19937_64 rng{settings.seed};
  Publisher publisher{settings.publishTo, settings.header};
  LoopResult out;
  u32 sinceRegret = settings.regretEvery;
  std::string coverage = "not yet summed";
  std::optional<CoverageSummary> covered;
  std::string lastReport;
  bool endPhase = false;
  bool auditDone = false;
  u32 items = 0;

  while (!device.stopped() && !Signal::stopRequested() && (!settings.maxItems || items < settings.maxItems)) {
    device.keepAnchor();
    if (device.stopped()) { break; }

    Readings const readings{db, env, card};
    model.fit(readings.observations());
    Publication const publication = publish(scope, candidates, readings);
    out.T = publication.T();
    publisher.write(publication, out.regret);
    if (settings.watch) { settings.watch->state(stateOf(publication, readings, model, db, env, coverage, out)); }

    std::map<const Candidate*, RegisterOffer> const offers = registerOffers(db, env, card, candidates, readings);
    RegisterOffer const none{};
    auto filter = [&](const Candidate& c) {
      auto const it = offers.find(&c);
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
                                 .newFloor = settings.newFloor,
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
    u64 const began = device.now();

    // The budget counts every session of the campaign, so a run resumed with a larger one goes on where it stopped.
    out.elapsed = laneTime(db, env).total();
    if (settings.budget > 0 && out.elapsed >= settings.budget) {
      out.budgetSpent = true;
      // Its last word, from where it stands now: the report is otherwise written after the checks below.
      lastReport = reportJson({.version = VERSION,
                               .settings = settings.settings,
                               .state = "budget spent",
                               .elapsed = out.elapsed,
                               .budget = settings.budget,
                               .T = publication.T(),
                               .regret = out.regret,
                               .lanes = laneTime(db, env),
                               .coverage = covered,
                               .publication = &publication,
                               .audit = auditSummary(db, env, readings, publication, true)});
      writeReport(settings.publishTo, lastReport);
      for (const std::string& line : auditReport(auditSummary(db, env, readings, publication, true), true)) {
        log("tune: %s\n", line.c_str());
      }
      break;
    }
    if (settings.budget > 0 && !endPhase && out.elapsed >= settings.budget * (1 - settings.reserve)) {
      endPhase = true;
      log("tune: %s of the %s budget used; the last %s are for confirming what is published\n",
          durationText(out.elapsed).c_str(), durationText(settings.budget).c_str(),
          durationText(settings.budget * settings.reserve).c_str());
    }

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
      covered = summarize(card, candidates, readings, filter);
      coverage = toString(*covered);
      log("tune: coverage: %s\n", coverage.c_str());
      log("tune: lanes: %s\n", toString(laneTime(db, env)).c_str());
      if (settings.watch) { settings.watch->state(stateOf(publication, readings, model, db, env, coverage, out)); }
      if (settings.stop > 0 && out.regret <= settings.stop) {
        out.converged = true;
        break;
      }
    }

    // Where the run stands, rewritten each item; the coverage as of the last regret estimate.
    auto report = [&](const char* state, bool budgetSpent) {
      lastReport = reportJson({.version = VERSION,
                               .settings = settings.settings,
                               .state = state,
                               .elapsed = out.elapsed,
                               .budget = settings.budget,
                               .T = publication.T(),
                               .regret = out.regret,
                               .lanes = laneTime(db, env),
                               .coverage = covered,
                               .publication = &publication,
                               .audit = auditSummary(db, env, readings, publication, budgetSpent)});
      writeReport(settings.publishTo, lastReport);
    };
    report("running", false);

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
    if (endPhase && !auditDone) {
      proposal = endStep(context, publication, db, env, device.session(), began, settings.explore, out.elapsed,
                         settings.budget, settings.target, rng);
      if (!proposal && auditState(db, env, readings).declared) {
        auditDone = true;
        for (const std::string& line : auditReport(auditSummary(db, env, readings, publication, false), false)) {
          log("tune: %s\n", line.c_str());
        }
      }
    }
    LaneTime const time = laneTime(db, env);
    double elapsed = 0;
    for (double const s : time.seconds) { elapsed += s; }
    for (Lane const lane : laneOrder(settings.lanes, time)) {
      if (proposal) { break; }
      if (!settings.lanes.serves(lane)) { continue; }
      switch (lane) {
      case Lane::Model: proposal = modelStep(); break;
      case Lane::Coverage: proposal = coverageStep(context, settings.explore, elapsed, rng); break;
      case Lane::Restarts: proposal = restartStep(context, settings.explore, elapsed, db, env, rng); break;
      case Lane::Interactions: proposal = interactionStep(context, settings.explore, elapsed, rng); break;
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

    recordItem(proposal->record.empty() ? std::string{toString(proposal->lane)} : proposal->record, proposal->what,
               proposal->packet, proposal->ordinal, proposal->candidate->subject.fft, proposal->candidate->subject.kind,
               proposal->options, began);
    double const meanUs = std::exp(model.predict(model.mean(), proposal->candidate->subject, proposal->options));
    (void)db.add(PredRow{.sess = device.session(), .mean = meanUs, .draw = proposal->predictedUs, .ts = began});
    callOf(*proposal->candidate, proposal->options,
           {.n = items, .kind = "call", .predicted = proposal->predictedUs, .mean = meanUs, .why = proposal->why});
  }
  // A run stopped otherwise says so: the report written last, its state replaced.
  if (!out.budgetSpent && !lastReport.empty() && !settings.publishTo.empty()) {
    std::string const running = "\"state\": \"running\"";
    if (size_t const at = lastReport.find(running); at != std::string::npos) {
      lastReport.replace(at, running.size(),
                         std::string{"\"state\": \""} + (out.converged ? "converged" : "stopped") + '"');
    }
    writeReport(settings.publishTo, lastReport);
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
                                    {.stop = command.budget > 0 ? 0 : command.stop,
                                     .sweep = command.sweep,
                                     .lanes = command.lanes,
                                     .explore = command.explore,
                                     .budget = command.budget,
                                     .reserve = command.reserve,
                                     .target = command.target,
                                     .newFloor = command.newFloor,
                                     .seed = seed,
                                     .tailQueues = command.tailQueues,
                                     .publishTo = fs::current_path(),
                                     .header = headerFor(*scope, command),
                                     .settings = settingsWord(*scope, command),
                                     .watch = dashboard ? &*dashboard : nullptr});
  dashboard.reset();

  log("tune: %u calls and %u accuracy readings; T %.2f us/it; expected regret %s%s\n", result.calls,
      result.gateReadings, result.T, percent(result.regret).c_str(),
      result.converged       ? ", below the stop"
        : result.budgetSpent ? ", the budget spent"
        : bench.deviceLost() ? ", stopped by a lost device"
                             : "");
  if (command.budget > 0) {
    log("tune: %s of the %s budget used\n", durationText(result.elapsed).c_str(), durationText(command.budget).c_str());
  }
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
