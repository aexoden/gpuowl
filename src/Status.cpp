// Copyright (C) Jason Lynch

#include "Status.h"

#include "Bootstrap.h"
#include "Eligibility.h"
#include "Emit.h"
#include "log.h"
#include "Objective.h"

#include <cinttypes>
#include <ctime>
#include <limits>

namespace tune {

namespace {

[[nodiscard]] std::string percent(double fraction) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.3g%%", fraction * 100);
  return buf;
}

[[nodiscard]] std::string localTime(u64 ts) {
  std::time_t const t = std::time_t(ts);
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &t);
#else
  localtime_r(&t, &local);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &local);
  return buf;
}

[[nodiscard]] Attempt attemptOf(const TuneDB& db, const TryRow& row) {
  const UseConfig* const options = db.findCfg(row.cfg);
  return {.row = row, .options = options ? configText(*options) : "?"};
}

[[nodiscard]] std::string describe(const SessRow& s) {
  return "session " + std::to_string(s.id) + (s.tune.empty() ? "," : ", a run") + " started " + localTime(s.start) +
    (s.gen ? ", generation " + std::to_string(s.gen) : "");
}

// "-fft <spec> -use <options>", which reproduces the attempt, and where it was made.
[[nodiscard]] std::string reproduce(const Attempt& a) {
  std::string const use = a.options == "-" ? "" : " -use " + a.options;
  return "-fft " + a.row.fft + use + "   (" + toString(a.row.kind) + " at " + std::to_string(a.row.exponent) + ")";
}

}  // namespace

std::string_view completeLines(std::string_view text) {
  size_t const end = text.rfind('\n');
  return end == std::string_view::npos ? std::string_view{} : text.substr(0, end + 1);
}

std::string ago(u64 seconds) {
  if (seconds < 120) { return std::to_string(seconds) + " s"; }
  if (seconds < 120 * 60) { return std::to_string(seconds / 60) + " min"; }
  if (seconds < 48 * 3600) { return std::to_string(seconds / 3600) + " h"; }
  return std::to_string(seconds / 86'400) + " days";
}

TuneStatus statusOf(const Scheduler& scheduler, TuneDB& db, u32 env, double stop, const Activity& activity) {
  TuneStatus out;
  out.activity = activity;

  const SessRow* newest = nullptr;
  for (const SessRow& s : db.sessions()) {
    if (!newest || s.id > newest->id) { newest = &s; }
    if (s.env == env && (!out.latest || s.id > out.latest->id)) { out.latest = s; }
  }

  // Read from outside, every open attempt looks like one a process died holding, the running one's included.
  for (const TryRow& row : db.diedHolding()) {
    if (activity.held == true && newest && row.sess == newest->id) {
      out.measuring = attemptOf(db, row);
    } else if (db.envOf(row.sess) == env) {
      out.faults.push_back(attemptOf(db, row));
    }
  }

  // Answered here, as the running process will answer it: otherwise the queue would rule its entry out as a death.
  if (out.measuring && !db.attached()) { (void)db.add(DoneRow{.sess = out.measuring->row.sess, .ts = activity.now}); }

  BootstrapState const state = scheduler.bootstrapState(db, env);
  Objective const published{db, env, scheduler.scope()};
  Objective const valuing{db, env, scheduler.scope(), Gating::Assumed};
  out.T = published.T();
  out.measured = published.measured();
  out.valuedT = valuing.T();
  out.stop = stop;
  out.floor = stop * valuing.T();

  QueueReport report;
  report.left = scheduler.admissible(db, env, valuing, out.floor);
  report.valuedT = out.valuedT;
  report.floor = out.floor;
  RunSummary summary = summarize(scheduler, db, env, out.latest ? out.latest->id : 0, report, stop);
  out.families = std::move(summary.families);
  out.heldBy = std::move(summary.heldBy);
  out.phase = scheduler.phase(state, report.left, published, out.floor).text;

  for (const Item& item : report.left) {
    if (!worthRunning(item, out.floor)) {
      ++out.notWorth;
      continue;
    }
    out.unlisted += item.unlisted;
    if (out.next.size() < STATUS_NEXT) {
      out.next.push_back({.item = item, .label = itemLabel(scheduler, item)});
    } else {
      ++out.moreWorth;
    }
  }

  TuneStatus::Accuracy& a = out.accuracy;
  if (std::optional<SelectionFile> const file =
        emit(db, scheduler.lines(db, env), Provenance{.ts = 0, .db = {}, .env = env})) {
    for (const SelectionEntry& e : file->entries) {
      ++a.entries;
      a.exact += e.evidence == Evidence::NotApplicable;
      a.confirmed += e.evidence == Evidence::Confirmed;
      a.unvalidated += e.evidence == Evidence::Unvalidated;

      std::optional<FFTConfig> const fft = parseFft(e.fft);
      if (!fft) { continue; }
      u64 const table = interval(*fft, e.emin).hi;
      a.belowTable += e.reach < table;
      a.aboveTable += e.reach > table;
    }
    a.limits = u32(file->limits.size());
  }
  a.owed = u32(gatesOwed(db, env).size());
  a.rejected = u32(rejectedSets(db, env).size());

  for (Fault& f : faultsOf(db, env)) {
    if (f.what == Fault::What::Wrong) { out.wrong.push_back(std::move(f)); }
  }
  return out;
}

void logStatus(const TuneStatus& s, const DbEnv& env, const std::string& valuedAs) {
  const Activity& activity = s.activity;
  std::string const written = activity.written && activity.now >= activity.written
    ? "; tunedb.txt last written " + ago(activity.now - activity.written) + " ago"
    : "";
  log("tune: status: env %u, %s (%s)%s\n", env.id, env.gpu.empty() ? "unnamed device" : env.gpu.c_str(),
      env.toEnv().label().c_str(), written.c_str());

  if (activity.held == true) {
    if (s.measuring) {
      const TryRow& row = s.measuring->row;
      std::string const since =
        row.ts && activity.now >= row.ts ? ", for " + ago(activity.now - row.ts) + " so far" : "";
      log("tune: status: a process holds the database, measuring %s%s\n", reproduce(*s.measuring).c_str(),
          since.c_str());
    } else {
      log("tune: status: a process holds the database\n");
    }
  } else if (activity.held == false) {
    log("tune: status: nothing holds the database\n");
  }
  if (s.latest) { log("tune: status: the latest session on env %u is %s\n", env.id, describe(*s.latest).c_str()); }

  log("tune: status: valued as %s\n", valuedAs.c_str());
  log("tune: status: T %.3f us/it, %.1f%% of the weight on measured entries\n", s.T, s.measured * 100);
  if (!s.phase.empty()) { log("tune: status: a run started now would begin in %s\n", s.phase.c_str()); }

  for (const RunSummary::Family& f : s.families) {
    log("tune: status: %s: %s\n", typeName(f.type), familyCounts(f, s.heldBy).c_str());
  }

  const TuneStatus::Accuracy& a = s.accuracy;
  log("tune: status: accuracy: %u %s published (%u exact arithmetic, %u confirmed, %u unvalidated); %u held below the "
      "table's reach, %u raised above it; %u %s for options no entry runs; %u %s owed a reading, %u rejected\n",
      a.entries, a.entries == 1 ? "entry" : "entries", a.exact, a.confirmed, a.unvalidated, a.belowTable, a.aboveTable,
      a.limits, a.limits == 1 ? "limit" : "limits", a.owed, a.owed == 1 ? "set" : "sets", a.rejected);

  std::string const threshold = s.stop > 0 ? "stop=" + percent(s.stop) + " of T" : "anything";
  if (s.next.empty()) {
    log("tune: status: nothing is worth %s, so a run started now would end at once\n", threshold.c_str());
  } else {
    log("tune: status: next, as a run started now would rank them:\n");
    for (size_t i = 0; i < s.next.size(); ++i) {
      const Item& item = s.next[i].item;
      // What runs by rule is priced at the least a price can be where nothing else prices it.
      std::string worth = byRule(item) ? "by rule" : "";
      if (!byRule(item) || item.value > std::numeric_limits<double>::min()) {
        char buf[96];
        snprintf(buf, sizeof(buf), "worth %.4f us/it (%s of T)", item.value,
                 percent(s.valuedT > 0 ? item.value / s.valuedT : 0).c_str());
        worth += (worth.empty() ? "" : ", ") + std::string{buf};
      }
      log("tune: status:   %zu. %s %s: %s, ~%.0f s\n", i + 1, toString(item.kind), s.next[i].label.c_str(),
          worth.c_str(), item.seconds);
    }
    if (s.moreWorth || s.notWorth) {
      char below[96] = "worth nothing";
      if (s.stop > 0) { snprintf(below, sizeof(below), "worth less than %s (%.4f us/it)", threshold.c_str(), s.floor); }
      log("tune: status:   and %u more worth running, %u %s\n", s.moreWorth, s.notWorth, below);
    }
    if (s.unlisted) { log("tune: status:   and up to %" PRIu64 " more not listed yet\n", s.unlisted); }
  }

  if (!s.faults.empty()) {
    log("tune: status: %zu %s held out, having been measured when the device went away ('-tune reset,fft=<spec>' "
        "drops everything measured on that shape, these among it):\n",
        s.faults.size(), s.faults.size() == 1 ? "configuration is" : "configurations are");
    for (const Attempt& fault : s.faults) { log("tune: status:   %s\n", reproduce(fault).c_str()); }
  }
  if (!s.wrong.empty()) { logFaults(s.wrong, "tune: status: "); }
}

}  // namespace tune
