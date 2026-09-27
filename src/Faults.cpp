// Copyright (C) Jason Lynch

#include "Faults.h"

#include "log.h"
#include "Stats.h"
#include "UseResolve.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace tune {

std::vector<Fault> faultsOf(const TuneDB& db, u32 env) {
  std::map<u32, u32> gens;
  for (const SessRow& s : db.sessions()) { gens[s.id] = s.gen; }
  auto optionsOf = [&](u32 cfg) {
    const UseConfig* const options = db.findCfg(cfg);
    return options ? configText(*options) : std::string{"?"};
  };

  std::vector<Fault> out;
  std::set<std::tuple<Fault::What, std::string, TestKind, u64, std::string>> seen;
  auto add = [&](Fault f) {
    if (seen.emplace(f.what, f.fft, f.kind, f.exponent, f.options).second) { out.push_back(std::move(f)); }
  };

  for (const TryRow& row : db.diedHolding()) {
    if (db.envOf(row.sess) != env) { continue; }
    add({.what = Fault::What::Lost,
         .fft = row.fft,
         .kind = row.kind,
         .exponent = row.exponent,
         .options = optionsOf(row.cfg),
         .ts = row.ts,
         .gen = gens[row.sess]});
  }
  for (const RunRow& row : db.mergedRuns()) {
    if (db.envOf(row.sess) != env) { continue; }
    if (row.m.status != Status::Err && row.m.status != Status::Lost) { continue; }
    add({.what = row.m.status == Status::Err ? Fault::What::Wrong : Fault::What::Lost,
         .fft = row.fft,
         .kind = row.kind,
         .exponent = row.exponent,
         .options = optionsOf(row.cfg),
         .ts = row.m.ts,
         .gen = gens[row.sess]});
  }

  std::ranges::stable_sort(out, {}, &Fault::ts);
  return out;
}

std::string reproduce(const Fault& f) {
  std::string const use = f.options == "-" ? "" : " -use " + f.options;
  return "-fft " + f.fft + use + "   (" + toString(f.kind) + " at " + std::to_string(f.exponent) + ")";
}

const char* faultText(Fault::What what) {
  return what == Fault::What::Lost ? "took the device down" : "computed a wrong answer";
}

std::vector<std::string> faultLines(const std::vector<Fault>& faults) {
  std::vector<std::string> out;
  for (Fault::What const what : {Fault::What::Lost, Fault::What::Wrong}) {
    auto const n = std::ranges::count(faults, what, &Fault::what);
    if (!n) { continue; }
    out.push_back(std::to_string(n) + (n == 1 ? " configuration " : " configurations ") + faultText(what) +
                  (what == Fault::What::Lost ? " and will not be built again on this device; a kernel or driver bug, "
                                               "worth reporting with the line that reproduces it:"
                                             : " and is held out of what is published; a kernel or driver bug, worth "
                                               "reporting with the line that reproduces it:"));
    for (const Fault& f : faults) {
      if (f.what == what) { out.push_back("  " + reproduce(f)); }
    }
  }
  return out;
}

void logFaults(const std::vector<Fault>& faults, const char* prefix) {
  for (const std::string& line : faultLines(faults)) { log("%s%s\n", prefix, line.c_str()); }
}

}  // namespace tune
