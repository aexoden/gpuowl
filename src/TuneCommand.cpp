// Copyright (C) Jason Lynch

#include "TuneCommand.h"

#include "TuneDB.h"

#include <cmath>
#include <limits>
#include <optional>

namespace tune {

namespace {

// Upstream's own tuner's option words (tune.cpp), which took -tune before this tuner did: a word of its given here is
// a command meant for -oldtune.
bool isUpstreamWord(std::string_view token) {
  std::string_view const key = token.substr(0, token.find('='));
  for (std::string_view const word : {"noconfig", "inplace", "fp64", "ntt", "nofp32", "fp32", "fp6431", "1k256", "m61",
                                      "pfa", "quick", "minexp", "maxexp", "exp", "regs", "variants", "fine", "tk"}) {
    if (key == word) { return true; }
  }
  return false;
}

std::vector<std::string_view> splitOn(std::string_view text, char separator) {
  std::vector<std::string_view> out;
  for (size_t at = 0; at <= text.size();) {
    size_t const end = std::min(text.find(separator, at), text.size());
    out.push_back(text.substr(at, end - at));
    at = end + 1;
  }
  return out;
}

void parseWorkload(std::string_view text, ScopeArgs& out) {
  size_t const dash = text.find('-');
  out.lo = parseExponent(text.substr(0, dash), "workload=");
  out.hi = dash == std::string_view::npos ? out.lo : parseExponent(text.substr(dash + 1), "workload=");
  if (out.hi < out.lo) { throw std::string{"-tune: workload= is empty: its first exponent is the larger"}; }
}

void parseKinds(std::string_view text, ScopeArgs& out) {
  out.kinds.clear();
  for (std::string_view const one : splitOn(text, '+')) {
    std::optional<TestKind> const kind = parseTestKind(one);
    if (!kind) { throw "-tune: kinds= does not know '" + std::string{one} + "'. Accepted: prp, ll, prp+ll"; }
    if (std::ranges::find(out.kinds, *kind) == out.kinds.end()) { out.kinds.push_back(*kind); }
  }
}

// A percentage of T, or a bare 0.  A bare fraction is refused rather than guessed at: 0.1 could mean 0.1% or 10%.
double parseStop(std::string_view text) {
  bool const percent = text.ends_with('%');
  std::optional<double> const value = parseDouble(percent ? text.substr(0, text.size() - 1) : text);
  if (!value || *value < 0 || *value >= 100 || (!percent && *value != 0)) {
    throw std::string{"-tune: stop= takes a percentage below 100%, such as stop=0.1%, or 0 to run until stopped"};
  }
  return *value / 100;
}

std::string percentText(double fraction) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%g%%", fraction * 100);
  return fraction == 0 ? "0" : buf;
}

}  // namespace

const char* toString(TuneVerb verb) {
  switch (verb) {
  case TuneVerb::Run: return "run";
  case TuneVerb::Status: return "status";
  case TuneVerb::Emit: return "emit";
  case TuneVerb::Scope: return "scope";
  }
  return "?";
}

TuneCommand parseTuneCommand(std::string_view text) {
  TuneCommand out;
  std::vector<std::string_view> tokens = text.empty() ? std::vector<std::string_view>{} : splitOn(text, ',');
  if (!tokens.empty()) {
    for (TuneVerb const verb : {TuneVerb::Status, TuneVerb::Emit, TuneVerb::Scope}) {
      if (tokens.front() == toString(verb)) {
        out.verb = verb;
        tokens.erase(tokens.begin());
        break;
      }
    }
  }
  std::string const who = out.verb == TuneVerb::Run ? "-tune" : std::string{"-tune "} + toString(out.verb);

  for (std::string_view const token : tokens) {
    if (isUpstreamWord(token)) {
      throw who + ": '" + std::string{token} + "' is an option of upstream's tuner, which is -oldtune";
    }
    size_t const eq = token.find('=');
    std::string_view const key = token.substr(0, eq);
    std::string_view const value = eq == std::string_view::npos ? std::string_view{} : token.substr(eq + 1);
    if (eq == std::string_view::npos || value.empty()) {
      throw who + ": '" + std::string{token} + "' is not a setting";
    }

    if (key == "workload") {
      parseWorkload(value, out.scope);
    } else if (key == "probe") {
      out.scope.probe = parseExponent(value, "probe=");
    } else if (key == "probeWeight") {
      std::optional<double> const w = parseDouble(value);
      if (!w || *w < 0 || *w > 1) { throw std::string{"-tune: probeWeight= takes a fraction from 0 to 1"}; }
      out.scope.probeWeight = *w;
    } else if (key == "kinds") {
      parseKinds(value, out.scope);
    } else if (key == "pfaShapes") {
      if (value == "ntt") {
        out.scope.pfaShapes = PfaShapes::Ntt;
      } else if (value == "all") {
        out.scope.pfaShapes = PfaShapes::All;
      } else if (value == "none") {
        out.scope.pfaShapes = PfaShapes::None;
      } else {
        throw std::string{"-tune: pfaShapes= takes ntt, all or none"};
      }
    } else if (key == "tailQueues") {
      std::optional<TailQueues> const q = parseTailQueues(value);
      if (!q) { throw std::string{"-tune: tailQueues= takes ntt, all or none"}; }
      out.tailQueues = *q;
    } else if (key == "stop" && out.verb != TuneVerb::Scope) {
      out.stop = parseStop(value);
    } else if (key == "sweep" && out.verb == TuneVerb::Run) {
      std::optional<double> const share = parseDouble(value);
      if (!share || *share < 0 || *share > 1) { throw std::string{"-tune: sweep= takes a fraction from 0 to 1"}; }
      out.sweep = *share;
    } else if (key == "lanes" && out.verb == TuneVerb::Run) {
      std::optional<Timetable> const lanes = parseTimetable(value);
      if (!lanes) {
        throw std::string{"-tune: lanes= takes model, or shares of model/coverage/restarts/interactions adding up to "
                          "100, as 70/20/10/0 or 70/20/10/0~30/30/20/20@4h"};
      }
      out.lanes = *lanes;
    } else if (key == "contenders" && out.verb == TuneVerb::Run) {
      if (value == "all") {
        out.explore.contenders = std::numeric_limits<double>::infinity();
      } else {
        std::optional<double> const margin =
          value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1)) : std::optional<double>{};
        if (!margin || *margin < 0) {
          throw std::string{"-tune: contenders= takes a percentage, such as contenders=15%, or all"};
        }
        out.explore.contenders = *margin / 100;
      }
    } else if (key == "randomReach" && out.verb == TuneVerb::Run) {
      std::optional<Reach> const reach = parseReach(value);
      if (!reach) {
        throw std::string{"-tune: randomReach= takes a factor of at least 1 and the time it takes to double, such as "
                          "randomReach=1.5x@30m, or all"};
      }
      out.explore.reach = *reach;
    } else if (key == "dashboard" && out.verb == TuneVerb::Run) {
      if (value != "0" && value != "1") { throw std::string{"-tune: dashboard= takes 0 or 1"}; }
      out.dashboard = value == "1";
    } else if (key == "seed" && out.verb == TuneVerb::Run) {
      std::optional<u64> const seed = parseInt<u64>(value);
      if (!seed) { throw std::string{"-tune: seed= takes a number"}; }
      out.seed = *seed;
    } else {
      throw who + ": '" + std::string{key} + "' is not a setting it takes";
    }
  }

  if (out.scope.lo && out.scope.probe && (out.scope.probe < out.scope.lo || out.scope.probe > out.scope.hi)) {
    throw "-tune: probe=" + std::to_string(out.scope.probe) + " is outside workload=" + std::to_string(out.scope.lo) +
      '-' + std::to_string(out.scope.hi);
  }
  return out;
}

std::string settingsWord(const RunScope& scope, const TuneCommand& command) {
  std::string kinds;
  for (const Grid& g : scope.grids) { kinds += (kinds.empty() ? "" : "+") + std::string{toString(g.kind)}; }
  char weight[32];
  snprintf(weight, sizeof(weight), "%g", scope.probeWeight);
  return "workload=" + std::to_string(scope.lo) + '-' + std::to_string(scope.hi) +
    ",probe=" + std::to_string(scope.probe) + ",probeWeight=" + weight + ",kinds=" + kinds +
    ",pfaShapes=" + toString(scope.pfaShapes) + ",tailQueues=" + toString(command.tailQueues) +
    ",stop=" + percentText(command.stop);
}

}  // namespace tune
