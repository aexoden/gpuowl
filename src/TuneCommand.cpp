// Copyright (C) Jason Lynch

#include "TuneCommand.h"

#include "TuneDB.h"

#include <algorithm>
#include <cctype>
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

// ",powerHold=<f>" or ",powerHold=off" where the setting is not the default, which a settings word leaves out.
std::string powerHoldText(const TuneCommand& command) {
  if (command.powerHold == POWER_HOLD) { return {}; }
  if (command.powerHold == 0) { return ",powerHold=off"; }
  char buf[32];
  snprintf(buf, sizeof(buf), ",powerHold=%g", command.powerHold);
  return buf;
}

// ",heat=<t>s,blocks=<n>,heater=off,compileAhead=off" for whichever is not the default: how a call was timed, which its readings depend
// on.
std::string callShapeText(const TuneCommand& command) {
  std::string out;
  char buf[32];
  if (command.callHeat != CALL_HEAT_SEC) {
    snprintf(buf, sizeof(buf), ",heat=%gs", command.callHeat);
    out += buf;
  }
  if (command.callBlocks != BLOCKS_PER_CALL) { out += ",blocks=" + std::to_string(command.callBlocks); }
  if (!command.heater) { out += ",heater=off"; }
  if (!command.compileAhead) { out += ",compileAhead=off"; }
  return out;
}

// ",within=<p>%|all,top=<n>|all" for whichever is not the default: what the contender table shows.
std::string contestText(const TuneCommand& command) {
  std::string out;
  if (command.contest.within != WITHIN) {
    out += ",within=" + (std::isinf(command.contest.within) ? std::string{"all"} : percentText(command.contest.within));
  }
  if (command.contest.top != TOP) {
    out += ",top=" + (command.contest.top == SIZE_MAX ? std::string{"all"} : std::to_string(command.contest.top));
  }
  return out;
}

}  // namespace

const char* toString(TuneVerb verb) {
  switch (verb) {
  case TuneVerb::Run: return "run";
  case TuneVerb::Status: return "status";
  case TuneVerb::Emit: return "emit";
  case TuneVerb::Scope: return "scope";
  case TuneVerb::Explain: return "explain";
  }
  return "?";
}

namespace {

// ",search=staged,margin=<p>%,restarts=<p>%" for a staged run, each setting only where it is not the default: how the
// run chose what to measure, which a session's readings do not depend on, but its record should say.
std::string searchText(const TuneCommand& command) {
  if (command.search != Search::Staged) { return {}; }
  std::string out = ",search=staged";
  if (command.staged.margin != MARGIN) {
    out += ",margin=" + (std::isinf(command.staged.margin) ? std::string{"all"} : percentText(command.staged.margin));
  }
  if (command.staged.restarts != RESTARTS) { out += ",restarts=" + percentText(command.staged.restarts); }
  return out;
}

}  // namespace

TuneCommand parseTuneCommand(std::string_view text) {
  TuneCommand out;
  std::vector<std::string_view> tokens = text.empty() ? std::vector<std::string_view>{} : splitOn(text, ',');
  if (!tokens.empty()) {
    for (TuneVerb const verb : {TuneVerb::Status, TuneVerb::Emit, TuneVerb::Scope, TuneVerb::Explain}) {
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

    if (out.verb == TuneVerb::Explain && key == "fft") {
      if (!std::ranges::all_of(value, [](char c) { return std::isalnum(u8(c)) || c == ':'; })) {
        throw std::string{"-tune explain: fft= takes an FFT spec, such as fft=512:15:256:001, or its shape alone"};
      }
      out.explain.fft = value;
    } else if (out.verb == TuneVerb::Explain && std::isupper(u8(key.front()))) {
      // A -use option: tune settings are lowercase, option keys upper.
      if (!out.explain.options) { out.explain.options.emplace(); }
      (*out.explain.options)[std::string{key}] = value;
    } else if (key == "workload") {
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
    } else if (key == "within" && (out.verb == TuneVerb::Run || out.verb == TuneVerb::Status)) {
      std::optional<double> const p =
        value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1)) : std::optional<double>{};
      if (value == "all") {
        out.contest.within = std::numeric_limits<double>::infinity();
      } else if (!p || *p < 0) {
        throw who + ": within= takes a percentage, such as within=5%, or all";
      } else {
        out.contest.within = *p / 100;
      }
    } else if (key == "top" && (out.verb == TuneVerb::Run || out.verb == TuneVerb::Status)) {
      std::optional<u32> const n = value == "all" ? std::optional<u32>{} : parseInt<u32>(value);
      if (value == "all") {
        out.contest.top = SIZE_MAX;
      } else if (!n) {
        throw who + ": top= takes a count of rivals per line, or all";
      } else {
        out.contest.top = *n;
      }
    } else if (key == "search" && out.verb == TuneVerb::Run) {
      if (value != "model" && value != "staged") { throw std::string{"-tune: search= takes model or staged"}; }
      out.search = value == "staged" ? Search::Staged : Search::Model;
    } else if (key == "margin" && out.verb == TuneVerb::Run) {
      std::optional<double> const margin = value == "all" ? std::optional<double>{INFINITY}
        : value.ends_with('%')                            ? parseDouble(value.substr(0, value.size() - 1))
                                                          : std::optional<double>{};
      if (!margin || *margin < 0) { throw std::string{"-tune: margin= takes a percentage, such as margin=25%, or all"}; }
      out.staged.margin = *margin / 100;
    } else if (key == "restarts" && out.verb == TuneVerb::Run) {
      std::optional<double> const share = value == "0" ? std::optional<double>{0}
        : value.ends_with('%')                         ? parseDouble(value.substr(0, value.size() - 1))
                                                       : std::optional<double>{};
      if (!share || *share < 0 || *share > 100) {
        throw std::string{"-tune: restarts= takes a percentage, such as restarts=10%, or 0"};
      }
      out.staged.restarts = *share / 100;
    } else if (key == "budget" && out.verb == TuneVerb::Run) {
      std::optional<double> const budget = value == "0" ? std::optional<double>{0} : parseDuration(value);
      if (!budget) {
        throw std::string{"-tune: budget= takes a span of tuning time, such as budget=2h or budget=90m, or 0"};
      }
      out.budget = *budget;
    } else if (key == "reserve" && out.verb == TuneVerb::Run) {
      std::optional<double> const share =
        value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1)) : std::optional<double>{};
      if (!share || *share < 0 || *share >= 100) {
        throw std::string{"-tune: reserve= takes a percentage below 100%, such as reserve=25%"};
      }
      out.reserve = *share / 100;
    } else if (key == "target" && out.verb == TuneVerb::Run) {
      std::optional<double> const share =
        value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1)) : std::optional<double>{};
      if (!share || *share < 0 || *share >= 100) {
        throw std::string{"-tune: target= takes a percentage, such as target=0.5%"};
      }
      out.target = *share / 100;
    } else if (key == "frontSweep" && out.verb == TuneVerb::Run) {
      std::optional<double> const band = value == "0" ? std::optional<double>{0}
        : value.ends_with('%')                        ? parseDouble(value.substr(0, value.size() - 1))
                                                      : std::optional<double>{};
      if (!band || *band < 0) {
        throw std::string{"-tune: frontSweep= takes a percentage, such as frontSweep=5%, or 0"};
      }
      out.explore.sweepBand = *band / 100;
    } else if (key == "newFloor" && out.verb == TuneVerb::Run) {
      std::optional<double> const floor = parseDouble(value);
      if (!floor || *floor < 0 || *floor > 1) { throw std::string{"-tune: newFloor= takes a fraction from 0 to 1"}; }
      out.newFloor = *floor;
    } else if (key == "restartCut" && out.verb == TuneVerb::Run) {
      if (value == "all") {
        out.explore.restartCut = std::numeric_limits<double>::infinity();
      } else {
        std::optional<double> const margin =
          value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1)) : std::optional<double>{};
        if (!margin || *margin < 0) {
          throw std::string{"-tune: restartCut= takes a percentage, such as restartCut=50%, or all"};
        }
        out.explore.restartCut = *margin / 100;
      }
    } else if (key == "shortCalls" && out.verb == TuneVerb::Run) {
      std::optional<double> const factor = value == "0" ? std::optional<double>{0}
        : value.ends_with('x')                          ? parseDouble(value.substr(0, value.size() - 1))
                                                        : std::optional<double>{};
      if (!factor || (*factor != 0 && *factor < 1)) {
        throw std::string{"-tune: shortCalls= takes a factor of at least 1, such as shortCalls=2x, or 0"};
      }
      out.shortCalls = *factor;
    } else if (key == "randomReach" && out.verb == TuneVerb::Run) {
      std::optional<Reach> const reach = parseReach(value);
      if (!reach) {
        throw std::string{"-tune: randomReach= takes a factor of at least 1 and the time it takes to double, such as "
                          "randomReach=1.5x@30m, or all"};
      }
      out.explore.reach = *reach;
    } else if ((key == "soak" || key == "soakMargin") && out.verb == TuneVerb::Run) {
      std::optional<double> const share = value == "0" ? std::optional<double>{0}
        : value.ends_with('%')                         ? parseDouble(value.substr(0, value.size() - 1))
                                                       : std::optional<double>{};
      if (!share || *share < 0 || *share >= 100) {
        throw "-tune: " + std::string{key} + "= takes a percentage below 100%, such as " + std::string{key} +
          "=10%, or 0";
      }
      (key == "soak" ? out.soak.share : out.soak.margin) = *share / 100;
    } else if ((key == "soakTop" || key == "soakRounds") && out.verb == TuneVerb::Run) {
      std::optional<u32> const n = parseInt<u32>(value);
      if (!n || *n < 1 || *n > 16) { throw "-tune: " + std::string{key} + "= takes a count from 1 to 16"; }
      (key == "soakTop" ? out.soak.top : out.soak.rounds) = *n;
    } else if ((key == "soakWarm" || key == "soakSlice") && out.verb == TuneVerb::Run) {
      std::optional<double> const t = value == "0" ? std::optional<double>{0} : parseDuration(value);
      if (!t || (key == "soakSlice" && *t < 1)) {
        throw "-tune: " + std::string{key} + "= takes a span of time, such as 2m or 90s" +
          (key == "soakWarm" ? ", or 0" : "");
      }
      (key == "soakWarm" ? out.soak.warm : out.soak.slice) = *t;
    } else if ((key == "duel" || key == "duelMargin") && out.verb == TuneVerb::Run) {
      std::optional<double> const share = value == "0" ? std::optional<double>{0}
        : value == "all"                               ? std::optional<double>{key == "duel" ? 100 : INFINITY}
        : value.ends_with('%')                         ? parseDouble(value.substr(0, value.size() - 1))
                                                       : std::optional<double>{};
      if (!share || *share < 0 || (key == "duel" && *share > 100)) {
        throw "-tune: " + std::string{key} + "= takes a percentage, such as " + std::string{key} + "=" +
          (key == "duel" ? "15%, all or 0" : "3%, or all");
      }
      (key == "duel" ? out.duel.share : out.duel.margin) = *share / 100;
    } else if (key == "duelEpsilon" && out.verb == TuneVerb::Run) {
      std::optional<double> const epsilon =
        value == "0" ? std::optional<double>{0}
        : value.ends_with('%') ? parseDouble(value.substr(0, value.size() - 1))
                               : std::optional<double>{};
      if (!epsilon || *epsilon < 0 || *epsilon >= 100) {
        throw std::string{"-tune: duelEpsilon= takes a percentage, such as duelEpsilon=0.3%, or 0"};
      }
      out.duel.epsilon = *epsilon / 100;
    } else if (key == "duelPairs" && out.verb == TuneVerb::Run) {
      std::optional<u32> const n = parseInt<u32>(value);
      if (!n || *n < 2 || *n > 64) { throw std::string{"-tune: duelPairs= takes a count from 2 to 64"}; }
      out.duel.pairs = *n;
    } else if (key == "heat" && out.verb == TuneVerb::Run) {
      std::optional<double> const heat = value == "0" ? std::optional<double>{0} : parseDuration(value);
      if (!heat || *heat > 600) { throw std::string{"-tune: heat= takes a span of time up to 10m, such as 3s, or 0"}; }
      out.callHeat = *heat;
    } else if (key == "blocks" && out.verb == TuneVerb::Run) {
      std::optional<u32> const blocks = parseInt<u32>(value);
      if (!blocks || *blocks < 1 || *blocks > 64) { throw std::string{"-tune: blocks= takes a count from 1 to 64"}; }
      out.callBlocks = *blocks;
    } else if (key == "heater" && out.verb == TuneVerb::Run) {
      if (value != "on" && value != "off") { throw std::string{"-tune: heater= takes on or off"}; }
      out.heater = value == "on";
    } else if (key == "compileAhead" && out.verb == TuneVerb::Run) {
      if (value != "on" && value != "off") { throw std::string{"-tune: compileAhead= takes on or off"}; }
      out.compileAhead = value == "on";
    } else if (key == "powerHold" && out.verb != TuneVerb::Scope) {
      std::optional<double> const hold = value == "off" ? std::optional<double>{0} : parseDouble(value);
      if (!hold || *hold < 0 || *hold > 1) {
        throw std::string{"-tune: powerHold= takes the share of the power limit a card holds, such as 0.95, or off"};
      }
      out.powerHold = *hold;
    } else if (key == "tag") {
      if (!std::ranges::all_of(value, [](char c) { return std::isalnum(u8(c)) || c == '.' || c == '_' || c == '-'; })) {
        throw std::string{"-tune: tag= takes a word of letters, digits, '.', '_' and '-', such as tag=875mhz"};
      }
      out.tag = value;
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

  if (out.verb == TuneVerb::Explain && out.explain.fft.empty()) {
    throw std::string{"-tune explain: fft= names the FFT to explain, such as -tune explain,fft=512:15:256:001"};
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
    ",stop=" + percentText(command.stop) + (command.tag.empty() ? "" : ",tag=" + command.tag) +
    powerHoldText(command) + callShapeText(command) + searchText(command) + contestText(command);
}

}  // namespace tune
