// Copyright (C) Jason Lynch

#include "Production.h"

#include "Args.h"
#include "Gate.h"
#include "log.h"
#include "TuneDB.h"

#include <algorithm>
#include <mutex>
#include <set>
#include <utility>

namespace tune {

namespace {

// Each of these lines is a standing fact about the configuration rather than news about the exponent in hand, so it is
// said once however many tasks a run works through.
void logOnce(const std::string& line) {
  static std::mutex mutex;
  static std::set<std::string> said;

  std::lock_guard const lock{mutex};
  if (said.insert(line).second) { log("%s", line.c_str()); }
}

std::string joined(const std::vector<std::string>& keys) {
  std::string s;
  for (const std::string& key : keys) { s += (s.empty() ? "" : ", ") + key; }
  return s;
}

// The keys the config files set, '!' lines included. The command line is deliberate and momentary; these are the ones
// a machine keeps, and the ones upstream's own tuner wrote into config.txt.
std::set<std::string> configFileKeys(const Args& args) {
  std::set<std::string> keys;

  for (const auto& [key, value] : args.flags) {
    if (!args.cliKeys.contains(key)) { keys.insert(key); }
  }

  for (const UseLine& line : args.perFftConfig) {
    for (const auto& [key, value] : line.uses) { keys.insert(key); }
  }

  return keys;
}

}  // namespace

std::optional<FFTConfig> pinnedFft(const Args& args) {
  if (args.fftSpec.empty()) { return {}; }

  // Upstream's own parser, so that "8M", "512:15:512" and a full spec name here exactly what the shape scan would run
  // for them. It reports what it refuses by throwing, which is the shape scan's to do rather than this walk's.
  try {
    return FFTConfig{args.fftSpec};
  } catch (...) { return {}; }
}

Regime runRegime(const Args& args, const FFTConfig& fft, u64 E) {
  Regime regime = regimeOf(fft, E);
  regime.longCarry = regime.longCarry || args.carry == CARRY_64;
  return regime;
}

u64 publishedReach(const SelectionFile& file, const Env& env, const FFTConfig& fft, TestKind kind,
                   const UseConfig& options, u64 E) {
  std::optional<u64> lowest;
  Regime const regime = regimeOf(fft, E);
  UseConfig const rounding = roundingOf(env, fft, options);

  for (const SelectionEntry& entry : file.entries) {
    if (entry.kind != kind || entry.fft != fft.spec() || entry.regime != regime) { continue; }
    if (roundingOf(env, fft, entry.opts) != rounding) { continue; }

    // An entry that holds exactly the table's reach for its band was read there and nowhere past it, which says nothing
    // against a reach another entry of the same arithmetic was measured to above it.
    if (entry.reach == interval(fft, entry.emin).hi) { continue; }

    lowest = std::min(lowest.value_or(entry.reach), entry.reach);
  }

  return lowest.value_or(maxExp(fft));
}

std::vector<std::string> shadowedKeys(const Args& args, const Env& env, const SelectionFile& file,
                                      const SelectionEntry& entry, const FFTConfig& fft) {
  UseConfig const resolved =
    resolveConfig(args, fft, entry.kind, fittedTo(file.layersFor(entry), env, fft, entry.kind));

  std::vector<std::string> out;

  for (const auto& [key, value] : resolved) {
    auto const at = entry.opts.find(key);
    if (at != entry.opts.end() && at->second == value) { continue; }

    const Option* const option = findOption(key);
    if (!option || option->appliesTo(env, fft, resolved)) { out.push_back(key); }
  }

  return out;
}

std::optional<Choice> chooseFrom(const SelectionFile& file, const Args& args, const Env& env, u64 E, TestKind kind) {
  std::optional<FFTConfig> const pinned = pinnedFft(args);

  // An -fft spec that does not parse is the shape scan's to refuse, in the words it already refuses it in.
  if (!args.fftSpec.empty() && !pinned) { return {}; }

  std::optional<Choice> best;

  for (const SelectionEntry& entry : file.entries) {
    if (entry.kind != kind || E < entry.emin) { continue; }
    if (best && entry.cost >= best->entry->cost) { continue; }

    auto const fft = parseFft(entry.fft);
    if (!fft) { continue; }
    if (pinned && pinned->spec() != fft->spec()) { continue; }

    // Only reachable where -fftOverdrive has carried E past the interval the entry covers, which can leave it running
    // kernels the entry says nothing about.
    if (regimeOf(*fft, E) != entry.regime) { continue; }

    std::vector<std::string> shadowed = shadowedKeys(args, env, file, entry, *fft);

    // -carry long runs the expanded carry kernels whatever the bits per word say, so an entry measured in a short
    // regime did not measure what is about to run -- the same kind of difference as an overridden key, and named the
    // same way.
    if (runRegime(args, *fft, E) != entry.regime) { shadowed.insert(shadowed.begin(), "-carry"); }

    UseConfig options = resolveConfig(args, *fft, kind, fittedTo(file.layersFor(entry), env, *fft, kind));

    // Every entry of the arithmetic that will actually run holds it to the lowest reach any of them measured, whether
    // or not this one was overridden. An override also costs whatever the options it was measured under bought above
    // the table, conservative whether they raised the reach or lowered it.
    u64 reach = std::min(entry.reach, publishedReach(file, env, *fft, kind, options, entry.reach));
    if (!shadowed.empty()) { reach = std::min(reach, maxExp(*fft)); }
    if (double(E) > double(reach) * args.fftOverdrive) { continue; }

    best = Choice{
      .fft = *fft, .options = std::move(options), .entry = entry, .shadowed = std::move(shadowed), .reach = reach};
  }

  return best;
}

fs::path selectionPath(const Args& args) {
  fs::path path = "selection.txt";
  if (!fs::exists(path) && !args.masterDir.empty()) { path = args.masterDir / "selection.txt"; }
  return path;
}

Choice choose(const Args& args, const Env& env, u64 E, TestKind kind) {
  // Read afresh for each task rather than cached: the file is small, a task is not, and a tuning run publishing beside
  // a production run is then answered with what it has published rather than with what it had at startup.
  std::optional<SelectionFile> const file = readSelection(selectionPath(args));

  if (file) {
    if (std::optional<Choice> chosen = chooseFrom(*file, args, env, E, kind)) {
      if (!chosen->shadowed.empty()) {
        logOnce("Selection entry " + chosen->entry->id + " (" + chosen->fft.spec() + ") was measured with " +
                joined(chosen->shadowed) + " set otherwise, so it runs only to " + to_string(chosen->reach) +
                " rather than to the " + to_string(chosen->entry->reach) + " it measured\n");
      }

      if (double(E) > double(chosen->reach)) {
        logOnce("Warning: " + chosen->fft.spec() + " runs " + to_string(E) + " past its validated reach " +
                to_string(chosen->reach) + ", as -fftOverdrive asks\n");
      }

      return std::move(*chosen);
    }
  }

  SelectionLayers layers;
  if (file) { layers = SelectionLayers{.global = file->global, .family = file->family, .entry = {}}; }

  auto const scan = [&](u64 ask) {
    FFTConfig const fft = FFTConfig::bestFit(args, ask, args.fftSpec, env.hasFP64);
    UseConfig options = resolveConfig(args, fft, kind, fittedTo(layers, env, fft, kind));
    u64 const limit = file ? publishedReach(*file, env, fft, kind, options, E) : maxExp(fft);

    return Choice{.fft = fft, .options = std::move(options), .entry = {}, .shadowed = {}, .reach = limit};
  };

  // The shape scan knows nothing about what has been published, so a configuration whose own measurement stopped short
  // of the fitted table can come back out of it. Where it does, ask again for something whose table limit is past this
  // one's -- necessarily a different and larger FFT -- rather than run past a limit this machine measured.
  u64 ask = E;
  std::optional<Choice> restricted;

  for (int attempt = 0; attempt < 8; ++attempt) {
    std::optional<Choice> candidate;

    try {
      candidate = scan(ask);
    } catch (...) {
      // The first ask failing is the shape scan refusing the exponent, which is its own answer to give. A later one
      // failing means there is nothing larger to move to, and the restricted answer is the only one there is.
      if (!restricted) { throw; }
      break;
    }

    u64 const table = maxExp(candidate->fft);

    if (candidate->reach >= table || double(E) <= double(candidate->reach) * args.fftOverdrive) { return *candidate; }

    std::string const said = candidate->fft.spec() + " is published as reaching only " + to_string(candidate->reach) +
      " under the options this run resolves for it";

    // One shape was named, so there is nothing else to ask for; upstream warns and runs a spec that is too small for
    // the exponent, and an explicit choice is no less explicit for having been measured.
    if (!args.fftSpec.empty()) {
      logOnce("Warning: " + said + ", and " + to_string(E) + " is past that\n");
      return *candidate;
    }

    logOnce("Note: " + said + "; looking past it for " + to_string(E) + "\n");
    restricted = std::move(candidate);

    if (table + 1 <= ask) { break; }
    ask = table + 1;
  }

  log("Warning: every FFT the scan offers for %s is published as reaching less far; running %s\n", to_string(E).c_str(),
      restricted->fft.spec().c_str());
  return std::move(*restricted);
}

void reportShadowing(const Args& args, const Env& env) {
  std::optional<SelectionFile> const file = readSelection(selectionPath(args));
  if (!file || file->entries.empty()) { return; }

  std::set<std::string> const fromConfig = configFileKeys(args);
  std::set<std::string> shadowing;

  for (const SelectionEntry& entry : file->entries) {
    auto const fft = parseFft(entry.fft);
    if (!fft) { continue; }

    for (const std::string& key : shadowedKeys(args, env, *file, entry, *fft)) {
      if (fromConfig.contains(key)) { shadowing.insert(key); }
    }
  }

  if (shadowing.empty()) { return; }

  log("Note: the config files set %s, which shadow what %s publishes; remove them to run what was measured\n",
      joined({shadowing.begin(), shadowing.end()}).c_str(), selectionPath(args).string().c_str());
}

}  // namespace tune
