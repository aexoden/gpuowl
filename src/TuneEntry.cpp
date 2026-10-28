#include "TuneEntry.h"
#include "Args.h"
#include "BuildId.h"
#include "CycleFile.h"
#include "Eligibility.h"

#include <cassert>
#include <cmath>
#include <cinttypes>
#include <algorithm>
#include <cctype>
#include <iterator>
#include <optional>
#include <sstream>

// Returns whether *results* was updated.
bool TuneEntry::update(vector<TuneEntry>& results) const {
  if (!std::isfinite(cost)) { return false; }   // a failed timing (see Gpu::timePRP) must never be recorded
  u64 const maxExp = this->maxExp();
  [[maybe_unused]] bool didErase = false;

  int i{};
  for (i = int(results.size()) - 1; i >= 0 && results[i].cost > cost; --i) {
    if (results[i].maxExp() <= maxExp) {
      results.erase(std::next(results.begin(), i));
      didErase = true;
    }
  }

  if (i >= 0 && results[i].maxExp() >= maxExp) {
    assert(!didErase);
    return false;
  }

  results.insert(std::next(results.begin(), i + 1), *this);
  return true;
}

// Returns whether entry *e* represents an improvement over *results* (i.e. would update the results).
bool TuneEntry::willUpdate(const vector<TuneEntry>& results) const {
  u64 const maxExp = this->maxExp();
  for (const auto& r : results) {
    if (r.cost > cost) {
      break;
    } if (r.maxExp() >= maxExp) {
      return false;
    }
  }
  return true;
}

// A file of this name in the run directory, else in the -pool directory; the run directory's where neither has one.
static fs::path findFile(const Args& args, const char* name) {
  if (!fs::exists(name) && fs::exists(args.masterDir / name)) { return args.masterDir / name; }
  return name;
}

static fs::path tuneFilePath(const Args& args, bool ll = false) {
  if (ll) {
    for (const fs::path& path : {fs::path{"tune-ll.txt"}, args.masterDir / "tune-ll.txt"}) {
      if (fs::exists(path)) { return path; }
    }
  }
  fs::path tuneFile = "tune.txt";
  if (!fs::exists(tuneFile)) {
    tuneFile = args.masterDir / "tune.txt";
  }
  return tuneFile;
}

// Lines starting with ';' or '#', and blank lines, are comments: readTuneFile() skips them and writeTuneFile() keeps them
static bool isCommentLine(const string& line) {
  auto const first = line.find_first_not_of(" \t\r\n");
  return first == string::npos || line[first] == ';' || line[first] == '#';
}

// A tune.txt line is "<cost> <fft spec> # <maxExp>[, KEY=VAL]...".  The KEY=VALs are -use settings for that FFT alone,
// taking priority over config.txt (but not over the command line).  A line of the tuner's own file names between the
// spec and the '#' the test kind, the carry regime and the exponent range it holds instead of a maxExp:
// "<cost> <fft spec> <kind> <regime> <lo>-<hi> [# KEY=VAL[, KEY=VAL]...]".
// Returns nothing for a line that is not an entry.  Throws (const char*) for an FFT spec this build can't parse, or words
// between the spec and the '#' it can't read.
static optional<TuneEntry> parseTuneLine(const string& line) {
  char specBuf[32];
  double cost{};
  int specEnd{};
  if (sscanf(line.c_str(), "%lf %31s%n", &cost, specBuf, &specEnd) < 2) { return {}; }
  string kind, regime;
  u64 lo{}, hi{};
  {
    size_t const end = std::min(line.find('#'), line.size());
    std::istringstream words{line.substr(size_t(specEnd), end - std::min(end, size_t(specEnd)))};
    for (string word; words >> word;) {
      unsigned long long a{}, b{};
      int used{};
      if (word == "prp" || word == "ll") {
        kind = word;
      } else if (tune::parseRegime(word)) {
        regime = word;
      } else if (sscanf(word.c_str(), "%llu-%llu%n", &a, &b, &used) == 2 && size_t(used) == word.size() && a <= b) {
        lo = a;
        hi = b;
      } else {
        throw "tune.txt line";
      }
    }
  }
  vector<KeyVal> uses;
  u64 maxExp{hi};
  if (auto pos = line.find('#'); pos != string::npos) {
    for (auto& kv : Args::splitUses(line.substr(pos + 1))) {
      // The first item is the maxExp comment, not a setting
      if (!kv.first.empty() && std::ranges::all_of(kv.first, [](char c) { return isdigit(c); })) {
        if (!maxExp) { maxExp = strtoull(kv.first.c_str(), nullptr, 10); }
        continue;
      }
      uses.push_back(std::move(kv));
    }
  }
  FFTConfig const fft{specBuf};
  // A maxExp below the FFT's own lowers it: the line's settings were found to hold only that far.  Written by a build with
  // a lower table, it is a little conservative until the line is written again.
  return TuneEntry{cost, fft, std::move(uses), maxExp && maxExp < fft.maxExp() ? maxExp : 0, kind, regime, lo};
}

// Every line of a tune file as written, in the file's order; a line this build cannot read is logged and skipped.
static vector<TuneEntry> readLines(const fs::path& path) {
  vector<TuneEntry> out;
  File fi = File::openRead(path);
  if (!fi) { return out; }
  fi.allowUnterminatedLastLine();
  for (const string& line : fi) {
    if (isCommentLine(line)) { continue; }
    try {
      if (optional<TuneEntry> e = parseTuneLine(line)) {
        out.push_back(std::move(*e));
      } else {
        log("%s line '%s' ignored\n", path.filename().string().c_str(), rstripNewline(line).c_str());
      }
    } catch (const char*) {
      log("%s line '%s' ignored\n", path.filename().string().c_str(), rstripNewline(line).c_str());
    }
  }
  return out;
}

// The build named on the first line of a tune file the tuner wrote; nothing for any other file.
static optional<u64> tunerBuild(const fs::path& path) {
  File fi = File::openRead(path);
  if (!fi) { return {}; }
  fi.allowUnterminatedLastLine();
  for (const string& line : fi) {
    string const magic = string{"# "} + TUNE_FILE_MAGIC + ", build ";
    if (!line.starts_with(magic)) { return {}; }
    return strtoull(line.c_str() + magic.size(), nullptr, 16);
  }
  return {};
}

bool TuneEntry::tunerFile(const Args& args) { return tunerBuild(findFile(args, "tune.txt")).has_value(); }

vector<TuneEntry> TuneEntry::tunedLines(const Args& args, bool ll) {
  fs::path const path = findFile(args, "tune.txt");
  vector<TuneEntry> own, fallback;
  if (optional<u64> const build = tunerBuild(path)) {
    static bool warned = false;
    if (*build != tune::buildFingerprint() && !warned) {
      warned = true;
      log("Warning: tune.txt was measured with other kernels (build %016" PRIx64 ", this one is %016" PRIx64 "); its costs "
          "and settings may not hold for this build.  Run the build that wrote it, or tune this one.\n",
          *build, tune::buildFingerprint());
    }
    for (TuneEntry& e : readLines(path)) { (e.kind == (ll ? "ll" : "prp") ? own : fallback).push_back(std::move(e)); }
    if (!ll) { fallback.clear(); }
  } else if (fs::path const llPath = findFile(args, "tune-ll.txt"); ll && fs::exists(llPath)) {
    own = readLines(llPath);
    fallback = readLines(path);
  } else if (ll) {
    fallback = readLines(path);
  } else {
    own = readLines(path);
  }
  for (TuneEntry& e : fallback) { e.fallback = true; }
  std::ranges::stable_sort(own, {}, &TuneEntry::cost);
  std::ranges::stable_sort(fallback, {}, &TuneEntry::cost);
  std::ranges::move(fallback, std::back_inserter(own));
  return own;
}

vector<TuneEntry> TuneEntry::readTuneFile(const Args& args, bool ll) {
  fs::path const tuneFile = tuneFilePath(args, ll);

  // if (!fs::exists(tuneFile)) { log("Tune file %s not found\n", tuneFile.string().c_str()); }

  vector<TuneEntry> results;
  File fi = File::openRead(tuneFile);
  if (!fi) { return {}; }
  fi.allowUnterminatedLastLine();

  for (const string& line : fi) {
    if (isCommentLine(line)) { continue; }
    try {
      optional<TuneEntry> const e = parseTuneLine(line);
      if (!e) {
        log("tune.txt line '%s' ignored\n", line.c_str());
        continue;
      }
      // Insert through update() so the list is a proper cost/maxExp frontier whatever order the file is in.  The file
      // was written sorted, but maxExp comes from the bits-per-word tables of the build that reads it, so rows written
      // by an older build can be out of order or dominated by a cheaper row; those are dropped here.
      if (!e->update(results) && args.verbose) {
        log("tune.txt line '%s' ignored, a cheaper FFT covers its exponents\n", rstripNewline(line).c_str());
      }
    } catch (const char*) {
      // e.g. a row from an older build whose variant encoding is no longer valid: skip it, keep the other rows
      log("tune.txt line '%s' ignored\n", rstripNewline(line).c_str());
    }
  }
  if (args.verbose && !results.empty()) { log("Read %u entries from %s\n", u32(results.size()), tuneFile.string().c_str()); }
  return results;
}

const TuneEntry* TuneEntry::pick(const vector<TuneEntry>& lines, u64 E, double overdrive, bool fp64, bool doLog) {
  for (const TuneEntry& e : lines) {
    // The first acceptable is the best as they're sorted by cost, a test's own kind first.  An FFT too large for E (below
    // minBpw) would make the Gpu constructor throw "FFT size too large", so those are skipped too.
    if (E < e.minExp || E > e.maxExp() * overdrive || E / float(e.fft.size()) < e.fft.minBpw()) { continue; }
    if (!fp64 && e.fft.FFT_FP64) { continue; }
    // A line names the carry regime it was measured in; where E would run in another (a build whose tables moved the
    // boundary), its cost and settings are not of what would run.
    if (!e.regime.empty()) {
      if (string const would = tune::regimeOf(e.fft, E).label(); would != e.regime) {
        if (doLog) {
          log("tune.txt: %s holds %" PRIu64 " but would run it as %s, not as the %s it was measured in; passed over\n",
              e.fft.spec().c_str(), E, would.c_str(), e.regime.c_str());
        }
        continue;
      }
    }
    if (doLog && e.fallback) {
      log("No LL line of the tune file holds %" PRIu64 "; running %s by its line for PRP\n", E, e.fft.spec().c_str());
    }
    return &e;
  }
  return nullptr;
}

// Unlike readTuneFile() this also looks at lines a cheaper FFT dominates, so that a user can try settings on any FFT by adding
// them to its tune.txt line.
vector<KeyVal> TuneEntry::usesFor(const Args& args, const FFTConfig& fft, u64 E, bool ll) {
  string const spec = fft.spec();
  vector<TuneEntry> const lines = tunedLines(args, ll);
  const TuneEntry* first = nullptr;
  for (const TuneEntry& e : lines) {
    if (e.fft.spec() != spec) { continue; }
    if (E >= e.minExp && E <= e.maxExp()) { return e.uses; }
    if (!first) { first = &e; }
  }
  return first ? first->uses : vector<KeyVal>{};
}

void TuneEntry::writeTuneFile(const vector<TuneEntry>& results) {
  // The comments of the tune.txt being replaced, each run of them with the max exponent of the entry that followed it (or none at
  // the end).  A run is written again before the first entry whose max exponent is at least that, so it stays above its entry,
  // or above the entry that took its place.
  vector<pair<string, optional<u64>>> comments;
  {
    File fi = File::openRead("tune.txt");
    if (fi) {
      fi.allowUnterminatedLastLine();
      string run;
      for (const string& line : fi) {
        if (isCommentLine(line)) {
          run += rstripNewline(line) + '\n';
          continue;
        }
        if (run.empty()) { continue; }
        try {
          if (optional<TuneEntry> const e = parseTuneLine(line)) { comments.emplace_back(std::move(run), e->maxExp()); run.clear(); }
        } catch (const char*) {}
      }
      if (!run.empty()) { comments.emplace_back(std::move(run), std::nullopt); }
    }
    // In case the file was not in max exponent order; the end run stays last
    std::ranges::stable_sort(comments, {}, [](const auto& c) { return c.second.value_or(UINT64_MAX); });
  }

  [[maybe_unused]] u64 prevMaxExp{};
  [[maybe_unused]] double prevCost{};
  CycleFile tune{"tune.txt"};
  auto c = comments.begin();
  for (const TuneEntry& r : results) {
    u64 const maxExp = r.maxExp();
    for (; c != comments.end() && c->second && *c->second <= maxExp; ++c) { tune->printf("%s", c->first.c_str()); }
    assert(r.cost >= prevCost && maxExp > prevMaxExp);
    prevCost = r.cost;
    prevMaxExp = maxExp;
    string uses;
    for (const auto& [key, val] : r.uses) { uses += ", " + key + '=' + val; }
    tune->printf("%6.1f %14s # %" PRIu64 "%s\n", r.cost, r.fft.spec().c_str(), maxExp, uses.c_str());
  }
  for (; c != comments.end(); ++c) { tune->printf("%s", c->first.c_str()); }
}
