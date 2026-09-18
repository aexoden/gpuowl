// Copyright (C) Jason Lynch

#include "Selection.h"

#include "CycleFile.h"
#include "File.h"
#include "log.h"
#include "Sha3Hash.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <system_error>

namespace tune {

namespace {

const char* const HEADER = "# prpll selection v1";

std::string costText(double cost) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", cost);
  return buf;
}

std::vector<std::pair<std::string, std::string>> pairsOf(const UseConfig& config) {
  return {config.begin(), config.end()};
}

UseConfig configOf(const std::vector<std::pair<std::string, std::string>>& pairs) {
  return {pairs.begin(), pairs.end()};
}

bool entryIsSound(const SelectionEntry& e, const FFTConfig& fft, std::string_view where) {
  auto const refuse = [&](const std::string& why) {
    log("%s: entry %s (%s) %s\n", std::string{where}.c_str(), e.id.c_str(), e.fft.c_str(), why.c_str());
    return false;
  };

  if (!std::isfinite(e.cost) || e.cost < 0) { return refuse("has no usable cost"); }
  if (!isWriteableConfig(e.opts)) { return refuse("holds an option this format cannot write back"); }

  if (e.emin < minExp(fft)) {
    return refuse("starts at " + to_string(e.emin) + ", below the " + to_string(minExp(fft)) + " this FFT accepts");
  }

  if (intervals(fft, e.emin, e.reach).size() != 1) {
    return refuse("does not cover exactly one regime over [" + to_string(e.emin) + ", " + to_string(e.reach) + "]");
  }

  return true;
}

}  // namespace

std::string entryId(const std::string& fft, TestKind kind, Regime regime, const UseConfig& opts) {
  std::string const key = fft + ' ' + toString(kind) + ' ' + regime.label() + ' ' + configText(opts);
  char buf[32];
  snprintf(buf, sizeof(buf), "%016" PRIx64, SHA3::hash(key)[0]);
  return buf;
}

SelectionLayers SelectionFile::layersFor(const SelectionEntry& entry) const {
  return SelectionLayers{.global = global, .family = family, .entry = pairsOf(entry.opts)};
}

bool finalize(SelectionFile& file) {
  if (!isWriteableConfig(configOf(file.global))) {
    log("The selection file's global use line holds an option this format cannot write back\n");
    return false;
  }

  for (const UseLine& line : file.family) {
    if (!isWriteableConfig(configOf(line.uses))) {
      log("The selection file's '%s' use line holds an option this format cannot write back\n",
          line.selector.spec().c_str());
      return false;
    }
  }

  std::set<std::string> ids;

  for (SelectionEntry& e : file.entries) {
    auto const fft = parseFft(e.fft);
    if (!fft) {
      log("Selection entry '%s' is not an FFT spec\n", e.fft.c_str());
      return false;
    }

    e.fft = fft->spec();
    e.regime = regimeOf(*fft, e.emin);

    if (!entryIsSound(e, *fft, "The selection file")) { return false; }

    e.id = entryId(e.fft, e.kind, e.regime, e.opts);

    if (!ids.insert(e.id).second) {
      log("Selection entry '%s' is a second entry for one configuration in one regime\n", e.fft.c_str());
      return false;
    }
  }

  std::ranges::stable_sort(file.entries, {}, &SelectionEntry::cost);
  return true;
}

std::optional<SelectionFile> parseSelection(std::string_view text, std::string_view name) {
  SelectionFile file;
  std::map<std::string, UseConfig> opts;

  bool sawHeader = false;
  bool sawGlobal = false;
  bool ok = true;
  u32 lineNo = 0;

  for (const std::string& raw : split(std::string{text}, '\n')) {
    ++lineNo;
    std::string const line = rstripNewline(raw);

    auto refuse = [&](const std::string& why) {
      log("%s:%u: %s\n", std::string{name}.c_str(), lineNo, why.c_str());
      ok = false;
    };

    if (line.empty()) { continue; }

    if (line.starts_with('#')) {
      if (!sawHeader) {
        if (line != HEADER) {
          refuse("not a selection file (expected '" + std::string{HEADER} + "')");
          break;
        }

        sawHeader = true;
      } else if (file.provenance.empty() && line.starts_with("# written ")) {
        file.provenance = line.substr(2);
      } else {
        file.unknown.push_back(line);
      }

      continue;
    }

    if (!sawHeader) {
      refuse("missing the '" + std::string{HEADER} + "' header line");
      break;
    }

    auto const fields = splitFields(line);
    if (!fields) {
      refuse("a quoted field is left open");
      continue;
    }

    std::vector<std::string> const& f = *fields;
    if (f.empty()) { continue; }

    std::string const& tag = f[0];

    if (tag == "use") {
      if (f.size() >= 2 && f[1] == "!") {
        try {
          UseLine parsed = parseUseLine(line.substr(line.find('!')));
          if (!isWriteableConfig(configOf(parsed.uses))) {
            refuse("this family line holds an option this format cannot write back");
          } else {
            file.family.push_back(std::move(parsed));
          }
        } catch (...) { refuse("malformed family line"); }
      } else if (f.size() == 2) {
        auto const config = parseConfigText(f[1]);

        if (!config) {
          refuse("'" + f[1] + "' is not an option set");
        } else if (!isWriteableConfig(*config)) {
          refuse("'" + f[1] + "' holds an option this format cannot write back");
        } else if (sawGlobal) {
          refuse("a second global use line; there is one, and the later would silently replace the earlier");
        } else {
          file.global = pairsOf(*config);
          sawGlobal = true;
        }
      } else {
        refuse("a use line takes one option set, or '!' and a selector");
      }
    } else if (tag == "entry") {
      if (f.size() != 8) {
        refuse("entry has " + to_string(f.size()) + " fields, expected 8");
        continue;
      }

      SelectionEntry e;

      e.id = f[1];
      auto const cost = parseNonNegative(f[2]);
      auto const kind = parseTestKind(f[4]);
      auto const emin = parseInt<u64>(f[5]);
      auto const reach = parseInt<u64>(f[6]);
      auto const evidence = parseEvidence(f[7]);
      if (!cost) { refuse("'" + f[2] + "' is not a cost"); }
      if (!kind) { refuse("'" + f[4] + "' is not a test kind"); }
      if (!emin || !reach) { refuse("entry has a malformed interval"); }
      if (!evidence) { refuse("'" + f[7] + "' is not an evidence state"); }

      auto const fft = parseFft(f[3]);
      if (!fft) {
        refuse("'" + f[3] + "' is not an FFT spec");
        continue;
      }

      if (!cost || !kind || !emin || !reach || !evidence) { continue; }
      e.fft = fft->spec();
      e.cost = *cost;
      e.kind = *kind;
      e.emin = *emin;
      e.reach = *reach;
      e.evidence = *evidence;
      e.regime = regimeOf(*fft, e.emin);
      file.entries.push_back(std::move(e));
    } else if (tag == "opts") {
      if (f.size() != 3) {
        refuse("opts has " + to_string(f.size()) + " fields, expected 3");
        continue;
      }

      auto const config = parseConfigText(f[2]);
      if (!config) {
        refuse("'" + f[2] + "' is not an option set");
      } else if (!opts.emplace(f[1], *config).second) {
        refuse("a second option set for entry " + f[1]);
      }
    } else {
      file.unknown.push_back(line);
    }
  }

  if (!ok) { return {}; }

  for (SelectionEntry& e : file.entries) {
    auto const it = opts.find(e.id);

    if (it == opts.end()) {
      log("%s: entry %s has no opts line\n", std::string{name}.c_str(), e.id.c_str());
      return {};
    }

    e.opts = it->second;
    opts.erase(it);

    if (!entryIsSound(e, *parseFft(e.fft), name)) { return {}; }

    std::string const expected = entryId(e.fft, e.kind, e.regime, e.opts);
    if (e.id != expected) {
      log("%s: entry %s does not hash to its own id (%s)\n", std::string{name}.c_str(), e.id.c_str(), expected.c_str());
      return {};
    }
  }

  for (const auto& [id, unused] : opts) {
    log("%s: an opts line for %s, which is not an entry\n", std::string{name}.c_str(), id.c_str());
    return {};
  }

  return file;
}

std::optional<SelectionFile> readSelection(const fs::path& path) {
  File file = File::openRead(path);
  if (!file) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
      log("Can't read '%s'\n", path.string().c_str());
      return {};
    }

    return SelectionFile{};
  }

  return parseSelection(file.readAll(), path.string());
}

std::string text(const SelectionFile& file) {
  std::string out = std::string{HEADER} + '\n';
  if (!file.provenance.empty()) { out += "# " + file.provenance + '\n'; }

  UseConfig global{file.global.begin(), file.global.end()};
  out += "use   " + configText(global) + '\n';

  for (const UseLine& line : file.family) {
    UseConfig uses{line.uses.begin(), line.uses.end()};
    out += "use ! " + line.selector.spec() + ' ' + configText(uses) + '\n';
  }

  for (const SelectionEntry& e : file.entries) {
    out += "entry " + e.id + ' ' + costText(e.cost) + ' ' + e.fft + ' ' + toString(e.kind) + ' ' + to_string(e.emin) +
      ' ' + to_string(e.reach) + ' ' + toString(e.evidence) + '\n';
    out += "opts  " + e.id + ' ' + configText(e.opts) + '\n';
  }

  for (const std::string& line : file.unknown) { out += line + '\n'; }

  return out;
}

void writeSelection(const fs::path& path, const SelectionFile& file) {
  CycleFile out{path};
  out->write(text(file));
}

}  // namespace tune
