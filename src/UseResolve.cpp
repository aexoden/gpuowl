// Copyright (C) Jason Lynch

#include "UseResolve.h"

#include "Args.h"
#include "FFTVariants.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace tune {

namespace {

constexpr FFT_TYPES KNOWN_TYPES[] = {FFT64, FFT3161, FFT3261, FFT61, FFT323161, FFT3231, FFT6431, FFT31, FFT32};

[[noreturn]] void refuse(std::string_view text, const char* why) {
  log("Invalid FFT selector '%.*s': %s\n", int(text.size()), text.data(), why);
  throw "Invalid FFT selector";
}

// A decimal count with an optional K or M suffix, as FFT specs write sizes. Unlike the spec parser, this rejects
// anything else, so "51x" is not read as 51.
std::optional<u32> parseCount(std::string_view s) {
  u32 multiple = 1;

  if (!s.empty() && (s.back() == 'k' || s.back() == 'K')) {
    multiple = 1024;
    s.remove_suffix(1);
  } else if (!s.empty() && (s.back() == 'm' || s.back() == 'M')) {
    multiple = 1024 * 1024;
    s.remove_suffix(1);
  }

  if (s.empty() || s.size() > 6 || !std::ranges::all_of(s, [](char c) { return std::isdigit(u8(c)); })) { return {}; }

  return u32(std::stoul(std::string{s})) * multiple;
}

std::vector<std::string_view> splitOn(std::string_view s, char sep) {
  std::vector<std::string_view> parts;

  for (size_t start = 0;;) {
    size_t const end = s.find(sep, start);
    parts.push_back(s.substr(start, end - start));
    if (end == std::string_view::npos) { return parts; }
    start = end + 1;
  }
}

// Applies the lines that select this FFT and kind, weakest first. The sort is stable, so among equal ranks the last
// line read is applied last.
void applyLines(const std::vector<UseLine>& lines, const FFTConfig& fft, TestKind kind, UseConfig& config) {
  std::vector<const UseLine*> matching;

  for (const UseLine& line : lines) {
    if (line.selector.matches(fft, kind)) { matching.push_back(&line); }
  }

  std::ranges::stable_sort(matching, {}, [](const UseLine* line) { return line->selector.rank(); });

  for (const UseLine* line : matching) {
    for (const auto& [k, v] : line->uses) { config[k] = v; }
  }
}

} // namespace

const char* toString(TestKind kind) { return kind == TestKind::LL ? "ll" : "prp"; }

FFTSelector FFTSelector::parse(std::string_view text) {
  FFTSelector sel;
  std::vector<std::string_view> parts = splitOn(text, ':');
  auto next = parts.begin();

  if (next->empty()) { refuse(text, "empty"); }

  if (std::isalpha(u8(next->front()))) {
    if (*next == "prp") {
      sel.kind = TestKind::PRP;
    } else if (*next == "ll") {
      sel.kind = TestKind::LL;
    } else {
      refuse(text, "expected a test kind (prp or ll) or an FFT spec");
    }

    if (++next == parts.end()) { return sel; }
  }

  // As in an FFT spec, a leading number below 60 is the FFT type and a missing one means FP64.
  std::optional<u32> first = parseCount(*next);

  if (!first) { refuse(text, "expected a number"); }

  if (*first < 60) {
    if (std::ranges::find(KNOWN_TYPES, FFT_TYPES(*first)) == std::end(KNOWN_TYPES)) {
      refuse(text, "unknown FFT type");
    }
    sel.type = FFT_TYPES(*first);
    ++next;
  } else {
    sel.type = FFT64;
  }

  auto const rest = size_t(parts.end() - next);
  if (rest == 0) {
    if (*first >= 60) { refuse(text, "expected an FFT type or width:middle:height"); }
    return sel;
  }

  if (rest < 3 || rest > 5) { refuse(text, "expected width:middle:height[:variant[:carry]]"); }

  std::optional<u32> const w = parseCount(next[0]);
  std::optional<u32> const m = parseCount(next[1]);
  std::optional<u32> const h = parseCount(next[2]);

  if (!w || !m || !h) { refuse(text, "expected width:middle:height"); }
  if (*w != 256 && *w != 512 && *w != 1024 && *w != 4096) { refuse(text, "width must be 256, 512, 1K or 4K"); }
  if (*m < 2 || *m > 16) { refuse(text, "middle must be between 2 and 16"); }
  if (*h != 256 && *h != 512 && *h != 1024) { refuse(text, "height must be 256, 512 or 1K"); }
  if (*sel.type != FFT64 && *sel.type != FFT32 && (*m & (*m - 1))) {
    refuse(text, "NTT middle must be a power of two");
  }
  sel.width = *w;
  sel.middle = *m;
  sel.height = *h;

  if (rest >= 4) {
    std::optional<u32> const v = parseCount(next[3]);
    if (!v || next[3].size() != 3 || variant_W(*v) >= N_VARIANT_W || variant_M(*v) >= N_VARIANT_M ||
        variant_H(*v) >= N_VARIANT_H) {
      refuse(text, "invalid variant");
    }
    sel.variant = *v;
  }

  if (rest == 5) {
    if (next[4] != "0" && next[4] != "1") { refuse(text, "carry must be 0 (32-bit) or 1 (64-bit)"); }
    sel.carry = next[4] == "0" ? CARRY_32 : CARRY_64;
  }

  // FFTConfig names its variant and carry canonically, so a selector specifying the same kernels differently must fold
  // the same way or it would silently match nothing.
  if (sel.variant) {
    FFTShape const shape{*sel.type, sel.width, sel.middle, sel.height};
    if (sel.carry) { sel.carry = canonicalCarry(shape, *sel.variant, *sel.carry); }
    sel.variant = canonicalVariant(shape, *sel.variant);
  }

  return sel;
}

bool FFTSelector::matches(const FFTConfig& fft, TestKind testKind) const {
  if (kind && *kind != testKind) { return false; }
  if (type && *type != fft.shape.fft_type) { return false; }
  if (width && (width != fft.shape.width || middle != fft.shape.middle || height != fft.shape.height)) { return false; }
  if (variant && *variant != fft.variant) { return false; }
  return !carry || *carry == fft.carry;
}

u32 FFTSelector::rank() const {
  u32 const parts = (type ? 1 : 0) + (width ? 1 : 0) + (variant ? 1 : 0) + (carry ? 1 : 0);
  return (kind ? 5 : 0) + parts;
}

std::string FFTSelector::spec() const {
  std::string s = kind ? std::string{toString(*kind)} : "";
  auto add = [&s](const std::string& part) { s += (s.empty() ? "" : ":") + part; };

  if (type && (*type != FFT64 || !width)) { add(std::to_string(*type)); }
  if (width) { add(numberK(width) + ':' + std::to_string(middle) + ':' + numberK(height)); }
  if (variant) {
    add(std::to_string(variant_W(*variant)) + std::to_string(variant_M(*variant)) +
        std::to_string(variant_H(*variant)));
  }
  if (carry && *carry != CARRY_AUTO) { add(*carry == CARRY_32 ? "0" : "1"); }

  return s;
}

UseLine parseUseLine(std::string_view line) {
  auto bad = [line](const char* why) {
    log("Config line '%.*s' not understood (%s); expected '! <fft-selector> <key=value,...>'\n", int(line.size()),
        line.data(), why);
    throw "Invalid config line";
  };
  auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };

  // Split by hand rather than with a fixed-width sscanf, which truncated long option lists mid-value.
  std::vector<std::string_view> tokens;

  if (line.empty() || line.front() != '!') { bad("no leading '!'"); }

  for (size_t p = 1; p < line.size();) {
    if (isSpace(line[p])) {
      ++p;
      continue;
    }

    if (line[p] == '#') { break; }

    size_t end = p;
    while (end < line.size() && !isSpace(line[end])) { ++end; }
    tokens.push_back(line.substr(p, end - p));
    p = end;
  }

  if (tokens.size() != 2) {
    bad(tokens.size() < 2 ? "missing selector or options" : "unexpected text after the options");
  }

  UseLine out{FFTSelector::parse(tokens[0]), Args::splitUses(std::string{tokens[1]})};
  for (const auto& [k, v] : out.uses) {
    if (k.empty()) { bad("empty key"); }
  }
  if (out.uses.empty()) { bad("no options"); }
  return out;
}

UseConfig resolveConfig(const Args& args, const FFTConfig& fft, TestKind kind, const SelectionLayers& selection) {
  UseConfig config;

  for (const auto& [k, v] : selection.global) { config[k] = v; }
  applyLines(selection.family, fft, kind, config);
  for (const auto& [k, v] : selection.entry) { config[k] = v; }

  // args.flags merges the config files and the command line, in that order.
  for (const auto& [k, v] : args.flags) { config[k] = v; }
  applyLines(args.perFftConfig, fft, kind, config);
  for (const std::string& k : args.cliKeys) {
    if (auto it = args.flags.find(k); it != args.flags.end()) { config[k] = it->second; }
  }

  return config;
}

void resolveInto(Args& args, const FFTConfig& fft, TestKind kind,
                 const std::vector<std::pair<std::string, std::string>>& extraConf) {
  UseConfig resolved = resolveConfig(args, fft, kind);
  for (const auto& [k, v] : extraConf) { resolved[k] = v; }
  args.flags = std::move(resolved);
  args.perFftConfig.clear();
}

Takeover takeOverConfig(Args& args) {
  std::set<std::string> fromConfig;
  std::set<std::string> fromCommandLine;
  std::set<std::string> kept;

  for (const UseLine& line : args.perFftConfig) {
    for (const auto& [k, v] : line.uses) { fromConfig.insert(k); }
  }

  for (const auto& [k, v] : args.flags) {
    if (!args.cliKeys.contains(k)) {
      fromConfig.insert(k);
      continue;
    }

    const Option* const option = findOption(k);
    if (option && option->kind != Kind::Tunable) {
      kept.insert(k);
    } else {
      fromCommandLine.insert(k);
    }
  }

  std::erase_if(args.flags, [&kept](const auto& kv) { return !kept.contains(kv.first); });

  args.perFftConfig.clear();
  args.cliKeys = kept;

  return {{fromConfig.begin(), fromConfig.end()}, {fromCommandLine.begin(), fromCommandLine.end()}};
}

std::string describe(const Takeover& takeover) {
  auto join = [](const std::vector<std::string>& keys) {
    std::string s;
    for (const std::string& k : keys) { s += (s.empty() ? "" : ", ") + k; }
    return s;
  };

  if (takeover.configKeys.empty() && takeover.commandLineKeys.empty()) {
    return "Tuning from built-in defaults; no -use settings to ignore";
  }

  std::string s = "Tuning ignores these -use settings";
  if (!takeover.configKeys.empty()) { s += "; from the config files: " + join(takeover.configKeys); }
  if (!takeover.commandLineKeys.empty()) { s += "; from the command line: " + join(takeover.commandLineKeys); }
  return s;
}

} // namespace tune
