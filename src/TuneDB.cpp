// Copyright (C) Jason Lynch

#include "TuneDB.h"

#include "CycleFile.h"
#include "File.h"
#include "log.h"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <limits>
#include <system_error>

namespace tune {

namespace {

const char* const TWO_WRITERS = ".  Two processes writing one database do this; give each its own";

std::string quoted(const std::string& s) {
  bool const needs = s.empty() || s.find(' ') != std::string::npos || s.find('\t') != std::string::npos;
  return needs ? '"' + s + '"' : s;
}

std::optional<bool> parseBool(std::string_view text) {
  if (text == "0") { return false; }
  if (text == "1") { return true; }
  return {};
}

std::string driftText(double drift) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.4f", drift);
  return buf;
}

std::string hex16(u64 x) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%016" PRIx64, x);
  return buf;
}

std::optional<std::string> canonicalFft(const std::string& spec) {
  auto const fft = parseFft(spec);
  if (!fft) { return {}; }
  return fft->spec();
}

// Digits with the optional K or M suffix.
std::optional<u32> parseSize(std::string_view text) {
  u32 multiple = 1;
  if (!text.empty() && (text.back() == 'K' || text.back() == 'k')) {
    multiple = 1024;
    text.remove_suffix(1);
  } else if (!text.empty() && (text.back() == 'M' || text.back() == 'm')) {
    multiple = 1024 * 1024;
    text.remove_suffix(1);
  }

  auto const n = parseInt<u32>(text);
  if (!n || *n == 0 || *n > std::numeric_limits<u32>::max() / multiple) { return {}; }
  return *n * multiple;
}

std::optional<enum FFT_TYPES> fftType(u32 value) {
  for (enum FFT_TYPES const type : {FFT64, FFT3161, FFT3261, FFT61, FFT323161, FFT3231, FFT6431, FFT31, FFT32}) {
    if (value == u32(type)) { return type; }
  }

  return {};
}

std::pair<std::string_view, std::string_view> keyValue(std::string_view field) {
  auto const eq = field.find('=');
  if (eq == std::string_view::npos) { return {field, {}}; }
  return {field.substr(0, eq), field.substr(eq + 1)};
}

}  // namespace

std::optional<std::vector<std::string>> splitFields(std::string_view line) {
  std::vector<std::string> out;
  std::string cur;
  bool inQuote = false;
  bool have = false;

  for (char const c : line) {
    if (c == '"') {
      inQuote = !inQuote;
      have = true;
    } else if (!inQuote && (c == ' ' || c == '\t' || c == '\r')) {
      if (have) { out.push_back(cur); }
      cur.clear();
      have = false;
    } else {
      cur += c;
      have = true;
    }
  }

  if (inQuote) { return {}; }
  if (have) { out.push_back(cur); }
  return out;
}

std::optional<double> parseDouble(std::string_view text) {
  double value{};
  auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) { return {}; }
  return value;
}

std::optional<double> parseNonNegative(std::string_view text) {
  auto const value = parseDouble(text);
  if (!value || *value < 0) { return {}; }
  return value;
}

std::optional<double> parsePositive(std::string_view text) {
  auto const value = parseDouble(text);
  if (!value || *value <= 0) { return {}; }
  return value;
}

bool isWriteableField(std::string_view value) { return value.find_first_of("\"\n\r") == std::string_view::npos; }

bool isWriteableConfig(const UseConfig& config) {
  auto const plain = [](std::string_view s) { return s.find_first_of(",= \t\"\n\r") == std::string_view::npos; };
  return std::ranges::all_of(config,
                             [&](const auto& kv) { return !kv.first.empty() && plain(kv.first) && plain(kv.second); });
}

std::optional<FFTConfig> parseFft(std::string_view spec) {
  std::vector<std::string> parts = split(std::string{spec}, ':');

  enum FFT_TYPES type = FFT64;
  if (auto const lead = parseSize(parts.front()); lead && *lead < 60) {
    auto const named = fftType(*lead);
    if (!named) { return {}; }
    type = *named;
    parts.erase(parts.begin());
  }

  if (parts.size() < 3 || parts.size() > 5) { return {}; }

  auto const width = parseSize(parts[0]);
  auto const middle = parseSize(parts[1]);
  auto const height = parseSize(parts[2]);
  if (!width || !middle || !height) { return {}; }
  if (*width != 256 && *width != 512 && *width != 1024 && *width != 4096) { return {}; }
  if (*middle < 2 || *middle > 16) { return {}; }
  if (*height != 256 && *height != 512 && *height != 1024) { return {}; }
  if (type != FFT64 && type != FFT32 && (*middle & (*middle - 1))) { return {}; }

  u32 variant = LAST_VARIANT;
  if (parts.size() >= 4) {
    auto const named = parseInt<u32>(parts[3]);
    if (!named || variant_W(*named) >= N_VARIANT_W || variant_M(*named) >= N_VARIANT_M ||
        variant_H(*named) >= N_VARIANT_H) {
      return {};
    }
    variant = *named;
  }

  enum CARRY_KIND carry = CARRY_AUTO;
  if (parts.size() == 5) {
    if (parts[4] == "0") {
      carry = CARRY_32;
    } else if (parts[4] == "1") {
      carry = CARRY_64;
    } else {
      return {};
    }
  }

  return FFTConfig{FFTShape{type, *width, *middle, *height}, variant, carry};
}

const char* toString(Evidence evidence) {
  switch (evidence) {
  case Evidence::Unvalidated: return "unvalidated";
  case Evidence::Confirmed: return "confirmed";
  case Evidence::Rejected: return "rejected";
  case Evidence::Unavailable: return "unavailable";
  case Evidence::NotApplicable: return "n/a";
  }
  return "unvalidated";
}

std::optional<Evidence> parseEvidence(std::string_view text) {
  for (Evidence e : {Evidence::Unvalidated, Evidence::Confirmed, Evidence::Rejected, Evidence::Unavailable,
                     Evidence::NotApplicable}) {
    if (text == toString(e)) { return e; }
  }

  return {};
}

std::string configText(const UseConfig& config) {
  std::string out;
  for (const auto& [key, value] : config) { out += (out.empty() ? "" : ",") + key + '=' + value; }
  return out.empty() ? "-" : out;
}

std::optional<UseConfig> parseConfigText(std::string_view text) {
  UseConfig config;
  if (text == "-") { return config; }
  if (text.empty()) { return {}; }

  size_t at = 0;
  while (at <= text.size()) {
    size_t const comma = std::min(text.find(',', at), text.size());
    std::string_view const item = text.substr(at, comma - at);
    if (item.empty()) { return {}; }

    auto const [key, value] = keyValue(item);
    bool const bare = item.find('=') == std::string_view::npos;
    config[std::string{key}] = bare ? "1" : std::string{value};
    at = comma + 1;
  }

  return config;
}

Env DbEnv::toEnv() const {
  return Env{.isAmd = isAmd,
             .isNvidia = isNvidia,
             .cudaBackend = cudaBackend,
             .noAsm = noAsm,
             .computeCapability = computeCapability,
             .pdlLaunch = pdlLaunch,
             .deviceName = gpu,
             .driverVersion = driver};
}

bool DbEnv::sameMachine(const DbEnv& other) const {
  return gpu == other.gpu && name == other.name && driver == other.driver && isAmd == other.isAmd &&
    isNvidia == other.isNvidia && cudaBackend == other.cudaBackend && noAsm == other.noAsm &&
    pdlLaunch == other.pdlLaunch && computeCapability == other.computeCapability && machine == other.machine &&
    build == other.build;
}

DbEnv dbEnvOf(const Env& env) {
  return DbEnv{.gpu = env.deviceName,
               .name = env.deviceName,
               .driver = env.driverVersion,
               .isAmd = env.isAmd,
               .isNvidia = env.isNvidia,
               .cudaBackend = env.cudaBackend,
               .noAsm = env.noAsm,
               .pdlLaunch = env.pdlLaunch,
               .computeCapability = env.computeCapability,
               .machine = {},
               .build = 0,
               .extra = {}};
}

std::string formatRow(const DbEnv& row) {
  std::string out = "env   " + to_string(row.id) + " gpu=" + quoted(row.gpu) + " name=" + quoted(row.name) +
    " drv=" + quoted(row.driver) + " vendor=" +
    (row.isAmd        ? "amd"
       : row.isNvidia ? "nvidia"
                      : "-") +
    " be=" + (row.cudaBackend ? "cuda" : "ocl") + " cc=" + to_string(row.computeCapability) +
    " noasm=" + (row.noAsm ? "1" : "0") + " pdl=" + (row.pdlLaunch ? "1" : "0") +
    " machine=" + (row.machine.empty() ? "-" : quoted(row.machine)) + " build=" + hex16(row.build);
  for (const std::string& field : row.extra) { out += ' ' + quoted(field); }
  return out;
}

std::string formatCfgRow(u32 id, const UseConfig& config) {
  return "cfg   " + to_string(id) + ' ' + configText(config);
}

std::string formatRow(const SessRow& row) {
  return "sess  " + to_string(row.id) + " env=" + to_string(row.env) + " start=" + to_string(row.start) +
    " gen=" + to_string(row.gen) + " anchor=" + (row.anchor.empty() ? "-" : quoted(row.anchor)) +
    (row.alarmed ? " alarmed=1" : "");
}

namespace {

std::string measurementText(const Measurement& m) {
  char buf[192];
  snprintf(buf, sizeof(buf), "%.3f %.3f %u %u %s %s %" PRIu64, m.mean, m.stddev, m.blocks, m.calls,
           driftText(m.drift).c_str(), toString(m.status), m.ts);
  return buf;
}

}  // namespace

std::string formatRow(const RunRow& row) {
  return "run   " + to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + to_string(row.exponent) +
    ' ' + row.regime.label() + ' ' + to_string(row.cfg) + ' ' + measurementText(row.m);
}

std::string formatRow(const TryRow& row) {
  return "try   " + to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + to_string(row.exponent) +
    ' ' + to_string(row.cfg) + ' ' + to_string(row.ts);
}

std::string formatRow(const NogoRow& row) {
  return "nogo  " + to_string(row.sess) + ' ' + row.fft + ' ' + row.key + '=' + row.val + ' ' + to_string(row.ts);
}

std::string formatRow(const RoeRow& row) {
  char buf[128];
  snprintf(buf, sizeof(buf), "%.2f %u %.6f %s", row.z, row.n, row.maxRoe, row.checkOk ? "ok" : "fail");
  return "roe   " + to_string(row.sess) + ' ' + row.fft + ' ' + to_string(row.exponent) + ' ' + to_string(row.cfg) +
    ' ' + buf + ' ' + to_string(row.ts);
}

std::string formatRow(const ReachRow& row) {
  return "reach " + to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + row.regime.label() + ' ' +
    to_string(row.cfg) + ' ' + to_string(row.reach) + ' ' + toString(row.evidence) + ' ' + to_string(row.ts);
}

std::string formatRow(const RefRow& row) {
  return "ref   " + to_string(row.sess) + ' ' + row.fft + ' ' + to_string(row.exponent) + ' ' + to_string(row.iters) +
    ' ' + hex16(row.res64) + ' ' + to_string(row.ts);
}

const DbEnv* TuneDB::findEnv(u32 id) const {
  auto const it = std::ranges::find(envs_, id, &DbEnv::id);
  return it == envs_.end() ? nullptr : &*it;
}

const SessRow* TuneDB::findSession(u32 id) const {
  auto const it = std::ranges::find(sessions_, id, &SessRow::id);
  return it == sessions_.end() ? nullptr : &*it;
}

const UseConfig* TuneDB::findCfg(u32 id) const {
  auto const it = cfgs_.find(id);
  return it == cfgs_.end() ? nullptr : &it->second;
}

namespace {

template<typename Row> bool pushCanonical(std::vector<Row>& into, const Row& row) {
  auto const fft = canonicalFft(row.fft);
  if (!fft) { return false; }
  into.push_back(row);
  into.back().fft = *fft;
  return true;
}

}  // namespace

bool TuneDB::add(const DbEnv& row) {
  if (findEnv(row.id)) { return false; }
  if (!isWriteableField(row.gpu) || !isWriteableField(row.name) || !isWriteableField(row.driver) ||
      !isWriteableField(row.machine)) {
    return false;
  }
  if (!std::ranges::all_of(row.extra, [](const std::string& f) { return isWriteableField(f); })) { return false; }

  envs_.push_back(row);
  return true;
}

bool TuneDB::addCfg(u32 id, UseConfig config) {
  if (!isWriteableConfig(config)) { return false; }
  return cfgs_.emplace(id, std::move(config)).second;
}

bool TuneDB::add(const SessRow& row) {
  if (findSession(row.id) || !findEnv(row.env) || !isWriteableField(row.anchor)) { return false; }
  sessions_.push_back(row);
  return true;
}

bool TuneDB::add(const RunRow& row) {
  if (!findSession(row.sess) || !findCfg(row.cfg)) { return false; }
  Measurement const& m = row.m;
  if (!std::isfinite(m.mean) || m.mean < 0 || !std::isfinite(m.stddev) || m.stddev < 0) { return false; }
  if (!parsePositive(driftText(m.drift))) { return false; }
  return pushCanonical(runs_, row);
}

bool TuneDB::add(const TryRow& row) {
  if (!findSession(row.sess) || !findCfg(row.cfg)) { return false; }
  return pushCanonical(tries_, row);
}

bool TuneDB::add(const NogoRow& row) {
  auto const plain = [](std::string_view s) { return s.find_first_of(" \t\"\n\r") == std::string_view::npos; };
  if (!findSession(row.sess) || row.key.empty() || !plain(row.key) || !plain(row.val)) { return false; }
  return pushCanonical(nogos_, row);
}

bool TuneDB::add(const RoeRow& row) {
  if (!findSession(row.sess) || !findCfg(row.cfg)) { return false; }
  if (!std::isfinite(row.z) || !std::isfinite(row.maxRoe) || row.maxRoe < 0) { return false; }
  return pushCanonical(roes_, row);
}

bool TuneDB::add(const ReachRow& row) {
  if (!findSession(row.sess) || !findCfg(row.cfg)) { return false; }
  return pushCanonical(reaches_, row);
}

bool TuneDB::add(const RefRow& row) {
  if (!findSession(row.sess)) { return false; }
  return pushCanonical(refs_, row);
}

void TuneDB::clear() {
  envs_.clear();
  cfgs_.clear();
  sessions_.clear();
  runs_.clear();
  tries_.clear();
  nogos_.clear();
  roes_.clear();
  reaches_.clear();
  refs_.clear();
  unknown_.clear();
}

bool TuneDB::parse(std::string_view text, std::string_view name) {
  clear();

  bool sawHeader = false;
  u32 lineNo = 0;
  bool ok = true;

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
          refuse("not a tune database (expected '" + std::string{HEADER} + "')");
          break;
        }

        sawHeader = true;
      } else {
        unknown_.push_back(line);
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

    auto const known = tag == "env" || tag == "cfg" || tag == "sess" || tag == "run" || tag == "try" || tag == "nogo" ||
      tag == "roe" || tag == "reach" || tag == "ref";
    if (known && f.size() < 2) {
      refuse(tag + " row has no fields");
      continue;
    }

    auto sessionOf = [&]() -> std::optional<u32> {
      auto const id = parseInt<u32>(f[1]);

      if (!id) {
        refuse("'" + f[1] + "' is not a session id");
      } else if (!findSession(*id)) {
        refuse("names session " + f[1] + ", which is not declared");
        return {};
      }

      return id;
    };

    auto cfgOf = [&](const std::string& field, std::optional<u32>& out) {
      out = parseInt<u32>(field);

      if (!out) {
        refuse("'" + field + "' is not a cfg id");
      } else if (!findCfg(*out)) {
        refuse("names cfg " + field + ", which is not declared");
        out.reset();
      }
    };

    if (tag == "env") {
      DbEnv e;
      auto const id = parseInt<u32>(f[1]);

      if (!id) {
        refuse("'" + f[1] + "' is not an env id");
        continue;
      }

      e.id = *id;

      bool bad = false;
      for (size_t i = 2; i < f.size() && !bad; ++i) {
        auto const [key, val] = keyValue(f[i]);

        auto number = [&](auto& field) {
          auto const v = parseInt<std::remove_reference_t<decltype(field)>>(val);
          if (v) {
            field = *v;
          } else {
            refuse("env field '" + f[i] + "' is not a number");
            bad = true;
          }
        };

        auto flag = [&](bool& field) {
          auto const v = parseBool(val);
          if (v) {
            field = *v;
          } else {
            refuse("env field '" + f[i] + "' is not 0 or 1");
            bad = true;
          }
        };

        if (key == "gpu") {
          e.gpu = val;
        } else if (key == "name") {
          e.name = val;
        } else if (key == "drv") {
          e.driver = val;
        } else if (key == "vendor") {
          e.isAmd = val == "amd";
          e.isNvidia = val == "nvidia";
          if (!e.isAmd && !e.isNvidia && val != "-") {
            refuse("env field '" + f[i] + "' is not amd, nvidia or -");
            bad = true;
          }
        } else if (key == "be") {
          e.cudaBackend = val == "cuda";
          if (!e.cudaBackend && val != "ocl") {
            refuse("env field '" + f[i] + "' is not ocl or cuda");
            bad = true;
          }
        } else if (key == "cc") {
          number(e.computeCapability);
        } else if (key == "noasm") {
          flag(e.noAsm);
        } else if (key == "pdl") {
          flag(e.pdlLaunch);
        } else if (key == "machine") {
          e.machine = val == "-" ? "" : std::string{val};
        } else if (key == "build") {
          auto const v = parseInt<u64>(val, 16);
          if (v) {
            e.build = *v;
          } else {
            refuse("env field '" + f[i] + "' is not a hex fingerprint");
            bad = true;
          }
        } else {
          e.extra.push_back(f[i]);
        }
      }

      if (bad) { continue; }
      if (findEnv(e.id)) {
        refuse("env " + to_string(e.id) + " declared twice" + TWO_WRITERS);
      } else if (!add(e)) {
        refuse("env " + to_string(e.id) + " holds a value this format cannot write back");
      }
    } else if (tag == "cfg") {
      auto const id = parseInt<u32>(f[1]);
      if (!id) {
        refuse("'" + f[1] + "' is not a cfg id");
        continue;
      }

      auto const config = f.size() >= 3 ? parseConfigText(f[2]) : std::optional<UseConfig>{};
      if (f.size() != 3 || !config) {
        refuse("cfg " + f[1] + " is not an option set");
        continue;
      }

      if (findCfg(*id)) {
        refuse("cfg " + f[1] + " declared twice" + TWO_WRITERS);
      } else if (!addCfg(*id, *config)) {
        refuse("cfg " + f[1] + " holds a key or value this format cannot write back");
      }
    } else if (tag == "sess") {
      SessRow s;
      auto const id = parseInt<u32>(f[1]);
      if (!id) {
        refuse("'" + f[1] + "' is not a session id");
        continue;
      }
      s.id = *id;

      bool bad = false;
      for (size_t i = 2; i < f.size() && !bad; ++i) {
        auto const [key, val] = keyValue(f[i]);
        auto number = [&](auto& field) {
          auto const v = parseInt<std::remove_reference_t<decltype(field)>>(val);
          if (v) {
            field = *v;
          } else {
            refuse("session field '" + f[i] + "' is not a number");
            bad = true;
          }
        };

        if (key == "env") {
          number(s.env);
        } else if (key == "start") {
          number(s.start);
        } else if (key == "gen") {
          number(s.gen);
        } else if (key == "anchor") {
          s.anchor = val == "-" ? "" : std::string{val};
        } else if (key == "alarmed") {
          auto const v = val.empty() ? std::optional{true} : parseBool(val);
          if (v) {
            s.alarmed = *v;
          } else {
            refuse("session field '" + f[i] + "' is not 0 or 1");
            bad = true;
          }
        } else {
          refuse("unknown session field '" + f[i] + "'");
          bad = true;
        }
      }

      if (bad) { continue; }
      if (!findEnv(s.env)) {
        refuse("session " + to_string(s.id) + " names env " + to_string(s.env) + ", which is not declared");
        continue;
      }
      if (!isWriteableField(s.anchor)) {
        refuse("session " + to_string(s.id) + " holds a value this format cannot write back");
      } else if (!add(s)) {
        refuse("session " + to_string(s.id) + " declared twice" + TWO_WRITERS);
      }
    } else if (tag == "run" || tag == "try" || tag == "nogo" || tag == "roe" || tag == "reach" || tag == "ref") {
      size_t const want = tag == "run" ? 14
        : tag == "try"                 ? 7
        : tag == "nogo"                ? 5
        : tag == "roe"                 ? 10
        : tag == "reach"               ? 9
                                       : 7;
      if (f.size() != want) {
        refuse(tag + " row has " + to_string(f.size()) + " fields, expected " + to_string(want));
        continue;
      }

      std::optional<u32> const sess = sessionOf();
      auto const fft = canonicalFft(f[2]);
      if (!fft) { refuse("'" + f[2] + "' is not an FFT spec"); }
      if (!sess || !fft) { continue; }

      if (tag == "run") {
        auto const kind = parseTestKind(f[3]);
        auto const exponent = parseInt<u64>(f[4]);
        auto const regime = parseRegime(f[5]);
        std::optional<u32> cfg;
        cfgOf(f[6], cfg);
        auto const mean = parseNonNegative(f[7]);
        auto const sd = parseNonNegative(f[8]);
        auto const blocks = parseInt<u32>(f[9]);
        auto const calls = parseInt<u32>(f[10]);
        auto const drift = parsePositive(f[11]);
        auto const status = parseStatus(f[12]);
        if (!kind) { refuse("'" + f[3] + "' is not a test kind"); }
        if (!exponent) { refuse("'" + f[4] + "' is not an exponent"); }
        if (!regime) { refuse("'" + f[5] + "' is not a regime"); }
        if (!mean || !sd || !drift) { refuse("run row has a malformed timing"); }
        if (!blocks || !calls) { refuse("run row has a malformed block or call count"); }
        if (!status) { refuse("'" + f[12] + "' is not a status"); }
        auto const ts = parseInt<u64>(f[13]);
        if (!ts) { refuse("'" + f[13] + "' is not a timestamp"); }
        if (!kind || !exponent || !regime || !cfg || !mean || !sd || !blocks || !calls || !drift || !status || !ts) {
          continue;
        }
        RunRow const row{.sess = *sess,
                         .fft = *fft,
                         .kind = *kind,
                         .exponent = *exponent,
                         .regime = *regime,
                         .cfg = *cfg,
                         .m = {.mean = *mean,
                               .stddev = *sd,
                               .blocks = *blocks,
                               .calls = *calls,
                               .drift = *drift,
                               .status = *status,
                               .ts = *ts}};
        if (!add(row)) { refuse("run row holds a value this format cannot write back"); }
      } else if (tag == "try") {
        auto const kind = parseTestKind(f[3]);
        auto const exponent = parseInt<u64>(f[4]);
        std::optional<u32> cfg;
        cfgOf(f[5], cfg);
        auto const ts = parseInt<u64>(f[6]);
        if (!kind) { refuse("'" + f[3] + "' is not a test kind"); }
        if (!exponent) { refuse("'" + f[4] + "' is not an exponent"); }
        if (!ts) { refuse("'" + f[6] + "' is not a timestamp"); }
        if (!kind || !exponent || !cfg || !ts) { continue; }
        tries_.push_back(
          TryRow{.sess = *sess, .fft = *fft, .kind = *kind, .exponent = *exponent, .cfg = *cfg, .ts = *ts});

      } else if (tag == "nogo") {
        auto const [key, val] = keyValue(f[3]);
        auto const ts = parseInt<u64>(f[4]);
        bool const shaped = f[3].find('=') != std::string::npos && !key.empty();
        if (!shaped) { refuse("'" + f[3] + "' is not a KEY=VALUE"); }
        if (!ts) { refuse("'" + f[4] + "' is not a timestamp"); }
        if (!shaped || !ts) { continue; }
        NogoRow const row{.sess = *sess, .fft = *fft, .key = std::string{key}, .val = std::string{val}, .ts = *ts};
        if (!add(row)) { refuse("'" + f[3] + "' cannot be written back as a KEY=VALUE"); }
      } else if (tag == "roe") {
        auto const exponent = parseInt<u64>(f[3]);
        std::optional<u32> cfg;
        cfgOf(f[4], cfg);
        auto const z = parseDouble(f[5]);
        auto const n = parseInt<u32>(f[6]);
        auto const maxRoe = parseNonNegative(f[7]);
        bool const checkOk = f[8] == "ok";
        auto const ts = parseInt<u64>(f[9]);
        if (!exponent) { refuse("'" + f[3] + "' is not an exponent"); }
        if (!z || !n || !maxRoe) { refuse("roe row has a malformed reading"); }
        if (f[8] != "ok" && f[8] != "fail") { refuse("'" + f[8] + "' is not ok or fail"); }
        if (!ts) { refuse("'" + f[9] + "' is not a timestamp"); }
        if (!exponent || !cfg || !z || !n || !maxRoe || (f[8] != "ok" && f[8] != "fail") || !ts) { continue; }
        RoeRow const row{.sess = *sess,
                         .fft = *fft,
                         .exponent = *exponent,
                         .cfg = *cfg,
                         .z = *z,
                         .n = *n,
                         .maxRoe = *maxRoe,
                         .checkOk = checkOk,
                         .ts = *ts};
        if (!add(row)) { refuse("roe row holds a value this format cannot write back"); }

      } else if (tag == "reach") {
        auto const kind = parseTestKind(f[3]);
        auto const regime = parseRegime(f[4]);
        std::optional<u32> cfg;
        cfgOf(f[5], cfg);
        auto const reach = parseInt<u64>(f[6]);
        auto const evidence = parseEvidence(f[7]);
        auto const ts = parseInt<u64>(f[8]);
        if (!kind) { refuse("'" + f[3] + "' is not a test kind"); }
        if (!regime) { refuse("'" + f[4] + "' is not a regime"); }
        if (!reach) { refuse("'" + f[6] + "' is not a reach"); }
        if (!evidence) { refuse("'" + f[7] + "' is not an evidence state"); }
        if (!ts) { refuse("'" + f[8] + "' is not a timestamp"); }
        if (!kind || !regime || !cfg || !reach || !evidence || !ts) { continue; }
        reaches_.push_back(ReachRow{.sess = *sess,
                                    .fft = *fft,
                                    .kind = *kind,
                                    .regime = *regime,
                                    .cfg = *cfg,
                                    .reach = *reach,
                                    .evidence = *evidence,
                                    .ts = *ts});

      } else {
        auto const exponent = parseInt<u64>(f[3]);
        auto const iters = parseInt<u64>(f[4]);
        auto const res64 = parseInt<u64>(f[5], 16);
        auto const ts = parseInt<u64>(f[6]);
        if (!exponent || !iters) { refuse("ref row has a malformed exponent or iteration count"); }
        if (!res64) { refuse("'" + f[5] + "' is not a residue"); }
        if (!ts) { refuse("'" + f[6] + "' is not a timestamp"); }
        if (!exponent || !iters || !res64 || !ts) { continue; }
        refs_.push_back(
          RefRow{.sess = *sess, .fft = *fft, .exponent = *exponent, .iters = *iters, .res64 = *res64, .ts = *ts});
      }
    } else {
      unknown_.push_back(line);
    }
  }

  if (!ok) { clear(); }
  return ok;
}

bool TuneDB::load(const fs::path& path) {
  clear();

  File file = File::openRead(path);
  if (!file) {
    std::error_code ec;
    if (!fs::exists(path, ec)) { return true; }
    log("Can't read '%s'\n", path.string().c_str());
    return false;
  }

  return parse(file.readAll(), path.string());
}

std::string TuneDB::text() const {
  std::string out = std::string{HEADER} + '\n';

  for (const DbEnv& e : envs_) { out += formatRow(e) + '\n'; }
  for (const auto& [id, config] : cfgs_) { out += formatCfgRow(id, config) + '\n'; }
  for (const SessRow& s : sessions_) { out += formatRow(s) + '\n'; }
  for (const RunRow& r : runs_) { out += formatRow(r) + '\n'; }
  for (const TryRow& r : tries_) { out += formatRow(r) + '\n'; }
  for (const NogoRow& r : nogos_) { out += formatRow(r) + '\n'; }
  for (const RoeRow& r : roes_) { out += formatRow(r) + '\n'; }
  for (const ReachRow& r : reaches_) { out += formatRow(r) + '\n'; }
  for (const RefRow& r : refs_) { out += formatRow(r) + '\n'; }
  for (const std::string& line : unknown_) { out += line + '\n'; }

  return out;
}

void TuneDB::save(const fs::path& path) const {
  CycleFile out{path};
  out->write(text());
}

}  // namespace tune
