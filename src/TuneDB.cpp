// Copyright (C) Jason Lynch

#include "TuneDB.h"

#include "BuildId.h"
#include "FFTVariants.h"
#include "File.h"
#include "fs.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <system_error>

#ifdef _WIN32
#include <io.h>
#include <sys/locking.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/sysmacros.h>
#endif
#endif

namespace tune {

namespace {

std::string quoted(const std::string& s) {
  bool const needs = s.empty() || s.find_first_of(" \t") != std::string::npos;
  return needs ? '"' + s + '"' : s;
}

std::optional<bool> parseBool(std::string_view text) {
  if (text == "0") { return false; }
  if (text == "1") { return true; }
  return {};
}

std::string fixed(double x, int digits) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", digits, x);
  return buf;
}

std::string hex16(u64 x) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%016" PRIx64, x);
  return buf;
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

// Digits with an optional K suffix, as FFT specs spell sizes.
std::optional<u32> parseSize(std::string_view text) {
  u32 multiple = 1;
  if (!text.empty() && (text.back() == 'K' || text.back() == 'k')) {
    multiple = 1024;
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

u64 now() { return u64(std::time(nullptr)); }

std::string planText(const RegisterPlan& plan) {
  std::string out;
  for (auto const& [blocks, regs] : plan.occupancies) {
    out += (out.empty() ? "" : "+") + std::to_string(blocks) + ':' + std::to_string(regs);
  }
  if (plan.lift) { out += out.empty() ? "-1" : "+-1"; }
  return out.empty() ? "-" : out;
}

std::optional<RegisterPlan> parsePlan(std::string_view k0, std::string_view text) {
  auto const base = parseInt<int>(k0);
  if (!base) { return {}; }
  RegisterPlan plan{.k0 = *base};
  if (text == "-") { return plan; }
  for (size_t at = 0; at <= text.size();) {
    size_t const plus = std::min(text.find('+', at), text.size());
    std::string_view const item = text.substr(at, plus - at);
    at = plus + 1;
    if (item == "-1") {
      plan.lift = true;
      continue;
    }
    size_t const colon = item.find(':');
    if (colon == std::string_view::npos) { return {}; }
    auto const blocks = parseInt<int>(item.substr(0, colon));
    auto const regs = parseInt<int>(item.substr(colon + 1));
    if (!blocks || !regs) { return {}; }
    plan.occupancies.emplace_back(*blocks, *regs);
  }
  return plan;
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

std::optional<FFTConfig> parseFft(std::string_view spec) {
  std::vector<std::string> parts = split(std::string{spec}, ':');

  enum FFT_TYPES type = FFT64;
  if (auto const lead = parseSize(parts.front()); lead && *lead < 256) {
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

  // Only what FFTShape can name: a spec it would refuse, or read as some other shape, names no FFT.
  FFTShape const shape{type, *width, *middle, *height};
  if (!std::ranges::any_of(FFTShape::allShapes(), [&](const FFTShape& s) { return s.spec() == shape.spec(); })) {
    return {};
  }

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

  return canonical(FFTConfig{shape, variant, carry});
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

  for (size_t at = 0; at <= text.size();) {
    size_t const comma = std::min(text.find(',', at), text.size());
    std::string_view const item = text.substr(at, comma - at);
    at = comma + 1;
    auto const [key, value] = keyValue(item);
    if (key.empty() || item.find('=') == std::string_view::npos) { return {}; }
    if (!config.emplace(std::string{key}, std::string{value}).second) { return {}; }
  }
  return config;
}

bool isWriteableConfig(const UseConfig& config) {
  auto const plain = [](std::string_view s) { return s.find_first_of(",= \t\"\n\r") == std::string_view::npos; };
  return std::ranges::all_of(config,
                             [&](const auto& kv) { return !kv.first.empty() && plain(kv.first) && plain(kv.second); });
}

Env DbEnv::toEnv() const {
  return Env{.isAmd = isAmd,
             .isNvidia = isNvidia,
             .cudaBackend = cudaBackend,
             .noAsm = noAsm,
             .hasFP64 = hasFP64,
             .amdBuiltins = amdBuiltins,
             .computeCapability = computeCapability,
             .pdlLaunch = pdlLaunch,
             .maxWorkGroupSize = maxWorkGroupSize,
             .amdWavefront = amdWavefront,
             .deviceName = gpu,
             .driverVersion = driver};
}

bool DbEnv::sameCard(const DbEnv& other) const {
  return gpu == other.gpu && driver == other.driver && isAmd == other.isAmd && isNvidia == other.isNvidia &&
    cudaBackend == other.cudaBackend && noAsm == other.noAsm && pdlLaunch == other.pdlLaunch &&
    computeCapability == other.computeCapability && hasFP64 == other.hasFP64 && amdBuiltins == other.amdBuiltins &&
    maxWorkGroupSize == other.maxWorkGroupSize && amdWavefront == other.amdWavefront && machine == other.machine;
}

DbEnv dbEnvOf(const Env& env, std::string machine) {
  return DbEnv{.gpu = env.deviceName,
               .driver = env.driverVersion,
               .isAmd = env.isAmd,
               .isNvidia = env.isNvidia,
               .cudaBackend = env.cudaBackend,
               .noAsm = env.noAsm,
               .pdlLaunch = env.pdlLaunch,
               .computeCapability = env.computeCapability,
               .hasFP64 = env.hasFP64,
               .amdBuiltins = env.amdBuiltins,
               .maxWorkGroupSize = env.maxWorkGroupSize,
               .amdWavefront = env.amdWavefront,
               .machine = std::move(machine),
               .build = buildFingerprint()};
}

std::string formatRow(const DbEnv& row) {
  const char* const vendor = row.isAmd ? "amd" : row.isNvidia ? "nvidia" : "-";
  return "env " + std::to_string(row.id) + " gpu=" + quoted(row.gpu) + " drv=" + quoted(row.driver) +
    " vendor=" + vendor + " be=" + (row.cudaBackend ? "cuda" : "ocl") + " cc=" + std::to_string(row.computeCapability) +
    " noasm=" + (row.noAsm ? "1" : "0") + " pdl=" + (row.pdlLaunch ? "1" : "0") + " fp64=" + (row.hasFP64 ? "1" : "0") +
    " builtins=" + (row.amdBuiltins ? "1" : "0") + " wg=" + std::to_string(row.maxWorkGroupSize) +
    " wave=" + std::to_string(row.amdWavefront) + " machine=" + (row.machine.empty() ? "-" : quoted(row.machine)) +
    " build=" + hex16(row.build);
}

std::string formatCfgRow(u32 id, const UseConfig& config) {
  return "cfg " + std::to_string(id) + ' ' + configText(config);
}

std::string formatRow(const SessRow& row) {
  std::string out = "sess " + std::to_string(row.id) + " env=" + std::to_string(row.env) +
    " start=" + std::to_string(row.start) + " gen=" + std::to_string(row.gen) +
    " anchor=" + (row.anchor.empty() ? "-" : row.anchor);
  if (!row.settings.empty()) { out += " settings=" + quoted(row.settings); }
  return out;
}

std::string formatRow(const CallRow& row) {
  return "call " + std::to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' +
    std::to_string(row.exponent) + ' ' + std::to_string(row.cfg) + ' ' + toString(row.status) + ' ' +
    fixed(row.mean, 3) + ' ' + fixed(row.sd, 3) + ' ' + std::to_string(row.blocks) + ' ' + std::to_string(row.dropped) +
    ' ' + fixed(row.drift, 4) + ' ' + fixed(row.buildSec, 2) + ' ' + fixed(row.timedSec, 2) + ' ' +
    std::to_string(row.ts);
}

std::string formatRow(const TryRow& row) {
  return "try " + std::to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' +
    std::to_string(row.exponent) + ' ' + std::to_string(row.cfg) + ' ' + std::to_string(row.ts);
}

std::string formatRow(const DoneRow& row) { return "done " + std::to_string(row.sess) + ' ' + std::to_string(row.ts); }

std::string formatRow(const RoeRow& row) {
  return "roe " + std::to_string(row.sess) + ' ' + row.fft + ' ' + std::to_string(row.exponent) + ' ' +
    std::to_string(row.cfg) + ' ' + (row.checkOk ? "ok" : "fail") + ' ' + fixed(row.z, 2) + ' ' +
    std::to_string(row.n) + ' ' + fixed(row.maxRoe, 6) + ' ' + std::to_string(row.ts);
}

std::string formatRow(const AnchorRow& row) {
  return "anchor " + std::to_string(row.sess) + ' ' + row.fft + ' ' + std::to_string(row.exponent) + ' ' +
    std::to_string(row.cfg) + ' ' + fixed(row.mean, 3) + ' ' + fixed(row.ratio, 4) + ' ' + std::to_string(row.ts);
}

std::string formatRow(const RegsRow& row) {
  const KernelResources& r = row.resources;
  return "regs " + std::to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + row.key + ' ' +
    std::to_string(r.regs) + ' ' + std::to_string(r.sharedBytes) + ' ' + std::to_string(r.threads) + ' ' +
    std::to_string(r.localBytes) + ' ' + std::to_string(row.plan.k0) + ' ' + planText(row.plan) + ' ' +
    std::to_string(row.ts);
}

std::string formatRow(const RefRow& row) {
  return "ref " + std::to_string(row.sess) + ' ' + row.fft + ' ' + std::to_string(row.exponent) + ' ' +
    std::to_string(row.iters) + ' ' + hex16(row.res64) + ' ' + std::to_string(row.ts);
}

std::string formatRow(const ItemRow& row) {
  return "item " + std::to_string(row.sess) + ' ' + row.lane + ' ' + row.what + ' ' + std::to_string(row.packet) + ' ' +
    std::to_string(row.ordinal) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + std::to_string(row.cfg) + ' ' +
    std::to_string(row.ts);
}

std::string formatRow(const KresRow& row) {
  const KernelResources& r = row.resources;
  return "kres " + std::to_string(row.sess) + ' ' + row.fft + ' ' + toString(row.kind) + ' ' + row.key + ' ' +
    std::to_string(r.regs) + ' ' + std::to_string(r.sharedBytes) + ' ' + std::to_string(r.threads) + ' ' +
    std::to_string(r.localBytes) + ' ' + std::to_string(row.cfg) + ' ' + std::to_string(row.ts);
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

u32 TuneDB::envOf(u32 sess) const {
  const SessRow* const s = findSession(sess);
  return s ? s->env : 0;
}

bool TuneDB::append(const std::string& line) {
  lines_.push_back(line);
  if (appendTo_.empty()) { return true; }

  std::error_code ec;
  bool const fresh = !fs::exists(appendTo_, ec) || fs::file_size(appendTo_, ec) == 0;

  bool ok = true;
  int failedWith = 0;
  {
    File out = File::openAppend(appendTo_);
    if (fresh) { ok = out.printf("%s\n", HEADER) >= 0; }
    ok = ok && out.printf("%s\n", line.c_str()) >= 0;
    // A buffered write can succeed and the flush behind it fail.
    out.flush();
    ok = ok && !ferror(out.get());
    if (!ok) { failedWith = errno; }
  }

  if (!ok) {
    lines_.pop_back();
    log("tune-db: could not write to '%s', so the record was not kept: %s\n", appendTo_.string().c_str(),
        strerror(failedWith));
  }
  return ok;
}

bool TuneDB::lockForWriting(const fs::path& path) {
  // A sibling rather than the database itself, which a rewrite may replace through a rename.
  fs::path const claimPath = path + ".lock";
  File claim = File::openAppend(claimPath);

#ifdef _WIN32
  int const fd = _fileno(claim.get());
  if (_lseek(fd, 0, SEEK_SET) == -1 || _locking(fd, _LK_NBLCK, 1)) {
    log("tune-db: another process is writing '%s', so this one will not touch it (%s)\n", path.string().c_str(),
        strerror(errno));
    return false;
  }
#else
  int const fd = fileno(claim.get());

  // Close on exec: a flock belongs to the open file description, and the recovery from a lost GPU is exec of this same
  // command (Restart.h), whose new image would otherwise be refused by its own predecessor's descriptor.
  if (int const flags = fcntl(fd, F_GETFD); flags == -1 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
    log("tune-db: could not claim '%s' safely across a restart (%s)\n", claimPath.string().c_str(), strerror(errno));
    return false;
  }
  if (flock(fd, LOCK_EX | LOCK_NB)) {
    log("tune-db: another process is writing '%s', so this one will not touch it (%s)\n", path.string().c_str(),
        strerror(errno));
    return false;
  }
#endif
  lock_ = std::move(claim);
  return true;
}

std::optional<bool> TuneDB::writerHolds(const fs::path& path) {
#ifdef __linux__
  // Read off the kernel's table of locks: any probe that takes the lock, even shared and for an instant, would turn
  // away a writer starting in that instant.
  struct stat claim{};
  if (stat((path + ".lock").c_str(), &claim)) {
    if (errno == ENOENT) { return false; }
    return {};
  }
  std::ifstream in{"/proc/locks"};
  if (!in) { return {}; }
  std::string const locks{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
  return lockListed(locks, major(claim.st_dev), minor(claim.st_dev), claim.st_ino);
#else
  (void)path;
  return {};
#endif
}

bool lockListed(std::string_view locks, u32 major, u32 minor, u64 inode) {
  for (size_t at = 0; at < locks.size();) {
    size_t const eol = std::min(locks.find('\n', at), locks.size());
    std::string_view const line = locks.substr(at, eol - at);
    at = eol + 1;

    std::vector<std::string_view> fields;
    for (size_t from = 0; from < line.size();) {
      size_t const space = std::min(line.find(' ', from), line.size());
      if (space > from) { fields.push_back(line.substr(from, space - from)); }
      from = space + 1;
    }

    // "<n>: <class> <mode> <type> <pid> <maj>:<min>:<inode> <start> <end>", with "->" after the number for a waiter.
    if (fields.size() < 6 || fields[1] == "->") { continue; }
    std::string_view const where = fields[5];
    size_t const first = where.find(':');
    size_t const second = first == std::string_view::npos ? first : where.find(':', first + 1);
    if (second == std::string_view::npos) { continue; }

    auto const maj = parseInt<u32>(where.substr(0, first), 16);
    auto const min = parseInt<u32>(where.substr(first + 1, second - first - 1), 16);
    auto const ino = parseInt<u64>(where.substr(second + 1));
    if (maj == major && min == minor && ino == inode) { return true; }
  }
  return false;
}

void TuneDB::answered(u32 sess) { open_.erase(sess); }

template<typename Row> bool TuneDB::record(std::vector<Row>& into, Row row) {
  auto const fft = parseFft(row.fft);
  if (!fft || !findSession(row.sess)) { return false; }
  row.fft = fft->spec();
  if (!append(formatRow(row))) { return false; }
  answered(row.sess);
  into.push_back(std::move(row));
  return true;
}

bool TuneDB::add(const CallRow& row) {
  if (!findCfg(row.cfg) || row.exponent == 0) { return false; }
  return record(calls_, row);
}

bool TuneDB::add(const TryRow& row) {
  if (!findCfg(row.cfg) || row.exponent == 0) { return false; }
  if (!record(tries_, row)) { return false; }
  open_[row.sess] = tries_.back();
  return true;
}

bool TuneDB::add(const DoneRow& row) {
  if (!findSession(row.sess) || !append(formatRow(row))) { return false; }
  answered(row.sess);
  return true;
}

bool TuneDB::add(const RoeRow& row) {
  if (!findCfg(row.cfg) || row.exponent == 0) { return false; }
  return record(roes_, row);
}

bool TuneDB::add(const AnchorRow& row) {
  if (!findCfg(row.cfg) || row.exponent == 0 || !(row.mean > 0) || !(row.ratio > 0)) { return false; }
  return record(anchors_, row);
}

bool TuneDB::add(const RegsRow& row) {
  if (row.key.empty() || row.key.find_first_of(" \t\"") != std::string::npos) { return false; }
  return record(regs_, row);
}

bool TuneDB::add(const RefRow& row) {
  if (row.exponent == 0 || row.iters == 0) { return false; }
  return record(refs_, row);
}

bool TuneDB::add(const KresRow& row) {
  if (row.key.empty() || row.key.find_first_of(" \t\"") != std::string::npos || !findCfg(row.cfg)) { return false; }
  return record(kres_, row);
}

bool TuneDB::add(const ItemRow& row) {
  auto const word = [](const std::string& w) { return !w.empty() && w.find_first_of(" \t\"") == std::string::npos; };
  auto const fft = parseFft(row.fft);
  if (!word(row.lane) || !word(row.what) || !fft || !findSession(row.sess) || !findCfg(row.cfg)) { return false; }
  ItemRow out = row;
  out.fft = fft->spec();
  if (!append(formatRow(out))) { return false; }
  items_.push_back(std::move(out));
  return true;
}

namespace {

template<typename Rows, typename Id> u32 nextId(const Rows& rows, Id id) {
  u32 top = 0;
  for (const auto& row : rows) { top = std::max(top, std::invoke(id, row)); }
  return top + 1;
}

}  // namespace

u32 TuneDB::internEnv(const DbEnv& env) {
  for (const DbEnv& e : envs_) {
    if (e.sameMachine(env)) { return e.id; }
  }
  DbEnv row = env;
  row.id = nextId(envs_, &DbEnv::id);
  if (!append(formatRow(row))) { return 0; }
  envs_.push_back(row);
  return row.id;
}

u32 TuneDB::findCfgId(const UseConfig& config) const {
  for (const auto& [id, c] : cfgs_) {
    if (c == config) { return id; }
  }
  return 0;
}

u32 TuneDB::internCfg(const UseConfig& config) {
  if (u32 const id = findCfgId(config)) { return id; }
  if (!isWriteableConfig(config)) { return 0; }
  u32 const id = cfgs_.empty() ? 1 : cfgs_.rbegin()->first + 1;
  if (!append(formatCfgRow(id, config))) { return 0; }
  cfgs_.emplace(id, config);
  return id;
}

u32 TuneDB::beginSession(u32 env, const std::string& anchor, u32 gen, const std::string& settings, u64 start) {
  if (!findEnv(env) || settings.find('"') != std::string::npos) { return 0; }
  SessRow const row{.id = nextId(sessions_, &SessRow::id),
                    .env = env,
                    .start = start ? start : now(),
                    .gen = gen,
                    .anchor = anchor,
                    .settings = settings};
  if (!append(formatRow(row))) { return 0; }
  sessions_.push_back(row);
  return row.id;
}

std::string TuneDB::envAnchor(u32 env) const {
  for (const SessRow& s : sessions_) {
    if (s.env == env && !s.anchor.empty()) { return s.anchor; }
  }
  for (const AnchorRow& a : anchors_) {
    if (envOf(a.sess) == env) { return a.fft + '@' + std::to_string(a.exponent); }
  }
  return {};
}

const AnchorRow* TuneDB::envBaseline(u32 env) const {
  std::string const anchor = envAnchor(env);
  if (anchor.empty()) { return nullptr; }
  for (const AnchorRow& a : anchors_) {
    if (envOf(a.sess) == env && a.fft + '@' + std::to_string(a.exponent) == anchor) { return &a; }
  }
  return nullptr;
}

const RegsRow* TuneDB::findRegs(u32 env, const std::string& fft, TestKind kind, const std::string& key) const {
  for (const RegsRow& r : regs_) {
    if (r.fft == fft && r.kind == kind && r.key == key && envOf(r.sess) == env) { return &r; }
  }
  return nullptr;
}

std::vector<TryRow> TuneDB::diedHolding() const {
  std::vector<TryRow> out;
  for (const auto& [sess, row] : open_) { out.push_back(row); }
  return out;
}

bool TuneDB::diedOn(u32 env, const std::string& fft, TestKind kind, u64 exponent, u32 cfg) const {
  return std::ranges::any_of(open_, [&](const auto& entry) {
    const TryRow& t = entry.second;
    return t.fft == fft && t.kind == kind && t.exponent == exponent && t.cfg == cfg && envOf(t.sess) == env;
  });
}

void TuneDB::clear() {
  lines_.clear();
  envs_.clear();
  cfgs_.clear();
  sessions_.clear();
  calls_.clear();
  tries_.clear();
  roes_.clear();
  anchors_.clear();
  regs_.clear();
  refs_.clear();
  items_.clear();
  kres_.clear();
  unknown_.clear();
  open_.clear();
}

namespace {

// The fields of a record kind this build reads, and how many.
struct Shape {
  const char* tag;
  size_t fields;
};

constexpr Shape SHAPES[] = {{"cfg", 3},    {"call", 15}, {"try", 7}, {"done", 3},  {"roe", 10},
                            {"anchor", 8}, {"regs", 12}, {"ref", 7}, {"item", 10}, {"kres", 11}};

}  // namespace

bool TuneDB::parse(std::string_view text, std::string_view name) {
  clear();
  fs::path const attached = std::exchange(appendTo_, {});

  std::vector<std::string_view> lines;
  for (size_t at = 0; at < text.size();) {
    size_t const eol = std::min(text.find('\n', at), text.size());
    lines.push_back(text.substr(at, eol - at));
    at = eol + 1;
  }
  bool const lastComplete = text.empty() || text.back() == '\n';

  bool good = true;
  size_t lineNo = 0;
  for (std::string_view const line : lines) {
    ++lineNo;
    bool const last = lineNo == lines.size();
    auto refuse = [&](const std::string& why) {
      // A last line with no end is a record the writer died in the middle of: dropped, not fatal.
      if (last && !lastComplete) {
        log("tune-db: %.*s:%zu: an unfinished last record is ignored (%s)\n", int(name.size()), name.data(), lineNo,
            why.c_str());
        return;
      }
      log("tune-db: %.*s:%zu: %s\n", int(name.size()), name.data(), lineNo, why.c_str());
      good = false;
    };

    if (lineNo == 1) {
      if (line != HEADER) {
        log("tune-db: %.*s is not a database this build reads (its first line is not '%s')\n", int(name.size()),
            name.data(), HEADER);
        appendTo_ = attached;
        return false;
      }
      continue;
    }
    if (line.empty() || line.front() == '#') {
      lines_.emplace_back(line);
      continue;
    }

    auto const fields = splitFields(line);
    if (!fields || fields->empty()) {
      refuse("unreadable record");
      continue;
    }
    const std::vector<std::string>& f = *fields;
    std::string const& tag = f[0];

    for (const Shape& s : SHAPES) {
      if (tag == s.tag && f.size() != s.fields) {
        refuse(tag + " record has " + std::to_string(f.size()) + " fields, expected " + std::to_string(s.fields));
      }
    }
    size_t const before = lines_.size();

    if (tag == "env") {
      DbEnv row;
      auto const id = f.size() > 1 ? parseInt<u32>(f[1]) : std::nullopt;
      bool ok = id && *id > 0 && !findEnv(*id);
      if (id) { row.id = *id; }
      for (size_t i = 2; ok && i < f.size(); ++i) {
        auto const [key, value] = keyValue(f[i]);
        auto setBool = [&](bool& to) {
          auto const b = parseBool(value);
          if (b) { to = *b; }
          return b.has_value();
        };
        auto setU32 = [&](u32& to) {
          auto const n = parseInt<u32>(value);
          if (n) { to = *n; }
          return n.has_value();
        };
        if (key == "gpu") {
          row.gpu = value;
        } else if (key == "drv") {
          row.driver = value;
        } else if (key == "vendor") {
          row.isAmd = value == "amd";
          row.isNvidia = value == "nvidia";
        } else if (key == "be") {
          ok = value == "cuda" || value == "ocl";
          row.cudaBackend = value == "cuda";
        } else if (key == "cc") {
          ok = setU32(row.computeCapability);
        } else if (key == "noasm") {
          ok = setBool(row.noAsm);
        } else if (key == "pdl") {
          ok = setBool(row.pdlLaunch);
        } else if (key == "fp64") {
          ok = setBool(row.hasFP64);
        } else if (key == "builtins") {
          ok = setBool(row.amdBuiltins);
        } else if (key == "wg") {
          ok = setU32(row.maxWorkGroupSize);
        } else if (key == "wave") {
          ok = setU32(row.amdWavefront);
        } else if (key == "machine") {
          row.machine = value == "-" ? "" : std::string{value};
        } else if (key == "build") {
          auto const b = parseInt<u64>(value, 16);
          ok = b.has_value();
          if (b) { row.build = *b; }
        }
      }
      if (!ok) {
        refuse("malformed env record");
      } else {
        envs_.push_back(row);
        lines_.emplace_back(line);
      }
    } else if (tag == "cfg" && f.size() == 3) {
      auto const id = parseInt<u32>(f[1]);
      auto const config = parseConfigText(f[2]);
      if (!id || *id == 0 || !config || cfgs_.contains(*id)) {
        refuse("malformed cfg record");
      } else {
        cfgs_.emplace(*id, *config);
        lines_.emplace_back(line);
      }
    } else if (tag == "sess") {
      SessRow row;
      auto const id = f.size() > 1 ? parseInt<u32>(f[1]) : std::nullopt;
      bool ok = id && *id > 0 && !findSession(*id);
      if (id) { row.id = *id; }
      bool sawEnv = false;
      for (size_t i = 2; ok && i < f.size(); ++i) {
        auto const [key, value] = keyValue(f[i]);
        if (key == "env") {
          auto const e = parseInt<u32>(value);
          ok = e && findEnv(*e);
          sawEnv = true;
          if (e) { row.env = *e; }
        } else if (key == "start") {
          auto const s = parseInt<u64>(value);
          ok = s.has_value();
          if (s) { row.start = *s; }
        } else if (key == "gen") {
          auto const g = parseInt<u32>(value);
          ok = g.has_value();
          if (g) { row.gen = *g; }
        } else if (key == "anchor") {
          row.anchor = value == "-" ? "" : std::string{value};
        } else if (key == "settings") {
          row.settings = value;
        }
      }
      if (!ok || !sawEnv) {
        refuse("malformed sess record");
      } else {
        sessions_.push_back(row);
        lines_.emplace_back(line);
      }
    } else if (tag == "call" && f.size() == 15) {
      auto const sess = parseInt<u32>(f[1]);
      auto const kind = parseTestKind(f[3]);
      auto const exponent = parseInt<u64>(f[4]);
      auto const cfg = parseInt<u32>(f[5]);
      auto const status = parseStatus(f[6]);
      auto const mean = parseNonNegative(f[7]);
      auto const sd = parseNonNegative(f[8]);
      auto const blocks = parseInt<u32>(f[9]);
      auto const dropped = parseInt<u32>(f[10]);
      auto const drift = parsePositive(f[11]);
      auto const build = parseNonNegative(f[12]);
      auto const timed = parseNonNegative(f[13]);
      auto const ts = parseInt<u64>(f[14]);
      if (!sess || !kind || !exponent || !cfg || !status || !mean || !sd || !blocks || !dropped || !drift || !build ||
          !timed || !ts ||
          !add(CallRow{.sess = *sess,
                       .fft = f[2],
                       .kind = *kind,
                       .exponent = *exponent,
                       .cfg = *cfg,
                       .status = *status,
                       .mean = *mean,
                       .sd = *sd,
                       .blocks = *blocks,
                       .dropped = *dropped,
                       .drift = *drift,
                       .buildSec = *build,
                       .timedSec = *timed,
                       .ts = *ts})) {
        refuse("malformed call record");
      }
    } else if (tag == "try" && f.size() == 7) {
      auto const sess = parseInt<u32>(f[1]);
      auto const kind = parseTestKind(f[3]);
      auto const exponent = parseInt<u64>(f[4]);
      auto const cfg = parseInt<u32>(f[5]);
      auto const ts = parseInt<u64>(f[6]);
      if (!sess || !kind || !exponent || !cfg || !ts ||
          !add(TryRow{.sess = *sess, .fft = f[2], .kind = *kind, .exponent = *exponent, .cfg = *cfg, .ts = *ts})) {
        refuse("malformed try record");
      }
    } else if (tag == "done" && f.size() == 3) {
      auto const sess = parseInt<u32>(f[1]);
      auto const ts = parseInt<u64>(f[2]);
      if (!sess || !ts || !add(DoneRow{.sess = *sess, .ts = *ts})) { refuse("malformed done record"); }
    } else if (tag == "roe" && f.size() == 10) {
      auto const sess = parseInt<u32>(f[1]);
      auto const exponent = parseInt<u64>(f[3]);
      auto const cfg = parseInt<u32>(f[4]);
      bool const checkKnown = f[5] == "ok" || f[5] == "fail";
      auto const z = parseDouble(f[6]);
      auto const n = parseInt<u32>(f[7]);
      auto const maxRoe = parseNonNegative(f[8]);
      auto const ts = parseInt<u64>(f[9]);
      if (!sess || !exponent || !cfg || !checkKnown || !z || !n || !maxRoe || !ts ||
          !add(RoeRow{.sess = *sess,
                      .fft = f[2],
                      .exponent = *exponent,
                      .cfg = *cfg,
                      .checkOk = f[5] == "ok",
                      .z = *z,
                      .n = *n,
                      .maxRoe = *maxRoe,
                      .ts = *ts})) {
        refuse("malformed roe record");
      }
    } else if (tag == "anchor" && f.size() == 8) {
      auto const sess = parseInt<u32>(f[1]);
      auto const exponent = parseInt<u64>(f[3]);
      auto const cfg = parseInt<u32>(f[4]);
      auto const mean = parsePositive(f[5]);
      auto const ratio = parsePositive(f[6]);
      auto const ts = parseInt<u64>(f[7]);
      if (!sess || !exponent || !cfg || !mean || !ratio || !ts ||
          !add(AnchorRow{.sess = *sess,
                         .fft = f[2],
                         .exponent = *exponent,
                         .cfg = *cfg,
                         .mean = *mean,
                         .ratio = *ratio,
                         .ts = *ts})) {
        refuse("malformed anchor record");
      }
    } else if (tag == "regs" && f.size() == 12) {
      auto const sess = parseInt<u32>(f[1]);
      auto const kind = parseTestKind(f[3]);
      auto const regs = parseInt<int>(f[5]);
      auto const shared = parseInt<int>(f[6]);
      auto const threads = parseInt<int>(f[7]);
      auto const local = parseInt<int>(f[8]);
      auto const plan = parsePlan(f[9], f[10]);
      auto const ts = parseInt<u64>(f[11]);
      if (!sess || !kind || !regs || !shared || !threads || !local || !plan || !ts ||
          !add(RegsRow{.sess = *sess,
                       .fft = f[2],
                       .kind = *kind,
                       .key = f[4],
                       .resources = {.regs = *regs, .localBytes = *local, .sharedBytes = *shared, .threads = *threads},
                       .plan = *plan,
                       .ts = *ts})) {
        refuse("malformed regs record");
      }
    } else if (tag == "ref" && f.size() == 7) {
      auto const sess = parseInt<u32>(f[1]);
      auto const exponent = parseInt<u64>(f[3]);
      auto const iters = parseInt<u64>(f[4]);
      auto const res64 = parseInt<u64>(f[5], 16);
      auto const ts = parseInt<u64>(f[6]);
      if (!sess || !exponent || !iters || !res64 || !ts ||
          !add(
            RefRow{.sess = *sess, .fft = f[2], .exponent = *exponent, .iters = *iters, .res64 = *res64, .ts = *ts})) {
        refuse("malformed ref record");
      }
    } else if (tag == "kres" && f.size() == 11) {
      auto const sess = parseInt<u32>(f[1]);
      auto const kind = parseTestKind(f[3]);
      auto const regs = parseInt<int>(f[5]);
      auto const shared = parseInt<int>(f[6]);
      auto const threads = parseInt<int>(f[7]);
      auto const local = parseInt<int>(f[8]);
      auto const cfg = parseInt<u32>(f[9]);
      auto const ts = parseInt<u64>(f[10]);
      if (!sess || !kind || !regs || !shared || !threads || !local || !cfg || !ts ||
          !add(KresRow{.sess = *sess,
                       .fft = std::string{f[2]},
                       .kind = *kind,
                       .key = std::string{f[4]},
                       .resources = {.regs = *regs, .localBytes = *local, .sharedBytes = *shared, .threads = *threads},
                       .cfg = *cfg,
                       .ts = *ts})) {
        refuse("malformed kres record");
      }
    } else if (tag == "item" && f.size() == 10) {
      auto const sess = parseInt<u32>(f[1]);
      auto const packet = parseInt<u32>(f[4]);
      auto const ordinal = parseInt<u32>(f[5]);
      auto const kind = parseTestKind(f[7]);
      auto const cfg = parseInt<u32>(f[8]);
      auto const ts = parseInt<u64>(f[9]);
      if (!sess || !packet || !ordinal || !kind || !cfg || !ts ||
          !add(ItemRow{.sess = *sess,
                       .lane = std::string{f[2]},
                       .what = std::string{f[3]},
                       .packet = *packet,
                       .ordinal = *ordinal,
                       .fft = std::string{f[6]},
                       .kind = *kind,
                       .cfg = *cfg,
                       .ts = *ts})) {
        refuse("malformed item record");
      }
    } else if (std::ranges::none_of(SHAPES, [&](const Shape& s) { return tag == s.tag; })) {
      // A record kind a later build writes: kept, and written back where it was.
      unknown_.emplace_back(line);
      lines_.emplace_back(line);
    }

    // add() wrote the canonical form; the file keeps the record as it was spelled.
    if (lines_.size() > before) { lines_.back() = std::string{line}; }
  }

  appendTo_ = attached;
  if (!good) { clear(); }
  return good;
}

bool TuneDB::load(const fs::path& path) {
  File file = File::openRead(path);
  if (!file) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
      clear();
      return true;
    }
    log("tune-db: can't read '%s'\n", path.string().c_str());
    return false;
  }
  return parse(file.readAll(), path.string());
}

std::string TuneDB::text() const {
  std::string out = std::string{HEADER} + '\n';
  for (const std::string& line : lines_) { out += line + '\n'; }
  return out;
}

}  // namespace tune
