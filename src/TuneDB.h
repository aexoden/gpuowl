// Copyright (C) Jason Lynch

// Measurement database for the tuning system.
//
// Line-oriented text, one record per line. Every row belongs to a session, every session to an env.

#pragma once

#include "common.h"
#include "Eligibility.h"
#include "OptionSpace.h"
#include "UseResolve.h"

#include <charconv>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// Whitespace-separated fields, with double quotes protecting one that contains spaces or tabs. Empty for a line with no
// fields at all; nothing for a line that leaves a quote open.
[[nodiscard]] std::optional<std::vector<std::string>> splitFields(std::string_view line);

template<typename T> [[nodiscard]] std::optional<T> parseInt(std::string_view text, int base = 10) {
  T value{};
  auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (ec != std::errc{} || end != text.data() + text.size()) { return {}; }
  return value;
}

[[nodiscard]] std::optional<double> parseDouble(std::string_view text);
[[nodiscard]] std::optional<double> parseNonNegative(std::string_view text);
[[nodiscard]] std::optional<double> parsePositive(std::string_view text);

// Whether a value survives being written and read back.
[[nodiscard]] bool isWriteableField(std::string_view value);
[[nodiscard]] bool isWriteableConfig(const UseConfig& config);

[[nodiscard]] std::optional<FFTConfig> parseFft(std::string_view spec);

enum class Status : u8 { Ok, Err, NoCompile, Unsupported, Lost };

[[nodiscard]] const char* toString(Status status);
[[nodiscard]] std::optional<Status> parseStatus(std::string_view text);

enum class Evidence : u8 { Unvalidated, Confirmed, Rejected, Unavailable, NotApplicable };

[[nodiscard]] const char* toString(Evidence evidence);
[[nodiscard]] std::optional<Evidence> parseEvidence(std::string_view text);

struct DbEnv {
  u32 id = 0;
  std::string gpu;   // the device name as the driver reports it
  std::string name;  // the board name, where the platform reports one separately
  std::string driver;
  bool isAmd = false;
  bool isNvidia = false;
  bool cudaBackend = false;
  bool noAsm = false;
  bool pdlLaunch = false;
  u32 computeCapability = 0;

  // Which physical card, as a PCI address; empty for a machine that cannot tell.
  std::string machine{};

  // A hash of the kernel sources; placeholder for now.
  u64 build = 0;

  // Fields of this row that weren't recognized.
  std::vector<std::string> extra{};

  [[nodiscard]] Env toEnv() const;

  // Everything a measurement's validity depends on, the kernel build included: two rows that disagree may not be
  // compared, even where the card is the same one.
  [[nodiscard]] bool sameMachine(const DbEnv& other) const;
};

[[nodiscard]] DbEnv dbEnvOf(const Env& env);

struct SessRow {
  u32 id = 0;
  u32 env = 0;
  u64 start = 0;
  u32 gen = 0;

  std::string anchor;  // "<spec>@<exponent>": the configuration this session's drift is measured against

  // The session had a drift alarm.
  bool alarmed = false;
};

// One timing with its error bar, in microseconds per iteration.
struct Measurement {
  double mean = 0;
  double stddev = 0;
  u32 blocks = 0;
  u32 calls = 0;
  double drift = 1;
  Status status = Status::Ok;
  u64 ts = 0;
};

// One timing of one configuration at one exponent.
struct RunRow {
  u32 sess = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  Regime regime{};
  u32 cfg = 0;
  Measurement m;
};

// A configuration about to be built and run; useful for tracing a configuration that caused a crash.
struct TryRow {
  u32 sess = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  u32 cfg = 0;
  u64 ts = 0;
};

// One key=value that will not build on one FFT.
struct NogoRow {
  u32 sess = 0;
  std::string fft;
  std::string key;
  std::string val;
  u64 ts = 0;
};

struct RoeRow {
  u32 sess = 0;
  std::string fft;
  u64 exponent = 0;
  u32 cfg = 0;
  double z = 0;
  u32 n = 0;
  double maxRoe = 0;
  bool checkOk = false;
  u64 ts = 0;
};

struct ReachRow {
  u32 sess = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  u32 cfg = 0;
  u64 reach = 0;
  Evidence evidence = Evidence::Unvalidated;
  u64 ts = 0;
};

// One LL residue reading.
struct RefRow {
  u32 sess = 0;
  std::string fft;
  u64 exponent = 0;
  u64 iters = 0;
  u64 res64 = 0;
  u64 ts = 0;
};

class TuneDB {
public:
  static constexpr const char* DEFAULT_NAME = "tunedb.txt";
  static constexpr const char* HEADER = "# prpll tunedb v1";

  // Replaces the contents with `text`. Returns false on failure.
  [[nodiscard]] bool parse(std::string_view text, std::string_view name);

  // Reads the file. A path that does not exist is an empty database; one that exists and cannot be read is a failure.
  [[nodiscard]] bool load(const fs::path& path);

  // The database as it would be written.
  [[nodiscard]] std::string text() const;

  // Rewrites the whole file, through a temporary and a rename, ensuring atomicity. Assumes only a single process is
  // writing to the file.
  void save(const fs::path& path) const;

  [[nodiscard]] const std::vector<DbEnv>& envs() const { return envs_; }
  [[nodiscard]] const std::map<u32, UseConfig>& cfgs() const { return cfgs_; }
  [[nodiscard]] const std::vector<SessRow>& sessions() const { return sessions_; }
  [[nodiscard]] const std::vector<RunRow>& runs() const { return runs_; }
  [[nodiscard]] const std::vector<TryRow>& tries() const { return tries_; }
  [[nodiscard]] const std::vector<NogoRow>& nogos() const { return nogos_; }
  [[nodiscard]] const std::vector<RoeRow>& roes() const { return roes_; }
  [[nodiscard]] const std::vector<ReachRow>& reaches() const { return reaches_; }
  [[nodiscard]] const std::vector<RefRow>& refs() const { return refs_; }
  [[nodiscard]] const std::vector<std::string>& unknownRows() const { return unknown_; }

  [[nodiscard]] const DbEnv* findEnv(u32 id) const;
  [[nodiscard]] const SessRow* findSession(u32 id) const;
  [[nodiscard]] const UseConfig* findCfg(u32 id) const;

  [[nodiscard]] bool add(const DbEnv& row);
  [[nodiscard]] bool addCfg(u32 id, UseConfig config);
  [[nodiscard]] bool add(const SessRow& row);
  [[nodiscard]] bool add(const RunRow& row);
  [[nodiscard]] bool add(const TryRow& row);
  [[nodiscard]] bool add(const NogoRow& row);
  [[nodiscard]] bool add(const RoeRow& row);
  [[nodiscard]] bool add(const ReachRow& row);
  [[nodiscard]] bool add(const RefRow& row);

private:
  void clear();

  std::vector<DbEnv> envs_;
  std::map<u32, UseConfig> cfgs_;
  std::vector<SessRow> sessions_;
  std::vector<RunRow> runs_;
  std::vector<TryRow> tries_;
  std::vector<NogoRow> nogos_;
  std::vector<RoeRow> roes_;
  std::vector<ReachRow> reaches_;
  std::vector<RefRow> refs_;
  std::vector<std::string> unknown_;
};

[[nodiscard]] std::string formatRow(const DbEnv& row);
[[nodiscard]] std::string formatCfgRow(u32 id, const UseConfig& config);
[[nodiscard]] std::string formatRow(const SessRow& row);
[[nodiscard]] std::string formatRow(const RunRow& row);
[[nodiscard]] std::string formatRow(const TryRow& row);
[[nodiscard]] std::string formatRow(const NogoRow& row);
[[nodiscard]] std::string formatRow(const RoeRow& row);
[[nodiscard]] std::string formatRow(const ReachRow& row);
[[nodiscard]] std::string formatRow(const RefRow& row);

[[nodiscard]] std::string configText(const UseConfig& config);
[[nodiscard]] std::optional<UseConfig> parseConfigText(std::string_view text);

}  // namespace tune
