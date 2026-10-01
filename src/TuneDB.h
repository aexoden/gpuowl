// Copyright (C) Jason Lynch

// Measurement database for the tuning system.
//
// Line-oriented text, one record per line. Every row belongs to a session, every session to an env.

#pragma once

#include "common.h"
#include "Eligibility.h"
#include "File.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "UseResolve.h"

#include <charconv>
#include <filesystem>
#include <map>
#include <set>
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

  // What a command with no device open cannot ask the card: which FFT types and variants it can build at all.
  bool hasFP64 = true;
  bool amdBuiltins = true;

  // Which physical card, as a PCI address; empty for a machine that cannot tell.
  std::string machine{};

  // Which kernels were measured, as BuildId.h's fingerprint of their sources.
  u64 build = 0;

  // Fields of this row that weren't recognized.
  std::vector<std::string> extra{};

  [[nodiscard]] Env toEnv() const;

  // Everything a measurement's validity depends on, the kernel build included: two rows that disagree may not be
  // compared, even where the card is the same one.
  [[nodiscard]] bool sameMachine(const DbEnv& other) const;

  // The same physical card under the same driver and backend, whatever kernels were built for it.  This is the
  // relation `adopt` works across, and the only one it works across.
  [[nodiscard]] bool sameCard(const DbEnv& other) const;
};

[[nodiscard]] DbEnv dbEnvOf(const Env& env);

struct SessRow {
  u32 id = 0;
  u32 env = 0;
  u64 start = 0;
  u32 gen = 0;

  std::string anchor;  // "<spec>@<exponent>": the configuration this session's drift is measured against

  // A tuning run's settings, resolved: what it valued its items against, so that its standing can be reported as it
  // saw it.  Empty for a session that was not a tuning run.
  std::string tune{};

  // The session had a drift alarm.
  bool alarmed = false;
};

// Pending work a tuning run was scoped against: `count` assignments of one kind at one exponent.  What its grid was
// weighted by, which a later reader has no other way to recover once the worktodo has moved on.
struct WorkRow {
  u32 sess = 0;
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  u32 count = 0;
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

// A session that declared an attempt and then ended with nothing to say about it.
struct DoneRow {
  u32 sess = 0;
  u64 ts = 0;
};

// One key=value a build of one FFT failed with when it alone had moved.
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

  // A hash of the error samples behind z, which the statistics above only summarise: two readings with equal ones
  // rounded identically.  0 where the reading carries none.
  u64 fp = 0;

  u64 ts = 0;
};

// One timing of the session's drift anchor.
struct AnchorRow {
  u32 sess = 0;
  std::string fft;
  u64 exponent = 0;
  u32 cfg = 0;
  double mean = 0;

  // Against the first reading taken for this env, which is what a later session divides its own rows by.
  double ratio = 1;

  u64 ts = 0;
};

// A session whose anchor stayed past the drift alarm.  Folded into the session row it names, which is where a rewrite
// puts it: a session's row is written before its anchor has ever been timed, so the flag cannot be set there in place.
struct AlarmRow {
  u32 sess = 0;
  u64 ts = 0;
};

// A restart: the `k`th draw of an entry's restart sequence, declared before its first call, so that its rows can be
// told apart from the moves of the search whatever the workload and the defaults are later.
struct JumpRow {
  u32 sess = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  u32 cfg = 0;
  u32 k = 0;
  u64 ts = 0;
};

// A combination of the tier below's best answers (tier 2 or 3), declared before its first call, so that what the device
// learns from combinations is kept apart from what it learns from single moves.
struct ComboRow {
  u32 sess = 0;
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  u32 cfg = 0;
  u32 tier = 0;
  u64 ts = 0;
};

// Which configuration a bootstrap family is searched on at one probe, chosen once and kept: after the defaults sweep
// the cheapest reading of the type there.  Recorded, since the choice reads measurements that later calls keep adding
// to, and a search that moved would begin again.
struct BootRow {
  u32 sess = 0;
  std::string fft;
  u64 probe = 0;
  u64 ts = 0;
};

// One entry of a round of the halving, and the calls of search it had had when the round began, which its calls in the
// round are counted from.
struct RoundMember {
  std::string fft;
  TestKind kind = TestKind::PRP;
  Regime regime{};
  u64 from = 0;

  bool operator==(const RoundMember&) const = default;
};

// A round of the halving as it began: its entries, and the calls of search each is to have in it.  Recorded, since the
// calls a round is made of move the gaps its pool would be chosen by, and a pool chosen afresh would drop an entry
// before it had its calls.  A round of one entry ends its halving.  A round of none says only that the env's rounds are
// recorded from here on, so that none is inferred from calls made before.
struct RoundRow {
  u32 sess = 0;

  // The round's place among the env's rounds, from 1.
  u32 n = 0;

  // Its place in its halving, from 1; 0 for a round of no entries.
  u32 round = 0;

  u64 calls = 0;
  u64 ts = 0;
  std::vector<RoundMember> members;
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

  // Claims the right to write this database, or returns false and says why.
  [[nodiscard]] bool lockForWriting(const fs::path& path);

  // Whether another process holds that right now, found without taking any lock, so that asking can never turn a
  // writer away; nothing where this platform cannot tell.
  [[nodiscard]] static std::optional<bool> writerHolds(const fs::path& path);

  // Appends every row added from now on to `path`, as it is added. Call after load(), and only while holding the lock
  // above.
  void attach(const fs::path& path);
  [[nodiscard]] bool attached() const { return !appendTo_.empty(); }

  [[nodiscard]] const std::vector<DbEnv>& envs() const { return envs_; }
  [[nodiscard]] const std::map<u32, UseConfig>& cfgs() const { return cfgs_; }
  [[nodiscard]] const std::vector<SessRow>& sessions() const { return sessions_; }
  [[nodiscard]] const std::vector<WorkRow>& works() const { return works_; }
  [[nodiscard]] const std::vector<RunRow>& runs() const { return runs_; }
  [[nodiscard]] const std::vector<TryRow>& tries() const { return tries_; }
  [[nodiscard]] const std::vector<NogoRow>& nogos() const { return nogos_; }
  [[nodiscard]] const std::vector<RoeRow>& roes() const { return roes_; }
  [[nodiscard]] const std::vector<AnchorRow>& anchors() const { return anchors_; }
  [[nodiscard]] const std::vector<RefRow>& refs() const { return refs_; }
  [[nodiscard]] const std::vector<JumpRow>& jumps() const { return jumps_; }
  [[nodiscard]] const std::vector<ComboRow>& combos() const { return combos_; }
  [[nodiscard]] const std::vector<BootRow>& boots() const { return boots_; }
  [[nodiscard]] const std::vector<RoundRow>& rounds() const { return rounds_; }
  [[nodiscard]] const std::vector<std::string>& unknownRows() const { return unknown_; }

  // One row per distinct measurement, duplicates folded as a running mean and pooled variance with the calls summed
  // and failure statuses sticky.  The rows themselves stay as the file spelled them: what merges is the reading, not
  // the record of which session took it.  A row whose options use an access mode the table has since withdrawn is
  // left out, as if it had never been taken.
  [[nodiscard]] std::vector<RunRow> mergedRuns() const;

  // The surviving reading for each key.  A later roe replaces the earlier one rather than pooling with it: it is
  // evidence about a configuration, not a sample.
  [[nodiscard]] std::vector<RoeRow> latestRoes() const;

  // Replaces the rows with the folded views above and drops option sets no surviving row names.
  [[nodiscard]] bool compact();

  // Drops everything measured on `env`, or on one shape of it: the timings, the evidence derived from them, the keys
  // recorded as unbuildable and the attempts standing against the env.  A kernel change can undo any of those, which
  // is what makes this worth asking for.  The env and its sessions stay, so the env keeps the anchor it is pinned to.
  [[nodiscard]] bool reset(u32 env, std::string_view fft = {});

  // Restamps every row of `from` as a row of `into`, which is the user saying that the kernels that moved were not
  // the ones these rows measured.  Refused unless the two envs are the same card: a different card's rows were never
  // comparable, whatever was built for it.  Rows the two have in common merge from then on like any duplicates.
  [[nodiscard]] bool adopt(u32 from, u32 into);

  // The env `adopt` takes for `into` when the user names none: the most recent one that is this card under other
  // kernels.  0 when there is none.
  [[nodiscard]] u32 adoptCandidate(u32 into) const;

  [[nodiscard]] const DbEnv* findEnv(u32 id) const;
  [[nodiscard]] const SessRow* findSession(u32 id) const;
  [[nodiscard]] const UseConfig* findCfg(u32 id) const;

  [[nodiscard]] bool add(const DbEnv& row);
  [[nodiscard]] bool addCfg(u32 id, UseConfig config);
  [[nodiscard]] bool add(const SessRow& row);
  [[nodiscard]] bool add(const WorkRow& row);
  [[nodiscard]] bool add(const RunRow& row);
  [[nodiscard]] bool add(const TryRow& row);
  [[nodiscard]] bool add(const NogoRow& row);
  [[nodiscard]] bool add(const RoeRow& row);
  [[nodiscard]] bool add(const AnchorRow& row);
  [[nodiscard]] bool add(const AlarmRow& row);
  [[nodiscard]] bool add(const RefRow& row);
  [[nodiscard]] bool add(const JumpRow& row);
  [[nodiscard]] bool add(const ComboRow& row);
  [[nodiscard]] bool add(const BootRow& row);
  [[nodiscard]] bool add(const RoundRow& row);
  [[nodiscard]] bool add(const DoneRow& row);

  // The id an identical entry already has, or a fresh one.
  [[nodiscard]] u32 internEnv(const DbEnv& env);
  [[nodiscard]] u32 internCfg(const UseConfig& config);

  // The id this configuration already has, or 0.
  [[nodiscard]] u32 findCfgId(const UseConfig& config) const;

  // Opens a session on `env`. `start` is its wall-clock time, or 0 for now.
  [[nodiscard]] u32 beginSession(u32 env, const std::string& anchor, u32 gen = 0, u64 start = 0,
                                 const std::string& tune = {});

  // The anchor this env is pinned to, as its earliest session that named one spells it, or else as its earliest anchor
  // reading does; empty when it has none. An env compares its rows against readings of one configuration at one
  // exponent, so the first session to time an anchor fixes it for the rest of the env's life.
  [[nodiscard]] std::string envAnchor(u32 env) const;

  // The first anchor reading taken for this env under the anchor it is pinned to; nullptr when there is none.
  [[nodiscard]] const AnchorRow* envBaseline(u32 env) const;

  // The env a row belongs to, through its session; 0 when the session is unknown.
  [[nodiscard]] u32 envOf(u32 sess) const;

  void closeTry(u32 sess);

  // Stops `sess` writing anything further, for good.
  void sealSession(u32 sess);
  [[nodiscard]] bool isSealed(u32 sess) const;

  // Every attempt this database has standing against it.
  [[nodiscard]] std::vector<TryRow> diedHolding() const;

  // Whether this configuration was what a previous generation was holding when it died.
  [[nodiscard]] bool diedOn(u32 env, u32 cfg, TestKind kind, const std::string& fft, u64 exponent) const;

  // Whether `config` sets a key to a value a build of `fft` failed with in this env when that key alone had moved: a
  // hint that it may fail again, which orders what is tried.  The failure itself is a verdict on the configuration that
  // failed, and on nothing else.
  [[nodiscard]] bool failedWith(u32 env, const std::string& fft, const UseConfig& config) const;

private:
  // Whether the database may be rewritten in place, saying why not when it may not.
  [[nodiscard]] bool rewritable(const char* what) const;

  void clear();

  // Maintains the open-attempt state a row implies.
  void noteTry(const TryRow& row);
  void noteRow(u32 sess, u64 ts);

  // False if the line did not reach the file, having said so. Silently true when nothing is attached.
  [[nodiscard]] bool append(const std::string& line);

  // Canonicalises `row`'s FFT spec, writes it, and keeps it only if the write landed.
  template<typename Row> [[nodiscard]] bool record(std::vector<Row>& into, Row row);

  fs::path appendTo_;

  File lock_;

  std::vector<DbEnv> envs_;
  std::map<u32, UseConfig> cfgs_;
  std::vector<SessRow> sessions_;
  std::vector<WorkRow> works_;
  std::vector<RunRow> runs_;
  std::vector<TryRow> tries_;
  std::vector<NogoRow> nogos_;
  std::vector<RoeRow> roes_;
  std::vector<AnchorRow> anchors_;
  std::vector<RefRow> refs_;
  std::vector<JumpRow> jumps_;
  std::vector<ComboRow> combos_;
  std::vector<BootRow> boots_;
  std::vector<RoundRow> rounds_;
  std::vector<std::string> unknown_;

  std::map<u32, TryRow> open_;
  std::map<u32, u64> answered_;

  std::set<u32> sealed_;

  std::set<u32> live_;
};

[[nodiscard]] std::string formatRow(const DbEnv& row);
[[nodiscard]] std::string formatCfgRow(u32 id, const UseConfig& config);
[[nodiscard]] std::string formatRow(const SessRow& row);
[[nodiscard]] std::string formatRow(const WorkRow& row);
[[nodiscard]] std::string formatRow(const RunRow& row);
[[nodiscard]] std::string formatRow(const TryRow& row);
[[nodiscard]] std::string formatRow(const NogoRow& row);
[[nodiscard]] std::string formatRow(const RoeRow& row);
[[nodiscard]] std::string formatRow(const AnchorRow& row);
[[nodiscard]] std::string formatRow(const AlarmRow& row);
[[nodiscard]] std::string formatRow(const RefRow& row);
[[nodiscard]] std::string formatRow(const JumpRow& row);
[[nodiscard]] std::string formatRow(const ComboRow& row);
[[nodiscard]] std::string formatRow(const BootRow& row);
[[nodiscard]] std::string formatRow(const RoundRow& row);
[[nodiscard]] std::string formatRow(const DoneRow& row);

// Whether `locks`, in the form of Linux's /proc/locks, lists a lock held on the file at (major, minor, inode); a
// process waiting for one does not hold it.
[[nodiscard]] bool lockListed(std::string_view locks, u32 major, u32 minor, u64 inode);

[[nodiscard]] std::string configText(const UseConfig& config);
[[nodiscard]] std::optional<UseConfig> parseConfigText(std::string_view text);

}  // namespace tune
