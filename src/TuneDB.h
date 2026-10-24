// Copyright (C) Jason Lynch

// The tuner's measurement database: one line per record, appended as it is taken, so that a run that stops for any
// reason has lost at most the call it was in.  Every record belongs to a session and every session to an env, which
// is one card under one driver, backend and kernel build.  Nothing the tuner decides is stored: every decision is a
// function of these records and the run's settings.

#pragma once

#include "common.h"
#include "File.h"
#include "Occupancy.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "TestKind.h"

#include <charconv>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// Whitespace-separated fields, with double quotes protecting one that contains spaces or tabs.  Nothing for a line
// that leaves a quote open.
[[nodiscard]] std::optional<std::vector<std::string>> splitFields(std::string_view line);

template<typename T> [[nodiscard]] std::optional<T> parseInt(std::string_view text, int base = 10) {
  T value{};
  auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (ec != std::errc{} || end != text.data() + text.size()) { return {}; }
  return value;
}

[[nodiscard]] std::optional<double> parseDouble(std::string_view text);

// An FFT spec as upstream spells it, under its canonical name (FFTVariants.h); nothing for one that names no FFT.
[[nodiscard]] std::optional<FFTConfig> parseFft(std::string_view spec);

// "KEY=VAL,KEY=VAL,..." in key order, or "-" for none.
[[nodiscard]] std::string configText(const UseConfig& config);
[[nodiscard]] std::optional<UseConfig> parseConfigText(std::string_view text);

// Whether every key and value survives being written into a record and read back.
[[nodiscard]] bool isWriteableConfig(const UseConfig& config);

struct DbEnv {
  u32 id = 0;
  std::string gpu{};     // the device name as the driver reports it
  std::string driver{};  // the driver version
  bool isAmd = false;
  bool isNvidia = false;
  bool cudaBackend = false;
  bool noAsm = false;
  bool pdlLaunch = false;
  u32 computeCapability = 0;
  bool hasFP64 = true;
  bool amdBuiltins = true;
  u32 maxWorkGroupSize = 1024;
  u32 amdWavefront = 0;
  std::string machine{};  // which physical card, as a PCI address; empty where the platform cannot tell
  u64 build = 0;          // the kernels that were measured (BuildId.h)

  [[nodiscard]] Env toEnv() const;

  // The same card under the same driver and backend, whatever kernels were built for it.
  [[nodiscard]] bool sameCard(const DbEnv& other) const;

  // Everything a reading's meaning depends on, the kernel build included: readings of two envs that differ here are
  // never compared.
  [[nodiscard]] bool sameMachine(const DbEnv& other) const { return sameCard(other) && build == other.build; }
};

[[nodiscard]] DbEnv dbEnvOf(const Env& env, std::string machine = {});

struct SessRow {
  u32 id = 0;
  u32 env = 0;
  u64 start = 0;
  u32 gen = 0;             // the restart generation that opened it
  std::string anchor{};    // "<spec>@<exponent>", the configuration its drift is measured against; empty for none
  std::string settings{};  // a tuning run's settings word, resolved; empty for any other session
};

// One timed call: a Gpu built for one configuration and timed over a number of blocks.  The independent observation
// the model learns from.
struct CallRow {
  u32 sess = 0;
  std::string fft{};
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  u32 cfg = 0;
  Status status = Status::Ok;
  double mean = 0;   // microseconds per iteration over the blocks kept, as measured
  double sd = 0;     // the kept blocks' standard deviation
  u32 blocks = 0;    // kept
  u32 dropped = 0;   // spikes left out
  double drift = 1;  // the anchor's ratio when the call was taken: mean / drift is comparable across sessions
  double buildSec = 0;
  double timedSec = 0;
  u64 ts = 0;

  [[nodiscard]] bool ok() const { return status == Status::Ok && blocks > 0; }
  [[nodiscard]] double cost() const { return mean / drift; }
};

// A configuration about to be built and run, closed by the next record of its session; one left open names what a
// process died holding.
struct TryRow {
  u32 sess = 0;
  std::string fft{};
  TestKind kind = TestKind::PRP;
  u64 exponent = 0;
  u32 cfg = 0;
  u64 ts = 0;
};

// A session that ended with an attempt open and nothing to say about it.
struct DoneRow {
  u32 sess = 0;
  u64 ts = 0;
};

// One rounding-error reading, taken on the PRP kernels whatever kind the configuration is published for.
struct RoeRow {
  u32 sess = 0;
  std::string fft{};
  u64 exponent = 0;
  u32 cfg = 0;
  bool checkOk = false;
  double z = 0;
  u32 n = 0;
  double maxRoe = 0;
  u64 ts = 0;
};

// One timing of the env's drift anchor.
struct AnchorRow {
  u32 sess = 0;
  std::string fft{};
  u64 exponent = 0;
  u32 cfg = 0;
  double mean = 0;
  double ratio = 1;  // against the env's first anchor reading
  u64 ts = 0;
};

// A REG*-limited kernel as a build that left it to the compiler compiled it, and what the occupancy model offers its
// key on this device.  The first such build of an FFT writes it, key by key.
struct RegsRow {
  u32 sess = 0;
  std::string fft{};
  TestKind kind = TestKind::PRP;
  std::string key{};
  KernelResources resources{};
  RegisterPlan plan{};
  u64 ts = 0;
};

// The residue an LL reading of `fft` reached after `iters` iterations from the seed.  Readings of one exponent and
// iteration count must agree whatever FFT and options produced them.
struct RefRow {
  u32 sess = 0;
  std::string fft{};
  u64 exponent = 0;
  u64 iters = 0;
  u64 res64 = 0;
  u64 ts = 0;
};

// A REG*-limited kernel's resources as one configuration built it, the key left to the compiler: written where they
// differ from every such reading of the key before, so that what structural options do to a kernel's registers can be
// seen, which the `regs` record of the first build cannot show.
struct KresRow {
  u32 sess = 0;
  std::string fft{};
  TestKind kind = TestKind::PRP;
  std::string key{};
  KernelResources resources{};
  u32 cfg = 0;
  u64 ts = 0;
};

// What the model expected of the item before it: the posterior mean's estimate, and the draw's that chose it (the
// same for a lane that asks no draw).  Written after the item record it belongs to.  The draw's is optimistic by its
// selection; the mean's is what calibration is measured by.
struct PredRow {
  u32 sess = 0;
  double mean = 0;  // microseconds per iteration
  double draw = 0;
  u64 ts = 0;
};

// What the tuning loop chose to measure next and why, written before the attempt: the lane it served, what kind of
// proposal it was, and for an item of a packet -- a run of calls that go together, such as a restart's local search --
// the packet and its place in it.  An item's time is until the next item of its session, or the session's last record.
struct ItemRow {
  u32 sess = 0;
  std::string lane{};  // a word
  std::string what{};  // a word
  u32 packet = 0;      // 0 for none
  u32 ordinal = 0;
  std::string fft{};
  TestKind kind = TestKind::PRP;
  u32 cfg = 0;
  u64 ts = 0;
};

class TuneDB {
public:
  static constexpr const char* DEFAULT_NAME = "tunedb.txt";
  static constexpr const char* HEADER = "# prpll tunedb v2";

  // Replaces the contents with `text`; false, having said why, for text that is not a database this build reads.
  [[nodiscard]] bool parse(std::string_view text, std::string_view name);

  // A path that does not exist is an empty database; one that exists and cannot be read is a failure.
  [[nodiscard]] bool load(const fs::path& path);

  [[nodiscard]] std::string text() const;

  // Claims the right to write this database, or returns false and says why.  Held until the process ends, and not
  // passed on to the image a restart executes.
  [[nodiscard]] bool lockForWriting(const fs::path& path);

  // Whether another process holds that right, found without taking it; nothing where the platform cannot tell.
  [[nodiscard]] static std::optional<bool> writerHolds(const fs::path& path);

  // Appends every record added from now on to `path`.  Call after load(), and only while holding the lock.
  void attach(const fs::path& path) { appendTo_ = path; }

  [[nodiscard]] const std::vector<DbEnv>& envs() const { return envs_; }
  [[nodiscard]] const std::map<u32, UseConfig>& cfgs() const { return cfgs_; }
  [[nodiscard]] const std::vector<SessRow>& sessions() const { return sessions_; }
  [[nodiscard]] const std::vector<CallRow>& calls() const { return calls_; }
  [[nodiscard]] const std::vector<TryRow>& tries() const { return tries_; }
  [[nodiscard]] const std::vector<RoeRow>& roes() const { return roes_; }
  [[nodiscard]] const std::vector<AnchorRow>& anchors() const { return anchors_; }
  [[nodiscard]] const std::vector<RegsRow>& regs() const { return regs_; }
  [[nodiscard]] const std::vector<RefRow>& refs() const { return refs_; }
  [[nodiscard]] const std::vector<ItemRow>& items() const { return items_; }
  [[nodiscard]] const std::vector<KresRow>& kres() const { return kres_; }
  [[nodiscard]] const std::vector<PredRow>& preds() const { return preds_; }
  [[nodiscard]] const std::vector<std::string>& unknownRows() const { return unknown_; }

  [[nodiscard]] const DbEnv* findEnv(u32 id) const;
  [[nodiscard]] const SessRow* findSession(u32 id) const;
  [[nodiscard]] const UseConfig* findCfg(u32 id) const;

  // The env a record belongs to, through its session; 0 where the session is unknown.
  [[nodiscard]] u32 envOf(u32 sess) const;

  // The id an equal env or option set already has, or a fresh one with its record written.  0 where it could not be
  // written.
  [[nodiscard]] u32 internEnv(const DbEnv& env);
  [[nodiscard]] u32 internCfg(const UseConfig& config);
  [[nodiscard]] u32 findCfgId(const UseConfig& config) const;

  // Opens a session on `env`; 0 where its record could not be written.
  [[nodiscard]] u32 beginSession(u32 env, const std::string& anchor, u32 gen, const std::string& settings,
                                 u64 start = 0);

  // Each returns false, having said why, where the record did not reach the file; it is then not kept either.
  [[nodiscard]] bool add(const CallRow& row);
  [[nodiscard]] bool add(const TryRow& row);
  [[nodiscard]] bool add(const DoneRow& row);
  [[nodiscard]] bool add(const RoeRow& row);
  [[nodiscard]] bool add(const AnchorRow& row);
  [[nodiscard]] bool add(const RegsRow& row);
  [[nodiscard]] bool add(const RefRow& row);
  // Unlike the others, an item answers no attempt: it is written before one.
  [[nodiscard]] bool add(const ItemRow& row);
  [[nodiscard]] bool add(const KresRow& row);
  [[nodiscard]] bool add(const PredRow& row);

  // The anchor an env is pinned to, as its earliest session that named one spells it, else as its earliest anchor
  // reading does; empty where it has none.
  [[nodiscard]] std::string envAnchor(u32 env) const;

  // The env's first reading of the anchor it is pinned to; nullptr where there is none.
  [[nodiscard]] const AnchorRow* envBaseline(u32 env) const;

  // The `regs` record of one key of an FFT in `env`, the first written; nullptr where there is none.
  [[nodiscard]] const RegsRow* findRegs(u32 env, const std::string& fft, TestKind kind, const std::string& key) const;

  // Every attempt a process died holding: open at the end of its session with nothing after it.
  [[nodiscard]] std::vector<TryRow> diedHolding() const;

  // Whether session `sess` has an attempt no record has answered yet.
  [[nodiscard]] bool attemptOpen(u32 sess) const { return open_.contains(sess); }

  // Whether a process died holding this configuration.
  [[nodiscard]] bool diedOn(u32 env, const std::string& fft, TestKind kind, u64 exponent, u32 cfg) const;

private:
  // Writes `line` when attached; false, having said why, where it did not land.
  [[nodiscard]] bool append(const std::string& line);

  // Canonicalises the record's FFT, writes it and keeps it if the write landed.
  template<typename Row> [[nodiscard]] bool record(std::vector<Row>& into, Row row);

  // Each record of a session answers the attempt it has open.
  void answered(u32 sess);

  void clear();

  fs::path appendTo_{};
  File lock_{};

  // Every record in file order, as spelled: what text() writes back.
  std::vector<std::string> lines_;

  std::vector<DbEnv> envs_;
  std::map<u32, UseConfig> cfgs_;
  std::vector<SessRow> sessions_;
  std::vector<CallRow> calls_;
  std::vector<TryRow> tries_;
  std::vector<RoeRow> roes_;
  std::vector<AnchorRow> anchors_;
  std::vector<RegsRow> regs_;
  std::vector<RefRow> refs_;
  std::vector<ItemRow> items_;
  std::vector<KresRow> kres_;
  std::vector<PredRow> preds_;
  std::vector<std::string> unknown_;

  // Per session, the attempt it has open, if any.
  std::map<u32, TryRow> open_;
};

[[nodiscard]] std::string formatRow(const DbEnv& row);
[[nodiscard]] std::string formatCfgRow(u32 id, const UseConfig& config);
[[nodiscard]] std::string formatRow(const SessRow& row);
[[nodiscard]] std::string formatRow(const CallRow& row);
[[nodiscard]] std::string formatRow(const TryRow& row);
[[nodiscard]] std::string formatRow(const DoneRow& row);
[[nodiscard]] std::string formatRow(const RoeRow& row);
[[nodiscard]] std::string formatRow(const AnchorRow& row);
[[nodiscard]] std::string formatRow(const RegsRow& row);
[[nodiscard]] std::string formatRow(const RefRow& row);
[[nodiscard]] std::string formatRow(const ItemRow& row);
[[nodiscard]] std::string formatRow(const KresRow& row);
[[nodiscard]] std::string formatRow(const PredRow& row);

// Whether `locks`, in the form of Linux's /proc/locks, lists a lock held on the file at (major, minor, inode).
[[nodiscard]] bool lockListed(std::string_view locks, u32 major, u32 minor, u64 inode);

}  // namespace tune
