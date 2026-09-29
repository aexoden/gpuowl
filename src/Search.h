// Copyright (C) Jason Lynch

// One entry's search: what it offers to measure next, from the rows of an env and what the search has already been
// through.  What the offers are worth, and which entry is served next, are decided by whoever asks: a search is given
// a price for each kind of offer and leaves out what is worth nothing, and otherwise ranks nothing.
//
// The configurations are the pure half's (Probe.h); what is kept here is what makes asking again cheap -- the probe
// lists already built and how far each has been answered, the restart draws and how far their sequence has been read
// -- and how often this process has tried each configuration without its row concluding.

#pragma once

#include "common.h"
#include "Eligibility.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "Probe.h"
#include "Stats.h"
#include "TuneDB.h"
#include "UseResolve.h"
#include "Value.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace tune {

// Enough calls for any configuration to conclude twice over.  One tried this often without concluding is failing in a
// way that records nothing, and trying it again would only repeat that.
inline constexpr u32 MAX_ATTEMPTS = 2 * MIN_CALLS;

// An entry: one configuration at the automatic carry, in one test kind and one regime band.
struct Baseline {
  FFTConfig fft;
  TestKind kind;

  // Where the entry would be eligible, and so the band whose workload weight its value is taken over.
  Interval band;

  // Where it is timed when nothing has been started on it: the probe where the band holds it, and otherwise the largest
  // prime the band reaches.
  u64 exponent;

  // "<spec> <kind> <regime>"
  [[nodiscard]] std::string label() const;

  [[nodiscard]] EntryKey key() const;
};

// The row a measurement resumes from: the exponent it was started at, and the calls it has.
struct Partial {
  u64 exponent = 0;
  u32 calls = 0;
};

// Where each entry stands, from the rows of an env.
struct Progress {
  // Settled by any concluded row: the baseline is the first measurement, and whatever measured it second is not one.
  // Not by the lines it was taken under, which move as entries are tuned: every row names all it ran, so one taken
  // under earlier lines is still a measurement of what it ran.
  std::set<EntryKey> settled;

  // A failure is a verdict on the options it was taken under, which a repeat would only repeat; by the canonical set.
  std::set<EntrySet> failed;

  // By the canonical option set as well: calls under other options do not pool with the ones a measurement will make.
  std::map<EntrySet, Partial> partial;

  // Every option set a row has concluded, canonical: a probe of that very set is answered by one, and a probe of the
  // same step from another best set is taken later for one.
  std::map<EntryKey, std::vector<UseConfig>> concluded;

  // The option sets a row has concluded or failed, and every option set each entry has rows of at all, with when the
  // last of them was written.
  std::set<EntrySet> answered;
  std::map<EntryKey, std::map<std::string, u64>> sets;
};

// Where each entry stands, from the rows `env` has.  A row the device lost under is not recorded, so it says nothing
// either way.
[[nodiscard]] Progress progressOf(const TuneDB& db, u32 env, const Env& device);

// Draws in a row that repeat a configuration already seen, after which a restart space is taken to be spent.  A space
// with one point left unseen among n survives this many draws with probability (1 - 1/n)^256: for n = 64, under 2%,
// and the next process starts the count again.
inline constexpr u32 RESTART_REPEATS = 256;

// How many option sets an entry measures between one restart draw and the next, whether or not it is at a local
// optimum: a descent can settle in a basin long before its steps run out, and the steps of a large stage would keep a
// restart waiting for as long as they last.
inline constexpr u32 RESTART_PERIOD = 32;

// How far one entry's restart sequence has been read.  Only what can never become runnable again is passed -- a draw a
// row answers, a hold, a draw given up on -- so the cursor never goes back, and a long stretch of such draws is read
// once rather than on every re-score.
struct RestartScan {
  u32 next = 0;
  u32 repeats = 0;
  bool exhausted = false;

  // The configurations drawn or measured so far, as their canonical text.
  std::set<std::string> seen;
};

// Moves `scan` to the first draw from where it stands that `runnable` accepts, and returns it; `text` names the
// configuration a draw is.  Nothing once RESTART_REPEATS draws in a row have repeated configurations in `seen`, which
// is what a space whose every point has been measured or ruled out looks like: a long run of new configurations that
// cannot run is the space still being explored, however long it is.
[[nodiscard]] std::optional<u32> nextRunnable(RestartScan& scan, const std::function<std::string(u32)>& text,
                                              const std::function<bool(u32)>& runnable);

// What a search offers.
enum class Offer : u8 {
  Lines,    // the default lines, all at once, alone or laid over the best set
  Probe,    // a step within a group, or of one key
  Combo,    // a point of a combination of the tier below's answers
  Restart,  // the next draw of the entry's restart sequence
};

struct Candidate {
  Offer kind = Offer::Probe;

  // What the configuration is built with.
  UseConfig options{};

  // The one key it moves, which a build failure can be pinned on, if there is one; and what it is, for the log.
  std::string moved{};
  std::string what{};

  // Where it is measured: the entry's timing exponent, or where a row of it was started.
  u64 exponent = 0;

  // Calls already recorded against it, which is what makes it a resumption.
  u32 calls = 0;

  // A restart's: which draw of the sequence it is.
  u32 draw = 0;

  // 2 or 3 for a combination, and 1 for anything else.
  u32 tier = 1;

  // The first offer from a stage listed only in part: at most how many more points the stage has.
  u64 unlisted = 0;

  // What the best set it is a step from costs, which the gain it is priced by is a gain on; and that price.
  double cost = 0;
  double value = 0;
};

// What one re-score gives every entry's search.
struct SearchContext {
  const Env& device;
  const Strategy& strategy;
  const TuneDB& db;
  u32 env;
  const Progress& progress;

  // The lines as they stand, which every entry tries first.
  const Defaults& lines;

  // Whether an entry jumps: once it has no step left, and once each RESTART_PERIOD option sets.
  bool restarts = false;
};

// What an offer of `kind` from a best set costing `cost` is worth.  A search leaves out what is worth nothing.
using Worth = std::function<double(Offer kind, double cost)>;

class EntrySearch {
public:
  explicit EntrySearch(Baseline entry);

  [[nodiscard]] const Baseline& entry() const { return entry_; }

  // What the entry offers now, in the order the search would take it.  `readings` are its publishable option sets,
  // canonical and cheapest first, so the first is its best set; there must be at least one.
  //
  // Offered first, but for a restart draw that is due: the lines, which carry what the best entries found, all at once,
  // the one jump most likely to pay before any single step; and where the entry's best set is not the built-in
  // defaults, that set with the lines laid over it.  Each is offered while no row has measured it, so as the lines move
  // each entry tries them again.  Then the steps of each structural branch the strategy searches, each from that
  // branch's best set -- only the entry's best set steps into other branches -- in the order probesOf() lists them, and
  // a combination only once its branch has nothing of a lower tier left, since it combines what those found; and after
  // all of those, the steps the entry's rows took from another best set, which may do otherwise from this one, priced
  // as a jump is (Offer::Restart), since what they did there is some evidence against them here.  The next draw of the
  // restart sequence at a local optimum of the declared moves, and ahead of everything else once the entry has measured
  // RESTART_PERIOD option sets since the last draw was declared, or until a draw begun is finished, priced then as a
  // step is.  Whatever sets a value a build of the FFT failed with (TuneDB::failedWith()) after everything else.
  // Nothing a row of that very configuration concluded or recorded a failure of, nothing an earlier generation's death
  // keeps out, and nothing tried MAX_ATTEMPTS times in this process.
  [[nodiscard]] std::vector<Candidate> offers(const SearchContext& context, std::span<const Reading> readings,
                                              const Worth& worth);

  // Records that a call was made of `options`, whatever came of it.
  void tried(const Env& device, const UseConfig& options);

private:
  // probesOf(), which is pure, for one best set and whether it steps into other branches, listing at most `listed`
  // points of each stage; each probe's configuration as text, and of the entry's rows checked against it so far,
  // whether one is of that configuration and whether one took the same step against another background (sameStep()).
  // A row that says either of a probe always will, so each is checked once.
  struct ListMemo {
    std::string from;
    u32 listed = PROBE_WINDOW;
    ProbeList list;
    std::vector<std::string> texts;
    std::vector<bool> answered;
    std::vector<bool> seen;
    std::set<std::string> checked;
  };

  // The memo for `best`, rebuilt when `from` -- what the readings of its branch say, which only the combo tiers read
  // -- changes: a best set changes rarely, and each re-score asks again.  With `widen`, rebuilt listing twice as many
  // points of each stage.  A probe the rebuilt list shares with the old one keeps what the rows answered it.
  [[nodiscard]] ListMemo& probeList(const SearchContext& context, const UseConfig& best,
                                    std::span<const Reading> readings, bool structuralSteps, std::string from,
                                    bool widen = false);

  // restartOf(), canonical: a draw takes one enumeration of the axes per axis.
  [[nodiscard]] const UseConfig& draw(const Env& device, u32 k);

  // The next draw worth making, if the space has anything left to offer.
  [[nodiscard]] std::optional<Candidate> nextRestart(const SearchContext& context);

  // Whether the entry has measured RESTART_PERIOD option sets since its last draw was declared, or since it was first
  // measured where none has been; or has begun its last draw and not finished it.
  [[nodiscard]] bool restartDue(const SearchContext& context) const;

  // Whether `options` is still worth asking for at `exponent`: not tried MAX_ATTEMPTS times in this process, and not
  // what an earlier generation died on there.
  [[nodiscard]] bool runnable(const SearchContext& context, const UseConfig& options, u64 exponent) const;

  // Where `canonical` is measured, and the calls a row started there has.
  void resume(const SearchContext& context, const std::string& canonical, Candidate& candidate) const;

  Baseline entry_;
  EntryKey key_;

  std::map<std::string, ListMemo> lists_;
  std::map<u32, UseConfig> draws_;
  RestartScan scan_;

  // Calls asked for, by canonical option set.
  std::map<std::string, u32> attempts_;
};

}  // namespace tune
