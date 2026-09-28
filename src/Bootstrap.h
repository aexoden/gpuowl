// Copyright (C) Jason Lynch

// The global bootstrap: for each FFT type worth tuning, the option set every configuration of that type runs at until
// it has been tuned itself, found by racing the option groups on one configuration at the exponent that matters most,
// and then combinations of the groups' best answers.  The whole combination tree, whatever the run searches its
// entries with: every entry not yet tuned inherits this answer, so an interaction missed here is missed everywhere.
//
// Nothing here is remembered between items.  Where every family stands -- which race it is in, what each candidate has
// read, which races are decided and what they decided -- is recomputed from the database each time it is asked for, so
// a race interrupted by a stop loses nothing and a later process carries it on from the rows.
//
// The winners are the selection file's first default lines: keys every tuned family agrees on make the global line,
// keys they disagree on a line per family.  Both are transcripts of races that actually ran.  Once entries are
// published, publishedLines() draws the lines from them instead, the same way.

#pragma once

#include "common.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Probe.h"
#include "Stats.h"
#include "TuneDB.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace tune {

// A race is decided when the leader's interval, this many standard errors wide on each side, clears every rival's.
inline constexpr double RACE_CONFIDENCE = 2.0;

// Or when every rival still overlapping the leader is within this fraction of it: closer than that, which of the two
// wins is not worth the calls it would take to say.
inline constexpr double RACE_MARGIN = 0.0025;

// A candidate that has had this many calls without separating from the leader is tied with it.
inline constexpr u32 RACE_MAX_CALLS = 16;

// A group is raced again from its winner until the incumbent holds, and at most this many times.  A later round reads
// the same rows as the one before, so it cannot undo a separation and the rounds end on their own; the bound is a
// backstop.
inline constexpr u32 GROUP_ROUNDS = 8;

// The largest gain a family is raced in the hope of.  Short of the gain prior's own tail: a race costs tens of minutes,
// and the rare gain past this would not repay it for every family that far off the pace.  A family beyond it still has
// its baselines measured, at the lines the others decide.
inline constexpr double RACE_GAIN = 0.32;

[[nodiscard]] const char* typeName(enum FFT_TYPES type);

// The tunable part of `config` as the kernels would see it: a key that does not apply, that the kernels cannot read, or
// that is at its default is dropped, since leaving it in would give one configuration two names.
[[nodiscard]] UseConfig canonicalConfig(const Env& env, const FFTConfig& fft, const UseConfig& config);

// What tells two builds of `fft` apart: canonical where the option table knows a key well enough to say it changes
// nothing, and as written everywhere else.
[[nodiscard]] UseConfig builtAs(const Env& env, const FFTConfig& fft, const UseConfig& config);

// Every configuration one step from `background` within `group`: one key moved to another of its values, or for LOADS
// and STORES one access class moved to another of its modes.  Canonical, without duplicates, and without `background`
// itself.
struct Move {
  UseConfig config;

  // The key that moved; the build failure of a move is that key's, and nothing else's.
  std::string key;

  // "WMUL=1", "LOADS=20"
  std::string text;
};

[[nodiscard]] std::vector<Move> movesWithin(const Env& env, const FFTConfig& fft, const UseConfig& background,
                                            Group group);

// One candidate of a race, as the database stands.
struct RaceEntry {
  UseConfig config;
  std::string text;

  // Nothing until it has been measured.
  std::optional<Measurement> m;

  // A failure, a hold, or attempts that recorded nothing: it takes no further part.
  bool out = false;
};

enum class RaceHow : u8 { Pending, Separated, Margin, Alone };

[[nodiscard]] const char* toString(RaceHow how);

struct RaceResult {
  RaceHow how = RaceHow::Pending;

  // Into the entries; decided races only.  Nothing where every candidate is out.
  std::optional<size_t> winner;

  // Where the race is pending: the candidates to call next, the first most in need of a call.  Where only one needs a
  // call, the rest of the race follows it, as what to call in between.
  std::vector<size_t> next;
};

// Pure.  Every candidate is called MIN_CALLS times first.  After that the leader is the lowest cost, and a rival is
// settled once it has separated from the leader, is within RACE_MARGIN of it, or has had RACE_MAX_CALLS calls; the race
// is decided when every rival is settled.  A tie goes to the candidate nearest the built-in defaults, since a key moved
// for no measurable gain is a key moved for nothing; then to the one with the most calls, which once a race is decided
// is its winner -- the incumbent of every later race, and so the one whose calls keep pooling -- so that evidence
// accumulating inside the margin cannot flip a decision the margin already called a tie; then to the lower cost.  Until
// then the unsettled candidates and the leader are called, fewest calls first, and never one of them twice running
// while the race holds another.
[[nodiscard]] RaceResult decideRace(const std::vector<RaceEntry>& entries);

// One family's bootstrap configuration: the smallest shape of the type whose default variant holds the probe, at that
// variant and the automatic carry.
struct Family {
  enum FFT_TYPES type = FFT64;
  FFTConfig fft;
};

enum class FamilyPhase : u8 {
  Unread,   // no reading at the built-in defaults yet, so nothing can be said about it
  Held,     // it cannot be measured at the defaults at all
  Skipped,  // not worth tuning, or bootstrapping was turned off
  Waiting,  // worth tuning, behind a cheaper family
  Racing,
  Done,
};

[[nodiscard]] const char* toString(FamilyPhase phase);

// One decided race, for the log.
struct Decision {
  // A group, or the groups a combination combined ("Tail+Height", "all").
  std::string stage;
  u32 round = 0;
  RaceHow how = RaceHow::Pending;
  std::string winner;  // the move that won, or "the incumbent"
  double cost = 0;
  double se = 0;
  u32 candidates = 0;
};

struct FamilyState {
  Family family;
  FamilyPhase phase = FamilyPhase::Unread;

  // At the built-in defaults, anchor corrected; zero until read.
  double reading = 0;

  // What its decided races add up to, canonical.
  UseConfig decided{};

  std::vector<Decision> decisions{};

  // While racing: the group or the combination.
  std::string stage{};
  std::vector<RaceEntry> entries{};
  RaceResult race{};
};

// One call the bootstrap wants made.
struct Turn {
  size_t family = 0;
  UseConfig config;
  std::string key;   // what moved, for a build failure to be pinned on; empty for the incumbent
  std::string text;  // "Width WMUL=1", or "defaults"
  u32 calls = 0;

  // 2 or 3 for a point of a combination, 1 otherwise.
  u32 tier = 1;
};

struct BootstrapState {
  std::vector<FamilyState> families;

  // What to call next, most wanted first; empty once every family is done or skipped.
  std::vector<Turn> turns;

  // The lines the families decided so far make, and whether any family has a race still to run.
  Defaults defaults;
  bool complete = false;
};

class Bootstrap {
public:
  Bootstrap() = default;
  // `comboTiers` is how many tiers of the combination tree are raced after the groups: 1 races the groups alone.
  Bootstrap(Env env, u64 probe, std::vector<Family> families, bool enabled = true, u32 comboTiers = COMBO_TIERS);

  // Where every family stands against what `env` has measured.  `excluded` names the candidates this process has tried
  // too often without recording anything, by configText().
  [[nodiscard]] BootstrapState state(const TuneDB& db, u32 env, const std::set<std::string>& excluded = {}) const;

  // As state(), with each family on the configuration named rather than the one the database records.
  [[nodiscard]] BootstrapState stateOf(const std::vector<Family>& families, const TuneDB& db, u32 env,
                                       const std::set<std::string>& excluded = {}) const;

  // The families, each on the configuration `env` recorded for it at this probe (a `boot` row).  Where none is
  // recorded: on the one it was built with if its races began there, which is how a bootstrap begun before the choice
  // was recorded goes on; else on the type's cheapest concluded reading at the built-in defaults at the probe, which
  // after the defaults sweep is the FFT the search would tune first; else on the one it was built with.  A run records
  // the choice once the sweep is done, so that it does not move under the races as readings are added.
  [[nodiscard]] std::vector<Family> familiesIn(const TuneDB& db, u32 env) const;

  // Whether `env` has recorded a configuration for every family at this probe.
  [[nodiscard]] bool chosen(const TuneDB& db, u32 env) const;

  // The families with no configuration recorded at this probe, on the one familiesIn() gives them.
  [[nodiscard]] std::vector<Family> unrecorded(const TuneDB& db, u32 env) const;

  [[nodiscard]] const Env& env() const { return env_; }
  [[nodiscard]] u64 probe() const { return probe_; }
  // As built: each type's smallest shape whose default variant holds the probe, before any choice is recorded.
  [[nodiscard]] const std::vector<Family>& families() const { return families_; }
  [[nodiscard]] bool enabled() const { return enabled_; }

private:
  Env env_;
  u64 probe_ = 0;
  std::vector<Family> families_;
  bool enabled_ = false;
  u32 comboTiers_ = COMBO_TIERS;

  // The last probesOf() for each family's combination tier, which is pure, and what it was asked: a decided family's
  // stages are asked for again on every re-score.
  mutable std::map<std::pair<size_t, u32>, std::pair<std::string, ProbeList>> stageLists_;
};

// The families a run over `baselines` bootstraps: for each type some baseline belongs to, its smallest shape whose
// default variant holds `probe` and that `env` can compile.  In type order.
[[nodiscard]] std::vector<Family> bootstrapFamilies(const Env& env, u64 probe, const std::vector<FFTConfig>& inScope);

// The lines the decided families make: a key every family it applies to agrees on goes on the global line, and a key
// they disagree on goes on the line of each family that moved it.  A family that kept the built-in value of a disputed
// key needs no line for it, since the global line does not set it.
[[nodiscard]] Defaults defaultLines(const Env& env, const std::vector<std::pair<Family, UseConfig>>& decided);

// What a configuration with no option set of its own runs at under `defaults`, canonical.
[[nodiscard]] UseConfig underDefaults(const Env& env, const FFTConfig& fft, TestKind kind, const Defaults& defaults);

// The lines published beside `published`, and the one jump the search tries first on an entry still at the built-in
// defaults: the best evidence there is for each FFT type, split as defaultLines() splits the bootstrap's.  A type's
// evidence is its `kind` entry that production would run at `probe`, or where none covers it the entry nearest it,
// and failing both the set its bootstrap decided.  An entry published at the built-in defaults is none: it has not been
// searched.
// So the lines start as the bootstrap's and, as entries are tuned further, carry what they found to every FFT nothing
// has been published for.  A key held at a value that changes the rounding is left at its default, since nothing reads
// the accuracy of what the lines are applied to.
[[nodiscard]] Defaults publishedLines(const Env& env, u64 probe, TestKind kind,
                                      const std::vector<SelectionEntry>& published, const BootstrapState& bootstrap);

}  // namespace tune
