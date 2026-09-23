// Copyright (C) Jason Lynch

// The global bootstrap: for each FFT type worth tuning, the option set every configuration of that type runs at until
// it has been tuned itself, found by racing the option groups on one configuration at the exponent that matters most.
//
// Nothing here is remembered between items.  Where every family stands -- which race it is in, what each candidate has
// read, which races are decided and what they decided -- is recomputed from the database each time it is asked for, so
// a race interrupted by a stop loses nothing and a later process carries it on from the rows.
//
// The winners become the selection file's default lines: keys every tuned family agrees on make the global line, keys
// they disagree on a line per family.  Both are transcripts of races that actually ran.

#pragma once

#include "common.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Stats.h"
#include "TuneDB.h"

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
  Group group = Group::None;
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

  // While racing.
  Group group = Group::None;
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
  Bootstrap(Env env, u64 probe, std::vector<Family> families, bool enabled = true);

  // Where every family stands against what `env` has measured.  `excluded` names the candidates this process has tried
  // too often without recording anything, by configText().
  [[nodiscard]] BootstrapState state(const TuneDB& db, u32 env, const std::set<std::string>& excluded = {}) const;

  [[nodiscard]] const Env& env() const { return env_; }
  [[nodiscard]] u64 probe() const { return probe_; }
  [[nodiscard]] const std::vector<Family>& families() const { return families_; }
  [[nodiscard]] bool enabled() const { return enabled_; }

private:
  Env env_;
  u64 probe_ = 0;
  std::vector<Family> families_;
  bool enabled_ = false;
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

}  // namespace tune
