// Copyright (C) Jason Lynch

// The bootstrap: for each FFT type worth tuning, its fastest configuration at the probe is searched first, for a
// bounded number of calls, by the same search every entry has.  What it finds is published like any entry's, and so
// reaches the default lines, which carry it to every configuration of the type nothing has been published for yet, and
// which every entry tries as a step of its own.  The bootstrap only says which entries are searched first, and how far.
//
// Nothing here is remembered between items: where every family stands is recomputed from the database each time it
// is asked for, so a stop loses nothing and a later process carries on from the rows.

#pragma once

#include "common.h"
#include "Emit.h"
#include "FFTConfig.h"
#include "OptionSpace.h"
#include "Probe.h"
#include "Stats.h"
#include "TuneDB.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace tune {

// The largest gain a family is searched in the hope of.  Short of the gain prior's own tail: a family that far off the
// pace would take a gain too rare to repay the search to come level with the cheapest.  A family beyond it still has
// its entries measured, and searched as any entry is.
inline constexpr double BOOTSTRAP_GAIN = 0.32;

// How many rounds of the halving's calls each family's search is given.
inline constexpr u32 BOOTSTRAP_ROUNDS = 4;

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

// One FFT type, and the configuration its bootstrap searches: as built, the smallest shape of the type whose default
// variant holds the probe, at that variant and the automatic carry.
struct Family {
  enum FFT_TYPES type = FFT64;
  FFTConfig fft;
};

enum class FamilyPhase : u8 {
  Unread,   // no reading at the built-in defaults yet, so nothing can be said about it
  Held,     // it cannot be measured at the defaults at all
  Skipped,  // not worth searching, or bootstrapping was turned off
  Owed,     // worth searching, and short of its calls
  Served,   // it has had its calls
};

[[nodiscard]] const char* toString(FamilyPhase phase);

struct FamilyState {
  Family family;
  FamilyPhase phase = FamilyPhase::Unread;

  // At the probe, anchor corrected: at the built-in defaults, and the cheapest under any options; zero until read.
  double reading = 0;
  double best = 0;

  // Its calls of search: every call at the probe at an option set other than the built-in defaults.
  u64 calls = 0;
};

struct BootstrapState {
  std::vector<FamilyState> families;

  // The test kind the families are read and searched in.
  TestKind kind = TestKind::PRP;

  // How many calls of search each family worth searching is given.
  u64 budget = 0;

  // Whether no family is still to be read or searched.
  bool complete = false;
};

class Bootstrap {
public:
  Bootstrap() = default;
  // `kind` is the test kind the families are read and searched in.
  Bootstrap(Env env, u64 probe, std::vector<Family> families, bool enabled = true, TestKind kind = TestKind::PRP);

  // Where every family stands against what `env` has measured, each given `budget` calls of search.  A family is read
  // at the built-in defaults first; then, cheapest first, searched while it has had fewer than `budget` calls and is
  // within BOOTSTRAP_GAIN of the cheapest family, each as it is tuned so far.
  [[nodiscard]] BootstrapState state(const TuneDB& db, u32 env, u64 budget) const;

  // The families, each on the configuration `env` recorded for it at this probe (a `boot` row).  Where none is
  // recorded: on the type's cheapest concluded reading at the built-in defaults at the probe, which after the defaults
  // sweep is the FFT the search would tune first; else on the one it was built with.  A run records the choice once
  // the sweep is done, so that it does not move under the search as readings are added.
  [[nodiscard]] std::vector<Family> familiesIn(const TuneDB& db, u32 env) const;

  // Whether `env` has recorded a configuration for every family at this probe.
  [[nodiscard]] bool chosen(const TuneDB& db, u32 env) const;

  // The families with no configuration recorded at this probe, on the one familiesIn() gives them.
  [[nodiscard]] std::vector<Family> unrecorded(const TuneDB& db, u32 env) const;

  [[nodiscard]] const Env& env() const { return env_; }
  [[nodiscard]] u64 probe() const { return probe_; }
  [[nodiscard]] TestKind kind() const { return kind_; }
  // As built: each type's smallest shape whose default variant holds the probe, before any choice is recorded.
  [[nodiscard]] const std::vector<Family>& families() const { return families_; }
  [[nodiscard]] bool enabled() const { return enabled_; }

private:
  Env env_;
  u64 probe_ = 0;
  std::vector<Family> families_;
  bool enabled_ = false;
  TestKind kind_ = TestKind::PRP;
};

// The families a run over `baselines` bootstraps: for each type some baseline belongs to, its smallest shape whose
// default variant holds `probe` and that `env` can compile.  In type order.
[[nodiscard]] std::vector<Family> bootstrapFamilies(const Env& env, u64 probe, const std::vector<FFTConfig>& inScope);

// The lines the families' option sets make: a key every family it applies to agrees on goes on the global line, and a
// key they disagree on goes on the line of each family that moved it.  A family that kept the built-in value of a
// disputed key needs no line for it, since the global line does not set it.  Keys coupled with one another
// (keysCoupledWith) go on the global line only when the families agree on all of them.
[[nodiscard]] Defaults defaultLines(const Env& env, const std::vector<std::pair<Family, UseConfig>>& decided);

// What a configuration with no option set of its own runs at under `defaults`, canonical.  With `over`, that set with
// the lines laid on top of it: a key the lines set takes their value, refitted against the rest of `over`, and every
// other key keeps its own, but for a key coupled with one the lines set, which takes the lines' value or its default.
[[nodiscard]] UseConfig underDefaults(const Env& env, const FFTConfig& fft, TestKind kind, const Defaults& defaults,
                                      const UseConfig& over = {});

// The lines published beside `published`, and what every entry tries as a step of its own: the best evidence there is
// for each FFT type, split as defaultLines() splits them.  A type's evidence is its `kind` entry that production would
// run at `probe`, or where none covers it the entry nearest it.  An entry published at the built-in defaults is none:
// it has not been searched.  So the lines carry what the best entries found to every FFT nothing has been published
// for.  A key held at a value that changes the rounding is left at its default, since nothing reads the accuracy of
// what the lines are applied to.
[[nodiscard]] Defaults publishedLines(const Env& env, u64 probe, TestKind kind,
                                      const std::vector<SelectionEntry>& published);

}  // namespace tune
