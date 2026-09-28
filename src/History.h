// Copyright (C) Jason Lynch

// What the time spent tuning has bought, read from the database rather than from one process's memory, so that it
// spans every session, resumption and restart an env has had.
//
// Every measurement row carries its timestamp, so the database as it stood at any moment is the file with the later
// rows left out; the objective over that is what production would have run then.  And at the probe, where the anchor
// race and the bootstrap always time configurations at the built-in defaults, what production runs now can be set
// against what an untuned run could have had there.

#pragma once

#include "common.h"
#include "Objective.h"
#include "OptionSpace.h"
#include "TuneDB.h"
#include "Tuner.h"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tune {

// How many moments of the past a history rebuilds at most, and how long it may spend doing so before it keeps what it
// has.
inline constexpr size_t HISTORY_POINTS = 64;
inline constexpr double HISTORY_BUDGET_SEC = 10;

// The text of a database as it stood at `ts`: every row stamped later left out; envs, option sets, sessions, pending
// work and kinds this build does not know kept.
[[nodiscard]] std::string asOf(std::string_view text, u64 ts);

// One session of an env: from its start to the last row it wrote.
struct SessionSpan {
  u32 sess = 0;
  u64 start = 0;
  u64 end = 0;

  [[nodiscard]] double seconds() const { return double(end - start); }
};

// The sessions of `env` in `text`, oldest first.
[[nodiscard]] std::vector<SessionSpan> spansOf(std::string_view text, u32 env);

// The moment `active` seconds of measuring into `spans`.
[[nodiscard]] u64 momentAt(std::span<const SessionSpan> spans, double active);

// Seconds of measuring in `spans`, leaving out session `except`.
[[nodiscard]] double measuringTime(std::span<const SessionSpan> spans, u32 except = 0);

// The order to rebuild `n` evenly spaced moments in, so that however many are rebuilt they span the whole range: both
// ends, then each gap halved in turn.
[[nodiscard]] std::vector<size_t> spreadOrder(size_t n);

// Where the objective stood at one moment.
struct HistoryPoint {
  // Seconds of measuring on the env before it.
  double active = 0;

  double T = 0;
  double measured = 0;

  // What production ran at the probe, where a measured entry served it.
  std::optional<double> probe;
};

// The kind the probe is valued in: PRP where the workload has a PRP grid, else the first kind that has one.
[[nodiscard]] TestKind probeKind(const RunScope& scope);

// The objective over the database `text` as it stood at up to `points` moments evenly spread over `env`'s measuring,
// valued over `scope`, oldest first.  Rebuilt in spreadOrder(), stopping once `more` says no, so that a history cut
// short still spans the whole range, coarsely.
[[nodiscard]] std::vector<HistoryPoint> replay(std::string_view text, u32 env, const RunScope& scope, size_t points,
                                               const std::function<bool()>& more);

// What production runs at the probe, against the cheapest configuration measured there at the built-in defaults, both
// at the cost the selection file ranks by.
struct Benefit {
  TestKind kind = TestKind::PRP;
  u64 probe = 0;

  // Nothing where no measured entry serves the probe yet.
  std::optional<Cost> now;

  // Empty where nothing was measured at the probe at the built-in defaults.
  std::string untunedFft;
  double untuned = 0;

  // The share of the untuned time per iteration production no longer spends, where both are known.
  [[nodiscard]] std::optional<double> saving() const;
};

[[nodiscard]] Benefit benefitOf(const TuneDB& db, u32 env, const Env& device, const Objective& objective, TestKind kind,
                                u64 probe);

// "at <probe> production runs <fft> at <us> us/it, <p>% less per iteration than ..."
[[nodiscard]] std::string benefitText(const Benefit& benefit);

// What an exponent would cost untuned: the cheapest entry of `env` measured at the built-in defaults whose interval
// holds it, at the cost the selection file ranks by.  What every gain the tuning made is a gain over, whichever FFT it
// was made on.
class Untuned {
public:
  Untuned(const TuneDB& db, u32 env, const Env& device);

  // Nothing where no reading at the built-in defaults serves `E`.
  [[nodiscard]] std::optional<Cost> at(TestKind kind, u64 E) const;

private:
  std::vector<SelectionEntry> entries_;
};

}  // namespace tune
