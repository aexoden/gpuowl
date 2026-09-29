// Copyright (C) Jason Lynch

// What a tuning run shows while it runs, beyond the line each item ends with: every PROGRESS_EVERY_SEC a line saying
// where the run stands against its start and its stopping rule; a line every HEARTBEAT_EVERY_SEC while one call goes
// on, so that a long call is not mistaken for a hang; and, where stdout is a terminal, one line at the bottom of it,
// redrawn in place, with the run's clock, what is being measured and for how long.  The drawn line never reaches the
// log file, and the log's lines scroll above it.
//
// The text is a pure function of what the queue reports and of the clock.  RunView adds the clock, a thread that keeps
// it moving while the run is blocked in a call, and the terminal.

#pragma once

#include "common.h"
#include "log.h"
#include "Scheduler.h"

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace tune {

inline constexpr double PROGRESS_EVERY_SEC = 300;
inline constexpr double HEARTBEAT_EVERY_SEC = 60;
inline constexpr double LIVE_TICK_SEC = 0.5;

// Where a run stands, as the queue saw it after its latest item.
struct RunProgress {
  u32 items = 0;
  u32 anchors = 0;
  std::map<ItemKind, QueueReport::Spent> spent;

  // T before the first item, which the prior prices where nothing is measured yet, and T now with the share of the
  // workload's weight on measured entries.
  double startT = 0;
  double T = 0;
  double measured = 0;

  double stop = 0;
  double floor = 0;

  // The most any item valued by its expected saving is worth, in us/it: what the stopping rule compares with `floor`.
  // Then the items worth a call now, those that run by rule included, and the seconds they are expected to take.
  double mostWorth = 0;
  u32 worthRunning = 0;
  double worthSeconds = 0;

  // At most how many more points worth running the stages listed in part have past what is listed of them.
  u64 unlisted = 0;

  bool bootstrapComplete = false;

  // What the run is doing now, with its own totals; empty where nobody said.
  Phase phase{};
};

// Where the run `sofar` stands, with T now at `T`, `measured` of the weight on measured entries and `ranked` what
// admissible() gives, against `stop` of T with `floor` the value that is.
[[nodiscard]] RunProgress progressOf(const QueueReport& sofar, double T, double measured,
                                     const std::vector<Item>& ranked, double stop, double floor,
                                     bool bootstrapComplete);

// A span of seconds as m:ss, or h:mm:ss from an hour.
[[nodiscard]] std::string clockText(double seconds);

// The periodic line, `elapsed` seconds into the run.
[[nodiscard]] std::string progressLine(const RunProgress& p, double elapsed);

// That `what` is still being measured, `onIt` seconds after it began.
[[nodiscard]] std::string heartbeatLine(const std::string& what, double onIt);

// The line drawn at the bottom of a terminal `cols` wide, never wider than cols - 1, so that it cannot wrap.  `what`
// is empty between calls.
[[nodiscard]] std::string liveLine(const RunProgress& p, const std::string& what, double elapsed, double onIt,
                                   size_t cols);

// Whether the periodic line is due `elapsed` seconds into the run, the last having been written at `last`.
[[nodiscard]] bool progressDue(double elapsed, double last);

// How many heartbeats a call `onIt` seconds long has earned.
[[nodiscard]] u32 heartbeatsDue(double onIt);

// The bytes that redraw the bottom line as `line`.
[[nodiscard]] std::string redraw(const std::string& line);

// What to write for `text`, which log() is sending to stdout, where `drawn` says whether the bottom line is on the
// screen: the line cleared, the text, and the line drawn below it as `line` once the text has ended its own line.
struct Paint {
  std::string bytes;
  bool drawn = false;
};
[[nodiscard]] Paint repaint(std::string_view text, bool drawn, const std::string& line);

// Whether stdout is a terminal the bottom line can be drawn on.
[[nodiscard]] bool liveTerminal();

// One item the queue has finished, as the line it logs says it.
struct Finished {
  u32 n = 0;
  ItemKind kind = ItemKind::Baseline;
  std::string label;

  // Whether it was a step of the bootstrap.
  bool bootstrap = false;
  u64 exponent = 0;

  // " (call <n>)" or " (resumed at call <n>)", as the log writes it after the exponent; empty for a reading.
  std::string call;

  bool completed = false;
  double seconds = 0;

  // A timing's.
  double usPerIt = 0;

  // A gate's or a reach's: the reading, and what the gate made of it.
  bool reads = false;
  double z = 0;
  bool checkOk = true;
  std::string outcome;

  // T before the item and after it.
  double before = 0;
  double after = 0;

  // Where the item moved its entry's best set: the new one, as the log says it.
  std::string best;
};

// What the queue ranks from, for a watch that wants more of it than RunProgress holds.  Valid only for the call it is
// passed to, on the thread running the queue.
struct QueueState {
  const Scheduler& scheduler;
  const TuneDB& db;
  u32 env = 0;
  const Objective& objective;
  const std::vector<Item>& ranked;
  double floor = 0;
};

// What runQueue() tells whoever is watching it.
class Watch {
public:
  virtual ~Watch() = default;

  // Before each call the queue makes, named as the line logged once it is done names it.
  virtual void measuring(const std::string& what) = 0;

  // Once before the first item, and again after each item and each anchor reading.
  virtual void progress(const RunProgress& p) = 0;

  // Straight after each progress().
  virtual void state(const QueueState& /*q*/) {}

  // After each item, as its line is logged.
  virtual void finished(const Finished& /*f*/) {}
};

// Several watches, each told everything in the order given.
class Watches final : public Watch {
public:
  explicit Watches(std::vector<Watch*> watches) : watches_{std::move(watches)} {}

  void measuring(const std::string& what) override;
  void progress(const RunProgress& p) override;
  void state(const QueueState& q) override;
  void finished(const Finished& f) override;

private:
  std::vector<Watch*> watches_;
};

// The periodic line and the heartbeat through log(), and with `live` the bottom line on the terminal, for as long as
// it exists.  One at a time: it takes log()'s stdout copy and the hook that runs before a restart.
class RunView final : public Watch {
public:
  explicit RunView(bool live);
  ~RunView() override;

  RunView(const RunView&) = delete;
  RunView& operator=(const RunView&) = delete;

  void measuring(const std::string& what) override;
  void progress(const RunProgress& p) override;

private:
  using Clock = std::chrono::steady_clock;

  void tick();
  void paint(std::string_view text);
  void draw(const std::string& line);
  void close();

  bool live_;
  Clock::time_point start_;
  LogLink link_;

  // What the ticker reads, under state_.
  std::mutex state_;
  std::condition_variable wake_;
  bool stopping_ = false;
  RunProgress progress_{};
  bool seen_ = false;
  std::string what_;
  Clock::time_point whatSince_{};
  u32 beats_ = 0;
  double lastProgress_ = 0;

  // The terminal, under screen_, which log()'s stdout sink takes with log()'s own lock held: nothing calls log()
  // holding it.
  std::mutex screen_;
  bool drawn_ = false;
  bool closed_ = false;
  std::string shown_;

  std::thread ticker_;
};

}  // namespace tune
