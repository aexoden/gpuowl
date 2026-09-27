// Copyright (C) Jason Lynch

// A full-screen view of a tuning run, asked for with dashboard=1: what is being measured now, what the tuning has
// bought over every session of the env, what production runs over the workload and how that moved, what the run did
// last and will do next, the rest of the log, and every configuration that took the device down or computed a wrong
// answer.
//
// It replaces only what the run writes to the terminal.  The log file gets every line it gets without it, the periodic
// progress line and the heartbeat included, and nothing drawn here reaches it.
//
// The frame is a pure function of a Board and the terminal's size.  DashboardView fills the Board from what runQueue()
// tells its Watch, and owns the terminal: the alternate screen, a thread that repaints it so the clocks move while the
// run is blocked in a call, and giving the screen back on every way out that can be caught.

#pragma once

#include "common.h"
#include "Faults.h"
#include "History.h"
#include "Progress.h"
#include "Scheduler.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace tune {

inline constexpr double DASHBOARD_TICK_SEC = 0.5;

// How many items finished, lines of the log and items ranked next the board keeps.
inline constexpr size_t RECENT_KEPT = 64;
inline constexpr size_t EVENTS_KEPT = 128;
inline constexpr size_t NEXT_KEPT = 16;

// How many faults are listed before the rest are counted.
inline constexpr size_t FAULTS_SHOWN = 5;

// From this width the panels below the charts sit in two columns.
inline constexpr size_t TWO_COLUMNS_FROM = 160;

// From this height the charts are drawn three rows tall.
inline constexpr size_t TALL_CHARTS_FROM = 50;

enum class Style : u8 { Plain, Bold, Dim, Good, Bad, Warn, Heading };

struct Span {
  std::string text;
  Style style = Style::Plain;
};

// One row of the screen, never wider than the frame it is in.
using Line = std::vector<Span>;

// The columns `line` takes: every glyph drawn here is one column wide.
[[nodiscard]] size_t columns(std::string_view text);
[[nodiscard]] size_t columns(const Line& line);
[[nodiscard]] std::string plain(const Line& line);

// One stretch of the workload production runs one way: the entry that serves it, or the prior where none does.
struct Stretch {
  TestKind kind = TestKind::PRP;
  u64 lo = 0;
  u64 hi = 0;
  double weight = 0;

  std::string fft;
  bool measured = false;

  // What an iteration costs there now, and what it cost at the run's start, both weighted over the stretch's points.
  double us = 0;
  double was = 0;

  // Whether the run changed what production runs anywhere in it, and whether all of it was measured at the run's
  // start: a prior is an estimate, so a change from one is a first measurement rather than a gain or a loss.
  bool moved = false;
  bool wasMeasured = true;
};

// `now`'s points grouped into stretches of one entry, in exponent order within each kind, set against `start`, the
// same grid's points when the run began.
[[nodiscard]] std::vector<Stretch> stretchesOf(const std::vector<ObjectivePoint>& now,
                                               const std::vector<ObjectivePoint>& start);

// One item the queue would take, as far as NEXT_KEPT.
struct NextUp {
  ItemKind kind = ItemKind::Baseline;
  std::string label;
  bool byRule = false;
  double value = 0;
  double seconds = 0;
};

// One item finished, and when.
struct Recent {
  Finished item;
  std::string at;
};

// What the queue is doing, in words, from what it ranks: its by-rule work first, since that runs ahead of everything.
[[nodiscard]] std::string phaseOf(const std::vector<Item>& ranked, bool anythingWorthRunning);

struct Board {
  std::string device;
  std::string scope;
  u32 generation = 0;

  // Seconds this process has run, and the env's measuring before it, over how many sessions.
  double elapsed = 0;
  double before = 0;
  u32 sessions = 0;

  bool started = false;
  RunProgress progress{};
  std::string phase;
  std::string what;
  double onIt = 0;

  Benefit benefit{};
  bool historyPending = false;

  // The database's history, then a point for every item this process has run.
  std::vector<HistoryPoint> history;

  std::vector<Stretch> production;
  std::vector<NextUp> next;

  // Newest first.
  std::deque<Recent> recent;
  std::deque<std::string> events;

  std::vector<Fault> faults;

  bool unicode = true;
};

// The screen, `rows` by `cols`, with nothing in its last column.
[[nodiscard]] std::vector<Line> frame(const Board& board, size_t rows, size_t cols);

// The bytes that draw `lines` from the top of the screen, clearing what is left of each row and below the last.
[[nodiscard]] std::string paint(const std::vector<Line>& lines, bool color);

// A line log() sent to stdout, as the events panel lists it: "HH:MM:SS text", its date and "tune: " dropped; nothing
// for the lines the other panels already show -- an item's closing line, the progress line, the heartbeat.
[[nodiscard]] std::optional<std::string> eventOf(std::string_view line);

// Whether the terminal takes UTF-8, by the locale's environment variables, and colour, unless NO_COLOR is set.
[[nodiscard]] bool utf8Terminal();
[[nodiscard]] bool colorTerminal();

class DashboardView final : public Watch {
public:
  struct Setup {
    std::string device;
    std::string scope;
    u32 generation = 0;

    // The database, and the env and session this run measures in.
    fs::path db;
    u32 env = 0;
    u32 session = 0;
    Env card;

    RunScope runScope;
  };

  // Takes the terminal, then rebuilds the database's history while it shows that it is doing so.
  explicit DashboardView(Setup setup);
  ~DashboardView() override;

  DashboardView(const DashboardView&) = delete;
  DashboardView& operator=(const DashboardView&) = delete;

  void measuring(const std::string& what) override;
  void progress(const RunProgress& p) override;
  void state(const QueueState& q) override;
  void finished(const Finished& f) override;

private:
  using Clock = std::chrono::steady_clock;

  void tick();
  void take(std::string_view text);
  void close();

  Setup setup_;
  TestKind kind_;
  bool color_;
  Clock::time_point start_;

  // Everything below, under mutex_, which log()'s stdout sink takes with log()'s own lock held: nothing calls log()
  // holding it.
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
  bool closed_ = false;
  Board board_;
  Clock::time_point whatSince_{};
  std::vector<ObjectivePoint> startPoints_;
  bool haveStart_ = false;
  std::string partial_;

  std::thread ticker_;
};

}  // namespace tune
