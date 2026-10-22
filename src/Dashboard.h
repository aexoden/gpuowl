// Copyright (C) Jason Lynch

// A full-screen view of a tuning run, asked for with dashboard=1: what is being measured now, what is published and
// what it costs, how T and the expected regret have moved, what the model has learnt and how well it predicts, the
// items just finished, the rest of the log, and every configuration that failed or took the device down.
//
// It replaces only what the run writes to the terminal: the log file gets every line it gets without it.  The frame is
// a pure function of a Board and the terminal's size.

#pragma once

#include "common.h"
#include "Watch.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace tune {

inline constexpr double DASHBOARD_TICK_SEC = 0.5;
inline constexpr size_t RECENT_KEPT = 64;
inline constexpr size_t EVENTS_KEPT = 128;
inline constexpr size_t HISTORY_KEPT = 512;

enum class Style : u8 { Plain, Bold, Dim, Good, Bad, Warn, Heading };

struct Span {
  std::string text;
  Style style = Style::Plain;
};

// One row of the screen, never wider than the frame it is in.
using Row = std::vector<Span>;

[[nodiscard]] size_t columns(std::string_view text);
[[nodiscard]] size_t columns(const Row& row);
[[nodiscard]] std::string plain(const Row& row);

struct Board {
  std::string device{};
  std::string scope{};
  u32 generation = 0;
  double elapsed = 0;
  std::string what{};  // being measured now
  double onIt = 0;
  bool started = false;
  StateReport state{};
  std::vector<double> T;          // after each item
  std::vector<double> regret;     // at each estimate
  std::deque<ItemReport> recent;  // newest first
  std::deque<std::string> events;
  bool unicode = true;
};

// The screen, `rows` by `cols`, with nothing in its last column.
[[nodiscard]] std::vector<Row> frame(const Board& board, size_t rows, size_t cols);

// The bytes that draw `rows` from the top of the screen, clearing what is left of each row and below the last.
[[nodiscard]] std::string paint(const std::vector<Row>& rows, bool color);

// A line log() sent to stdout, as the events panel lists it: "HH:MM:SS text", its date and "tune: " dropped; nothing
// for the item lines the recent panel already shows.
[[nodiscard]] std::optional<std::string> eventOf(std::string_view line);

[[nodiscard]] bool utf8Terminal();
[[nodiscard]] bool colorTerminal();

// Whether stdout is a terminal a dashboard can take.
[[nodiscard]] bool canTakeTerminal();

class DashboardView final : public Watch {
public:
  DashboardView(std::string device, std::string scope, u32 generation);
  ~DashboardView() override;

  DashboardView(const DashboardView&) = delete;
  DashboardView& operator=(const DashboardView&) = delete;

  void measuring(const std::string& what) override;
  void finished(const ItemReport& item) override;
  void state(const StateReport& state) override;

private:
  using Clock = std::chrono::steady_clock;

  void tick();
  void take(std::string_view text);
  void close();

  bool color_;
  Clock::time_point start_;
  Clock::time_point whatSince_{};

  // Everything below, under mutex_, which log()'s stdout sink takes with log()'s own lock held: nothing calls log()
  // holding it.
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
  bool closed_ = false;
  Board board_;
  std::string partial_;
  std::thread ticker_;
};

}  // namespace tune
