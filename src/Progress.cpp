// Copyright (C) Jason Lynch

#include "Progress.h"

#include "Restart.h"
#include "Status.h"
#include "Summary.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace tune {

namespace {

constexpr std::string_view CLEAR = "\r\x1b[K";

// The narrowest `what` is elided to before the figures beside it are given up, and then the line cut at the edge.
constexpr size_t MIN_WHAT = 48;

#ifdef __GNUC__
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

// A line here is far shorter than the buffer.
[[nodiscard]] std::string format(const char* fmt, ...) {
  char buf[512];
  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  return buf;
}

// `s` in `width` characters, the middle taken out: a label's entry is at its start and its exponent at its end.
[[nodiscard]] std::string elide(const std::string& s, size_t width) {
  if (s.size() <= width) { return s; }
  if (width <= 3) { return s.substr(0, width); }
  size_t const keep = width - 3;
  size_t const head = (keep + 1) / 2;
  return s.substr(0, head) + "..." + s.substr(s.size() - (keep - head));
}

[[nodiscard]] size_t terminalColumns() {
#ifndef _WIN32
  winsize ws{};
  if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) && ws.ws_col) { return ws.ws_col; }
#endif
  return 80;
}

void writeOut(std::string_view bytes) {
  fwrite(bytes.data(), 1, bytes.size(), stdout);
  fflush(stdout);
}

}  // namespace

RunProgress progressOf(const QueueReport& sofar, double T, double measured, const std::vector<Item>& ranked,
                       double stop, double floor, bool bootstrapComplete) {
  RunProgress out{.items = sofar.items,
                  .anchors = sofar.anchors,
                  .spent = sofar.spent,
                  .startT = sofar.startT,
                  .T = T,
                  .measured = measured,
                  .stop = stop,
                  .floor = floor,
                  .bootstrapComplete = bootstrapComplete};
  for (const Item& item : ranked) {
    if (!byRule(item)) { out.mostWorth = std::max(out.mostWorth, item.value); }
    if (worthRunning(item, floor)) {
      ++out.worthRunning;
      out.worthSeconds += item.seconds;
      out.unlisted += item.unlisted;
    }
  }
  return out;
}

std::string clockText(double seconds) {
  auto const s = u64(std::max(0.0, seconds));
  if (s < 3600) { return format("%u:%02u", u32(s / 60), u32(s % 60)); }
  return format("%u:%02u:%02u", u32(s / 3600), u32(s / 60 % 60), u32(s % 60));
}

std::string progressLine(const RunProgress& p, double elapsed) {
  std::string const spent = spentText(p.spent);
  std::string out = format("%s in, %u %s and %u anchor %s%s%s; T %.3f -> %.3f us/it, %.1f%% of the weight on measured "
                           "entries; ",
                           clockText(elapsed).c_str(), p.items, p.items == 1 ? "item" : "items", p.anchors,
                           p.anchors == 1 ? "reading" : "readings", spent.empty() ? "" : ": ", spent.c_str(), p.startT,
                           p.T, 100 * p.measured);

  if (!p.worthRunning) { return out + "nothing is worth running now"; }

  if (p.mostWorth > 0) {
    out += p.stop > 0 && p.floor > 0
      ? format("the most an item is worth is %.4f us/it, %.1fx the stop floor of %.4f (stop=%g%% of T); ", p.mostWorth,
               p.mostWorth / p.floor, p.floor, 100 * p.stop)
      : format("the most an item is worth is %.4f us/it, and stop=0 runs until stopped; ", p.mostWorth);
  }
  out += format("%u %s worth running now, ~%s by the queue's estimates", p.worthRunning,
                p.worthRunning == 1 ? "item is" : "items are", ago(u64(std::lround(p.worthSeconds))).c_str());
  return p.unlisted ? out + format(", and up to %" PRIu64 " more not listed yet", p.unlisted) : out;
}

std::string heartbeatLine(const std::string& what, double onIt) {
  return "still measuring " + what + ", " + clockText(onIt) + " so far";
}

std::string liveLine(const RunProgress& p, const std::string& what, double elapsed, double onIt, size_t cols) {
  size_t const width = cols > 1 ? cols - 1 : 0;

  std::string const head = "[" + clockText(elapsed) + "] ";
  std::string const body = what.empty() ? "choosing the next item" : what;
  std::string const clock = what.empty() ? "" : " (" + clockText(onIt) + ")";

  // What is being measured comes first; where the terminal is too narrow for it, these are given up from the last.
  std::vector<std::string> extras;
  if (p.T > 0) { extras.push_back(format(" | T %.3f us/it (%.0f%% measured)", p.T, 100 * p.measured)); }
  if (p.mostWorth > 0 && p.stop > 0 && p.floor > 0) {
    extras.push_back(format(" | worth %.1fx stop", p.mostWorth / p.floor));
  } else if (p.mostWorth > 0) {
    extras.push_back(format(" | worth %.4f us/it", p.mostWorth));
  }

  size_t const want = std::min(body.size(), MIN_WHAT);
  auto fixed = [&] {
    size_t n = head.size() + clock.size();
    for (const std::string& e : extras) { n += e.size(); }
    return n;
  };
  while (!extras.empty() && fixed() + want > width) { extras.pop_back(); }

  std::string tail = clock;
  for (const std::string& e : extras) { tail += e; }
  if (head.size() + tail.size() + want <= width) {
    return head + elide(body, width - head.size() - tail.size()) + tail;
  }
  return (head + body + tail).substr(0, width);
}

bool progressDue(double elapsed, double last) { return elapsed - last >= PROGRESS_EVERY_SEC; }

u32 heartbeatsDue(double onIt) { return onIt > 0 ? u32(onIt / HEARTBEAT_EVERY_SEC) : 0; }

std::string redraw(const std::string& line) { return line.empty() ? std::string{CLEAR} : "\r" + line + "\x1b[K"; }

Paint repaint(std::string_view text, bool drawn, const std::string& line) {
  Paint out;
  if (drawn) { out.bytes = CLEAR; }
  out.bytes += text;
  if (!text.empty() && text.back() == '\n' && !line.empty()) {
    out.bytes += redraw(line);
    out.drawn = true;
  }
  return out;
}

bool liveTerminal() {
#ifdef _WIN32
  // A console takes the escapes only once asked to, and not every host can be.
  return false;
#else
  if (!isatty(STDOUT_FILENO)) { return false; }
  const char* const term = getenv("TERM");
  return term && *term && std::string_view{term} != "dumb";
#endif
}

RunView::RunView(bool live) : live_{live}, start_{Clock::now()}, link_{logLink()} {
  if (live_) {
    setStdoutSink([this](std::string_view text) { paint(text); });
    restart::beforeExec([this] { close(); });
  }
  ticker_ = std::thread{[this] { tick(); }};
}

RunView::~RunView() {
  {
    std::unique_lock const lock{state_};
    stopping_ = true;
  }
  wake_.notify_all();
  ticker_.join();

  if (live_) {
    restart::beforeExec(nullptr);
    close();
    setStdoutSink(nullptr);
  }
}

void RunView::measuring(const std::string& what) {
  std::unique_lock const lock{state_};
  what_ = what;
  whatSince_ = Clock::now();
  beats_ = 0;
}

void RunView::progress(const RunProgress& p) {
  std::string line;
  {
    std::unique_lock const lock{state_};
    double const elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
    bool const bootstrapDone = seen_ && p.bootstrapComplete && !progress_.bootstrapComplete;
    if (seen_ && (bootstrapDone || progressDue(elapsed, lastProgress_))) {
      lastProgress_ = elapsed;
      line = progressLine(p, elapsed);
    }
    seen_ = true;
    progress_ = p;
    what_.clear();
  }
  if (!line.empty()) { log("tune: progress: %s\n", line.c_str()); }
}

void RunView::tick() {
  // A thread of its own starts with no log file, and the heartbeat belongs in the run's.
  LogLinkScope const scope{link_};

  std::unique_lock lock{state_};
  while (!stopping_) {
    auto const now = Clock::now();
    double const elapsed = std::chrono::duration<double>(now - start_).count();
    double const onIt = what_.empty() ? 0 : std::chrono::duration<double>(now - whatSince_).count();

    std::string beat;
    if (u32 const due = heartbeatsDue(onIt); !what_.empty() && due > beats_) {
      beats_ = due;
      beat = heartbeatLine(what_, onIt);
    }
    std::string const line = live_ ? liveLine(progress_, what_, elapsed, onIt, terminalColumns()) : "";

    lock.unlock();
    if (!beat.empty()) { log("tune: %s\n", beat.c_str()); }
    if (live_) { draw(line); }
    lock.lock();

    wake_.wait_for(lock, std::chrono::duration<double>(LIVE_TICK_SEC), [this] { return stopping_; });
  }
}

void RunView::paint(std::string_view text) {
  std::unique_lock const lock{screen_};
  if (closed_) {
    writeOut(text);
    return;
  }
  Paint const p = repaint(text, drawn_, shown_);
  writeOut(p.bytes);
  drawn_ = p.drawn;
}

void RunView::draw(const std::string& line) {
  std::unique_lock const lock{screen_};
  if (closed_) { return; }
  writeOut(redraw(line));
  shown_ = line;
  drawn_ = !line.empty();
}

void RunView::close() {
  std::unique_lock const lock{screen_};
  if (drawn_) { writeOut(CLEAR); }
  drawn_ = false;
  closed_ = true;
}

}  // namespace tune
