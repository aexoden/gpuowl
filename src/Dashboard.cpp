// Copyright (C) Jason Lynch

#include "Dashboard.h"

#include "log.h"
#include "Restart.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace tune {

namespace {

// The alternate screen with the cursor hidden and no wrapping at the right edge, and back.
constexpr std::string_view ENTER = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[H\x1b[2J";
constexpr std::string_view LEAVE = "\x1b[?7h\x1b[?25h\x1b[?1049l";

// From this width the panels below the charts sit in two columns; from this height the charts are three rows tall.
constexpr size_t TWO_COLUMNS_FROM = 160;
constexpr size_t TALL_CHARTS_FROM = 50;

// How many recent calls the model panel judges its predictions over.
constexpr size_t CALIBRATION_CALLS = 20;

#ifdef __GNUC__
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

std::string format(const char* fmt, ...) {
  char buf[512];
  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  return buf;
}

struct Glyphs {
  std::string_view ellipsis;
  std::string_view bar;
  std::string_view sep;
  std::array<std::string_view, 8> blocks;
};

constexpr Glyphs UNICODE{"…", "│", "  ·  ", {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}};
constexpr Glyphs ASCII{"...", "|", "  |  ", {"_", ".", "-", "~", "=", "+", "*", "#"}};

const Glyphs& glyphs(const Board& b) { return b.unicode ? UNICODE : ASCII; }

bool isContinuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

// The first `width` columns of `text`.
std::string head(std::string_view text, size_t width) {
  size_t cols = 0;
  size_t at = 0;
  while (at < text.size()) {
    if (!isContinuation(text[at])) {
      if (cols == width) { break; }
      ++cols;
    }
    ++at;
  }
  return std::string{text.substr(0, at)};
}

// `row` cut to `width` columns, the ellipsis ending it where anything was cut.
Row fit(const Row& row, size_t width, std::string_view ellipsis) {
  if (columns(row) <= width) { return row; }
  size_t const e = std::min(columns(ellipsis), width);
  Row out;
  size_t used = 0;
  for (const Span& s : row) {
    size_t const room = width - e - used;
    if (!room) { break; }
    std::string const part = head(s.text, room);
    used += columns(part);
    out.push_back({part, s.style});
  }
  out.push_back({head(ellipsis, e), out.empty() ? Style::Plain : out.back().style});
  return out;
}

Row padded(Row row, size_t width) {
  if (size_t const n = columns(row); n < width) { row.push_back({std::string(width - n, ' '), Style::Plain}); }
  return row;
}

// `left` with `right` at the right edge of `width`, or as much of both as fits, left first.
Row spread(Row left, const Row& right, size_t width, std::string_view ellipsis) {
  size_t const l = columns(left);
  size_t const r = columns(right);
  if (l + 2 + r <= width) {
    left.push_back({std::string(width - l - r, ' '), Style::Plain});
    left.insert(left.end(), right.begin(), right.end());
    return left;
  }
  left.push_back({"  ", Style::Plain});
  left.insert(left.end(), right.begin(), right.end());
  return fit(left, width, ellipsis);
}

std::string clockText(double seconds) {
  u64 const s = u64(std::max(0.0, seconds));
  return s >= 3600 ? format("%" PRIu64 ":%02" PRIu64 ":%02" PRIu64, s / 3600, s / 60 % 60, s % 60)
                   : format("%" PRIu64 ":%02" PRIu64, s / 60, s % 60);
}

std::string percentText(double fraction) {
  return std::isfinite(fraction) ? format("%.2f%%", 100 * fraction) : std::string{"unknown"};
}

Row heading(const char* title, const std::string& note = {}) {
  return {{title, Style::Heading}, {note.empty() ? "" : "  " + note, Style::Dim}};
}

// --- the panels ---

Row titleRow(const Board& b, size_t width) {
  std::string const sep{glyphs(b).sep};
  Row left{{"PRPLL -tune", Style::Heading}, {"  " + b.device, Style::Plain}, {sep + b.scope, Style::Dim}};
  Row right{{"this run " + clockText(b.elapsed), Style::Plain}};
  if (b.generation) { right.push_back({sep + "generation " + std::to_string(b.generation), Style::Warn}); }
  if (columns(left) + 2 + columns(right) > width) { left.pop_back(); }
  return spread(std::move(left), right, width, glyphs(b).ellipsis);
}

Row statusRow(const Board& b) {
  if (!b.started) { return {{"starting", Style::Bold}}; }
  std::string const sep{glyphs(b).sep};
  const StateReport& s = b.state;
  Row out{{format("T %.2f us/it", s.T / std::max(s.coverage, 1e-9)), Style::Bold},
          {format(" over %.1f%% of the workload", 100 * s.coverage), Style::Plain},
          {sep + "expected regret " + percentText(s.regret), std::isfinite(s.regret) ? Style::Plain : Style::Dim},
          {sep + format("%u calls, %u accuracy readings", s.calls, s.gates), Style::Plain}};
  return out;
}

Row nowRow(const Board& b) {
  if (b.what.empty()) { return {{"now  ", Style::Heading}, {"choosing", Style::Dim}}; }
  return {{"now  ", Style::Heading}, {b.what, Style::Bold}, {"  " + clockText(b.onIt), Style::Dim}};
}

// `values` as `height` rows of blocks, the lowest one eighth of a row and the highest the whole height.
std::vector<Row> chart(const std::vector<double>& values, size_t height, size_t width, const Glyphs& g) {
  std::vector<double> v = values;
  if (v.size() > width) {
    std::vector<double> squeezed;
    for (size_t c = 0; c < width; ++c) { squeezed.push_back(v[c * v.size() / width]); }
    v = std::move(squeezed);
  }
  double const lo = v.empty() ? 0 : *std::ranges::min_element(v);
  double const hi = v.empty() ? 0 : *std::ranges::max_element(v);
  u32 const levels = u32(height) * 8;
  std::vector<Row> rows(height);
  for (double const x : v) {
    u32 const level = hi - lo < 1e-12 ? levels / 2 : 1 + u32(std::lround((x - lo) / (hi - lo) * (levels - 1)));
    for (size_t r = 0; r < height; ++r) {
      i64 const cell = std::clamp<i64>(i64(level) - i64(height - 1 - r) * 8, 0, 8);
      std::string const glyph = cell ? std::string{g.blocks[size_t(cell - 1)]} : " ";
      if (!rows[r].empty()) {
        rows[r].back().text += glyph;
      } else {
        rows[r].push_back({glyph, Style::Good});
      }
    }
  }
  return rows;
}

void addChart(std::vector<Row>& out, const char* name, const std::vector<double>& values, size_t height, size_t width,
              const Glyphs& g, bool percent) {
  auto text = [&](double x) { return percent ? percentText(x) : format("%.2f", x); };
  if (values.empty()) {
    out.push_back({{format("  %s: nothing yet", name), Style::Dim}});
    return;
  }
  double const lo = *std::ranges::min_element(values);
  double const hi = *std::ranges::max_element(values);
  out.push_back(fit(
    {{format("  %s: now %s, from %s", name, text(values.back()).c_str(), text(values.front()).c_str()), Style::Plain}},
    width, g.ellipsis));
  size_t const cw = width > 18 ? width - 18 : 1;
  std::vector<Row> const rows = chart(values, height, cw, g);
  for (size_t r = 0; r < height; ++r) {
    Row row{{"    ", Style::Plain}};
    row.insert(row.end(), rows[r].begin(), rows[r].end());
    row = padded(std::move(row), 4 + cw);
    if (r == 0) { row.push_back({"  " + text(hi), Style::Dim}); }
    if (r + 1 == height && height > 1) { row.push_back({"  " + text(lo), Style::Dim}); }
    out.push_back(fit(row, width, g.ellipsis));
  }
}

using Panel = std::vector<Row> (*)(const Board&, size_t width, size_t rows);

std::vector<Row> faultsPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Row> out;
  if (b.state.faults.empty()) { return out; }
  out.push_back({{"FAULTS", Style::Bad}, {format("  %zu", b.state.faults.size()), Style::Dim}});
  for (const std::string& f : b.state.faults) {
    if (out.size() >= std::min<size_t>(rows, 6)) { break; }
    out.push_back(fit({{"  " + f, Style::Warn}}, width, glyphs(b).ellipsis));
  }
  return out;
}

std::vector<Row> publishedPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Row> out{heading("PUBLISHED", format("%zu lines", b.state.lines.size()))};
  for (const std::string& line : b.state.lines) {
    if (out.size() >= rows) { break; }
    out.push_back(fit({{"  " + line, Style::Plain}}, width, glyphs(b).ellipsis));
  }
  return out;
}

std::vector<Row> modelPanel(const Board& b, size_t width, size_t) {
  std::vector<double> ratios;
  for (const ItemReport& r : b.recent) {
    // The posterior mean's estimate, not the draw's that chose the set: that one is optimistic by its selection.
    double const predicted = r.mean > 0 ? r.mean : r.predicted;
    if (r.measured > 0 && predicted > 0) { ratios.push_back(predicted / r.measured); }
    if (ratios.size() == CALIBRATION_CALLS) { break; }
  }
  std::string calibration = "no predictions yet";
  if (!ratios.empty()) {
    double logs = 0;
    for (double const r : ratios) { logs += std::log(r); }
    std::vector<double> errors;
    for (double const r : ratios) { errors.push_back(std::abs(std::log(r))); }
    std::ranges::nth_element(errors, errors.begin() + i64(errors.size() / 2));
    calibration = format("last %zu calls: mean predicted/measured %.3f, median error %.1f%%", ratios.size(),
                         std::exp(logs / double(ratios.size())), 100 * errors[errors.size() / 2]);
  }
  return {heading("MODEL", format("%u readings", b.state.observations)),
          fit({{"  " + calibration, Style::Plain}}, width, glyphs(b).ellipsis),
          fit({{"  spreads " + b.state.spreads, Style::Dim}}, width, glyphs(b).ellipsis)};
}

std::vector<Row> searchPanel(const Board& b, size_t width, size_t) {
  if (b.state.lanes.empty() && b.state.explored.empty()) { return {}; }
  return {heading("SEARCH"), fit({{"  lanes " + b.state.lanes, Style::Plain}}, width, glyphs(b).ellipsis),
          fit({{"  coverage " + b.state.explored, Style::Dim}}, width, glyphs(b).ellipsis)};
}

std::vector<Row> recentPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Row> out{heading("RECENT")};
  for (const ItemReport& r : b.recent) {
    if (out.size() >= rows) { break; }
    Row row{{format("  %4u ", r.n), Style::Dim},
            {r.kind == "gate" ? "gate " : "", Style::Plain},
            {r.what, Style::Plain},
            {"  " + r.result, Style::Bold}};
    if (r.predicted > 0) { row.push_back({format("  (predicted %.1f)", r.predicted), Style::Dim}); }
    row.push_back({"  " + r.options, Style::Dim});
    out.push_back(fit(row, width, glyphs(b).ellipsis));
  }
  return out;
}

std::vector<Row> eventsPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Row> out{heading("EVENTS")};
  for (const std::string& e : b.events) {
    if (out.size() >= rows) { break; }
    out.push_back(fit({{"  " + e, Style::Plain}}, width, glyphs(b).ellipsis));
  }
  return out;
}

// Panels one under another in `rows`, each given what is left, with a blank row between.
std::vector<Row> stack(const Board& b, std::initializer_list<Panel> panels, size_t width, size_t rows) {
  std::vector<Row> out;
  for (Panel const p : panels) {
    if (out.size() >= rows) { break; }
    std::vector<Row> const part = p(b, width, rows - out.size());
    if (part.empty()) { continue; }
    if (!out.empty()) { out.emplace_back(); }
    for (const Row& row : part) { out.push_back(fit(row, width, glyphs(b).ellipsis)); }
  }
  out.resize(std::min(out.size(), rows));
  return out;
}

}  // namespace

size_t columns(std::string_view text) {
  return size_t(std::ranges::count_if(text, [](char c) { return !isContinuation(c); }));
}

size_t columns(const Row& row) {
  size_t n = 0;
  for (const Span& s : row) { n += columns(s.text); }
  return n;
}

std::string plain(const Row& row) {
  std::string out;
  for (const Span& s : row) { out += s.text; }
  return out;
}

std::vector<Row> frame(const Board& b, size_t rows, size_t cols) {
  size_t const width = cols > 1 ? cols - 1 : 0;
  const Glyphs& g = glyphs(b);
  std::vector<Row> out;
  if (!rows || !width) { return out; }

  out.push_back(fit(titleRow(b, width), width, g.ellipsis));
  out.push_back(fit(statusRow(b), width, g.ellipsis));
  out.push_back(fit(nowRow(b), width, g.ellipsis));
  out.emplace_back();

  size_t const height = rows >= TALL_CHARTS_FROM ? 3 : 1;
  if (rows >= 24) {
    addChart(out, "T, us/it", b.T, height, width, g, false);
    addChart(out, "expected regret", b.regret, height, width, g, true);
    out.emplace_back();
  }

  if (out.size() < rows) {
    size_t const left = rows - out.size();
    if (width >= TWO_COLUMNS_FROM) {
      std::string const bar = " " + std::string{g.bar} + " ";
      size_t const lw = (width - columns(bar)) / 2;
      size_t const rw = width - columns(bar) - lw;
      std::vector<Row> const l =
        stack(b, {faultsPanel, publishedPanel, modelPanel, searchPanel, recentPanel}, lw, left);
      std::vector<Row> const r = stack(b, {eventsPanel}, rw, left);
      for (size_t i = 0; i < left; ++i) {
        Row row = padded(i < l.size() ? l[i] : Row{}, lw);
        row.push_back({bar, Style::Dim});
        if (i < r.size()) { row.insert(row.end(), r[i].begin(), r[i].end()); }
        out.push_back(fit(row, width, g.ellipsis));
      }
    } else {
      std::vector<Row> const all =
        stack(b, {faultsPanel, publishedPanel, modelPanel, searchPanel, recentPanel, eventsPanel}, width, left);
      out.insert(out.end(), all.begin(), all.end());
    }
  }

  out.resize(std::min(out.size(), rows));
  for (Row& row : out) {
    // Trailing blanks cost bytes on every repaint and draw nothing.
    while (!row.empty() && row.back().text.find_first_not_of(' ') == std::string::npos) { row.pop_back(); }
  }
  return out;
}

std::string paint(const std::vector<Row>& rows, bool color) {
  auto sgr = [](Style s) -> std::string_view {
    switch (s) {
    case Style::Plain: return "";
    case Style::Bold: return "\x1b[1m";
    case Style::Dim: return "\x1b[2m";
    case Style::Good: return "\x1b[32m";
    case Style::Bad: return "\x1b[1;31m";
    case Style::Warn: return "\x1b[33m";
    case Style::Heading: return "\x1b[1;36m";
    }
    return "";
  };
  std::string out = "\x1b[H";
  for (size_t i = 0; i < rows.size(); ++i) {
    for (const Span& s : rows[i]) {
      std::string_view const on = color ? sgr(s.style) : "";
      out += on;
      out += s.text;
      if (!on.empty()) { out += "\x1b[0m"; }
    }
    out += "\x1b[K";
    // A newline in the bottom row would scroll the screen.
    if (i + 1 < rows.size()) { out += "\r\n"; }
  }
  return out + "\x1b[J";
}

std::optional<std::string> eventOf(std::string_view line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.remove_suffix(1); }
  std::string time;
  // "YYYYMMDD HH:MM:SS <context> text", as log() prefixes every line.
  auto const digits = [&](size_t from, size_t n) {
    return line.size() >= from + n &&
      std::all_of(line.begin() + i64(from), line.begin() + i64(from + n), [](char c) { return c >= '0' && c <= '9'; });
  };
  if (digits(0, 8) && line.size() >= 17 && line[8] == ' ' && line[11] == ':' && line[14] == ':') {
    time = std::string{line.substr(9, 8)};
    line.remove_prefix(17);
    while (!line.empty() && line.front() == ' ') { line.remove_prefix(1); }
  }
  if (line.find_first_not_of(' ') == std::string_view::npos) { return std::nullopt; }

  constexpr std::string_view TUNE = "tune: ";
  if (line.starts_with(TUNE)) {
    std::string_view const rest = line.substr(TUNE.size());
    size_t const dot = rest.find(". ");
    bool const item = dot != std::string_view::npos && dot > 0 &&
      std::all_of(rest.begin(), rest.begin() + i64(dot), [](char c) { return c >= '0' && c <= '9'; });
    if (item) { return std::nullopt; }
    line = rest;
  }
  return (time.empty() ? std::string(8, ' ') : time) + " " + std::string{line};
}

bool utf8Terminal() {
  for (const char* name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
    const char* const value = getenv(name);
    if (!value || !*value) { continue; }
    std::string v{value};
    std::ranges::transform(v, v.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return v.find("utf-8") != std::string::npos || v.find("utf8") != std::string::npos;
  }
  return false;
}

bool colorTerminal() {
  const char* const no = getenv("NO_COLOR");
  return !no || !*no;
}

bool canTakeTerminal() {
#ifdef _WIN32
  return false;
#else
  const char* const term = getenv("TERM");
  return isatty(STDOUT_FILENO) && term && std::string_view{term} != "dumb";
#endif
}

namespace {

std::atomic<bool> screenTaken{false};

void writeOut(std::string_view bytes) {
  fwrite(bytes.data(), 1, bytes.size(), stdout);
  fflush(stdout);
}

void giveScreenBack() {
  if (screenTaken.exchange(false)) { writeOut(LEAVE); }
}

void takeScreen() {
  // For an exit() from anywhere, which no destructor on the way would see.
  static std::once_flag registered;
  std::call_once(registered, [] { std::atexit(giveScreenBack); });
  if (!screenTaken.exchange(true)) { writeOut(ENTER); }
}

std::pair<size_t, size_t> terminalSize() {
#ifndef _WIN32
  winsize ws{};
  if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) && ws.ws_col && ws.ws_row) { return {ws.ws_row, ws.ws_col}; }
#endif
  return {24, 80};
}

}  // namespace

DashboardView::DashboardView(std::string device, std::string scope, u32 generation) :
  color_{colorTerminal()}, start_{Clock::now()} {
  board_.device = std::move(device);
  board_.scope = std::move(scope);
  board_.generation = generation;
  board_.unicode = utf8Terminal();

  takeScreen();
  setStdoutSink([this](std::string_view s) { take(s); });
  restart::beforeExec([this] { close(); });
  ticker_ = std::thread{[this] { tick(); }};
}

DashboardView::~DashboardView() {
  {
    std::unique_lock const lock{mutex_};
    stopping_ = true;
  }
  wake_.notify_all();
  ticker_.join();

  restart::beforeExec(nullptr);
  close();
  setStdoutSink(nullptr);
}

void DashboardView::measuring(const std::string& what) {
  std::unique_lock const lock{mutex_};
  board_.what = what;
  whatSince_ = Clock::now();
}

void DashboardView::finished(const ItemReport& item) {
  std::unique_lock const lock{mutex_};
  board_.recent.push_front(item);
  if (board_.recent.size() > RECENT_KEPT) { board_.recent.pop_back(); }
  board_.what.clear();
}

void DashboardView::state(const StateReport& s) {
  std::unique_lock const lock{mutex_};
  bool const newRegret = std::isfinite(s.regret) && s.regret != board_.state.regret;
  board_.state = s;
  board_.started = true;
  if (s.coverage > 0) {
    board_.T.push_back(s.T / s.coverage);
    if (board_.T.size() > HISTORY_KEPT) { board_.T.erase(board_.T.begin()); }
  }
  if (newRegret) { board_.regret.push_back(s.regret); }
}

void DashboardView::take(std::string_view text) {
  std::unique_lock const lock{mutex_};
  if (closed_) {
    writeOut(text);
    return;
  }
  partial_ += text;
  for (size_t nl = partial_.find('\n'); nl != std::string::npos; nl = partial_.find('\n')) {
    if (std::optional<std::string> event = eventOf(std::string_view{partial_}.substr(0, nl))) {
      board_.events.push_front(std::move(*event));
      if (board_.events.size() > EVENTS_KEPT) { board_.events.pop_back(); }
    }
    partial_.erase(0, nl + 1);
  }
}

void DashboardView::tick() {
  std::unique_lock lock{mutex_};
  while (!stopping_) {
    if (!closed_) {
      auto const now = Clock::now();
      board_.elapsed = std::chrono::duration<double>(now - start_).count();
      board_.onIt = board_.what.empty() ? 0 : std::chrono::duration<double>(now - whatSince_).count();
      auto const [rows, cols] = terminalSize();
      writeOut(paint(frame(board_, rows, cols), color_));
    }
    wake_.wait_for(lock, std::chrono::duration<double>(DASHBOARD_TICK_SEC), [this] { return stopping_; });
  }
}

void DashboardView::close() {
  std::unique_lock const lock{mutex_};
  if (closed_) { return; }
  giveScreenBack();
  closed_ = true;
}

}  // namespace tune
