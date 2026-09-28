// Copyright (C) Jason Lynch

#include "Dashboard.h"

#include "File.h"
#include "log.h"
#include "Restart.h"
#include "Status.h"
#include "Summary.h"

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
#include <iterator>
#include <map>
#include <mutex>
#include <tuple>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace tune {

namespace {

// The alternate screen with the cursor hidden and no wrapping at the right edge, and back.
constexpr std::string_view ENTER = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[H\x1b[2J";
constexpr std::string_view LEAVE = "\x1b[?7h\x1b[?25h\x1b[?1049l";

// What a measured share is taken to be whole at: the objective sums its weights in floating point.
constexpr double WHOLE = 0.9995;

#ifdef __GNUC__
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

[[nodiscard]] std::string format(const char* fmt, ...) {
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
  std::string_view arrow;
};

constexpr Glyphs UNICODE{"…", "│", "  ·  ", {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}, "→"};
constexpr Glyphs ASCII{"...", "|", "  |  ", {"_", ".", "-", "~", "=", "+", "*", "#"}, "->"};

[[nodiscard]] const Glyphs& glyphs(const Board& b) { return b.unicode ? UNICODE : ASCII; }

[[nodiscard]] bool isContinuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

// The first `width` columns of `text`.
[[nodiscard]] std::string head(std::string_view text, size_t width) {
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

// The last `width` columns of `text`.
[[nodiscard]] std::string tail(std::string_view text, size_t width) {
  size_t cols = 0;
  size_t at = text.size();
  while (at > 0 && cols < width) {
    --at;
    while (at > 0 && isContinuation(text[at])) { --at; }
    ++cols;
  }
  return std::string{text.substr(at)};
}

// `text` in `width` columns, its end given up for the ellipsis.
[[nodiscard]] std::string clipEnd(std::string_view text, size_t width, std::string_view ellipsis) {
  if (columns(text) <= width) { return std::string{text}; }
  size_t const e = columns(ellipsis);
  if (width <= e) { return head(text, width); }
  return head(text, width - e) + std::string{ellipsis};
}

// `text` in `width` columns, its middle given up: a label's entry is at its start and its exponent at its end.
[[nodiscard]] std::string clipMiddle(std::string_view text, size_t width, std::string_view ellipsis) {
  if (columns(text) <= width) { return std::string{text}; }
  size_t const e = columns(ellipsis);
  if (width <= e + 1) { return head(text, width); }
  size_t const keep = width - e;
  size_t const front = (keep + 1) / 2;
  return head(text, front) + std::string{ellipsis} + tail(text, keep - front);
}

// `line` cut to `width` columns, the ellipsis ending it where anything was cut.
[[nodiscard]] Line fit(const Line& line, size_t width, std::string_view ellipsis) {
  if (columns(line) <= width) { return line; }
  size_t const e = std::min(columns(ellipsis), width);
  Line out;
  size_t used = 0;
  for (const Span& s : line) {
    size_t const room = width - e - used;
    if (!room) { break; }
    std::string const part = head(s.text, room);
    used += columns(part);
    out.push_back({part, s.style});
  }
  out.push_back({head(ellipsis, e), out.empty() ? Style::Plain : out.back().style});
  return out;
}

[[nodiscard]] Line padded(Line line, size_t width) {
  if (size_t const n = columns(line); n < width) { line.push_back({std::string(width - n, ' '), Style::Plain}); }
  return line;
}

// `left` with `right` at the right edge of `width`, or as much of both as fits, left first.
[[nodiscard]] Line spread(Line left, const Line& right, size_t width, std::string_view ellipsis) {
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

[[nodiscard]] std::string localClock(const char* pattern, std::time_t t) {
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &t);
#else
  localtime_r(&t, &local);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), pattern, &local);
  return buf;
}

[[nodiscard]] std::string plural(u64 n, const char* one, const char* many) {
  return std::to_string(n) + " " + (n == 1 ? one : many);
}

// --- the panels ---

[[nodiscard]] Line headingLine(const char* title, const std::string& note) {
  return {{title, Style::Heading}, {note.empty() ? "" : "  " + note, Style::Dim}};
}

[[nodiscard]] Line titleLine(const Board& b, size_t width) {
  std::string const sep{glyphs(b).sep};
  Line left{{"PRPLL -tune", Style::Heading}, {"  " + b.device, Style::Plain}, {sep + b.scope, Style::Dim}};
  Line right{
    {"this run " + clockText(b.elapsed), Style::Plain},
    {sep + "measured " + clockText(b.before + b.elapsed) + " over " + plural(b.sessions, "session", "sessions"),
     Style::Dim}};
  if (b.generation) { right.push_back({sep + "generation " + std::to_string(b.generation), Style::Warn}); }
  // A restart's generation matters more than the scope, which the operator typed.
  if (columns(left) + 2 + columns(right) > width) { left.pop_back(); }
  if (columns(left) + 2 + columns(right) > width) {
    right.erase(right.begin() + 1, right.end() - (b.generation ? 1 : 0));
  }
  return spread(std::move(left), right, width, glyphs(b).ellipsis);
}

[[nodiscard]] Line statusLine(const Board& b) {
  if (!b.started) { return {{"starting", Style::Bold}}; }
  std::string const sep{glyphs(b).sep};
  const RunProgress& p = b.progress;
  Line out{{p.phase.text.empty() ? b.phase : p.phase.text, Style::Bold}};
  out.push_back({sep + format("T %.3f us/it, %.1f%% measured", p.T, 100 * p.measured), Style::Plain});
  if (p.mostWorth > 0) {
    out.push_back({sep +
                     (p.stop > 0 && p.floor > 0
                        ? format("best item worth %.4f us/it, %.1fx the stop floor", p.mostWorth, p.mostWorth / p.floor)
                        : format("best item worth %.4f us/it, stop=0", p.mostWorth)),
                   Style::Plain});
  }
  if (p.worthRunning) {
    out.push_back({sep +
                     format("%s worth running, ~%s", plural(p.worthRunning, "item", "items").c_str(),
                            ago(u64(std::lround(p.worthSeconds))).c_str()),
                   Style::Dim});
  }
  if (p.unlisted) { out.push_back({format(", up to %" PRIu64 " more not listed yet", p.unlisted), Style::Dim}); }
  return out;
}

void addFaults(std::vector<Line>& out, const Board& b, size_t width) {
  if (b.faults.empty()) { return; }
  auto const lost = std::ranges::count(b.faults, Fault::What::Lost, &Fault::what);
  auto const wrong = std::ranges::count(b.faults, Fault::What::Wrong, &Fault::what);
  std::string counts;
  if (lost) { counts += plural(u64(lost), "configuration", "configurations") + " took the device down"; }
  if (wrong) {
    counts +=
      (counts.empty() ? "" : ", ") + plural(u64(wrong), "configuration", "configurations") + " computed wrong answers";
  }
  const Glyphs& g = glyphs(b);
  out.push_back(
    fit({{"FAULTS", Style::Bad},
         {"  " + counts + ": kernel or driver bugs, held out of the search; -tune status lists them", Style::Bad}},
        width, g.ellipsis));
  size_t shown = 0;
  for (auto it = b.faults.rbegin(); it != b.faults.rend() && shown < FAULTS_SHOWN; ++it, ++shown) {
    std::string when = it->ts ? localClock("%Y-%m-%d %H:%M", std::time_t(it->ts)) : "";
    if (it->gen) { when += (when.empty() ? "" : ", ") + std::string{"generation "} + std::to_string(it->gen); }
    out.push_back(fit({{it->what == Fault::What::Lost ? "  lost   " : "  wrong  ", Style::Bad},
                       {reproduce(*it), Style::Plain},
                       {when.empty() ? "" : "   " + when, Style::Dim}},
                      width, g.ellipsis));
  }
  if (b.faults.size() > shown) {
    out.push_back({{"  and " + std::to_string(b.faults.size() - shown) + " more", Style::Dim}});
  }
  out.emplace_back();
}

[[nodiscard]] Line nowLine(const Board& b, size_t width) {
  Line out{{"NOW", Style::Heading}, {"  ", Style::Plain}};
  if (b.what.empty()) {
    out.push_back({b.started ? "choosing the next item" : "starting", Style::Dim});
    return fit(out, width, glyphs(b).ellipsis);
  }
  std::string const clock = "  (" + clockText(b.onIt) + ")";
  size_t const fixed = columns(out) + columns(clock);
  out.push_back({clipMiddle(b.what, width > fixed ? width - fixed : 0, glyphs(b).ellipsis), Style::Bold});
  out.push_back({clock, Style::Dim});
  return fit(out, width, glyphs(b).ellipsis);
}

struct Series {
  std::string name;
  std::vector<std::optional<double>> values;

  // What to say where nothing is drawn yet.
  std::string empty = "nothing measured yet";
};

// The value in force at each of `width` columns spanning `total` seconds of measuring: the last point at or before the
// column's end, or nothing before the first.
[[nodiscard]] Series resample(const std::vector<HistoryPoint>& points, double total, size_t width, bool probe) {
  Series out;
  out.values.resize(width);
  if (!probe) { out.empty = "not measured over the whole workload yet"; }
  size_t p = 0;
  std::optional<size_t> last;
  for (size_t c = 0; c < width; ++c) {
    double const end = total * double(c + 1) / double(width);
    while (p < points.size() && points[p].active <= end) { last = p++; }
    if (!last) { continue; }
    const HistoryPoint& h = points[*last];
    // Where the prior still prices part of the workload T is partly an estimate, and the step from the estimate to
    // what was measured would set the scale of the whole chart.
    if (probe) {
      out.values[c] = h.probe;
    } else if (h.measured >= WHOLE) {
      out.values[c] = h.T;
    }
  }
  return out;
}

// `series` as `height` rows of blocks, the lowest value one eighth of a row and the highest the whole height.
[[nodiscard]] std::vector<Line> chart(const Series& series, size_t height, const Glyphs& g) {
  double lo = 0;
  double hi = 0;
  bool any = false;
  for (const std::optional<double>& v : series.values) {
    if (!v) { continue; }
    lo = any ? std::min(lo, *v) : *v;
    hi = any ? std::max(hi, *v) : *v;
    any = true;
  }
  u32 const levels = u32(height) * 8;
  std::vector<Line> rows(height);
  for (size_t c = 0; c < series.values.size(); ++c) {
    const std::optional<double>& v = series.values[c];
    u32 const level = !v ? 0 : hi - lo < 1e-9 ? levels / 2 : 1 + u32(std::lround((*v - lo) / (hi - lo) * (levels - 1)));
    Style const style = Style::Good;
    for (size_t r = 0; r < height; ++r) {
      i64 const cell = std::clamp<i64>(i64(level) - i64(height - 1 - r) * 8, 0, 8);
      std::string const glyph = cell ? std::string{g.blocks[size_t(cell - 1)]} : " ";
      Line& row = rows[r];
      if (!row.empty() && row.back().style == style) {
        row.back().text += glyph;
      } else {
        row.push_back({glyph, style});
      }
    }
  }
  return rows;
}

[[nodiscard]] std::pair<double, double> rangeOf(const Series& s) {
  double lo = 0;
  double hi = 0;
  bool any = false;
  for (const std::optional<double>& v : s.values) {
    if (!v) { continue; }
    lo = any ? std::min(lo, *v) : *v;
    hi = any ? std::max(hi, *v) : *v;
    any = true;
  }
  return {lo, hi};
}

[[nodiscard]] std::optional<double> lastOf(const Series& s) {
  for (auto it = s.values.rbegin(); it != s.values.rend(); ++it) {
    if (*it) { return **it; }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<double> firstOf(const Series& s) {
  for (const std::optional<double>& v : s.values) {
    if (v) { return *v; }
  }
  return std::nullopt;
}

// Each chart has a row of its own saying what it is, so the margins only hold the scale.
constexpr size_t CHART_LEFT = 4;
constexpr size_t CHART_RIGHT = 14;

// `s` under a row naming it and saying where it has gone, with its highest value beside its top row and its lowest
// beside its bottom one.
void addChart(std::vector<Line>& out, const Series& s, size_t height, size_t width, const Glyphs& g) {
  size_t const cw = width - CHART_LEFT - CHART_RIGHT;
  std::vector<Line> const rows = chart(s, height, g);
  auto const [lo, hi] = rangeOf(s);
  std::optional<double> const first = firstOf(s);
  std::optional<double> const last = lastOf(s);

  std::string title = "  " + s.name + ": ";
  title += last ? format("now %.3f us/it, from %.3f", *last, *first) : s.empty;
  out.push_back(fit({{title, Style::Plain}}, width, g.ellipsis));

  for (size_t r = 0; r < height; ++r) {
    std::string right;
    if (last && height == 1) {
      right = format("  %.0f-%.0f", lo, hi);
    } else if (last && r == 0) {
      right = format("  %.3f", hi);
    } else if (last && r + 1 == height) {
      right = format("  %.3f", lo);
    }
    Line line{{std::string(CHART_LEFT, ' '), Style::Plain}};
    line.insert(line.end(), rows[r].begin(), rows[r].end());
    line = padded(std::move(line), CHART_LEFT + cw);
    line.push_back({right, Style::Dim});
    out.push_back(fit(line, width, g.ellipsis));
  }
}

// Under a chart: measuring time from the env's first session to now, and where this process began, which is placed
// first; the ends are labelled where that leaves room.
void addAxis(std::vector<Line>& out, const Board& b, size_t width, double total, const Glyphs& g) {
  size_t const cw = width - CHART_LEFT - CHART_RIGHT;
  std::string axis(cw, ' ');
  std::vector<bool> taken(cw, false);
  auto place = [&](size_t at, std::string_view text) {
    if (at + text.size() > cw) { return false; }
    for (size_t i = at > 0 ? at - 1 : 0; i < std::min(cw, at + text.size() + 1); ++i) {
      if (taken[i]) { return false; }
    }
    axis.replace(at, text.size(), text);
    std::fill(taken.begin() + i64(at), taken.begin() + i64(at + text.size()), true);
    return true;
  };
  size_t const mark = std::min(cw - 1, size_t(double(cw) * b.before / total));
  std::string_view constexpr AFTER = "^ this run";
  std::string_view constexpr BEFORE = "this run ^";
  if (!place(mark, AFTER) && !(mark + 1 >= BEFORE.size() && place(mark + 1 - BEFORE.size(), BEFORE))) {
    (void)place(mark, "^");
  }
  (void)place(0, "0:00");
  std::string const to = clockText(total) + " measured";
  if (to.size() <= cw) { (void)place(cw - to.size(), to); }
  out.push_back(fit({{std::string(CHART_LEFT, ' '), Style::Plain}, {axis, Style::Dim}}, width, g.ellipsis));
}

void addBenefit(std::vector<Line>& out, const Board& b, size_t width, size_t chartHeight) {
  const Glyphs& g = glyphs(b);
  std::optional<double> const saving = b.benefit.saving();
  std::string_view constexpr TITLE = "BENEFIT  ";
  Style const style = saving && *saving > 0 ? Style::Good : Style::Plain;
  std::string text = b.benefit.probe ? benefitText(b.benefit) : "nothing measured yet";
  if (b.historyPending) { text += "  (reading the database's history" + std::string{g.ellipsis} + ")"; }
  // The one line worth reading whole, so it takes a second row rather than lose its end.
  std::string rest;
  if (size_t const room = width > TITLE.size() ? width - TITLE.size() : 0; columns(text) > room) {
    if (size_t const cut = text.rfind(' ', room); cut != std::string::npos && cut > room / 2) {
      rest = text.substr(cut + 1);
      text.resize(cut);
    }
  }
  out.push_back(fit({{std::string{TITLE.substr(0, 7)}, Style::Heading}, {"  " + text, style}}, width, g.ellipsis));
  if (!rest.empty()) { out.push_back(fit({{std::string(TITLE.size(), ' ') + rest, style}}, width, g.ellipsis)); }

  if (!chartHeight || width < CHART_LEFT + CHART_RIGHT + 16) { return; }
  size_t const cw = width - CHART_LEFT - CHART_RIGHT;
  double const total = std::max(b.before + b.elapsed, b.history.empty() ? 0.0 : b.history.back().active);
  if (total <= 0) { return; }

  Series T = resample(b.history, total, cw, false);
  T.name = "T, the time per iteration over the workload";
  addChart(out, T, chartHeight, width, g);
  addAxis(out, b, width, total, g);
  out.emplace_back();
  Series probe = resample(b.history, total, cw, true);
  probe.name = "what production runs at the probe " + std::to_string(b.benefit.probe);
  addChart(out, probe, chartHeight, width, g);
  addAxis(out, b, width, total, g);
}

[[nodiscard]] std::vector<Line> productionPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Line> out;
  if (!rows) { return out; }
  const Glyphs& g = glyphs(b);
  out.push_back(fit(
    headingLine("PRODUCTION", "what selection.txt runs over the workload; change against the fastest untuned there"),
    width, g.ellipsis));
  if (b.production.empty()) {
    if (rows > 1) { out.push_back({{"  nothing yet", Style::Dim}}); }
    return out;
  }

  // Past what fits, the stretches that carry the least weight are counted rather than listed.
  std::vector<size_t> shown(b.production.size());
  for (size_t i = 0; i < shown.size(); ++i) { shown[i] = i; }
  size_t const room = rows - 1;
  double hidden = 0;
  if (shown.size() > room) {
    size_t const keep = room ? room - 1 : 0;
    std::ranges::stable_sort(shown,
                             [&](size_t x, size_t y) { return b.production[x].weight > b.production[y].weight; });
    for (size_t i = keep; i < shown.size(); ++i) { hidden += b.production[shown[i]].weight; }
    shown.resize(keep);
    std::ranges::sort(shown);
  }

  bool const kinds =
    std::ranges::any_of(b.production, [&](const Stretch& s) { return s.kind != b.production[0].kind; });
  auto rangeOf = [](const Stretch& s) { return format("%" PRIu64 "-%" PRIu64, s.lo, s.hi); };
  size_t rangeWidth = 0;
  size_t fftWidth = 0;
  for (size_t const i : shown) {
    rangeWidth = std::max(rangeWidth, rangeOf(b.production[i]).size());
    fftWidth = std::max(fftWidth, b.production[i].fft.size());
  }
  for (size_t const i : shown) {
    const Stretch& s = b.production[i];
    Line line{{"  ", Style::Plain}};
    if (kinds) { line.push_back({std::string{toString(s.kind)} + " ", Style::Dim}); }
    std::string range = rangeOf(s);
    range.resize(rangeWidth, ' ');
    line.push_back({range, Style::Plain});
    line.push_back({format("  %5.1f%%  ", 100 * s.weight), Style::Dim});
    std::string fft = s.fft;
    fft.resize(fftWidth, ' ');
    line.push_back({fft, s.measured ? Style::Plain : Style::Dim});
    line.push_back({format("  %9.3f us/it", s.us), s.measured ? Style::Plain : Style::Dim});
    if (!s.measured) {
      line.push_back({"  prior", Style::Dim});
    } else if (s.moved && !s.wasMeasured) {
      line.push_back({"  not read at the defaults", Style::Dim});
    } else if (!s.moved) {
      line.push_back({"  untuned", Style::Dim});
    } else if (s.was > 0) {
      double const change = (s.us - s.was) / s.was;
      line.push_back({format("  %+.2f%%", 100 * change),
                      change < 0     ? Style::Good
                        : change > 0 ? Style::Bad
                                     : Style::Dim});
    }
    out.push_back(fit(line, width, g.ellipsis));
  }
  if (shown.size() < b.production.size()) {
    out.push_back(
      {{format("  and %zu more stretches, %.1f%% of the weight", b.production.size() - shown.size(), 100 * hidden),
        Style::Dim}});
  }
  return out;
}

[[nodiscard]] std::vector<Line> samplesPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Line> out;
  if (!rows) { return out; }
  const Glyphs& g = glyphs(b);
  out.push_back(
    fit(headingLine("SAMPLES", "across the workload: the fastest at the built-in defaults, and what runs there now"),
        width, g.ellipsis));
  if (b.samples.empty()) {
    if (rows > 1) { out.push_back({{"  nothing yet", Style::Dim}}); }
    return out;
  }

  size_t fftWidth = 0;
  for (const Sample& s : b.samples) {
    if (s.untuned) { fftWidth = std::max(fftWidth, s.untuned->fft.size()); }
  }
  bool const kinds = std::ranges::any_of(b.samples, [&](const Sample& s) { return s.kind != b.samples[0].kind; });
  for (const Sample& s : b.samples) {
    if (out.size() == rows) { break; }
    Line line{{s.probe ? "* " : "  ", Style::Bold}};
    if (kinds) { line.push_back({std::string{toString(s.kind)} + " ", Style::Dim}); }
    line.push_back({format("%10" PRIu64 "  ", s.exponent), Style::Plain});
    std::string from = s.untuned ? format("%s %9.3f", s.untuned->fft.c_str(), s.untuned->us) : "not read";
    from.resize(std::max(from.size(), fftWidth + 10), ' ');
    line.push_back({from, Style::Dim});
    line.push_back({"  " + std::string{g.arrow} + "  ", Style::Dim});
    bool const measured = s.now && s.now->measured();
    line.push_back({s.now ? format("%s %9.3f", s.now->fft.c_str(), s.now->us) : std::string{"-"},
                    measured ? Style::Plain : Style::Dim});
    if (!measured) {
      line.push_back({"  prior", Style::Dim});
    } else if (s.untuned && s.untuned->us > 0) {
      double const change = (s.now->us - s.untuned->us) / s.untuned->us;
      line.push_back({format("  %+.2f%%", 100 * change),
                      change < 0     ? Style::Good
                        : change > 0 ? Style::Bad
                                     : Style::Dim});
    }
    out.push_back(fit(line, width, g.ellipsis));
  }
  return out;
}

[[nodiscard]] Line recentLine(const Recent& r, size_t width, const Glyphs& g) {
  const Finished& f = r.item;
  double const change = f.after - f.before;
  bool const better = !f.best.empty() || change < -1e-9;

  Line right;
  if (!f.completed) {
    right.push_back({"gave no reading", Style::Bad});
  } else if (f.reads) {
    // The outcome is the part worth reading, but never at the cost of which set it was for.
    right.push_back({format("z %.2f", f.z), f.checkOk ? Style::Plain : Style::Bad});
    std::string const outcome = ", " + f.outcome;
    right.push_back({clipEnd(outcome, std::max<size_t>(24, width * 2 / 5), g.ellipsis), Style::Dim});
  } else {
    right.push_back({format("%9.3f us/it", f.usPerIt), better ? Style::Good : Style::Plain});
    right.push_back({format("  %5.1f s", f.seconds), Style::Dim});
    right.push_back({format("  T %+.3f", change),
                     change < -1e-9    ? Style::Good
                       : change > 1e-9 ? Style::Bad
                                       : Style::Dim});
  }

  std::string const left = format("%5u  ", f.n) + r.at + "  " + toString(f.kind) + " " + f.label + " at " +
    std::to_string(f.exponent) + f.call;
  size_t const rightWidth = columns(right);
  size_t const leftWidth = width > rightWidth + 3 ? width - rightWidth - 3 : 0;
  Line line{{" " + clipMiddle(left, leftWidth, g.ellipsis), better ? Style::Good : Style::Plain}};
  line = padded(std::move(line), leftWidth + 3);
  line.insert(line.end(), right.begin(), right.end());
  return fit(line, width, g.ellipsis);
}

[[nodiscard]] std::vector<Line> recentPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Line> out;
  if (!rows) { return out; }
  const Glyphs& g = glyphs(b);
  out.push_back(fit(headingLine("RECENT", "items this run, newest first"), width, g.ellipsis));
  if (b.recent.empty() && rows > 1) { out.push_back({{"  nothing yet", Style::Dim}}); }
  for (const Recent& r : b.recent) {
    if (out.size() >= rows) { break; }
    out.push_back(recentLine(r, width, g));
    if (!r.item.best.empty() && out.size() < rows) {
      out.push_back(fit({{"      now best at " + r.item.best, Style::Good}}, width, g.ellipsis));
    }
  }
  return out;
}

[[nodiscard]] std::vector<Line> nextPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Line> out;
  if (!rows) { return out; }
  const Glyphs& g = glyphs(b);
  out.push_back(fit(headingLine("NEXT", "what the queue ranks highest, best rate first"), width, g.ellipsis));
  if (b.next.empty()) {
    if (rows > 1) { out.push_back({{b.started ? "  nothing is worth running" : "  not ranked yet", Style::Dim}}); }
    return out;
  }
  size_t const listed = std::min(b.next.size(), rows > 2 ? rows - 2 : rows - 1);
  for (size_t i = 0; i < listed; ++i) {
    const NextUp& n = b.next[i];
    std::string const right =
      (n.byRule ? std::string{"by rule"} : format("%.4f us/it", n.value)) + ", ~" + ago(u64(std::lround(n.seconds)));
    std::string const left = format("  %2zu. ", i + 1) + toString(n.kind) + " " + n.label;
    size_t const leftWidth = width > columns(right) + 2 ? width - columns(right) - 2 : 0;
    Line line{{clipMiddle(left, leftWidth, g.ellipsis), Style::Plain}};
    line = padded(std::move(line), leftWidth + 2);
    line.push_back({right, n.byRule ? Style::Warn : Style::Dim});
    out.push_back(fit(line, width, g.ellipsis));
  }
  if (out.size() < rows && b.progress.worthRunning > listed) {
    std::string more = format("  and %u more worth running", b.progress.worthRunning - u32(listed));
    if (b.progress.unlisted) { more += format(", up to %" PRIu64 " more not listed yet", b.progress.unlisted); }
    out.push_back(fit({{more, Style::Dim}}, width, g.ellipsis));
  }
  return out;
}

[[nodiscard]] Style eventStyle(std::string_view text) {
  for (std::string_view bad : {"Warning", "warning", "could not", "Could not", "was lost", "went away", "fault"}) {
    if (text.find(bad) != std::string_view::npos) { return Style::Bad; }
  }
  for (std::string_view good : {"is now best", "the default lines are now", "bootstrap complete"}) {
    if (text.find(good) != std::string_view::npos) { return Style::Good; }
  }
  return Style::Plain;
}

[[nodiscard]] std::vector<Line> eventsPanel(const Board& b, size_t width, size_t rows) {
  std::vector<Line> out;
  if (!rows) { return out; }
  const Glyphs& g = glyphs(b);
  out.push_back(fit(headingLine("EVENTS", "the rest of the log, newest first"), width, g.ellipsis));
  if (b.events.empty() && rows > 1) { out.push_back({{"  nothing yet", Style::Dim}}); }
  for (const std::string& e : b.events) {
    if (out.size() >= rows) { break; }
    out.push_back({{"  " + clipEnd(e, width > 2 ? width - 2 : 0, g.ellipsis), eventStyle(e)}});
  }
  return out;
}

using Panel = std::vector<Line> (*)(const Board&, size_t, size_t);

// The rows each of `want` gets out of `rows`, a row at a time to each in turn that wants more, and anything left over
// to the last.
[[nodiscard]] std::vector<size_t> allocate(const std::vector<size_t>& want, size_t rows) {
  std::vector<size_t> out(want.size(), 0);
  bool given = true;
  while (rows && given) {
    given = false;
    for (size_t i = 0; i < want.size() && rows; ++i) {
      if (out[i] < want[i]) {
        ++out[i];
        --rows;
        given = true;
      }
    }
  }
  if (!out.empty()) { out.back() += rows; }
  return out;
}

[[nodiscard]] size_t wantOf(const Board& b, Panel panel) {
  if (panel == productionPanel) { return 1 + std::max<size_t>(1, b.production.size()); }
  if (panel == samplesPanel) { return 1 + std::max<size_t>(1, b.samples.size()); }
  if (panel == nextPanel) { return 2 + std::max<size_t>(1, b.next.size()); }
  if (panel == recentPanel) {
    size_t n = 1;
    for (const Recent& r : b.recent) { n += r.item.best.empty() ? 1 : 2; }
    return std::max<size_t>(n, 2);
  }
  return 1 + std::max<size_t>(1, b.events.size());
}

// `panels` stacked in `rows` rows of `width`, a blank row between each two.
[[nodiscard]] std::vector<Line> stack(const Board& b, const std::vector<Panel>& panels, size_t width, size_t rows) {
  std::vector<Line> out;
  if (panels.empty()) { return out; }
  size_t const gaps = panels.size() - 1;
  if (rows <= gaps) { return out; }
  std::vector<size_t> want;
  for (Panel p : panels) { want.push_back(wantOf(b, p)); }
  std::vector<size_t> const got = allocate(want, rows - gaps);
  for (size_t i = 0; i < panels.size(); ++i) {
    if (i) { out.emplace_back(); }
    std::vector<Line> lines = panels[i](b, width, got[i]);
    lines.resize(std::min(lines.size(), got[i]));
    lines.resize(got[i]);
    std::ranges::move(lines, std::back_inserter(out));
  }
  return out;
}

}  // namespace

size_t columns(std::string_view text) {
  return size_t(std::ranges::count_if(text, [](char c) { return !isContinuation(c); }));
}

size_t columns(const Line& line) {
  size_t n = 0;
  for (const Span& s : line) { n += columns(s.text); }
  return n;
}

std::string plain(const Line& line) {
  std::string out;
  for (const Span& s : line) { out += s.text; }
  return out;
}

std::vector<ObjectivePoint> untunedPoints(const std::vector<ObjectivePoint>& now, const Untuned& untuned) {
  std::vector<ObjectivePoint> out = now;
  for (ObjectivePoint& p : out) { p.cost = untuned.at(p.kind, p.exponent); }
  return out;
}

std::vector<u64> sampleExponents(const std::vector<ObjectivePoint>& points, TestKind kind, u64 probe, size_t n) {
  std::vector<u64> all;
  bool probed = false;
  for (const ObjectivePoint& p : points) {
    if (p.kind != kind || !p.cost) { continue; }
    if (p.exponent == probe) { probed = true; }
    if (p.weight > 0) { all.push_back(p.exponent); }
  }
  std::ranges::sort(all);
  auto const [first, last] = std::ranges::unique(all);
  all.erase(first, last);
  if (!n) { return {}; }

  std::vector<u64> out;
  if (probed) { out.push_back(probe); }
  size_t const spread = n - out.size();
  if (spread && !all.empty()) {
    for (size_t k = 0; k < spread; ++k) {
      size_t const i = spread == 1 ? 0 : k * (all.size() - 1) / (spread - 1);
      out.push_back(all[i]);
    }
  }
  std::ranges::sort(out);
  auto const [from, to] = std::ranges::unique(out);
  out.erase(from, to);
  return out;
}

std::vector<Stretch> stretchesOf(const std::vector<ObjectivePoint>& now, const std::vector<ObjectivePoint>& reference,
                                 const std::vector<SelectionEntry>& entries) {
  std::map<std::pair<TestKind, u64>, const Cost*> before;
  for (const ObjectivePoint& p : reference) {
    if (p.cost) { before[{p.kind, p.exponent}] = &*p.cost; }
  }

  std::vector<const ObjectivePoint*> points;
  for (const ObjectivePoint& p : now) {
    if (p.cost) { points.push_back(&p); }
  }
  std::ranges::stable_sort(points, [](const ObjectivePoint* a, const ObjectivePoint* b) {
    return std::tuple{a->kind, a->exponent} < std::tuple{b->kind, b->exponent};
  });

  std::vector<Stretch> out;
  std::string key;
  // Weighted, and plain for a stretch whose every point weighs nothing (a probe with probeWeight=0, say).
  double usWeighted = 0;
  double wasWeighted = 0;
  double weights = 0;
  double usPlain = 0;
  double wasPlain = 0;
  u32 count = 0;
  auto close = [&] {
    if (out.empty() || !count) { return; }
    Stretch& s = out.back();
    s.us = weights > 0 ? usWeighted / weights : usPlain / count;
    s.was = weights > 0 ? wasWeighted / weights : wasPlain / count;
  };
  for (const ObjectivePoint* p : points) {
    const Cost& c = *p->cost;
    std::string const k = std::string{toString(p->kind)} + " " + (c.measured() ? c.entry : "prior " + c.fft);
    if (out.empty() || k != key || out.back().kind != p->kind) {
      close();
      out.push_back({.kind = p->kind, .lo = p->exponent, .fft = c.fft, .entry = c.entry, .measured = c.measured()});
      key = k;
      usWeighted = wasWeighted = weights = usPlain = wasPlain = 0;
      count = 0;
    }
    Stretch& s = out.back();
    s.hi = p->exponent;
    s.weight += p->weight;
    auto const it = before.find({p->kind, p->exponent});
    const Cost* const was = it == before.end() ? nullptr : it->second;
    double const wasUs = was ? was->us : c.us;
    usWeighted += p->weight * c.us;
    wasWeighted += p->weight * wasUs;
    weights += p->weight;
    usPlain += c.us;
    wasPlain += wasUs;
    ++count;
    if (!was || was->entry != c.entry || std::abs(was->us - c.us) > 1e-9) { s.moved = true; }
    if (!was || !was->measured()) { s.wasMeasured = false; }
  }
  close();

  // The grid only samples the workload, so two stretches meet somewhere between their points: where the first's entry
  // stops serving, or the second's starts, and where neither says, just before the second's first point.
  std::map<std::string, const SelectionEntry*> byId;
  for (const SelectionEntry& e : entries) { byId.emplace(e.id, &e); }
  auto intervalOf = [&](const Stretch& s) -> const SelectionEntry* {
    auto const at = s.measured ? byId.find(s.entry) : byId.end();
    return at == byId.end() ? nullptr : at->second;
  };
  for (size_t i = 1; i < out.size(); ++i) {
    Stretch& prev = out[i - 1];
    Stretch& cur = out[i];
    if (prev.kind != cur.kind || prev.hi + 1 >= cur.lo) { continue; }
    u64 boundary = cur.lo - 1;
    if (const SelectionEntry* e = intervalOf(prev); e && e->reach >= prev.hi && e->reach < cur.lo) {
      boundary = e->reach;
    } else if (const SelectionEntry* f = intervalOf(cur); f && f->emin > prev.hi && f->emin <= cur.lo) {
      boundary = f->emin - 1;
    }
    prev.hi = boundary;
    cur.lo = boundary + 1;
  }
  return out;
}

std::string phaseOf(const std::vector<Item>& ranked, bool anythingWorthRunning) {
  if (std::ranges::any_of(ranked, [](const Item& i) { return i.kind == ItemKind::Bootstrap; })) {
    return "bootstrap: racing each FFT type's options";
  }
  if (std::ranges::any_of(ranked, [](const Item& i) { return i.kind == ItemKind::Gate; })) {
    return "accuracy gate: reading what the table owes";
  }
  if (std::ranges::any_of(ranked, [](const Item& i) { return i.cover; })) { return "covering the workload"; }
  return anythingWorthRunning ? "searching" : "nothing left worth running";
}

std::vector<Line> frame(const Board& b, size_t rows, size_t cols) {
  size_t const width = cols > 1 ? cols - 1 : 0;
  const Glyphs& g = glyphs(b);
  std::vector<Line> out;
  if (!rows || !width) { return out; }

  out.push_back(titleLine(b, width));
  out.push_back(fit(statusLine(b), width, g.ellipsis));
  out.emplace_back();
  addFaults(out, b, width);
  out.push_back(nowLine(b, width));
  out.emplace_back();

  // The charts are the first thing given up on a short screen, then their height.
  size_t const below = 12;
  size_t chartHeight = rows >= TALL_CHARTS_FROM ? 3 : 1;
  auto benefitRows = [&](size_t h) { return 2 + (h ? 2 * (h + 2) + 1 : 0) + 1; };
  while (chartHeight && out.size() + benefitRows(chartHeight) + below > rows) { --chartHeight; }
  if (chartHeight == 2) { chartHeight = 1; }
  addBenefit(out, b, width, chartHeight);
  out.emplace_back();

  if (out.size() < rows) {
    size_t const left = rows - out.size();
    if (width >= TWO_COLUMNS_FROM) {
      std::string const bar = " " + std::string{g.bar} + " ";
      size_t const lw = (width - columns(bar)) / 2;
      size_t const rw = width - columns(bar) - lw;
      std::vector<Line> const l = stack(b, {productionPanel, samplesPanel, recentPanel}, lw, left);
      std::vector<Line> const r = stack(b, {nextPanel, eventsPanel}, rw, left);
      for (size_t i = 0; i < left; ++i) {
        Line line = padded(i < l.size() ? l[i] : Line{}, lw);
        line.push_back({bar, Style::Dim});
        if (i < r.size()) { line.insert(line.end(), r[i].begin(), r[i].end()); }
        out.push_back(fit(line, width, g.ellipsis));
      }
    } else {
      std::vector<Line> const all =
        stack(b, {productionPanel, samplesPanel, recentPanel, nextPanel, eventsPanel}, width, left);
      out.insert(out.end(), all.begin(), all.end());
    }
  }

  out.resize(std::min(out.size(), rows));
  for (Line& line : out) {
    // Trailing blanks cost bytes on every repaint and draw nothing.
    while (!line.empty() && line.back().text.find_first_not_of(' ') == std::string::npos) { line.pop_back(); }
  }
  return out;
}

std::string paint(const std::vector<Line>& lines, bool color) {
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
  for (size_t i = 0; i < lines.size(); ++i) {
    for (const Span& s : lines[i]) {
      std::string_view const on = color ? sgr(s.style) : "";
      out += on;
      out += s.text;
      if (!on.empty()) { out += "\x1b[0m"; }
    }
    out += "\x1b[K";
    // A newline in the bottom row would scroll the screen.
    if (i + 1 < lines.size()) { out += "\r\n"; }
  }
  return out + "\x1b[J";
}

std::optional<std::string> eventOf(std::string_view line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.remove_suffix(1); }
  std::string time;
  // "YYYYMMDD HH:MM:SS <context> text", as log() prefixes every line; a line after the first of one log() is bare.
  auto const digits = [&](size_t from, size_t n) {
    return line.size() >= from + n &&
      std::all_of(line.begin() + from, line.begin() + from + n, [](char c) { return c >= '0' && c <= '9'; });
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
      std::all_of(rest.begin(), rest.begin() + dot, [](char c) { return c >= '0' && c <= '9'; });
    if (item || rest.starts_with("progress: ") || rest.starts_with("still measuring ")) { return std::nullopt; }
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

[[nodiscard]] std::pair<size_t, size_t> terminalSize() {
#ifndef _WIN32
  winsize ws{};
  if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) && ws.ws_col && ws.ws_row) { return {ws.ws_row, ws.ws_col}; }
#endif
  return {24, 80};
}

[[nodiscard]] std::string readText(const fs::path& path) {
  File file = File::openRead(path);
  return file ? file.readAll() : std::string{};
}

}  // namespace

DashboardView::DashboardView(Setup setup) :
  setup_{std::move(setup)}, kind_{probeKind(setup_.runScope)}, color_{colorTerminal()}, start_{Clock::now()} {
  std::string const text = readText(setup_.db);
  std::vector<SessionSpan> const spans = spansOf(text, setup_.env);

  board_.device = setup_.device;
  board_.scope = setup_.scope;
  board_.generation = setup_.generation;
  board_.before = measuringTime(spans, setup_.session);
  board_.sessions = u32(std::max<size_t>(1, spans.size()));
  board_.benefit.kind = kind_;
  board_.benefit.probe = setup_.runScope.probe;
  board_.historyPending = true;
  board_.unicode = utf8Terminal();

  takeScreen();
  setStdoutSink([this](std::string_view s) { take(s); });
  restart::beforeExec([this] { close(); });
  ticker_ = std::thread{[this] { tick(); }};

  auto const until = Clock::now() + std::chrono::duration<double>(HISTORY_BUDGET_SEC);
  std::vector<HistoryPoint> history =
    replay(text, setup_.env, setup_.runScope, HISTORY_POINTS, [&] { return Clock::now() < until; });
  // The session this process opened is in the file already, but nothing of it has been measured: its own points come
  // from the queue.
  std::erase_if(history, [&](const HistoryPoint& p) { return p.active > board_.before + 1; });

  std::unique_lock const lock{mutex_};
  history.insert(history.end(), board_.history.begin(), board_.history.end());
  board_.history = std::move(history);
  board_.historyPending = false;
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

void DashboardView::progress(const RunProgress& p) {
  std::unique_lock const lock{mutex_};
  board_.progress = p;
  board_.started = true;
  board_.what.clear();
}

void DashboardView::state(const QueueState& q) {
  Untuned const untuned{q.db, q.env, setup_.card};
  std::vector<Stretch> production =
    stretchesOf(q.objective.points(), untunedPoints(q.objective.points(), untuned), q.objective.entries());
  std::vector<Sample> samples;
  for (u64 const E : sampleExponents(q.objective.points(), kind_, setup_.runScope.probe, SAMPLES_SHOWN)) {
    samples.push_back({.kind = kind_,
                       .exponent = E,
                       .probe = E == setup_.runScope.probe,
                       .untuned = untuned.at(kind_, E),
                       .now = q.objective.cStar(kind_, E)});
  }
  Benefit benefit = benefitOf(q.db, q.env, setup_.card, q.objective, kind_, setup_.runScope.probe);
  std::vector<Fault> faults = faultsOf(q.db, q.env);

  std::vector<NextUp> next;
  bool worth = false;
  for (const Item& item : q.ranked) {
    if (!worthRunning(item, q.floor)) { continue; }
    worth = true;
    if (next.size() == NEXT_KEPT) { break; }
    next.push_back({.kind = item.kind,
                    .label = itemLabel(q.scheduler, item),
                    .byRule = byRule(item),
                    .value = item.value,
                    .seconds = item.seconds});
  }
  std::string phase = phaseOf(q.ranked, worth);

  std::optional<Cost> const probe = q.objective.cStar(kind_, setup_.runScope.probe);
  double const elapsed = std::chrono::duration<double>(Clock::now() - start_).count();

  std::unique_lock const lock{mutex_};
  board_.production = std::move(production);
  board_.samples = std::move(samples);
  board_.benefit = std::move(benefit);
  board_.faults = std::move(faults);
  board_.next = std::move(next);
  board_.phase = std::move(phase);
  board_.history.push_back({.active = board_.before + elapsed,
                            .T = q.objective.T(),
                            .measured = q.objective.measured(),
                            .probe = probe && probe->measured() ? std::optional<double>{probe->us} : std::nullopt});
}

void DashboardView::finished(const Finished& f) {
  std::string const at = localClock("%H:%M:%S", std::time(nullptr));
  std::unique_lock const lock{mutex_};
  board_.recent.push_front({.item = f, .at = at});
  if (board_.recent.size() > RECENT_KEPT) { board_.recent.pop_back(); }
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
