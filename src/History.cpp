// Copyright (C) Jason Lynch

#include "History.h"

#include "LLCheck.h"
#include "Stats.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <utility>

namespace tune {

namespace {

// The row kinds whose last field is the moment they were written, and whose second is the session that wrote them.
constexpr std::array<std::string_view, 11> STAMPED{"run",   "try", "done", "nogo",  "roe", "anchor",
                                                   "alarm", "ref", "jump", "combo", "boot"};

struct Fields {
  std::string_view kind;
  std::string_view second;
  std::string_view last;
};

[[nodiscard]] Fields fieldsOf(std::string_view line) {
  auto const isSpace = [](char c) { return c == ' ' || c == '\t'; };
  Fields out;
  size_t at = 0;
  std::string_view* slot = &out.kind;
  while (at < line.size()) {
    while (at < line.size() && isSpace(line[at])) { ++at; }
    if (at == line.size()) { break; }
    size_t end = at;
    while (end < line.size() && !isSpace(line[end])) { ++end; }
    std::string_view const field = line.substr(at, end - at);
    if (slot) {
      *slot = field;
      slot = slot == &out.kind ? &out.second : nullptr;
    }
    out.last = field;
    at = end;
  }
  return out;
}

[[nodiscard]] bool stamped(std::string_view kind) { return std::ranges::find(STAMPED, kind) != STAMPED.end(); }

// The value of `key=` among a line's fields.
[[nodiscard]] std::optional<u64> fieldValue(std::string_view line, std::string_view key) {
  for (size_t at = line.find(key); at != std::string_view::npos; at = line.find(key, at + 1)) {
    if (at > 0 && line[at - 1] != ' ' && line[at - 1] != '\t') { continue; }
    std::string_view rest = line.substr(at + key.size());
    return parseInt<u64>(rest.substr(0, rest.find_first_of(" \t")));
  }
  return std::nullopt;
}

// Each line of `text`, without its newline.
template<typename F> void eachLine(std::string_view text, F&& f) {
  for (size_t at = 0; at < text.size();) {
    size_t const nl = text.find('\n', at);
    size_t const end = nl == std::string_view::npos ? text.size() : nl;
    f(text.substr(at, end - at), nl == std::string_view::npos ? end : nl + 1);
    at = nl == std::string_view::npos ? text.size() : nl + 1;
  }
}

}  // namespace

std::string asOf(std::string_view text, u64 ts) {
  std::string out;
  out.reserve(text.size());
  size_t from = 0;
  eachLine(text, [&](std::string_view line, size_t next) {
    Fields const f = fieldsOf(line);
    bool keep = true;
    if (stamped(f.kind) && f.last != f.kind) {
      std::optional<u64> const when = parseInt<u64>(f.last);
      keep = !when || *when <= ts;
    }
    if (keep) { out.append(text.substr(from, next - from)); }
    from = next;
  });
  return out;
}

std::vector<SessionSpan> spansOf(std::string_view text, u32 env) {
  std::map<u32, SessionSpan> spans;
  std::map<u32, u64> latest;
  eachLine(text, [&](std::string_view line, size_t) {
    Fields const f = fieldsOf(line);
    if (f.kind == "sess") {
      std::optional<u32> const id = parseInt<u32>(f.second);
      std::optional<u64> const of = fieldValue(line, "env=");
      std::optional<u64> const start = fieldValue(line, "start=");
      if (id && of && start && *of == env) { spans[*id] = {.sess = *id, .start = *start, .end = *start}; }
    } else if (stamped(f.kind)) {
      std::optional<u32> const sess = parseInt<u32>(f.second);
      std::optional<u64> const when = parseInt<u64>(f.last);
      // A failure row written with no time says nothing about when the session was measuring.
      if (sess && when && *when) { latest[*sess] = std::max(latest[*sess], *when); }
    }
  });

  std::vector<SessionSpan> out;
  for (auto& [id, span] : spans) {
    if (auto const it = latest.find(id); it != latest.end()) { span.end = std::max(span.end, it->second); }
    out.push_back(span);
  }
  std::ranges::sort(out, [](const SessionSpan& a, const SessionSpan& b) {
    return std::pair{a.start, a.sess} < std::pair{b.start, b.sess};
  });
  return out;
}

u64 momentAt(std::span<const SessionSpan> spans, double active) {
  double before = 0;
  for (const SessionSpan& s : spans) {
    if (active <= before + s.seconds()) { return s.start + u64(std::max(0.0, active - before)); }
    before += s.seconds();
  }
  return spans.empty() ? 0 : spans.back().end;
}

double measuringTime(std::span<const SessionSpan> spans, u32 except) {
  double out = 0;
  for (const SessionSpan& s : spans) {
    if (s.sess != except) { out += s.seconds(); }
  }
  return out;
}

std::vector<size_t> spreadOrder(size_t n) {
  std::vector<size_t> out;
  if (n == 0) { return out; }
  out.push_back(0);
  if (n == 1) { return out; }
  out.push_back(n - 1);
  std::deque<std::pair<size_t, size_t>> gaps{{0, n - 1}};
  while (!gaps.empty()) {
    auto const [a, b] = gaps.front();
    gaps.pop_front();
    if (b - a < 2) { continue; }
    size_t const mid = a + (b - a) / 2;
    out.push_back(mid);
    gaps.emplace_back(a, mid);
    gaps.emplace_back(mid, b);
  }
  return out;
}

TestKind probeKind(const RunScope& scope) {
  if (scope.grid(TestKind::PRP)) { return TestKind::PRP; }
  return scope.grids.empty() ? TestKind::PRP : scope.grids.front().kind;
}

std::vector<HistoryPoint> replay(std::string_view text, u32 env, const RunScope& scope, size_t points,
                                 const std::function<bool()>& more) {
  std::vector<SessionSpan> const spans = spansOf(text, env);
  if (spans.empty() || points == 0) { return {}; }
  double const total = measuringTime(spans);
  size_t const n = total > 0 ? points : 1;
  TestKind const kind = probeKind(scope);

  std::vector<std::optional<HistoryPoint>> at(n);
  for (size_t const k : spreadOrder(n)) {
    if (!more()) { break; }
    double const active = n > 1 ? total * double(k) / double(n - 1) : total;
    TuneDB db;
    if (!db.parse(asOf(text, momentAt(spans, active)), "tunedb.txt as it stood")) { continue; }
    Objective const objective{db, env, scope};
    std::optional<Cost> const probe = objective.cStar(kind, scope.probe);
    at[k] = HistoryPoint{.active = active,
                         .T = objective.T(),
                         .measured = objective.measured(),
                         .probe = probe && probe->measured() ? std::optional<double>{probe->us} : std::nullopt};
  }

  std::vector<HistoryPoint> out;
  for (const std::optional<HistoryPoint>& p : at) {
    if (p) { out.push_back(*p); }
  }
  return out;
}

std::optional<double> Benefit::saving() const {
  if (!now || !now->measured() || untuned <= 0) { return std::nullopt; }
  return (untuned - now->us) / untuned;
}

Benefit benefitOf(const TuneDB& db, u32 env, const Env& device, const Objective& objective, TestKind kind, u64 probe) {
  Benefit out;
  out.kind = kind;
  out.probe = probe;
  if (std::optional<Cost> const now = objective.cStar(kind, probe); now && now->measured()) { out.now = now; }

  for (const RunRow& row : db.mergedRuns()) {
    if (row.exponent != probe || row.kind != kind || !row.m.ok() || db.envOf(row.sess) != env) { continue; }
    std::optional<FFTConfig> const fft = parseFft(row.fft);
    const UseConfig* const options = db.findCfg(row.cfg);
    if (!fft || !options || !atBuiltInDefaults(device, *fft, *options)) { continue; }
    // Priced as the selection file prices what it publishes, so that the two are compared like for like.
    if (double const cost = pessimisticCost(row.m); out.untunedFft.empty() || cost < out.untuned) {
      out.untunedFft = row.fft;
      out.untuned = cost;
    }
  }
  return out;
}

std::string benefitText(const Benefit& b) {
  char buf[320];
  if (!b.now) {
    snprintf(buf, sizeof(buf), "at %" PRIu64 " nothing measured is published yet", b.probe);
    return buf;
  }
  int const n = snprintf(buf, sizeof(buf), "at %" PRIu64 " production runs %s at %.3f us/it", b.probe,
                         b.now->fft.c_str(), b.now->us);
  std::string out{buf, size_t(std::max(n, 0))};
  if (std::optional<double> const saving = b.saving()) {
    snprintf(buf, sizeof(buf),
             ", %.1f%% %s per iteration than the best measured there at the built-in defaults (%s, "
             "%.3f us/it)",
             100 * std::abs(*saving), *saving >= 0 ? "less" : "more", b.untunedFft.c_str(), b.untuned);
    out += buf;
  } else {
    out += "; nothing was measured there at the built-in defaults to set it against";
  }
  return out;
}

}  // namespace tune
