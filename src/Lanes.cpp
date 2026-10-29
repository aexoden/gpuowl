// Copyright (C) Jason Lynch

#include "Lanes.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <vector>

namespace tune {

namespace {

constexpr const char* LANE_NAMES[LANES] = {"model", "coverage", "restarts", "interactions"};

}  // namespace

const char* toString(Lane lane) { return LANE_NAMES[size_t(lane)]; }

std::optional<Lane> parseLane(std::string_view text) {
  for (size_t i = 0; i < LANES; ++i) {
    if (text == LANE_NAMES[i]) { return Lane(i); }
  }
  return {};
}

LaneShares Timetable::at(double elapsed) const {
  double const f = rampSeconds > 0 ? std::clamp(elapsed / rampSeconds, 0.0, 1.0) : 1.0;
  LaneShares out{};
  for (size_t i = 0; i < LANES; ++i) { out[i] = start[i] + f * (end[i] - start[i]); }
  return out;
}

LaneShares Timetable::owed(double elapsed) const {
  double const ramp = std::min(elapsed, std::max(rampSeconds, 0.0));
  LaneShares const atRamp = at(ramp);
  LaneShares out{};
  for (size_t i = 0; i < LANES; ++i) {
    // The ramp's trapezium, then the end share for whatever came after it.
    out[i] = ramp * (start[i] + atRamp[i]) / 2 + (elapsed - ramp) * end[i];
  }
  return out;
}

std::optional<Timetable> parseTimetable(std::string_view text) {
  if (text == "model") { return Timetable{.start = {1, 0, 0, 0}, .end = {1, 0, 0, 0}, .rampSeconds = 0}; }

  auto split = [](std::string_view s) -> std::optional<LaneShares> {
    LaneShares out{};
    size_t i = 0;
    double sum = 0;
    for (size_t at = 0; at <= s.size(); ++i) {
      size_t const slash = std::min(s.find('/', at), s.size());
      if (i >= LANES) { return {}; }
      double v = 0;
      auto const [end, ec] = std::from_chars(s.data() + at, s.data() + slash, v);
      if (ec != std::errc{} || end != s.data() + slash || v < 0) { return {}; }
      out[i] = v / 100;
      sum += v;
      at = slash + 1;
    }
    if (i != LANES || std::abs(sum - 100) > 1e-9) { return {}; }
    return out;
  };

  Timetable out;
  size_t const tilde = text.find('~');
  if (tilde == std::string_view::npos) {
    std::optional<LaneShares> const one = split(text);
    if (!one) { return {}; }
    return Timetable{.start = *one, .end = *one, .rampSeconds = 0};
  }
  size_t const at = text.find('@', tilde);
  if (at == std::string_view::npos || at + 2 > text.size()) { return {}; }
  std::optional<LaneShares> const start = split(text.substr(0, tilde));
  std::optional<LaneShares> const end = split(text.substr(tilde + 1, at - tilde - 1));
  std::string_view const span = text.substr(at + 1);
  double const unit = span.back() == 'h' ? 3600 : span.back() == 'm' ? 60 : 0;
  double hours = 0;
  auto const [stop, ec] = std::from_chars(span.data(), span.data() + span.size() - 1, hours);
  if (!start || !end || unit == 0 || ec != std::errc{} || stop != span.data() + span.size() - 1 || hours <= 0) {
    return {};
  }
  return Timetable{.start = *start, .end = *end, .rampSeconds = hours * unit};
}

std::string toString(const Timetable& t) {
  auto split = [](const LaneShares& s) {
    std::string out;
    for (size_t i = 0; i < LANES; ++i) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%g", s[i] * 100);
      out += (i ? "/" : "") + std::string{buf};
    }
    return out;
  };
  if (t.start == LaneShares{1, 0, 0, 0} && t.end == t.start) { return "model"; }
  if (t.start == t.end || t.rampSeconds <= 0) { return split(t.end); }
  char buf[32];
  double const minutes = t.rampSeconds / 60;
  if (std::fmod(minutes, 60) == 0) {
    std::snprintf(buf, sizeof(buf), "%gh", minutes / 60);
  } else {
    std::snprintf(buf, sizeof(buf), "%gm", minutes);
  }
  return split(t.start) + '~' + split(t.end) + '@' + buf;
}

std::optional<double> parseDuration(std::string_view text) {
  double seconds = 0;
  bool any = false;
  while (!text.empty()) {
    double n = 0;
    auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (ec != std::errc{} || end == text.data() + text.size() || n < 0) { return {}; }
    char const unit = *end;
    if (unit != 'h' && unit != 'm' && unit != 's') { return {}; }
    seconds += n * (unit == 'h' ? 3600 : unit == 'm' ? 60 : 1);
    any = true;
    text.remove_prefix(size_t(end - text.data()) + 1);
  }
  if (!any) { return {}; }
  return seconds;
}

std::string durationText(double seconds) {
  long const minutes = std::lround(seconds / 60);
  char buf[32];
  if (minutes >= 60 && minutes % 60) {
    std::snprintf(buf, sizeof(buf), "%ldh%ldm", minutes / 60, minutes % 60);
  } else if (minutes >= 60) {
    std::snprintf(buf, sizeof(buf), "%ldh", minutes / 60);
  } else {
    std::snprintf(buf, sizeof(buf), "%ldm", minutes);
  }
  return buf;
}

double Reach::at(double elapsed) const {
  if (!std::isfinite(start)) { return start; }
  return start * std::exp2(std::max(0.0, elapsed) / doubling);
}

std::optional<Reach> parseReach(std::string_view text) {
  if (text == "all") { return Reach{.start = std::numeric_limits<double>::infinity()}; }
  size_t const x = text.find("x@");
  if (x == std::string_view::npos || x + 3 > text.size()) { return {}; }
  double start = 0;
  auto const [end, ec] = std::from_chars(text.data(), text.data() + x, start);
  std::string_view const span = text.substr(x + 2);
  double const unit = span.back() == 'h' ? 3600 : span.back() == 'm' ? 60 : 0;
  double count = 0;
  auto const [stop, ec2] = std::from_chars(span.data(), span.data() + span.size() - 1, count);
  if (ec != std::errc{} || end != text.data() + x || start < 1 || unit == 0 || ec2 != std::errc{} ||
      stop != span.data() + span.size() - 1 || count <= 0) {
    return {};
  }
  return Reach{.start = start, .doubling = count * unit};
}

std::string toString(const Reach& reach) {
  if (!std::isfinite(reach.start)) { return "all"; }
  char buf[48];
  double const minutes = reach.doubling / 60;
  if (std::fmod(minutes, 60) == 0) {
    std::snprintf(buf, sizeof(buf), "%gx@%gh", reach.start, minutes / 60);
  } else {
    std::snprintf(buf, sizeof(buf), "%gx@%gm", reach.start, minutes);
  }
  return buf;
}

double LaneTime::total() const {
  double out = duty;
  for (double const s : seconds) { out += s; }
  return out;
}

LaneTime laneTime(const TuneDB& db, u32 env) {
  // Every session's last record, which ends its last item.
  std::map<u32, u64> last;
  auto seen = [&](u32 sess, u64 ts) {
    if (db.envOf(sess) == env) { last[sess] = std::max(last[sess], ts); }
  };
  for (const CallRow& r : db.calls()) { seen(r.sess, r.ts); }
  for (const RoeRow& r : db.roes()) { seen(r.sess, r.ts); }
  for (const AnchorRow& r : db.anchors()) { seen(r.sess, r.ts); }
  for (const ItemRow& r : db.items()) { seen(r.sess, r.ts); }
  for (const SoakRow& r : db.soaks()) { seen(r.sess, r.ts); }

  LaneTime out;
  std::map<u32, std::vector<const ItemRow*>> bySession;
  for (const ItemRow& r : db.items()) {
    if (db.envOf(r.sess) == env) { bySession[r.sess].push_back(&r); }
  }
  for (const auto& [sess, items] : bySession) {
    for (size_t i = 0; i < items.size(); ++i) {
      u64 const until = i + 1 < items.size() ? items[i + 1]->ts : last[sess];
      double const seconds = std::max(1.0, double(until) - double(items[i]->ts));
      if (std::optional<Lane> const lane = parseLane(items[i]->lane)) {
        out.seconds[size_t(*lane)] += seconds;
      } else {
        out.duty += seconds;
      }
    }
  }
  for (const CallRow& r : db.calls()) {
    if (db.envOf(r.sess) == env && !bySession.contains(r.sess)) {
      out.seconds[size_t(Lane::Model)] += std::max(1.0, r.buildSec + r.timedSec);
    }
  }
  return out;
}

std::string toString(const LaneTime& time) {
  double elapsed = 0;
  for (double const s : time.seconds) { elapsed += s; }
  std::string out;
  for (size_t i = 0; i < LANES; ++i) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s%s %.1f min (%.0f%%)", i ? ", " : "", LANE_NAMES[i], time.seconds[i] / 60,
                  elapsed > 0 ? 100 * time.seconds[i] / elapsed : 0.0);
    out += buf;
  }
  char buf[48];
  std::snprintf(buf, sizeof(buf), "; duty %.1f min", time.duty / 60);
  return out + buf;
}

std::array<Lane, LANES> laneOrder(const Timetable& timetable, const LaneTime& time) {
  double elapsed = 0;
  for (double const s : time.seconds) { elapsed += s; }
  LaneShares const owed = timetable.owed(elapsed);
  std::array<Lane, LANES> out{Lane::Model, Lane::Coverage, Lane::Restarts, Lane::Interactions};
  std::ranges::stable_sort(out, std::greater<>{}, [&](Lane l) {
    size_t const i = size_t(l);
    // A lane with no share is owed nothing, and goes last.
    return timetable.start[i] == 0 && timetable.end[i] == 0 ? -1e300 : owed[i] - time.seconds[i];
  });
  return out;
}

}  // namespace tune
