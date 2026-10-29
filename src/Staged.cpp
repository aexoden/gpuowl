// Copyright (C) Jason Lynch

#include "Staged.h"

#include "Branches.h"
#include "Coverage.h"
#include "Prior.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace tune {

namespace {

constexpr std::array<const char*, STAGES> STAGE_NAMES{"types",    "coverage",  "sweep", "transfer",
                                                      "variants", "structure", "local"};

// What the stages know of every entry: one per subject, the first of its candidates.
struct Entry {
  const Candidate* c = nullptr;
  const Reading* best = nullptr;  // its fastest successful reading
  bool tried = false;             // anything read of it, failures included
  double expected = 0;            // microseconds per iteration: its best, else the scaled prior
  double behind = 0;              // log units: expected against the least expected serving its timing exponent
  double measuredBehind = 0;      // its best against the best measured serving its timing exponent; infinite unread
};

// "PRP|3" for an entry's kind and FFT type, "PRP|3:1K:13:512" for its kind and shape.
std::string typeKey(const Candidate& c) {
  return std::string{toString(c.subject.kind)} + '|' + std::to_string(int(c.subject.fft.shape.fft_type));
}
std::string shapeKey(const Candidate& c) {
  return std::string{toString(c.subject.kind)} + '|' + std::to_string(int(c.subject.fft.shape.fft_type)) + ':' +
    c.subject.fft.shape.spec();
}

double median(std::vector<double> v) {
  std::ranges::sort(v);
  size_t const n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

class Field {
public:
  explicit Field(const AcquireContext& context) : context_{context} {
    std::set<ConfigKey> seen;
    for (const std::vector<Candidate>& kind : context.candidates) {
      for (const Candidate& c : kind) {
        if (!seen.insert(Readings::keyOf(c.subject, {})).second) { continue; }
        Entry e{.c = &c};
        for (const Reading* r : context.readings.of(c.subject)) {
          e.tried = true;
          if (!r->failed && r->calls > 0 && (!e.best || r->mean < e.best->mean)) { e.best = r; }
        }
        entries_.push_back(e);
      }
    }

    // What the card's readings make of the stated prior: per kind and type, else per kind, the median of the read
    // entries' best over what the prior says of them.
    std::map<std::string, std::vector<double>> byType;
    std::map<TestKind, std::vector<double>> byKind;
    for (const Entry& e : entries_) {
      if (!e.best) { continue; }
      double const ratio = e.best->mean / statedPrior(e.c->subject.fft.shape);
      byType[typeKey(*e.c)].push_back(ratio);
      byKind[e.c->subject.kind].push_back(ratio);
    }
    for (Entry& e : entries_) {
      if (e.best) {
        e.expected = e.best->mean;
        continue;
      }
      double scale = 1;
      if (auto const t = byType.find(typeKey(*e.c)); t != byType.end()) {
        scale = median(t->second);
      } else if (auto const k = byKind.find(e.c->subject.kind); k != byKind.end()) {
        scale = median(k->second);
      }
      e.expected = statedPrior(e.c->subject.fft.shape) * scale;
    }

    double const inf = std::numeric_limits<double>::infinity();
    for (Entry& e : entries_) {
      double frontier = inf;
      double measured = inf;
      for (const Entry& o : entries_) {
        if (o.c->subject.kind != e.c->subject.kind || !o.c->interval.contains(e.c->timing)) { continue; }
        frontier = std::min(frontier, o.expected);
        if (o.best) { measured = std::min(measured, o.best->mean); }
      }
      e.behind = std::log(e.expected) - std::log(frontier);
      e.measuredBehind = e.best ? std::log(e.best->mean) - std::log(measured) : inf;
    }
  }

  [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }

  [[nodiscard]] bool fresh(const Candidate& c, const UseConfig& options) const {
    return !context_.readings.find(c.subject, options) && !context_.blocked(c, options);
  }

  [[nodiscard]] const Env& env() const { return context_.model.env(); }

  [[nodiscard]] std::shared_ptr<const ConditionalSpace> space(const Candidate& c) const {
    return spaceOf(env(), c.subject.fft, context_.filter(c));
  }

  [[nodiscard]] const AcquireContext& context() const { return context_; }

  // The entries' indices, the least behind first by `by`.
  [[nodiscard]] std::vector<size_t> ordered(double Entry::* by) const {
    std::vector<size_t> out(entries_.size());
    std::iota(out.begin(), out.end(), size_t{0});
    std::ranges::stable_sort(out, {}, [&](size_t i) { return entries_[i].*by; });
    return out;
  }

  // The contenders' indices, the least behind first.
  [[nodiscard]] std::vector<size_t> contenders(double margin) const {
    std::vector<size_t> out;
    for (size_t const i : ordered(&Entry::measuredBehind)) {
      if (entries_[i].measuredBehind <= std::log1p(margin)) { out.push_back(i); }
    }
    return out;
  }

  // Each kind and type's leading entry: the read one least behind, where it is within `margin`.
  [[nodiscard]] std::vector<size_t> leaders(double margin) const {
    std::vector<size_t> out;
    std::set<std::string> seen;
    for (size_t const i : contenders(margin)) {
      if (seen.insert(typeKey(*entries_[i].c)).second) { out.push_back(i); }
    }
    return out;
  }

  [[nodiscard]] const Entry* leaderOf(const Candidate& c, double margin) const {
    for (size_t const i : leaders(margin)) {
      if (typeKey(*entries_[i].c) == typeKey(c)) { return &entries_[i]; }
    }
    return nullptr;
  }

private:
  const AcquireContext& context_;
  std::vector<Entry> entries_;
};

// Each stage hands its items in order to `emit`, which says whether to stop.
using Emit = std::function<bool(Proposal)>;

Proposal item(Stage stage, const Entry& e, UseConfig options, std::string why) {
  return Proposal{.candidate = e.c,
                  .options = std::move(options),
                  .lane = Lane::Coverage,
                  .what = toString(stage),
                  .predictedUs = e.expected,
                  .why = std::string{"staged "} + toString(stage) + ": " + why,
                  .record = STAGED};
}

std::string percentBehind(double logBehind) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.1f%%", std::expm1(logBehind) * 100);
  return buf;
}

// The first look at a group's entries: the first of `order` at its defaults that may be read.
bool firstLook(const Field& f, Stage stage, const std::vector<size_t>& order, const std::string& why,
               const Emit& emit) {
  for (size_t const i : order) {
    const Entry& e = f.entries()[i];
    if (!f.fresh(*e.c, {})) { continue; }
    return emit(item(stage, e, {}, why + ", expected " + percentBehind(e.behind) + " behind"));
  }
  return false;
}

// Groups of entries by `key`, each in the order `within` gives, the groups in the order of their first entry's `by`.
std::vector<std::vector<size_t>> groups(const Field& f, const std::function<std::string(const Candidate&)>& key,
                                        double Entry::* by) {
  std::map<std::string, size_t> index;
  std::vector<std::vector<size_t>> out;
  for (size_t const i : f.ordered(by)) {
    auto const [it, added] = index.emplace(key(*f.entries()[i].c), out.size());
    if (added) { out.emplace_back(); }
    out[it->second].push_back(i);
  }
  return out;
}

bool anyTried(const Field& f, const std::vector<size_t>& group) {
  return std::ranges::any_of(group, [&](size_t i) { return f.entries()[i].tried; });
}

bool typesStage(const Field& f, const Emit& emit) {
  for (const std::vector<size_t>& g : groups(f, typeKey, &Entry::behind)) {
    if (anyTried(f, g)) { continue; }
    if (firstLook(f, Stage::Types, g, "the first look at its FFT type", emit)) { return true; }
  }
  return false;
}

bool coverageStage(const Field& f, double margin, const Emit& emit) {
  for (std::vector<size_t> g : groups(f, shapeKey, &Entry::behind)) {
    if (anyTried(f, g) || f.entries()[g.front()].behind > std::log1p(margin)) { continue; }
    // The shape's entry serving the most weight first.
    std::ranges::stable_sort(g, std::greater<>{}, [&](size_t i) { return f.entries()[i].c->weight(); });
    if (firstLook(f, Stage::Coverage, g, "the first look at its shape", emit)) { return true; }
  }
  return false;
}

bool variantsStage(const Field& f, double margin, const Emit& emit) {
  for (const std::vector<size_t>& g : groups(f, shapeKey, &Entry::measuredBehind)) {
    if (!(f.entries()[g.front()].measuredBehind <= std::log1p(margin))) { continue; }
    for (size_t const i : g) {
      const Entry& e = f.entries()[i];
      if (e.tried) { continue; }
      std::string const why =
        "the first look at another entry of a shape " + percentBehind(f.entries()[g.front()].measuredBehind) + " behind";
      // At its type leader's best set, so that it is compared tuned with what it would replace; else at its defaults.
      if (const Entry* const leader = f.leaderOf(*e.c, margin)) {
        UseConfig const options = fitted(f.env(), e.c->subject.fft, leader->best->options);
        if (f.fresh(*e.c, options)) {
          if (emit(item(Stage::Variants, e, options, why + ", with the best set of its type's leader"))) { return true; }
          continue;
        }
      }
      if (f.fresh(*e.c, {}) && emit(item(Stage::Variants, e, {}, why))) { return true; }
    }
  }
  return false;
}

bool sweepStage(const Field& f, double margin, const Emit& emit) {
  for (size_t const i : f.leaders(margin)) {
    const Entry& e = f.entries()[i];
    std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
    const std::vector<BranchVariable>& vars = space->variables();

    // Each option's position in its context, as anything read of the leader has had it.
    std::set<std::tuple<size_t, size_t, size_t>> read;
    for (const Reading* r : f.context().readings.of(e.c->subject)) {
      std::vector<size_t> const states = space->statesOf(r->options);
      for (size_t v = 0; v < vars.size(); ++v) { read.emplace(v, space->contextOf(v, states), states[v]); }
    }
    std::vector<size_t> const seed = space->statesOf(e.best->options);
    for (size_t v = 0; v < vars.size(); ++v) {
      if (vars[v].axis.option->structural || seed[v] == vars[v].inactive()) { continue; }
      size_t const at = space->contextOf(v, seed);
      const BranchVariable::Context& ctx = vars[v].context(at);
      for (size_t p = 0; p < ctx.offered.size(); ++p) {
        if (!ctx.offered[p] || p == seed[v] || read.contains({v, at, p})) { continue; }
        std::vector<size_t> states = seed;
        states[v] = p;
        settle(*space, states, v);
        UseConfig const options = space->configOf(states);
        if (!f.fresh(*e.c, options)) { continue; }
        if (emit(item(Stage::Sweep, e, options, vars[v].axis.label(p) + " from the best set of its type's leader"))) {
          return true;
        }
      }
    }
  }
  return false;
}

bool transferStage(const Field& f, double margin, const Emit& emit) {
  for (size_t const i : f.contenders(margin)) {
    const Entry& e = f.entries()[i];
    const Entry* const leader = f.leaderOf(*e.c, margin);
    if (!leader || leader == &e) { continue; }
    UseConfig const options = fitted(f.env(), e.c->subject.fft, leader->best->options);
    if (!f.fresh(*e.c, options)) { continue; }
    if (emit(item(Stage::Transfer, e, options, "the best set of its type's leader, " + leader->c->subject.fft.spec()))) {
      return true;
    }
  }
  return false;
}

bool structureStage(const Field& f, double margin, const Emit& emit) {
  for (size_t const i : f.contenders(margin)) {
    const Entry& e = f.entries()[i];
    std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
    const std::vector<BranchVariable>& vars = space->variables();
    std::vector<size_t> const seed = space->statesOf(e.best->options);
    for (size_t v = 0; v < vars.size(); ++v) {
      if (!vars[v].axis.option->structural) { continue; }
      const BranchVariable::Context& ctx = vars[v].context(space->contextOf(v, seed));
      if (!ctx.reference) { continue; }
      for (size_t p = 0; p < ctx.offered.size(); ++p) {
        if (!ctx.offered[p] || p == seed[v]) { continue; }
        std::vector<size_t> states = seed;
        states[v] = p;
        settle(*space, states, v);
        UseConfig const options = space->configOf(states);
        if (!f.fresh(*e.c, options)) { continue; }
        if (emit(item(Stage::Structure, e, options,
                      vars[v].axis.label(p) + " from the entry's best set, " + percentBehind(e.measuredBehind) +
                        " behind"))) {
          return true;
        }
      }
    }
  }
  return false;
}

// One contender's structural branch, from its fastest set.
struct Unit {
  size_t entry = 0;
  const Reading* seed = nullptr;
  double behind = 0;
};

std::vector<Unit> units(const Field& f, double margin) {
  std::vector<Unit> out;
  for (size_t const i : f.contenders(margin)) {
    const Entry& e = f.entries()[i];
    double const frontier = std::log(e.best->mean) - e.measuredBehind;
    for (const BranchArchive& a : archiveOf(f.env(), f.context().readings.of(e.c->subject))) {
      double const behind = std::log(a.seeds.front()->mean) - frontier;
      if (behind <= std::log1p(margin)) { out.push_back({.entry = i, .seed = a.seeds.front(), .behind = behind}); }
    }
  }
  std::ranges::stable_sort(out, {}, &Unit::behind);
  return out;
}

// Every single move of an option within the unit's branch, from its fastest set, that nothing has read.
std::vector<std::pair<UseConfig, std::string>> movesOf(const Field& f, const Unit& u) {
  const Entry& e = f.entries()[u.entry];
  std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
  const std::vector<BranchVariable>& vars = space->variables();
  std::vector<size_t> const seed = space->statesOf(u.seed->options);
  std::vector<std::pair<UseConfig, std::string>> out;
  for (size_t v = 0; v < vars.size(); ++v) {
    if (vars[v].axis.option->structural || seed[v] == vars[v].inactive()) { continue; }
    const BranchVariable::Context& ctx = vars[v].context(space->contextOf(v, seed));
    for (size_t p = 0; p < ctx.offered.size(); ++p) {
      if (!ctx.offered[p] || p == seed[v]) { continue; }
      std::vector<size_t> states = seed;
      states[v] = p;
      settle(*space, states, v);
      UseConfig options = space->configOf(states);
      if (f.fresh(*e.c, options)) { out.emplace_back(std::move(options), vars[v].axis.label(p)); }
    }
  }
  return out;
}

Proposal localItem(const Field& f, const Unit& u, UseConfig options, const std::string& move, size_t rank) {
  Proposal p = item(Stage::Local, f.entries()[u.entry], std::move(options),
                    move + " from the best set of a branch " + percentBehind(u.behind) + " behind, ranked " +
                      std::to_string(rank + 1));
  p.predictedUs = u.seed->mean;
  return p;
}

}  // namespace

const char* toString(Stage stage) { return STAGE_NAMES[size_t(stage)]; }

std::optional<Stage> parseStage(std::string_view text) {
  for (size_t i = 0; i < STAGES; ++i) {
    if (text == STAGE_NAMES[i]) { return Stage(i); }
  }
  return {};
}

std::optional<Proposal> stagedStep(const AcquireContext& context, const StagedSettings& settings,
                                   std::mt19937_64& rng) {
  Field const f{context};
  std::optional<Proposal> out;
  Emit const first = [&](Proposal p) {
    out = std::move(p);
    return true;
  };
  if (typesStage(f, first) || coverageStage(f, settings.margin, first) || sweepStage(f, settings.margin, first) ||
      transferStage(f, settings.margin, first) || variantsStage(f, settings.margin, first) ||
      structureStage(f, settings.margin, first)) {
    return out;
  }

  // A branch drawn by 1/rank, the least behind first; one with nothing left unread is passed over.
  std::vector<Unit> const all = units(f, settings.margin);
  std::vector<double> weights(all.size());
  for (size_t k = 0; k < all.size(); ++k) { weights[k] = 1.0 / double(k + 1); }
  while (std::ranges::any_of(weights, [](double w) { return w > 0; })) {
    size_t const k = std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng);
    std::vector<std::pair<UseConfig, std::string>> moves = movesOf(f, all[k]);
    if (moves.empty()) {
      weights[k] = 0;
      continue;
    }
    auto& [options, label] = moves[std::uniform_int_distribution<size_t>{0, moves.size() - 1}(rng)];
    return localItem(f, all[k], std::move(options), label, k);
  }
  return {};
}

StagesLeft stagesLeft(const AcquireContext& context, const StagedSettings& settings) {
  Field const f{context};
  StagesLeft out{};
  auto counter = [&](Stage s) {
    return Emit{[&out, s](const Proposal&) {
      ++out[size_t(s)];
      return false;
    }};
  };
  (void)typesStage(f, counter(Stage::Types));
  (void)coverageStage(f, settings.margin, counter(Stage::Coverage));
  (void)sweepStage(f, settings.margin, counter(Stage::Sweep));
  (void)transferStage(f, settings.margin, counter(Stage::Transfer));
  (void)variantsStage(f, settings.margin, counter(Stage::Variants));
  (void)structureStage(f, settings.margin, counter(Stage::Structure));
  for (const Unit& u : units(f, settings.margin)) {
    if (!movesOf(f, u).empty()) { ++out[size_t(Stage::Local)]; }
  }
  return out;
}

std::string toString(const StagesLeft& left) {
  std::string out;
  for (size_t i = 0; i < STAGES; ++i) {
    out += std::string{i ? ", " : ""} + STAGE_NAMES[i] + ' ' + std::to_string(left[i]);
  }
  return out;
}

}  // namespace tune
