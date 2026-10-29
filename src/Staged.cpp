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

constexpr std::array<const char*, STAGES> STAGE_NAMES{"types",     "coverage", "sweep",     "transfer",
                                                      "structure", "variants", "positions", "local"};

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

  // The entries the front runs: at each grid point the one whose best is cheapest there, the heaviest first by the
  // weight of the points each holds.
  [[nodiscard]] std::vector<size_t> holders() const {
    std::map<size_t, double> weight;
    for (const Grid& g : context_.scope.grids) {
      for (const GridPoint& p : g.points) {
        const Entry* best = nullptr;
        for (const Entry& e : entries_) {
          if (e.best && e.c->subject.kind == g.kind && e.c->interval.contains(p.exponent) &&
              (!best || e.best->mean < best->best->mean)) {
            best = &e;
          }
        }
        if (best) { weight[size_t(best - entries_.data())] += p.weight; }
      }
    }
    std::vector<size_t> out;
    for (const auto& [i, w] : weight) { out.push_back(i); }
    std::ranges::stable_sort(out, std::greater<>{}, [&](size_t i) { return weight[i]; });
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

// The positions `space`'s entry has read, in their contexts: as every entry `alike` takes has read them, each reading's
// set fitted to the entry.
std::set<std::tuple<size_t, size_t, size_t>> positionsRead(const Field& f, const Entry& e, const ConditionalSpace& space,
                                                           const std::function<bool(const Candidate&)>& alike) {
  const std::vector<BranchVariable>& vars = space.variables();
  std::set<std::tuple<size_t, size_t, size_t>> out;
  for (const Entry& o : f.entries()) {
    if (!o.tried || !alike(*o.c)) { continue; }
    for (const Reading* r : f.context().readings.of(o.c->subject)) {
      std::vector<size_t> const states =
        space.statesOf(o.c == e.c ? r->options : fitted(f.env(), e.c->subject.fft, r->options));
      for (size_t v = 0; v < vars.size(); ++v) { out.emplace(v, space.contextOf(v, states), states[v]); }
    }
  }
  return out;
}

bool sweepStage(const Field& f, const Emit& emit) {
  for (size_t const i : f.holders()) {
    const Entry& e = f.entries()[i];
    std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
    const std::vector<BranchVariable>& vars = space->variables();
    // A position any entry of the shape has read is swept, so a sibling taking the points over does not start over.
    std::set<std::tuple<size_t, size_t, size_t>> const read =
      positionsRead(f, e, *space, [&](const Candidate& c) { return shapeKey(c) == shapeKey(*e.c); });
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
        if (emit(item(Stage::Sweep, e, options, vars[v].axis.label(p) + " from the best set of an entry the front runs"))) {
          return true;
        }
      }
    }
  }
  return false;
}

// A contender: an entry read within the margin, or one nothing has read of a shape read within it, which stands where
// the shape's best does.  The least behind first.
struct Contender {
  size_t entry = 0;
  double behind = 0;
};

std::vector<Contender> contenders(const Field& f, double margin) {
  std::map<std::string, double> shapes;
  for (const Entry& e : f.entries()) {
    if (!e.best) { continue; }
    auto const [it, added] = shapes.emplace(shapeKey(*e.c), e.measuredBehind);
    if (!added) { it->second = std::min(it->second, e.measuredBehind); }
  }
  std::vector<Contender> out;
  for (size_t i = 0; i < f.entries().size(); ++i) {
    const Entry& e = f.entries()[i];
    if (e.best) {
      if (e.measuredBehind <= std::log1p(margin)) { out.push_back({.entry = i, .behind = e.measuredBehind}); }
    } else if (auto const it = shapes.find(shapeKey(*e.c)); !e.tried && it != shapes.end()) {
      if (it->second <= std::log1p(margin)) { out.push_back({.entry = i, .behind = it->second}); }
    }
  }
  std::ranges::stable_sort(out, {}, &Contender::behind);
  return out;
}

// A contender's first look: at its type leader's best set, so that it is compared tuned with what it would replace;
// else at its defaults.
std::optional<Proposal> variantOf(const Field& f, const Contender& k, double margin) {
  const Entry& e = f.entries()[k.entry];
  if (e.tried) { return {}; }
  std::string const why = "the first look at another entry of a shape " + percentBehind(k.behind) + " behind";
  if (const Entry* const leader = f.leaderOf(*e.c, margin)) {
    UseConfig options = fitted(f.env(), e.c->subject.fft, leader->best->options);
    if (f.fresh(*e.c, options)) {
      return item(Stage::Variants, e, std::move(options), why + ", at the best set of its type's leader");
    }
  }
  if (f.fresh(*e.c, {})) { return item(Stage::Variants, e, {}, why); }
  return {};
}

std::optional<Proposal> transferOf(const Field& f, const Contender& k, double margin) {
  const Entry& e = f.entries()[k.entry];
  const Entry* const leader = f.leaderOf(*e.c, margin);
  if (!e.best || !leader || leader == &e) { return {}; }
  UseConfig options = fitted(f.env(), e.c->subject.fft, leader->best->options);
  if (!f.fresh(*e.c, options)) { return {}; }
  return item(Stage::Transfer, e, std::move(options), "the best set of its type's leader, " + leader->c->subject.fft.spec());
}

// The single moves from `seed` of the variables `structural` says, in order, each set as fitted: those `wanted` takes.
std::optional<Proposal> firstMove(const Field& f, const Entry& e, const UseConfig& seedSet, bool structural,
                                  const std::function<bool(size_t v, size_t context, size_t p)>& wanted, Stage stage,
                                  const std::string& from) {
  std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
  const std::vector<BranchVariable>& vars = space->variables();
  std::vector<size_t> const seed = space->statesOf(seedSet);
  for (size_t v = 0; v < vars.size(); ++v) {
    if (vars[v].axis.option->structural != structural || seed[v] == vars[v].inactive()) { continue; }
    size_t const at = space->contextOf(v, seed);
    const BranchVariable::Context& ctx = vars[v].context(at);
    if (!ctx.reference) { continue; }
    for (size_t p = 0; p < ctx.offered.size(); ++p) {
      if (!ctx.offered[p] || p == seed[v] || !wanted(v, at, p)) { continue; }
      std::vector<size_t> states = seed;
      states[v] = p;
      settle(*space, states, v);
      UseConfig options = space->configOf(states);
      if (f.fresh(*e.c, options)) { return item(stage, e, std::move(options), vars[v].axis.label(p) + from); }
    }
  }
  return {};
}

// A position of an option, structural or not as asked, that no entry of the contender's shape -- every variant and regime
// of it -- has read in its context, from the contender's best set: the sweep, on every shape near the front.
std::optional<Proposal> positionsOf(const Field& f, const Contender& k, bool structural) {
  const Entry& e = f.entries()[k.entry];
  if (!e.best) { return {}; }
  std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
  std::set<std::tuple<size_t, size_t, size_t>> const read =
    positionsRead(f, e, *space, [&](const Candidate& c) { return shapeKey(c) == shapeKey(*e.c); });
  auto unread = [&](size_t v, size_t at, size_t p) { return !read.contains({v, at, p}); };
  std::string const from = " from the entry's best set, a position its shape has not read, " + percentBehind(k.behind) +
    " behind";
  return firstMove(f, e, e.best->options, structural, unread, structural ? Stage::Structure : Stage::Positions, from);
}

// A single move from the best set of one of the contender's structural branches within the margin, the branch drawn by
// 1/rank of how far behind it is; nothing where every branch's moves have been read.  Without `rng`, any.
std::optional<Proposal> localOf(const Field& f, const Contender& k, double margin, std::mt19937_64* rng) {
  const Entry& e = f.entries()[k.entry];
  if (!e.best) { return {}; }
  double const frontier = std::log(e.best->mean) - e.measuredBehind;
  std::vector<const Reading*> seeds;
  for (const BranchArchive& a : archiveOf(f.env(), f.context().readings.of(e.c->subject))) {
    if (std::log(a.seeds.front()->mean) - frontier <= std::log1p(margin)) { seeds.push_back(a.seeds.front()); }
  }
  std::shared_ptr<const ConditionalSpace> const space = f.space(*e.c);
  const std::vector<BranchVariable>& vars = space->variables();
  std::vector<double> weights(seeds.size());
  for (size_t b = 0; b < seeds.size(); ++b) { weights[b] = 1.0 / double(b + 1); }
  while (std::ranges::any_of(weights, [](double w) { return w > 0; })) {
    size_t const b = rng ? std::discrete_distribution<size_t>{weights.begin(), weights.end()}(*rng)
                         : size_t(std::ranges::find_if(weights, [](double w) { return w > 0; }) - weights.begin());
    weights[b] = 0;
    std::vector<size_t> const seed = space->statesOf(seeds[b]->options);
    std::vector<std::pair<UseConfig, std::string>> moves;
    for (size_t v = 0; v < vars.size(); ++v) {
      if (vars[v].axis.option->structural || seed[v] == vars[v].inactive()) { continue; }
      const BranchVariable::Context& ctx = vars[v].context(space->contextOf(v, seed));
      for (size_t p = 0; p < ctx.offered.size(); ++p) {
        if (!ctx.offered[p] || p == seed[v]) { continue; }
        std::vector<size_t> states = seed;
        states[v] = p;
        settle(*space, states, v);
        UseConfig options = space->configOf(states);
        if (f.fresh(*e.c, options)) { moves.emplace_back(std::move(options), vars[v].axis.label(p)); }
      }
    }
    if (moves.empty()) { continue; }
    auto& [options, label] = moves[rng ? std::uniform_int_distribution<size_t>{0, moves.size() - 1}(*rng) : 0];
    Proposal p = item(Stage::Local, e, std::move(options),
                      label + " from the best set of branch " + std::to_string(b + 1) + " of the entry, " +
                        percentBehind(std::log(seeds[b]->mean) - frontier) + " behind");
    p.predictedUs = seeds[b]->mean;
    return p;
  }
  return {};
}

// The contenders of one shape, the least behind first, where the shape's best stands.
struct Shape {
  std::vector<Contender> members;
  double behind = 0;
};

std::vector<Shape> shapesOf(const Field& f, const std::vector<Contender>& all) {
  std::map<std::string, size_t> index;
  std::vector<Shape> out;
  for (const Contender& k : all) {
    auto const [it, added] = index.emplace(shapeKey(*f.entries()[k.entry].c), out.size());
    if (added) { out.push_back({.members = {}, .behind = k.behind}); }
    out[it->second].members.push_back(k);
  }
  return out;
}

// The shape's next item.  Its lead, the entry read least behind, takes its type leader's set, then each structural
// position the shape has not read: a branch is tried on every shape near the front before anything finer.  Then the
// first looks at its other entries, their type leader's set on each, every other position the shape has not read from
// the lead, and last single moves from the best of each entry's branches, the lead's first.
std::optional<Proposal> workOf(const Field& f, const Shape& s, double margin, std::mt19937_64* rng) {
  auto const lead = std::ranges::find_if(s.members, [&](const Contender& k) { return f.entries()[k.entry].best; });
  if (lead == s.members.end()) { return {}; }
  if (std::optional<Proposal> p = transferOf(f, *lead, margin)) { return p; }
  if (std::optional<Proposal> p = positionsOf(f, *lead, true)) { return p; }
  for (const Contender& k : s.members) {
    if (std::optional<Proposal> p = variantOf(f, k, margin)) { return p; }
  }
  for (const Contender& k : s.members) {
    if (std::optional<Proposal> p = transferOf(f, k, margin)) { return p; }
  }
  if (std::optional<Proposal> p = positionsOf(f, *lead, false)) { return p; }
  if (std::optional<Proposal> p = localOf(f, *lead, margin, rng)) { return p; }
  for (const Contender& k : s.members) {
    if (std::optional<Proposal> p = localOf(f, k, margin, rng)) { return p; }
  }
  return {};
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
  if (typesStage(f, first) || coverageStage(f, settings.margin, first) || sweepStage(f, first)) {
    return out;
  }

  // A shape drawn by 1/rank of how far behind it is, the least behind first, and an equal floor for each, so that a
  // shape far down the order is still reached; one with nothing left to read is passed over.
  std::vector<Shape> const shapes = shapesOf(f, contenders(f, settings.margin));
  std::vector<double> weights(shapes.size());
  for (size_t k = 0; k < shapes.size(); ++k) { weights[k] = 1.0 / double(k + 1) + 1.0 / double(shapes.size()); }
  while (std::ranges::any_of(weights, [](double w) { return w > 0; })) {
    size_t const k = std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng);
    if (std::optional<Proposal> p = workOf(f, shapes[k], settings.margin, &rng)) {
      p->why += ", shape " + std::to_string(k + 1) + " of " + std::to_string(shapes.size());
      return p;
    }
    weights[k] = 0;
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
  (void)sweepStage(f, counter(Stage::Sweep));
  for (const Shape& shape : shapesOf(f, contenders(f, settings.margin))) {
    if (std::optional<Proposal> const p = workOf(f, shape, settings.margin, nullptr)) {
      ++out[size_t(*parseStage(p->what))];
    }
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
