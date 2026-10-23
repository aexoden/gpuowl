// Copyright (C) Jason Lynch

#include "Explore.h"

#include "Branches.h"
#include "Coverage.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace tune {

namespace {

// How many tries a step makes for a configuration that is neither read nor blocked before it gives up.
constexpr u32 TRIES = 24;

// One entry per subject: the first of its candidates.
std::vector<const Candidate*> entriesOf(const AcquireContext& context) {
  std::vector<const Candidate*> out;
  std::set<ConfigKey> seen;
  for (const std::vector<Candidate>& kind : context.candidates) {
    for (const Candidate& c : kind) {
      if (seen.insert(Readings::keyOf(c.subject, {})).second) { out.push_back(&c); }
    }
  }
  return out;
}

const Reading* fastest(const AcquireContext& context, const Candidate& c) {
  const Reading* best = nullptr;
  for (const Reading* r : context.readings.of(c.subject)) {
    if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
  }
  return best;
}

// How much dearer than what else serves its timing exponent an entry is, in log units: measured where it has been,
// else as the posterior mean expects of its defaults.
std::vector<double> behind(const AcquireContext& context, const std::vector<const Candidate*>& entries) {
  std::vector<double> value(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    const Reading* const best = fastest(context, *entries[i]);
    value[i] = best ? std::log(best->mean) : context.model.predict(context.model.mean(), entries[i]->subject, {});
  }
  std::vector<double> out(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    double frontier = value[i];
    for (size_t j = 0; j < entries.size(); ++j) {
      if (entries[j]->subject.kind == entries[i]->subject.kind && entries[j]->interval.contains(entries[i]->timing)) {
        frontier = std::min(frontier, value[j]);
      }
    }
    out[i] = value[i] - frontier;
  }
  return out;
}

// One of `indices`: at random with probability RANDOM_SHARE, else the least behind.
std::pair<size_t, bool> pick(const std::vector<size_t>& indices, const std::vector<double>& behindBy,
                             std::mt19937_64& rng) {
  if (std::uniform_real_distribution<double>{0, 1}(rng) < RANDOM_SHARE) {
    return {indices[std::uniform_int_distribution<size_t>{0, indices.size() - 1}(rng)], true};
  }
  return {*std::ranges::min_element(indices, {}, [&](size_t i) { return behindBy[i]; }), false};
}

// Every variable after `from` made legal again where its ancestors moved: kept where it still is, else at its
// reference, or inactive.
void settle(const ConditionalSpace& space, std::vector<size_t>& states, size_t from) {
  const std::vector<BranchVariable>& vars = space.variables();
  for (size_t u = from + 1; u < vars.size(); ++u) {
    const BranchVariable::Context& c = vars[u].context(space.contextOf(u, states));
    if (!c.reference) {
      states[u] = vars[u].inactive();
    } else if (states[u] == vars[u].inactive() || !c.offered[states[u]]) {
      states[u] = *c.reference;
    }
  }
}

// Moves within an entry's space that keep every variable legal.
class Mutator {
public:
  Mutator(const ConditionalSpace& space, std::mt19937_64& rng) : space_{space}, vars_{space.variables()}, rng_{rng} {}

  // A legal state of `v` where its ancestors are: inactive, or one of its offered positions at random.
  size_t anyLegal(size_t v, const std::vector<size_t>& states) const {
    const BranchVariable::Context& c = vars_[v].context(space_.contextOf(v, states));
    if (!c.reference) { return vars_[v].inactive(); }
    std::vector<size_t> offered;
    for (size_t p = 0; p < c.offered.size(); ++p) {
      if (c.offered[p]) { offered.push_back(p); }
    }
    return offered[std::uniform_int_distribution<size_t>{0, offered.size() - 1}(rng_)];
  }

  // Moves `v` to another legal position, and what it opens at random where `dependents`, else to their references;
  // false where it has nowhere else to go.
  bool move(std::vector<size_t>& states, size_t v, bool dependents) const {
    const BranchVariable::Context& c = vars_[v].context(space_.contextOf(v, states));
    if (!c.reference) { return false; }
    std::vector<size_t> others;
    for (size_t p = 0; p < c.offered.size(); ++p) {
      if (c.offered[p] && p != states[v]) { others.push_back(p); }
    }
    if (others.empty()) { return false; }
    states[v] = others[std::uniform_int_distribution<size_t>{0, others.size() - 1}(rng_)];
    for (size_t u = v + 1; u < vars_.size(); ++u) {
      const BranchVariable::Context& cu = vars_[u].context(space_.contextOf(u, states));
      if (!cu.reference) {
        states[u] = vars_[u].inactive();
      } else if (dependents && std::ranges::find(vars_[u].ancestors, v) != vars_[u].ancestors.end()) {
        states[u] = anyLegal(u, states);
      } else if (states[u] == vars_[u].inactive() || !cu.offered[states[u]]) {
        states[u] = *cu.reference;
      }
    }
    return true;
  }

  // A variable that can move, a structural one with probability `structural` where there is one.
  std::optional<size_t> someVariable(const std::vector<size_t>& states, double structural) const {
    std::vector<size_t> plain;
    std::vector<size_t> branching;
    for (size_t v = 0; v < vars_.size(); ++v) {
      const BranchVariable::Context& c = vars_[v].context(space_.contextOf(v, states));
      if (!c.reference || std::ranges::count(c.offered, true) < 2) { continue; }
      (vars_[v].axis.option->structural ? branching : plain).push_back(v);
    }
    bool const useBranching =
      !branching.empty() && (plain.empty() || std::uniform_real_distribution<double>{0, 1}(rng_) < structural);
    const std::vector<size_t>& from = useBranching ? branching : plain;
    if (from.empty()) { return {}; }
    return from[std::uniform_int_distribution<size_t>{0, from.size() - 1}(rng_)];
  }

private:
  const ConditionalSpace& space_;
  const std::vector<BranchVariable>& vars_;
  std::mt19937_64& rng_;
};

// The chance a mutation moves a structural variable, with what it opens.
constexpr double STRUCTURAL_MOVE = 0.25;

bool fresh(const AcquireContext& context, const Candidate& c, const UseConfig& options) {
  return !context.readings.find(c.subject, options) && !context.blocked(c, options);
}

double predictedUs(const AcquireContext& context, const Candidate& c, const UseConfig& options) {
  return std::exp(context.model.predict(context.model.mean(), c.subject, options));
}

// The best set read on another entry like `c` -- of its shape, else of its type -- fitted to it.
std::optional<UseConfig> transferred(const AcquireContext& context, const Candidate& c, std::mt19937_64& rng) {
  const Env& env = context.model.env();
  std::vector<const Reading*> shape;
  std::vector<const Reading*> type;
  std::map<ConfigKey, const Reading*> best;
  for (const auto& [key, r] : context.readings.all()) {
    if (r.failed || r.calls == 0 || r.kind != c.subject.kind || r.fft.spec() == c.subject.fft.spec()) { continue; }
    if (r.fft.shape.fft_type != c.subject.fft.shape.fft_type) { continue; }
    ConfigKey const entry{.fft = key.fft, .kind = key.kind, .regime = key.regime, .options = {}};
    if (auto& b = best[entry]; !b || r.mean < b->mean) { b = &r; }
  }
  for (const auto& [entry, r] : best) {
    (r->fft.shape.spec() == c.subject.fft.shape.spec() ? shape : type).push_back(r);
  }
  const std::vector<const Reading*>& from = shape.empty() ? type : shape;
  if (from.empty()) { return {}; }
  const Reading* const r = from[std::uniform_int_distribution<size_t>{0, from.size() - 1}(rng)];
  return fitted(env, c.subject.fft, r->options);
}

// How much likelier a branch's move is to read a variable the branch opens than one every branch has.
constexpr double OPENED_WEIGHT = 3;

// The next set of a contender's least developed branch.
std::optional<Proposal> branchStep(const AcquireContext& context, const std::vector<const Candidate*>& entries,
                                   const std::vector<double>& behindBy, double contenders, std::mt19937_64& rng) {
  const Env& env = context.model.env();
  struct Open {
    size_t entry;
    BranchArchive archive;
    double behind;  // the branch's fastest set, against what else serves the entry's timing exponent
  };
  std::vector<Open> open;
  for (size_t i = 0; i < entries.size(); ++i) {
    if (!(behindBy[i] <= std::log1p(contenders))) { continue; }
    const Reading* const best = fastest(context, *entries[i]);
    if (!best) { continue; }
    for (BranchArchive& a : archiveOf(env, context.readings.of(entries[i]->subject))) {
      double const behind = behindBy[i] + std::log(a.seeds.front()->mean) - std::log(best->mean);
      open.push_back({.entry = i, .archive = std::move(a), .behind = behind});
    }
  }
  if (open.empty()) { return {}; }

  size_t fewest = SIZE_MAX;
  for (const Open& o : open) { fewest = std::min(fewest, o.archive.configurations); }
  std::vector<size_t> least;
  std::vector<double> behind;
  for (size_t k = 0; k < open.size(); ++k) {
    behind.push_back(open[k].behind);
    if (open[k].archive.configurations == fewest) { least.push_back(k); }
  }
  auto const [k, random] = pick(least, behind, rng);
  const Open& o = open[k];
  const Candidate& c = *entries[o.entry];
  std::shared_ptr<const ConditionalSpace> const space = spaceOf(env, c.subject.fft, context.filter(c));
  const std::vector<BranchVariable>& vars = space->variables();
  std::vector<size_t> const defaults = space->statesOf({});

  // What the branch has read: each variable's position in its context there, failures included.
  std::set<std::tuple<size_t, size_t, size_t>> read;
  for (const Reading* r : context.readings.of(c.subject)) {
    if (branchOf(env, c.subject.fft, r->options) != o.archive.branch) { continue; }
    std::vector<size_t> const states = space->statesOf(r->options);
    for (size_t v = 0; v < vars.size(); ++v) { read.emplace(v, space->contextOf(v, states), states[v]); }
  }

  auto propose = [&](const UseConfig& options, const char* how) -> std::optional<Proposal> {
    if (branchOf(env, c.subject.fft, options) != o.archive.branch || !fresh(context, c, options)) { return {}; }
    UseConfig const shown = canonicalConfig(env, c.subject.fft, o.archive.branch);
    return Proposal{.candidate = &c,
                    .options = options,
                    .lane = Lane::Coverage,
                    .what = "branch",
                    .packet = 0,
                    .ordinal = u32(o.archive.configurations) + 1,
                    .predictedUs = predictedUs(context, c, options),
                    .why = std::string{"coverage: "} + how + " in branch " +
                      (shown.empty() ? std::string{"of the defaults"} : configText(shown)) + ", set " +
                      std::to_string(o.archive.configurations + 1) + (random ? ", drawn at random" : "")};
  };

  if (std::optional<Proposal> p = propose(fitted(env, c.subject.fft, o.archive.branch), "the defaults")) { return p; }

  std::vector<size_t> const seed = space->statesOf(o.archive.seeds.front()->options);
  struct Move {
    size_t v;
    size_t p;
  };
  std::vector<Move> moves;
  std::vector<double> weights;
  for (size_t v = 0; v < vars.size(); ++v) {
    if (vars[v].axis.option->structural || seed[v] == vars[v].inactive()) { continue; }
    size_t const context = space->contextOf(v, seed);
    const BranchVariable::Context& ctx = vars[v].context(context);
    for (size_t p = 0; p < ctx.offered.size(); ++p) {
      if (!ctx.offered[p] || p == seed[v] || read.contains({v, context, p})) { continue; }
      moves.push_back({.v = v, .p = p});
      weights.push_back(defaults[v] == vars[v].inactive() ? OPENED_WEIGHT : 1);
    }
  }
  for (u32 attempt = 0; attempt < TRIES && !moves.empty(); ++attempt) {
    size_t const m = std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng);
    std::vector<size_t> states = seed;
    states[moves[m].v] = moves[m].p;
    settle(*space, states, moves[m].v);
    if (std::optional<Proposal> p = propose(space->configOf(states), "a position unread")) { return p; }
    weights[m] = 0;
    if (std::ranges::all_of(weights, [](double w) { return w == 0; })) { break; }
  }
  return {};
}

}  // namespace

std::optional<Proposal> coverageStep(const AcquireContext& context, double contenders, std::mt19937_64& rng) {
  std::vector<const Candidate*> const entries = entriesOf(context);
  if (entries.empty()) { return {}; }
  std::vector<double> const behindBy = behind(context, entries);

  // Half to the contenders' branches, half to breadth; either lends its turn to the other.
  bool const branches = std::bernoulli_distribution{0.5}(rng);
  if (branches) {
    if (std::optional<Proposal> p = branchStep(context, entries, behindBy, contenders, rng)) { return p; }
  }

  // An entry's level is how many sets of it have been tried, failures included: rounds go to the least tried.
  std::vector<size_t> unlooked;
  std::vector<size_t> level(entries.size(), 0);
  size_t lowest = SIZE_MAX;
  for (size_t i = 0; i < entries.size(); ++i) {
    level[i] = context.readings.of(entries[i]->subject).size();
    if (level[i] == 0) {
      if (!context.blocked(*entries[i], {})) { unlooked.push_back(i); }
    } else {
      lowest = std::min(lowest, level[i]);
    }
  }
  std::vector<size_t> developing;
  for (size_t i = 0; i < entries.size(); ++i) {
    if (level[i] > 0 && level[i] == lowest) { developing.push_back(i); }
  }
  if (unlooked.empty() && developing.empty()) {
    return branches ? std::nullopt : branchStep(context, entries, behindBy, contenders, rng);
  }

  bool const look = developing.empty() || (!unlooked.empty() && std::bernoulli_distribution{0.5}(rng));
  if (look) {
    auto const [i, random] = pick(unlooked, behindBy, rng);
    const Candidate& c = *entries[i];
    return Proposal{.candidate = &c,
                    .options = {},
                    .lane = Lane::Coverage,
                    .what = "first-look",
                    .packet = 0,
                    .ordinal = 1,
                    .predictedUs = predictedUs(context, c, {}),
                    .why = std::string{"coverage: first look, "} +
                      (random ? "drawn at random" : "least behind of " + std::to_string(unlooked.size())) + " unread"};
  }

  auto const [i, random] = pick(developing, behindBy, rng);
  const Candidate& c = *entries[i];
  const Reading* const best = fastest(context, c);
  UseConfig const base = best ? best->options : UseConfig{};
  std::shared_ptr<const ConditionalSpace> const space = spaceOf(context.model.env(), c.subject.fft, context.filter(c));
  Mutator const mutator{*space, rng};
  u32 const ordinal = u32(level[i]) + 1;
  // A first packet: a set read on a like entry, a structural move with what it opens at its defaults, mutations.
  static constexpr const char* KINDS[] = {"transfer", "structural", "mutation", "mutation"};
  for (u32 attempt = 0; attempt < TRIES; ++attempt) {
    std::string kind = KINDS[(level[i] - 1 + attempt) % std::size(KINDS)];
    std::optional<UseConfig> options;
    if (kind == "transfer") {
      options = transferred(context, c, rng);
    } else {
      std::vector<size_t> states = space->statesOf(base);
      std::optional<size_t> const v = mutator.someVariable(states, kind == "structural" ? 1.0 : STRUCTURAL_MOVE);
      if (v && mutator.move(states, *v, false)) { options = space->configOf(states); }
    }
    if (options && fresh(context, c, *options)) {
      return Proposal{.candidate = &c,
                      .options = *options,
                      .lane = Lane::Coverage,
                      .what = kind,
                      .packet = 0,
                      .ordinal = ordinal,
                      .predictedUs = predictedUs(context, c, *options),
                      .why = "coverage: " + kind + ", set " + std::to_string(ordinal) + " of the entry" +
                        (random ? ", drawn at random" : "")};
    }
  }
  return {};
}

std::optional<Proposal> restartStep(const AcquireContext& context, const TuneDB& db, u32 env, std::mt19937_64& rng) {
  std::vector<const Candidate*> const entries = entriesOf(context);
  if (entries.empty()) { return {}; }
  const Env& card = context.model.env();

  // The open restart: the last one, while it has steps left and its start was read.
  u32 top = 0;
  std::map<u32, std::vector<const ItemRow*>> restarts;
  for (const ItemRow& r : db.items()) {
    if (db.envOf(r.sess) != env) { continue; }
    top = std::max(top, r.packet);
    if (r.lane == toString(Lane::Restarts) && r.packet) { restarts[r.packet].push_back(&r); }
  }
  if (!restarts.empty()) {
    auto const& [packet, steps] = *restarts.rbegin();
    const Candidate* candidate = nullptr;
    const Reading* best = nullptr;
    for (const Candidate* c : entries) {
      if (c->subject.fft.spec() != steps.front()->fft || c->subject.kind != steps.front()->kind) { continue; }
      for (const ItemRow* s : steps) {
        const UseConfig* const options = db.findCfg(s->cfg);
        const Reading* const r = options ? context.readings.find(c->subject, *options) : nullptr;
        if (r && !r->failed && r->calls > 0 && (!best || r->mean < best->mean)) {
          best = r;
          candidate = c;
        }
      }
      if (candidate) { break; }
    }
    // Where among its steps the best was read: a restart still improving goes on.
    size_t bestAt = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
      const UseConfig* const options = db.findCfg(steps[i]->cfg);
      if (best && options && *options == best->options) { bestAt = i + 1; }
    }
    bool const improving = bestAt + 2 > steps.size();
    if (best && (steps.size() < RESTART_PACKET || (improving && steps.size() < RESTART_MOST))) {
      std::shared_ptr<const ConditionalSpace> const space =
        spaceOf(card, candidate->subject.fft, context.filter(*candidate));
      Mutator const mutator{*space, rng};
      u32 const ordinal = u32(steps.size()) + 1;
      for (u32 attempt = 0; attempt < TRIES; ++attempt) {
        std::vector<size_t> states = space->statesOf(best->options);
        std::optional<size_t> const v = mutator.someVariable(states, STRUCTURAL_MOVE);
        if (!v || !mutator.move(states, *v, true)) { continue; }
        // Halfway, a second move with the first: what moves together may pay where neither does alone.
        if (ordinal == RESTART_PACKET / 2 + 1) {
          if (std::optional<size_t> const w = mutator.someVariable(states, 0); w && *w != *v) {
            (void)mutator.move(states, *w, false);
          }
        }
        UseConfig const options = space->configOf(states);
        if (!fresh(context, *candidate, options)) { continue; }
        return Proposal{.candidate = candidate,
                        .options = options,
                        .lane = Lane::Restarts,
                        .what = "local",
                        .packet = packet,
                        .ordinal = ordinal,
                        .predictedUs = predictedUs(context, *candidate, options),
                        .why = "restart " + std::to_string(packet) + ", step " + std::to_string(ordinal) +
                          (ordinal > RESTART_PACKET ? " (still improving)" : " of " + std::to_string(RESTART_PACKET)) +
                          " from its own best (" + std::to_string(best->mean).substr(0, 6) + " us/it)"};
      }
    }
  }

  // A new start: an entry drawn uniformly, or uniformly within a (kind, type, shape) drawn uniformly; then a complete
  // configuration drawn three ways.
  std::map<std::string, std::vector<const Candidate*>> buckets;
  for (const Candidate* c : entries) {
    buckets[std::string{toString(c->subject.kind)} + '|' + std::to_string(int(c->subject.fft.shape.fft_type)) + '|' +
            c->subject.fft.shape.spec()]
      .push_back(c);
  }
  u32 const packet = top + 1;
  for (u32 attempt = 0; attempt < TRIES; ++attempt) {
    const Candidate* c = nullptr;
    if (std::bernoulli_distribution{0.5}(rng)) {
      c = entries[std::uniform_int_distribution<size_t>{0, entries.size() - 1}(rng)];
    } else {
      auto it = buckets.begin();
      std::advance(it, std::uniform_int_distribution<size_t>{0, buckets.size() - 1}(rng));
      c = it->second[std::uniform_int_distribution<size_t>{0, it->second.size() - 1}(rng)];
    }
    std::shared_ptr<const ConditionalSpace> const space = spaceOf(card, c->subject.fft, context.filter(*c));
    const std::vector<BranchVariable>& vars = space->variables();
    Mutator const mutator{*space, rng};
    std::vector<size_t> const defaults = space->statesOf({});
    std::vector<size_t> states = defaults;

    std::string kind;
    switch (std::uniform_int_distribution<int>{0, 2}(rng)) {
    case 0:
      kind = "random";
      for (size_t v = 0; v < vars.size(); ++v) { states[v] = mutator.anyLegal(v, states); }
      break;
    case 1: kind = "structural"; break;
    default: {
      // Whole option groups from two of the entry's seeds, and a mutation.
      std::vector<const Reading*> seeds;
      for (const BranchArchive& a : archiveOf(card, context.readings.of(c->subject))) {
        seeds.insert(seeds.end(), a.seeds.begin(), a.seeds.end());
      }
      if (seeds.size() < 2) {
        kind = "structural";
        break;
      }
      kind = "recombine";
      std::shuffle(seeds.begin(), seeds.end(), rng);
      std::map<Group, bool> fromFirst;
      UseConfig mixed;
      for (const Reading* parent : {seeds[0], seeds[1]}) {
        for (const auto& [key, value] : parent->options) {
          const Option* const o = findOption(key);
          if (!o) { continue; }
          auto const [it, added] = fromFirst.try_emplace(o->group, std::bernoulli_distribution{0.5}(rng));
          if (it->second == (parent == seeds[0])) { mixed[key] = value; }
        }
      }
      states = space->statesOf(fitted(card, c->subject.fft, mixed));
      if (std::optional<size_t> const v = mutator.someVariable(states, STRUCTURAL_MOVE)) {
        (void)mutator.move(states, *v, true);
      }
    }
    }
    if (kind == "structural") {
      // Every structural variable at random, and what that opens that the defaults do not, at random too.
      for (size_t v = 0; v < vars.size(); ++v) {
        bool const opened = defaults[v] == vars[v].inactive();
        if (vars[v].axis.option->structural || opened) {
          states[v] = mutator.anyLegal(v, states);
        } else {
          const BranchVariable::Context& ctx = vars[v].context(space->contextOf(v, states));
          bool const keep = ctx.reference && states[v] != vars[v].inactive() && ctx.offered[states[v]];
          states[v] = keep ? states[v] : ctx.reference ? *ctx.reference : vars[v].inactive();
        }
      }
    }

    UseConfig const options = space->configOf(states);
    if (!fresh(context, *c, options)) { continue; }
    return Proposal{.candidate = c,
                    .options = options,
                    .lane = Lane::Restarts,
                    .what = kind,
                    .packet = packet,
                    .ordinal = 1,
                    .predictedUs = predictedUs(context, *c, options),
                    .why = "restart " + std::to_string(packet) + ": " + kind + " start"};
  }
  return {};
}

}  // namespace tune
