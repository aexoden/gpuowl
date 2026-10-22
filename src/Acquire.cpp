// Copyright (C) Jason Lynch

#include "Acquire.h"

#include "Optimize.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <thread>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tune {

namespace {

constexpr double NONE = std::numeric_limits<double>::infinity();

// How many of the leading candidates at a point an untried step may be taken on.
constexpr size_t UNTRIED_LEADERS = 8;

// How many grid points choose() tries before it gives up, and how many draws it takes for a challenger at each.
constexpr u32 POINT_TRIES = 8;
constexpr u32 CHALLENGER_DRAWS = 4;

struct Valued {
  UseConfig options{};
  double value = NONE;  // log microseconds per iteration
};

// The model's work yields to anything else wanting the CPU: another tuner's timed loop on the same machine busy-waits
// on its device, and a stall there is read as the configuration's cost.
void lowerPriority() {
#ifdef __linux__
  (void)setpriority(PRIO_PROCESS, pid_t(syscall(SYS_gettid)), 19);
#endif
}

// The value `config` gives `key`, as text; empty where it gives none.
std::string useText(const UseConfig& config, const std::string& key) {
  auto const it = config.find(key);
  return it == config.end() ? std::string{} : it->second;
}

// Runs f(0) .. f(n-1) across the machine's threads.
template<typename F> void parallelFor(size_t n, F&& f) {
  size_t const threads = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
  if (threads <= 1) {
    for (size_t i = 0; i < n; ++i) { f(i); }
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::jthread> pool;
  for (size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      lowerPriority();
      for (size_t i; (i = next++) < n;) { f(i); }
    });
  }
}

// A few axes of `options` on `c`, chosen at random by `seed`.
std::vector<std::string> someAxes(const AcquireContext& context, const Candidate& c, const UseConfig& options,
                                  u64 seed) {
  // Weighted toward axes with positions no FFT of the type has been measured at: one reading of a position teaches
  // every FFT of the type what it does.  A position read on other FFTs but not this one counts for as much as its
  // axis's effect was learnt to vary between shapes, or between the variants of one: L2_STRIPING's best stripe
  // depends on the shape, and what it did elsewhere hides what it would do here.
  const Model& model = context.model;
  std::vector<std::pair<std::string, double>> axes;
  for (const Axis& a : axesOf(model.env(), c.subject.fft, options, context.filter(c))) {
    double const otherShape = std::min(1.0, model.shapeDeparture(a.name) / model.effectSpread());
    double const otherVariant = std::min(1.0, model.entryDeparture(a.name) / model.effectSpread());
    double untried = 0;
    for (size_t p = 0; p < a.values.size(); ++p) {
      if (p == a.reference) { continue; }
      if (!model.measured(c.subject, a.label(p))) {
        ++untried;
      } else if (!model.measuredOnShape(c.subject, a.label(p))) {
        untried += otherShape;
      } else if (!model.measuredAt(c.subject, a.label(p))) {
        untried += otherVariant;
      }
    }
    axes.emplace_back(a.name, 1 + UNTRIED_WEIGHT * untried / double(a.values.size()));
  }
  std::mt19937_64 rng{seed};
  // Mostly one axis: a reading that moves one thing says what that thing does, where one that moves several leaves the
  // fit to share its difference among them.
  size_t const m = 1 + std::discrete_distribution<size_t>{FREE_AXES_WEIGHTS.begin(), FREE_AXES_WEIGHTS.end()}(rng);
  std::vector<std::string> names;
  while (names.size() < m && !axes.empty()) {
    std::vector<double> weights;
    for (auto const& [name, w] : axes) { weights.push_back(w); }
    size_t const pick = std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng);
    names.push_back(axes[pick].first);
    axes.erase(axes.begin() + ptrdiff_t(pick));
  }
  return names;
}

// One posterior draw, and what it makes of each candidate, worked out once.
class DrawView {
public:
  DrawView(const AcquireContext& context, Draw draw) : context_{context}, draw_{std::move(draw)} {}

  // Works out the candidates of `candidates` holding `E` that pass the screen, across the machine's threads.
  void valueAt(const std::vector<Candidate>& candidates, u64 E) {
    std::vector<const Candidate*> todo;
    for (const Candidate* c : screen(candidates, E)) {
      if (!cache_.contains(c)) { todo.push_back(c); }
    }
    std::vector<Valued> values(todo.size());
    parallelFor(todo.size(), [&](size_t i) { values[i] = work(*todo[i]); });
    for (size_t i = 0; i < todo.size(); ++i) { cache_.emplace(todo[i], std::move(values[i])); }
  }

  const Valued& valued(const Candidate& c) {
    auto const it = cache_.find(&c);
    if (it != cache_.end()) { return it->second; }
    return cache_.emplace(&c, work(c)).first->second;
  }

  [[nodiscard]] double predict(const Subject& subject, const UseConfig& options) const {
    return context_.model.predict(draw_, subject, options);
  }

  // The candidates at `E` worth optimising in full under this draw.
  std::vector<const Candidate*> screen(const std::vector<Candidate>& candidates, u64 E) const {
    std::vector<const Candidate*> at;
    for (const Candidate& c : candidates) {
      if (c.interval.contains(E)) { at.push_back(&c); }
    }
    if (at.size() <= SCREENED + SCREEN_EXTRA) { return at; }

    auto value = [&](const Candidate* c) {
      const Reading* best = nullptr;
      for (const Reading* r : context_.readings.of(c->subject)) {
        if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
      }
      return context_.model.predict(context_.model.mean(), c->subject, best ? best->options : UseConfig{});
    };
    std::vector<std::pair<double, const Candidate*>> ranked;
    if (context_.means) {
      context_.means->screen(at, value);
      for (const Candidate* c : at) { ranked.emplace_back(context_.means->screenValue(c), c); }
    } else {
      for (const Candidate* c : at) { ranked.emplace_back(value(c), c); }
    }
    std::ranges::sort(ranked, {}, &std::pair<double, const Candidate*>::first);

    std::vector<const Candidate*> out;
    for (size_t i = 0; i < SCREENED; ++i) { out.push_back(ranked[i].second); }
    std::mt19937_64 rng{draw_.seed ^ E};
    for (size_t i = 0; i < SCREEN_EXTRA; ++i) {
      out.push_back(ranked[SCREENED + rng() % (ranked.size() - SCREENED)].second);
    }
    return out;
  }

private:
  Valued work(const Candidate& c) const {
    Valued out;
    auto offer = [&](const UseConfig& options) {
      if (context_.blocked(c, options)) { return; }
      if (double const v = context_.model.predict(draw_, c.subject, options); v < out.value) {
        out = {.options = options, .value = v};
      }
    };

    AxisFilter const filter = context_.filter(c);

    // The best measured set is where both proposals start from; with none, the built-in defaults.
    const Reading* incumbent = nullptr;
    for (const Reading* r : context_.readings.of(c.subject)) {
      if (r->failed || r->calls == 0 || context_.blocked(c, r->options)) { continue; }
      offer(r->options);
      if (!incumbent || r->mean < incumbent->mean) { incumbent = r; }
    }
    UseConfig const around = incumbent ? incumbent->options : UseConfig{};

    // Exploration starts half the time from another structural branch -- an INPLACE value other than the incumbent's,
    // from its best measured set there or else its defaults -- since the keys a branch opens (L2_STRIPING under
    // INPLACE=1, the layout keys under INPLACE=0) are no axes of the other's, and a search around one never reaches
    // them.
    u64 const seed = draw_.seed ^ termHash(c.subject.fft.spec() + toString(c.subject.kind) + c.subject.regime.label());
    UseConfig start = around;
    if (seed % 2) {
      const Env& env = context_.model.env();
      std::string const here = useText(branchOf(env, c.subject.fft, around), "INPLACE");
      for (const Axis& axis : axesOf(env, c.subject.fft, around, filter)) {
        if (axis.option->key != "INPLACE") { continue; }
        std::vector<UseConfig> others;
        for (size_t i = 0; i < axis.values.size(); ++i) {
          if (std::to_string(axis.values[i].first) == here) { continue; }
          const Reading* best = nullptr;
          for (const Reading* r : context_.readings.of(c.subject)) {
            if (r->failed || r->calls == 0 || context_.blocked(c, r->options)) { continue; }
            if (useText(branchOf(env, c.subject.fft, r->options), "INPLACE") != std::to_string(axis.values[i].first)) {
              continue;
            }
            if (!best || r->mean < best->mean) { best = r; }
          }
          UseConfig other;
          place(other, axis, i);
          others.push_back(best ? best->options : fitted(env, c.subject.fft, other));
        }
        if (!others.empty()) { start = others[(seed / 2) % others.size()]; }
      }
    }

    const Optimum* mean = context_.means ? context_.means->find(&c) : nullptr;
    Optimum computed;
    if (!mean) {
      computed = meanStep(c, around, filter);
      if (context_.means) { context_.means->put(&c, computed); }
      mean = &computed;
    }
    offer(mean->options);
    Subspace const subspace{.around = start, .free = someAxes(context_, c, start, seed)};
    offer(optimize(context_.model, draw_, c.subject, filter, &subspace).options);
    if (std::optional<UseConfig> const step = bestStep(c, around, filter)) { offer(*step); }
    return out;
  }

  // The single-axis move from `around` this draw likes best.  A random subspace holds any one axis only now and then,
  // so a move whose effect here is uncertain -- a stripe width read on other shapes but not this one -- would wait long
  // for a draw to look at it; a move changes one effect, so the draw's noise cannot stack across axes the way a free
  // optimum's can.
  std::optional<UseConfig> bestStep(const Candidate& c, const UseConfig& around, const AxisFilter& filter) const {
    const Model& model = context_.model;
    std::vector<Axis> const axes = axesOf(model.env(), c.subject.fft, around, filter);
    Energy const e = model.energy(draw_, c.subject, around, axes);
    double best = 0;
    std::optional<std::pair<size_t, size_t>> move;
    for (size_t a = 0; a < axes.size(); ++a) {
      if (axes[a].option->structural) { continue; }
      size_t const from = axes[a].current;
      for (size_t p = 0; p < axes[a].values.size(); ++p) {
        if (p == from) { continue; }
        double delta = e.unary[a][p] - e.unary[a][from];
        for (const Energy::Pair& pair : e.pairs) {
          if (pair.a == a) {
            size_t const nb = axes[pair.b].values.size();
            size_t const b = axes[pair.b].current;
            delta += pair.table[p * nb + b] - pair.table[from * nb + b];
          } else if (pair.b == a) {
            size_t const nb = axes[a].values.size();
            size_t const o = axes[pair.a].current;
            delta += pair.table[o * nb + p] - pair.table[o * nb + from];
          }
        }
        if (delta < best) {
          best = delta;
          move = {a, p};
        }
      }
    }
    if (!move) { return {}; }
    UseConfig step = around;
    place(step, axes[move->first], move->second);
    return fitted(model.env(), c.subject.fft, step);
  }

  // The mean's optimum, moving at most MEAN_STEP_AXES axes from `around`: those its unbounded optimum gains most from
  // one at a time, then minimised together.  Unbounded, it stacks every effect the fit has spread over axes that were
  // only ever measured moving together, and predicts a saving no configuration has.
  Optimum meanStep(const Candidate& c, const UseConfig& around, const AxisFilter& filter) const {
    const Model& model = context_.model;
    Draw const mean = model.mean();
    Optimum const full = optimize(model, mean, c.subject, filter);

    std::vector<Axis> const axes = axesOf(model.env(), c.subject.fft, around, filter);
    Energy const e = model.energy(mean, c.subject, around, axes);
    std::vector<std::pair<double, std::string>> moves;
    for (size_t a = 0; a < axes.size(); ++a) {
      std::pair<int, int> const to = positionOf(model.env(), c.subject.fft, full.options, axes[a]);
      if (to == axes[a].values[axes[a].current]) { continue; }
      auto const at = std::ranges::find(axes[a].values, to);
      double const gain =
        at == axes[a].values.end() ? 0 : e.unary[a][size_t(at - axes[a].values.begin())] - e.unary[a][axes[a].current];
      moves.emplace_back(gain, axes[a].name);
    }
    if (moves.size() <= MEAN_STEP_AXES) { return full; }

    std::ranges::sort(moves);
    Subspace subspace{.around = around, .free = {}};
    for (size_t i = 0; i < MEAN_STEP_AXES; ++i) { subspace.free.push_back(moves[i].second); }
    return optimize(model, mean, c.subject, filter, &subspace);
  }

  const AcquireContext& context_;
  Draw draw_;
  std::map<const Candidate*, Valued> cache_;
};

DrawView takeDraw(const AcquireContext& context, std::mt19937_64& rng) {
  std::chi_squared_distribution<double> chi{DRAW_TAIL_DOF};
  double const tail = std::sqrt(DRAW_TAIL_DOF / std::max(chi(rng), 1e-9));
  return DrawView{context, context.model.draw(rng() | 1, context.scale * tail)};
}

// The cheapest candidate at `E` under one draw.
std::pair<const Candidate*, Valued> winnerAt(DrawView& view, const std::vector<Candidate>& candidates, u64 E) {
  view.valueAt(candidates, E);
  const Candidate* best = nullptr;
  Valued bestValue;
  for (const Candidate* c : view.screen(candidates, E)) {
    const Valued& v = view.valued(*c);
    if (v.value < bestValue.value) {
      best = c;
      bestValue = v;
    }
  }
  return {best, bestValue};
}

bool wellMeasured(const Readings& readings, const Candidate& c, const UseConfig& options) {
  const Reading* const r = readings.find(c.subject, options);
  return r && r->calls >= WELL_MEASURED;
}

}  // namespace

const Optimum* MeanOptima::find(const Candidate* c) const {
  std::lock_guard const lock{mutex_};
  auto const it = optima_.find(c);
  return it == optima_.end() ? nullptr : &it->second;
}

void MeanOptima::put(const Candidate* c, Optimum o) {
  std::lock_guard const lock{mutex_};
  optima_.try_emplace(c, std::move(o));
}

double MeanOptima::screenValue(const Candidate* c) const {
  std::lock_guard const lock{mutex_};
  auto const it = screened_.find(c);
  return it == screened_.end() ? NONE : it->second;
}

void MeanOptima::screen(const std::vector<const Candidate*>& candidates,
                        const std::function<double(const Candidate*)>& value) {
  std::vector<const Candidate*> todo;
  {
    std::lock_guard const lock{mutex_};
    for (const Candidate* c : candidates) {
      if (!screened_.contains(c)) { todo.push_back(c); }
    }
  }
  std::vector<double> values(todo.size());
  parallelFor(todo.size(), [&](size_t i) { values[i] = value(todo[i]); });
  std::lock_guard const lock{mutex_};
  for (size_t i = 0; i < todo.size(); ++i) { screened_.try_emplace(todo[i], values[i]); }
}

std::optional<Choice> choose(const AcquireContext& context, std::mt19937_64& rng) {
  std::vector<size_t> kinds;
  for (size_t k = 0; k < context.scope.grids.size() && k < context.candidates.size(); ++k) {
    if (!context.candidates[k].empty()) { kinds.push_back(k); }
  }
  if (kinds.empty()) { return {}; }

  std::uniform_real_distribution<double> uniform{0, 1};
  for (u32 attempt = 0; attempt < POINT_TRIES; ++attempt) {
    size_t const k = kinds[std::uniform_int_distribution<size_t>{0, kinds.size() - 1}(rng)];
    const Grid& grid = context.scope.grids[k];
    std::vector<double> weights;
    for (const GridPoint& p : grid.points) { weights.push_back(p.weight); }
    u64 const E = grid.points[std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng)].exponent;

    DrawView first = takeDraw(context, rng);
    auto const [winner, valued] = winnerAt(first, context.candidates[k], E);
    if (!winner) { continue; }

    bool const settled = wellMeasured(context.readings, *winner, valued.options);
    if (!settled && uniform(rng) < TOP_TWO_BETA) {
      return Choice{.candidate = winner,
                    .options = valued.options,
                    .predictedUs = std::exp(valued.value),
                    .why = "the draw's cheapest at " + std::to_string(E)};
    }

    for (u32 d = 0; d < CHALLENGER_DRAWS; ++d) {
      DrawView other = takeDraw(context, rng);
      auto const [challenger, cv] = winnerAt(other, context.candidates[k], E);
      if (!challenger || (challenger == winner && cv.options == valued.options)) { continue; }
      if (wellMeasured(context.readings, *challenger, cv.options)) { continue; }
      return Choice{.candidate = challenger,
                    .options = cv.options,
                    .predictedUs = std::exp(cv.value),
                    .why = "a challenger at " + std::to_string(E)};
    }
    if (!settled) {
      return Choice{.candidate = winner,
                    .options = valued.options,
                    .predictedUs = std::exp(valued.value),
                    .why = "the draw's cheapest at " + std::to_string(E) + ", unchallenged"};
    }
  }
  return {};
}

const char* toString(Novelty novelty) {
  switch (novelty) {
  case Novelty::Type: return "new to the type";
  case Novelty::Shape: return "new to the shape";
  case Novelty::Entry: return "new to the FFT";
  case Novelty::Set: return "a new set";
  }
  return "?";
}

std::optional<Choice> untriedStep(const AcquireContext& context, std::mt19937_64& rng, bool positionsOnly) {
  std::vector<size_t> kinds;
  for (size_t k = 0; k < context.scope.grids.size() && k < context.candidates.size(); ++k) {
    if (!context.candidates[k].empty()) { kinds.push_back(k); }
  }
  if (kinds.empty()) { return {}; }

  const Model& model = context.model;
  struct Step {
    const Candidate* candidate;
    UseConfig options;
    Novelty novelty;
    double weight;
  };
  for (u32 attempt = 0; attempt < POINT_TRIES; ++attempt) {
    size_t const k = kinds[std::uniform_int_distribution<size_t>{0, kinds.size() - 1}(rng)];
    const Grid& grid = context.scope.grids[k];
    std::vector<double> weights;
    for (const GridPoint& p : grid.points) { weights.push_back(p.weight); }
    u64 const E = grid.points[std::discrete_distribution<size_t>{weights.begin(), weights.end()}(rng)].exponent;

    DrawView const view{context, model.mean()};
    std::vector<const Candidate*> leading = view.screen(context.candidates[k], E);
    if (leading.size() > UNTRIED_LEADERS) { leading.resize(UNTRIED_LEADERS); }

    // The newest kind of step any leading candidate has, then one of those at random: a position no FFT of the type
    // was read at teaches every one of them, one read on other shapes only teaches this shape as far as the option's
    // effect varies between shapes, and so on down to a set nothing measured.
    std::vector<Step> steps;
    for (const Candidate* c : leading) {
      const Reading* best = nullptr;
      for (const Reading* r : context.readings.of(c->subject)) {
        if (!r->failed && r->calls > 0 && (!best || r->mean < best->mean)) { best = r; }
      }
      UseConfig const around = best ? best->options : UseConfig{};
      for (const Axis& axis : axesOf(model.env(), c->subject.fft, around, context.filter(*c))) {
        double const otherShape = std::min(1.0, model.shapeDeparture(axis.name) / model.effectSpread());
        double const otherVariant = std::min(1.0, model.entryDeparture(axis.name) / model.effectSpread());
        for (size_t p = 0; p < axis.values.size(); ++p) {
          if (p == axis.current) { continue; }
          std::string const label = axis.label(p);
          Step step{.candidate = c, .options = around, .novelty = Novelty::Set, .weight = 1};
          if (p != axis.reference && !model.measured(c->subject, label)) {
            step.novelty = Novelty::Type;
          } else if (p != axis.reference && !model.measuredOnShape(c->subject, label)) {
            step = {.candidate = c, .options = around, .novelty = Novelty::Shape, .weight = otherShape};
          } else if (p != axis.reference && !model.measuredAt(c->subject, label)) {
            step = {.candidate = c, .options = around, .novelty = Novelty::Entry, .weight = otherVariant};
          } else if (positionsOnly) {
            continue;
          }
          if (!steps.empty() && step.novelty > steps.front().novelty) { continue; }
          place(step.options, axis, p);
          step.options = fitted(model.env(), c->subject.fft, step.options);
          if (context.readings.find(c->subject, step.options) || context.blocked(*c, step.options)) { continue; }
          if (!steps.empty() && step.novelty < steps.front().novelty) { steps.clear(); }
          steps.push_back(std::move(step));
        }
      }
    }
    if (steps.empty()) { continue; }
    std::vector<double> w;
    for (const Step& s : steps) { w.push_back(std::max(s.weight, 1e-6)); }
    const Step& step = steps[std::discrete_distribution<size_t>{w.begin(), w.end()}(rng)];
    return Choice{.candidate = step.candidate,
                  .options = step.options,
                  .predictedUs = std::exp(model.predict(model.mean(), step.candidate->subject, step.options)),
                  .why = std::string{"an untried step ("} + toString(step.novelty) + ") at " + std::to_string(E)};
  }
  return {};
}

double expectedRegret(const AcquireContext& context, const Publication& publication, std::mt19937_64& rng, u32 draws) {
  std::vector<double> regrets;
  for (u32 i = 0; i < draws; ++i) {
    DrawView view = takeDraw(context, rng);
    double excess = 0;
    double cost = 0;
    for (size_t k = 0; k < context.scope.grids.size() && k < context.candidates.size(); ++k) {
      const Grid& grid = context.scope.grids[k];
      const KindPublication* const pub = publication.of(grid.kind);
      for (const GridPoint& p : grid.points) {
        const Line* const line = pub ? pub->at(p.exponent) : nullptr;
        if (!line) { return NONE; }
        const FFTConfig& fft = line->candidate->subject.fft;
        Subject const there{.fft = fft, .kind = grid.kind, .regime = regimeOf(fft, p.exponent)};
        double const published = std::exp(view.predict(there, line->reading->options));
        double const best = std::exp(winnerAt(view, context.candidates[k], p.exponent).second.value);
        excess += p.weight * std::max(0.0, published - best);
        cost += p.weight * published;
      }
    }
    regrets.push_back(cost > 0 ? excess / cost : 0);
  }
  if (regrets.empty()) { return 0; }
  std::ranges::nth_element(regrets, regrets.begin() + regrets.size() / 2);
  return regrets[regrets.size() / 2];
}

}  // namespace tune
