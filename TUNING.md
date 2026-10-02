# Tuning PRPLL

PRPLL can run each exponent on many different FFTs, and each FFT has dozens of `-use` options that change how its
kernels are built. Which combination is fastest depends on the GPU, the driver, the backend (OpenCL or CUDA) and the
exponent. `-tune` measures this on your GPU, for the exponents you actually test, and publishes the result in a file
that normal PRPLL runs read automatically.

In short:

```sh
prpll -dir <your run directory> -device <N> -tune     # tune; stop it with Ctrl-C whenever you like
prpll -dir <your run directory> -device <N>           # a normal run now uses what was measured
```

Everything below explains what those two commands do and how to adjust them.

- [How it works](#how-it-works)
- [Running a tune](#running-a-tune)
- [Settings](#settings)
- [Watching and stopping a run](#watching-and-stopping-a-run)
- [How a normal run uses the result](#how-a-normal-run-uses-the-result)
- [Subcommands](#subcommands)
- [Files](#files)
- [Envs: one set of measurements per card, driver and build](#envs-one-set-of-measurements-per-card-driver-and-build)
- [Several GPUs](#several-gpus)
- [Other tools: `-measure` and `-options`](#other-tools--measure-and--options)
- [The previous tuner: `-oldtune`](#the-previous-tuner--oldtune)
- [Exit statuses](#exit-statuses)
- [Questions](#questions)


## How it works

**One number is optimised.** The tuner works on the expected time per iteration over your workload, called `T` in
its output. For each exponent you are likely to test, it asks which FFT and options a normal run would pick there and
how long an iteration of that takes, then averages over the exponents, weighted by how much of your work is at each
one. Everything the tuner does is an attempt to lower `T`, and it spends its time on whatever it expects to lower `T`
the most per second of measuring.

**Everything published was measured.** An FFT is only ever published with the exact option set it was timed under,
and only for the exponents where that timing applies. Nothing is inferred from one FFT and applied to another. Costs
are published pessimistically (the mean plus two standard errors), so a configuration that got one lucky reading
cannot beat one that is well measured.

**Accuracy is checked, not assumed.** Before a floating-point configuration is published, its rounding error is read
on your GPU at the top of the exponent range it would serve. Some options change the rounding error (for example
`TAIL_TRIGS`, `TABMUL_CHAIN`, `MM_CHAIN`); a configuration that sets one of those must also read no worse than the same
configuration without it. Where the error is too high, the configuration is published only up to the exponent where it
is safe, or not at all; where the margin is large, it may be allowed slightly past PRPLL's standard limit.
Configurations that produce a wrong residue while being measured are recorded as excluded and are never used again.

A run goes through these stages, though it interleaves them and you do not need to manage any of them:

1. **Scope.** Work out the exponents to tune for (the *workload*) and the single exponent that matters most (the
   *probe*). By default both come from your worktodo files.
2. **Drift anchor.** Pick one reference configuration and re-time it every few minutes, so that a GPU that slows down
   as it warms up (or speeds up as something else stops) does not distort the comparison.
3. **Coverage.** Make sure every exponent in the workload has an FFT published for it: where none has, time the
   FFT most likely to be cheapest there (its default variant first) and read its rounding error. This runs before
   anything else, so a short run still covers the whole workload.
4. **Defaults sweep.** Time every FFT within 10% of the fastest one somewhere in the workload at PRPLL's built-in
   defaults, those at the probe exponent first. This is the untuned map: what each of them costs before any option is
   changed, which is what every later gain is a gain over, and what the dashboard compares against. An FFT not timed
   yet that is expected to be faster than anything measured is timed first, and the FFTs it may put out of reach wait
   on it: if it is as fast as expected they are never timed, and if not they join the sweep, whose count then grows.
5. **Bootstrap.** Once the sweep has read the FFTs at the probe exponent, for each FFT type (FP64, the NTTs, the
   hybrids), search the type's fastest FFT there ahead of the rest of its type: `4 × roundCalls` measurements each (64
   by default), by the same search every FFT gets. What it finds is published like any other result, and so becomes
   the type's default options straight away. A type that is so much slower than the fastest one that no plausible
   option gain could close the gap is not searched first.
6. **Search.** For the FFTs that are competitive, first try the default options as they stand, then other option
   sets one step at a time (how big a step is depends on `strategy=`), and combine the best answers of different
   option groups. A random option set is tried to escape a local optimum: once no step is left, and after every 32
   option sets an FFT measures whether or not steps are left. Whenever the default options change, an FFT tries them
   again, laid over the best options it has found itself and then as they are. The search is first spread over the
   contenders, the FFTs within 10% of the fastest, or within 10% of the fastest when both are at their defaults, since
   which of them tunes best cannot be told from their defaults:
   once the sweep has read them all at the probe exponent, each gets an equal share in rounds, the slower half
   dropping out after each round and the share doubling, until one is left (see `contenders=`). The search then goes
   wherever the next step is expected to gain most, until the FFT left has had as many measurements again as the
   rounds gave out; then the contenders as they stand are halved again, those dropped earlier among them, with every
   share twice as long as last time. So an FFT whose gain lies deep in its search is come back to, for longer each
   time. FFTs further off the pace are timed at their defaults as they become worth it. The sweep, the bootstrap and
   the search take turns, a measurement at a time, so none of them waits for the others to finish.
7. **Accuracy checks.** Read the rounding error of any published configuration whose options change it.

**The default options follow the search.** Once FFTs are published, each FFT type's default options are those of its
best published FFT: the one a normal run would use at the probe exponent, or where none of that type reaches it, the
one nearest it. A type with nothing searched published has no line of its own. Options that change the rounding
error stay at their defaults on these lines, since nothing reads the rounding error of an FFT that merely runs under
them. Every FFT is first timed at the built-in defaults, and the lines are the first step of its search; every
published FFT names the options it was measured with, so it runs as measured whatever the lines come to say.

`selection.txt` is rewritten after every measurement. It holds nothing until the first FFTs have been timed and had
their rounding error read; soon after, it covers the whole workload at the built-in defaults, and from then on it
improves with each measurement, so what it holds is always usable. The run ends by itself when nothing left is expected to lower `T` by more than 0.1%, or when
you press Ctrl-C.


## Running a tune

Run it in the directory your normal PRPLL runs use (`-dir`), on the GPU you want to tune (`-device`, `-pci` or
`-uid`), with nothing else running on that GPU:

```sh
prpll -dir ~/prpll/gpu0 -device 0 -tune
```

What to expect:

- **Duration.** Each measurement builds the FFT's kernels and times about 5000 iterations: from several seconds to a
  minute or more, depending on the FFT and the GPU. Choosing the drift anchor takes a few minutes, the bootstrap
  64 measurements (by default) per FFT type it searches first, and a run that goes until it stops by itself usually
  takes many hours. The workload is covered first; the defaults sweep, which on a wide workload can take hours, then
  takes turns with the search, so options are tuned from the start (`-tune status` shows how far each has got). The
  most valuable measurements come first, and you can stop whenever you need the GPU and resume later.
- **Resuming.** Run the same command again. Every measurement is saved in `tunedb.txt` as it is taken, so nothing but
  the measurement in progress is lost.
- **Your settings are set aside.** While tuning, every `-use` option from `config.txt` (including `!` lines) and from
  the command line is ignored, as is `-carry`, so that every measurement starts from the built-in defaults. The run
  says which ones it is ignoring. The exceptions are options the tuner never searches, such as `NO_ASM`, which stay in
  force when given on the **command line**, so a GPU that needs them can still be tuned. `-fft` is also ignored: the
  tuner chooses among every FFT itself.
- **Block size.** Measurements use the PRP block size (`-block`, default 1000), so tune with the same `-block` you run
  with.
- **One test at a time.** Measurements run one test at a time. If you run several tests at once on one GPU
  (`-workers`), the published costs will not reflect that.
- **Crashes.** If the GPU faults or the driver loses the device during a measurement, PRPLL restarts itself (up to 16
  times, or the value of the environment variable `PRPLL_MAX_RESTARTS`) and carries on. The configuration that was
  being measured is never built again on that GPU. `-tune status` lists such configurations, and `-tune reset` clears
  them if a driver or kernel change has fixed the cause.

A run begins by reporting what it will do:

```text
tune: Tuning from built-in defaults; no -use settings to ignore
tune: pending work read from worktodo-0.txt
tune: workload 112159852-170349795 (2 assignments pending, from 5% below to 25% above)
tune: probe 118063003 (the most-populated 2% bin of the pending work), carrying 50% of the weight
tune: prp grid: 65 exponents, spread across the range
tune:   118063003  75.0%    870.302 us/it  prior, from 1K:13:256  (probe)
tune:   136279841  25.0%   1013.328 us/it  prior, from 1K:15:256
tune: T = 906.059 us/it against env 1, 0.0% of the weight on measured entries
tune: 945 entries could serve the workload; each measured one is searched by strategy=hybrid (maxPermute=4, maxPoints=64, comboTop=3, comboTiers=3), then by random restarts; the run goes on until nothing is expected to lower T by 0.1% of it
tune: bootstrap at 118063003 once the workload is covered and every FFT within 10% of the fastest has been read at the built-in defaults: each type's fastest there is searched first, for 64 calls
```

The costs marked `prior` are estimates, used only to decide what to measure first. They are replaced by measurements
as the run goes on. Once the sweep is done, the run names the FFT each type is searched on first:

```text
tune: bootstrap at 118063003 over FFT64 1K:13:256:101, FFT3161 1:1K:8:256:202, FFT3261 2:1K:8:256:212, FFT61 3:1K:16:256:202, FFT323161 4:1K:8:256:212, FFT6431 51:1K:8:256:212
```


## Settings

Settings go after `-tune` as **one word, separated by commas, with no spaces**:

```sh
prpll -tune workload=100M-140M,probe=118063003,stop=0.5%
```

Exponents may be written in full or with a `K`, `M` or `G` suffix (`118063003`, `140M`).

### What to tune for

**`workload=<lo>-<hi>`**: the range of exponents to tune for, weighed evenly across it (64 points spread over it, plus
the probe). FFTs that cannot serve any exponent in the range are never measured. A single exponent
(`workload=118063003`) is allowed. The exponents in your worktodo do not weigh in: a worktodo holds a few days of the
months of work a tune is used for.

- Default: from 5% below the lowest exponent in your worktodo files to 25% above the highest, since the assignments
  you receive next tend to be larger. With no pending work, `100M-400M`.
- The worktodo files read are the ones a normal run would read: `worktodo-<N>.txt` for each worker, the pool's
  `worktodo.txt` when `-pool` is given, and a `worktodo.txt` in the run directory. Exponents that are not prime, and
  lines for anything other than Mersenne numbers, are skipped. `Cert` lines count as PRP work.

**`probe=<E>`**: the exponent that matters most. The bootstrap searches each FFT type at this exponent first, and it
carries extra weight in `T`.

- Default: the most common exponent in your worktodo files (the most populated 2%-wide band, represented by its
  average). With no pending work, the geometric middle of the workload.
- It is taken to the prime at or below the value given.
- A probe outside a workload you named is refused. A probe outside the range derived from your worktodo widens that
  range to reach it.

**`probeWeight=<0..1>`**: how much of the weight in `T` sits on the probe alone; the rest is spread evenly across the
workload. Default `0.5`. `0` tunes the whole range evenly; `1`
tunes for the probe exponent and nothing else.

**`kinds=prp|ll|prp+ll`**: which tests to tune for. Default `prp`. LL work is only worth tuning for if you actually
run LL tests; with `prp+ll` each kind gets its own measurements and its own share of `T`. An LL measurement's residue is
checked against the residue two different FFTs agree on at their built-in defaults, since there is no known LL residue
to compare with. The bootstrap searches in PRP when PRP is among the kinds and in LL otherwise, so an LL-only tune takes
no PRP measurements.

Examples:

```sh
# Tune for the worktodo's range and a way past it, whatever it holds (the default).
prpll -tune

# First-time checks around 118M, with most of the effort on the exponent in hand.
prpll -tune workload=110M-130M,probe=118063003,probeWeight=0.8

# A GPU that will spend its life on 330M-340M work, with nothing in its worktodo yet.
prpll -tune workload=330M-340M

# Treat the whole range evenly.
prpll -tune workload=100M-400M,probeWeight=0
```

To see what a set of settings would tune for without opening a GPU, put `scope` in front of them
([`-tune scope`](#-tune-scope)).

### How long to run

**`stop=<P>%|0`**: the run ends once nothing left is expected to lower `T` by `P` percent. Default `0.1%`. A larger
value gives a shorter run (`stop=1%`); `stop=0` runs until you stop it with Ctrl-C. The bootstrap, the accuracy
checks, the coverage of the workload, the first two halvings of the search and any later one once begun (see
`contenders=` and `halvings=`) and a measurement once begun, which one more call finishes, are always completed,
whatever `stop=` says.

A value without `%` is refused, except `0`: `0.1` could mean either 0.1% or 10%.

### How to search

These change how the options of each FFT are searched. The defaults are the recommended ones.

**`bootstrap=0|1`**: whether to search each FFT type's fastest FFT first, for `4 × roundCalls` measurements, once the
defaults sweep is done (default `1`). With `bootstrap=0` the search goes wherever it is expected to gain most from the
start, and a type's default options wait until one of its FFTs is searched that way. This is mainly useful to compare
against.

**`strategy=<S>`**: what one step of the per-FFT search is.

- `hybrid` (default): options are searched in groups of related options, trying combinations within a group (a large
  group in pieces of `maxPermute` options, and at most `maxPoints` combinations each), then the best answers of
  different groups are combined.
- `groups`: the groups alone, without combining them. The same as `hybrid` with `comboTiers=1`.
- `single`: one option at a time.
- `permute:<KEY>+<KEY>...`: every combination of exactly the options named, e.g.
  `strategy=permute:PAD+TAIL_KERNELS+IN_SIZEX`. The number of combinations grows quickly.

Every step starts from the best option set the FFT has shown so far (for an option that changes which other options
exist, such as `INPLACE`, from the best set of each of its values), and moves only the options of that step; every
other option stays where the best set has it. When the best set changes, the steps are taken from the new one; a
step already measured from an earlier best set is taken again only after every step never measured, since what it did
there says something, though not everything, about what it does here. So two changes in different pieces of a group,
neither of which helps alone, are tried together only by the combinations (below) or by chance. The steps that switch
an option of the `INPLACE` kind come before any other step of the FFT, so the search learns early which side of each
is the faster one and spends its time there.

Under `hybrid` and `groups` the pieces then take turns: each piece of each group gets a step before any gets another,
counting the steps it has already had, so a large piece such as `Memory`'s does not keep the rest of the groups
waiting until it is done. Where both sides of an option of the `INPLACE` kind have been measured, each piece is
searched on every side, the faster side's taking as many turns as all the other sides' together. The combinations of two groups start once those two
groups have no step left, while the other groups are still being searched. A measurement that was begun is finished
before anything else of its FFT, and is ranked by what its first call read rather than by what a step of its FFT is
expected to gain, so one whose first call beat everything else is taken ahead of steps expected to gain less, even
on an FFT that is otherwise out of the running. Under `single` each option's values are tried one after another, in turn.

Whatever the strategy, an FFT whose search has run out of steps is occasionally tried with a random option set, which
is what lets the search find combinations no step would reach.

**`maxPermute=<N>|all`** (`hybrid` and `groups`): how many options of one group are permuted together. A group with
more is split into pieces of this many, in a fixed order, and no step moves options in two pieces at once. `all` keeps
each group whole. Default `4`. The register limits of the `Cuda` group (`REGMI64`, `REGTS31` and the like) are never
permuted: each limits one kernel, run on its own, so each is searched one value at a time whatever `maxPermute` is,
and the combinations try the best of them together.

**`maxPoints=<N>|all`** (`hybrid` and `groups`): how many combinations are tried in each piece of a group, and in each
combination of groups. A piece's combinations are listed one option moved first, then two, and so on, so a cut keeps
the smaller moves: with the defaults, the larger groups are cut before any step moves all four of their options at
once. `all` tries every combination. Default `64`.

Raising either is how to spend more time on each FFT; every combination is a measurement. How many each piece holds
depends on the card, the backend and the FFT, and a run says it as it starts, one line per FFT type; `-tune scope`
says it group by group, and takes these settings too, so the effect of a value can be seen before a run spends
anything on it:

```text
tune: strategy=hybrid (maxPermute=4, maxPoints=64, comboTop=3, comboTiers=3) steps from one best set of each type, before the combinations above them:
tune:   FFT64 1K:7:256:212: 195 steps from each best set (Placement 6, Memory 93 of 178, Queues 3, Tail 23, Width 6, Height 4, Cuda 60), 4601 with maxPermute=all
tune:     Memory: 7 options in bins of 149 and 29 points, 64 and 29 offered; 4499 with maxPermute=all
tune:     Cuda: 6 options in bins of 7, 13, 13, 13 and 13 points, all offered; 1 structural step
```

On NVIDIA the one piece the defaults cut is `Memory`'s first, at 64 of its 149 combinations, so `maxPoints=150` lists
every piece whole; `maxPermute=all` makes `Memory` one piece of 4499. On an AMD Radeon Pro VII nothing is cut, and the
largest piece with `maxPermute=all` is `Placement`'s, about 400. A run lists a large piece 64 combinations at a time,
in the same order, and more as those are measured, so it keeps moving however large the pieces are; the progress line,
`-tune status` and the summary add `and up to <N> more not listed yet` for what lies beyond. The bootstrap searches the
same way.

**`comboTop=<N>`** (`hybrid` only): how many of each group's best answers are carried into the combinations. Default
`3`. An answer the measurements cannot tell from the last of those, within the noise of both, is carried as well:
which of several equally good answers happens to read first says nothing about which combines best. A group's answers
are ranked by what each did against the same settings of every other group, from measurements that differ only within
the group, so an answer is not credited with what another group's setting did for it; an answer that was only ever
measured beside other changes is not carried.

**`comboTiers=1|2|3`** (`hybrid` only): how widely groups are combined. `1`: none (the same as `strategy=groups`);
`2`: groups that share kernels are combined, and so are the pieces of one group (`Memory combined`, `Cuda combined`),
each piece's best answers with the others'; `3`: everything is combined. Default `3`.

**`contenders=<N>`**: how many FFTs the search is first spread over. The contenders are the FFTs within 10% of the
fastest one somewhere in the workload, as measured now or as measured at the built-in defaults: an FFT that is behind
only because the fastest has been searched and it has not is still a contender. They are taken nearest first as they
stand now, one variant of each shape before a second variant of any, so that the search looks at several shapes rather
than the variants of one; of two equally near, the one that runs more of the workload goes first. Each is searched for
`roundCalls` calls, then the slower half drops out, the rest get twice as many calls, and so on until one is left. An
FFT stays in a round until it has had its calls, however far ahead another pulls meanwhile, and a round is recorded in
the database, so an interrupted run picks it up where it stopped. When the workload changes (another range, or other
`kinds=`), a round goes on with the FFTs in it that the new workload runs; one it no longer runs keeps the calls it is
owed, and has them, before anything else is decided, as soon as a later tune's workload runs it again. The FFT left is
then searched by expected gain for as many calls as the rounds gave out, after which the contenders as they stand then,
including those that dropped out and those that have come within 10% since, are halved again, each round twice as long
as the matching round of the last halving. A round gives each FFT in it its calls whatever the search expects them to
gain, since what it is for is a gain several steps away that no single step shows. A halving, once begun, runs to its
end whatever `stop=` says, so a short run still looks at more than one shape. The first two halvings begin whatever
`stop=` says too (see `halvings=`); `stop=` decides whether a later one begins, which it does only while a step of one
of its FFTs is worth `stop=` on its own. `0` ranks the search by expected gain from the start, which tends to spend
everything on whichever FFT was fastest before tuning. Default `16`.

**`roundCalls=<N>`**: how many calls each contender is searched for in the first round of the first halving; each
later round doubles it, and so does each later halving. Only calls made during a round count towards it. Default
`16`, so 16 contenders take about 1000 calls in the first halving.

**`halvings=<N>`**: how many halvings of the search begin whatever `stop=` says. How much a step is expected to gain is
judged one step at a time, so a gain that only shows after several steps (an option that helps only beside another, or
two options that each cost a little alone) is invisible to `stop=`; the second halving's longer rounds are what find
most of those. Later halvings begin only while a step of one of their FFTs is worth `stop=` on its own. `1` gives a
shorter run that may miss such gains. Default `2`, so with 16 contenders about 3000 calls are made in the first two
halvings before `stop=` can end a run.

### Output

**`tunetxt=0|1`**: also write `tune.txt`, the file older PRPLL binaries and `-oldtune` use, after every publish.
Default `0`, so an existing `tune.txt` is never overwritten. See [Files](#files) for what it holds.

**`dashboard=0|1`**: show the run as a full-screen view instead of scrolling lines; see
[The dashboard](#the-dashboard). Default `0`. It changes only what the terminal shows: the log file is the same either
way, and the setting is not part of what the run records, so `-tune status` and a resumed run are unaffected.


## Watching and stopping a run

Each measurement ends with a line saying what was measured, what it cost, and what it did to `T`:

```text
tune: 7. baseline 256:3:256:002 prp short32 at the built-in defaults at 5999137: 55.897 us/it, 2.2 s; T 51.224 -> 51.224 us/it
tune: 8. probe 256:3:256:212 prp short32 Placement INPLACE=0 at 5999137: 59.929 us/it, 2.0 s; T 51.224 -> 51.224 us/it
tune: 9. baseline 256:3:256:002 prp short32 at the built-in defaults at 5999137 (resumed at call 2): 55.910 us/it, 1.9 s; T 51.224 -> 51.224 us/it
```

A measurement also says what its row is now ranked at, as `ranked at <us> over <n> calls`: `selection.txt` and the
search rank a row by its mean plus two standard errors, so a configuration measured only twice can read faster than the
one in use and still rank behind it until more calls narrow its error. The tuner gives those calls first to the
configuration most likely to beat the one in use.

The defaults sweep, the bootstrap and the search take turns, a measurement at a time, so none of them holds the
others up. The bootstrap says which FFT it searches first for each type, which types it does not, and why:

```text
tune: bootstrap: FFT64 256:3:256:202 is searched first, for 16 calls, from 50.412 us/it at the built-in defaults
tune: bootstrap: FFT3161 1:256:2:256:202 is not searched first: at 172.119 us/it it would take more than a 32% gain to bring it level with the cheapest type
...
tune: bootstrap: FFT64 256:3:256:202 has had its 16 calls of search, and is at 47.925 us/it
tune: bootstrap complete; the default lines are SHUFL_BYTES_H=16,SHUFL_BYTES_W=16
```

Besides those lines:

- Every 5 minutes, and once when the bootstrap finishes, a `tune: progress:` line gives the time spent, what the run
  is doing and how far through it it is, the measurements taken by kind, how `T` has moved, and how much measuring is
  currently worth doing:

  ```text
  tune: progress: 4:21 in, halving: round 1 of 2, 4 contenders, 9 of 16 calls; 111 items and 1 anchor reading: 40 baseline (1.5 min), 65 probe (2.2 min), 6 gate (0.2 min); T 41.107 -> 47.985 us/it, 100.0% of the weight on measured entries; 1422 items are worth running now, ~51 min by the queue's estimates
  ```

  The first part counts through what the run is doing: `covering the workload: 62.0% of its weight measured` or
  `accuracy gate: 3 readings owed`, which go before anything else, or else whichever of `defaults sweep: 23 of 70 FFTs
  read at the built-in defaults`, `bootstrap: FFT64 256:3:256:202 prp short32, 12 of 16 calls of search; type 1 of 1`
  and `halving: round 2 of 4, 8 contenders, 150 of 256 calls` (or `searching by expected gain`) are taking turns,
  joined by ` + `. "Items worth running now" is only what the queue can take next.

- When the default options move to follow a newly published FFT, the run says what they are now:

  ```text
  tune: the default lines are now MULTI_Q=1,SHUFL_BYTES_H=16; ! 0 LOADS=30050; ! 51 LDSPAD_H=0,ZEROHACK_H=0,ZEROHACK_W=0, from the best sets published
  ```

- A measurement still going after a minute prints `tune: still measuring ...` each minute, so that a slow compile is not
  mistaken for a hang.
- On a terminal, the bottom line is redrawn in place with the measurement in progress. It is never written to the log
  file.

Any configuration that has taken the device down, or computed a wrong answer, is listed as soon as a run starts (so
after a restart the new process says it again) and again in the summary, each with the command line that reproduces
it. Neither should ever happen: either is a kernel or driver bug, worth reporting with that line. A wrong answer is
only concluded from two readings: a call whose Gerbicz check fails (or, for LL, whose residue disagrees with the
reference) is read again at once, since a card without ECC flips a bit now and then, and only a second failure is
recorded.

```text
tune: 1 configuration took the device down and will not be built again on this device; a kernel or driver bug, worth reporting with the line that reproduces it:
tune:   -fft 1K:10:256:010 -use FAST_BARRIER=1,INPLACE=1,LOADS=10000,WMUL=1   (prp at 95537053)
```

When the run ends, whether by itself or by Ctrl-C, it prints a summary: what was measured, why it stopped, per FFT type
what was measured and what was not (and how large a gain would have been needed to justify measuring it), what it has
learnt about how much a change of options tends to gain, and the drift of the GPU over the session. Stopped fifteen
minutes into a tune of a small workload:

```text
tune: summary: 426 items and 3 anchor readings: 40 baseline (1.5 min), 276 probe (9.1 min), 82 combo (2.6 min), 6 restart (0.2 min), 22 gate (0.7 min)
tune: summary: stopped before the queue was done; T 46.652 us/it, and an item is worth running at 0.0467 us/it (stop=0.1%)
tune: summary: FFT64: 18 of 513 entries measured, 495 not; the closest, 256:4:256:000 prp short32, needed a gain of 18.5% to be worth stop=0.1% of T (the gains learnt give that a 0.0667% chance), and was worth 0.0072 us/it
tune: summary: FFT3161: 1 of 10 entries measured, 9 not; the closest, 1:256:4:256:202 prp short32, needed a gain of 85.7% to be worth stop=0.1% of T (the gains learnt give that a 0% chance), and was worth 0.0000 us/it
...
tune: summary: left: 7029 probe, the best worth 0.0531 us/it (0.114% of T): 256:3:256:201 prp short32 the default lines ENABLE_RESTRICT=1,LOADS=23505,OLD_FENCE=0,SHUFL_BYTES_H=16,SHUFL_BYTES_W=16,STORES=3,ZEROHACK_H=0,ZEROHACK_W=0
...
tune: summary: drift: 3 readings of 256:3:256:212@5999137, ratio 1.0000 -> 1.0008 (1.0000 to 1.0017), within the warning
tune: T 41.107 -> 46.652 us/it, stopped before the queue was done
tune: benefit: at 5999137 production runs 256:3:256:202 at 46.652 us/it, 7.6% less per iteration than the best measured there at the built-in defaults (256:3:256:202, 50.488 us/it); env 1 has been measured for 15:01 over 1 session
```

The `benefit` line is what all the tuning so far has bought at the probe exponent: what a normal run now spends per
iteration there, against the fastest configuration measured there at the built-in options (the drift anchor race and
the bootstrap always measure some), for example:

```text
tune: benefit: at 89843291 production runs 1K:10:256:010 at 516.247 us/it, 17.1% less per iteration than the best measured there at the built-in defaults (1K:10:256:212, 622.880 us/it); env 1 has been measured for 2:57:26 over 3 sessions
```

**Ctrl-C** stops cleanly at any point: the measurement in progress is abandoned, and `selection.txt` already holds
everything measured before it. Pressing Ctrl-C is not treated as a failure of the configuration being measured.

**While it runs**, `prpll -dir <same directory> -tune status` in another terminal shows where it stands, without
disturbing it ([`-tune status`](#-tune-status)).

### The dashboard

With `dashboard=1` and a terminal on stdout, the run takes over the terminal's alternate screen, as `top` or `less` do,
and redraws it twice a second. Quitting gives back the screen you had before, and the summary prints there as usual.
Every line still goes to the log file exactly as it does without the dashboard. Without a terminal (output redirected
to a file, say), the setting is ignored with a note. From top to bottom:

- **The header**: the device, the workload and probe, how long this run and every session of this env so far have
  measured, and the restart generation when a restart has happened. Below it, what the run is doing and how far
  through it it is (the phase from the `progress` line), `T`, what the best item left is worth, and how much is left
  worth running.
- **FAULTS**, in red, only when there are any: every configuration that took the device down or computed a wrong
  answer on this env, newest first, each as the `-fft`/`-use` line that reproduces it.
- **NOW**: the measurement in progress and how long it has taken so far.
- **BENEFIT**: the `benefit` line described above, and under it two charts over all the measuring this env has had,
  across every session and restart, each with a line of its own saying what it is and where it has gone, its highest
  and lowest values at its right, and a time axis below it: `T` over the workload, and what a normal run would spend
  per iteration at the probe. Both fall as the tuning pays. They are rebuilt from the timestamps in `tunedb.txt` when
  the run starts (for up to 10 seconds; a long history is then drawn more coarsely), so they need nothing that an older
  build did not write. `T` is charted only from when every exponent of the workload had a measured FFT, since until
  then it still includes estimates. Each chart's `from` is its first value: where `T` was first measured over the whole
workload, and what production first ran at the probe. `^ this run` marks where the current run began.
- **PRODUCTION**: what `selection.txt` runs over the workload, one row per stretch of exponents served by the same FFT
  and options: its share of the workload, its cost, and how that compares with the fastest FFT measured there at the
  built-in defaults, which is what the tuning as a whole bought, the choice of FFT included. `untuned` marks a stretch
  still run at that FFT's defaults, `not read at the defaults` one with no such reading to compare with, and `prior` one
  nothing measured serves yet. On a small screen the stretches carrying the least weight are counted rather than
  listed.
- **SAMPLES**: up to seven exponents spread over the workload, the probe among them (marked `*`): for each, the fastest
  FFT at the built-in defaults and its cost, what a normal run would use there now and its cost, and the change. Where
  the tuning moved an exponent to another FFT, this is where it shows.
- **RECENT**: the measurements this run has made, newest first, with their results and, for a timing, what its row is
  ranked at and over how many calls; a new best set for an FFT is shown in green with the set.
- **NEXT**: what the queue would measure next and what each is worth.
- **EVENTS**: every other line the log gets (the bootstrap's decisions, the default options moving, notes and
  warnings), newest first.

From 160 columns the lower panels sit side by side; from 50 rows the charts are drawn taller. It works in any size, but
at least 120x40 shows everything usefully. Set `NO_COLOR` for no colours; a locale that is not UTF-8 gets plain ASCII.
If the process is killed by something that cannot be caught (`kill -9`), the terminal may be left in the alternate
screen: `reset` restores it.


## How a normal run uses the result

A normal PRPLL run (no `-tune`) looks for `selection.txt` in its run directory, and then in the `-pool` directory. For
each exponent it takes the cheapest published entry that covers that exponent, and runs it with the options published
for it. The file is read again for every new task, so a run picks up newly published results without being restarted.

Where no published entry covers an exponent (or there is no `selection.txt`), PRPLL falls back to its usual choice of
FFT (the fastest in `tune.txt`, or the smallest FFT that fits), but still under `selection.txt`'s default `use` lines,
and still honouring any exponent limits and exclusions the tuner published. Where there is a `selection.txt` but
nothing in it covers the exponent, PRPLL says so, once per exponent, with the range the file does cover and a workload
that would cover this exponent too:

```text
Note: no entry in selection.txt covers 79999987 (its prp entries cover 36700158-77888224), so PRPLL's own choice of FFT runs it under the file's default lines, which were not measured on it; a tuning run over it, such as -tune workload=36700158-79999987, would publish one
```

The tuner is the way to cover it: widen the workload (or add the assignment to the worktodo) and run `-tune` at least
until the coverage stage has published an FFT for it.

**Your own settings still win.** `config.txt` is never written by the tuner, and options are taken, highest priority
first, from:

1. `-use` on the command line
2. `!` lines in `config.txt` (see `prpll -h`)
3. plain `-use` lines in `config.txt`
4. `selection.txt`: the options of the entry chosen
5. `selection.txt`: the default line of the FFT type
6. `selection.txt`: the global default line
7. PRPLL's built-in defaults

So a `-use` line left in `config.txt` (for example by `-oldtune`) overrides what the tuner measured. When that happens,
PRPLL says so at startup:

```text
Note: the config files set TAIL_KERNELS, which shadow what selection.txt publishes; remove them to run what was measured
```

and an entry run under options other than the ones it was measured with is limited to the exponent range every
configuration of that FFT is allowed, since the tuner's accuracy checks no longer apply to it. Remove the `-use` lines
from `config.txt` to get the full benefit.

**`-fft` narrows the choice** to the entries of that FFT; where none covers the exponent, the named FFT still runs,
under `selection.txt`'s default lines. An FFT the tuner found computing wrong answers is only ever run when `-fft`
names it, and then with a warning.


## Subcommands

A word at the start of the settings chooses a subcommand instead of a tuning run. All but `accuracy` open no GPU,
so they work on a machine without one, and read the files in the run directory (`-dir`).

Where the database holds measurements from more than one GPU, driver or build, the subcommands act on the one
measured with the kernels of the binary you are running. `env=<id>` names another one; see [Envs](#envs-one-set-of-measurements-per-card-driver-and-build).

### `-tune scope`

What a tuning run would tune for: the workload, the probe, the grid of exponents `T` is taken over, their weights, and
`T` with what the database holds so far (or estimates only, where nothing is measured).

```sh
prpll -tune scope
prpll -tune scope,workload=330M-340M
```

Takes `workload=`, `probe=`, `probeWeight=`, `kinds=` and `env=`, and the search's `strategy=`, `maxPermute=`,
`maxPoints=`, `comboTop=` and `comboTiers=`. It changes nothing, and without a database it creates none. With a
database it also says, group by group, how many steps the search takes from one best set of each FFT type (see
[`maxPoints=`](#how-to-search)); without one, which card the database is for is not known, and a run says it as it
starts instead.

```text
tune: pending work read from no worktodo file
tune: workload 100000000-400000000 (the default range, there being no pending work)
tune: probe 199999991 (the geometric centre of the range, the prime at or below 200000000), carrying 50% of the weight
tune: prp grid: 65 exponents, spread across the range
tune:   100000000   0.8%    728.591 us/it  prior, from 1K:11:256
...
tune:   199999991  50.0%   1522.063 us/it  prior, from 1K:11:512  (probe)
tune:   ... and 53 more
tune: T = 1595.272 us/it with nothing measured, 0.0% of the weight on measured entries
```

### `-tune status`

Where the measurements stand: which GPU and build they are for, whether a run is in progress and what it is measuring,
the latest run and its settings, `T` and how much of the workload is measured, per FFT type how many FFTs are measured
and how many are not, what the accuracy checks made of the published entries, the next few measurements a run would
take, and any configurations that crashed the GPU or computed a wrong answer.

```sh
prpll -tune status
prpll -tune status,stop=1%        # what would a run with stop=1% still do?
```

```text
tune: status: env 1, Tesla P100-PCIE-16GB (nvidia,cuda,cc600,pdl); tunedb.txt last written 101 min ago
tune: status: nothing holds the database
tune: status: the latest session on env 1 is session 2, a run started 2026-09-26 17:55
tune: status: valued as session 2's run, over the 0 assignments pending when it started: workload=67000000-80000000,probe=67513549,probeWeight=0,kinds=prp+ll,bootstrap=1,strategy=hybrid,maxPermute=all,maxPoints=all,comboTop=3,comboTiers=3,contenders=16,roundCalls=16,stop=0
tune: status: T 1103.939 us/it, 100.0% of the weight on measured entries
tune: status: a run started now would begin in defaults sweep: 2 of 464 FFTs read at the built-in defaults + halving: round 1 of 4, 16 contenders, 0 of 256 calls
tune: status: FFT64: 172 of 3394 entries measured, 462 not, 2760 waiting on the halving
...
tune: status: accuracy: 19 entries published (0 exact arithmetic, 7 confirmed, 12 unvalidated); 0 held below the table's reach, 0 raised above it; 0 limits for options no entry runs; 0 sets owed a reading, 0 rejected
tune: status: next, as a run started now would rank them:
tune: status:   1. baseline 256:15:512:002 ll short32 at the built-in defaults: by rule, ~23 s
tune: status:   2. probe 1K:4:512:102 ll short32 Placement INPLACE=0: by rule, ~24 s
tune: status:   3. baseline 256:15:512:102 ll short32 at the built-in defaults: by rule, ~23 s
tune: status:   4. probe 1K:4:512:102 ll short32 Width SHUFL_BYTES_W=4: by rule, ~24 s
tune: status:   5. baseline 256:15:512:200 ll short32 at the built-in defaults: by rule, ~23 s
tune: status:   and 3574 more worth running, 0 worth nothing
tune: status:   and up to 102345 more not listed yet
```

It is safe to use while a run is in progress (whether a run holds the database can only be told on Linux). It
evaluates the queue with the latest run's settings and the pending work that run started with; any setting given
replaces that run's (`workload=` or `probe=` replaces both), so you can see what a different run would do next. The
"next" list is what a run started now would do; a run in progress may differ slightly, having learnt things it has not
yet written down.
It takes `env=`, and every setting a run takes except `tunetxt=` and `dashboard=`.

### `-tune emit`

Rewrite `selection.txt` from `tunedb.txt`. A run already does this after every measurement, so it is only needed after
`reset`, `adopt` or `compact`, to get `tune.txt` (`emit,tunetxt=1`), or to recreate a deleted `selection.txt`.

```sh
prpll -tune emit
prpll -tune emit,tunetxt=1
```

The default `use` lines are chosen at the probe exponent (the best searched FFT published there, for each type), so
`emit` has to use the same probe as the run did. It derives it from the worktodo, as the run did; if your worktodo has changed since, give the
run's `probe=` (and `workload=`), which `-tune status` shows. Takes `workload=`, `probe=`, `probeWeight=`, `kinds=`,
`tunetxt=` and `env=`.

### `-tune reset`

Forget what was measured, so that the next run starts again: every measurement, every accuracy reading, and every
configuration recorded as failing or as having crashed the GPU. `fft=<spec>` limits it to one FFT (e.g.
`reset,fft=1K:13:256:212`); this is how to let the tuner retry a configuration that crashed the GPU once a driver
update has fixed it.

```sh
prpll -tune reset
prpll -tune reset,fft=1K:13:256:212
```

It is refused while a run is using the database. Run `-tune emit` afterwards to update `selection.txt`. The FFT the
drift anchor uses cannot be reset on its own; reset everything instead. Takes `fft=` and `env=`.

### `-tune adopt`

Measurements belong to the kernels they were measured with (see [Envs](#envs-one-set-of-measurements-per-card-driver-and-build)).
When you update PRPLL and its kernels have changed, the old measurements are no longer used. If you know the kernel
changes do not matter for your GPU, `adopt` takes the old measurements over as the new kernels' own:

```sh
prpll -tune adopt                  # this GPU's latest measurements, as the current kernels' own
prpll -tune adopt,from=1           # env 1's measurements, into its card's env under the current kernels
prpll -tune adopt,from=1,into=3
```

```text
tune: env 2 has taken over the rows of env 1
tune: adopt rewrote tunedb.txt
```

Then run `-tune` as usual: it carries on from where the adopted measurements left off. The old drift-anchor readings
were taken under the old kernels and are not kept, so the run may first spend a few minutes choosing its anchor again.

Without `from=`, `adopt` takes the most recent earlier env of the GPU. With no GPU open it cannot tell which card it is
running on, so where nothing has been measured under the current kernels yet and the database holds measurements from
more than one GPU (or driver, or backend), it lists them and asks for `from=<id>`. It is refused between different GPUs, drivers or backends, and between envs whose drift
anchors are different configurations. Run `-tune emit` afterwards to update `selection.txt`, or just start `-tune`.
Takes `from=<id>` and `into=<id>` (or `env=<id>`).

### `-tune compact`

Rewrite `tunedb.txt` with repeated measurements of one configuration merged into one line, and without the option sets
no measurement uses any more. It changes no result, only the size of the file. Refused while a run is using the
database. Takes nothing.

### `-tune accuracy`

For development: on the GPU, read the rounding error of every value of every `-use` option against the option set it
moved from, on each FFT type's bootstrap FFT at the probe (or on `fft=<spec>`), and report which options change the
rounding. The readings are recorded in `tunedb.txt`. `groups=<Group>+<Group>...` limits it to some option groups
(`Placement`, `Middle`, `Memory`, `Queues`, `Tail`, `Width`, `Height`, `Arith`, `Cuda`). Takes `workload=` and `probe=`
as a run does. A normal tuning run does not need it.


## Files

All in the run directory (`-dir`):

| File | Written by | Holds |
| --- | --- | --- |
| `tunedb.txt` | tuning runs, `reset`, `adopt`, `compact` | every measurement, with the exact options it was taken under. Appended to as a run goes. |
| `tunedb.txt.lock` | tuning runs and the subcommands that write | an empty file that is locked while the database is being written, so that two processes cannot write it at once |
| `selection.txt` | tuning runs (after every measurement), `emit` | what normal runs use: the default `use` lines, then one `entry` and `opts` line per published configuration, `limit` lines for options no entry runs that were measured to hold only to a lower exponent, and `exclude` lines for configurations that computed wrong answers. Replaced atomically. |
| `tune.txt` | tuning runs and `emit` with `tunetxt=1`; also `-oldtune` | the FFT list older binaries read, holding only FFTs measured as safe to their full standard range under default rounding |

`tunedb.txt` is worth keeping: it is the only record of the measurements, and deleting it means tuning from scratch.
`selection.txt` can always be recreated from it with `-tune emit`. Neither is meant to be edited by hand.

A `selection.txt` looks like this:

```text
# prpll selection v1
# written 1790139162 by v8.0-... from tunedb.txt env 1; T=933.8 over workload 118000000-119000000
use   INPLACE=0,LOADS=23004,SHUFL_BYTES_H=16
entry 7867c39c478e5237 933.832 512:13:512:202 prp 68157436 123603520 unvalidated
opts  7867c39c478e5237 INPLACE=0,LOADS=23004,SHUFL_BYTES_H=16
entry da60d95dd13fe95c 937.532 512:13:512:212 prp 68157436 124312360 unvalidated
opts  da60d95dd13fe95c INPLACE=0,LOADS=23004,SHUFL_BYTES_H=16
...
```

The `use` lines are the default options: a plain `use` line for every FFT, and `use ! <type>` lines for one FFT type
where the types disagree (`-` means none). Each `entry` gives a cost in microseconds per iteration, the FFT, the test
kind, the exponent range it serves, and the state of its accuracy evidence, and its `opts` line the options it was
measured with (`-` for the built-in defaults), naming every option the `use` lines set at the value it was measured
with, so that it runs as measured whatever they say: `n/a` (an NTT, which has no rounding error), `confirmed` (the rounding error was read
at the top of the range and found comfortably safe, or the range was cut to where it is), `unvalidated` (read at the
top of PRPLL's standard range and found above the level production warns at, but with less margin than `confirmed`
asks for; nothing more is owed, and the range is the one PRPLL itself would use), or `unavailable` (too few rounding errors occurred to
judge; only configurations that round exactly as the defaults do are published this way).


## Envs: one set of measurements per card, driver and build

A timing is only comparable with timings taken on the same GPU, with the same driver, backend (OpenCL or CUDA) and
kernel code. `tunedb.txt` therefore files each measurement under an *env*: a GPU model and its PCI slot where the
backend reports it, the driver version, the backend, a few capabilities (and whether `NO_ASM` was set), and a
fingerprint of the kernel source PRPLL was built with and of how the tuner times a call.

- **Updating the driver, switching backend, or moving to another card** starts a new env. Its measurements start from
  nothing: tune again.
- **Updating PRPLL** starts a new env only when the kernel source or the way a call is timed changed; changes to the
  rest of the program do not. If the kernels changed, either tune again or, if you are confident the change does not
  affect your GPU, use [`-tune adopt`](#-tune-adopt). If the timing changed, tune again: the old readings and the new
  ones do not compare.
- Measurements from other envs stay in the file and are simply not used. `-tune reset,env=<id>` drops them.

Where the subcommands cannot tell which env you mean, they list the envs in the database with their ids, so that you
can name one with `env=<id>`.


## Several GPUs

Tune each GPU in its own run directory, the one its normal runs use:

```sh
prpll -dir ~/prpll/gpu0 -device 0 -tune
prpll -dir ~/prpll/gpu1 -device 1 -tune
```

Both can run at once. Two GPUs of the same model are told apart by PCI slot where the backend reports it (not under
NVIDIA's OpenCL); even so, keep one directory per GPU, since each directory has one `selection.txt`. Two tuning runs
cannot share one directory: the second is refused while the first holds the database.

With `-pool`, a normal run looks for `selection.txt` in its own directory first, then in the pool directory. So for
several identical GPUs, you can tune one and copy its `selection.txt` into the pool directory for the others.


## Other tools: `-measure` and `-options`

**`-options [<fft>]`** lists every `-use` option this build knows, with its values and default resolved for this GPU
and the given FFT (default `512:15:512`), and how the tuner groups them. Use it to see what an option name means or
which values are allowed. It also checks the option table and exits with status 1 if the check fails.

**`-measure <fft>[,<setting>...]`** times one FFT repeatedly, exactly as the tuner times a configuration, and reports
whether its error bar is honest: how far readings move between runs compared with what each run's own spread predicts,
along with the cost of building it, its residue and its rounding error. It is a diagnostic for a GPU whose timings are
unstable, and records its readings in `tunedb.txt`. The options measured are those `-use` on the command line and
`config.txt` give (`-measure 1K:13:256 -use TAIL_KERNELS=2`); `selection.txt` is not read. Settings: `n=<calls>` (default 8),
`blocks=<per call>`, `block=<iterations>`, `exp=<E>` (default: the `-prp` exponent, else the top of the FFT's range),
`kind=prp|ll`, `anchor=<fft>` (time a second FFT alternately and correct for its drift), `roe=0|1` (read the rounding
error), `drift=0|1` (time the drift anchor; on by default), `drain=0|1`.


## The previous tuner: `-oldtune`

`-oldtune` is the tuner PRPLL had before (called `-tune` in upstream PRPLL), kept for comparison. It takes the same
options as before (`noconfig`, `inplace`, `fp64`, `ntt`, `nofp32`, `fp6431`, `minexp=`, `maxexp=`, `quick=`; see
`prpll -h`). It writes its option choices into `config.txt` and its FFT list into `tune.txt`.

The two do not mix well: the `-use` lines `-oldtune` adds to `config.txt` override everything `-tune` measured (see
[How a normal run uses the result](#how-a-normal-run-uses-the-result)). If you have used it, remove those lines
(marked `# New settings based on a -tune run` and `# These settings were slightly faster in a -tune run`) before
relying on `-tune`. A `-tune` run lists the `config.txt` keys that would override its results.

Giving one of the old tuner's options to `-tune` is refused with a message naming `-oldtune`.


## Exit statuses

| Status | Meaning |
| --- | --- |
| 0 | finished normally, including a run stopped with Ctrl-C |
| 1 | failed (for example, the database could not be read, or nothing could be published) |
| 2 | the command line could not be understood; nothing was done |
| 3 | the GPU was lost and could not be recovered by restarting |


## Questions

**How long should I let it run?** Until it stops by itself if you can; otherwise as long as you can spare. Progress is
front-loaded: the baselines of the FFTs that matter and the bootstrap come first, and each later measurement is chosen
because it is expected to help the most. `-tune status` shows how much is still worth measuring.

**My worktodo moved to a different exponent range. Do I start again?** No. Run `-tune` again: the new range is taken
from the worktodo, the measurements that still apply are kept, and only what is new is measured. If the probe
exponent changed, the bootstrap searches each type's fastest FFT there first, since the default options are chosen at
the probe.

**Can I keep running tests while tuning?** Not on the same GPU: anything else running on it distorts the
measurements. Other GPUs are unaffected.

**Does tuning touch my results or assignments?** No. It reads the worktodo files and never changes them.

**Why is an FFT I expected missing from `selection.txt`?** Either something cheaper covers the same exponents, it was
not measured yet (see `-tune status`), it failed an accuracy check, or it computed a wrong answer while being measured
and was excluded.
