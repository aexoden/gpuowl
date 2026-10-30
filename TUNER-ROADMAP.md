# Tuner synthesis: review and roadmap

The tracker for turning the three tuner branches into one tuner on a stable base. Tick items here as they land, with the
commit that did it, so the next session can see what is and is not done.

- Base: the `model2` branch, kept in sync with upstream (`gwoltman/master`) by cherry-picking.
- Reference branches, not developed further for now: `tune` (the first tuner) and `model` (`model2` without its last 29
  commits).
- Last reviewed: 2026-10-30.

## Status

### 1. Sync upstream onto model2

- [x] Cherry-pick upstream's 11 commits after `0e2bf4aa` (`bde9aa08` to `267c5a45`), as `40a1bd19` to `1bf09825`.
      `1f7afba9` was already here as `60611a65`.
- [x] `81de954a` (here `4df5182a`): `bestFits`/`bestFit` take both `ll` (which lines of the tune file an LL test reads)
      and upstream's `errorChecked` (whether a too-small forced `-fft` may run); `-fft` takes the settings of its
      tune-file line holding E.
- [x] `carryutil.cl` back to upstream's (`f74c2bae`): `bde9aa08` made `bar(WG)` a full barrier above a warp, which
      fixes the same `updateStats` hang the branch's change did.
- [x] Option table (`f74c2bae`): `SLEEP_WAIT` fixed (never searched; the user's own value stays in force while tuning);
      `ENABLE_SUBBARRIER` searched on CUDA builds on nVidia (values 0 and 1, and 2 at `WMUL=4`; inert without
      sub-workgroups wider than a warp); `ENABLE_BARSYNC` and `USE_REGISTER_BARSYNC` deprecated, as upstream removed
      both. The same two removed from the `-use` key list in `Gpu.cpp`. `ENABLE_SUBBARRIER` sits after `WMUL` and
      `TAIL_KERNELS` in the table (`40f804c3`), since a walk in table order must meet a key's parents first.
- [x] `make DEBUG=1 check` (GPU-free unit tests and source checkers) passes: 260 tests, locally and in CI from
      `40f804c3`.
- [x] CUDA build compiles (checked without a toolkit, against NVIDIA's `cuda.h` and `nvrtc.h` from PyPI; not linked
      against a real `libcuda`).
- [x] CUDA build run on a real card (recipe below), 2026-10-30, on a card with slow FP64: the proof verified, and
      all six PRP runs read `de62d6db1ad5092d` at 200000 iterations.
- [x] OpenCL build run on the same card: the proof verified, the PRP residues matched.
- [x] A short `-tune` on each build: both ran fine (2026-10-30).

GPU validation recipe, on a cloud card with the CUDA toolkit in `/usr/local/cuda`:

```sh
git clone -b model2 https://github.com/aexoden/gpuowl && cd gpuowl
make -j CUDA=1                                    # build-cuda/prpll
mkdir run && cd run
../build-cuda/prpll -verify ../test/786433-10.proof            # end-to-end arithmetic check
# 512-wide NTT FFTs, so that WMUL=4 is allowed (from width 1024 up WMUL is capped at 2) and nothing needs FP64:
# 2:512:4:512 is FP32+M61 (up to 72.7M), 1:512:4:512 is M31+M61 (up to 83.2M).
P="-prp 57885161 -iters 200000 -log 20000"
# Each run starts fresh: a PRP resumes from the savefiles in ./57885161/ otherwise.
for F in 2:512:4:512 1:512:4:512; do
  rm -rf 57885161; ../build-cuda/prpll $P -fft $F              # Gerbicz-checked PRP; note the residues
  rm -rf 57885161; ../build-cuda/prpll $P -fft $F -use ENABLE_SUBBARRIER=1
  rm -rf 57885161; ../build-cuda/prpll $P -fft $F -use ENABLE_SUBBARRIER=2,WMUL=4
done
../build-cuda/prpll -tune budget=30m                           # a short tune on the new base
```

A PRP residue does not depend on the FFT, so the residues at each log step must match across all six PRP runs, and
no `EE` lines may appear (`de62d6db1ad5092d` at 200000 iterations). On a card with fast FP64, `-fft 512:8:512` adds
the FP64 path.

### 2. Shared fixes upstream (the user sends these)

At each sync, drop model2's copy of any fix upstream has merged and take upstream's.

- [ ] Width trig offsets with radix 8 at one 256 pass (`aba32f4f`, `Gpu.cpp` `sharedH`)
- [ ] A pinned 32-bit carry caps FFT61, FFT31 and FFT32 (`FFTConfig::maxBpw`)
- [ ] `MM_CHAIN` and `MM2_CHAIN` default independently (`36db12ca`, `cl/fft-middle.cl`)
- [ ] CUDA graph slots split by LL and PRP (`fb4d4a43`)
- [ ] Context-lost tracking and CUDA sticky errors (`928ef5e4`, `8e6b447e`, `0a884453`, `793038d7`)
- [ ] Release the persisting L2 set-aside when a Gpu is destroyed (`193fbf7e`)
- [ ] Exit statuses 2 and 3, restart by re-exec (`69ac2820`)

### 3. Correctness gaps

- [x] A failed Gerbicz check during a soak goes through `checkedTwice` and is recorded as an error, as a failed call
      is (`80042c14`).
- [x] LL soaks check their residue: their chunks are call-sized and end where the LL reference does (`80042c14`).
- [x] Port tune's accuracy reference rule, since reworked on measurement (the slack and rule items below): a set
      that moves a rounding-changing key serves up to where it reads at least 93% of the z of the same set at default
      rounding read at the table top, and never below the floor. Where that reading failed its check or is below the
      floor, the floor alone holds; a set that rounds visibly where the reference hardly rounds is never trusted. One
      verdict (`Gate.h` `verdictFor`) serves publishing, the duels and the contender table.
- [x] Port `-tune accuracy` and the `roeStat` sample fingerprint: `roe` rows carry the samples' fingerprint (an
      eleventh field; older rows still read), and `-tune accuracy[,fft=<spec>][,groups=<Group>+...]` reads every
      value of every key against the set it moved from, bit for bit, and says where the option table has a key
      otherwise. Register limits are not swept. Resumes from what `tunedb.txt` holds.
- [x] Run `-tune accuracy` on a card of each kind available (it reads one FFT of each type in scope, or the one
      `fft=` names), and correct the option table's `accuracyImpact`, `defaultRounding` and `roundsAs` from what it
      reports; upstream `62117cea` changed FP32 rounding since the flags were measured. Done on nVidia; AMD below.
  - RTX A4000 (CUDA), `fft=2:512:4:512` (FP32+M61, variants 212 and 202) at 72729143, 2026-10-30: all 44 keys
    read, value by value, as the table has them. Defaults read z 38.80 (212) and 35.14 (202), max 0.2500 on both.
    What changes the rounding: `TABMUL_CHAIN32=1` (-18.36 Z), `TAIL_KERNELS` 0, 1 and 11 (worst -2.41 at 0),
    `TAIL_TRIGS32` 0 and 1 (worst -1.51 at 1).
  - Tesla P100 (CUDA), one FFT of each type in scope (1K:11:512 FP64, 2:1K:16:256 FP32+M61, 4:1K:8:256
    FP32+M31+M61, 51:1K:16:256 FP64+M31), every variant, 2026-10-30: every key as the table has it, but for
    `256RADIX8_H`, which the table had as suspected and which changes the rounding at both values; now `Yes`.
    `TAIL_KERNELS` 1 and 11 read identical gaps at all five exponents on both cards, as `roundsAs` has it.
  - Tesla P100 (OpenCL), `workload=60M-150M,repeat=5` (1K:10:256 FP64, 2:1K:8:256, 4:1K:4:256, 51:1K:8:256), every
    variant, 2026-10-30: every key as the table has it, `FAST_BARRIER` (OpenCL only) included; `256RADIX8_H` changes
    the rounding here too.
  - RTX A4000 (CUDA), 2:256:16:256:212: `256RADIX8_W` changes the rounding at both values (radix 8 read 4.76 the
    more accurate there); now `Yes`.
  - Not yet read: `FUSE_WEIGHT_BUTTERFLY`, which is offered on AMD alone, and anything on an AMD or Intel card.
- [x] `-tune accuracy,repeat=<n>`: every move that changed the rounding is read again beside its background at the
      primes just below, n exponents in all, and the report gives each move's gaps and each background's own z, with
      mean and sd. Progress lines give each new reading's distinct sample values and how many sit at the max.
- [x] Decide the slack against measured noise. `repeat=5`, 2026-10-30, gap mean and sd over five exponents:

      | Card, FFT | Move | Mean | sd |
      | --- | --- | --- | --- |
      | A4000, 2:512:4:512:212 (z 37.8) | `TABMUL_CHAIN32=1` | -16.09 | 1.35 |
      | | `TAIL_KERNELS=0` | -0.61 | 1.70 |
      | | `TAIL_KERNELS=1`, `=11` | -0.55 | 1.68 |
      | | `TAIL_TRIGS32=0` | -0.00 | 0.78 |
      | | `TAIL_TRIGS32=1` | -0.66 | 0.85 |
      | P100, 1K:11:512:212 (z 47.9) | `TABMUL_CHAIN=1` | -4.73 | 2.01 |
      | | `TAIL_KERNELS=0` | +0.10 | 1.76 |
      | | `TAIL_KERNELS=1`, `=11` | +0.03 | 1.77 |
      | | `TAIL_TRIGS=0` | +0.59 | 1.62 |
      | | `TAIL_TRIGS=1` | -0.44 | 1.67 |
      | P100, 1K:11:512:000 (z 32.7) | `MM_CHAIN=1` | +6.55 | 0.77 |
      | | `MM2_CHAIN=1` | -0.20 | 1.32 |
      | | `MM2_CHAIN=2` | +0.67 | 1.48 |

      The defaults' own z over the same exponents: sd 0.90 at z 37.8, 2.01 at z 47.9, 0.63 at z 32.7. One gap's sd
      is 0.8 to 2.0 Z, about 3.5% of z, so 0.5 Z failed about a third of the sets that cost nothing. Decided: within
      7% of the reference's z (about two sds at any z). `MM_CHAIN=1` is reliably more accurate, so it can make up
      for another key's cost in the same set; tune's "MM2_CHAIN=1 costs 2.1" was most likely one noisy reading.

      The P100's OpenCL sweep agrees. Across both cards and builds, about 150 gaps of keys that cost nothing
      (`TAIL_KERNELS`, `TAIL_TRIGS`, `TAIL_TRIGS32`, `MM2_CHAIN`) read no further below than 6.2%; gap sds ran 0.14
      to 2.1 Z, lower where the two readings move together. The real costs, as a share of the reference's z:
      `TABMUL_CHAIN32=1` -30 to -43%, `256RADIX8_H=1` -10 to -16% (sd as low as 0.14), `TABMUL_CHAIN=1` -6 to -10%;
      `MM_CHAIN=1` gains 14 to 23% everywhere it was read.
- [x] The reference rule, reworked: a set that changes the rounding was compared with its reference at each exponent
      the gate read, so a uniform cost beyond the slack rejected it at every exponent (`256RADIX8_H=1`,
      `TABMUL_CHAIN32=1`). Decided: hold it to its reference's z at the table top instead, the accuracy the default
      rounding is trusted with. z falls some 8 (FP64 at 18 bpw) to 20 (FP32 hybrids at 34 bpw) per 1% of exponent
      near the top, by the rule of thumb that errors double every half bit per word, so those settings keep all but
      about a percent of the range. The reference is read once, at the top.
- [x] Share rounding readings between sets that round alike (tune `roundingOf`), on nVidia only (CUDA and OpenCL,
      both swept): a set's readings include every passing reading of a set of the FFT whose `roundingOf` is the same,
      but not at an exponent where its own check failed. AMD and Intel read every set on its own (`sharesRounding`).
- [x] Run `-tune accuracy` on the OpenCL build: the P100, above.
- [ ] Sweep an AMD card (weeks away): `FUSE_WEIGHT_BUTTERFLY` and AMD's compiler; then let `sharesRounding` take it.
- [ ] `REACH_RESOLUTION` is 1% of the exponent, coarse beside z falling 8 to 20 per 1% near the top: a set held a
      little below its reference loses up to 1% of the range where 0.2% would do. Measure z against the exponent on a
      card before changing it.
- [ ] Publish an exclusion list of configurations that computed wrong answers, and have production honour it.
- [ ] Re-gate lines carried from an earlier build when the build changed rounding.
- [x] Hold a configuration that took the card down at every exponent and test kind, and withdraw it from publication
      (lines carried from earlier builds included), not only the (exponent, kind) it crashed at. A running process's
      own open attempt is the call in flight, not a death; `-tune status` beside a run treats the newest session so.
- [x] Failures go to `failures/` in the run directory: `failures/failures.log` holds one entry per configuration that
      took the device down, computed a wrong answer or would not build, with the command line that reproduces it;
      a failed build's log and the CUDA source and PTX dumps are saved beside it under unique names instead of
      overwriting each other in the run directory. Each tune run, and `-tune status`, says how many configurations
      are held out and points at the log; a run adds any entry the log lacks (a hard fault leaves no process to
      write one).
- [ ] A fault while only the heater runs (between calls) is not pinned on the configuration it heats: the heater
      holds no attempt record, and any record the main thread writes meanwhile would close one.

### 4. Output format

- [ ] Write the v2 lines to `selection.txt` under a first line of their own; production reads it before `tune.txt`.
- [ ] Refuse tune's `# prpll selection v1` file whole (tune's binary already refuses ours).
- [ ] Write `tune.txt` only on request, holding only full-reach, default-rounding lines.

### 5. Cost policies and production feedback

- [ ] Make cost interpretation a named, recorded setting: plain (mean + 2 SE), power floor, soak calibration; duels on
      or off.
- [ ] Defaults: duels on, `soak=0`; the power floor's default waits on the `-measure` results, which showed it does not
      work right on the two cards tried.
- [ ] Soak calibration per FFT type and kind at most, or reported without being applied; rank a soak group's members
      by their within-group ratios; let a duel stand unless both were soaked in the same group.
- [ ] `tune-feedback.txt` written whenever production runs from our `selection.txt`; record the worker count and leave
      shared stretches out of ratios; cap or rotate the file.

### 6. Strategies

- [ ] `search=classic`: port tune's `Classic.cpp` (upstream's own procedure) onto Bench, Readings and Publish.
- [ ] Compare `model`, `staged` and `classic` on the prpll-sim scenarios and on real cards with equal budgets, against
      production feedback.

Default strategy: `search=model`. `staged` moves one option at a time except by copying a type leader's best set and
through restarts, and has no long apples-to-apples comparison yet.

### 7. Smaller ports from tune

- [ ] `-tune reset` (clear a configuration that crashed the card once a driver fixes it). More pressing now that a
      death holds a configuration out everywhere: today the only way back is editing `tunedb.txt`.
- [ ] Validate `-use` keys against the option table (OR it with upstream's list in `Gpu.cpp`)
- [ ] `-options` flag (`dumpOptionSpace` exists)
- [ ] Note when `config.txt` keys shadow published settings
- [ ] Production fallback when a published line will not build on the device
- [ ] Warn when more workers share L2 than the entries were timed with
- [ ] `-measure` restarts after a lost device
- [ ] A `TUNING.md` for the synthesis, based on tune's

### 8. Defaults from data

- [ ] Once production feedback exists per card class, keep the strategy and cost policy whose published costs sit
      closest to what production sustained.

## Known issues

- `the_loop_publishes_what_the_simulated_card_rewards` (`tests/test_Tuner.cpp`) depends on its seed: within its 80
  items the model loop sometimes stops at a configuration 4.17% behind the planted optimum (650.3 against 624.3 us/it).
  Release builds, seeds 1-8, 2026-10-30: `b61ecb40` reached the optimum on 6 of 8 (seeds 1 and 2 stopped short),
  `40f804c3` on 7 of 8 (seed 5 stopped short). The default seed, 7, passes on both. Any change to the option table
  moves the draws, so a change can turn the default seed red without making the search worse: sweep seeds
  (`SEED=<n> build-release/prpll-test the_loop_publishes_what`) before concluding either way.

- One rounding reading's z carries sampling noise of about 2 to 4% of itself: z is (0.5 - mean) / beta + 0.577 with
  beta from the samples' sd, whose relative standard error over n = 2150 Gumbel samples is about sqrt(1.1 / n),
  2.3%, and the `repeat=` readings agree (above). A gap between two roundings is noisier still, as if they were
  hardly correlated. The relative slack allows for it. Accuracy readings in general still take one reading per
  exponent; at the floor that matters little, since z falls fast with the exponent there.
- FP32 hybrid samples are quantized: of 2150 samples, the FP32 hybrids' take 30 to 105 distinct values (A4000 and
  P100, CUDA and OpenCL), the FP64 and FP64+M31 ones 2146 to 2150. uF is a float near nF * M61, so uF / M61 is
  resolved only to a power-of-two step that grows with |nF|; the maxima sit on steps of 1/256 to 1/16 (0.2500,
  0.3125, 0.3555 = 91/256, 0.4375 = 7/16). z there is coarser than its digits suggest, and its tail can be wrong:
  one A4000 reading of `TABMUL_CHAIN32=1` at 2:512:4:512:212 (z 21.56) had a max of 0.4375 where its neighbours'
  were 0.32 to 0.34, a sample the Gumbel fit puts at about 1e-4 in 2150. Whether a Gumbel z is the right measure for
  those types, and whether their floor of 6 means anything, is a question for upstream.
- The Intel B570 hangs rather than running (it was also the desktop's display card); a project of its own.

## Decisions

| Question | Decision |
| --- | --- |
| File for the v2 lines | `selection.txt`, with a header of its own |
| Default strategy | `search=model`; `staged` stays opt-in |
| Production feedback | On whenever production runs from our `selection.txt` |
| Upstream fixes | The user sends them |
| `ENABLE_SUBBARRIER` | Searched, on CUDA builds only |
| Branches | Build on `model2`; `tune` and `model` decided later |
| Power floor (`powerHold`) | Open: `-measure` on two cards showed it does not work right |
| A configuration that took the card down | Held at every exponent and kind, and withdrawn; every failure logged in `failures/` |
| Accuracy slack for sets that change rounding | Within 7% of the z of the same set at default rounding (was 0.5 Z, as tune, until `repeat=` measured the noise) |
| Accuracy reference rule | Against the reference's z at the table top, not at the same exponent: a cost lowers the reach instead of rejecting |
| Shared rounding readings | On nVidia (CUDA and OpenCL swept); AMD and Intel once swept |

## Review (2026-10-30)

### Branches

`model` and `model2` share no history with master: they start from a squashed root (`ef954247`), which is upstream at
`0e2bf4aa` (2026-10-23) plus tune's fixes outside the tuner plus a rewritten tuner. `model2` is `model` plus 29 commits.

| | tune | model | model2 |
| --- | --- | --- | --- |
| History | Ordinary fork; last merged master at `ecdc8d2a` (2026-10-19) | Orphan, 21 commits | `model` + 29 commits |
| Tuner size (lines added) | src 23.7K, tests 19.9K | src 14.7K, tests 5.9K | src 20.1K, tests 7.7K |
| Search | Value-ranked queue, successive halving, bootstrap, `mode=classic` | Posterior model with Thompson sampling, lanes | `model`'s, plus `search=staged`, duels, soaks |
| Production reads | `selection.txt`; `tune.txt` only with `tunetxt=1` | `tune.txt` v1, `tune-ll.txt` | `tune.txt` v2 (kind, regime, range per line) |
| Build integration | Edits Makefile, CMakeLists.txt, vcxproj | `tune.mk`, `tune.cmake`, `PRPLL.tune.props` | same as `model` |
| GPU-free tests | 645 pass | not run | 260 pass |

### What tune has that model2 lacked at review time

1. Accuracy reference rule (roadmap 3). model2 carries the table's `accuracyImpact`, `defaultRounding` and `roundsAs`
   but nothing reads them; a set passes if its own Z clears production's danger floor (20 for FP64, 6 otherwise).
2. Output older binaries cannot misread (roadmap 4). An upstream binary parses model2's v2 `tune.txt`, drops kind,
   regime and range, and runs a lowered-reach line up to the table maximum.
3. Wrong-answer exclusions that production honours (roadmap 3).
4. `-tune accuracy`, reach under user overrides and `limit` records, shared rounding readings, `mode=classic`,
   `-tune reset`/`adopt`/`compact`, production fallbacks, `-options`, `TUNING.md` (roadmap 3, 6, 7).

Already ported intact: the second reading of a failed Gerbicz check, the LL reference two FFTs agree on, attempt records
that pin a fault on its configuration, the restart cap, database locking, regime-aware calls, the option table rules.
tune divides costs by the drift anchor; model2 deliberately stopped (`7ddd0709`), with split-half data showing the
division doubled disagreement.

### What model and model2 add over tune

Costs derived at read time from raw rows (so a cost policy can be re-applied to an existing database); carried lines
from earlier builds; separate build files; prpll-sim; heater, `compileAhead` and the bounded in-memory kernel cache;
duels; the drift anchor as a monitor only; `search=staged`; the conditional option space; the contender table and
`-tune explain`; one line per FFT, kind, regime and range; the whole known frontier published; `budget=` and the audit;
`tune-report.json` and `tag=`; spill records; `-measure` with telemetry (Linux only); power rows on every call.

### Soak and production feedback findings

Soak slices are `soak` rows in `tunedb.txt`. The separate file is `tune-feedback.txt`, appended by every normal run.

- A failed Gerbicz check in a soak slice is filed as "unsupported": no second reading, no error row.
- LL soaks have no check.
- One env-wide median ratio rescales every unsoaked reading and the model's observations, across types and kinds;
  each new soak moves it and can re-rank the database.
- Soaked costs pool slices across groups and sessions, losing the within-group pairing; two soaked configurations skip
  the duel ranking.
- Soak is on by default (`soak=10%`).
- Feedback is written by every run, tuned or not, without the worker count, and grows without limit.

### Upstream since the forks

Upstream added 21 non-merge commits after tune's last merge, 11 of them after model's base. Kernel changes start a new
env for every tuning database. The ones that touch tuner code: `bde9aa08` (`ENABLE_SUBBARRIER`; removes
`USE_REGISTER_BARSYNC` and the `ENABLE_BARSYNC` path), `af8b0b67` (`SLEEP_WAIT`), `81de954a` (`-fft` lists and
`errorChecked`), `1f7afba9` (in-memory kernel cache, already in model2), and before model's base `18ca4299`/`8083d9c5`
(`CS_FENCE`), `21f0dbad`/`994acfcf` (`CS_ATOMIC`), `62117cea` (FP32 weights, may change FP32 rounding) and `9e3ff659`
(removes `M61_SM120`).

Because model2 has no merge base with master, each sync is a cherry-pick. Making a commit whose tree is model2's and
whose parent is `0e2bf4aa` (`git commit-tree`) would restore a merge base; deferred until the branches are decided.
