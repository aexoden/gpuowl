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
- [ ] CUDA build run on a real card (recipe below).
- [ ] OpenCL build run on a real card (recipe below, with `build-release/prpll`).

GPU validation recipe, on a cloud card with the CUDA toolkit in `/usr/local/cuda`:

```sh
git clone -b model2 https://github.com/aexoden/gpuowl && cd gpuowl
make -j CUDA=1                                    # build-cuda/prpll
mkdir run && cd run
../build-cuda/prpll -verify ../test/786433-10.proof            # end-to-end arithmetic check
# A 512-wide FFT, so that WMUL=4 is allowed (from width 1024 up WMUL is capped at 2)
P="-prp 57885161 -fft 512:8:512 -iters 200000 -log 20000"
../build-cuda/prpll $P                                         # Gerbicz-checked PRP; note the residues
../build-cuda/prpll $P -use ENABLE_SUBBARRIER=1
../build-cuda/prpll $P -use ENABLE_SUBBARRIER=2,WMUL=4
../build-cuda/prpll -tune budget=30m                           # a short tune on the new base
```

The residues at each log step must match across the three PRP runs, and no `EE` lines may appear.

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

- [ ] A failed Gerbicz check during a soak goes through `checkedTwice` and is recorded as an error (today
      `classify()` files it as "unsupported" and nothing marks the configuration).
- [ ] LL soaks check their residue (`timeItersLL` has no check).
- [ ] Port tune's accuracy reference rule: a set that moves a rounding-changing key must read within 0.5 Z of the same
      set at default rounding, or its reach is cut (tune `Gate.cpp` `standardFor`, `accuracyReference`).
- [ ] Port `-tune accuracy` and the `roeStat` sample fingerprint; re-measure the option table's accuracy flags on
      current kernels (upstream `62117cea` changed FP32 rounding).
- [ ] Share rounding readings between sets that round alike (tune `roundingOf`).
- [ ] Publish an exclusion list of configurations that computed wrong answers, and have production honour it.
- [ ] Re-gate lines carried from an earlier build when the build changed rounding.
- [ ] Hold a configuration that took the card down at every exponent, not only the (exponent, kind) it crashed at.

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

- [ ] `-tune reset` (clear a configuration that crashed the card once a driver fixes it)
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
