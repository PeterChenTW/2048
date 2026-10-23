# 2048 Reinforcement Learning

An AI for the game 2048. The project started as a small DQN with a pygame GUI. It went through several rounds of DQN experiments and ended with a tabular **n-tuple network trained with afterstate TD(0)**, played with **expectimax search**. That approach is the current best.

All numbers in this README come from result files committed in this repo, and each section names its source file. Scores are the standard 2048 score (sum of merged tile values). Unless stated otherwise, evaluation uses fixed eval seeds starting at 1000, so different methods play the same games.

## Current best method: n-tuple + afterstate TD(0) + expectimax

- **Value function:** an n-tuple network, a lookup-table linear approximator, so it is not a neural network. It is trained by afterstate TD(0) with gamma = 1, reward = game score, alpha = 0.0025 and greedy action selection. Training runs for 60 minutes on a single core per seed, which works out to about 1.23–1.26 M games per seed (source: `results/ntuple/NTUPLE_RESULTS.md`).
- **Search:** exact expectimax. Max nodes take max over actions of reward + value. Chance nodes average over every empty cell, with a 2 at probability 0.9 and a 4 at 0.1. Leaves take the n-tuple V(afterstate), and a terminal position scores 0. There is no sampling and no pruning; the only speed-up is an exact transposition table. d = number of player moves searched, and d1 is identical to greedy play: per-step boards are byte-identical to the original evaluator.
- **No retraining for search.** The same t60 snapshots (`s1_t03600.bin`, `s2_t03600.bin`) are used at every depth.

### Results (source: `results/ntuple/EXPECTIMAX_RESULTS.md`, `results/ntuple/expectimax/summary.json`)

Games per arm: d1–d3 1000 (eval seeds 1000–1999), d4 200 (1000–1199), d5 100 (1000–1099).

| seed | d | games | mean (95% CI) | median | ≥8192 | ≥16384 | ms/move |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1000 | 107,915 [105,388, 110,441] | 121,776 | 60.6% | 0.0% | 0.0009 |
| 1 | 2 | 1000 | 137,801 [136,026, 139,576] | 143,618 | 92.4% | 1.5% | 0.0238 |
| 1 | 3 | 1000 | 160,828 [157,894, 163,761] | 151,146 | 96.7% | 14.4% | 0.3836 |
| 1 | 4 | 200 | 178,812 [169,990, 187,635] | 162,142 | 96.5% | 24.5% | 5.0172 |
| 1 | 5 | 100 | 199,893 [185,428, 214,358] | 167,070 | 97.0% | 38.0% | 65.9220 |
| 2 | 1 | 1000 | 101,971 [99,341, 104,601] | 112,644 | 54.1% | 0.1% | 0.0009 |
| 2 | 2 | 1000 | 138,580 [136,667, 140,494] | 144,284 | 90.1% | 2.1% | 0.0261 |
| 2 | 3 | 1000 | 163,359 [160,338, 166,379] | 154,460 | 97.6% | 15.3% | 0.4166 |
| 2 | 4 | 200 | 178,872 [170,340, 187,404] | 157,178 | 98.0% | 24.5% | 5.4455 |
| 2 | 5 | 100 | 190,626 [176,605, 204,646] | 157,166 | 96.0% | 34.0% | 71.1732 |

No arm reached 32768. ms/move for d1–d3 was measured single-process over 100 games. For d4/d5 it was taken from the parallel eval itself (8 processes, nice 19, shared CPU), so treat it as an order of magnitude only.

**Paired differences** (same eval seeds, mean ± 1.96·sd/√n):

| comparison | seed 1 | seed 2 |
|---|---|---|
| d3 − d1 | +52,913 [+48,990, +56,836] | +61,388 [+57,395, +65,380] |
| d4 − d3 | +16,792 [+5,743, +27,841] | +13,666 [+2,748, +24,583] |
| d5 − d3 | +41,996 [+23,904, +60,087] | +24,898 [+8,145, +41,651] |
| d5 − d4 (seeds 1000–1099) | +25,008 [+4,430, +45,587] | +10,579 [−10,117, +31,275] |

### Recommendation: d = 4

- d4 is clearly better than d3 on both seeds; neither CI includes 0.
- d5 over d4 is not established. On seed 2 the CI includes 0 (n = 100).
- d5 is about 13× slower per move than d4 (65.9 vs 5.02 ms on seed 1, 71.2 vs 5.45 ms on seed 2).

### The corner strategy is learned, not programmed

The agent keeps its largest tile in a corner, and nothing in the code tells it to. The training code (`ntuple/train.c`, `ntuple/ntuple.h`, `ntuple/common.h`) has no corner, monotonicity, snake or empty-cell features, no reward shaping, no search and no imitation. The only inputs are the game rules and the score. Corner metrics are computed only in the evaluators, as diagnostics, and the training code never reads them. The search in `ntuple_expectimax/expectimax.c` also uses only n-tuple values; its corner and monotonicity functions just write diagnostic CSV columns.

Corner behaviour emerges during training (source: `results/ntuple/NTUPLE_RESULTS.md` §3; seed 1, 1000 games per snapshot). Here "corner≥256" is the share of steps with the max tile in a corner, counted once the max tile is ≥256:

| training time | 5 s | 15 s | 30 s | 1 min | 8 min | 15 min | 60 min |
|---|---|---|---|---|---|---|---|
| mean score | 11,261 | 18,152 | 29,125 | 39,946 | 73,291 | 88,638 | 107,915 |
| corner≥256 | 0.4% | 8.3% | 46.6% | 60.4% | 74.1% | 80.8% | 84.0% |

A random policy scores 2.1% on the same metric. Seed 2 follows the same curve (§5.4 of the same file).

### Demo

`demo/expectimax/expectimax_demo.html` is a single offline HTML file. It shows result charts and four replayed games (seed 1 eval seeds 1813, 1713, 1364; seed 2 eval seed 1523), and each replay can switch between d1, d3, d4 and d5. These four eval seeds lie outside the d4/d5 statistics range, so the replays are illustrations only. Screenshots are `demo/expectimax/screenshot_mobile_412.png` and `screenshot_desktop_1440.png`; older versions are kept as `*_v1`/`*_v2`.

## DQN experiments (history)

Every DQN arm below was rejected or superseded. Code and results live on the experiment branches (see [Branches](#branches)). Unless noted, evaluation is greedy, 100 games, eval seeds 1000–1099.

**Baseline** (`results/baseline.md`, branch `dqn-revamp`): the original checkpoint `model/nn_prioritize_replay_no_invalid_gamma_099_2` (DQN + target net + PER, ~63k episodes) scores mean 7,738, ≥1024 18%. Over 1000 games it scores 7,984 on the old board and 8,237 on the new FastBoard. FastBoard has identical rules and runs about 24× faster than the old board (127,707 vs 5,274 steps/s).

**Step 2–3, 15-minute smoke runs** (`results/step2.md`, `results/compare/README.md`): fixing the DQN (linear Q output, masked Double DQN, PER importance-sampling weights) and trying one-hot + dueling and 3-step returns. No arm came near the baseline in 15 minutes (best 2,449). 3-step returns did not help.

**Step 4, 3 h active training** (`results/step4/RESULTS.md`): arm A (original control) reached a mean of 5,932 across 2 seeds with ≥1024 12%. Arm C (one-hot + dueling) reached 3,916 with ≥1024 2%. Neither met the target (mean > 15,000 and ≥1024 ≥ 80%) or beat the baseline. C learned more per env step but was limited by throughput.

**Round 1, five pre-registered experiments** (`results/experiments_STATUS.md`, branch `exp/qtd-analysis`/`exp/round2-vectorize`). BASE = arm A settings, 90 min active, seeds 101/202 (303 added). The rule is per seed: mean ≥ 1.2× BASE and ≥1024 +10pp.

| exp | change | result |
|---|---|---|
| #1 afterstate (E1-AS) | afterstate V head | **rejected** (2/3 seeds pass; s101 ×1.20 but ≥1024 only +5pp). Kept as new control C. |
| #2 capacity × n-step | replay size / n-step 3 | rejected (×0.93, ×1.00) |
| #3 linear reward × gamma | linear reward, gamma 0.99/0.997 | rejected (lin/log ×0.44–0.88) |
| #4 conv2x2 | 2×2 conv trunk | rejected (×2.3 updates/s but mean 2,774 vs 5,249 and 1,691 vs 3,095) |
| #5 restart pool | late-game restarts | rejected (s101 3,934 vs 5,249) |

Afterstate long runs (180 min) reached 7,898 (≥1024 27%; best snapshot t150 9,796 / 44%) on s101 and 10,349 (49%) on s202.

**Round 2** (pre-registered in `research/round2/REPORT.md`; results in `results/round2/` on `exp/integrate` and `results/round2_skip/` on `exp/round2-skip`):

| arm | change vs C | result |
|---|---|---|
| E | ε = 0 throughout | rejected: s101 ×0.96 / −3pp, s202 ×1.10 / +5pp, s303 stopped for futility |
| R | replay ratio 0.25 | rejected: s303 11,275 vs 9,746 (×1.16 / +13pp), the rest stopped for futility |
| S | `--value-skip l1` | rejected: ×0.50–0.72 on all 3 seeds, S also made only ~0.41× as many updates |
| V (vectorized TD target) | speed only | not trained: combined microbench speed-up ×1.05 was under the ×1.10 gate (commit 095338f) |

**Round 3** (`results/round3_sym_gamma/ROUND3_RESULTS.md`, branch `exp/round3-sym-gamma`; 180 min, compared with C t180):

| arm | change | seeds passing | result |
|---|---|---|---|
| G | raw reward, gamma 1.0, h(x) value rescale | 0/3 | rejected (×0.63–0.83) |
| Y | random D4 symmetry augmentation | 1/3 | rejected (×1.47, ×1.19, ×1.03) |
| GY | both | 0/3 | rejected (×0.72–0.93) |

The best DQN result was about 12k mean (Y s202: 12,317, ≥1024 65%), still below the 15,000 / 80% target. The n-tuple player reached about 100k mean (≥1024 ≥ 99%) after 60 minutes on one core, so the project moved to n-tuple + expectimax.

## Usage

### Requirements
- n-tuple / expectimax: a C compiler (gcc) on Linux, using only libc and libm.
- DQN branches: Python 3 with the packages in `requirements.txt` on those branches (torch, numpy, pygame, matplotlib, seaborn, pandas, pytest).

### n-tuple training (this branch, `ntuple/`)
```bash
cd ntuple
nice -n 19 make            # builds train, eval, record, test_ntuple
nice -n 19 make test       # rule / n-tuple unit checks
mkdir -p snapshots
# usage: ./train SEED TOTAL_SECONDS OUTDIR   (reported runs: SEED 1 and 2, 3600 s)
nice -n 19 ./train 1 3600 snapshots
# usage: ./eval SNAPSHOT|random SEED_START N OUT_CSV   (greedy, no learning)
nice -n 19 ./eval snapshots/s1_t03600.bin 1000 1000 s1_t60.csv
```
Snapshots are written at 5 s, 15 s, 30 s, 1, 2, 4, 8, 15, 30 and 60 min of training time (≈135 MB each, not committed). The pre-registered design is in `ntuple/NTUPLE_PREREG.md`. The helper scripts in `ntuple/` (`run_evals*.sh`, `summarize.py`, `monitor.py`, `throughput.py`) are copied unchanged from the original runs and contain absolute paths from that machine.

### Expectimax evaluation (this branch)
```bash
cd ntuple_expectimax
nice -n 19 gcc -O3 -march=native -o expectimax expectimax.c -lm
# usage: ./expectimax SNAPSHOT DEPTH SEED_START N OUT_CSV [DUMP_FILE K]
nice -n 19 ./expectimax path/to/s1_t03600.bin 4 1000 200 s1_d4.csv
python3 summarize.py          # rebuild the results tables
```
`run_all.sh` (d2/d3 + timing) and `run_d45.sh` (d4/d5) are the scripts used for the reported runs. They expect the snapshots under `../../../corner/ntuple/`. **Snapshot `.bin` files are not committed** (≈135 MB each); their sha256 hashes are in `ntuple_expectimax/snapshots.sha256`.

### Demo
Open `demo/expectimax/expectimax_demo.html` in any browser; it needs no server. To regenerate it, run `python3 build.py` in `demo/expectimax/`. That reads `template.html`, `traj_*.txt` and the summary JSON. `build.py` reads `results/ntuple/expectimax/summary.json` relative to its own location. `traj_dump.c` produced the replay trajectories.

### DQN training / evaluation (branches `dqn-revamp` … `exp/round3-sym-gamma`)
```bash
git checkout exp/round3-sym-gamma
pip install -r requirements.txt
python main.py train --help   # e.g. --arch onehot_afterstate --minutes 90 --budget-clock active --seed 101
python main.py play [--checkpoint PATH]           # pygame GUI
python eval.py --checkpoint PATH --games 100 --seed-start 1000 --board fast
```
The original single-script version (`python main.py`, which trains or plays with the old DQN) is what this branch still contains in `main.py`, `RL.py` and `game_2048.py`.

## Directory structure (this branch)

```
├── main.py, RL.py, game_2048.py   original DQN + pygame game (legacy)
├── model/, result/                 original DQN checkpoint and training plot
├── ntuple/                         n-tuple TD(0) training (train.c), eval, tests, Makefile, PREREG
├── ntuple_expectimax/              expectimax search (expectimax.c), n-tuple headers, eval.c,
│                                   run scripts, summarize.py, snapshots.sha256
├── results/ntuple/
│   ├── NTUPLE_RESULTS.md           n-tuple TD training results (seeds 1 and 2)
│   ├── EXPECTIMAX_RESULTS.md       expectimax d1–d5 results and paired CIs
│   ├── expectimax/                 summary.json, summary_table.md, logs, timing
│   └── rerun_s2/                   seed 2 rerun logs and eval summary
├── demo/expectimax/                offline HTML demo, build script, replays, screenshots
└── PROGRESS.md                     run log for this branch
```

## Branches

| branch | content |
|---|---|
| `main` | original project (legacy DQN) |
| `dqn-revamp` | DQN rewrite (FastBoard, eval.py, CLI), baseline, step 2–4, research REPORT |
| `s3-encoder-dueling`, `s3-nstep`, `s3-integrate` | step 3: one-hot + dueling, n-step, 15-min comparison |
| `exp/afterstate-gen`, `exp/afterstate-head`, `exp/afterstate` | round 1 #1 afterstate |
| `exp/replay-nstep` | round 1 #2 |
| `exp/linear-reward` | round 1 #3 |
| `exp/conv2x2` | round 1 #4 |
| `exp/restart-pool` | round 1 #5 |
| `exp/eps-episode`, `exp/active-clock` | per-episode ε schedule and active-time budget options |
| `exp/qtd-analysis` | Q/TD quantile analysis tool, round 1 status |
| `research/round2` | round 2 research report (pre-registration) |
| `exp/round2-vectorize` | V: TD-target speed-ups (not trained, gate missed) |
| `exp/integrate` | round 2 integration, C/E and R results |
| `exp/round2-skip` | round 2 S (value skip) |
| `exp/round3-sym-gamma` | round 3 G/Y/GY |
| `exp/ntuple-results` | n-tuple TD results |
| `exp/ntuple-expectimax` | **this branch**: expectimax search on the n-tuple network, demo |
