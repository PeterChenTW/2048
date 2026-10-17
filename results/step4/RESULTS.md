# Step 4 results: 3 h active training, eval.py 100 games (seeds 1000-1099, fast board)

Target: mean > 15,000 and reach-1024 >= 80%. Baseline (old checkpoint, 63k episodes): mean 7,737.52, reach-1024 18%.

## Per run, final snapshot

| run | t (min) | env steps | updates | updates/step | mean | median | max | reach_512 | reach_1024 | reach_2048 | target met | vs baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_s201 | 180 | 3,393,145 | 1,711,104 | 0.504 | 5,404.36 | 5,128 | 14,824 | 54% | 7% | 0% | no | -2,333 |
| A_s202 | 180 | 4,098,874 | 1,647,104 | 0.402 | 6,459.40 | 5,592 | 16,056 | 56% | 17% | 0% | no | -1,278 |
| C_s201 | 180 | 597,351 | 465,834 | 0.780 | 3,583.84 | 3,300 | 10,948 | 23% | 1% | 0% | no | -4,154 |
| C_s202 | 180 | 590,055 | 460,143 | 0.780 | 4,248.64 | 3,564 | 12,864 | 38% | 2% | 0% | no | -3,489 |

## Per arm, final snapshot: mean across seeds [min, max]

| arm | mean | median | max | reach_512 | reach_1024 | reach_2048 |
|---|---|---|---|---|---|---|
| A | 5,931.88 [5,404.36, 6,459.40] | 5,360 [5,128, 5,592] | 15,440 [14,824, 16,056] | 55% [54%, 56%] | 12% [7%, 17%] | 0% [0%, 0%] |
| C | 3,916.24 [3,583.84, 4,248.64] | 3,432 [3,300, 3,564] | 11,906 [10,948, 12,864] | 30% [23%, 38%] | 2% [1%, 2%] | 0% [0%, 0%] |

## Per arm, best snapshot per run (by mean): mean across seeds [min, max]

| arm | mean | median | max | reach_512 | reach_1024 | reach_2048 |
|---|---|---|---|---|---|---|
| A | 5,931.88 [5,404.36, 6,459.40] | 5,360 [5,128, 5,592] | 15,440 [14,824, 16,056] | 55% [54%, 56%] | 12% [7%, 17%] | 0% [0%, 0%] |
| C | 4,015.28 [3,781.92, 4,248.64] | 3,475 [3,386, 3,564] | 11,502 [10,140, 12,864] | 34% [30%, 38%] | 2% [2%, 2%] | 0% [0%, 0%] |

Best snapshot times: A_s201 t180, A_s202 t180, C_s201 t150, C_s202 t180

## Eval history (every 30 min)

| run | t | mean | median | max | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| A_s201 | 30 | 2,777.92 | 2,484 | 8,124 | 17% | 0% | 0% |
| A_s201 | 60 | 2,192.68 | 2,112 | 5,352 | 4% | 0% | 0% |
| A_s201 | 90 | 4,509.48 | 3,872 | 14,700 | 38% | 3% | 0% |
| A_s201 | 120 | 3,753.40 | 3,326 | 10,452 | 29% | 2% | 0% |
| A_s201 | 150 | 4,459.40 | 4,008 | 8,912 | 42% | 0% | 0% |
| A_s201 | 180 | 5,404.36 | 5,128 | 14,824 | 54% | 7% | 0% |
| A_s202 | 30 | 2,815.52 | 2,582 | 8,220 | 16% | 0% | 0% |
| A_s202 | 60 | 4,499.36 | 3,728 | 10,036 | 39% | 0% | 0% |
| A_s202 | 90 | 3,781.68 | 3,362 | 8,280 | 26% | 0% | 0% |
| A_s202 | 120 | 5,727.24 | 5,482 | 15,000 | 57% | 10% | 0% |
| A_s202 | 150 | 6,048.04 | 6,074 | 16,140 | 59% | 5% | 0% |
| A_s202 | 180 | 6,459.40 | 5,592 | 16,056 | 56% | 17% | 0% |
| C_s201 | 30 | 1,780.68 | 1,568 | 3,192 | 0% | 0% | 0% |
| C_s201 | 60 | 1,826.60 | 1,514 | 6,300 | 4% | 0% | 0% |
| C_s201 | 90 | 2,358.08 | 2,322 | 5,912 | 5% | 0% | 0% |
| C_s201 | 120 | 3,079.44 | 2,970 | 9,856 | 15% | 1% | 0% |
| C_s201 | 150 | 3,781.92 | 3,386 | 10,140 | 30% | 2% | 0% |
| C_s201 | 180 | 3,583.84 | 3,300 | 10,948 | 23% | 1% | 0% |
| C_s202 | 30 | 1,825.28 | 1,666 | 5,276 | 2% | 0% | 0% |
| C_s202 | 60 | 1,915.64 | 1,660 | 3,792 | 0% | 0% | 0% |
| C_s202 | 90 | 2,731.96 | 2,644 | 6,360 | 11% | 0% | 0% |
| C_s202 | 120 | 3,262.20 | 3,076 | 7,192 | 22% | 0% | 0% |
| C_s202 | 150 | 3,445.00 | 3,220 | 12,332 | 21% | 1% | 0% |
| C_s202 | 180 | 4,248.64 | 3,564 | 12,864 | 38% | 2% | 0% |

Curves: step4_eval_curves.png, step4_train_curves.png. Checkpoints (not committed): /workspace/rl2048-review/step4/runs/<run>/{final.pt,snapshots/t*.pt,state*}.

## Notes
- No run met the target (mean > 15,000 and reach-1024 >= 80%); none beat the baseline 7,737.52 (63k-episode checkpoint). Both arms were still rising at 3 h.
- Per-step update alignment was NOT achieved: C held 0.78 updates/env step; A's native 128 updates/episode fell to 0.50 / 0.40 as episodes lengthened. Per wall time A made ~3.6x more updates (1.65-1.71M vs 0.46M) and ~6x more env steps (3.4-4.1M vs 0.59M), because A's small network is much cheaper per step.
- Per env step, C learns faster (C ~3.9k mean at 0.59M steps; A was ~2.2k-4.5k at 0.8-1.0M steps), so C's deficit is a throughput (compute per step) problem, not stagnation.
- 0 NaN (C); no freezes (skipped_freeze_s = 0 for all runs). Done 2026-10-17 ~22:50 Taipei.
