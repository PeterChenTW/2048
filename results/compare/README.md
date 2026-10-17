# Step-3 comparison: 15-minute short trainings (2026-10-17 19:19-19:34 Taipei)

All arms: from scratch, training seed 101, 15-minute wall-clock budget, 2 torch threads each, 4 arms in parallel on an 8-CPU box.
Eval for every arm: `eval.py --games 100 --seed-start 1000 --board fast` (greedy, seeds 1000-1099).
Arms B-D use the new step-based epsilon: linear 0.15 -> 0.02 over 150k env steps (`--eps-schedule linear_steps --eps-min 0.02 --eps-decay-steps 150000`).

| arm | code | episodes | env steps | grad updates | loss first50 -> last50 | NaN |
|---|---|---|---|---|---|---|
| A control | original e1a034d (ReLU Q, max-target, gamma 0.9, 128 single-sample updates per episode, eps 0.15*0.999/replay), FastBoard, `tools/control_train_e1a034d.py` | 1,169 | 171,216 | 133,120 | 0.0101 -> 0.0109 | 0 |
| B step2-legacy | step-2 fixes, legacy conv net, n=1 | 2,199 | 318,928 | 79,701 | 0.0109 -> 0.0077 | 0 |
| C s3-onehot-dueling | 16ch log2 one-hot + (1,2)/(2,1) conv + dueling, n=1 | 928 | 156,760 | 39,159 | 0.0052 -> 0.0030 | 0 |
| D s3-onehot-dueling-n3 | as C, 3-step returns | 1,125 | 163,652 | 40,881 | 0.0271 -> 0.0347 | 0 |

| policy | mean | median | max | >=512 | >=1024 | >=2048 |
|---|---|---|---|---|---|---|
| baseline old ckpt (old board) | 7,737.52 | 7,446 | 16,952 | 77% | 18% | 0% |
| A control | 2,448.92 | 2,194 | 6,380 | 10% | 0% | 0% |
| B step2-legacy | 1,616.80 | 1,456 | 4,472 | 1% | 0% | 0% |
| C s3-onehot-dueling | 2,069.28 | 1,794 | 6,364 | 3% | 0% | 0% |
| D s3-onehot-dueling-n3 | 1,575.00 | 1,432 | 3,280 | 0% | 0% | 0% |

Plots: `learning_curve_episodes.png`, `learning_curve_steps.png` (raw + MA100, all arms), `loss_curve.png`. Regenerate with `python tools/plot_compare.py`.

Reading (single seed, 15 min, so noisy):
- No arm is near the baseline, which was trained far longer. 15 minutes is a smoke-level comparison, not a verdict.
- C (one-hot + dueling) has the best training curve per env step (MA100 ~2,000 vs ~1,500-1,700) and the lowest loss, at ~half the steps/s of the legacy net.
- A's greedy eval is highest, but A performs 133k single-sample updates vs ~40-80k batched updates (128 per update) for B-D, so arms differ in gradient compute per env step; its eval uses the original ReLU-output network.
- n-step 3 (D) did not help at this budget; its loss rises (larger, higher-variance targets).
- Next (step 4): longer runs (hours), 2-3 seeds per arm, more updates per env step (replay-every 1-2), then decide.
Checkpoints are on the box only (not committed): wt-s3/model/cmp_*.
