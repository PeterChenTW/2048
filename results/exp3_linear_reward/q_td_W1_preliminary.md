# Checkpoint Q / target / TD diagnostics

- Target weights are not saved: checkpoint network is BOTH online and target; these TD errors do not reconstruct historical training TD errors.
- Training replay in RL.py uses PER-weighted mean squared TD loss (not Huber), RMSprop, and clip_grad_norm_ (default threshold 1.0).
- Gradient diagnostics use uniformly sampled minibatches and unweighted mean squared TD loss, without optimizer steps. They do not reproduce PER sampling or importance weights. Fraction clipped uses threshold 1.0; the checkpoint threshold and its corresponding fraction are also recorded.
- States come from fresh epsilon-greedy FastBoard episodes, not the training replay buffer or restart pool. Each checkpoint induces its own state distribution.
- Returns honor checkpoint n_step and gamma; reward statistics describe individual environment rewards. The final partial episode is truncated at --states and bootstraps normally. Complete-episode and partial-episode statistics are separate.
- max_a Q(s) is over all four actions (unmasked); action selection and bootstrap targets are masked to legal actions.
- Per-step reward profile: neither the training CSV nor the checkpoint stores per-step rewards or the replay buffer, so rewards are sampled from fresh GREEDY (epsilon=0) episodes of each checkpoint (--reward-games, 1 thread). For each step the turn merge score m is recorded and the reward is computed with the training formulas (rewards.py / fast_board.py): log = log2(1+m)/16, linear = m/reward_scale, and -1 in both modes when no tile spawned. "own" uses the checkpoint reward mode; "as_log" and "as_linear" re-score the same trajectory with each formula.

| label | reward_mode | gamma | Q p5/p50/p95 | \|TD\| p50/p95/p99 | frac \|TD\|>1 | grad-norm p50/p95 | frac clipped | reward p50/p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BASE_s101 | log | 0.99 | 3.3411/5.3349/5.9813 | 0.15543/0.78861/1.3749 | 2.8% | 6.6263/9.7779 | 100.0% | 0.14512/0.34324 |
| BASE_s202 | log | 0.99 | 3.2333/5.1139/5.8424 | 0.16972/0.89444/1.5048 | 3.8% | 6.4861/8.3983 | 100.0% | 0.14512/0.33485 |
| E3-LIN99_s101 | linear | 0.99 | 1.4197/1.8474/2.3263 | 0.047293/0.43336/1.003 | 1.0% | 1.2837/2.3574 | 70.0% | 0.015625/0.15625 |
| E3-LIN99_s202 | linear | 0.99 | 1.143/1.7748/2.2824 | 0.059766/0.55031/1.0669 | 1.2% | 1.0267/2.2088 | 65.0% | 0.015625/0.15625 |

## Per-step reward (greedy rollouts)

| label | mode | games / steps | greedy mean score | reward | mean | p10 | p50 | p90 | p99 | frac 0 | frac -1 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| BASE_s101 | log | 10 / 4080 | 5913 | own (log) | 0.1615 | 0 | 0.1451 | 0.3153 | 0.4468 | 26.9% | 0.0% |
| BASE_s101 | log | 10 / 4080 | 5913 | as_log | 0.1615 | 0 | 0.1451 | 0.3153 | 0.4468 | 26.9% | 0.0% |
| BASE_s101 | log | 10 / 4080 | 5913 | as_linear | 0.05661 | 0 | 0.01562 | 0.125 | 0.5502 | 26.9% | 0.0% |
| BASE_s202 | log | 10 / 2744 | 3545 | own (log) | 0.1561 | 0 | 0.1451 | 0.3153 | 0.4436 | 28.1% | 0.0% |
| BASE_s202 | log | 10 / 2744 | 3545 | as_log | 0.1561 | 0 | 0.1451 | 0.3153 | 0.4436 | 28.1% | 0.0% |
| BASE_s202 | log | 10 / 2744 | 3545 | as_linear | 0.05047 | 0 | 0.01562 | 0.125 | 0.5312 | 28.1% | 0.0% |
| E3-LIN99_s101 | linear | 10 / 2192 | 2624 | own (linear) | 0.04677 | 0 | 0.01562 | 0.1234 | 0.5156 | 28.8% | 0.0% |
| E3-LIN99_s101 | linear | 10 / 2192 | 2624 | as_log | 0.1522 | 0 | 0.1451 | 0.3141 | 0.441 | 28.8% | 0.0% |
| E3-LIN99_s101 | linear | 10 / 2192 | 2624 | as_linear | 0.04677 | 0 | 0.01562 | 0.1234 | 0.5156 | 28.8% | 0.0% |
| E3-LIN99_s202 | linear | 10 / 1867 | 2026 | own (linear) | 0.0424 | 0 | 0.01562 | 0.09375 | 0.5 | 29.8% | 0.0% |
| E3-LIN99_s202 | linear | 10 / 1867 | 2026 | as_log | 0.1468 | 0 | 0.1451 | 0.2902 | 0.4382 | 29.8% | 0.0% |
| E3-LIN99_s202 | linear | 10 / 1867 | 2026 | as_linear | 0.0424 | 0 | 0.01562 | 0.09375 | 0.5 | 29.8% | 0.0% |

| label | m 0-0 | m 1-16 | m 17-128 | m 129-1024 | m 1025-inf |
|---|---:|---:|---:|---:|---:|
| BASE_s101 | 26.9% | 56.2% | 15.4% | 1.5% | 0.0% |
| BASE_s202 | 28.1% | 55.5% | 15.0% | 1.3% | 0.0% |
| E3-LIN99_s101 | 28.8% | 55.7% | 14.4% | 1.1% | 0.0% |
| E3-LIN99_s202 | 29.8% | 56.0% | 13.4% | 0.9% | 0.0% |

Episode scores use score_v2 (merge points). Means across all collected episode segments include the final partial episode; completed means exclude it.

- BASE_s101: 4000 steps; 4000 transitions; 13 complete / 14 collected episodes; mean score / steps 3647.71 / 285.71; completed mean score / steps 3731.69 / 291.77; 20 gradient batches; 0 optimizer updates; all metrics finite; 21.6s.
- BASE_s202: 4000 steps; 4000 transitions; 18 complete / 19 collected episodes; mean score / steps 2447.79 / 210.53; completed mean score / steps 2530.00 / 216.00; 20 gradient batches; 0 optimizer updates; all metrics finite; 13.8s.
- E3-LIN99_s101: 4000 steps; 4000 transitions; 21 complete / 22 collected episodes; mean score / steps 1978.36 / 181.82; completed mean score / steps 2026.67 / 185.33; 20 gradient batches; 0 optimizer updates; all metrics finite; 19.9s.
- E3-LIN99_s202: 4000 steps; 4000 transitions; 23 complete / 24 collected episodes; mean score / steps 1808.33 / 166.67; completed mean score / steps 1865.22 / 171.04; 20 gradient batches; 0 optimizer updates; all metrics finite; 5.5s.
