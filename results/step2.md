# Step 2 short-training check (2026-10-17, Taipei)

Code: step-2 fixes (linear Q output, masked Double DQN, act/target masking, PER IS weights beta 0.4->1.0, batched replay, gamma=0.99 from config.py, main.py train/play CLI).
Each run: `python main.py train --minutes 15 --threads 2`, replay every 4 env steps, batch 128.

| run | init | episodes | env steps | updates | loss first50 -> last50 | Q mean range | Q min..max | NaN |
|---|---|---|---|---|---|---|---|---|
| step2_ft | old ckpt | 1,387 | 331,420 | 82,824 | 0.00910 -> 0.00814 | 1.64 .. 4.46 | -5.47 .. 11.99 | 0 |
| step2_scratch | fresh | 2,279 | 369,016 | 92,223 | 0.0369 -> 0.00890 | -0.06 .. 5.49 | -4.28 .. 12.38 | 0 |

Eval: `python eval.py --games 100 --seed-start 1000 --board fast` (greedy).

| policy | mean | median | max | >=512 | >=1024 | >=2048 |
|---|---|---|---|---|---|---|
| baseline old ckpt (old board) | 7,737.52 | 7,446 | 16,952 | 77% | 18% | 0% |
| baseline old ckpt (fast board) | 7,755.48 | 7,388 | 16,444 | 77% | 20% | 0% |
| step2_ft (15 min) | 2,996.16 | 3,054 | 6,540 | 12% | 0% | 0% |
| step2_scratch (15 min) | 2,193.12 | 2,082 | 6,260 | 4% | 0% | 0% |

Reference only: 15 minutes is far too short. Fine-tuning drops below baseline because the old weights were
trained with a ReLU on the Q output and gamma 0.9, so the Q-value scale and targets shift under the fixed code.
Epsilon decays per update (0.999) and hits the 0.001 floor within ~100 episodes, so exploration is minimal; to be revisited in step 3/4.
