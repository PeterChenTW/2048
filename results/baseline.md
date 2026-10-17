# Baseline (before DQN revamp)

Date: 2026-10-17

## Checkpoint

model/nn_prioritize_replay_no_invalid_gamma_099_2 (DQN + target net + PER, raw tile-value input; ~63k training episodes per result/ plot).

## Old board (game_2048.Board), greedy, 100 games, seeds 1000–1099

| policy | games | mean | median | max | >=512 | >=1024 | >=2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DQN greedy | 100 | 7,738 | 7,446 | 16,952 | 77% | 18% | 0% |
| random (valid moves only) | 100 | 1,129 | 1,152 | 3,496 | 0% | 0% | 0% |

Max-tile distribution for DQN: 64:1, 128:2, 256:20, 512:59, 1024:18; random: 32:6, 64:36, 128:49, 256:9.

Note: score = score_v2 (sum of merged tile values); invalid moves masked; old get_invalid_moves consumes the global RNG.

## New board (fast_board.FastBoard)

Same checkpoint, same seeds (1000–1099), `python eval.py --board fast [--policy random]`.

| policy | games | mean | median | max | >=512 | >=1024 | >=2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DQN greedy | 100 | 7,755 | 7,388 | 16,444 | 77% | 20% | 0% |
| random (valid moves only) | 100 | 1,127 | 994 | 2,944 | 0% | 0% | 0% |

Max-tile distribution for DQN: 128:2, 256:21, 512:57, 1024:20; random: 32:1, 64:48, 128:40, 256:11.

`python eval.py --board old` reproduces the old-board table above exactly (mean 7,737.52, median 7,446, max 16,952).

### Larger sample: 1000 games, seeds 1000–1999, DQN greedy

| board | games | mean | median | max | >=512 | >=1024 | >=2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old (game_2048.Board) | 1000 | 7,984 | 7,520 | 20,932 | 77.4% | 21.1% | 0.1% |
| new (FastBoard) | 1000 | 8,237 | 7,818 | 25,548 | 80.1% | 23.9% | 0.5% |

Per-game score std is about 3,200–3,600, so the standard error of the difference of two 1000-game means is about 150; the +253 gap (~1.7 SE) is not statistically significant.

### Throughput benchmark

`python bench.py --seconds 10` (training/eval workload: get_invalid_moves + random valid move + move, reset on game end; 8-vCPU box, single thread, Python 3.13):

| board | steps/s |
| --- | --- |
| old Board | 5,274 |
| FastBoard | 127,707 |
| speedup | 24.2x |

(An earlier run gave 5,323 vs 130,851 = 24.6x.) Greedy DQN eval is dominated by the network forward pass, so the 100-game eval only drops from 22.8 s to 11.6 s.

### Notes on differences

- Rules are identical: `tests/test_fast_board_equivalence.py` checks move/merge/score/changed on 20,000 random boards x 4 actions, invalid moves and game_over, 500 full random-action games with per-step RNG-state sync (grid, score, score_v2, turns, invalid_move, reward, done, spawn position), 1,000 initial boards and the 90/10 spawn ratio.
- The only behavioral difference is RNG consumption: the old `get_invalid_moves()` deep-copies the board and calls `move()` on each copy, which spawns tiles using the global `random` module, so every legality check advances the RNG by several draws. `FastBoard.get_invalid_moves()` is pure and draws nothing, and each board owns its own `random.Random(seed)`. With the same seed, the spawn sequence therefore diverges after the first move, so per-seed games differ while the distribution is the same; the 100- and 1000-game means agree within noise.
- Legacy quirks are kept on purpose: `score` is the sum of spawned tiles (use `score_v2` for the real 2048 score), and `done` is only set when the last empty cell gets filled by a spawn.
