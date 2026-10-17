"""Seeded, headless evaluation of greedy DQN and random 2048 policies."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
import types

import numpy as np
import torch

DEFAULT_CHECKPOINT = 'model/nn_prioritize_replay_no_invalid_gamma_099_2'
ALL_MOVES = ['left', 'down', 'right', 'up']


def make_board(board, seed):
    """Construct a seeded board without requiring pygame for legacy evaluation."""
    if board == 'fast':
        from fast_board import FastBoard
        return FastBoard(rng=random.Random(seed))
    if board == 'old':
        # pygame is only needed by Game's GUI, never by Board. Restore the
        # module table after import so this stub cannot affect later GUI imports.
        if 'game_2048' not in sys.modules:
            previous = sys.modules.get('pygame')
            sys.modules['pygame'] = types.ModuleType('pygame')
            try:
                legacy = importlib.import_module('game_2048')
            finally:
                if previous is None:
                    sys.modules.pop('pygame', None)
                else:
                    sys.modules['pygame'] = previous
        else:
            legacy = sys.modules['game_2048']
        return legacy.Board(4)
    raise ValueError('board must be fast or old')


def load_agent(checkpoint):
    """Load legacy state dictionaries, including checkpoints saved on a GPU."""
    from RL import DDQNAgent
    agent = DDQNAgent(16, 4)
    try:
        agent.load(str(checkpoint))
    except Exception:
        # RL.load has neither map_location nor weights_only compatibility.
        try:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        except TypeError:  # older PyTorch
            state = torch.load(checkpoint, map_location='cpu')
        agent.model.load_state_dict(state)
    agent.model.cpu().eval()
    return agent


def evaluate(*, checkpoint=DEFAULT_CHECKPOINT, policy='greedy', games=100,
             seed_start=1000, board='fast', max_steps=5000):
    """Return JSON-compatible summary statistics and individual game records.

    Reach rates are fractions in [0, 1]. Steps count actual calls to move().
    """
    if policy not in ('greedy', 'random') or board not in ('fast', 'old'):
        raise ValueError('unknown policy or board')
    if games < 1 or max_steps < 0:
        raise ValueError('games must be positive and max_steps nonnegative')
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    # Tiny single-position convolutions run much faster with one CPU thread.
    if policy == 'greedy':
        torch.set_num_threads(1)
    try:
        agent = load_agent(checkpoint) if policy == 'greedy' else None
        records = []
        for i in range(games):
            seed = seed_start + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            instance = make_board(board, seed)
            action_rng = random.Random(seed + 10**6) if board == 'fast' else random
            steps = 0
            while not instance.done and steps < max_steps:
                invalid = instance.get_invalid_moves()
                if len(invalid) == 4:
                    break
                if agent is None:
                    action = action_rng.choice([a for a in range(4) if a not in invalid])
                else:
                    with torch.no_grad():
                        state = torch.tensor(instance.grid, dtype=torch.float32)
                        q_values = agent.model(state).reshape(4)
                        q_values[invalid] = -torch.inf
                        action = int(q_values.argmax().item())
                instance.move(ALL_MOVES[action])
                steps += 1
            records.append(dict(seed=seed, score=int(instance.score_v2),
                                max_tile=int(np.max(instance.grid)), steps=steps))
        scores = [record['score'] for record in records]
        distribution = Counter(record['max_tile'] for record in records)
        return dict(games=games, policy=policy, board=board,
                    mean=statistics.mean(scores), median=statistics.median(scores),
                    max=max(scores),
                    max_tile_distribution={str(k): v for k, v in sorted(distribution.items())},
                    reach_rates={str(tile): sum(r['max_tile'] >= tile for r in records) / games
                                 for tile in (512, 1024, 2048, 4096)},
                    elapsed_seconds=time.perf_counter() - started, per_game=records)
    finally:
        if policy == 'greedy':
            torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--policy', choices=('greedy', 'random'), default='greedy')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--board', choices=('fast', 'old'), default='fast')
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    if args.games < 1 or args.max_steps < 0:
        parser.error('--games must be positive and --max-steps nonnegative')
    result = evaluate(checkpoint=args.checkpoint, policy=args.policy, games=args.games,
                      seed_start=args.seed_start, board=args.board, max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_game'}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    main()
