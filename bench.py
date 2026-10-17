"""Benchmark the training workload (legality query, valid action, move)."""
import argparse
import random
import sys
import time
import types

sys.modules.setdefault('pygame', types.ModuleType('pygame'))
from game_2048 import Board
from fast_board import FastBoard


def measure(factory, seconds):
    actions = random.Random(42)
    board = factory()
    steps = 0
    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        invalid = board.get_invalid_moves()
        valid = [i for i in range(4) if i not in invalid]
        if not valid:
            board = factory()
            continue
        board.move(board.all_moves[actions.choice(valid)])
        steps += 1
        if board.done:
            board = factory()
    return steps / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    random.seed(42)
    old = measure(lambda: Board(4), args.seconds)
    rng = random.Random(42)
    fast = measure(lambda: FastBoard(rng), args.seconds)
    print(f'Board:     {old:,.0f} steps/s')
    print(f'FastBoard: {fast:,.0f} steps/s')
    print(f'Speedup:   {fast / old:.2f}x')


if __name__ == '__main__':
    main()
