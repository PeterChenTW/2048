"""Differential checks against the unmodified legacy implementation."""
import random
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault('pygame', types.ModuleType('pygame'))
from game_2048 import Board
from fast_board import FastBoard


def legacy(grid):
    board = Board.__new__(Board)
    board.size = 4
    board.grid = [list(row) for row in grid]
    board.all_moves = ['left', 'down', 'right', 'up']
    board.score = board.score_v2 = board.turns = board.invalid_move = board.get_score = 0
    board.done = False
    board.merge_this_turn = None
    board.new_tile_position = None
    board.state = np.array(grid)
    return board


def test_random_boards():
    rng = random.Random(187)
    for trial in range(20000):
        # Alternate full, sparse, and arbitrary occupancy, including terminal boards.
        values = ([2, 4, 8, 16, 32, 64, 128, 256] if trial % 3 == 0 else
                  [0]* (12 if trial % 3 == 1 else 3) + [2, 4, 8, 16, 32, 64, 128, 256])
        grid = [[rng.choice(values) for _ in range(4)] for _ in range(4)]
        fast = FastBoard.from_grid(grid, rng)
        old = legacy(grid)
        assert fast.game_over() == old.game_over()
        snapshot = random.getstate()
        try:
            expected_invalid = old.get_invalid_moves()
        finally:
            random.setstate(snapshot)
        rng_state = rng.getstate()
        assert fast.get_invalid_moves() == expected_invalid
        assert rng.getstate() == rng_state
        for action in range(4):
            reference = legacy(grid)
            for _ in range(action):
                reference.rotate()
            reference.grid = [reference.merge(row)[0] for row in reference.grid]
            for _ in range((4-action) % 4):
                reference.rotate()
            cells, score, changed = fast.pure_move(action)
            assert cells == sum(reference.grid, [])
            assert score == reference.get_score
            assert changed == (reference.grid != grid)


def test_full_games():
    actions = random.Random(871)
    for seed in range(500):
        old = Board(4, random_seed=seed)
        fast = FastBoard(random.Random(seed))
        steps = 0
        while True:
            action = actions.randrange(4)  # Include invalid actions.
            snapshot = random.getstate()
            old_state, old_reward, old_done = old.move(old.all_moves[action])
            fast.rng.setstate(snapshot)
            state, reward, done = fast.move(action)
            assert np.array_equal(state, old_state)
            assert fast.grid == old.grid
            for field in ('score', 'score_v2', 'turns', 'invalid_move', 'get_score',
                          'merge_this_turn', 'new_tile_position'):
                assert getattr(fast, field) == getattr(old, field), field
            assert reward == old_reward
            assert done == old_done
            assert fast.rng.getstate() == random.getstate()
            steps += 1
            assert steps < 10000
            if done or old.game_over():
                break


def test_initial_boards():
    for seed in range(1000):
        old = Board(4, random_seed=seed)
        fast = FastBoard(random.Random(seed))
        assert fast.grid == old.grid
        assert fast.score == old.score
        assert fast.new_tile_position == old.new_tile_position
        assert fast.rng.getstate() == random.getstate()


def test_spawn_statistics():
    rng = random.Random(19)
    fours = 0
    for _ in range(20000):
        board = FastBoard.from_grid([[0]*4 for _ in range(4)], rng)
        fours += board.add_random_tile() == 4
    assert 0.09 < fours / 20000 < 0.11


def test_directions_and_terminal_quirks():
    grid = [[0]*4 for _ in range(4)]
    grid[1][1] = 2
    for action, destination in enumerate(((1, 0), (3, 1), (1, 3), (0, 1))):
        cells, _, _ = FastBoard.from_grid(grid).pure_move(action)
        assert cells[4*destination[0]+destination[1]] == 2
    # A board already terminal never sets done: zero_count is zero.
    full = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    board = FastBoard.from_grid(full)
    assert board.game_over()
    assert board.get_invalid_moves() == [0, 1, 2, 3]
    assert board.move('left')[1:] == (-1, False)
