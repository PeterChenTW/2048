"""Unit tests for afterstate generation utilities."""
import random
import numpy as np
import pytest

from fast_board import FastBoard
from afterstate import afterstates, batch_afterstates, is_terminal


def _collect_game_positions(num_games=10, max_steps=100, seed=42):
    """Play random-policy games to collect a diverse set of board positions."""
    rng = random.Random(seed)
    positions = []
    for _ in range(num_games):
        board = FastBoard(rng=rng)
        positions.append(board.grid)
        for _ in range(max_steps):
            invalid = board.get_invalid_moves()
            valid = [a for a in range(4) if a not in invalid]
            if not valid:
                break
            action = rng.choice(valid)
            board.move(action)
            positions.append(board.grid)
            if board.done:
                break
    return positions


def test_afterstates_pure_move_and_invalid_moves():
    """Verify afterstates() matches pure_move and legal matches get_invalid_moves."""
    positions = _collect_game_positions(num_games=5, max_steps=50, seed=123)
    assert len(positions) > 30

    for grid in positions:
        board = FastBoard.from_grid(grid)
        after, merge, legal = afterstates(grid)

        assert after.shape == (4, 16)
        assert after.dtype == np.int64
        assert merge.shape == (4,)
        assert merge.dtype == np.int64
        assert legal.shape == (4,)
        assert legal.dtype == bool

        invalid = set(board.get_invalid_moves())
        orig_flat = np.array(grid, dtype=np.int64).reshape(16)

        for a in range(4):
            res, score, changed = board.pure_move(a)
            # legal[a] == changed == (a not in invalid)
            assert legal[a] == changed, f"Action {a}: legal={legal[a]} vs changed={changed}"
            assert legal[a] == (a not in invalid), f"Action {a}: legal={legal[a]} vs not_invalid={a not in invalid}"

            if legal[a]:
                assert np.array_equal(after[a], res), f"Action {a}: afterstate mismatch"
                assert merge[a] == score, f"Action {a}: merge score mismatch"
            else:
                assert np.array_equal(after[a], orig_flat), f"Action {a}: illegal afterstate should be unchanged"
                assert merge[a] == 0, f"Action {a}: illegal merge should be 0"


def test_afterstates_matches_fastboard_move():
    """Applying FastBoard.move(a) equals after[a] plus one new tile (2 or 4) in an empty cell."""
    positions = _collect_game_positions(num_games=5, max_steps=40, seed=456)

    tested_moves = 0
    for grid in positions:
        after, merge, legal = afterstates(grid)
        for a in range(4):
            if not legal[a]:
                continue

            # Clone board with controlled RNG
            board_clone = FastBoard.from_grid(grid, rng=random.Random(999))
            prev_score_v2 = board_clone.score_v2
            _, _, done = board_clone.move(a)

            # Merge score must match score_v2 delta
            assert board_clone.score_v2 - prev_score_v2 == merge[a]

            after_a = after[a]
            cells_after_move = board_clone._cells

            # If game became done or board is full, move might have spawned or not
            # If a tile was spawned:
            if board_clone.new_tile_position is not None:
                spawn_x, spawn_y = board_clone.new_tile_position
                spawn_idx = 4 * spawn_x + spawn_y

                # Exactly at spawn_idx, after_a must have been 0 (empty cell)
                assert after_a[spawn_idx] == 0
                # cells_after_move at spawn_idx must be 2 or 4
                assert cells_after_move[spawn_idx] in (2, 4)

                # All other 15 cells must be strictly identical
                for i in range(16):
                    if i != spawn_idx:
                        assert after_a[i] == cells_after_move[i], f"Mismatch at cell {i}"
            tested_moves += 1

    assert tested_moves > 50


def test_afterstates_does_not_mutate_or_consume_rng():
    """Calling afterstates() does not change board cells, grid, or consume rng."""
    rng = random.Random(777)
    board = FastBoard(rng=rng)
    board.move(0)

    grid_copy = [row[:] for row in board.grid]
    cells_copy = board._cells.copy()
    rng_state = rng.getstate()

    after, merge, legal = afterstates(board.grid)

    assert rng.getstate() == rng_state
    assert board.grid == grid_copy
    assert np.array_equal(board._cells, cells_copy)


def test_batch_afterstates():
    """batch_afterstates() matches per-board afterstates() results and handles shapes."""
    positions = _collect_game_positions(num_games=3, max_steps=30, seed=888)
    n = len(positions)
    assert n > 20

    # 1. As list of 4x4
    batch_after, batch_merge, batch_legal = batch_afterstates(positions)
    assert batch_after.shape == (n, 4, 16)
    assert batch_merge.shape == (n, 4)
    assert batch_legal.shape == (n, 4)

    for i in range(n):
        exp_after, exp_merge, exp_legal = afterstates(positions[i])
        assert np.array_equal(batch_after[i], exp_after)
        assert np.array_equal(batch_merge[i], exp_merge)
        assert np.array_equal(batch_legal[i], exp_legal)

    # 2. As (N, 16) array
    flat_arr = np.array([np.array(p).reshape(16) for p in positions], dtype=np.int64)
    batch_after_flat, batch_merge_flat, batch_legal_flat = batch_afterstates(flat_arr)
    assert np.array_equal(batch_after, batch_after_flat)
    assert np.array_equal(batch_merge, batch_merge_flat)
    assert np.array_equal(batch_legal, batch_legal_flat)

    # 3. Empty batch
    empty_after, empty_merge, empty_legal = batch_afterstates([])
    assert empty_after.shape == (0, 4, 16)
    assert empty_merge.shape == (0, 4)
    assert empty_legal.shape == (0, 4)


def test_terminal_board():
    """Terminal board has all moves illegal and is_terminal() == True."""
    # Constructed terminal board: checkerboard pattern with no adjacent matches and no zeros
    terminal_grid = [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
    assert is_terminal(terminal_grid) is True

    after, merge, legal = afterstates(terminal_grid)
    assert not legal.any()
    assert (merge == 0).all()
    flat_term = np.array(terminal_grid, dtype=np.int64).reshape(16)
    for a in range(4):
        assert np.array_equal(after[a], flat_term)

    # Non-terminal board: fresh board
    non_term_grid = [
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert is_terminal(non_term_grid) is False
    _, _, non_term_legal = afterstates(non_term_grid)
    assert non_term_legal.any()

    # Play until game over
    rng = random.Random(2026)
    board = FastBoard(rng=rng)
    while not board.done:
        invalid = board.get_invalid_moves()
        valid = [a for a in range(4) if a not in invalid]
        if not valid:
            break
        board.move(rng.choice(valid))

    if board.game_over():
        assert is_terminal(board.grid) is True
        _, _, term_legal = afterstates(board.grid)
        assert not term_legal.any()


def test_input_formats():
    """afterstates and is_terminal work with 4x4 lists, flat lists, and numpy arrays."""
    grid_4x4_list = [
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2, 4],
        [8, 16, 32, 64],
    ]
    grid_flat_list = [v for row in grid_4x4_list for v in row]
    grid_np_4x4 = np.array(grid_4x4_list, dtype=np.int32)
    grid_np_flat = np.array(grid_flat_list, dtype=np.int64)
    grid_tuple_4x4 = tuple(tuple(row) for row in grid_4x4_list)

    ref_after, ref_merge, ref_legal = afterstates(grid_4x4_list)
    ref_term = is_terminal(grid_4x4_list)

    for g in [grid_flat_list, grid_np_4x4, grid_np_flat, grid_tuple_4x4]:
        a, m, l = afterstates(g)
        assert np.array_equal(a, ref_after)
        assert np.array_equal(m, ref_merge)
        assert np.array_equal(l, ref_legal)
        assert is_terminal(g) == ref_term
