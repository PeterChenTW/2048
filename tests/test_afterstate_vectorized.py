"""Exact reference parity for vectorized afterstate expansion."""
import random

import numpy as np
import pytest

import afterstate
from afterstate import (
    EXPAND_MODES,
    batch_afterstates,
    batch_afterstates_reference,
    batch_afterstates_vectorized,
)
from fast_board import FastBoard


RANDOM_BOARD_COUNT = 6000


def _assert_equal(actual, expected):
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
        assert result.dtype == reference.dtype


@pytest.fixture(scope="module")
def comparison_boards():
    rng = np.random.default_rng(20261018)
    random_groups = []
    # Dense, mixed and sparse boards, all including exponents through 15.
    for empty_probability in (0.1, 0.5, 0.85):
        tiles = np.left_shift(1, rng.integers(1, 16, size=(2000, 16)))
        tiles[rng.random(tiles.shape) < empty_probability] = 0
        random_groups.append(tiles)
    random_boards = np.concatenate(random_groups)
    assert len(random_boards) == RANDOM_BOARD_COUNT >= 5000
    assert random_boards.max() == 2**15

    game_rng = random.Random(20261018)
    game_boards = []
    for _ in range(5):
        board = FastBoard(rng=game_rng)
        game_boards.append(np.asarray(board.grid, dtype=np.int64).reshape(16))
        for _ in range(64):
            invalid = board.get_invalid_moves()
            valid = [action for action in range(4) if action not in invalid]
            if not valid:
                break
            board.move(game_rng.choice(valid))
            game_boards.append(np.asarray(board.grid, dtype=np.int64).reshape(16))
            if board.done:
                break
    assert len(game_boards) > 100
    return np.concatenate([random_boards, np.asarray(game_boards)]), len(game_boards)


@pytest.mark.parametrize("matrix_shape", [False, True])
def test_random_and_game_boards(comparison_boards, matrix_shape, record_property):
    boards, game_count = comparison_boards
    assert len(boards) == RANDOM_BOARD_COUNT + game_count
    assert RANDOM_BOARD_COUNT >= 5000
    record_property("random_board_count", RANDOM_BOARD_COUNT)
    record_property("game_board_count", game_count)
    record_property("total_board_count", len(boards))
    grids = boards.reshape(-1, 4, 4) if matrix_shape else boards
    original = grids.copy()
    expected = batch_afterstates_reference(grids)
    actual = batch_afterstates_vectorized(grids)
    _assert_equal(actual, expected)
    np.testing.assert_array_equal(grids, original)
    after, merge, legal = actual
    assert after.shape == (len(boards), 4, 16)
    assert merge.shape == legal.shape == (len(boards), 4)
    assert after.dtype == merge.dtype == np.int64
    assert legal.dtype == np.bool_
    assert not np.shares_memory(after, grids)
    # Illegal actions preserve the input and have zero merge score.
    np.testing.assert_array_equal(after[~legal], np.broadcast_to(boards[:, None, :], after.shape)[~legal])
    assert np.all(merge[~legal] == 0)


EDGE_BOARDS = [
    pytest.param([[2, 4, 2, 4], [4, 2, 4, 2]] * 2,
                 [False] * 4, [2, 4, 2, 4], 0, id="terminal"),
    pytest.param([[2, 0, 0, 0]] + [[0] * 4] * 3,
                 [False, True, True, False], [2, 0, 0, 0], 0, id="partly-legal"),
    pytest.param([[2] * 4] * 4,
                 [True] * 4, [4, 4, 0, 0], 32, id="multiple-merges"),
    pytest.param([[0] * 4] * 4,
                 [False] * 4, [0] * 4, 0, id="all-zero"),
    pytest.param([[0, 2, 4, 8]] + [[0] * 4] * 3,
                 [True, True, False, False], [2, 4, 8, 0], 0, id="slide-no-merge"),
    pytest.param([[2, 0, 2, 0]] + [[0] * 4] * 3,
                 [True] * 3 + [False], [4, 0, 0, 0], 4, id="zeros-between-tiles"),
    pytest.param([[2, 2, 4, 0]] + [[0] * 4] * 3,
                 [True] * 3 + [False], [4, 4, 0, 0], 4, id="no-double-merge"),
    pytest.param([[2**15] * 4] * 4,
                 [True] * 4, [2**16, 2**16, 0, 0], 2**19, id="large-tiles"),
]


@pytest.mark.parametrize("grid, expected_legal, left_row, left_score", EDGE_BOARDS)
def test_edge_boards(grid, expected_legal, left_row, left_score):
    boards = np.asarray([grid], dtype=np.int64)
    actual = batch_afterstates_vectorized(boards)
    _assert_equal(actual, batch_afterstates_reference(boards))
    after, merge, legal = actual
    np.testing.assert_array_equal(legal[0], expected_legal)
    np.testing.assert_array_equal(after[0, 0, :4], left_row)
    assert merge[0, 0] == left_score


@pytest.mark.parametrize("tile", [1, 3, 6, -2, 2**16, 2**30, 2**60, -(2**63), 2**63 - 1])
def test_out_of_range_falls_back(tile, monkeypatch):
    # Include an ordinary board in the same batch to check whole-batch fallback.
    boards = np.zeros((2, 16), dtype=np.int64)
    boards[0, :4] = [2, 2, 0, 4]
    boards[1, 0] = tile
    expected = batch_afterstates_reference(boards)
    original = boards.copy()
    calls = []

    def reference_spy(grids):
        calls.append(grids.copy())
        return batch_afterstates_reference(grids)

    monkeypatch.setattr(afterstate, "batch_afterstates_reference", reference_spy)
    actual = batch_afterstates_vectorized(boards)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], original)
    _assert_equal(actual, expected)
    np.testing.assert_array_equal(boards, original)
    assert not np.shares_memory(actual[0], boards)


def test_supported_large_tiles_use_lookup(monkeypatch):
    boards = np.full((2, 16), 2**15, dtype=np.int64)
    boards[1, ::2] = 0
    expected = batch_afterstates_reference(boards)

    def unexpected_fallback(_):
        pytest.fail("Supported boards must use the vectorized path")

    monkeypatch.setattr(afterstate, "batch_afterstates_reference", unexpected_fallback)
    _assert_equal(batch_afterstates_vectorized(boards), expected)


@pytest.mark.parametrize("expand", EXPAND_MODES)
@pytest.mark.parametrize("matrix_shape", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_shapes_dtypes_and_no_alias(expand, matrix_shape, dtype):
    # A strided, read-only input must work without mutation or shared output.
    storage = np.zeros((3, 32), dtype=dtype)
    grids = storage[:, ::2]
    grids[:, :4] = [0, 2, 2, 4]
    if matrix_shape:
        grids = grids.reshape(3, 4, 4)
    original = grids.copy()
    grids.setflags(write=False)
    actual = batch_afterstates(grids, expand=expand)
    _assert_equal(actual, batch_afterstates_reference(original))
    for result, shape, expected_dtype in zip(
        actual, [(3, 4, 16), (3, 4), (3, 4)], [np.int64, np.int64, np.bool_]
    ):
        assert result.shape == shape
        assert result.dtype == expected_dtype
        assert not np.shares_memory(result, grids)
    actual[0][0, 0, 0] = -99
    np.testing.assert_array_equal(grids, original)
    _assert_equal(batch_afterstates(grids, expand=expand), batch_afterstates_reference(original))


@pytest.mark.parametrize("expand", EXPAND_MODES)
@pytest.mark.parametrize("grids", [[], np.empty((0, 16)), np.empty((0, 4, 4))])
def test_empty_batch(expand, grids):
    actual = batch_afterstates(grids, expand=expand)
    _assert_equal(actual, batch_afterstates_reference(grids))
    for result, shape, dtype in zip(
        actual, [(0, 4, 16), (0, 4), (0, 4)], [np.int64, np.int64, np.bool_]
    ):
        assert result.shape == shape
        assert result.dtype == dtype


def test_default_is_reference_without_table_build(monkeypatch):
    assert EXPAND_MODES == ("reference", "vectorized")
    boards = [[2, 2, 0, 0] * 4]
    expected = batch_afterstates_reference(boards)
    calls = []

    def reference_spy(grids):
        calls.append(grids)
        return expected

    def unexpected_table_build():
        pytest.fail("The default reference path must not build the lookup table")

    monkeypatch.setattr(afterstate, "batch_afterstates_reference", reference_spy)
    monkeypatch.setattr(afterstate, "_row_lookup_tables", unexpected_table_build)
    assert batch_afterstates(boards) is expected
    assert batch_afterstates(boards, expand="reference") is expected
    assert len(calls) == 2


def test_vectorized_dispatch(monkeypatch):
    boards = [[0] * 16]
    expected = batch_afterstates_reference(boards)
    monkeypatch.setattr(afterstate, "batch_afterstates_vectorized", lambda grids: expected)
    assert batch_afterstates(boards, expand="vectorized") is expected


@pytest.mark.parametrize("expand", ["invalid", "", None])
def test_invalid_expand_mode(expand):
    with pytest.raises(ValueError, match="expand mode"):
        batch_afterstates([], expand=expand)
