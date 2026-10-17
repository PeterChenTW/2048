"""Afterstate generation utilities for 2048.

An afterstate is the deterministic board state immediately after sliding and
merging tiles, before the random tile spawn.
"""
from typing import Tuple, Union
import numpy as np

from fast_board import _merge, _LINES


def afterstates(grid: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute afterstates, merge scores, and legality for all 4 actions.

    Action order matches FastBoard.all_moves: ['left', 'down', 'right', 'up']
    (0: left, 1: down, 2: right, 3: up).

    Args:
        grid: 4x4 or flat 16 board of raw tile values.

    Returns:
        after: np.ndarray int64 of shape (4, 16). For illegal actions, after[a] is
               the unchanged board.
        merge: np.ndarray int64 of shape (4,). For illegal actions, merge[a] = 0.
        legal: np.ndarray bool of shape (4,). True if action changes the board.
    """
    cells = np.asarray(grid, dtype=np.int64).reshape(16)
    cells_tuple = tuple(cells.tolist())

    after = np.empty((4, 16), dtype=np.int64)
    merge = np.zeros(4, dtype=np.int64)
    legal = np.zeros(4, dtype=bool)

    for a in range(4):
        res = list(cells_tuple)
        score = 0
        changed = False
        for i0, i1, i2, i3 in _LINES[a]:
            row = (cells_tuple[i0], cells_tuple[i1], cells_tuple[i2], cells_tuple[i3])
            merged_row, gained = _merge(row)
            if merged_row != row:
                changed = True
                res[i0], res[i1], res[i2], res[i3] = merged_row
                score += gained
        if changed:
            after[a] = res
            merge[a] = score
            legal[a] = True
        else:
            after[a] = cells
            merge[a] = 0
            legal[a] = False

    return after, merge, legal


def batch_afterstates(grids: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute afterstates for a batch of N boards.

    Args:
        grids: Array or sequence of N boards, each 4x4 or flat 16.

    Returns:
        after: np.ndarray int64 of shape (N, 4, 16).
        merge: np.ndarray int64 of shape (N, 4).
        legal: np.ndarray bool of shape (N, 4).
    """
    grids_arr = np.asarray(grids, dtype=np.int64)
    if grids_arr.size == 0:
        return (
            np.empty((0, 4, 16), dtype=np.int64),
            np.empty((0, 4), dtype=np.int64),
            np.empty((0, 4), dtype=bool),
        )

    if grids_arr.ndim == 3 and grids_arr.shape[1:] == (4, 4):
        grids_arr = grids_arr.reshape(-1, 16)
    elif grids_arr.ndim == 2 and grids_arr.shape[1] == 16:
        pass
    else:
        grids_arr = grids_arr.reshape(-1, 16)

    n = len(grids_arr)
    after = np.empty((n, 4, 16), dtype=np.int64)
    merge = np.empty((n, 4), dtype=np.int64)
    legal = np.empty((n, 4), dtype=bool)

    for i in range(n):
        after[i], merge[i], legal[i] = afterstates(grids_arr[i])

    return after, merge, legal


def is_terminal(grid: Union[np.ndarray, list, tuple]) -> bool:
    """Return True if no legal action can be taken from grid.

    Args:
        grid: 4x4 or flat 16 board of raw tile values.

    Returns:
        bool: True if no legal moves exist, False otherwise.
    """
    cells = np.asarray(grid, dtype=np.int64).reshape(16)
    cells_tuple = tuple(cells.tolist())
    for lines in _LINES:
        for a, b, c, d in lines:
            row = (cells_tuple[a], cells_tuple[b], cells_tuple[c], cells_tuple[d])
            if _merge(row)[0] != row:
                return False
    return True
