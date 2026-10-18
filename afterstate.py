"""Afterstate generation utilities for 2048.

An afterstate is the deterministic board state immediately after sliding and
merging tiles, before the random tile spawn.
"""
from functools import lru_cache
from typing import Tuple, Union
import numpy as np

from fast_board import _merge, _LINES


EXPAND_MODES = ('reference', 'vectorized')

# Four 4-bit exponents encode a row: 0 is empty, 1..15 are tiles 2..32768.
_ROW_SHIFTS = np.array([0, 4, 8, 12], dtype=np.uint16)
_LINE_INDICES = np.asarray(_LINES)
_ACTION_INDICES = np.arange(4)[:, None, None]


@lru_cache(maxsize=1)
def _row_lookup_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Build tables lazily from the board's exact merge rules.

    Store raw int64 output tiles so merging two 2**15 tiles retains 2**16.
    Reference-only callers never pay the table construction cost.
    """
    tiles = np.array([0] + [1 << exponent for exponent in range(1, 16)], dtype=np.int64)
    keys = np.arange(1 << 16, dtype=np.uint16)
    rows = tiles[(keys[:, None] >> _ROW_SHIFTS) & 15]
    after = np.empty((len(keys), 4), dtype=np.int64)
    merge = np.empty(len(keys), dtype=np.int64)
    for key, row in enumerate(rows.tolist()):
        after[key], merge[key] = _merge(tuple(row))
    after.setflags(write=False)
    merge.setflags(write=False)
    return after, merge


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


def batch_afterstates_reference(grids: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def batch_afterstates_vectorized(grids: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a batch with NumPy row lookups in the same action order as reference.

    Inputs outside {0, 2**1, ..., 2**15} fall back to reference for the whole
    batch, including tile 1 (exponent 0 is reserved for empty cells).
    """
    cells = np.asarray(grids, dtype=np.int64)
    if cells.size == 0:
        return batch_afterstates_reference(cells)
    cells = cells.reshape(-1, 16)
    unsupported = (cells != 0) & (
        (cells < 2) | (cells > (1 << 15)) | ((cells & (cells - 1)) != 0)
    )
    if np.any(unsupported):
        return batch_afterstates_reference(cells)

    exponents = np.zeros(cells.shape, dtype=np.float64)
    np.log2(cells, out=exponents, where=cells != 0)
    rows = exponents.astype(np.uint16)[:, _LINE_INDICES]
    keys = (rows[..., 0] | (rows[..., 1] << 4)
            | (rows[..., 2] << 8) | (rows[..., 3] << 12))
    row_after, row_merge = _row_lookup_tables()
    after = np.empty((len(cells), 4, 16), dtype=np.int64)
    # Scatter each oriented row back into the board's original coordinates.
    after[:, _ACTION_INDICES, _LINE_INDICES] = row_after[keys]
    merge = row_merge[keys].sum(axis=2, dtype=np.int64)
    legal = np.any(after != cells[:, None, :], axis=2)
    return after, merge, legal


def batch_afterstates(
    grids: Union[np.ndarray, list, tuple], expand: str = 'reference',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand boards using the selected implementation; default is reference."""
    if expand == 'reference':
        return batch_afterstates_reference(grids)
    if expand == 'vectorized':
        return batch_afterstates_vectorized(grids)
    raise ValueError(f"Unknown afterstate expand mode {expand!r}; expected one of {EXPAND_MODES}")


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
