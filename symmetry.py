"""D4 board symmetries (4 rotations x optional mirror) for 2048 replay augmentation.

2048's rules are invariant under the 8 symmetries of the square: moves map to
moves, merges and merge scores are preserved, and new tiles spawn uniformly on
empty cells. Hence the afterstate value V(x) satisfies V(g x) = V(x). These are
rules of the game only; no corner or monotonicity knowledge is encoded.

Boards are flat, row-major length-16 arrays (index r*4 + c).
Actions use FastBoard order: 0=left, 1=down, 2=right, 3=up.
"""
import numpy as np
import torch


def _cell_map(k, mirror):
    """Return perm such that transformed_flat = flat[perm]."""
    idx = np.arange(16).reshape(4, 4)
    if mirror:
        idx = idx[:, ::-1]
    idx = np.rot90(idx, k)  # counter-clockwise, k quarter turns
    return idx.reshape(16).copy()


# PERMS[i] for i = mirror*4 + k; PERMS[0] is the identity.
PERMS = np.stack([_cell_map(k, m) for m in (0, 1) for k in range(4)])

# Direction vectors (dr, dc) for actions 0..3 = left, down, right, up.
_DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def _transform_dir(i, d):
    """Image of a direction vector under symmetry i (derived numerically)."""
    perm = PERMS[i]
    # Place a marker pair and see where it moves: cell (1,1) and its neighbour.
    src = 5
    dst = (1 + d[0]) * 4 + (1 + d[1])
    inv = np.argsort(perm)  # cell j of original lands at position inv[j]
    a, b = divmod(int(inv[src]), 4), divmod(int(inv[dst]), 4)
    return (b[0] - a[0], b[1] - a[1])


# ACTION_MAPS[i][a] = action on the transformed board equivalent to action a.
ACTION_MAPS = np.array([[next(k for k, v in _DIRS.items() if v == _transform_dir(i, _DIRS[a]))
                         for a in range(4)] for i in range(8)])


def apply_np(boards, i):
    """Apply symmetry i to (16,) or (N, 16) numpy boards."""
    return np.asarray(boards)[..., PERMS[i]]


_PERMS_T = torch.as_tensor(PERMS, dtype=torch.long)


def apply_batch(boards, sym_ids):
    """Apply per-row symmetries to a (N, 16) tensor; sym_ids is a length-N int array."""
    index = _PERMS_T[torch.as_tensor(np.asarray(sym_ids), dtype=torch.long)]
    return torch.gather(boards, 1, index)
