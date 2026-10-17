"""Reward shaping functions for 2048 reinforcement learning."""
import math
import numpy as np


def shape_reward(merge_score, spawned, mode='log', scale=256.0) -> float:
    """Compute step reward given turn merge score and whether a tile spawned.

    Parameters
    ----------
    merge_score : int or float
        Points earned by merging tiles in this move (0 if no merge).
    spawned : bool
        Whether a new tile was spawned after the move.
    mode : {'log', 'linear'}, default='log'
        'log': np.log2(1 + merge_score) / 16 if spawned else -1.0
        'linear': merge_score / scale if spawned else -1.0
    scale : float, default=256.0
        Scale denominator for linear mode; must be positive.
    """
    if mode not in ('log', 'linear'):
        raise ValueError(f"unknown reward mode: {mode!r}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (scale > 0) or not math.isfinite(scale):
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    if not spawned:
        return -1.0
    if mode == 'log':
        return float(np.log2(1 + merge_score) / 16)
    return float(merge_score / scale)
