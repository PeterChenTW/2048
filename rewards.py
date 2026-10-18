"""Reward shaping functions for 2048 reinforcement learning."""
import math
import numpy as np


def _validate_options(mode, scale):
    if mode not in ('log', 'linear', 'raw'):
        raise ValueError(f"unknown reward mode: {mode!r}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (scale > 0) or not math.isfinite(scale):
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")


def merge_rewards(merge_score, mode='log', scale=256.0):
    """Shape scalar or batched merge scores before any float32 conversion."""
    _validate_options(mode, scale)
    merge_score = np.asarray(merge_score, dtype=np.float64)
    if mode == 'log':
        return np.log2(1 + merge_score) / 16
    if mode == 'raw':
        return merge_score  # unshaped game score; pair with value_rescale='h'
    return merge_score / scale


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
        'raw': merge_score if spawned else -1.0 (scale ignored)
    scale : float, default=256.0
        Scale denominator for linear mode; must be positive.
    """
    if not spawned:
        _validate_options(mode, scale)
        return -1.0
    return float(merge_rewards(merge_score, mode, scale))
