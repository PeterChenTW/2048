"""Afterstate value action-selection policy for 2048.

Given an afterstate value network V(x), action selection evaluates:
    score(s, a) = r(s, a) + gamma * V(after(s, a))   for legal a
                = -inf                               for illegal a
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence, Union
import numpy as np
import torch

from afterstate import afterstates, batch_afterstates
from rewards import merge_rewards


class LegalityMismatchError(ValueError, AssertionError):
    """Raised when caller-supplied invalid_moves disagrees with afterstate legality."""


class AfterstateScores(NamedTuple):
    """Container supporting both tuple unpacking and dict-key access."""

    after: np.ndarray
    rewards: np.ndarray
    legal: np.ndarray
    scores: torch.Tensor

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError as exc:
                raise KeyError(key) from exc
        return tuple.__getitem__(self, key)

    def keys(self):
        return self._fields

    def values(self):
        return tuple(self)

    def items(self):
        return zip(self._fields, self)

    def get(self, key, default=None):
        if isinstance(key, str):
            return getattr(self, key, default)
        return default

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._fields
        return tuple.__contains__(self, key)


def merge_reward(
    merge: Union[np.ndarray, Sequence[int], int, float],
    mode: str = "log",
    scale: float = 256.0,
) -> np.ndarray:
    """Compute vectorized merge reward in float32.

    Args:
        merge: Merge score(s) gained on the move.
        mode: 'log' -> log2(1 + merge) / 16, or 'linear' -> merge / scale.
        scale: Positive divisor for 'linear' mode (default 256.0).

    Returns:
        np.ndarray of dtype float32 with the same shape as ``merge``.
    """
    return np.asarray(merge_rewards(merge, mode, scale), dtype=np.float32)


def _normalize_grids(grids: Union[np.ndarray, torch.Tensor, Sequence]) -> np.ndarray:
    """Validate and reshape grids of shape (N, 16), (N, 4, 4), (16,), or (4, 4) into (N, 16) int64."""
    if isinstance(grids, torch.Tensor):
        arr = grids.detach().cpu().numpy()
    else:
        arr = np.asarray(grids)

    if arr.ndim == 1:
        if arr.shape != (16,):
            raise ValueError(f"Invalid 1D grid shape {arr.shape}; expected (16,).")
        return arr.astype(np.int64, copy=False).reshape(1, 16)
    if arr.ndim == 2:
        if arr.shape == (4, 4):
            return arr.astype(np.int64, copy=False).reshape(1, 16)
        if arr.shape[1] == 16:
            return arr.astype(np.int64, copy=False)
        raise ValueError(f"Invalid 2D grids shape {arr.shape}; expected (N, 16) or (4, 4).")
    if arr.ndim == 3:
        if arr.shape[1:] == (4, 4):
            return arr.astype(np.int64, copy=False).reshape(arr.shape[0], 16)
        raise ValueError(f"Invalid 3D grids shape {arr.shape}; expected (N, 4, 4).")
    raise ValueError(f"Invalid grids ndim {arr.ndim}; expected 1, 2, or 3.")


def _normalize_single_grid(grid: Union[np.ndarray, torch.Tensor, Sequence]) -> np.ndarray:
    """Validate and reshape a single board (16,) or (4, 4) into (16,) int64."""
    if isinstance(grid, torch.Tensor):
        arr = grid.detach().cpu().numpy()
    else:
        arr = np.asarray(grid)
    if arr.shape not in ((16,), (4, 4)):
        raise ValueError(f"Expected a single board of shape (16,) or (4, 4), got {arr.shape}.")
    return arr.astype(np.int64, copy=False).reshape(16)


def afterstate_scores(
    model,
    grids: Union[np.ndarray, torch.Tensor, Sequence],
    gamma: float = 0.99,
    reward_mode: str = "log",
    reward_scale: float = 256.0,
) -> AfterstateScores:
    """Compute afterstates, merge rewards, legality, and action scores r + gamma * V(after).

    Performs at most ONE batched ``torch.no_grad()`` forward pass of ``model``
    over the legal afterstates only.

    Args:
        model: Callable / nn.Module mapping raw board tensor (K, 16) -> (K,) values.
        grids: Board(s) of shape (N, 16), (N, 4, 4), (16,), or (4, 4).
        gamma: Discount factor multiplying V(after).
        reward_mode: 'log' or 'linear'.
        reward_scale: Scale divisor when ``reward_mode == 'linear'``.

    Returns:
        AfterstateScores (tuple/dict-compatible) with:
            after: np.ndarray int64 of shape (N, 4, 16)
            rewards: np.ndarray float32 of shape (N, 4) (0.0 for illegal)
            legal: np.ndarray bool of shape (N, 4)
            scores: torch.Tensor float32 of shape (N, 4) (-inf for illegal)
    """
    grids_2d = _normalize_grids(grids)
    n = grids_2d.shape[0]
    after, merge, legal = batch_afterstates(grids_2d)

    rewards = merge_reward(merge, mode=reward_mode, scale=reward_scale)
    rewards = np.where(legal, rewards, np.float32(0.0)).astype(np.float32, copy=False)

    scores = torch.full((n, 4), -torch.inf, dtype=torch.float32)
    if np.any(legal):
        legal_after = torch.as_tensor(after[legal], dtype=torch.float32)
        with torch.no_grad():
            v_legal = model(legal_after)
        v_legal_t = torch.as_tensor(v_legal, dtype=torch.float32).detach().cpu().reshape(-1)
        if v_legal_t.numel() != legal_after.shape[0]:
            raise ValueError(
                f"model returned {v_legal_t.numel()} values for {legal_after.shape[0]} legal afterstates"
            )
        r_legal = torch.as_tensor(rewards[legal], dtype=torch.float32)
        legal_mask = torch.as_tensor(legal, dtype=torch.bool)
        scores[legal_mask] = r_legal + float(gamma) * v_legal_t

    return AfterstateScores(after=after, rewards=rewards, legal=legal, scores=scores)


def greedy_actions(
    model,
    grids: Union[np.ndarray, torch.Tensor, Sequence],
    gamma: float = 0.99,
    reward_mode: str = "log",
    reward_scale: float = 256.0,
) -> np.ndarray:
    """Return greedy action indices (N,) int64 maximizing r(s,a) + gamma*V(after(s,a)) over legal a.

    Ties are broken toward the lowest action index (0..3).
    Raises ValueError if any board in ``grids`` has no legal action.
    """
    res = afterstate_scores(
        model,
        grids,
        gamma=gamma,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
    )
    if res.legal.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if not np.all(res.legal.any(axis=1)):
        raise ValueError("Cannot select greedy action: at least one board has no legal action.")
    scores_np = res.scores.detach().cpu().numpy()
    return np.argmax(scores_np, axis=1).astype(np.int64)


def select_action(
    model,
    grid: Union[np.ndarray, torch.Tensor, Sequence],
    invalid_moves: Iterable[int],
    epsilon: float,
    rng: np.random.Generator,
    gamma: float = 0.99,
    reward_mode: str = "log",
    reward_scale: float = 256.0,
) -> int:
    """Epsilon-greedy action selection over legal moves using afterstate values.

    Consumes ``rng`` identically to ``DDQNAgent.act``:
    one ``rng.random()`` call, followed by ``rng.choice(valid_actions)`` on exploration.
    Raises ValueError/AssertionError if ``invalid_moves`` disagrees with afterstate legality
    or if no legal actions exist.
    """
    flat_grid = _normalize_single_grid(grid)
    _, _, legal = afterstates(flat_grid)

    invalid_list = [int(a) for a in invalid_moves]
    if any(a < 0 or a >= 4 for a in invalid_list):
        raise LegalityMismatchError(f"invalid_moves must be in 0..3, got {invalid_list}")
    invalid_set = set(invalid_list)
    expected_invalid = {a for a in range(4) if not legal[a]}
    if invalid_set != expected_invalid:
        raise LegalityMismatchError(
            f"invalid_moves {sorted(invalid_set)} disagrees with afterstate illegality {sorted(expected_invalid)}"
        )

    valid_actions = [a for a in range(4) if legal[a]]
    if not valid_actions:
        raise ValueError("no valid actions")

    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    return int(
        greedy_actions(
            model,
            flat_grid,
            gamma=gamma,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
        )[0]
    )
