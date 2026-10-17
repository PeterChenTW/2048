"""N-step returns and masked Double DQN updates.

The accumulator turns a stream of 1-step transitions into overlapping
n-step transitions. ``double_dqn_targets`` and ``dqn_batch_update`` consume
those fields (reward sum, per-transition discount, next-action mask) without
depending on the replay buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class NStepTransition:
    """One transition whose reward already includes the discounted n-step sum.

    ``reward`` is ``R = sum_{i < k} gamma^i r_i``. ``next_state``, ``done`` and
    ``next_mask`` belong to the k-th step. ``discount`` is ``gamma^k`` with
    ``k <= n`` (shorter when the episode ends or is truncated inside the window).
    """

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    next_mask: Any
    discount: float


class _Pending:
    __slots__ = ("state", "action", "reward", "next_state", "done", "next_mask")

    def __init__(self, state, action, reward, next_state, done, next_mask):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.next_mask = next_mask


def _as_reward(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.reshape(value, -1)[0])
    return float(value)


def _as_done(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, np.ndarray):
        return bool(np.reshape(value, -1)[0])
    return bool(value)


class NStepAccumulator:
    """Overlapping n-step return over a single episode.

    ``push`` emits the oldest transition once ``n`` steps are buffered, and
    emits nothing before that. A terminal step emits every buffered transition
    with its own shortened horizon, ``done=True``, and clears the window.
    ``flush`` does the same for a non-terminal cut (time limit): every
    remaining transition bootstraps from the last ``next_state`` / ``next_mask``
    with ``done=False``.
    """

    def __init__(self, n: int, gamma: float):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise ValueError(f"n must be an integer >= 1, got {n!r}")
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be an integer >= 1, got {n}")
        try:
            gamma_f = float(gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gamma must be in [0, 1], got {gamma!r}") from exc
        if not math.isfinite(gamma_f) or gamma_f < 0.0 or gamma_f > 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
        self.n = n
        self.gamma = gamma_f
        self._buf: list[_Pending] = []

    def __len__(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, state, action, reward, next_state, done, next_mask) -> list[NStepTransition]:
        terminal = _as_done(done)
        self._buf.append(
            _Pending(state, action, _as_reward(reward), next_state, terminal, next_mask)
        )
        if terminal:
            emitted = [
                self._build(start, len(self._buf), done=True)
                for start in range(len(self._buf))
            ]
            self._buf.clear()
            return emitted
        if len(self._buf) == self.n:
            emitted = self._build(0, self.n, done=self._buf[-1].done)
            del self._buf[0]
            return [emitted]
        return []

    def flush(self) -> list[NStepTransition]:
        """Emit a truncated episode. Horizons end at the last observed step."""
        end = len(self._buf)
        emitted = [self._build(start, end, done=False) for start in range(end)]
        self._buf.clear()
        return emitted

    def _build(self, start: int, end: int, done: bool) -> NStepTransition:
        k = end - start
        # gamma**0 == 1 even when gamma is 0, so the leading reward is kept.
        reward = 0.0
        for i in range(k):
            reward += (self.gamma ** i) * self._buf[start + i].reward
        first = self._buf[start]
        last = self._buf[end - 1]
        return NStepTransition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            done=bool(done),
            next_mask=last.next_mask,
            discount=self.gamma ** k,
        )


def _q_values(net, states) -> torch.Tensor:
    values = net(states)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"Q network must return [batch, actions], got {tuple(values.shape)}")
    return values


@torch.no_grad()
def double_dqn_targets(
    online,
    target,
    rewards,
    next_states,
    dones,
    next_masks,
    discounts,
) -> torch.Tensor:
    """Masked Double DQN targets, with no gradient.

    The online net picks ``argmax`` over valid next actions. The target net
    evaluates that action. The bootstrap is 0 when ``done`` is set or the mask
    is all False, so an all-invalid row never contributes NaN or inf.
    ``target = rewards + discounts * bootstrap``.
    """
    online_q = _q_values(online, next_states)
    target_q = _q_values(target, next_states).to(dtype=online_q.dtype, device=online_q.device)
    batch, n_actions = online_q.shape
    if target_q.shape != online_q.shape:
        raise ValueError(
            f"online and target Q shapes differ: {tuple(online_q.shape)} vs {tuple(target_q.shape)}"
        )

    device = online_q.device
    dtype = online_q.dtype
    masks = torch.as_tensor(next_masks, device=device)
    if masks.dtype != torch.bool:
        masks = masks != 0
    masks = masks.reshape(batch, n_actions)

    done = torch.as_tensor(dones, device=device)
    if done.dtype != torch.bool:
        done = done != 0
    done = done.reshape(batch)

    reward_t = torch.as_tensor(rewards, device=device, dtype=dtype).reshape(batch)
    discount_t = torch.as_tensor(discounts, device=device, dtype=dtype).reshape(batch)

    # Match the legacy agent exactly: invalid actions never beat finite logits.
    # An all-invalid row still has a defined argmax and its bootstrap is discarded.
    selectable = online_q.masked_fill(~masks, -torch.inf)
    action = selectable.argmax(dim=1)
    chosen = target_q.gather(1, action.view(batch, 1)).squeeze(1)

    bootstrap_ok = masks.any(dim=1) & ~done
    zeros = torch.zeros(batch, device=device, dtype=dtype)
    bootstrap = torch.where(bootstrap_ok, chosen, zeros)
    return reward_t + discount_t * bootstrap


def dqn_batch_update(
    online,
    target,
    optimizer,
    states,
    actions,
    rewards,
    next_states,
    dones,
    next_masks,
    discounts,
    is_weights,
    grad_clip: float = 1.0,
    huber: bool = False,
):
    """One importance-weighted Double DQN step.

    Returns ``(stats, td_abs)``. ``td_abs`` is ``|Q(s, a) - y|`` per sample for
    prioritized replay. ``stats`` reports the weighted loss and the Q / target
    values from the forward that produced it. Raises ``FloatingPointError``
    when that loss is not finite, before any gradient is applied.
    """
    q_all = online(states)
    if q_all.ndim == 1:
        q_all = q_all.unsqueeze(0)
    actions_t = torch.as_tensor(actions, device=q_all.device, dtype=torch.long).reshape(-1)
    if actions_t.numel() != q_all.shape[0]:
        raise ValueError(
            f"actions length {actions_t.numel()} does not match batch {q_all.shape[0]}"
        )
    q = q_all.gather(1, actions_t.view(-1, 1)).squeeze(1)

    targets = double_dqn_targets(
        online, target, rewards, next_states, dones, next_masks, discounts
    ).to(dtype=q.dtype, device=q.device)
    if targets.shape != q.shape:
        raise ValueError(
            f"target shape {tuple(targets.shape)} does not match Q(s, a) shape {tuple(q.shape)}"
        )

    if huber:
        per_sample = F.huber_loss(q, targets, reduction="none")
    else:
        per_sample = (q - targets).pow(2)

    weights = torch.as_tensor(is_weights, device=q.device, dtype=q.dtype).reshape(-1)
    if weights.numel() != q.shape[0]:
        raise ValueError(
            f"is_weights length {weights.numel()} does not match batch {q.shape[0]}"
        )
    # Mean reduction matches ``nn.MSELoss`` when every weight is 1, and a
    # weight of 0 contributes neither loss nor gradient.
    loss = (weights.detach() * per_sample).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"non-finite loss: {loss.detach().cpu().item()!r}")

    td_abs = (q - targets).detach().abs()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip)
    optimizer.step()

    q_det = q.detach()
    td_np = td_abs.cpu().numpy()
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "q_mean": float(q_det.mean().cpu().item()),
        "q_min": float(q_det.min().cpu().item()),
        "q_max": float(q_det.max().cpu().item()),
        "target_mean": float(targets.detach().mean().cpu().item()),
        "td_abs_mean": float(td_abs.mean().cpu().item()),
    }
    return stats, td_np
