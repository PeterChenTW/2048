"""N-step returns, masked Double DQN targets, and one IS-weighted update."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nstep import NStepAccumulator, double_dqn_targets, dqn_batch_update


def _hand_return(rewards, gamma, start, k):
    total = 0.0
    for i in range(k):
        total += (gamma ** i) * rewards[start + i]
    return total, gamma ** k


def _push_seq(acc, rewards, gamma_unused=None, done_at=None, masks=None, states=None):
    """Push a list of rewards. Returns the list of emission lists."""
    del gamma_unused
    emissions = []
    for i, reward in enumerate(rewards):
        state = states[i] if states is not None else f"s{i}"
        next_state = states[i + 1] if states is not None else f"ns{i}"
        mask = masks[i] if masks is not None else (True, True, True, True)
        done = done_at == i
        emissions.append(
            acc.push(state, i % 4, reward, next_state, done, mask)
        )
    return emissions


class _Table(nn.Module):
    """Returns a fixed Q table, ignoring the state contents."""

    def __init__(self, q):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q, dtype=torch.float32))

    def forward(self, states):
        return self.q


class _OrthQ(nn.Module):
    """Q = state @ W with W shaped [features, actions]."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2, 4))

    def forward(self, x):
        return x @ self.w


class _BiasQ(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((4,), float(value)))

    def forward(self, x):
        return self.bias.expand(x.shape[0], -1)


def _fixed_batch(rewards, dones, masks, discounts, states=None, next_states=None, actions=None):
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    batch = rewards_t.shape[0]
    if states is None:
        states = torch.zeros(batch, 4)
    if next_states is None:
        next_states = torch.zeros(batch, 4)
    if actions is None:
        actions = torch.zeros(batch, dtype=torch.long)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards_t,
        "next_states": next_states,
        "dones": torch.tensor(dones, dtype=torch.bool),
        "next_masks": torch.tensor(masks, dtype=torch.bool),
        "discounts": torch.tensor(discounts, dtype=torch.float32),
    }


def test_rejects_bad_n_and_gamma():
    with pytest.raises(ValueError):
        NStepAccumulator(0, 0.9)
    with pytest.raises(ValueError):
        NStepAccumulator(-3, 0.9)
    with pytest.raises(ValueError):
        NStepAccumulator(1.5, 0.9)
    with pytest.raises(ValueError):
        NStepAccumulator(True, 0.9)
    with pytest.raises(ValueError):
        NStepAccumulator(2, -0.01)
    with pytest.raises(ValueError):
        NStepAccumulator(2, 1.01)
    with pytest.raises(ValueError):
        NStepAccumulator(2, float("nan"))
    NStepAccumulator(1, 0.0)
    NStepAccumulator(4, 1.0)
    assert len(NStepAccumulator(3, 0.9)) == 0


def test_n1_matches_one_step_transition():
    gamma = 0.9
    acc = NStepAccumulator(1, gamma)
    state = torch.tensor([1.0, 0.0])
    next_state = torch.tensor([0.0, 1.0])
    mask = np.array([True, False, True, False])
    out = acc.push(state, 2, 1.25, next_state, False, mask)
    assert len(out) == 1
    step = out[0]
    assert step.state is state
    assert step.action == 2
    assert step.reward == pytest.approx(1.25)
    assert step.next_state is next_state
    assert step.done is False
    assert step.next_mask is mask
    assert step.discount == pytest.approx(gamma)
    assert len(acc) == 0

    terminal_mask = [False, False, False, True]
    done_out = acc.push("s", 0, torch.tensor(3.0), "ns", True, terminal_mask)
    assert len(done_out) == 1
    assert done_out[0].reward == pytest.approx(3.0)
    assert done_out[0].discount == pytest.approx(gamma)
    assert done_out[0].done is True
    assert done_out[0].next_state == "ns"
    assert done_out[0].next_mask is terminal_mask
    assert len(acc) == 0
    # The terminal flush must not leak into the next episode.
    again = acc.push("fresh", 1, 4.0, "fresh-ns", False, mask)
    assert len(again) == 1
    assert again[0].state == "fresh"
    assert again[0].reward == pytest.approx(4.0)
    assert again[0].next_state == "fresh-ns"


def test_n3_rewards_and_discount_are_hand_computed():
    gamma = 0.5
    rewards = [1.0, 2.0, 4.0, 8.0, 16.0]
    masks = [np.array([i == a for a in range(4)]) for i in range(5)]
    states = [np.full(4, i, dtype=np.float64) for i in range(6)]
    acc = NStepAccumulator(3, gamma)
    emissions = _push_seq(acc, rewards, masks=masks, states=states)
    assert emissions[0] == []
    assert emissions[1] == []
    assert len(emissions[2]) == 1
    assert len(emissions[3]) == 1
    assert len(emissions[4]) == 1
    assert len(acc) == 2

    for emit_i, start in enumerate(range(3)):
        step = emissions[start + 2][0]
        expected_r, expected_d = _hand_return(rewards, gamma, start, 3)
        assert step.reward == pytest.approx(expected_r)
        assert step.discount == pytest.approx(expected_d)
        assert step.discount == pytest.approx(gamma ** 3)
        assert step.state is states[start]
        assert step.action == start % 4
        assert step.next_state is states[start + 3]
        assert step.done is False
        assert step.next_mask is masks[start + 2]
        assert emit_i == start

    # Explicit arithmetic for the first window: 1 + 0.5*2 + 0.25*4 = 3.
    assert emissions[2][0].reward == pytest.approx(3.0)
    assert emissions[2][0].discount == pytest.approx(0.125)
    # Second window: 2 + 0.5*4 + 0.25*8 = 6.
    assert emissions[3][0].reward == pytest.approx(6.0)
    # Third window: 4 + 0.5*8 + 0.25*16 = 12.
    assert emissions[4][0].reward == pytest.approx(12.0)


def test_emission_timing_and_buffer_length():
    acc = NStepAccumulator(3, 0.99)
    assert acc.push("s0", 0, 1.0, "ns0", False, "m0") == []
    assert len(acc) == 1
    assert acc.push("s1", 1, 1.0, "ns1", False, "m1") == []
    assert len(acc) == 2
    emitted = acc.push("s2", 2, 1.0, "ns2", False, "m2")
    assert len(emitted) == 1
    assert emitted[0].state == "s0"
    assert emitted[0].next_state == "ns2"
    assert len(acc) == 2
    emitted = acc.push("s3", 3, 1.0, "ns3", False, "m3")
    assert len(emitted) == 1
    assert emitted[0].state == "s1"
    assert emitted[0].next_mask == "m3"
    assert len(acc) == 2


def test_done_mid_window_flushes_shortened_horizons():
    gamma = 0.5
    acc = NStepAccumulator(3, gamma)
    masks = [[True, False, False, False], [False, True, False, False]]
    assert acc.push("s0", 0, 1.0, "ns0", False, masks[0]) == []
    assert len(acc) == 1
    emitted = acc.push("s1", 1, 4.0, "ns1", True, masks[1])
    assert len(acc) == 0
    assert len(emitted) == 2

    # k = 2 for the oldest: R = 1 + 0.5*4 = 3, discount = 0.25.
    assert emitted[0].state == "s0"
    assert emitted[0].action == 0
    assert emitted[0].reward == pytest.approx(3.0)
    assert emitted[0].discount == pytest.approx(gamma ** 2)
    assert emitted[0].next_state == "ns1"
    assert emitted[0].next_mask is masks[1]
    assert emitted[0].done is True

    # k = 1 for the terminal step itself.
    assert emitted[1].state == "s1"
    assert emitted[1].action == 1
    assert emitted[1].reward == pytest.approx(4.0)
    assert emitted[1].discount == pytest.approx(gamma)
    assert emitted[1].next_state == "ns1"
    assert emitted[1].next_mask is masks[1]
    assert emitted[1].done is True

    # A full window that ends exactly on done still returns every tail.
    acc2 = NStepAccumulator(3, gamma)
    rewards = [1.0, 2.0, 4.0]
    last = None
    for i, reward in enumerate(rewards):
        last = acc2.push(f"s{i}", i, reward, f"ns{i}", i == 2, f"m{i}")
    assert len(acc2) == 0
    assert len(last) == 3
    for start, step in enumerate(last):
        k = 3 - start
        expected_r, expected_d = _hand_return(rewards, gamma, start, k)
        assert step.reward == pytest.approx(expected_r)
        assert step.discount == pytest.approx(expected_d)
        assert step.done is True
        assert step.next_state == "ns2"
        assert step.next_mask == "m2"
        assert step.state == f"s{start}"


def test_flush_truncation_keeps_done_false():
    gamma = 0.5
    acc = NStepAccumulator(3, gamma)
    rewards = [8.0, 16.0]
    masks = ["m0", "m1"]
    acc.push("s0", 0, rewards[0], "ns0", False, masks[0])
    acc.push("s1", 1, rewards[1], "ns1", False, masks[1])
    assert len(acc) == 2
    emitted = acc.flush()
    assert len(acc) == 0
    assert acc.flush() == []
    assert len(emitted) == 2

    assert emitted[0].reward == pytest.approx(8.0 + gamma * 16.0)
    assert emitted[0].discount == pytest.approx(gamma ** 2)
    assert emitted[0].done is False
    assert emitted[0].next_state == "ns1"
    assert emitted[0].next_mask == "m1"
    assert emitted[0].state == "s0"

    assert emitted[1].reward == pytest.approx(16.0)
    assert emitted[1].discount == pytest.approx(gamma)
    assert emitted[1].done is False
    assert emitted[1].next_state == "ns1"
    assert emitted[1].next_mask == "m1"
    assert emitted[1].state == "s1"


def test_reset_between_episodes_has_no_leakage():
    gamma = 0.5
    acc = NStepAccumulator(3, gamma)
    acc.push("old0", 0, 100.0, "old-ns0", False, "old-m0")
    acc.push("old1", 1, 200.0, "old-ns1", False, "old-m1")
    assert len(acc) == 2
    acc.reset()
    assert len(acc) == 0
    assert acc.flush() == []

    rewards = [1.0, 2.0, 4.0]
    emissions = _push_seq(acc, rewards)
    assert len(emissions[2]) == 1
    assert emissions[2][0].reward == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert emissions[2][0].state == "s0"
    assert emissions[2][0].next_state == "ns2"
    assert "old" not in str(emissions[2][0].state)


def test_gamma_zero_and_one():
    rewards = [5.0, 7.0, 9.0]
    acc0 = NStepAccumulator(3, 0.0)
    emissions = _push_seq(acc0, rewards)
    step = emissions[2][0]
    assert step.reward == pytest.approx(5.0)
    assert step.discount == pytest.approx(0.0)
    tail = acc0.flush()
    # Remaining starts at rewards[1] and rewards[2]. gamma**i is 0 for i >= 1.
    assert len(tail) == 2
    assert tail[0].reward == pytest.approx(7.0)
    assert tail[0].discount == pytest.approx(0.0)
    assert tail[0].done is False
    assert tail[1].reward == pytest.approx(9.0)
    assert tail[1].discount == pytest.approx(0.0)

    acc1 = NStepAccumulator(3, 1.0)
    emissions = _push_seq(acc1, rewards)
    step = emissions[2][0]
    assert step.reward == pytest.approx(5.0 + 7.0 + 9.0)
    assert step.discount == pytest.approx(1.0)
    done_out = acc1.push("s-term", 0, 1.5, "ns-term", True, "m-term")
    # Buffer held the last two of the previous window, plus this terminal step.
    assert [s.done for s in done_out] == [True, True, True]
    assert done_out[0].reward == pytest.approx(7.0 + 9.0 + 1.5)
    assert done_out[0].discount == pytest.approx(1.0)
    assert done_out[1].reward == pytest.approx(9.0 + 1.5)
    assert done_out[2].reward == pytest.approx(1.5)
    assert done_out[2].discount == pytest.approx(1.0)
    assert all(s.next_mask == "m-term" for s in done_out)
    assert len(acc1) == 0


def test_next_mask_comes_from_kth_step():
    acc = NStepAccumulator(3, 0.9)
    masks = [
        np.array([True, False, False, False]),
        np.array([False, True, False, False]),
        np.array([False, False, True, True]),
    ]
    out = []
    for i, mask in enumerate(masks):
        out = acc.push(f"s{i}", i, float(i + 1), f"ns{i}", False, mask)
    assert len(out) == 1
    assert out[0].next_mask is masks[2]
    assert out[0].next_state == "ns2"
    assert np.array_equal(out[0].next_mask, [False, False, True, True])


def test_double_dqn_online_picks_target_evaluates_and_masks():
    # Online prefers action 1 (value 9) but that action is illegal.
    # Among legal actions {0, 2, 3} online prefers 2. Target value there is 30,
    # which is not the online value and not the target's own argmax (action 0).
    online = _Table([[0.0, 9.0, 5.0, 1.0], [4.0, 1.0, 3.0, 2.0]])
    target = _Table([[100.0, 1.0, 30.0, 8.0], [7.0, 50.0, 9.0, 11.0]])
    rewards = torch.tensor([1.5, 0.0])
    discounts = torch.tensor([0.5, 0.25])
    dones = torch.tensor([False, False])
    masks = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, True],
        ]
    )
    next_states = torch.zeros(2, 4)
    y = double_dqn_targets(online, target, rewards, next_states, dones, masks, discounts)
    # Row 0: 1.5 + 0.5 * 30 = 16.5. Row 1 legal argmax is action 0 (online 4 > 1, 2).
    # Target at action 0 is 7, so 0 + 0.25 * 7 = 1.75.
    assert y.shape == (2,)
    assert not y.requires_grad
    assert torch.allclose(y, torch.tensor([16.5, 1.75]))
    assert all(param.grad is None for param in online.parameters())

    # Illegal action stays unchosen even when its online logit is +inf.
    online_inf = _Table([[1.0, float("inf"), 2.0, 0.0]])
    target_inf = _Table([[7.0, 8.0, 9.0, 10.0]])
    y_inf = double_dqn_targets(
        online_inf,
        target_inf,
        torch.tensor([0.0]),
        torch.zeros(1, 4),
        torch.tensor([False]),
        torch.tensor([[True, False, True, True]]),
        torch.tensor([1.0]),
    )
    assert torch.allclose(y_inf, torch.tensor([9.0]))


def test_double_dqn_done_and_all_masked_bootstrap_zero_and_discounts():
    online = _Table([[float("inf"), float("nan"), 5.0, -1e20]])
    # Gathered target entry would be NaN if the all-masked row leaked action 0.
    target = _Table([[float("nan"), float("inf"), 123.0, -40.0]])
    rewards = torch.tensor([2.5, 4.0, 0.0, -1.0])
    discounts = torch.tensor([0.9, 0.3, 0.5, 0.25])
    dones = torch.tensor([False, True, False, False])
    masks = torch.tensor(
        [
            [False, False, False, False],  # all invalid -> bootstrap 0
            [True, True, True, True],  # done -> bootstrap 0
            [False, False, True, False],  # only action 2, target 123
            [True, False, False, False],  # action 0, but we vary discount below
        ]
    )
    # Last row uses a finite online/target pair; rebuild tables per-row via a stub.
    class _Rows(nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.rows = nn.Parameter(torch.tensor(rows, dtype=torch.float32))

        def forward(self, states):
            return self.rows

    online_rows = _Rows(
        [
            [float("inf"), float("nan"), 5.0, -1e20],
            [3.0, 8.0, 1.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [2.0, 9.0, 9.0, 9.0],
        ]
    )
    target_rows = _Rows(
        [
            [float("nan"), float("inf"), 123.0, -40.0],
            [10.0, 10.0, 10.0, 10.0],
            [1.0, 1.0, 123.0, 5.0],
            [8.0, 100.0, 100.0, 100.0],
        ]
    )
    y = double_dqn_targets(
        online_rows,
        target_rows,
        rewards,
        torch.zeros(4, 3),
        dones,
        masks,
        discounts,
    )
    expected = torch.tensor(
        [
            2.5,  # all-masked, reward only, finite despite NaN/inf Q
            4.0,  # done, reward only (would be 4 + 0.3*10 if done were ignored)
            0.0 + 0.5 * 123.0,
            -1.0 + 0.25 * 8.0,  # discount 0.25 is used; a fixed gamma would not
        ]
    )
    assert torch.allclose(y, expected)
    assert torch.isfinite(y).all()
    # Same nets, swapped discounts, unchanged rewards/masks: the live row moves.
    y_swap = double_dqn_targets(
        online_rows,
        target_rows,
        rewards,
        torch.zeros(4, 3),
        dones,
        masks,
        torch.tensor([0.9, 0.3, 0.0, 1.0]),
    )
    assert torch.allclose(y_swap[0], torch.tensor(2.5))
    assert torch.allclose(y_swap[1], torch.tensor(4.0))
    assert torch.allclose(y_swap[2], torch.tensor(0.0))
    assert torch.allclose(y_swap[3], torch.tensor(-1.0 + 8.0))


def test_double_dqn_with_real_dqn_matches_manual_formula():
    from RL import DQN

    torch.manual_seed(0)
    online, target_net = DQN(), DQN()
    with torch.no_grad():
        for param in target_net.parameters():
            param.add_(0.05)
    batch = 5
    next_states = torch.randn(batch, 16)
    rewards = torch.tensor([0.2, -1.0, 3.0, 0.0, 1.5])
    dones = torch.tensor([False, True, False, False, True])
    masks = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
            [False, False, False, False],
            [False, True, False, False],
            [True, True, False, True],
        ]
    )
    discounts = torch.tensor([0.9, 0.81, 0.1, 0.0, 0.5])
    y = double_dqn_targets(online, target_net, rewards, next_states, dones, masks, discounts)
    with torch.no_grad():
        online_q = online(next_states)
        target_q = target_net(next_states)
        neg = torch.finfo(online_q.dtype).min
        chosen_action = online_q.masked_fill(~masks, neg).argmax(dim=1)
        chosen = target_q.gather(1, chosen_action.view(-1, 1)).squeeze(1)
        bootstrap = torch.where(masks.any(1) & ~dones, chosen, torch.zeros_like(chosen))
        manual = rewards + discounts * bootstrap
    assert y.shape == (batch,)
    assert torch.allclose(y, manual)
    assert torch.isfinite(y).all()


def test_is_weights_scale_loss_and_zero_weight_has_no_gradient():
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    actions = torch.tensor([0, 1])
    rewards = [4.0, 6.0]
    dones = [True, True]
    masks = [[True, True, True, True], [True, True, True, True]]
    discounts = [0.9, 0.9]
    batch = _fixed_batch(
        rewards,
        dones,
        masks,
        discounts,
        states=states,
        next_states=torch.zeros(2, 2),
        actions=actions,
    )

    def fresh():
        online = _OrthQ()
        target = _OrthQ()
        optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
        return online, target, optimizer

    online_a, target_a, opt_a = fresh()
    stats_ones, td_ones = dqn_batch_update(
        online_a, target_a, opt_a, grad_clip=1e9, is_weights=torch.ones(2), **batch
    )
    # Q starts at 0 and both samples are terminal, so y = reward.
    # Mean squared error is (16 + 36) / 2 = 26. Quarter weights scale that by 0.25.
    assert stats_ones["loss"] == pytest.approx(26.0)
    assert td_ones.shape == (2,)
    assert td_ones == pytest.approx(np.array([4.0, 6.0], dtype=td_ones.dtype))
    assert stats_ones["td_abs_mean"] == pytest.approx(float(td_ones.mean()))
    assert stats_ones["target_mean"] == pytest.approx(5.0)
    assert stats_ones["q_mean"] == pytest.approx(0.0)
    assert stats_ones["q_min"] == pytest.approx(0.0)
    assert stats_ones["q_max"] == pytest.approx(0.0)

    online_b, target_b, opt_b = fresh()
    stats_quarter, _ = dqn_batch_update(
        online_b,
        target_b,
        opt_b,
        grad_clip=1e9,
        is_weights=torch.full((2,), 0.25),
        **batch,
    )
    assert stats_quarter["loss"] == pytest.approx(stats_ones["loss"] * 0.25)

    online_z, target_z, opt_z = fresh()
    before = online_z.w.detach().clone()
    stats_zero, _ = dqn_batch_update(
        online_z,
        target_z,
        opt_z,
        grad_clip=1e9,
        is_weights=torch.tensor([0.0, 1.0]),
        **batch,
    )
    # Sample 0 is the only user of w[0, 0]. A zero IS weight leaves it untouched.
    # Sample 1's unclipped grad w.r.t. w[1, 1] is (q - y) = -6, times lr 0.1.
    assert stats_zero["loss"] == pytest.approx(18.0)
    assert online_z.w[0, 0].item() == pytest.approx(0.0)
    assert online_z.w[0].detach().tolist() == pytest.approx(before[0].tolist())
    assert online_z.w[1, 1].item() == pytest.approx(0.6)
    assert online_z.w[1, 0].item() == pytest.approx(0.0)


def test_loss_decreases_and_td_abs_shape():
    torch.manual_seed(0)
    online = _BiasQ(0.0)
    target = _BiasQ(0.0)
    optimizer = torch.optim.SGD(online.parameters(), lr=0.05)
    batch = _fixed_batch(
        rewards=[1.0, 1.0, 1.0, 1.0],
        dones=[True, True, True, True],
        masks=[[True, False, True, True]] * 4,
        discounts=[0.99, 0.5, 0.0, 1.0],
        actions=torch.zeros(4, dtype=torch.long),
    )
    losses = []
    td_abs = None
    for _ in range(5):
        stats, td_abs = dqn_batch_update(
            online,
            target,
            optimizer,
            is_weights=np.ones(4, dtype=np.float64),
            **batch,
        )
        losses.append(stats["loss"])
        assert set(stats) == {
            "loss",
            "q_mean",
            "q_min",
            "q_max",
            "target_mean",
            "td_abs_mean",
        }
    assert td_abs.shape == (4,)
    assert isinstance(td_abs, np.ndarray)
    assert losses[0] == pytest.approx(1.0)
    assert losses[-1] < losses[0]
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    # Target net is not stepped.
    assert torch.equal(target.bias.detach(), torch.zeros(4))
    assert online.bias.detach()[0].item() > 0.0


def test_huber_loss_is_below_mse_for_large_errors():
    batch = _fixed_batch(
        rewards=[10.0, 10.0],
        dones=[True, True],
        masks=[[True, True, True, True], [True, True, True, True]],
        discounts=[1.0, 1.0],
    )

    def once(huber):
        online = _BiasQ(0.0)
        target = _BiasQ(0.0)
        optimizer = torch.optim.SGD(online.parameters(), lr=0.01)
        stats, td_abs = dqn_batch_update(
            online,
            target,
            optimizer,
            is_weights=torch.ones(2),
            huber=huber,
            **batch,
        )
        return stats, td_abs

    mse, td_mse = once(False)
    huber, td_huber = once(True)
    assert mse["loss"] == pytest.approx(100.0)
    assert huber["loss"] == pytest.approx(9.5)
    assert huber["loss"] < mse["loss"]
    assert td_mse.shape == (2,)
    assert td_huber == pytest.approx(np.array([10.0, 10.0]))


def test_non_finite_loss_raises_and_does_not_step():
    online = _BiasQ(0.0)
    target = _BiasQ(0.0)
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
    before = online.bias.detach().clone()
    batch = _fixed_batch(
        rewards=[float("nan"), 1.0],
        dones=[True, True],
        masks=[[True, True, True, True], [False, False, False, False]],
        discounts=[0.9, 0.9],
    )
    with pytest.raises(FloatingPointError):
        dqn_batch_update(
            online,
            target,
            optimizer,
            is_weights=torch.ones(2),
            **batch,
        )
    assert torch.equal(online.bias.detach(), before)
    assert optimizer.state == {}

    online_inf = _BiasQ(0.0)
    before_inf = online_inf.bias.detach().clone()
    batch_inf = _fixed_batch(
        rewards=[float("inf"), 0.0],
        dones=[True, True],
        masks=[[True, False, False, False]] * 2,
        discounts=[0.0, 0.0],
    )
    with pytest.raises(FloatingPointError):
        dqn_batch_update(
            online_inf,
            _BiasQ(0.0),
            torch.optim.SGD(online_inf.parameters(), lr=0.1),
            is_weights=[1.0, 1.0],
            **batch_inf,
        )
    assert torch.equal(online_inf.bias.detach(), before_inf)
