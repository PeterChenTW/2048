import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afterstate import afterstates, is_terminal
from afterstate_policy import (
    afterstate_scores,
    greedy_actions,
    merge_reward,
    select_action,
)
import eval as eval_mod
from fast_board import FastBoard
from value_net import OneHotValueNet


class LookupValueModel(nn.Module):
    """Stub V(afterstate) model returning values from a lookup table or default."""

    def __init__(self, table=None, default: float = 0.0):
        super().__init__()
        items = table.items() if isinstance(table, dict) else (table or [])
        self.table = {tuple(int(v) for v in k): float(val) for k, val in items}
        self.default = float(default)
        self.call_count = 0
        self.batch_sizes = []
        self.grad_enabled = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        arr = x.detach().cpu().numpy().astype(np.int64).reshape(-1, 16)
        self.batch_sizes.append(arr.shape[0])
        vals = [self.table.get(tuple(row.tolist()), self.default) for row in arr]
        return torch.tensor(vals, dtype=torch.float32)


class WeightedTileModel(nn.Module):
    """Deterministic stub model computing a weighted sum of tiles."""

    def __init__(self):
        super().__init__()
        weights = np.arange(1, 17, dtype=np.float32) / 16.0
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        flat = x.reshape(-1, 16).to(dtype=torch.float32)
        return (flat * self.weights).sum(dim=-1)


TERMINAL_GRID = np.array(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ],
    dtype=np.int64,
)


def test_merge_reward_modes():
    merges = np.array([[0, 4, 8, 256], [16, 0, 64, 512]], dtype=np.int64)
    log_r = merge_reward(merges, mode="log")
    assert log_r.shape == (2, 4)
    assert log_r.dtype == np.float32
    np.testing.assert_allclose(log_r, np.log2(1.0 + merges) / 16.0, rtol=1e-6)
    assert log_r[0, 0] == 0.0

    lin_r = merge_reward(merges, mode="linear", scale=256.0)
    assert lin_r.shape == (2, 4)
    assert lin_r.dtype == np.float32
    np.testing.assert_allclose(lin_r, merges / 256.0, rtol=1e-6)

    lin_r_custom = merge_reward(merges, mode="linear", scale=64.0)
    np.testing.assert_allclose(lin_r_custom, merges / 64.0, rtol=1e-6)

    with pytest.raises(ValueError):
        merge_reward(merges, mode="unknown")
    with pytest.raises(ValueError):
        merge_reward(merges, mode="linear", scale=0.0)
    with pytest.raises(ValueError):
        merge_reward(merges, mode="linear", scale=-10.0)


def test_afterstate_scores_shapes_and_single_batched_no_grad_forward():
    # Board with 3 legal moves (left, down, right) and 1 illegal move (up)
    grid = np.array(
        [
            [4, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    model = LookupValueModel(default=2.0)
    res = afterstate_scores(model, grid, gamma=0.99)

    # Supports both tuple unpacking and dict indexing
    after, rewards, legal, scores = res
    assert np.array_equal(after, res["after"])
    assert np.array_equal(rewards, res["rewards"])
    assert np.array_equal(legal, res["legal"])
    assert torch.equal(scores, res["scores"])

    assert after.shape == (1, 4, 16) and after.dtype == np.int64
    assert rewards.shape == (1, 4) and rewards.dtype == np.float32
    assert legal.shape == (1, 4) and legal.dtype == bool
    assert scores.shape == (1, 4) and scores.dtype == torch.float32

    assert legal.tolist() == [[True, True, True, False]]
    assert rewards[0, 3] == 0.0
    assert scores[0, 3].item() == -float("inf")
    assert torch.isfinite(scores[0, :3]).all()

    # Model called ONCE, under torch.no_grad(), over ONLY the 3 legal afterstates
    assert model.call_count == 1
    assert model.batch_sizes == [3]
    assert model.grad_enabled == [False]

    # Accepts (16,), (4, 4), (N, 16), (N, 4, 4)
    for shaped in (grid.reshape(16), grid.reshape(4, 4), grid.reshape(1, 16), grid.reshape(1, 4, 4)):
        out = afterstate_scores(model, shaped, gamma=0.99)
        assert torch.allclose(out.scores[:, :3], scores[:, :3])
        assert out.scores[0, 3].item() == -float("inf")


def test_argmax_includes_reward_term_and_gamma_v():
    # Action 0 (left): [8, 0, 0, 0, ...], merge = 8, legal = True
    # Action 1 (down): [..., 4, 4, 0, 0], merge = 0, legal = True
    # Action 2 (right): [0, 0, 0, 8, ...], merge = 8, legal = True
    # Action 3 (up): unchanged, merge = 0, legal = False
    grid = np.array(
        [
            [4, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    after, merge, legal = afterstates(grid)
    assert merge.tolist() == [8, 0, 8, 0]
    assert legal.tolist() == [True, True, True, False]

    # Case 1: Higher merge reward flips the choice vs V alone.
    # V alone prefers action 1 (V=0.10) over action 0 (V=0.0) and action 2 (V=-1.0),
    # but action 0 has merge reward log2(1+8)/16 = log2(9)/16 ~= 0.1981 > 0.99 * 0.10 = 0.099.
    model_reward_flips = LookupValueModel(
        table={
            tuple(after[0]): 0.0,
            tuple(after[1]): 0.10,
            tuple(after[2]): -1.0,
        }
    )
    choice1 = greedy_actions(model_reward_flips, grid, gamma=0.99, reward_mode="log")
    assert choice1.shape == (1,) and choice1.dtype == np.int64
    assert choice1[0] == 0

    # Case 2: gamma * V dominates the higher merge reward.
    # Action 1 has 0 merge reward, but V(after[1]) = 1.0 -> 0 + 0.99 * 1.0 = 0.99 > 0.1981 + 0.
    model_v_dominates = LookupValueModel(
        table={
            tuple(after[0]): 0.0,
            tuple(after[1]): 1.0,
            tuple(after[2]): 0.0,
        }
    )
    choice2 = greedy_actions(model_v_dominates, grid, gamma=0.99, reward_mode="log")
    assert choice2[0] == 1

    # When gamma = 0.0 on the exact same model, merge reward alone decides -> action 0.
    choice_gamma0 = greedy_actions(model_v_dominates, grid, gamma=0.0, reward_mode="log")
    assert choice_gamma0[0] == 0


def test_illegal_actions_never_chosen_even_if_v_huge():
    # On row 0 = [2, 0, 0, 0], action 0 (left) and action 3 (up) are ILLEGAL;
    # action 1 (down) and action 2 (right) are LEGAL.
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, 0] = 2
    after, _, legal = afterstates(grid)
    assert legal.tolist() == [False, True, True, False]

    # Assign enormous V to the illegal afterstate (unchanged board) and negative V to legal ones.
    model = LookupValueModel(
        table={
            tuple(after[0]): 1e9,   # illegal afterstate (unchanged board)
            tuple(after[1]): -5.0,  # legal (down)
            tuple(after[2]): -2.0,  # legal (right)
            tuple(after[3]): 1e9,   # illegal afterstate
        }
    )
    res = afterstate_scores(model, grid, gamma=0.99)
    assert res.scores[0, 0].item() == -float("inf")
    assert res.scores[0, 3].item() == -float("inf")
    # Only the 2 legal afterstates were passed to the model
    assert model.batch_sizes == [2]

    act = greedy_actions(model, grid, gamma=0.99)
    assert act[0] == 2


def test_batched_matches_per_board():
    rng = np.random.default_rng(123)
    boards = []
    for seed in range(10):
        b = FastBoard(rng=random.Random(seed))
        for _ in range(seed + 1):
            inv = b.get_invalid_moves()
            valid = [a for a in range(4) if a not in inv]
            if not valid:
                break
            b.move(FastBoard.all_moves[valid[int(rng.integers(0, len(valid)))]])
        boards.append(np.array(b.grid, dtype=np.int64))
    batch_4x4 = np.stack(boards, axis=0)
    batch_16 = batch_4x4.reshape(-1, 16)

    for model in (WeightedTileModel(), OneHotValueNet(channels=16, hidden=32)):
        model.eval()
        batched_res = afterstate_scores(model, batch_4x4, gamma=0.95, reward_mode="linear", reward_scale=128.0)
        batched_act = greedy_actions(model, batch_16, gamma=0.95, reward_mode="linear", reward_scale=128.0)

        for i in range(len(boards)):
            single_res = afterstate_scores(model, boards[i], gamma=0.95, reward_mode="linear", reward_scale=128.0)
            single_act = greedy_actions(model, boards[i].reshape(16), gamma=0.95, reward_mode="linear", reward_scale=128.0)

            assert np.array_equal(batched_res.after[i], single_res.after[0])
            np.testing.assert_allclose(batched_res.rewards[i], single_res.rewards[0])
            assert np.array_equal(batched_res.legal[i], single_res.legal[0])
            assert torch.allclose(batched_res.scores[i], single_res.scores[0], equal_nan=True)
            assert batched_act[i] == single_act[0]


def test_ties_break_to_lowest_action_index():
    const_model = LookupValueModel(default=1.0)

    # All 4 actions legal with 0 merge reward -> ties across [0, 1, 2, 3] -> picks 0
    grid_all_legal = np.zeros((4, 4), dtype=np.int64)
    grid_all_legal[1, 1] = 2
    assert greedy_actions(const_model, grid_all_legal, gamma=0.99)[0] == 0

    # Tile at top-left (0, 0): legal actions are [1 (down), 2 (right)] with equal score -> picks 1
    grid_1_2 = np.zeros((4, 4), dtype=np.int64)
    grid_1_2[0, 0] = 2
    assert greedy_actions(const_model, grid_1_2, gamma=0.99)[0] == 1

    # Tile at bottom-left (3, 0): left (0) and down (1) illegal; right (2) and up (3) legal -> picks 2
    grid_2_3 = np.zeros((4, 4), dtype=np.int64)
    grid_2_3[3, 0] = 2
    assert greedy_actions(const_model, grid_2_3, gamma=0.99)[0] == 2


def test_select_action_epsilon_rng_and_board_rng_untouched():
    # Tile at (0, 0): legal moves are [1, 2], invalid moves are [0, 3]
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, 0] = 2
    board_rng = random.Random(777)
    board = FastBoard.from_grid(grid, rng=board_rng)
    board_rng_state = board.rng.getstate()
    invalid = board.get_invalid_moves()
    assert invalid == [0, 3]

    after, _, _ = afterstates(grid)
    model = LookupValueModel(table={tuple(after[1]): 0.1, tuple(after[2]): 5.0})

    # epsilon=0.0 consumes EXACTLY one rng.random() and picks greedy action (2)
    rng = np.random.default_rng(42)
    ref_rng = np.random.default_rng(42)
    _ = ref_rng.random()

    action = select_action(model, board.grid, invalid, epsilon=0.0, rng=rng, gamma=0.99)
    assert action == 2
    assert rng.bit_generator.state == ref_rng.bit_generator.state
    assert board.rng.getstate() == board_rng_state

    # epsilon=1.0 picks only legal actions {1, 2} and roughly uniformly
    rng_explore = np.random.default_rng(12345)
    counts = {1: 0, 2: 0}
    for _ in range(400):
        a = select_action(model, board.grid, invalid, epsilon=1.0, rng=rng_explore, gamma=0.99)
        assert a in (1, 2)
        counts[a] += 1
    assert 150 <= counts[1] <= 250
    assert 150 <= counts[2] <= 250
    assert board.rng.getstate() == board_rng_state

    # Disagreement between invalid_moves and afterstate legality raises
    with pytest.raises((ValueError, AssertionError)):
        select_action(model, board.grid, [0], epsilon=0.0, rng=np.random.default_rng(0), gamma=0.99)
    with pytest.raises((ValueError, AssertionError)):
        select_action(model, board.grid, [0, 1, 3], epsilon=1.0, rng=np.random.default_rng(0), gamma=0.99)


def test_terminal_grid_raises():
    assert is_terminal(TERMINAL_GRID)
    model = LookupValueModel(default=10.0)

    res = afterstate_scores(model, TERMINAL_GRID, gamma=0.99)
    assert not np.any(res.legal)
    assert torch.all(res.scores == -torch.inf)
    assert model.call_count == 0

    with pytest.raises(ValueError):
        greedy_actions(model, TERMINAL_GRID, gamma=0.99)

    with pytest.raises(ValueError):
        select_action(model, TERMINAL_GRID, [0, 1, 2, 3], epsilon=0.0, rng=np.random.default_rng(0), gamma=0.99)


def test_eval_adapter_uses_greedy_action_and_takes_only_legal_moves(monkeypatch):
    stub_model = WeightedTileModel()

    class FakeAfterstateAgent:
        def __init__(self):
            self.calls = 0

        def greedy_action(self, grid, invalid):
            self.calls += 1
            fb = FastBoard.from_grid(grid)
            assert list(invalid) == fb.get_invalid_moves()
            act = int(greedy_actions(stub_model, grid, gamma=0.99)[0])
            assert act not in invalid
            return act

    fake_agent = FakeAfterstateAgent()
    monkeypatch.setattr(eval_mod, "load_agent", lambda checkpoint: fake_agent)

    result = eval_mod.evaluate(
        checkpoint="ignored",
        policy="greedy",
        games=2,
        seed_start=1000,
        board="fast",
        max_steps=300,
    )
    assert result["games"] == 2
    assert len(result["per_game"]) == 2
    total_steps = sum(r["steps"] for r in result["per_game"])
    assert total_steps > 0
    assert fake_agent.calls == total_steps
