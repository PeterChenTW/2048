"""Adversarial checks for vectorized afterstate expansion.

Covers full-game acting (including exploration), lookup-table reentry,
out-of-range fallback, and mixed batches. These tests are independent of the
stage-1/2 suites.
"""
import random

import numpy as np
import pytest
import torch
from torch import nn

from RL import load_checkpoint
from afterstate import (
    _LINE_INDICES,
    _row_lookup_tables,
    afterstates,
    batch_afterstates,
    batch_afterstates_reference,
    batch_afterstates_vectorized,
)
from afterstate_agent import AfterstateAgent
from afterstate_policy import greedy_actions, merge_reward, select_action
from config import DQNConfig
from fast_board import FastBoard, _LINES, _merge


class _TinyValue(nn.Module):
    """Deterministic stand-in so a full game does not run the 128-channel net."""

    def __init__(self, tilt=0.0):
        super().__init__()
        self.register_buffer("w", torch.linspace(-0.25, 0.4, 16))
        self.tilt = float(tilt)
        self.calls = 0

    def forward(self, states):
        self.calls += 1
        values = states.reshape(-1, 16).to(dtype=torch.float32)
        return (values * self.w).sum(-1) + self.tilt


def _agent(expand, seed, epsilon):
    agent = AfterstateAgent(
        16, 4,
        DQNConfig(arch="onehot_afterstate", seed=seed, afterstate_expand=expand,
                  memory_capacity=64, batch_size=8),
    )
    agent.model = _TinyValue(0.0)
    agent.target_model = _TinyValue(0.37)
    agent.epsilon = epsilon
    agent.rng = np.random.default_rng(seed + 99)
    return agent


def _play(expand, seed, epsilon, grid):
    """Mirror main.py: act, take that action's afterstate, then board.move."""
    agent = _agent(expand, seed, epsilon)
    board_rng = random.Random(seed + 3)
    board = FastBoard(rng=board_rng) if grid is None else FastBoard.from_grid(grid, rng=board_rng)
    trace = []
    explore_not_greedy = 0
    inputs_past_table = 0
    while not board.done:
        invalid = board.get_invalid_moves()
        if len(invalid) == 4:
            break
        raw = np.asarray(board.grid, dtype=np.int64).reshape(16)
        if int(raw.max()) >= 2 ** 16:
            inputs_past_table += 1
        state = torch.as_tensor(board.state, dtype=torch.float32)
        assert np.array_equal(state.detach().cpu().numpy().astype(np.int64).reshape(16), raw)
        legal = afterstates(raw)[2]
        valid = [action for action in range(4) if legal[action]]
        probe = np.random.default_rng()
        probe.bit_generator.state = agent.rng.bit_generator.state
        exploring = bool(probe.random() < epsilon)
        predicted = (int(probe.choice(valid)) if exploring
                     else int(greedy_actions(agent.model, raw, expand="reference")[0]))
        calls_before = agent.model.calls
        action = agent.act(state, invalid)
        if exploring:
            assert agent.model.calls == calls_before
        else:
            assert agent.model.calls == calls_before + 1
        if expand == "vectorized":
            after_state = agent.last_after_state
            assert after_state is agent.last_after_state
        else:
            assert agent.last_after_state is None
            after_state = afterstates(board.grid)[0][action]
        snapshot = np.array(after_state, dtype=np.int64, copy=True)
        after_state[...] = -7  # must not alias the board, the table, or the next call
        assert action == predicted
        pre_spawn, score, changed = board.pure_move(action)
        assert changed and score == int(afterstates(raw)[1][action])
        np.testing.assert_array_equal(pre_spawn, snapshot)
        np.testing.assert_array_equal(snapshot, afterstates(raw)[0][action])
        assert not np.shares_memory(snapshot, board._cells)
        if len(valid) > 1 and action != int(greedy_actions(agent.model, raw)[0]):
            explore_not_greedy += 1
        next_grid, reward, done = board.move(board.all_moves[action])
        spawned = np.asarray(next_grid, dtype=np.int64).reshape(16) - snapshot
        spawned_at = np.flatnonzero(spawned)
        assert spawned_at.size == 1
        assert int(snapshot[spawned_at[0]]) == 0
        assert int(spawned[spawned_at[0]]) in (2, 4)
        trace.append((
            action,
            tuple(int(v) for v in snapshot),
            tuple(int(v) for v in np.asarray(next_grid).reshape(-1)),
            float(reward),
            bool(done),
            int(agent.rng.bit_generator.state["state"]["state"]),
        ))
        if done:
            break
    assert trace
    return trace, explore_not_greedy, inputs_past_table, agent.model.calls


def _assert_modes_match(seed, epsilon, grid):
    reference, ref_explore, ref_hi, ref_calls = _play("reference", seed, epsilon, grid)
    vectorized, vec_explore, vec_hi, vec_calls = _play("vectorized", seed, epsilon, grid)
    assert reference == vectorized
    assert ref_explore == vec_explore
    assert ref_hi == vec_hi
    assert ref_calls == vec_calls
    return reference, ref_explore, ref_hi


@pytest.mark.parametrize("epsilon, min_explore", [(1.0, 1), (0.0, 0)])
def test_full_game_afterstate_matches_move_in_both_modes(epsilon, min_explore):
    trace, explore_not_greedy, hi_inputs = _assert_modes_match(2102, epsilon, None)
    assert len(trace) > 50
    if epsilon == 0.0:
        assert explore_not_greedy == 0
    else:
        assert explore_not_greedy >= min_explore
    assert hi_inputs == 0


def test_full_game_with_tiles_past_lookup_uses_same_actions():
    """2**16 and 2**17 are outside the row table and must fallback for the whole game."""
    grid = np.array([
        [2 ** 16, 2 ** 8, 2 ** 4, 2 ** 2],
        [2 ** 17, 2 ** 3, 0, 0],
        [0, 2 ** 5, 2 ** 6, 2 ** 7],
        [2, 0, 4, 8],
    ], dtype=np.int64)
    for epsilon, expect_explore in ((1.0, True), (0.0, False), (0.5, True)):
        trace, explore_not_greedy, hi_inputs = _assert_modes_match(3, epsilon, grid)
        assert hi_inputs == len(trace)
        assert (explore_not_greedy > 0) is expect_explore


def test_repeated_calls_ignore_cached_outputs_and_fallback_traffic():
    table_after, table_merge = _row_lookup_tables()
    # Every sampled key was produced by fast_board._merge, including the packed corners.
    shifts = np.array([0, 4, 8, 12], dtype=np.uint16)
    tiles = np.array([0] + [1 << exponent for exponent in range(1, 16)], dtype=np.int64)
    keys = np.unique(np.concatenate([
        np.arange(0, 1 << 16, 17, dtype=np.uint16),
        np.array([0, 1, 15, 65535], dtype=np.uint16),
    ]))
    decoded = tiles[(keys[:, None] >> shifts) & 15]
    for key, row in zip(keys.tolist(), decoded.tolist()):
        merged, score = _merge(tuple(int(value) for value in row))
        np.testing.assert_array_equal(table_after[key], merged)
        assert int(table_merge[key]) == score
    with pytest.raises(ValueError):
        table_after[0, 0] = 1

    normal = np.array([[2, 2, 0, 4] + [0] * 12, [2 ** 15] * 16], dtype=np.int64)
    expected = batch_afterstates_reference(normal)
    first = batch_afterstates_vectorized(normal)
    assert not np.shares_memory(first[0], normal)
    assert not np.shares_memory(first[0], table_after)
    assert not np.shares_memory(first[1], table_merge)
    first[0].fill(-5)
    first[1].fill(-5)
    second = batch_afterstates_vectorized(normal)
    for actual, reference in zip(second, expected):
        np.testing.assert_array_equal(actual, reference)
    assert not np.shares_memory(first[0], second[0])

    mixed = np.stack([
        normal[0],
        np.array([2 ** 16] + [0] * 15, dtype=np.int64),
        np.array([2 ** 17, 2, 2, 0] + [0] * 12, dtype=np.int64),
    ])
    mixed_expected = batch_afterstates_reference(mixed)
    mixed_actual = batch_afterstates_vectorized(mixed)
    mixed_actual[0].fill(9)
    rerun = batch_afterstates_vectorized(normal)
    for actual, reference in zip(rerun, expected):
        np.testing.assert_array_equal(actual, reference)
    restored = batch_afterstates_vectorized(mixed)
    for actual, reference in zip(restored, mixed_expected):
        np.testing.assert_array_equal(actual, reference)


def test_fallback_boundaries_and_mixed_batch_match_reference():
    # Each line index set is a permutation, so scattering cannot leave a cell stale.
    for lines in _LINES:
        assert sorted(index for line in lines for index in line) == list(range(16))
    assert _LINE_INDICES.shape == (4, 4, 4)

    hostile = [1, 3, 6, 12, 24, -1, -2, -4, 2 ** 16, 2 ** 17, 2 ** 18,
               2 ** 15 + 1, 2 ** 30, 2 ** 62, -(2 ** 63), 2 ** 63 - 1]
    boards = np.zeros((len(hostile) + 2, 16), dtype=np.int64)
    boards[0, :4] = [2, 2, 4, 8]
    boards[-1] = 2 ** 15
    for index, tile in enumerate(hostile, start=1):
        boards[index, index % 16] = tile
        boards[index, (index + 5) % 16] = 8
    _assert_expand_equal(boards)
    _assert_expand_equal(boards.reshape(-1, 4, 4))
    # Non-contiguous view of a mixed batch, plus a plain Python row with 2**17.
    stride = np.zeros((boards.shape[0], 32), dtype=np.int64)
    stride[:, ::2] = boards
    _assert_expand_equal(stride[:, ::2])
    python_rows = [
        [2, 2, 0, 0] * 4,
        [2 ** 16, 0, 2 ** 17, 4] + [0] * 12,
        [0] * 16,
        [-8, 2, 2, 1] + [0] * 12,
    ]
    _assert_expand_equal(python_rows)
    # Default dispatch stays on the reference implementation and is bit-exact.
    default = batch_afterstates(python_rows)
    reference = batch_afterstates_reference(python_rows)
    for actual, expected in zip(default, reference):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def _assert_expand_equal(grids):
    reference = batch_afterstates_reference(grids)
    vectorized = batch_afterstates_vectorized(grids)
    for actual, expected in zip(vectorized, reference):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    after, merge, legal = vectorized
    assert after.dtype == np.int64 and merge.dtype == np.int64 and legal.dtype == np.bool_
    np.testing.assert_array_equal(after[~legal], np.broadcast_to(
        np.asarray(grids, dtype=np.int64).reshape(after.shape[0], 1, 16), after.shape)[~legal])
    assert np.all(merge[~legal] == 0)


def test_default_reference_rng_order_and_td_definition():
    model = _TinyValue(0.0)
    grid = np.array([2, 2, 4, 0, 0, 8, 8, 2] + [0] * 8, dtype=np.int64)
    legal = afterstates(grid)[2]
    invalid = [int(action) for action in np.flatnonzero(~legal)]
    for epsilon in (0.0, 0.4, 1.0):
        generators = [np.random.default_rng(123) for _ in range(4)]
        default_action = select_action(model, grid, invalid, epsilon, generators[0])
        reference_action, reference_after = select_action(
            model, grid, invalid, epsilon, generators[1], expand="reference", return_after=True)
        vectorized_action, vectorized_after = select_action(
            model, grid, invalid, epsilon, generators[2], expand="vectorized", return_after=True)
        manual = generators[3]
        if manual.random() < epsilon:
            manual_action = int(manual.choice([action for action in range(4) if legal[action]]))
        else:
            manual_action = int(greedy_actions(model, grid)[0])
        assert default_action == reference_action == vectorized_action == manual_action
        states = [gen.bit_generator.state for gen in generators]
        assert states[0] == states[1] == states[2] == states[3]
        np.testing.assert_array_equal(reference_after, afterstates(grid)[0][manual_action])
        np.testing.assert_array_equal(vectorized_after, reference_after)
        assert vectorized_after.shape == (16,) and vectorized_after.dtype == np.int64
        assert not np.shares_memory(vectorized_after, reference_after)
        assert not np.shares_memory(vectorized_after, grid)

    agent = _agent("reference", 5, 0.0)
    rng = np.random.default_rng(1)
    boards = np.zeros((6, 16), dtype=np.int64)
    for index in range(4):
        cells = rng.integers(0, 16, size=8)
        boards[index, cells] = 2 ** rng.integers(1, 12, size=8)
    boards[3, 0] = 2 ** 16
    boards[3, 1] = 2 ** 17
    boards[4, 0] = 2
    boards[5] = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
                          dtype=np.int64).reshape(16)
    dones = np.array([False, False, True, False, False, False])
    gamma = agent.config.gamma
    after, merge, legal = batch_afterstates_reference(boards)
    rewards = np.where(legal, merge_reward(merge), np.float32(0.0)).astype(np.float32)
    scores = torch.full((len(boards), 4), -torch.inf)
    online = agent.model(torch.as_tensor(after[legal], dtype=torch.float32)).reshape(-1)
    scores[torch.as_tensor(legal)] = torch.as_tensor(rewards[legal]) + gamma * online
    manual_targets = torch.zeros(len(boards), dtype=torch.float32)
    for index, done in enumerate(dones):
        if done or not legal[index].any():
            continue
        choice = int(scores[index].argmax().item())
        value = agent.target_model(torch.as_tensor(after[index, choice][None], dtype=torch.float32))
        manual_targets[index] = torch.as_tensor(rewards[index, choice]) + gamma * value.reshape(-1)[0]
    states = torch.as_tensor(boards, dtype=torch.float32)
    done_tensor = torch.as_tensor(dones)
    for expand in ("reference", "vectorized"):
        agent.config.afterstate_expand = expand
        targets = agent._targets(states, done_tensor)
        assert torch.equal(targets, manual_targets)
        assert torch.all(targets[done_tensor] == 0)


def test_checkpoint_keeps_expand_mode_and_old_files_default_reference(tmp_path):
    agent = AfterstateAgent(
        16, 4,
        DQNConfig(arch="onehot_afterstate", seed=1, afterstate_expand="vectorized",
                  memory_capacity=32, batch_size=4),
    )
    agent.last_after_state = np.arange(16)
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)
    loaded = load_checkpoint(checkpoint)
    assert loaded.config.afterstate_expand == "vectorized"
    assert loaded.last_after_state is None
    for name, weight in agent.model.state_dict().items():
        assert torch.equal(weight, loaded.model.state_dict()[name])
    receiver = AfterstateAgent(
        16, 4,
        DQNConfig(arch="onehot_afterstate", seed=2, afterstate_expand="reference",
                  memory_capacity=32, batch_size=4),
    )
    receiver.last_after_state = np.ones(16, dtype=np.int64)
    receiver.load(checkpoint)
    assert receiver.config.afterstate_expand == "vectorized"
    assert receiver.last_after_state is None
    payload = torch.load(checkpoint, weights_only=True)
    del payload["config"]["afterstate_expand"]
    torch.save(payload, checkpoint)
    assert load_checkpoint(checkpoint).config.afterstate_expand == "reference"
    receiver.config.afterstate_expand = "vectorized"
    receiver.last_after_state = np.arange(16)
    receiver.load(checkpoint)
    assert receiver.config.afterstate_expand == "reference"
    assert receiver.last_after_state is None
