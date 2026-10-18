"""Expansion backend parity for policy, Double TD targets and the training driver."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import afterstate_policy as policy
from afterstate import EXPAND_MODES, afterstates, batch_afterstates_reference
from afterstate_agent import AfterstateAgent
from config import DQNConfig
from fast_board import FastBoard
from RL import load_checkpoint
from value_net import OneHotValueNet
# Importing the fixture reuses the stage-1 generator, including real game boards.
from test_afterstate_vectorized import comparison_boards, EDGE_BOARDS  # noqa: F401


class WeightedValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.arange(1, 17, dtype=torch.float32) / 65536)
        self.calls = 0

    def forward(self, states):
        self.calls += 1
        return (states.reshape(-1, 16) * self.weights).sum(dim=1)


@pytest.fixture(scope="module")
def boards(comparison_boards):
    generated, game_count = comparison_boards
    edges = np.asarray([case.values[0] for case in EDGE_BOARDS], dtype=np.int64).reshape(-1, 16)
    assert game_count > 100
    return np.concatenate([generated, edges])


def max_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    return float(np.max(np.abs(actual - expected), initial=0))


@pytest.mark.parametrize("reward_mode", ["log", "linear"])
def test_afterstate_scores_equivalence(boards, reward_mode, record_property):
    assert len(boards) >= 5000
    torch.manual_seed(2102)
    model = OneHotValueNet(channels=16, hidden=32).eval()
    reward_error = score_error = 0.0
    # Keep inference memory bounded while evaluating every generated board.
    for chunk in np.array_split(boards, 32):
        ref = policy.afterstate_scores(model, chunk, reward_mode=reward_mode, expand="reference")
        vec = policy.afterstate_scores(model, chunk, reward_mode=reward_mode, expand="vectorized")
        np.testing.assert_array_equal(vec.after, ref.after)
        np.testing.assert_array_equal(vec.legal, ref.legal)
        np.testing.assert_array_equal(torch.isneginf(vec.scores), torch.isneginf(ref.scores))
        assert torch.equal(torch.isfinite(vec.scores), torch.as_tensor(vec.legal))
        reward_error = max(reward_error, max_error(vec.rewards, ref.rewards))
        score_error = max(score_error, max_error(vec.scores[vec.legal], ref.scores[ref.legal]))
    assert reward_error <= 1e-6 and score_error <= 1e-6
    print(f"scores {reward_mode}: boards={len(boards)}, reward_max_error={reward_error:.9g}, "
          f"score_max_error={score_error:.9g}")
    record_property("board_count", len(boards))
    record_property("reward_max_error", reward_error)
    record_property("score_max_error", score_error)


@pytest.mark.parametrize("reward_mode", ["log", "linear"])
def test_double_td_target_equivalence(boards, reward_mode, record_property):
    # All density groups and real trajectories, followed by explicit edge cases.
    next_boards = np.concatenate([boards[np.linspace(0, len(boards) - 9, 2200, dtype=int)], boards[-8:]])
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=2102,
                                          reward_mode=reward_mode))
    # Make online and target nets different, holding BOTH fixed across backends.
    with torch.no_grad():
        agent.target_model.head[-1].bias.add_(0.25)
    dones = np.random.default_rng(2102).random(len(next_boards)) < 0.25
    dones[-8:] = False  # terminal checkerboard and all-zero board lack legal actions
    dones[0] = True
    states = torch.as_tensor(next_boards, dtype=torch.float32)
    dones = torch.as_tensor(dones)
    results = []
    for expand in EXPAND_MODES:
        agent.config.afterstate_expand = expand
        results.append(torch.cat([agent._targets(states[i:i + 128], dones[i:i + 128])
                                  for i in range(0, len(states), 128)]))
    error = max_error(results[1], results[0])
    assert error <= 1e-6
    no_action = ~batch_afterstates_reference(next_boards)[2].any(axis=1)
    assert no_action.any() and (no_action & ~dones.numpy()).any() and dones.any()
    assert torch.all(results[0][dones | torch.as_tensor(no_action)] == 0)
    assert torch.any(results[0][~dones & ~torch.as_tensor(no_action)] != 0)
    print(f"TD {reward_mode}: next_states={len(states)}, max_error={error:.9g}")
    record_property("next_state_count", len(states))
    record_property("td_target_max_error", error)


@pytest.mark.parametrize("epsilon", [0.0, 0.3, 1.0])
def test_select_action_rng_and_return_after(boards, epsilon):
    playable = boards[batch_afterstates_reference(boards)[2].any(axis=1)]
    selected = playable[np.linspace(0, len(playable) - 1, 1100, dtype=int)]
    model = WeightedValue()
    reference_rng = np.random.default_rng(2102)
    vectorized_rng = np.random.default_rng(2102)
    oracle_rng = np.random.default_rng(2102)
    for grid in selected:
        expected_after, _, legal = afterstates(grid)
        invalid = np.flatnonzero(~legal)
        valid = np.flatnonzero(legal).tolist()
        if oracle_rng.random() < epsilon:
            expected_action = int(oracle_rng.choice(valid))
        else:
            expected_action = int(policy.greedy_actions(model, grid)[0])
        ref_action, ref_after = policy.select_action(
            model, grid, invalid, epsilon, reference_rng, return_after=True)
        vec_action, vec_after = policy.select_action(
            model, grid, invalid, epsilon, vectorized_rng, expand="vectorized", return_after=True)
        assert ref_action == vec_action == expected_action
        assert reference_rng.bit_generator.state == vectorized_rng.bit_generator.state == oracle_rng.bit_generator.state
        for returned in (ref_after, vec_after):
            np.testing.assert_array_equal(returned, expected_after[expected_action])
            assert returned.dtype == np.int64 and returned.shape == (16,)
            assert not np.shares_memory(returned, grid)
    print(f"select_action: epsilon={epsilon}, boards={len(selected)}, actions/RNG/afterstate exact")


@pytest.mark.parametrize("expand", EXPAND_MODES)
@pytest.mark.parametrize("invalid", [[0], [0, 1, 3], [-1], [4]])
def test_invalid_moves_mismatch_preserves_rng(expand, invalid):
    grid = np.zeros(16, dtype=np.int64)
    grid[0] = 2  # invalid = [0, 3]
    rng = np.random.default_rng(2102)
    previous = rng.bit_generator.state
    with pytest.raises(policy.LegalityMismatchError):
        policy.select_action(WeightedValue(), grid, invalid, 0.3, rng, expand=expand)
    assert rng.bit_generator.state == previous


@pytest.mark.parametrize("expand", EXPAND_MODES)
def test_terminal_selection_preserves_rng(expand):
    rng = np.random.default_rng(2102)
    previous = rng.bit_generator.state
    with pytest.raises(ValueError, match="no valid actions"):
        policy.select_action(WeightedValue(), np.zeros(16), range(4), 0.3, rng, expand=expand)
    assert rng.bit_generator.state == previous


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
@pytest.mark.parametrize("return_after", [False, True])
def test_vectorized_expands_once_and_does_not_alias(monkeypatch, epsilon, return_after):
    grid = np.asarray([2, 2, 0, 0] + [0] * 12, dtype=np.int64)
    expected, _, legal = afterstates(grid)
    expand = policy.batch_afterstates
    buffers = []

    def spy(grids, **options):
        assert options == {"expand": "vectorized"}
        result = expand(grids, **options)
        buffers.append(result[0])
        return result

    monkeypatch.setattr(policy, "batch_afterstates", spy)
    monkeypatch.setattr(policy, "afterstates", lambda *_: pytest.fail("extra scalar expansion"))
    model = WeightedValue()
    result = policy.select_action(model, grid, np.flatnonzero(~legal), epsilon,
                                  np.random.default_rng(2102), expand="vectorized", return_after=return_after)
    assert len(buffers) == 1
    assert model.calls == (0 if epsilon == 1.0 else 1)
    if return_after:
        action, after = result
        np.testing.assert_array_equal(after, expected[action])
        assert not np.shares_memory(after, buffers[0])
        buffers[0].fill(-1)
        np.testing.assert_array_equal(after, expected[action])
    else:
        assert isinstance(result, int) and legal[result]


def test_agent_last_after_state_and_greedy_rng():
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=2102,
                                          afterstate_expand="vectorized"))
    grid = np.asarray([2, 2, 0, 0] + [0] * 12, dtype=np.int64)
    expected, _, legal = afterstates(grid)
    invalid = np.flatnonzero(~legal)
    action = agent.act(grid, invalid)
    saved = agent.last_after_state
    np.testing.assert_array_equal(saved, expected[action])
    assert not np.shares_memory(saved, grid)
    before = agent.rng.bit_generator.state
    vec_greedy = agent.greedy_action(grid, invalid)
    agent.config.afterstate_expand = "reference"
    assert agent.greedy_action(grid, invalid) == vec_greedy
    assert agent.rng.bit_generator.state == before
    agent.config.afterstate_expand = "vectorized"
    agent.act(grid, invalid)
    assert not np.shares_memory(saved, agent.last_after_state)
    agent.config.afterstate_expand = "reference"
    agent.act(grid, invalid)
    assert agent.last_after_state is None
    np.testing.assert_array_equal(saved, expected[action])


@pytest.mark.parametrize("expand", EXPAND_MODES)
def test_checkpoint_roundtrip_and_old_default(tmp_path, expand):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=2102,
                                          afterstate_expand=expand))
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["config"]["afterstate_expand"] == expand
    loaded = load_checkpoint(checkpoint)
    assert loaded.config.afterstate_expand == expand
    for name, weight in agent.model.state_dict().items():
        assert torch.equal(weight, loaded.model.state_dict()[name])
    receiver = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=2102))
    receiver.load(checkpoint)
    assert receiver.config.afterstate_expand == expand
    del payload["config"]["afterstate_expand"]
    torch.save(payload, checkpoint)
    assert load_checkpoint(checkpoint).config.afterstate_expand == "reference"
    receiver.config.afterstate_expand = "vectorized"
    receiver.load(checkpoint)
    assert receiver.config.afterstate_expand == "reference"


def test_main_short_episodes_csv_and_afterstate_reuse(tmp_path, monkeypatch):
    import config
    import main
    import afterstate

    original_move = FastBoard.move
    original_remember = AfterstateAgent.remember
    original_act = AfterstateAgent.act
    original_config = config.DQNConfig
    # Three short episodes, real optimizer and TD calculation, fixed logical clock.
    monkeypatch.setattr(config, "DQNConfig", lambda **kw: original_config(batch_size=2, **kw))
    progress = SimpleNamespace(step=0, expected=None, expand=None)

    def clock():
        return float(progress.step)

    def move(self, direction):
        state, reward, done = original_move(self, direction)
        progress.step += 1
        return state, reward, done or progress.step % 6 == 0

    def act(self, state, invalid):
        action = original_act(self, state, invalid)
        progress.expected = afterstates(state)[0][action].copy()
        return action

    def remember(self, after, next_state, done):
        np.testing.assert_array_equal(after, progress.expected)
        # The actual move matches the afterstate, with exactly one spawned tile.
        spawn_delta = next_state.reshape(16).numpy() - after
        assert np.count_nonzero(spawn_delta) == 1
        assert spawn_delta[spawn_delta != 0].item() in (2, 4)
        if progress.expand == "vectorized":
            assert after is self.last_after_state  # driver reused the acting result
        original_remember(self, after, next_state, done)

    monkeypatch.setattr(main, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(FastBoard, "move", move)
    monkeypatch.setattr(AfterstateAgent, "act", act)
    monkeypatch.setattr(AfterstateAgent, "remember", remember)
    rows_by_mode = []
    for expand in EXPAND_MODES:
        progress.step = 0
        progress.expand = expand
        if expand == "vectorized":
            # main imports this scalar path; vectorized acting uses batch instead.
            monkeypatch.setattr(afterstate, "afterstates", lambda *_: pytest.fail("main re-expanded board"))
        log = tmp_path / f"{expand}.csv"
        checkpoint = tmp_path / f"{expand}.pt"
        assert main.main(["train", "--arch", "onehot_afterstate", "--afterstate-expand", expand,
                          "--from-scratch", "--seed", "2102", "--threads", "1", "--episodes", "3",
                          "--minutes", "1", "--replay-every", "2", "--save-as", str(checkpoint),
                          "--log", str(log)]) == 0
        with log.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3 and all(int(row["steps"]) == 6 for row in rows)
        rows_by_mode.append(rows)
        summary = json.loads(log.with_suffix(".json").read_text())
        assert summary["afterstate_expand"] == expand and summary["updates"] == 9
        assert load_checkpoint(checkpoint).config.afterstate_expand == expand
    # Includes elapsed_s: the injected clock depends only on environment steps.
    assert rows_by_mode[0] == rows_by_mode[1]


@pytest.mark.parametrize("arch", ["legacy", "onehot_dueling", "onehot_conv2x2"])
def test_cli_rejects_vectorized_for_other_architectures(arch):
    import main
    with pytest.raises(ValueError, match="requires --arch onehot_afterstate"):
        main.main(["train", "--arch", arch, "--afterstate-expand", "vectorized", "--minutes", "0"])


def test_expand_defaults_and_invalid_mode():
    import main
    assert DQNConfig().afterstate_expand == main.parser().parse_args(["train"]).afterstate_expand == "reference"
    with pytest.raises(ValueError, match="expand mode"):
        policy.select_action(WeightedValue(), np.zeros(16), range(4), 0, np.random.default_rng(0), expand="bad")
