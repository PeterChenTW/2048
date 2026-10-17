"""Double afterstate TD semantics, PER, checkpoints, and opt-in training."""
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afterstate import afterstates
from afterstate_agent import AfterstateAgent
from config import DQNConfig
from eval import evaluate
from fast_board import FastBoard
from RL import DDQNAgent, load_checkpoint
from value_net import OneHotValueNet


REPO = Path(__file__).resolve().parents[1]
GRID = np.array([[2, 2, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0]])
TERMINAL = np.array([[2, 4, 2, 4], [4, 2, 4, 2],
                     [2, 4, 2, 4], [4, 2, 4, 2]])


class Lookup(nn.Module):
    def __init__(self, boards, values, default=0.0):
        super().__init__()
        self.table = {tuple(board): value for board, value in zip(boards, values)}
        self.offset = nn.Parameter(torch.tensor(float(default)))
        self.calls = []

    def forward(self, states):
        self.calls.append((len(states), torch.is_grad_enabled()))
        values = [self.table.get(tuple(row), 0.0) for row in states.detach().tolist()]
        return torch.tensor(values) + self.offset


def table_agent(**options):
    config = DQNConfig(arch='onehot_afterstate', memory_capacity=8, batch_size=1,
                       gamma=0.5, lr=0, seed=7, **options)
    agent = AfterstateAgent(16, 4, config)
    boards, _, _ = afterstates(GRID)
    # Online V alone prefers down (1), but r + gamma*V prefers left (0).
    # Target r + gamma*V would prefer right (2). Double selection must use online.
    agent.model = Lookup(boards, [1.0, 1.1, 0.0, 1000.0])
    agent.target_model = Lookup(boards, [3.0, 7.0, 100.0, 1000.0])
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0)
    return agent


def test_double_target_online_selects_target_evaluates_and_batches():
    agent = table_agent()
    next_states = torch.as_tensor(np.stack([GRID, GRID, TERMINAL]), dtype=torch.float32).reshape(-1, 16)
    values = agent._targets(next_states, torch.tensor([False, False, False]))
    reward = math.log2(5) / 16  # next-state left action merges 2+2
    assert values.tolist() == pytest.approx([reward + 0.5 * 3, reward + 0.5 * 3, 0])
    assert agent.model.calls == [(6, False)]  # three legal actions per live board
    assert agent.target_model.calls == [(2, False)]
    assert agent.target_model.offset.grad is None
    # The next merge reward changes the selected action as well as the target.
    assert values[0] != pytest.approx(0.5 * 3)
    assert values[0] != pytest.approx(0.5 * 7)
    assert values[0] != pytest.approx(reward + 0.5 * 100)


def test_terminal_targets_zero_without_network_evaluation():
    agent = table_agent()
    states = torch.as_tensor(np.stack([GRID, TERMINAL]), dtype=torch.float32).reshape(-1, 16)
    assert agent._targets(states, torch.tensor([True, True])).tolist() == [0, 0]
    assert agent.model.calls == agent.target_model.calls == []
    # Also recognize terminal boards even if the environment's done flag is false.
    assert agent._targets(states[1:], torch.tensor([False])).item() == 0
    assert agent.model.calls == agent.target_model.calls == []


def test_reward_options_apply_to_target_and_policy():
    agent = table_agent(epsilon_start=0)
    agent.config.reward_mode = 'linear'
    agent.config.reward_scale = 4.0
    target = agent._targets(torch.tensor(GRID).reshape(1, 16), torch.tensor([False]))
    assert target.item() == pytest.approx(1.0 + 0.5 * 3)
    assert agent.act(GRID, [3]) == agent.greedy_action(GRID, [3]) == 0


def test_replay_per_priorities_weights_schedules_and_stats():
    agent = table_agent(per_beta_anneal_steps=1)
    source = np.zeros(16, dtype=np.int64)
    agent.remember(source, GRID, False)
    assert len(agent.memory.data[0]) == 3  # no taken reward or action is stored
    source[0] = 999
    assert agent.memory.data[0][0][0, 0] == 0
    stats = agent.replay()
    expected = math.log2(5) / 16 + 0.5 * 3
    assert stats['target_mean'] == pytest.approx(expected)
    assert stats['loss'] == pytest.approx(expected ** 2)
    assert agent.memory.tree[7] == pytest.approx((expected + agent.config.per_eps) ** agent.config.per_alpha)
    assert agent.replay_calls == 1
    assert agent.beta == agent.config.per_beta_start
    assert agent.epsilon == pytest.approx(agent.config.epsilon_start * agent.config.epsilon_decay)
    assert agent.target_model.offset.grad is None
    assert all(np.isfinite(value) for value in stats.values())
    assert not stats['nan']
    assert set(stats) == {'loss', 'q_mean', 'q_min', 'q_max', 'target_mean', 'td_abs_mean',
                          'is_weight_mean', 'is_weight_min', 'beta', 'epsilon', 'nan'}
    agent.remember(np.zeros(16), TERMINAL, True)
    assert agent.memory.tree[8] == agent.memory.tree[7]  # max insertion priority
    agent.memory.update(8, agent.memory.tree[7] * 2)
    result = agent.replay(batch_size=8)
    assert agent.beta == agent.config.per_beta_end
    assert 0 < result['is_weight_min'] < result['is_weight_mean'] < 1
    assert agent.memory.tree[8] == pytest.approx(agent.config.per_eps ** agent.config.per_alpha)


def test_real_network_replay_and_nonfinite_guard():
    agent = AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', batch_size=2, seed=3))
    before = agent.model.head[-1].weight.detach().clone()
    agent.remember(afterstates(GRID)[0][0], GRID, False)
    agent.remember(afterstates(GRID)[0][2], TERMINAL, True)
    stats = agent.replay()
    assert not stats['nan']
    assert not torch.equal(before, agent.model.head[-1].weight)
    with torch.no_grad():
        agent.model.head[-1].bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        agent.replay()


def test_linear_epsilon_inheritance_and_replay_validation():
    agent = table_agent(epsilon_schedule='linear_steps', epsilon_decay_steps=10)
    agent.set_env_step(5)
    expected = (agent.config.epsilon_start + agent.config.epsilon_min) / 2
    assert agent.epsilon == pytest.approx(expected)
    with pytest.raises(ValueError):
        agent.replay()
    agent.remember(np.zeros(16), GRID, False)
    with pytest.raises(ValueError):
        agent.replay(batch_size=0)
    agent.replay()
    assert agent.epsilon == pytest.approx(expected)


def test_checkpoint_roundtrip_and_cross_arch_rejection(tmp_path):
    agent = AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', seed=9, gamma=0.8))
    assert isinstance(agent.model, OneHotValueNet)
    checkpoint = tmp_path / 'afterstate.pt'
    agent.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload['format'] == 'dqn2048-v2'
    assert payload['architecture'] == 'onehot_afterstate_v1'
    loaded = load_checkpoint(checkpoint)
    assert isinstance(loaded, AfterstateAgent)
    assert loaded.config.gamma == 0.8
    for key, value in agent.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
        assert torch.equal(value, loaded.target_model.state_dict()[key])
    rng = random.Random(13)
    for _ in range(8):
        board = FastBoard(rng=rng)
        invalid = board.get_invalid_moves()
        assert agent.greedy_action(board.grid, invalid) == loaded.greedy_action(board.grid, invalid)
    other = DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling'))
    with pytest.raises(ValueError, match='architecture'):
        other.load(checkpoint)
    other.save(tmp_path / 'dueling.pt')
    with pytest.raises(ValueError, match='architecture'):
        agent.load(tmp_path / 'dueling.pt')
    agent.load(checkpoint)


def test_invalid_agent_arch_and_nstep():
    with pytest.raises(ValueError, match='AfterstateAgent'):
        DDQNAgent(16, 4, DQNConfig(arch='onehot_afterstate'))
    with pytest.raises(ValueError, match='n_step=1'):
        AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', n_step=3))
    with pytest.raises(ValueError, match='requires arch'):
        AfterstateAgent(16, 4, DQNConfig())
    assert isinstance(AfterstateAgent(16, 4).model, OneHotValueNet)


def test_train_taken_reward_is_ignored_and_stores_before_spawn(tmp_path, monkeypatch):
    import config
    import main
    import nstep

    original_config = config.DQNConfig
    monkeypatch.setattr(config, 'DQNConfig', lambda **kwargs: original_config(batch_size=1, lr=0, **kwargs))
    monkeypatch.setattr(nstep, 'NStepAccumulator', lambda *args: pytest.fail('afterstate must bypass n-step'))
    original_move = FastBoard.move
    original_remember = AfterstateAgent.remember
    original_replay = AfterstateAgent.replay
    original_act = AfterstateAgent.act
    captures, targets, expected = [], [], []
    taken_reward = 0

    def act(self, state, invalid):
        action = original_act(self, state, invalid)
        expected.append(afterstates(state)[0][action].copy())
        return action

    def move(self, direction):
        next_grid, _, _ = original_move(self, direction)
        return next_grid, taken_reward, True

    def remember(self, after, next_state, done):
        original_remember(self, after, next_state, done)
        captures.append(self.memory.data[0])
        # Verify target semantics on the post-spawn board independently of env done.
        targets.append(self._targets(next_state.reshape(1, 16), torch.tensor([False])).item())

    def replay(self):
        stats = original_replay(self)
        assert stats['target_mean'] == 0  # env done overrides a still-playable board
        return stats

    monkeypatch.setattr(AfterstateAgent, 'act', act)
    monkeypatch.setattr(FastBoard, 'move', move)
    monkeypatch.setattr(AfterstateAgent, 'remember', remember)
    monkeypatch.setattr(AfterstateAgent, 'replay', replay)
    for i, taken_reward in enumerate([0, 1e9, float('nan')]):
        assert main.main(['train', '--arch', 'onehot_afterstate', '--from-scratch',
                          '--threads', '1', '--episodes', '1', '--replay-every', '1', '--seed', '11',
                          '--save-as', str(tmp_path / f'model{i}'), '--log', str(tmp_path / f'log{i}.csv')]) == 0
    assert targets == pytest.approx([targets[0]] * 3)
    for transition, after in zip(captures, expected):
        assert np.array_equal(transition[0].numpy().reshape(16), after)
        assert transition[1].sum() > transition[0].sum()  # next_state includes spawn
        assert transition[2] is True
    for transition in captures[1:]:
        assert torch.equal(captures[0][0], transition[0])
        assert torch.equal(captures[0][1], transition[1])


def test_train_cli_three_episodes_and_eval(tmp_path):
    checkpoint, log = tmp_path / 'trained.pt', tmp_path / 'train.csv'
    prefix = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /workspace/rl2048-review/.venv/bin/python'
    import shlex
    args = ['main.py', 'train', '--arch', 'onehot_afterstate', '--from-scratch',
            '--episodes', '3', '--threads', '1', '--seed', '101', '--eps-start', '1',
            '--eps-min', '1', '--replay-ratio', '0.02', '--save-as', str(checkpoint), '--log', str(log)]
    result = subprocess.run(prefix + ' ' + shlex.join(args), shell=True, cwd=REPO,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(log.with_suffix('.json').read_text())
    rows = list(csv.DictReader(log.open()))
    assert summary['arch'] == 'onehot_afterstate'
    assert summary['n_step'] == 1
    assert summary['episodes'] == len(rows) == 3
    assert summary['env_steps'] == sum(int(row['steps']) for row in rows)
    assert summary['updates'] > 0
    assert summary['nan_count'] == 0
    result = evaluate(checkpoint=checkpoint, games=2, board='fast', max_steps=100)
    assert result['games'] == len(result['per_game']) == 2
    assert all(0 < record['steps'] <= 100 for record in result['per_game'])
    assert math.isfinite(result['mean'])
    invalid = subprocess.run(prefix + ' main.py train --arch onehot_afterstate --n-step 3 '
                             '--threads 1 --from-scratch --episodes 1', shell=True, cwd=REPO,
                             capture_output=True, text=True, timeout=30)
    assert invalid.returncode != 0
    assert 'onehot_afterstate requires n_step=1' in invalid.stderr
