"""Integration checks for architecture selection, n-step discounts, and checkpoints."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DQNConfig
from dueling import OneHotDuelingDQN
from eval import evaluate
from RL import DQN, DDQNAgent, load_checkpoint
from nstep import NStepAccumulator


REPO = Path(__file__).resolve().parents[1]
LEGACY_CHECKPOINT = REPO / 'model/nn_prioritize_replay_no_invalid_gamma_099_2'


class Table(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = nn.Parameter(torch.tensor(rows, dtype=torch.float32))
        self.calls = 0

    def forward(self, states):
        self.calls += 1
        return self.rows[states.reshape(-1, 16)[:, 0].long()]


def agent_with_tables(**kwargs):
    config = DQNConfig(memory_capacity=8, batch_size=1, lr=0, seed=7, **kwargs)
    agent = DDQNAgent(16, 4, config)
    agent.model = Table([[1, 2, 3, 4], [10, 8, 6, 4]])
    agent.target_model = Table([[0, 0, 0, 0], [2, 7, 30, 90]])
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=config.lr)
    return agent


def _equal_state_dict(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_config_defaults_select_legacy_or_onehot():
    config = DQNConfig()
    assert config.arch == 'legacy'
    assert config.n_step == 1
    legacy = DDQNAgent(16, 4, config)
    assert isinstance(legacy.model, DQN)
    onehot = DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling', n_step=3))
    assert isinstance(onehot.model, OneHotDuelingDQN)
    assert onehot.model.architecture == 'onehot_dueling_v1'
    assert onehot.config.n_step == 3
    with pytest.raises(ValueError):
        DDQNAgent(16, 4, DQNConfig(arch='other'))


def test_checkpoint_factory_raw_legacy_vs_v2_roundtrip(tmp_path):
    legacy = DDQNAgent(16, 4, DQNConfig(arch='legacy', seed=1))
    legacy_path = tmp_path / 'legacy.pt'
    legacy.save(legacy_path)
    raw = torch.load(legacy_path, map_location='cpu', weights_only=True)
    assert 'format' not in raw
    assert 'conv1.weight' in raw
    restored = load_checkpoint(legacy_path)
    assert restored.config.arch == 'legacy'
    assert isinstance(restored.model, DQN)
    _equal_state_dict(raw, restored.model.state_dict())
    _equal_state_dict(restored.model.state_dict(), restored.target_model.state_dict())
    again = DDQNAgent(16, 4, DQNConfig(arch='legacy', seed=99))
    again.load(legacy_path)
    _equal_state_dict(legacy.model.state_dict(), again.model.state_dict())

    saved = DQNConfig(arch='onehot_dueling', n_step=3, gamma=0.5, lr=2e-4, seed=11,
                      memory_capacity=64, batch_size=8)
    onehot = DDQNAgent(16, 4, saved)
    onehot_path = tmp_path / 'onehot.pt'
    onehot.save(onehot_path)
    blob = torch.load(onehot_path, map_location='cpu', weights_only=True)
    assert blob['format'] == 'dqn2048-v2'
    assert blob['architecture'] == 'onehot_dueling_v1'
    assert blob['config']['arch'] == 'onehot_dueling'
    assert blob['config']['n_step'] == 3
    assert set(blob['state_dict']) == set(onehot.model.state_dict())
    loaded = load_checkpoint(onehot_path)
    assert isinstance(loaded.model, OneHotDuelingDQN)
    assert loaded.config.arch == 'onehot_dueling'
    assert loaded.config.n_step == 3
    assert loaded.config.gamma == pytest.approx(0.5)
    assert loaded.config.seed == 11
    assert loaded.config.memory_capacity == 64
    assert loaded.memory.capacity == 64
    _equal_state_dict(onehot.model.state_dict(), loaded.model.state_dict())
    _equal_state_dict(blob['state_dict'], loaded.model.state_dict())
    _equal_state_dict(loaded.model.state_dict(), loaded.target_model.state_dict())
    other = DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling', seed=2))
    other.load(onehot_path)
    _equal_state_dict(onehot.model.state_dict(), other.model.state_dict())

    with pytest.raises(ValueError, match='onehot_dueling'):
        DDQNAgent(16, 4, DQNConfig(arch='legacy')).load(onehot_path)


def test_legacy_checkpoint_into_onehot_raises():
    agent = DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling', seed=0))
    with pytest.raises(ValueError, match='onehot_dueling'):
        agent.load(LEGACY_CHECKPOINT)


def test_real_legacy_checkpoint_strict_load_and_short_eval():
    assert LEGACY_CHECKPOINT.is_file()
    raw = torch.load(LEGACY_CHECKPOINT, map_location='cpu', weights_only=True)
    agent = load_checkpoint(LEGACY_CHECKPOINT)
    assert agent.config.arch == 'legacy'
    assert isinstance(agent.model, DQN)
    incompatible = agent.model.load_state_dict(raw, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    fresh = DDQNAgent(16, 4, DQNConfig(arch='legacy', seed=0))
    fresh.load(LEGACY_CHECKPOINT)
    _equal_state_dict(raw, fresh.model.state_dict())
    _equal_state_dict(fresh.model.state_dict(), fresh.target_model.state_dict())
    result = evaluate(checkpoint=LEGACY_CHECKPOINT, policy='greedy', games=1,
                      board='fast', max_steps=200)
    assert result['games'] == 1
    assert result['policy'] == 'greedy'
    assert result['board'] == 'fast'
    record = result['per_game'][0]
    assert record['seed'] == 1000
    assert 0 < record['steps'] <= 200
    assert record['score'] >= 0
    assert math.isfinite(result['mean'])
    assert torch.isfinite(agent.model(torch.zeros(1, 16))).all()


@pytest.mark.parametrize('done,invalid,discount,expected', [
    # Next row online [10, 8, 6, 4], target [2, 7, 30, 90]. Action 0 is taken
    # from the zero state, whose online Q is 1. Passed discount is gamma**k
    # and must replace a fresh multiply by config.gamma (here 0.5).
    (False, (), 0.5 ** 3, 3 + (0.5 ** 3) * 2),
    (False, (0,), 0.5 ** 2, 3 + (0.5 ** 2) * 7),
    (True, (), 0.5 ** 1, 3),
    (False, (0, 1, 2, 3), 0.5 ** 3, 3),
])
def test_masked_replay_gamma_power_k(done, invalid, discount, expected):
    agent = agent_with_tables(gamma=0.5, n_step=3)
    agent.remember(torch.zeros(1, 16), 0, 3, torch.ones(1, 16), done,
                   next_invalid_moves=invalid, discount=discount)
    mask = agent.memory.data[0][-1]
    assert mask.dtype == bool
    assert mask.tolist() == [a not in invalid for a in range(4)]
    stats = agent.replay()
    assert stats['target_mean'] == pytest.approx(expected)
    assert stats['loss'] == pytest.approx((expected - 1) ** 2)
    if not done and invalid != (0, 1, 2, 3):
        assert stats['target_mean'] != pytest.approx(3 + agent.config.gamma * 2)
    assert agent.model.calls == 2
    assert agent.target_model.calls == 1
    assert agent.target_model.rows.grad is None


def test_remember_discount_defaults_to_gamma():
    agent = agent_with_tables(gamma=0.5)
    agent.remember(torch.zeros(1, 16), 0, 3, torch.ones(1, 16), False)
    assert agent.replay()['target_mean'] == pytest.approx(3 + 0.5 * 2)


def test_cli_train_onehot_nstep3_then_eval_two_games(tmp_path):
    checkpoint = tmp_path / 'checkpoints' / 'onehot.pt'
    log = tmp_path / 'logs' / 'train.csv'
    trained = subprocess.run(
        [sys.executable, 'main.py', 'train', '--from-scratch',
         '--arch', 'onehot_dueling', '--n-step', '3', '--episodes', '3',
         '--seed', '0', '--save-as', str(checkpoint), '--log', str(log)],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    assert 'Starting from scratch' in trained.stdout
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(int(row['nan_count']) == 0 for row in rows)
    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['arch'] == 'onehot_dueling'
    assert summary['n_step'] == 3
    assert summary['episodes'] == 3
    assert summary['nan_count'] == 0
    assert summary['loaded_checkpoint'] is None
    assert summary['updates'] > 0
    assert summary['checkpoint'] == str(checkpoint)
    assert summary['env_steps'] == sum(int(row['steps']) for row in rows)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert blob['format'] == 'dqn2048-v2'
    assert blob['architecture'] == 'onehot_dueling_v1'
    assert blob['config']['arch'] == 'onehot_dueling'
    assert blob['config']['n_step'] == 3
    assert blob['state_dict']

    eval_json = tmp_path / 'eval.json'
    evaluated = subprocess.run(
        [sys.executable, 'eval.py', '--checkpoint', str(checkpoint),
         '--games', '2', '--seed-start', '1000', '--board', 'fast',
         '--json', str(eval_json)],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    payload = json.loads(eval_json.read_text())
    printed = json.loads(evaluated.stdout)
    assert payload['games'] == 2
    assert printed['games'] == 2
    assert payload['policy'] == 'greedy'
    assert len(payload['per_game']) == 2
    for i, record in enumerate(payload['per_game']):
        assert record['seed'] == 1000 + i
        assert record['steps'] > 0
        assert record['score'] >= 0
        assert math.isfinite(record['score'])


@pytest.mark.parametrize('terminal', [False, True])
def test_accumulator_returns_flow_into_replay_with_shortened_horizons(terminal):
    accumulator = NStepAccumulator(3, 0.5)
    transitions = []
    for index, reward in enumerate([1, 2, 4]):
        transitions.extend(accumulator.push(torch.zeros(1, 16), 0, reward,
                                            torch.ones(1, 16), terminal and index == 2,
                                            [False, True, True, True]))
    transitions.extend(accumulator.flush())
    assert len(transitions) == 3
    for transition, reward, discount in zip(transitions, [3, 4, 4], [.125, .25, .5]):
        agent = agent_with_tables(gamma=.5, n_step=3)
        assert transition.reward == reward
        assert transition.discount == discount
        agent.remember(transition.state, transition.action, transition.reward,
                       transition.next_state, transition.done, (0,),
                       discount=transition.discount)
        assert agent.memory.data[0][-2] == discount
        # Online picks valid action 1; target evaluates it at 7.
        expected = reward if terminal else reward + discount * 7
        assert agent.replay()['target_mean'] == pytest.approx(expected)
