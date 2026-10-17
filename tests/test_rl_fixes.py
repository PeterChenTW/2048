"""Deterministic regression checks for masked Double DQN and prioritized replay."""
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DQNConfig
from RL import DQN, DDQNAgent, SumTree


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


def transition(agent, done=False, invalid=(), reward=3):
    agent.remember(torch.zeros(1, 16), 0, reward, torch.ones(1, 16), done, invalid)


def test_negative_q_and_strict_checkpoint():
    model = DQN()
    checkpoint = Path(__file__).resolve().parents[1] / 'model/nn_prioritize_replay_no_invalid_gamma_099_2'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    with torch.no_grad():
        model.fc3.weight.zero_()
        model.fc3.bias.fill_(-5)
    assert torch.equal(model(torch.zeros(2, 16)), torch.full((2, 4), -5.0))


@pytest.mark.parametrize('done,invalid,expected', [
    (False, (), 3 + .99 * 2),
    (True, (), 3),
    (False, (0,), 3 + .99 * 7),
    (False, (0, 1, 2, 3), 3),
])
def test_double_dqn_mask_and_terminal(done, invalid, expected):
    agent = agent_with_tables()
    transition(agent, done, invalid)
    mask = agent.memory.data[0][-1]
    assert mask.dtype == bool
    assert mask.tolist() == [a not in invalid for a in range(4)]
    stats = agent.replay()
    assert stats['target_mean'] == pytest.approx(expected)
    assert stats['loss'] == pytest.approx((expected - 1) ** 2)
    assert agent.model.calls == 2  # one current batch and one next-state batch
    assert agent.target_model.calls == 1
    assert agent.target_model.rows.grad is None
    assert agent.memory.tree[7] == pytest.approx((abs(expected - 1) + agent.config.per_eps) ** agent.config.per_alpha)


@pytest.mark.parametrize('epsilon', [0, 1])
def test_act_masks_invalid(epsilon):
    agent = agent_with_tables()
    agent.epsilon = epsilon
    for _ in range(100):
        assert agent.act(torch.zeros(1, 16), [1, 2, 3]) == 0
    with pytest.raises(ValueError, match='no valid'):
        agent.act(torch.zeros(1, 16), [0, 1, 2, 3])


def test_act_greedy_maps_argmax_back_to_global_action():
    agent = agent_with_tables()
    agent.epsilon = 0
    # state 0 -> Q=[1, 2, 3, 4]; state 1 -> Q=[10, 8, 6, 4]
    assert agent.act(torch.zeros(1, 16), []) == 3
    assert agent.act(torch.zeros(1, 16), [3]) == 2
    assert agent.act(torch.zeros(1, 16), [0, 3]) == 2
    assert agent.act(torch.zeros(1, 16), [2, 3]) == 1
    assert agent.act(torch.ones(1, 16), [0]) == 1
    assert agent.act(torch.ones(1, 16), [0, 1]) == 2
    assert not agent.q_values(torch.zeros(1, 16)).requires_grad


def test_is_weights_and_weighted_loss(monkeypatch):
    agent = agent_with_tables()
    transition(agent, done=True, reward=3)
    transition(agent, done=True, reward=5)
    agent.memory.update(7, 1)
    agent.memory.update(8, 3)
    # Force a batch containing each leaf once, independently of stratification.
    samples = iter([agent.memory.get(0), agent.memory.get(1)])
    monkeypatch.setattr(agent.memory, 'get', lambda value: next(samples))
    weights = (2 * np.array([.25, .75])) ** (-.4)
    weights /= weights.max()
    assert weights.max() == 1
    stats = agent.replay(2)
    assert stats['is_weight_mean'] == pytest.approx(weights.mean())
    assert stats['is_weight_min'] == pytest.approx(weights.min())
    assert stats['loss'] == pytest.approx(np.mean(weights * np.array([4, 16])))


def test_beta_anneals_and_epsilon_decays():
    agent = agent_with_tables(per_beta_anneal_steps=2, epsilon_start=.2,
                              epsilon_decay=.5, epsilon_min=.04)
    transition(agent, done=True)
    stats = [agent.replay() for _ in range(4)]
    assert [s['beta'] for s in stats] == pytest.approx([.4, .7, 1, 1])
    assert [s['epsilon'] for s in stats] == pytest.approx([.1, .05, .04, .04])


@pytest.mark.parametrize('capacity', [1, 3, 5, 8])
def test_sumtree_partial_and_boundary_sampling(capacity):
    tree = SumTree(capacity)
    with pytest.raises(ValueError):
        tree.get(0)
    for count in range(1, capacity + 1):
        tree.add(count, f'entry {count}')
        assert tree.size == count
        for value in [0, tree.total(), tree.total() + 10, *np.linspace(0, tree.total(), 100)]:
            idx, priority, data = tree.get(value)
            assert idx - capacity + 1 < count
            assert priority > 0
            assert data.startswith('entry ')
    tree.add(1, 'entry replacement')
    assert tree.size == capacity


@pytest.mark.parametrize('discount', [.2, .8])
def test_config_is_gamma_source(discount):
    agent = agent_with_tables(gamma=discount)
    transition(agent)
    assert agent.replay()['target_mean'] == pytest.approx(3 + discount * 2)
    source = (Path(__file__).resolve().parents[1] / 'RL.py').read_text()
    assert not re.search(r'\bgamma\s*=\s*[0-9.]', source)
    assert 'self.config.gamma' in source


def test_real_network_small_buffer_finite_stats(tmp_path):
    agent = DDQNAgent(16, 4, DQNConfig(memory_capacity=5, batch_size=3, seed=1))
    for i in range(3):
        state = torch.full((1, 16), float(i))
        agent.remember(state, i, float(i), state + 1, i == 2, (3,))
    stats = agent.replay()
    assert set(stats) == {'loss', 'q_mean', 'q_min', 'q_max', 'target_mean',
                          'td_abs_mean', 'is_weight_mean', 'is_weight_min', 'beta', 'epsilon', 'nan'}
    assert not stats['nan']
    assert all(np.isfinite(value) for value in stats.values())
    assert stats['q_min'] <= stats['q_mean'] <= stats['q_max']
    path = tmp_path / 'weights.pt'
    agent.save(path)
    agent.load(path)
    for key, value in agent.model.state_dict().items():
        assert torch.equal(value, agent.target_model.state_dict()[key])


def test_nonfinite_stats_raise():
    agent = agent_with_tables()
    transition(agent, reward=float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        agent.replay()
