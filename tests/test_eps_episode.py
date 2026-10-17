"""Tests for per-episode exponential epsilon exploration schedule."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from afterstate_agent import AfterstateAgent
from config import DQNConfig
from main import parser
from RL import DDQNAgent


def test_hand_computed_epsilon_episodes():
    # Arm A exact setup: start=0.15, decay=0.999, start_episode=128, floor=0.001
    cfg = DQNConfig(epsilon_schedule='episode_exp', epsilon_start=0.15,
                    epsilon_decay=0.999, epsilon_decay_start_episode=128,
                    epsilon_min=0.001)
    agent = DDQNAgent(16, 4, cfg)
    assert agent.epsilon == 0.15

    # Episodes 1 to 128: no decay
    for ep in range(1, 129):
        agent.end_episode(ep)
        assert agent.epsilon == 0.15

    # Episodes 129 onwards: decay once per episode
    for ep in range(129, 135):
        agent.end_episode(ep)
        k = ep - 128
        expected = 0.15 * (0.999 ** k)
        assert agent.epsilon == pytest.approx(expected, rel=1e-7)

    # Also test start_episode = 0
    cfg0 = DQNConfig(epsilon_schedule='episode_exp', epsilon_start=0.2,
                     epsilon_decay=0.9, epsilon_decay_start_episode=0,
                     epsilon_min=0.001)
    agent0 = DDQNAgent(16, 4, cfg0)
    for ep in range(1, 6):
        agent0.end_episode(ep)
        assert agent0.epsilon == pytest.approx(0.2 * (0.9 ** ep), rel=1e-7)


def test_floor_at_epsilon_min():
    cfg = DQNConfig(epsilon_schedule='episode_exp', epsilon_start=0.1,
                    epsilon_decay=0.5, epsilon_decay_start_episode=0,
                    epsilon_min=0.03)
    agent = DDQNAgent(16, 4, cfg)
    agent.end_episode(1)
    assert agent.epsilon == pytest.approx(0.05)
    agent.end_episode(2)
    assert agent.epsilon == pytest.approx(0.03)
    # Further episodes do not decay below epsilon_min
    agent.end_episode(3)
    assert agent.epsilon == pytest.approx(0.03)
    agent.end_episode(4)
    assert agent.epsilon == pytest.approx(0.03)


def test_replay_does_not_change_epsilon_in_episode_exp_ddqn():
    cfg = DQNConfig(epsilon_schedule='episode_exp', memory_capacity=4, batch_size=2,
                    epsilon_start=0.15, epsilon_decay=0.5, epsilon_min=0.01)
    agent = DDQNAgent(16, 4, cfg)
    agent.model = torch.nn.Linear(16, 4)
    agent.target_model = torch.nn.Linear(16, 4)
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.0)

    for i in range(4):
        agent.remember(torch.zeros(1, 16), 0, 1.0, torch.zeros(1, 16), False)

    for _ in range(5):
        stats = agent.replay()
        assert agent.epsilon == 0.15
        assert stats['epsilon'] == 0.15

    # end_episode should decay
    agent.end_episode(1)
    assert agent.epsilon == pytest.approx(0.15 * 0.5)


def test_replay_does_not_change_epsilon_in_episode_exp_afterstate():
    cfg = DQNConfig(arch='onehot_afterstate', epsilon_schedule='episode_exp',
                    memory_capacity=4, batch_size=2,
                    epsilon_start=0.15, epsilon_decay=0.5, epsilon_min=0.01)
    agent = AfterstateAgent(16, 4, cfg)
    agent.model = torch.nn.Linear(16, 1)
    agent.target_model = torch.nn.Linear(16, 1)
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.0)

    for i in range(4):
        agent.remember(torch.zeros(1, 16), torch.zeros(1, 16), False)

    for _ in range(5):
        stats = agent.replay()
        assert agent.epsilon == 0.15
        assert stats['epsilon'] == 0.15

    # AfterstateAgent inherits end_episode
    agent.end_episode(1)
    assert agent.epsilon == pytest.approx(0.15 * 0.5)


def test_update_exp_and_linear_steps_unchanged():
    # update_exp: end_episode is a no-op; replay decays
    cfg_update = DQNConfig(epsilon_schedule='update_exp', memory_capacity=4, batch_size=2,
                           epsilon_start=0.8, epsilon_decay=0.5, epsilon_min=0.1)
    agent_update = DDQNAgent(16, 4, cfg_update)
    agent_update.model = torch.nn.Linear(16, 4)
    agent_update.target_model = torch.nn.Linear(16, 4)
    agent_update.optimizer = torch.optim.SGD(agent_update.model.parameters(), lr=0.0)
    for _ in range(4):
        agent_update.remember(torch.zeros(1, 16), 0, 1.0, torch.zeros(1, 16), False)

    agent_update.end_episode(1)
    agent_update.end_episode(100)
    assert agent_update.epsilon == 0.8
    agent_update.replay()
    assert agent_update.epsilon == pytest.approx(0.4)

    # linear_steps: end_episode is a no-op; set_env_step decays
    cfg_linear = DQNConfig(epsilon_schedule='linear_steps', epsilon_start=0.8,
                           epsilon_min=0.1, epsilon_decay_steps=100)
    agent_linear = DDQNAgent(16, 4, cfg_linear)
    agent_linear.end_episode(1)
    assert agent_linear.epsilon == 0.8
    agent_linear.set_env_step(50)
    assert agent_linear.epsilon == pytest.approx(0.45)
    agent_linear.end_episode(2)
    assert agent_linear.epsilon == pytest.approx(0.45)


def test_cli_parsing_and_validation():
    p = parser()
    args = p.parse_args(['train', '--eps-schedule', 'episode_exp',
                         '--eps-decay', '0.99', '--eps-decay-start-episode', '10'])
    assert args.eps_schedule == 'episode_exp'
    assert args.eps_decay == 0.99
    assert args.eps_decay_start_episode == 10

    # eps-decay in (0, 1]: 1.0 is valid
    args1 = p.parse_args(['train', '--eps-decay', '1.0'])
    assert args1.eps_decay == 1.0

    # invalid choices or bounds
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--eps-schedule', 'invalid'])
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--eps-decay', '0'])
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--eps-decay', '-0.5'])
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--eps-decay', '1.1'])
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--eps-decay-start-episode', '-1'])

    # Agent validation
    with pytest.raises(ValueError, match='unknown epsilon schedule'):
        DDQNAgent(16, 4, DQNConfig(epsilon_schedule='unknown'))
    with pytest.raises(ValueError, match='epsilon_decay_start_episode'):
        DDQNAgent(16, 4, DQNConfig(epsilon_decay_start_episode=-5))
    with pytest.raises(ValueError, match='epsilon_decay_start_episode'):
        DDQNAgent(16, 4, DQNConfig(epsilon_decay_start_episode=True))
    with pytest.raises(ValueError, match='epsilon_decay'):
        DDQNAgent(16, 4, DQNConfig(epsilon_decay=0.0))
    with pytest.raises(ValueError, match='epsilon_decay'):
        DDQNAgent(16, 4, DQNConfig(epsilon_decay=1.5))
    with pytest.raises(ValueError, match='epsilon_decay'):
        DDQNAgent(16, 4, DQNConfig(epsilon_decay=False))


def test_seeded_train_episode_exp_csv_and_summary(tmp_path):
    checkpoint = tmp_path / 'trained.pt'
    log = tmp_path / 'train.csv'
    result = subprocess.run(
        [sys.executable, 'main.py', 'train', '--from-scratch', '--episodes', '5',
         '--arch', 'onehot_dueling', '--seed', '101', '--threads', '1',
         '--eps-schedule', 'episode_exp', '--eps-start', '0.15',
         '--eps-min', '0.001', '--eps-decay', '0.999',
         '--eps-decay-start-episode', '2',
         '--save-as', str(checkpoint), '--log', str(log)],
        cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr

    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['epsilon_schedule'] == 'episode_exp'
    assert summary['epsilon_decay'] == 0.999
    assert summary['epsilon_decay_start_episode'] == 2

    # Checkpoint contains config
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert blob['config']['epsilon_schedule'] == 'episode_exp'
    assert blob['config']['epsilon_decay'] == 0.999
    assert blob['config']['epsilon_decay_start_episode'] == 2

    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5

    # During episode 1: epsilon is 0.15. end_episode(1) does not decay (1 <= 2).
    # During episode 2: epsilon is 0.15. end_episode(2) does not decay (2 <= 2).
    # During episode 3: epsilon is 0.15. end_episode(3) decays (3 > 2) -> 0.15 * 0.999.
    # During episode 4: epsilon is 0.15 * 0.999. end_episode(4) decays (4 > 2) -> 0.15 * 0.999^2.
    # During episode 5: epsilon is 0.15 * 0.999^2.
    expected_epsilons = [
        0.15,
        0.15,
        0.15,
        0.15 * 0.999,
        0.15 * (0.999 ** 2),
    ]
    for row, expected in zip(rows, expected_epsilons):
        assert float(row['epsilon']) == pytest.approx(expected, rel=1e-7)

    # Verify summary for default schedule does not have epsilon_decay or epsilon_decay_start_episode
    log_default = tmp_path / 'default.csv'
    ckpt_default = tmp_path / 'default.pt'
    result_default = subprocess.run(
        [sys.executable, 'main.py', 'train', '--from-scratch', '--episodes', '1',
         '--arch', 'onehot_dueling', '--seed', '101', '--threads', '1',
         '--save-as', str(ckpt_default), '--log', str(log_default)],
        cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result_default.returncode == 0, result_default.stdout + result_default.stderr
    summary_default = json.loads(log_default.with_suffix('.json').read_text())
    assert 'epsilon_decay' not in summary_default
    assert 'epsilon_decay_start_episode' not in summary_default
