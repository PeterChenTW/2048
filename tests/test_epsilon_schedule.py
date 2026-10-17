"""Exploration schedules follow either optimizer updates or environment steps."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DQNConfig
from RL import DDQNAgent


REPO = Path(__file__).resolve().parents[1]


def test_linear_schedule_values():
    agent = DDQNAgent(16, 4, DQNConfig(epsilon_schedule='linear_steps',
                                     epsilon_start=0.8, epsilon_min=0.1,
                                     epsilon_decay_steps=200_000))
    for step, expected in [(0, 0.8), (100_000, 0.45),
                           (200_000, 0.1), (300_000, 0.1)]:
        agent.set_env_step(step)
        assert agent.epsilon == pytest.approx(expected)


@pytest.mark.parametrize('schedule', ['linear_steps', 'update_exp'])
def test_replay_epsilon_schedule(schedule):
    config = DQNConfig(epsilon_schedule=schedule, memory_capacity=2, batch_size=1,
                       epsilon_start=0.8, epsilon_min=0.1, epsilon_decay=0.5)
    agent = DDQNAgent(16, 4, config)
    agent.model = torch.nn.Linear(16, 4)
    agent.target_model = torch.nn.Linear(16, 4)
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0)
    agent.remember(torch.zeros(1, 16), 0, 1, torch.zeros(1, 16), True)
    agent.set_env_step(100_000)
    expected = (config.epsilon_start - (config.epsilon_start - config.epsilon_min) * 0.5
                if schedule == 'linear_steps' else 0.8)
    assert agent.epsilon == expected
    for _ in range(5):
        stats = agent.replay()
        if schedule == 'update_exp':
            expected = max(config.epsilon_min, expected * config.epsilon_decay)
        assert agent.epsilon == expected
        assert stats['epsilon'] == expected


def test_defaults_preserve_update_schedule():
    config = DQNConfig()
    assert config.epsilon_schedule == 'update_exp'
    assert config.epsilon_decay_steps == 200_000
    agent = DDQNAgent(16, 4, config)
    agent.set_env_step(1_000_000)
    assert agent.epsilon == config.epsilon_start


def test_cli_linear_schedule_reaches_agent(tmp_path):
    checkpoint = tmp_path / 'trained.pt'
    log = tmp_path / 'train.csv'
    result = subprocess.run(
        [sys.executable, 'main.py', 'train', '--from-scratch', '--episodes', '2',
         '--arch', 'onehot_dueling', '--seed', '0',
         '--eps-schedule', 'linear_steps', '--eps-start', '0.8',
         '--eps-min', '0.1', '--eps-decay-steps', '200000',
         '--save-as', str(checkpoint), '--log', str(log)],
        cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(log.with_suffix('.json').read_text())
    expected_config = dict(epsilon_schedule='linear_steps', epsilon_start=0.8,
                           epsilon_min=0.1, epsilon_decay_steps=200_000)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for field, expected in expected_config.items():
        assert summary[field] == expected
        assert blob['config'][field] == expected
    assert summary['episodes'] == 2
    assert summary['updates'] > 0
    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    total_steps = 0
    for row in rows:
        total_steps += int(row['steps'])
        expected = max(0.1, 0.8 - 0.7 * min(total_steps / 200_000, 1))
        assert float(row['epsilon']) == pytest.approx(expected)
    assert total_steps == summary['env_steps']
