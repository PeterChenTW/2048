"""Exercise the experiment union and freeze the pre-experiment defaults."""
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
import torch

from config import DQNConfig
from eval import evaluate
from main import FIELDS


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_DEFAULTS = {
    'gamma': 0.99, 'lr': 1e-4, 'memory_capacity': 6000, 'batch_size': 128,
    'per_alpha': 0.6, 'per_beta_start': 0.4, 'per_beta_end': 1.0,
    'per_beta_anneal_steps': 100_000, 'per_eps': 1e-5, 'abs_error_upper': 1.0,
    'epsilon_start': 0.15, 'epsilon_min': 0.001, 'epsilon_decay': 0.999,
    'epsilon_schedule': 'update_exp', 'epsilon_decay_steps': 200_000,
    'target_update_episodes': 50, 'grad_clip': 1.0, 'seed': None,
    'arch': 'legacy', 'n_step': 1,
}
ORIGINAL_FIELDS = [
    'episode', 'score_v2', 'max_tile', 'steps', 'epsilon', 'beta',
    'loss', 'q_mean', 'q_min', 'q_max', 'td_abs_mean', 'is_weight_mean',
    'nan_count', 'elapsed_s',
]


def assert_finite_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_numbers(item)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value)


def test_original_defaults_and_csv_fields_unchanged():
    defaults = asdict(DQNConfig())
    assert {key: defaults[key] for key in ORIGINAL_DEFAULTS} == ORIGINAL_DEFAULTS
    assert defaults['per_max_priority'] == 'scan'
    assert defaults['track_replay_age'] is False
    assert defaults['reward_mode'] == 'log'
    assert defaults['reward_scale'] == 256.0
    assert defaults['restart_frac'] == 0.0
    assert defaults['restart_pool_size'] == 5000
    assert FIELDS == ORIGINAL_FIELDS
    # test_exp5_restart also compares real default CSV rows to frozen 0f541e0 rows.


@pytest.mark.parametrize('arch,n_step', [('legacy', 1), ('legacy', 3),
                                          ('onehot_afterstate', 1)])
def test_driver_records_completed_step_at_replay_insertion(tmp_path, monkeypatch, arch, n_step):
    import main
    from fast_board import FastBoard
    from RL import DDQNAgent
    from afterstate_agent import AfterstateAgent

    completed = 0
    inserted = []
    original_move = FastBoard.move
    agent_class = AfterstateAgent if arch == 'onehot_afterstate' else DDQNAgent
    original_remember = agent_class.remember

    def three_moves(self, direction):
        nonlocal completed
        state, reward, _ = original_move(self, direction)
        completed += 1
        return state, reward, completed == 3

    def record_insertion(self, *args, **kwargs):
        original_remember(self, *args, **kwargs)
        slot = (self.memory.pointer - 1) % self.memory.capacity
        inserted.append((int(self.step_at_insert[slot]), completed))

    monkeypatch.setattr(FastBoard, 'move', three_moves)
    monkeypatch.setattr(agent_class, 'remember', record_insertion)
    assert main.main([
        'train', '--from-scratch', '--episodes', '1', '--threads', '1', '--seed', '7',
        '--arch', arch,
        '--n-step', str(n_step), '--track-replay-age',
        '--save-as', str(tmp_path / 'age.pt'), '--log', str(tmp_path / 'age.csv'),
    ]) == 0
    assert len(inserted) == completed == 3
    assert all(step == actual for step, actual in inserted)


@pytest.mark.parametrize('arch,n_step,episodes', [('onehot_conv2x2', 3, 4),
                                                 ('onehot_afterstate', 1, 3)])
def test_combined_training_and_checkpoint_evaluation(tmp_path, arch, n_step, episodes):
    checkpoint = tmp_path / 'combined.pt'
    log = tmp_path / 'combined.csv'
    trained = subprocess.run(
        ['nice', '-n', '19', sys.executable, 'main.py', 'train', '--from-scratch',
         '--arch', arch, '--n-step', str(n_step), '--memory-capacity', '2000',
         '--per-max-priority', 'tree', '--track-replay-age',
         '--reward-mode', 'linear', '--gamma', '0.997', '--restart-frac', '0.3',
         '--episodes', str(episodes), '--threads', '1', '--seed', '7',
         '--save-as', str(checkpoint), '--log', str(log)],
        cwd=REPO, env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        capture_output=True, text=True, timeout=90,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    with log.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ORIGINAL_FIELDS + [
            'age_steps_mean', 'age_updates_mean', 'start', 'start_max_tile']
        rows = list(reader)
    assert len(rows) == episodes
    for row in rows:
        assert row['start'] in ('fresh', 'restart')
        assert int(row['nan_count']) == 0
        for key in ('age_steps_mean', 'age_updates_mean'):
            assert math.isfinite(float(row[key]))
        assert all(math.isfinite(float(value)) for key, value in row.items()
                   if value and key != 'start')
    summary = json.loads(log.with_suffix('.json').read_text())
    expected = dict(arch=arch, n_step=n_step, memory_capacity=2000,
                    per_max_priority='tree', track_replay_age=True,
                    reward_mode='linear', reward_scale=256.0, gamma=0.997,
                    episodes=episodes, nan_count=0)
    assert {key: summary[key] for key in expected} == expected
    assert {'fresh_episodes', 'restart_episodes', 'fresh_env_steps',
            'restart_env_steps', 'pool_size_final', 'pool_adds',
            'late_stage_fraction', 'score_last_100_fresh_mean'} <= summary.keys()
    assert {
        'arch', 'n_step', 'memory_capacity', 'per_max_priority', 'track_replay_age',
        'episodes', 'gamma', 'reward_mode', 'reward_scale', 'epsilon_schedule',
        'epsilon_start', 'epsilon_min', 'epsilon_decay_steps', 'env_steps', 'updates',
        'replay_ratio_requested', 'replay_ratio_actual', 'duration_s',
        'score_last_100_mean', 'loss_first_50_mean', 'loss_last_50_mean',
        'q_stats_ranges', 'nan_count', 'stopped_reason', 'interrupted',
        'checkpoint', 'loaded_checkpoint',
    } <= summary.keys()
    assert summary['env_steps'] == sum(int(row['steps']) for row in rows)
    assert summary['fresh_episodes'] + summary['restart_episodes'] == episodes
    assert summary['fresh_env_steps'] + summary['restart_env_steps'] == summary['env_steps']
    assert summary['updates'] > 0
    assert summary['stopped_reason'] is None
    assert_finite_numbers(summary)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['architecture'] == f'{arch}_v1'
    for key in ('arch', 'n_step', 'memory_capacity', 'per_max_priority',
                'track_replay_age', 'reward_mode', 'reward_scale', 'gamma'):
        assert payload['config'][key] == expected[key]
    assert payload['config']['restart_frac'] == 0.3
    assert all(torch.isfinite(tensor).all() for tensor in payload['state_dict'].values())
    evaluated = evaluate(checkpoint=checkpoint, games=2, board='fast')
    assert evaluated['games'] == len(evaluated['per_game']) == 2
    assert all(game['steps'] > 0 for game in evaluated['per_game'])
    assert_finite_numbers(evaluated)


def test_afterstate_tree_and_scan_insertion_priorities_match(monkeypatch):
    from afterstate_agent import AfterstateAgent

    agents = [AfterstateAgent(16, 4, DQNConfig(
        arch='onehot_afterstate', memory_capacity=5, batch_size=2,
        per_max_priority=mode, track_replay_age=True, seed=7, lr=0))
        for mode in ('scan', 'tree')]
    scan, tree = agents
    tree_max_calls = []
    original_max = tree.memory.max_priority

    def tracked_max():
        value = original_max()
        tree_max_calls.append(value)
        return value

    monkeypatch.setattr(tree.memory, 'max_priority', tracked_max)
    terminal = np.array([[2, 4, 2, 4], [4, 2, 4, 2],
                         [2, 4, 2, 4], [4, 2, 4, 2]])
    for step in range(16):  # Repeatedly wrap a non-power-of-two replay capacity.
        slot = scan.memory.pointer
        expected_priority = (np.max(scan.memory.tree[-scan.memory.capacity:])
                             or scan.config.abs_error_upper)
        for agent in agents:
            agent.set_env_step(step + 1)
            agent.remember(terminal, terminal, True)
            assert agent.memory.tree[slot + agent.memory.capacity - 1] == expected_priority
            assert agent.step_at_insert[slot] == step + 1
            assert agent.updates_at_insert[slot] == agent.replay_calls
            # Exercise increases and decreases, including replacing the maximum.
            agent.memory.update(slot + agent.memory.capacity - 1, (step % 4 + 1) / 3)
        np.testing.assert_array_equal(scan.memory.tree, tree.memory.tree)
        if step % 3 == 2:
            results = [agent.replay() for agent in agents]
            assert results[0] == results[1]
            assert results[0]['age_steps_mean'] >= 0
            assert results[0]['age_updates_mean'] >= 0
            assert_finite_numbers(results)
            np.testing.assert_array_equal(scan.memory.tree, tree.memory.tree)
    assert len(tree_max_calls) == 16


@pytest.mark.parametrize('mode,scale', [('log', 256.0), ('linear', 128.0)])
def test_afterstate_merge_rewards_match_board_shaping(mode, scale):
    from afterstate import afterstates
    from afterstate_policy import merge_reward
    from fast_board import FastBoard
    from rewards import shape_reward
    import random

    board = FastBoard(rng=random.Random(7))
    for step in range(100):
        _, merges, legal = afterstates(board.grid)
        rewards = merge_reward(merges, mode, scale)
        expected = np.asarray([shape_reward(int(score), True, mode, scale)
                               for score in merges], dtype=np.float32)
        np.testing.assert_array_equal(rewards, expected)
        actions = np.flatnonzero(legal)
        if not len(actions):
            break
        action = int(actions[step % len(actions)])
        _, env_reward, done = board.move(board.all_moves[action])
        if mode == 'log':
            assert shape_reward(board.get_score, True) == env_reward
            assert rewards[action] == np.float32(env_reward)
        if done:
            break
