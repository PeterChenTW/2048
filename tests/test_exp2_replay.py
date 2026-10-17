import csv
import json
import numpy as np
import pytest
import torch

from config import DQNConfig
from main import parser, FIELDS, train
from RL import DDQNAgent, SumTree


def test_default_config_invariance():
    cfg = DQNConfig()
    assert cfg.memory_capacity == 6000
    assert cfg.per_max_priority == 'scan'
    assert cfg.track_replay_age is False

    agent = DDQNAgent(16, 4)
    assert agent.memory.capacity == 6000
    assert agent.config.per_max_priority == 'scan'
    assert agent.config.track_replay_age is False
    assert agent.step_at_insert is None
    assert agent.updates_at_insert is None
    assert agent.memory.max_tree is None

    # remember and replay with defaults should not contain age keys
    state = torch.zeros((1, 16), dtype=torch.float32)
    agent.remember(state, 0, 1.0, state, False)
    stats = agent.replay(batch_size=1)
    assert 'age_steps_mean' not in stats
    assert 'age_updates_mean' not in stats


def test_invalid_per_max_priority():
    with pytest.raises(ValueError, match="unknown per_max_priority"):
        DDQNAgent(16, 4, DQNConfig(per_max_priority='invalid'))


def test_cli_parsing():
    p = parser()
    args = p.parse_args(['train'])
    assert args.memory_capacity is None
    assert args.per_max_priority is None
    assert args.track_replay_age is False

    args = p.parse_args(['train', '--memory-capacity', '100000',
                         '--per-max-priority', 'tree', '--track-replay-age'])
    assert args.memory_capacity == 100000
    assert args.per_max_priority == 'tree'
    assert args.track_replay_age is True

    with pytest.raises(SystemExit):
        p.parse_args(['train', '--memory-capacity', '0'])

    with pytest.raises(SystemExit):
        p.parse_args(['train', '--per-max-priority', 'heap'])


def test_tree_scan_parity():
    seed = 42
    capacity = 50
    batch_size = 16

    cfg_scan = DQNConfig(memory_capacity=capacity, batch_size=batch_size,
                         per_max_priority='scan', seed=seed, lr=1e-4)
    cfg_tree = DQNConfig(memory_capacity=capacity, batch_size=batch_size,
                         per_max_priority='tree', seed=seed, lr=1e-4)

    agent_scan = DDQNAgent(16, 4, cfg_scan)
    agent_tree = DDQNAgent(16, 4, cfg_tree)

    agent_tree.model.load_state_dict(agent_scan.model.state_dict())
    agent_tree.target_model.load_state_dict(agent_scan.target_model.state_dict())

    # Feed identical stream of transitions and replay periodically
    rng = np.random.default_rng(123)
    for i in range(250):  # Wraps capacity 50 five times
        state = rng.random((1, 16)).astype(np.float32)
        action = int(rng.integers(0, 4))
        reward = float(rng.uniform(0.0, 10.0))
        next_state = rng.random((1, 16)).astype(np.float32)
        done = bool(rng.random() < 0.1)
        invalid = [int(a) for a in rng.choice(4, size=int(rng.integers(0, 3)), replace=False)]

        # Before insert, verify max priorities match
        max_p_scan = np.max(agent_scan.memory.tree[-agent_scan.memory.capacity:])
        if max_p_scan == 0:
            max_p_scan = agent_scan.config.abs_error_upper
        max_p_tree = agent_tree.memory.max_priority()
        if max_p_tree == 0:
            max_p_tree = agent_tree.config.abs_error_upper

        assert max_p_scan == pytest.approx(max_p_tree, abs=1e-12)

        agent_scan.remember(state, action, reward, next_state, done, invalid)
        agent_tree.remember(state, action, reward, next_state, done, invalid)

        # Trees should match exactly
        np.testing.assert_allclose(agent_scan.memory.tree, agent_tree.memory.tree, rtol=1e-10)

        if agent_scan.memory.size >= batch_size and i % 5 == 0:
            stats_scan = agent_scan.replay()
            stats_tree = agent_tree.replay()

            for k in stats_scan:
                assert stats_scan[k] == pytest.approx(stats_tree[k], rel=1e-5, abs=1e-6)
            np.testing.assert_allclose(agent_scan.memory.tree, agent_tree.memory.tree, rtol=1e-10)


def test_age_tracking_hand_example():
    cfg = DQNConfig(memory_capacity=5, batch_size=2, track_replay_age=True, seed=999)
    agent = DDQNAgent(16, 4, cfg)

    # Insert 3 transitions at specific env steps
    agent.set_env_step(10)
    state = torch.zeros((1, 16), dtype=torch.float32)
    agent.remember(state, 0, 1.0, state, False)

    agent.set_env_step(25)
    agent.remember(state, 1, 2.0, state, False)

    agent.set_env_step(40)
    agent.remember(state, 2, 3.0, state, False)

    assert list(agent.step_at_insert[:3]) == [10, 25, 40]
    assert list(agent.updates_at_insert[:3]) == [0, 0, 0]

    # Replay at env_step 100
    agent.set_env_step(100)
    stats1 = agent.replay(batch_size=2)
    assert agent.replay_calls == 1
    # Samples must be from entries 0, 1, or 2
    # Possible step ages are: (100-10)=90, (100-25)=75, (100-40)=60
    assert 60.0 <= stats1['age_steps_mean'] <= 90.0
    # Before replay1 was called, replay_calls was 0, and all transitions were inserted at replay_calls 0
    assert stats1['age_updates_mean'] == 0.0

    # Insert 4th transition at env_step 110 (replay_calls is now 1)
    agent.set_env_step(110)
    agent.remember(state, 3, 4.0, state, False)
    assert agent.step_at_insert[3] == 110
    assert agent.updates_at_insert[3] == 1

    # Replay at env_step 150
    agent.set_env_step(150)
    stats2 = agent.replay(batch_size=2)
    assert agent.replay_calls == 2
    # At replay2, replay_calls was 1. Items 0-2 have updates_at_insert=0 (age=1), item 3 has 1 (age=0)
    assert 0.0 <= stats2['age_updates_mean'] <= 1.0

    # Overwrite buffer (capacity 5): insert entries 4 and 5 (5 overwrites index 0)
    agent.set_env_step(200)
    agent.remember(state, 0, 5.0, state, False)  # slot 4
    agent.set_env_step(250)
    agent.remember(state, 1, 6.0, state, False)  # slot 0 (wrap-around)
    assert agent.step_at_insert[0] == 250
    assert agent.updates_at_insert[0] == 2


def test_main_summary_and_csv_with_age(tmp_path):
    p = parser()
    log_csv = tmp_path / 'log.csv'
    chk = tmp_path / 'model.pt'
    args = p.parse_args(['train', '--from-scratch', '--episodes', '2',
                         '--memory-capacity', '100', '--per-max-priority', 'tree',
                         '--track-replay-age', '--save-as', str(chk), '--log', str(log_csv)])
    train(args)

    assert log_csv.exists()
    summary_file = log_csv.with_suffix('.json')
    assert summary_file.exists()

    summary = json.loads(summary_file.read_text())
    assert summary['memory_capacity'] == 100
    assert summary['per_max_priority'] == 'tree'
    assert summary['track_replay_age'] is True

    with log_csv.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == list(FIELDS) + ['age_steps_mean', 'age_updates_mean']


def test_main_summary_and_csv_without_age(tmp_path):
    p = parser()
    log_csv = tmp_path / 'log_no_age.csv'
    chk = tmp_path / 'model_no_age.pt'
    args = p.parse_args(['train', '--from-scratch', '--episodes', '1',
                         '--save-as', str(chk), '--log', str(log_csv)])
    train(args)

    assert log_csv.exists()
    summary = json.loads(log_csv.with_suffix('.json').read_text())
    assert summary['memory_capacity'] == 6000
    assert summary['per_max_priority'] == 'scan'
    assert summary['track_replay_age'] is False

    with log_csv.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == FIELDS
