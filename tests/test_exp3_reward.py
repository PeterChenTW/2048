"""Tests for Experiment 3: Linear score reward shaping and gamma configuration."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from config import DQNConfig
from fast_board import FastBoard
from main import parser
from rewards import shape_reward
from RL import DDQNAgent, load_checkpoint


def test_log_mode_equals_fast_board_move_reward():
    """Verify log mode returns bit-identical rewards to FastBoard.move across seeded games."""
    for seed in range(5):
        rng = random.Random(seed)
        board = FastBoard(rng=rng)
        while not board.done and board.turns < 200:
            invalid = board.get_invalid_moves()
            if len(invalid) == 4:
                break
            action = rng.choice(range(4))
            _, env_reward, _ = board.move(board.all_moves[action])
            spawned = (env_reward != -1)
            shaped = shape_reward(board.get_score, spawned, mode='log')
            assert shaped == env_reward
            if action in invalid:
                assert env_reward == -1
                assert not spawned
                assert shaped == -1


def test_linear_mode_sum_and_scale():
    """Verify sum of merge_score equals score_v2 and linear reward == merge_score / scale."""
    for seed in (42, 123):
        rng = random.Random(seed)
        board = FastBoard(rng=rng)
        total_merge_score = 0
        scale = 256.0
        while not board.done and board.turns < 250:
            invalid = board.get_invalid_moves()
            if len(invalid) == 4:
                break
            action = rng.choice([a for a in range(4) if a not in invalid])
            _, env_reward, _ = board.move(board.all_moves[action])
            spawned = (env_reward != -1)
            if spawned:
                total_merge_score += board.get_score
                linear_reward = shape_reward(board.get_score, spawned, mode='linear', scale=scale)
                assert linear_reward == board.get_score / scale
            else:
                assert shape_reward(board.get_score, spawned, mode='linear', scale=scale) == -1
        assert total_merge_score == board.score_v2


def test_no_spawn_move_gives_negative_one():
    """Verify both modes return -1 on no-spawn transitions."""
    for mode in ('log', 'linear'):
        assert shape_reward(0, False, mode=mode) == -1
        assert shape_reward(128, False, mode=mode) == -1
        assert shape_reward(1024, False, mode=mode, scale=512.0) == -1

    # Also on a real FastBoard when executing an invalid move
    grid = [
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 4096],
        [8192, 16384, 32768, 65536],
    ]
    board = FastBoard.from_grid(grid)
    for action in range(4):
        _, env_reward, _ = board.move(board.all_moves[action])
        assert env_reward == -1
        assert shape_reward(board.get_score, False, mode='log') == -1
        assert shape_reward(board.get_score, False, mode='linear') == -1


def test_cli_parsing_and_config():
    """Verify CLI parsing including --gamma, --reward-mode, and --reward-scale."""
    p = parser()
    args = p.parse_args(['train', '--reward-mode', 'linear', '--reward-scale', '512',
                         '--gamma', '0.997'])
    assert args.reward_mode == 'linear'
    assert args.reward_scale == 512.0
    assert args.gamma == 0.997

    # Construct DQNConfig with parsed args like main.py
    config_args = {}
    if args.gamma is not None:
        config_args['gamma'] = args.gamma
    if args.reward_mode is not None:
        config_args['reward_mode'] = args.reward_mode
    if args.reward_scale is not None:
        config_args['reward_scale'] = args.reward_scale

    config = DQNConfig(**config_args)
    assert config.reward_mode == 'linear'
    assert config.reward_scale == 512.0
    assert config.gamma == 0.997

    # Default CLI parsing
    args_default = p.parse_args(['train'])
    assert args_default.reward_mode is None
    assert args_default.reward_scale is None
    assert args_default.gamma is None

    config_default = DQNConfig()
    assert config_default.reward_mode == 'log'
    assert config_default.reward_scale == 256.0
    assert config_default.gamma == 0.99


def test_v2_checkpoint_roundtrip_keeps_reward_mode(tmp_path):
    """Verify v2 checkpoint saves and restores reward_mode, reward_scale, and gamma."""
    cfg = DQNConfig(arch='onehot_dueling', reward_mode='linear',
                    reward_scale=128.0, gamma=0.997)
    agent = DDQNAgent(16, 4, config=cfg)
    ckpt_path = tmp_path / 'chk_v2.pt'
    agent.save(str(ckpt_path))

    loaded_agent = load_checkpoint(str(ckpt_path))
    assert loaded_agent.config.reward_mode == 'linear'
    assert loaded_agent.config.reward_scale == 128.0
    assert loaded_agent.config.gamma == 0.997
    assert loaded_agent.config.arch == 'onehot_dueling'


def test_invalid_mode_and_scale_raises():
    """Verify invalid reward mode and scale raise ValueError in shape_reward and DDQNAgent."""
    with pytest.raises(ValueError, match='unknown reward mode'):
        shape_reward(10, True, mode='exponential')

    with pytest.raises(ValueError, match='scale must be a finite positive number'):
        shape_reward(10, True, mode='linear', scale=0)

    with pytest.raises(ValueError, match='scale must be a finite positive number'):
        shape_reward(10, True, mode='linear', scale=-10.0)

    with pytest.raises(ValueError, match='unknown reward mode'):
        DDQNAgent(16, 4, config=DQNConfig(reward_mode='bad'))

    with pytest.raises(ValueError, match='reward_scale must be a finite positive number'):
        DDQNAgent(16, 4, config=DQNConfig(reward_scale=0))

    with pytest.raises(ValueError, match='reward_scale must be a finite positive number'):
        DDQNAgent(16, 4, config=DQNConfig(reward_scale=-5.0))


def test_train_summary_json_contains_reward_fields(tmp_path):
    """Verify main.py train includes reward_mode, reward_scale, and gamma in summary JSON."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    ckpt = tmp_path / 'ckpt.pt'
    log = tmp_path / 'log.csv'
    res = subprocess.run(
        [sys.executable, 'main.py', 'train', '--from-scratch', '--episodes', '1',
         '--arch', 'onehot_dueling', '--reward-mode', 'linear', '--reward-scale', '512',
         '--gamma', '0.997', '--save-as', str(ckpt), '--log', str(log), '--threads', '1'],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    summary_path = log.with_suffix('.json')
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary['reward_mode'] == 'linear'
    assert summary['reward_scale'] == 512.0
    assert summary['gamma'] == 0.997
