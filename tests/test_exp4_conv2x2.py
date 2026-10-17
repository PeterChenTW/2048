"""Unit and integration tests for Experiment #4: OneHotConv2x2DQN ('onehot_conv2x2')."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DQNConfig
from conv2x2 import OneHotConv2x2DQN
from dueling import OneHotDuelingDQN
from eval import evaluate
from RL import DDQNAgent, load_checkpoint


def _sample_boards(batch_size: int = 6, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    exponents = rng.integers(0, 16, size=(batch_size, 4, 4))
    boards = np.where(exponents == 0, 0, 1 << exponents).astype(np.float32)
    return torch.from_numpy(boards)


def _equal_state_dict(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_conv2x2_output_and_feature_shapes() -> None:
    torch.manual_seed(7)
    model = OneHotConv2x2DQN()
    assert OneHotConv2x2DQN.architecture == "onehot_conv2x2_v1"
    assert model.architecture == "onehot_conv2x2_v1"

    boards_n44 = _sample_boards(batch_size=5, seed=10)
    boards_n16 = boards_n44.reshape(5, 16)
    single_44 = boards_n44[0]
    single_16 = boards_n16[0]

    feat_n44 = model.forward_features(boards_n44)
    feat_n16 = model.forward_features(boards_n16)
    feat_44 = model.forward_features(single_44)
    feat_16 = model.forward_features(single_16)

    assert feat_n44.shape == (5, 256)
    assert feat_n16.shape == (5, 256)
    assert feat_44.shape == (1, 256)
    assert feat_16.shape == (1, 256)
    assert torch.allclose(feat_n44, feat_n16, atol=1e-6)
    assert torch.allclose(feat_44, feat_16, atol=1e-6)

    q_n44 = model(boards_n44)
    q_n16 = model(boards_n16)
    q_44 = model(single_44)
    q_16 = model(single_16)

    assert q_n44.shape == (5, 4)
    assert q_n16.shape == (5, 4)
    assert q_44.shape == (1, 4)
    assert q_16.shape == (1, 4)
    assert torch.allclose(q_n44, q_n16, atol=1e-6)
    assert torch.allclose(q_44, q_16, atol=1e-6)


def test_conv2x2_parameter_count() -> None:
    model = OneHotConv2x2DQN()
    dueling_model = OneHotDuelingDQN()
    conv2x2_params = sum(p.numel() for p in model.parameters())
    dueling_params = sum(p.numel() for p in dueling_model.parameters())
    print(f"OneHotConv2x2DQN parameter count: {conv2x2_params} (vs OneHotDuelingDQN: {dueling_params})")

    # Exact breakdown:
    # conv1: (16 * 2 * 2 + 1) * 64 = 4,160
    # conv2: (64 * 2 * 2 + 1) * 128 = 32,896
    # fc:    (512 + 1) * 256 = 131,328
    # head.value_stream: (256 + 1) * 256 + (256 + 1) * 1 = 66,049
    # head.advantage_stream: (256 + 1) * 256 + (256 + 1) * 4 = 66,820
    # total = 301,253
    assert conv2x2_params == 301_253
    assert conv2x2_params < dueling_params


def test_conv2x2_gradients_reach_every_parameter() -> None:
    torch.manual_seed(99)
    model = OneHotConv2x2DQN()
    boards = torch.tensor(
        [
            [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            [2, 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536],
        ],
        dtype=torch.float32,
    )
    q = model(boards)
    weights = torch.tensor([[1.0, 2.0, -0.5, 0.25], [-1.0, 0.5, 1.5, -0.25]])
    loss = (q * weights).sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} received no gradient"
        assert torch.any(param.grad != 0), f"Parameter {name} has all-zero gradient"


def test_conv2x2_save_load_checkpoint_roundtrip_and_cross_arch_errors(tmp_path: Path) -> None:
    cfg = DQNConfig(arch="onehot_conv2x2", n_step=3, gamma=0.95, lr=1e-4, seed=21,
                    memory_capacity=64, batch_size=8)
    agent = DDQNAgent(16, 4, cfg)
    assert isinstance(agent.model, OneHotConv2x2DQN)

    ckpt_path = tmp_path / "conv2x2.pt"
    agent.save(ckpt_path)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert blob["format"] == "dqn2048-v2"
    assert blob["architecture"] == "onehot_conv2x2_v1"
    assert blob["config"]["arch"] == "onehot_conv2x2"

    loaded = load_checkpoint(ckpt_path)
    assert isinstance(loaded.model, OneHotConv2x2DQN)
    assert loaded.config.arch == "onehot_conv2x2"
    assert loaded.config.n_step == 3
    assert loaded.config.gamma == pytest.approx(0.95)
    _equal_state_dict(agent.model.state_dict(), loaded.model.state_dict())
    _equal_state_dict(loaded.model.state_dict(), loaded.target_model.state_dict())

    boards = _sample_boards(batch_size=4, seed=123)
    with torch.no_grad():
        assert torch.equal(agent.model(boards), loaded.model(boards))

    fresh = DDQNAgent(16, 4, DQNConfig(arch="onehot_conv2x2", seed=99))
    fresh.load(ckpt_path)
    _equal_state_dict(agent.model.state_dict(), fresh.model.state_dict())

    # Cross-arch load errors: conv2x2 ckpt into onehot_dueling agent and vice versa
    dueling_agent = DDQNAgent(16, 4, DQNConfig(arch="onehot_dueling", seed=5))
    with pytest.raises(ValueError, match="onehot_conv2x2"):
        dueling_agent.load(ckpt_path)

    dueling_ckpt = tmp_path / "dueling.pt"
    dueling_agent.save(dueling_ckpt)
    with pytest.raises(ValueError, match="onehot_dueling"):
        agent.load(dueling_ckpt)

    legacy_agent = DDQNAgent(16, 4, DQNConfig(arch="legacy", seed=5))
    with pytest.raises(ValueError, match="onehot_conv2x2"):
        legacy_agent.load(ckpt_path)

    # Also verify eval.evaluate works with conv2x2 checkpoint
    eval_res = evaluate(checkpoint=ckpt_path, policy="greedy", games=1, board="fast", max_steps=40)
    assert eval_res["games"] == 1
    assert 0 < eval_res["per_game"][0]["steps"] <= 40
    assert math.isfinite(eval_res["mean"])


def test_conv2x2_agent_remember_and_replay_batches() -> None:
    cfg = DQNConfig(arch="onehot_conv2x2", memory_capacity=64, batch_size=8, lr=1e-4, seed=42)
    agent = DDQNAgent(16, 4, cfg)
    boards = _sample_boards(batch_size=20, seed=77).reshape(20, 16)

    for i in range(16):
        state = boards[i]
        next_state = boards[i + 1]
        action = i % 4
        reward = float((i % 5) * 0.1)
        done = (i == 15)
        invalid = (0,) if (i % 3 == 0 and not done) else ()
        agent.remember(state, action, reward, next_state, done, next_invalid_moves=invalid)

    for _ in range(4):
        stats = agent.replay()
        assert not stats["nan"]
        assert math.isfinite(stats["loss"])
        assert math.isfinite(stats["q_mean"])
        assert math.isfinite(stats["td_abs_mean"])
    assert agent.replay_calls == 4
    assert len(agent.losses) == 4


def test_main_conv2x2_requires_from_scratch_on_legacy_ckpt() -> None:
    import main as main_mod

    with pytest.raises(ValueError, match="--arch onehot_conv2x2 requires --from-scratch"):
        main_mod.main(["train", "--arch", "onehot_conv2x2", "--episodes", "1"])

