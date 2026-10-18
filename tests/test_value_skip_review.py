"""Adversarial checks for the afterstate L1-skip arm (proposal S).

These tests pin behavior that the implementation tests do not isolate:
target sync, refusing to load an S checkpoint as ``none``, the same-seed
conv-init phenomenon, and gradient flow through the skip path itself.
"""
from pathlib import Path
import subprocess

import pytest
import torch

from afterstate_agent import AfterstateAgent
from config import DQNConfig
from eval import load_agent
from RL import load_checkpoint
from value_net import OneHotValueNet

BOARDS = torch.tensor([
    [2, 2, 4, 0, 4, 8, 0, 0, 16, 0, 32, 0, 0, 0, 0, 0],
    [0, 2, 0, 4, 8, 8, 16, 0, 32, 64, 0, 0, 128, 256, 512, 1024],
    [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2],
], dtype=torch.float32)
L2_FEATURES = 2176
SKIP_FEATURES = 3072


def _assert_synced(agent):
    assert agent.model.architecture == "onehot_afterstate_skip_v1"
    assert agent.target_model.architecture == "onehot_afterstate_skip_v1"
    assert agent.model.value_skip == agent.target_model.value_skip == "l1"
    assert agent.model.trunk_features == agent.target_model.trunk_features == L2_FEATURES + SKIP_FEATURES
    assert list(agent.model.state_dict()) == list(agent.target_model.state_dict())
    for key, value in agent.model.state_dict().items():
        assert torch.equal(value, agent.target_model.state_dict()[key]), key
    assert torch.equal(agent.model(BOARDS), agent.target_model(BOARDS))


def test_skip_target_model_architecture_matches_after_sync():
    agent = AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip="l1", seed=17,
        batch_size=2, memory_capacity=8, per_max_priority="tree"))
    _assert_synced(agent)

    for i in range(4):
        agent.remember(BOARDS[i % 3], BOARDS[(i + 1) % 3], False)
    stats = agent.replay()
    assert not stats["nan"]
    diverged = [
        key for key, value in agent.model.state_dict().items()
        if not torch.equal(value, agent.target_model.state_dict()[key])
    ]
    assert diverged, "replay left the online net identical to the target"
    assert agent.target_model.architecture == "onehot_afterstate_skip_v1"
    assert agent.model.architecture == "onehot_afterstate_skip_v1"

    agent.update_target_model()
    _assert_synced(agent)


def test_skip_checkpoint_cannot_be_loaded_as_none(tmp_path):
    path = tmp_path / "skip.pt"
    source = AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip="l1", seed=9))
    expected = source.model(BOARDS).detach()
    source.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["architecture"] == "onehot_afterstate_skip_v1"
    assert payload["config"]["value_skip"] == "l1"
    assert payload["config"]["arch"] == "onehot_afterstate"

    none_agent = AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip="none", seed=9))
    assert none_agent.model.architecture == "onehot_afterstate_v1"
    with pytest.raises(ValueError, match="value_skip"):
        none_agent.load(path)
    assert none_agent.model.architecture == "onehot_afterstate_v1"
    assert none_agent.config.value_skip == "none"
    with pytest.raises(RuntimeError, match="size mismatch"):
        none_agent.model.load_state_dict(payload["state_dict"], strict=True)

    for loaded in (load_checkpoint(path), load_agent(path)):
        assert loaded.config.value_skip == "l1"
        assert loaded.model.value_skip == loaded.target_model.value_skip == "l1"
        assert loaded.model.architecture == loaded.target_model.architecture == "onehot_afterstate_skip_v1"
        assert torch.equal(loaded.model(BOARDS), expected)
        assert torch.equal(loaded.target_model(BOARDS), expected)
    action = load_agent(path).greedy_action(BOARDS[0].tolist(), [])
    assert action in (0, 1, 2, 3)


def test_same_seed_conv_init_phenomenon_is_recorded():
    """Record whether one seed gives S and none the same conv initialization.

    S_PREREG does not require the full initial weights to match: the value
    head's input width changed, so the same seed must not reproduce C's head.
    Conv equality is only an observation. It holds because both arms build
    the same discarded dueling head before the value head, and that head is
    what consumes the RNG after the convolutions.
    """
    torch.manual_seed(202)
    none = OneHotValueNet(value_skip="none")
    torch.manual_seed(202)
    skip = OneHotValueNet(value_skip="l1")

    none_conv = none.trunk.state_dict()
    skip_conv = skip.trunk.state_dict()
    assert list(none_conv) == list(skip_conv)
    matched = [key for key in none_conv if torch.equal(none_conv[key], skip_conv[key])]
    # Recorded phenomenon, not a prereg requirement: every conv tensor matches.
    assert matched == list(none_conv)

    assert [name for name, _ in none.named_parameters()] == [name for name, _ in skip.named_parameters()]
    expected_conv = {
        "conv_h": ((16, 128), (1, 2)),
        "conv_v": ((16, 128), (2, 1)),
        "conv_hh": ((128, 64), (1, 2)),
        "conv_hv": ((128, 64), (2, 1)),
        "conv_vh": ((128, 64), (1, 2)),
        "conv_vv": ((128, 64), (2, 1)),
    }
    for name, ((in_channels, out_channels), kernel) in expected_conv.items():
        left, right = getattr(none.trunk, name), getattr(skip.trunk, name)
        assert type(left) is type(right) is torch.nn.Conv2d
        assert left.in_channels == right.in_channels == in_channels
        assert left.out_channels == right.out_channels == out_channels
        assert left.kernel_size == right.kernel_size == kernel
        assert left.stride == right.stride == (1, 1)
        assert left.padding == right.padding == (0, 0)
        assert left.dilation == right.dilation == (1, 1)
        assert left.bias is not None and right.bias is not None

    assert none.head[0].in_features == L2_FEATURES
    assert skip.head[0].in_features == L2_FEATURES + SKIP_FEATURES
    assert none.head[0].out_features == skip.head[0].out_features == 256
    assert none.head[2].in_features == skip.head[2].in_features == 256
    assert none.head[0].weight.shape != skip.head[0].weight.shape
    assert OneHotValueNet.architecture == "onehot_afterstate_v1"
    assert none.architecture == "onehot_afterstate_v1"
    assert skip.architecture == "onehot_afterstate_skip_v1"


def test_l1_skip_path_alone_sends_gradients_to_conv_h_and_conv_v():
    """Zero the head columns that read L2. conv_h / conv_v must still get grad.

    A nonzero conv gradient from an ordinary loss can come entirely from the
    L2 stack. This closes that hole: with the L2 columns removed, the only
    remaining path is the post-ReLU h and v concatenation.
    """
    torch.manual_seed(11)
    net = OneHotValueNet(value_skip="l1")
    assert net.trunk_features == net.head[0].in_features == L2_FEATURES + SKIP_FEATURES
    assert net.hidden == 256

    with torch.no_grad():
        net.trunk.conv_h.bias.fill_(0.5)
        net.trunk.conv_v.bias.fill_(0.5)
        net.head[0].weight.zero_()
        net.head[0].bias.fill_(1.0)
        net.head[0].weight[:, L2_FEATURES:] = 1.0
        net.head[2].weight.fill_(1.0)
        net.head[2].bias.zero_()

    features = net.forward_features(BOARDS)
    assert features.shape == (3, L2_FEATURES + SKIP_FEATURES)
    assert (features[:, L2_FEATURES:L2_FEATURES + 1536] > 0).any()
    assert (features[:, L2_FEATURES + 1536:] > 0).any()

    net.zero_grad(set_to_none=True)
    net(BOARDS).sum().backward()

    for name in ("conv_h", "conv_v"):
        layer = getattr(net.trunk, name)
        assert layer.weight.grad is not None, name
        assert layer.bias.grad is not None, name
        assert layer.weight.grad.abs().sum() > 0, name
        assert layer.bias.grad.abs().sum() > 0, name
    for name in ("conv_hh", "conv_hv", "conv_vh", "conv_vv"):
        grad = getattr(net.trunk, name).weight.grad
        assert grad is not None, name
        assert torch.equal(grad, torch.zeros_like(grad)), name


def test_bitwise_baseline_commit_predates_skip_source():
    """The 33ae2ba tree archived by the bitwise test is the pre-skip network.

    ``tests/test_value_skip.py`` runs that tree in a child process. This check
    only locks the commit contents so a moved or rewritten baseline cannot
    silently become the worktree under test.
    """
    repo = Path(__file__).resolve().parents[1]
    text = subprocess.check_output(
        ["git", "-C", str(repo), "show", "33ae2ba:value_net.py"],
        text=True, timeout=30)
    assert "value_skip" not in text
    assert "_L1SkipDirectionalTrunk" not in text
    assert 'architecture: str = "onehot_afterstate_v1"' in text
    rl = subprocess.check_output(
        ["git", "-C", str(repo), "show", "33ae2ba:RL.py"],
        text=True, timeout=30)
    assert "onehot_afterstate_skip_v1" not in rl
    assert "value_skip" not in rl
