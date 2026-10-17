import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dueling import OneHotDuelingDQN
from value_net import OneHotValueNet


def _boards(n, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], dtype=np.float32)
    return torch.as_tensor(vals[rng.integers(0, len(vals), size=(n, 16))])


def test_output_shapes_all_input_forms():
    net = OneHotValueNet()
    b = _boards(5)
    assert net(b).shape == (5,)
    assert net(b.reshape(5, 4, 4)).shape == (5,)
    assert net(b[0]).shape == (1,)
    assert net(b).dtype == torch.float32
    assert net.forward_features(b).shape == (5, net.trunk_features)


def test_gradients_reach_every_parameter():
    net = OneHotValueNet()
    net(_boards(16)).sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    # the head's last layer gets gradient; trunk convs must too
    assert any(p.grad.abs().sum() > 0 for p in net.trunk.conv_h.parameters())


def test_deterministic_for_fixed_seed():
    torch.manual_seed(123)
    a = OneHotValueNet()
    torch.manual_seed(123)
    b = OneHotValueNet()
    x = _boards(8)
    assert torch.equal(a(x), b(x))


def test_initial_outputs_small():
    torch.manual_seed(0)
    net = OneHotValueNet()
    with torch.no_grad():
        v = net(_boards(256))
    assert v.abs().max() < 1.0


def test_state_dict_roundtrip():
    torch.manual_seed(1)
    a = OneHotValueNet(channels=32, hidden=64)
    torch.manual_seed(2)
    b = OneHotValueNet(channels=32, hidden=64)
    x = _boards(8)
    assert not torch.equal(a(x), b(x))
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(x), b(x))
    assert OneHotValueNet.architecture == "onehot_afterstate_v1"


def test_dueling_state_dict_keys_unchanged():
    keys = list(OneHotDuelingDQN().state_dict().keys())
    expected_prefixes = ["conv_h", "conv_v", "conv_hh", "conv_hv", "conv_vh", "conv_vv",
                         "head.value_stream.0", "head.value_stream.2",
                         "head.advantage_stream.0", "head.advantage_stream.2"]
    assert keys == [f"{p}.{s}" for p in expected_prefixes for s in ("weight", "bias")]
    # value net trunk keys are a subset of the Q net's trunk keys
    vkeys = [k[len("trunk."):] for k in OneHotValueNet().state_dict() if k.startswith("trunk.")]
    assert vkeys == [k for k in keys if not k.startswith("head.")]


def test_param_count_logged(capsys):
    net = OneHotValueNet()
    n = sum(p.numel() for p in net.parameters())
    q = sum(p.numel() for p in OneHotDuelingDQN().parameters())
    with capsys.disabled():
        print(f"\nOneHotValueNet params={n} OneHotDuelingDQN params={q}")
    assert 0 < n < q


def test_large_batch_forward():
    net = OneHotValueNet()
    with torch.no_grad():
        v = net(_boards(4 * 128))
    assert v.shape == (512,) and torch.isfinite(v).all()


def test_conv2x2_extension_point_and_unknown_trunk():
    with pytest.raises(NotImplementedError):
        OneHotValueNet(trunk="conv2x2")
    with pytest.raises(ValueError):
        OneHotValueNet(trunk="bogus")
