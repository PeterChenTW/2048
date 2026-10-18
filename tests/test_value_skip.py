"""S preregistration: L1 skip correctness and bitwise compatibility with 33ae2ba."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch
import torch.nn.functional as F

from afterstate_agent import AfterstateAgent
from config import DQNConfig
from encoding import log2_onehot_torch
from eval import load_agent
import main
from RL import DDQNAgent, load_checkpoint
from value_net import OneHotValueNet

REPO = Path(__file__).resolve().parents[1]
BOARDS = torch.tensor([
    [2, 2, 4, 0, 4, 8, 0, 0, 16, 0, 32, 0, 0, 0, 0, 0],
    [0, 2, 0, 4, 8, 8, 16, 0, 32, 64, 0, 0, 128, 256, 512, 1024],
    [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2],
], dtype=torch.float32)


def assert_equal_tree(actual, expected, path="root"):
    """Use exact equality throughout, including tensors nested in optimizer state."""
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), path
        for key in expected:
            assert_equal_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equal_tree(a, b, f"{path}[{i}]")
    else:
        assert actual == expected, path


@pytest.mark.parametrize("skip,features,params,tag", [
    ("none", 2176, 631_809, "onehot_afterstate_v1"),
    ("l1", 5248, 1_418_241, "onehot_afterstate_skip_v1"),
])
def test_shapes_gradients_and_preregistered_parameter_counts(skip, features, params, tag):
    torch.manual_seed(101)
    net = OneHotValueNet(value_skip=skip)
    assert net.architecture == tag
    assert net.trunk_features == net.head[0].in_features == features
    assert net.head[0].out_features == net.head[2].in_features == 256
    assert net.head[2].out_features == 1
    assert torch.equal(net.head[2].bias, torch.zeros(1))
    assert sum(p.numel() for p in net.parameters()) == params
    for boards, size in [(BOARDS[0], 1), (BOARDS, 3), (BOARDS.reshape(3, 4, 4), 3)]:
        features_out = net.forward_features(boards)
        out = net(boards)
        assert features_out.shape == (size, features)
        assert out.shape == (size,) and out.dtype == torch.float32
        assert torch.isfinite(features_out).all() and torch.isfinite(out).all()
    net(BOARDS).square().sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert net.trunk.conv_h.weight.grad.abs().sum() > 0
    assert net.trunk.conv_v.weight.grad.abs().sum() > 0


def test_skip_features_are_post_relu_and_in_preregistered_order():
    net = OneHotValueNet(value_skip="l1")
    trunk = net.trunk
    onehot = log2_onehot_torch(BOARDS)
    h = F.relu(trunk.conv_h(onehot))
    v = F.relu(trunk.conv_v(onehot))
    expected = torch.cat([
        F.relu(trunk.conv_hh(h)).flatten(1),
        F.relu(trunk.conv_hv(h)).flatten(1),
        F.relu(trunk.conv_vh(v)).flatten(1),
        F.relu(trunk.conv_vv(v)).flatten(1),
        h.flatten(1), v.flatten(1),
    ], dim=1)
    assert torch.equal(net.forward_features(BOARDS), expected)
    assert (expected >= 0).all()
    # Both arms retain exactly the same convolutional modules and initial weights.
    torch.manual_seed(202)
    control = OneHotValueNet()
    torch.manual_seed(202)
    skip = OneHotValueNet(value_skip="l1")
    assert_equal_tree(skip.trunk.state_dict(), control.trunk.state_dict())
    assert list(skip.state_dict()) == list(control.state_dict())
    assert sum(p.numel() for p in control.trunk.parameters()) == 74_240


@pytest.mark.parametrize("skip", ["none", "l1"])
def test_checkpoint_roundtrip_all_loaders_and_finite_replay(tmp_path, skip):
    agent = AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip=skip, seed=303,
        batch_size=2, memory_capacity=8, per_max_priority="tree"))
    for i in range(6):
        agent.remember(BOARDS[i % 3], BOARDS[(i + 1) % 3], i % 3 == 1)
    stats = agent.replay()
    assert not stats["nan"]
    assert all(torch.isfinite(p).all() for p in agent.model.parameters())
    expected = agent.model(BOARDS).detach()
    checkpoint = tmp_path / f"{skip}.pt"
    agent.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["config"]["value_skip"] == skip
    assert payload["architecture"] == agent.model.architecture
    direct = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", value_skip=skip))
    direct.load(checkpoint)
    for loaded in [direct, load_checkpoint(checkpoint), load_agent(checkpoint)]:
        assert isinstance(loaded, AfterstateAgent)
        assert loaded.config.value_skip == loaded.model.value_skip == skip
        assert_equal_tree(loaded.model.state_dict(), agent.model.state_dict())
        assert_equal_tree(loaded.target_model.state_dict(), agent.model.state_dict())
        assert torch.equal(loaded.model(BOARDS), expected)


@pytest.mark.parametrize("tag,config_arch,skip", [
    ("onehot_afterstate_skip_v1", "onehot_afterstate", "none"),
    ("onehot_afterstate_skip_v1", "onehot_afterstate", None),
    ("onehot_afterstate_v1", "onehot_afterstate", "l1"),
    ("onehot_afterstate_v1", "onehot_afterstate", "bogus"),
    ("onehot_dueling_v1", "onehot_dueling", "l1"),
    ("onehot_conv2x2_v1", "onehot_conv2x2", "l1"),
    ("legacy", "legacy", "l1"),
    ("onehot_afterstate_skip_v1", "onehot_dueling", "l1"),
])
def test_checkpoint_architecture_config_conflicts_rejected(tmp_path, tag, config_arch, skip):
    config = {"arch": config_arch}
    if skip is not None:
        config["value_skip"] = skip
    path = tmp_path / "conflicting.pt"
    torch.save(dict(format="dqn2048-v2", architecture=tag,
                    config=config, state_dict={}), path)
    direct = AfterstateAgent(16, 4)
    for loader in [direct.load, load_checkpoint, load_agent]:
        with pytest.raises(ValueError, match="architecture conflicts with its config"):
            loader(path)


@pytest.mark.parametrize("arch", ["legacy", "onehot_dueling", "onehot_conv2x2"])
def test_l1_rejected_for_other_architectures(arch):
    with pytest.raises(ValueError, match="requires arch=onehot_afterstate"):
        DDQNAgent(16, 4, DQNConfig(arch=arch, value_skip="l1"))
    with pytest.raises(ValueError, match="requires arch=onehot_afterstate"):
        main.main(["train", "--from-scratch", "--arch", arch, "--value-skip", "l1"])


def test_invalid_skip_values_rejected():
    with pytest.raises(ValueError, match="unknown value_skip"):
        OneHotValueNet(value_skip="bogus")
    with pytest.raises(ValueError, match="unknown value_skip"):
        AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", value_skip="bogus"))
    with pytest.raises(ValueError, match="requires trunk"):
        OneHotValueNet(trunk="conv2x2", value_skip="l1")
    with pytest.raises(SystemExit):
        main.parser().parse_args(["train", "--value-skip", "bogus"])


def test_cli_parse():
    assert DQNConfig().value_skip == "none"
    assert main.parser().parse_args(["train"]).value_skip == "none"
    for skip in ("none", "l1"):
        args = main.parser().parse_args([
            "train", "--arch", "onehot_afterstate", "--value-skip", skip])
        assert args.value_skip == skip and args.arch == "onehot_afterstate"


@pytest.mark.parametrize("saved_skip,requested_skip", [("l1", "none"), ("none", "l1")])
def test_resume_skip_mismatch_rejected_before_any_output(tmp_path, saved_skip, requested_skip):
    path = tmp_path / "checkpoint.pt"
    AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip=saved_skip)).save(path)
    output = tmp_path / "must-not-exist.pt"
    log = tmp_path / "must-not-exist.csv"
    with pytest.raises(ValueError, match="--value-skip"):
        main.main(["train", "--arch", "onehot_afterstate", "--value-skip", requested_skip,
                   "--checkpoint", str(path), "--minutes", "0", "--save-as", str(output),
                   "--log", str(log)])
    assert not output.exists() and not log.exists()
    direct = AfterstateAgent(16, 4, DQNConfig(
        arch="onehot_afterstate", value_skip=requested_skip))
    with pytest.raises(ValueError, match="value_skip"):
        direct.load(path)


@pytest.mark.parametrize("skip", ["none", "l1"])
def test_cli_zero_steps_summary_metadata_and_matching_resume(tmp_path, skip):
    # A zero-minute budget only constructs/saves the model; it runs no game or training.
    path = tmp_path / "checkpoint.pt"
    log = tmp_path / "zero.csv"
    common = ["train", "--arch", "onehot_afterstate", "--value-skip", skip,
              "--minutes", "0", "--save-as", str(path), "--log", str(log)]
    assert main.main([*common, "--from-scratch"]) == 0
    assert main.main([*common, "--checkpoint", str(path)]) == 0
    summary = json.loads(log.with_suffix(".json").read_text())
    assert summary["value_skip"] == skip
    assert summary["arch"] == "onehot_afterstate"
    assert summary["episodes"] == summary["env_steps"] == summary["updates"] == 0
    assert torch.load(path, weights_only=True)["config"]["value_skip"] == skip


# Run exactly this script against both repositories. No new value_skip argument is
# supplied: the new code must match the actual pre-change default at every step.
REGRESSION_SCRIPT = r"""
import copy
from pathlib import Path
import sys
import torch
from afterstate import afterstates
from afterstate_agent import AfterstateAgent
from config import DQNConfig

torch.set_num_threads(1)
torch.manual_seed(101)
out = Path(sys.argv[1])
mode = sys.argv[2]
boards = torch.tensor([
    [2, 2, 4, 0, 4, 8, 0, 0, 16, 0, 32, 0, 0, 0, 0, 0],
    [0, 2, 0, 4, 8, 8, 16, 0, 32, 64, 0, 0, 128, 256, 512, 1024],
    [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2],
], dtype=torch.float32)
agent = AfterstateAgent(16, 4, DQNConfig(
    arch="onehot_afterstate", seed=101, memory_capacity=16, batch_size=4,
    per_max_priority=mode))

def snapshot():
    with torch.no_grad():
        forward = agent.model(boards).clone()
    return dict(
        state_dict=copy.deepcopy(agent.model.state_dict()),
        target_state_dict=copy.deepcopy(agent.target_model.state_dict()),
        optimizer=copy.deepcopy(agent.optimizer.state_dict()),
        forward=forward,
        priorities=torch.tensor(agent.memory.tree.copy()),
        max_tree=None if agent.memory.max_tree is None else torch.tensor(agent.memory.max_tree.copy()),
        agent_rng=copy.deepcopy(agent.rng.bit_generator.state),
        torch_rng=torch.get_rng_state().clone(),
        epsilon=agent.epsilon, beta=agent.beta, replay_calls=agent.replay_calls)

snapshots = [snapshot()]
agent.save(out / "initial_checkpoint.pt")
for i in range(12):
    board = boards[i % 3]
    after, _, legal = afterstates(board)
    choices = legal.nonzero()[0]
    stored = after[int(choices[i % len(choices)])] if len(choices) else board
    agent.remember(stored, boards[(i + 1) % 3], i % 3 == 1)
for _ in range(5):
    stats = agent.replay()
    assert not stats["nan"]
    snapshots.append(snapshot())
agent.save(out / "trained_checkpoint.pt")
torch.save(dict(boards=boards, snapshots=snapshots), out / "snapshots.pt")
"""


@pytest.fixture(scope="module")
def archived_baseline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("value_skip_baseline_33ae2ba")
    repo = directory / "repo"
    repo.mkdir()
    archive = directory / "baseline.tar"
    with archive.open("wb") as stream:
        result = subprocess.run(
            ["nice", "-n", "19", "git", "-C", str(REPO), "archive", "33ae2ba"],
            stdout=stream, stderr=subprocess.PIPE,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    with tarfile.open(archive) as stream:
        stream.extractall(repo, filter="data")
    return repo


@pytest.fixture(scope="module", params=["scan", "tree"])
def regression_results(archived_baseline, tmp_path_factory, request):
    results = {}
    for label, repo in [("old", archived_baseline), ("new", REPO)]:
        output = tmp_path_factory.mktemp(f"skip_{label}_{request.param}")
        result = subprocess.run(
            ["nice", "-n", "19", sys.executable, "-c", REGRESSION_SCRIPT,
             str(output), request.param],
            cwd=repo, env={**os.environ, "PYTHONPATH": str(repo),
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        results[label] = (output, torch.load(output / "snapshots.pt", weights_only=True))
    return results


def test_default_matches_33ae2ba_bitwise_through_five_replays(regression_results):
    old = regression_results["old"][1]
    new = regression_results["new"][1]
    assert len(old["snapshots"]) == len(new["snapshots"]) == 6
    assert old["snapshots"][-1]["replay_calls"] == 5
    assert old["snapshots"][0]["agent_rng"] != old["snapshots"][-1]["agent_rng"]
    assert any(not torch.equal(value, old["snapshots"][-1]["state_dict"][key])
               for key, value in old["snapshots"][0]["state_dict"].items())
    assert_equal_tree(new, old)


@pytest.mark.parametrize("checkpoint,index", [("initial_checkpoint.pt", 0),
                                              ("trained_checkpoint.pt", -1)])
def test_real_33ae2ba_checkpoint_all_loaders_bitwise(regression_results, checkpoint, index):
    output, old = regression_results["old"]
    path = output / checkpoint
    payload = torch.load(path, weights_only=True)
    assert payload["architecture"] == "onehot_afterstate_v1"
    assert "value_skip" not in payload["config"]
    direct = AfterstateAgent(16, 4)
    direct.load(path)
    for loaded in [direct, load_checkpoint(path), load_agent(path)]:
        assert loaded.config.value_skip == "none"
        assert_equal_tree(loaded.model.state_dict(), old["snapshots"][index]["state_dict"])
        assert torch.equal(loaded.model(old["boards"]), old["snapshots"][index]["forward"])
