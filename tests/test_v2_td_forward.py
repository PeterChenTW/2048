"""V2 A1/A2/B: TD equivalence, ordinary targets, CLI and checkpoints."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

import afterstate_agent as agent_module
import afterstate_policy as policy_module
import dueling
import main
from afterstate import EXPAND_MODES, batch_afterstates_reference
from afterstate_agent import AfterstateAgent
from config import DQNConfig
from RL import DDQNAgent, load_checkpoint
from value_net import OneHotValueNet
from test_afterstate_vectorized import comparison_boards, EDGE_BOARDS  # noqa: F401


@pytest.fixture(scope="module")
def next_boards(comparison_boards):
    generated, game_count = comparison_boards
    assert game_count > 100
    selected = generated[np.linspace(0, len(generated) - 1, 2200, dtype=int)]
    edges = np.asarray([case.values[0] for case in EDGE_BOARDS], dtype=np.int64).reshape(-1, 16)
    return np.concatenate([selected, edges])


@pytest.mark.parametrize("expand", EXPAND_MODES)
@pytest.mark.parametrize("reward_mode", ["log", "linear"])
@pytest.mark.parametrize("forward", ["reference", "fast"])
def test_inference_td_and_online_argmax(next_boards, expand, reward_mode, forward, monkeypatch, record_property):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=2102,
                                          afterstate_expand=expand, reward_mode=reward_mode))
    # Use the real network with a smaller trunk to bound suite cost.
    torch.manual_seed(2102)
    agent.model = OneHotValueNet(channels=16, hidden=32).eval()
    agent.target_model = OneHotValueNet(channels=16, hidden=32).eval()
    agent.target_model.load_state_dict(agent.model.state_dict())
    with torch.no_grad():
        agent.target_model.head[-1].bias.add_(0.25)
    states = torch.as_tensor(next_boards, dtype=torch.float32)
    dones = torch.as_tensor(np.random.default_rng(2102).random(len(states)) < 0.25)
    dones[-len(EDGE_BOARDS):] = False
    dones[0] = True
    no_action = torch.as_tensor(~batch_afterstates_reference(next_boards)[2].any(axis=1))
    assert len(states) >= 2000 and states.max() >= 2**15
    assert dones.any() and (no_action & ~dones).any()
    choices = []
    forward_modes = []
    original_scores = agent_module.afterstate_scores

    def scores(*args, **kwargs):
        result = original_scores(*args, **kwargs)
        choices.append(result.scores.argmax(dim=1).numpy().copy())
        return result

    def check_forward(module, inputs):
        forward_modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

    monkeypatch.setattr(agent_module, "afterstate_scores", scores)
    original_fast = agent_module._td_afterstate_scores

    def fast_scores(*args, **kwargs):
        result = original_fast(*args, **kwargs)
        choices.append(result.scores.argmax(dim=1).numpy().copy())
        return result

    monkeypatch.setattr(agent_module, "_td_afterstate_scores", fast_scores)
    hooks = [net.register_forward_pre_hook(check_forward) for net in (agent.model, agent.target_model)]
    results, actions = [], []
    try:
        for backend, enabled in (("reference", False), (forward, False), (forward, True)):
            agent.config.td_forward = backend
            agent.config.td_inference_mode = enabled
            choices.clear()
            forward_modes.clear()
            targets = torch.cat([agent._targets(states[i:i + 128], dones[i:i + 128])
                                 for i in range(0, len(states), 128)])
            assert forward_modes and all(mode == (False, enabled) for mode in forward_modes)
            assert not targets.requires_grad and not torch.is_inference(targets)
            assert torch.isfinite(targets).all()
            assert torch.all(targets[dones | no_action] == 0)
            assert torch.any(targets[~(dones | no_action)] != 0)
            results.append(targets)
            actions.append(np.concatenate(choices))
    finally:
        for hook in hooks:
            hook.remove()
    error = max(float((results[0] - result).abs().max()) for result in results[1:])
    consistency = min(float(np.mean(actions[0] == action)) for action in actions[1:])
    assert error <= 1e-5
    assert consistency == 1.0
    print(f"V2 {forward}/{expand}/{reward_mode}, inference=off/on: next_states={len(states)}, "
          f"TD max_error={error:.9g}, online argmax={consistency:.2%}")
    record_property("td_target_max_error", error)
    record_property("online_argmax_consistency", consistency)
    record_property("next_state_count", len(states))


@pytest.mark.parametrize("expand", EXPAND_MODES)
def test_inference_ties_choose_first_index(expand, monkeypatch):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=4,
                                          afterstate_expand=expand))
    with torch.no_grad():
        for parameter in agent.model.parameters():
            parameter.zero_()
    states = torch.full((4, 16), 2.0)  # All four actions have equal reward and value.
    states[1].zero_()
    states[1, 0] = 2  # Index 0 illegal; tied valid actions 1 and 2 must choose 1.
    original = agent_module.afterstate_scores
    actions = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        actions.append(result.scores.argmax(dim=1).tolist())
        return result

    monkeypatch.setattr(agent_module, "afterstate_scores", capture)
    original_fast = agent_module._td_afterstate_scores

    def capture_fast(*args, **kwargs):
        result = original_fast(*args, **kwargs)
        actions.append(result.scores.argmax(dim=1).tolist())
        return result

    monkeypatch.setattr(agent_module, "_td_afterstate_scores", capture_fast)
    for forward in ("reference", "fast"):
        agent.config.td_forward = forward
        for enabled in (False, True):
            agent.config.td_inference_mode = enabled
            agent._targets(states, torch.zeros(4, dtype=torch.bool))
    assert actions == [[0, 1, 0, 0]] * 4


@pytest.mark.parametrize("expand", EXPAND_MODES)
@pytest.mark.parametrize("forward", ["reference", "fast"])
@pytest.mark.parametrize("enabled", [False, True])
def test_full_replay_stats_backward_and_weights(next_boards, expand, forward, enabled):
    agents = [AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=73,
                                            afterstate_expand=expand, batch_size=16,
                                            memory_capacity=32, td_inference_mode=inference,
                                            td_forward=backend,
                                            track_replay_age=True)) for backend, inference in
              (("reference", False), (forward, enabled))]
    for agent in agents:
        for i, board in enumerate(next_boards[:32]):
            agent.set_env_step(i)
            agent.remember(board, board, i % 5 == 0)
    before = agents[1].model.head[-1].weight.detach().clone()
    for _ in range(2):
        stats = [agent.replay() for agent in agents]
        assert stats[0].keys() == stats[1].keys()
        for key in stats[0]:
            assert stats[0][key] == pytest.approx(stats[1][key], abs=1e-5, rel=0)
        for agent in agents:
            assert not agent.config.td_inference_mode or not torch.is_inference_mode_enabled()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.model.parameters())
            assert all(p.grad is None for p in agent.target_model.parameters())
        for first, second in zip(agents[0].model.parameters(), agents[1].model.parameters()):
            torch.testing.assert_close(first, second, atol=1e-5, rtol=0)
        np.testing.assert_allclose(agents[0].memory.tree, agents[1].memory.tree, atol=1e-5, rtol=0)
        assert agents[0].rng.bit_generator.state == agents[1].rng.bit_generator.state
    assert not torch.equal(before, agents[1].model.head[-1].weight)


@pytest.mark.parametrize("case", ["active", "all_done", "no_legal", "empty"])
@pytest.mark.parametrize("forward", ["reference", "fast"])
def test_targets_are_ordinary_tensors_for_autograd(case, forward):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=8,
                                          td_inference_mode=True, td_forward=forward))
    states = torch.full((2, 16), 2.0)
    dones = torch.zeros(2, dtype=torch.bool)
    if case == "all_done":
        dones.fill_(True)
    elif case == "no_legal":
        states.zero_()
    elif case == "empty":
        states, dones = states[:0], dones[:0]
    targets = agent._targets(states, dones)
    assert not torch.is_inference(targets) and not targets.requires_grad
    q = torch.ones_like(targets, requires_grad=True)
    # Multiplication saves targets for backward and catches escaped inference tensors.
    (q * targets).sum().backward()
    torch.testing.assert_close(q.grad, targets)
    ((q - targets).square().sum()).backward()


def test_cli_defaults_and_flags():
    args = main.parser().parse_args(["train"])
    assert not args.td_inference_mode and args.td_forward == "reference"
    assert args.interop_threads is None
    assert DQNConfig().td_inference_mode is False and DQNConfig().td_forward == "reference"
    args = main.parser().parse_args(["train", "--arch", "onehot_afterstate", "--td-inference-mode",
                                     "--td-forward", "fast", "--interop-threads", "1"])
    assert args.td_inference_mode and args.td_forward == "fast" and args.interop_threads == 1
    for options in (["--td-forward", "unknown"], ["--interop-threads", "0"],
                    ["--interop-threads", "-1"]):
        with pytest.raises(SystemExit):
            main.parser().parse_args(["train", *options])


@pytest.mark.parametrize("arch", ["legacy", "onehot_dueling", "onehot_conv2x2"])
@pytest.mark.parametrize("options,fields", [(["--td-inference-mode"], {"td_inference_mode": True}),
                                          (["--td-forward", "fast"], {"td_forward": "fast"})])
def test_non_afterstate_rejects_td_options(arch, options, fields):
    with pytest.raises(ValueError, match="onehot_afterstate"):
        DDQNAgent(16, 4, DQNConfig(arch=arch, **fields))
    args = main.parser().parse_args(["train", "--arch", arch, *options])
    with pytest.raises(ValueError, match="onehot_afterstate"):
        main.train(args)


def test_unknown_backend_and_fast_empty():
    with pytest.raises(ValueError, match="td_forward"):
        AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", td_forward="unknown"))
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", td_forward="fast"))
    assert agent._targets(torch.zeros((1, 16)), torch.tensor([False])).tolist() == [0.0]


@pytest.mark.parametrize("enabled", [False, True])
def test_fast_reuses_encoding_and_only_forwards_legal_chosen(enabled, monkeypatch):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=9,
                                          td_forward="fast", td_inference_mode=enabled))
    states = torch.tensor([[2, 0, 0, 0] + [0] * 12, [0] * 16, [2] * 16,
                           [2, 4, 2, 4, 4, 2, 4, 2] * 2, [4] * 16], dtype=torch.float32)
    dones = torch.tensor([False, False, False, False, True])
    after, _, legal = batch_afterstates_reference(states[~dones].numpy())
    encodings, batches = [], []
    original = policy_module.log2_onehot_torch

    def encode(x):
        result = original(x)
        encodings.append((x.clone(), result.clone()))
        return result

    def unexpected_encoding(x):
        raise AssertionError("fast must not re-encode target boards")

    def hook(net, args, kwargs):
        assert kwargs == {"encoded": True}
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() == enabled
        batches.append(args[0].clone())

    monkeypatch.setattr(policy_module, "log2_onehot_torch", encode)
    monkeypatch.setattr(dueling, "log2_onehot_torch", unexpected_encoding)
    hooks = [net.register_forward_pre_hook(hook, with_kwargs=True)
             for net in (agent.model, agent.target_model)]
    try:
        agent._targets(states, dones)
    finally:
        for handle in hooks:
            handle.remove()
    assert len(encodings) == 1 and len(batches) == 2
    torch.testing.assert_close(encodings[0][0], torch.from_numpy(after[legal]))
    torch.testing.assert_close(batches[0], encodings[0][1])
    assert len(batches[0]) == int(legal.sum()) and len(batches[1]) == int(legal.any(axis=1).sum())
    for chosen in batches[1]:
        assert any(torch.equal(chosen, board) for board in batches[0])


def test_thread_setup_order_default_and_error(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(("intra", n)))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: calls.append(("interop", n)))
    main._configure_threads(main.parser().parse_args(["train"]))
    assert calls == [("intra", 1)]  # Default must never call the inter-op setter.
    calls.clear()
    main._configure_threads(main.parser().parse_args(["train", "--interop-threads", "1", "--threads", "2"]))
    assert calls == [("interop", 1), ("intra", 2)]

    def initialized(n):
        raise RuntimeError("cannot set after parallel work")

    monkeypatch.setattr(torch, "set_num_interop_threads", initialized)
    with pytest.raises(RuntimeError, match="once, before any PyTorch parallel work"):
        main._configure_threads(main.parser().parse_args(["train", "--interop-threads", "1"]))


def test_real_interop_setup_once_in_fresh_process():
    code = """
import torch
import main
args = main.parser().parse_args(['train', '--interop-threads', '1'])
main._configure_threads(args)
assert torch.get_num_interop_threads() == 1
try:
    main._configure_threads(args)
except RuntimeError as exc:
    assert '--interop-threads must be set once' in str(exc)
else:
    raise AssertionError('a second inter-op setter must fail clearly')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("forward", ["reference", "fast"])
def test_checkpoint_roundtrip_and_old_defaults(tmp_path, enabled, forward):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=9,
                                          td_inference_mode=enabled, td_forward=forward))
    path = tmp_path / "agent.pt"
    agent.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["config"]["td_inference_mode"] == enabled
    assert payload["config"]["td_forward"] == forward
    receiver = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=10))
    receiver.load(path)
    for loaded in (receiver, load_checkpoint(path)):
        assert loaded.config.td_inference_mode == enabled and loaded.config.td_forward == forward
        for key, value in agent.model.state_dict().items():
            assert torch.equal(value, loaded.model.state_dict()[key])
            assert torch.equal(value, loaded.target_model.state_dict()[key])
    del payload["config"]["td_inference_mode"], payload["config"]["td_forward"]
    torch.save(payload, path)
    receiver.config.td_inference_mode, receiver.config.td_forward = True, "fast"
    receiver.load(path)
    for loaded in (receiver, load_checkpoint(path)):
        assert loaded.config.td_inference_mode is False and loaded.config.td_forward == "reference"


@pytest.mark.parametrize("arch", ["legacy", "onehot_afterstate"])
@pytest.mark.parametrize("resume", [False, True])
def test_cli_config_and_summary_without_training(tmp_path, monkeypatch, arch, resume):
    checkpoint, output, log = (tmp_path / name for name in ("in.pt", "out.pt", "episodes.csv"))
    options = ["train", "--arch", arch, "--minutes", "0", "--save-as", str(output),
               "--log", str(log), "--interop-threads", "1"]
    if arch == "onehot_afterstate":
        options += ["--td-inference-mode", "--td-forward", "fast"]
    if resume:
        # Opposite saved settings prove that CLI overrides load() during setup.
        cls = AfterstateAgent if arch == "onehot_afterstate" else DDQNAgent
        cls(16, 4, DQNConfig(arch=arch)).save(checkpoint)
        options += ["--checkpoint", str(checkpoint)]
    else:
        options += ["--from-scratch"]
    calls = []
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: calls.append(n))
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    assert main.train(main.parser().parse_args(options)) == 0
    summary = json.loads(log.with_suffix(".json").read_text())
    assert calls == [1]
    assert summary["interop_threads"] == summary["interop_threads_requested"] == 1
    assert summary["threads"] == 1 and summary["env_steps"] == summary["updates"] == 0
    assert summary["td_inference_mode"] == (arch == "onehot_afterstate")
    assert summary["td_forward"] == ("fast" if arch == "onehot_afterstate" else "reference")
    loaded = load_checkpoint(output)
    assert loaded.config.td_inference_mode == summary["td_inference_mode"]
    assert loaded.config.td_forward == summary["td_forward"]
