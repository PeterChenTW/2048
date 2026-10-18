"""V2 獨立審查的對抗測試：TD 定義、fallback、checkpoint、bench variant。"""
import numpy as np
import pytest
import torch
from torch import nn

import afterstate
import tools.bench_afterstate_expand as bench
from afterstate import batch_afterstates
from afterstate_agent import AfterstateAgent
from afterstate_policy import afterstate_scores
from config import DQNConfig
from encoding import log2_onehot_torch
from RL import load_checkpoint
from value_net import OneHotValueNet


# 每個動作恰有一個合法方向，用來檢查 compact index 不是永遠選 0。
SINGLE_LEGAL = np.array([
    [0, 0, 16, 128, 0, 0, 4, 32, 0, 0, 128, 8, 0, 0, 2, 4],
    [16, 32, 4, 2, 4, 2, 8, 16, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 64, 4, 64],
], dtype=np.int64)

# 2**16 / 2**17 會讓 vectorized 整批 fallback；中間夾一個普通盤。
FALLBACK = np.array([
    [2 ** 16, 2 ** 8, 2 ** 4, 2 ** 2] + [0] * 12,
    [2, 2, 0, 4] + [0] * 12,
    [2 ** 17, 2 ** 3, 0, 0] + [0] * 12,
    [2 ** 16, 0, 2 ** 17, 4] + [0] * 12,
], dtype=np.int64)


def _manual_targets(agent, next_states, dones):
    """預設路徑的 TD 定義：online 對 r+γV 取 argmax，target 評估同一個 afterstate。"""
    with torch.no_grad():
        targets = torch.zeros(len(next_states), dtype=torch.float32)
        active = torch.nonzero(~dones, as_tuple=True)[0]
        if not len(active):
            return targets
        result = afterstate_scores(agent.model, next_states[active], **agent._reward_options())
        rows = np.flatnonzero(result.legal.any(axis=1))
        if len(rows):
            choices = result.scores[rows].argmax(dim=1).numpy()
            chosen = torch.as_tensor(result.after[rows, choices], dtype=torch.float32)
            values = agent.target_model(chosen).reshape(-1)
            rewards = torch.as_tensor(result.rewards[rows, choices], dtype=torch.float32)
            targets[active[rows]] = rewards + agent.config.gamma * values
        return targets


def _small_agent(**options):
    defaults = dict(arch="onehot_afterstate", seed=21, memory_capacity=32, batch_size=8)
    defaults.update(options)
    config = DQNConfig(**defaults)
    agent = AfterstateAgent(16, 4, config)
    torch.manual_seed(21)
    agent.model = OneHotValueNet(channels=8, hidden=16)
    agent.target_model = OneHotValueNet(channels=8, hidden=16)
    agent.target_model.load_state_dict(agent.model.state_dict())
    with torch.no_grad():
        agent.target_model.head[-1].bias.add_(0.2)
    agent.optimizer = torch.optim.RMSprop(agent.model.parameters(), lr=config.lr)
    return agent


def test_default_fast_match_formula_on_edges():
    agent = _small_agent()
    assert agent.config.td_forward == "reference" and agent.config.td_inference_mode is False
    pieces = [SINGLE_LEGAL, FALLBACK, np.zeros((1, 16), dtype=np.int64), np.full((1, 16), 2, dtype=np.int64)]
    states = torch.as_tensor(np.concatenate(pieces), dtype=torch.float32)
    dones = torch.zeros(len(states), dtype=torch.bool)
    dones[-1] = True  # 全 2 盤標成 done，目標必須是 0，且不影響其他列
    for expand in ("reference", "vectorized"):
        for mode in ("log", "linear"):
            agent.config.afterstate_expand = expand
            agent.config.reward_mode = mode
            agent.config.reward_scale = 3.0
            for inference in (False, True):
                agent.config.td_inference_mode = inference
                manual = _manual_targets(agent, states, dones)
                for forward in ("reference", "fast"):
                    agent.config.td_forward = forward
                    got = agent._targets(states, dones)
                    assert torch.equal(got, manual)
                    assert not got.requires_grad and not torch.is_inference(got)
                    assert not torch.is_inference_mode_enabled()
                    q = torch.ones_like(got, requires_grad=True)
                    (q * got).sum().backward()
                    torch.testing.assert_close(q.grad, got)


def test_all_done_batch_is_zero_and_skips_forward():
    agent = _small_agent(td_forward="fast", td_inference_mode=True)
    states = torch.as_tensor(np.concatenate([FALLBACK, SINGLE_LEGAL]), dtype=torch.float32)
    dones = torch.ones(len(states), dtype=torch.bool)
    calls = []
    hook = lambda *args, **kwargs: calls.append(1)
    handles = [net.register_forward_pre_hook(hook) for net in (agent.model, agent.target_model)]
    try:
        for forward in ("reference", "fast"):
            agent.config.td_forward = forward
            for inference in (False, True):
                agent.config.td_inference_mode = inference
                calls.clear()
                got = agent._targets(states, dones)
                assert calls == []
                assert torch.equal(got, torch.zeros(len(states)))
                assert not torch.is_inference(got)
    finally:
        for handle in handles:
            handle.remove()


def test_single_legal_action_target_is_that_afterstate():
    agent = _small_agent(td_forward="fast")
    states = torch.as_tensor(SINGLE_LEGAL, dtype=torch.float32)
    dones = torch.zeros(len(states), dtype=torch.bool)
    after, _, legal = batch_afterstates(SINGLE_LEGAL)
    assert np.all(legal.sum(axis=1) == 1)
    only = np.flatnonzero(legal[0])  # 形狀檢查用；逐列在下面比
    assert only.size == 1

    def capture(forward):
        seen = []

        def hook(module, args, kwargs):
            seen.append((args[0].detach().cpu().clone(), dict(kwargs)))

        handles = [net.register_forward_pre_hook(hook, with_kwargs=True)
                   for net in (agent.model, agent.target_model)]
        agent.config.td_forward = forward
        try:
            targets = agent._targets(states, dones)
        finally:
            for handle in handles:
                handle.remove()
        return targets, seen

    manual = _manual_targets(agent, states, dones)
    for forward in ("reference", "fast"):
        targets, seen = capture(forward)
        assert torch.equal(targets, manual)
        assert len(seen) == 2  # online 一批合法盤、target 一批選中盤
        online, target = seen
        assert torch.equal(online[0], target[0])
        chosen = after[np.arange(len(after)), legal.argmax(axis=1)]
        if forward == "reference":
            assert target[1] == {}
            assert torch.equal(target[0], torch.as_tensor(chosen, dtype=torch.float32))
        else:
            assert target[1] == {"encoded": True}
            assert torch.equal(target[0], log2_onehot_torch(torch.from_numpy(chosen)))


def test_tied_argmax_uses_lowest_legal_index():
    """動作 0 非法、1 與 2 獎勵同為 0 時，必須選 index 1 的 afterstate。"""
    agent = _small_agent(td_forward="fast")
    with torch.no_grad():
        for parameter in list(agent.model.parameters()) + list(agent.target_model.parameters()):
            parameter.zero_()
    board = np.array([2, 0, 0, 0] + [0] * 12, dtype=np.int64)
    after, _, legal = batch_afterstates(board.reshape(1, 16))
    assert legal[0].tolist() == [False, True, True, False]
    assert not np.array_equal(after[0, 1], after[0, 2])
    states = torch.as_tensor(board.reshape(1, 16), dtype=torch.float32)
    dones = torch.zeros(1, dtype=torch.bool)

    def chosen_input(forward):
        seen = []

        def hook(module, args, kwargs):
            seen.append(args[0].detach().cpu().clone())

        handles = [net.register_forward_pre_hook(hook, with_kwargs=True)
                   for net in (agent.model, agent.target_model)]
        agent.config.td_forward = forward
        try:
            agent._targets(states, dones)
        finally:
            for handle in handles:
                handle.remove()
        return seen[-1]

    raw = torch.as_tensor(after[0, 1].reshape(1, 16), dtype=torch.float32)
    onehot = log2_onehot_torch(torch.from_numpy(after[0, 1].reshape(1, 16)))
    wrong = log2_onehot_torch(torch.from_numpy(after[0, 2].reshape(1, 16)))
    assert torch.equal(chosen_input("reference"), raw)
    fast_input = chosen_input("fast")
    assert torch.equal(fast_input, onehot)
    assert not torch.equal(fast_input, wrong)


def test_vectorized_fallback_matches_reference_formula():
    agent = _small_agent(td_forward="fast", afterstate_expand="vectorized")
    states = torch.as_tensor(FALLBACK, dtype=torch.float32)
    dones = torch.tensor([False, False, True, False])
    calls = {"n": 0}
    original = afterstate.batch_afterstates_reference

    def counted(grids):
        calls["n"] += 1
        return original(grids)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(afterstate, "batch_afterstates_reference", counted)
    try:
        for inference in (False, True):
            agent.config.td_inference_mode = inference
            manual = _manual_targets(agent, states, dones)
            for forward in ("fast", "reference"):
                calls["n"] = 0
                agent.config.td_forward = forward
                got = agent._targets(states, dones)
                assert calls["n"] == 1  # 整批含 2**16/2**17，vectorized 必須 fallback 一次
                assert torch.equal(got, manual)
                assert torch.equal(got[dones], torch.zeros(int(dones.sum())))
                assert torch.any(got[~dones] != 0)
    finally:
        monkeypatch.undo()


class _Lookup(nn.Module):
    def __init__(self, boards, values):
        super().__init__()
        self.table = {tuple(int(v) for v in board): value for board, value in zip(boards, values)}
        self.offset = nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, states):
        self.seen.append(states.detach().cpu().clone())
        values = [self.table.get(tuple(int(v) for v in row), 0.0) for row in states.detach().tolist()]
        return torch.tensor(values, dtype=torch.float32) + self.offset


def test_fast_double_td_selects_online_and_evaluates_that_afterstate():
    from afterstate import afterstates

    grid = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    terminal = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    boards, _, _ = afterstates(grid)
    agent = _small_agent(td_forward="fast", gamma=0.5)
    agent.model = _Lookup(boards, [1.0, 1.1, 0.0, 1000.0])
    agent.target_model = _Lookup(boards, [3.0, 7.0, 100.0, 1000.0])
    states = torch.as_tensor(np.stack([grid, terminal, grid]), dtype=torch.float32).reshape(-1, 16)
    dones = torch.tensor([False, False, True])
    for inference in (False, True):
        agent.model.seen.clear()
        agent.target_model.seen.clear()
        agent.config.td_inference_mode = inference
        got = agent._targets(states, dones)
        reward = float(np.float32(np.log2(1 + 4) / 16))
        assert got[1].item() == 0.0 and got[2].item() == 0.0
        assert got[0].item() == pytest.approx(reward + 0.5 * 3.0, abs=1e-6, rel=0)
        assert len(agent.model.seen) == 1 and len(agent.model.seen[0]) == 3
        assert len(agent.target_model.seen) == 1
        chosen = agent.target_model.seen[0]
        expected = torch.as_tensor(boards[0].reshape(1, 16), dtype=torch.float32)
        assert torch.equal(chosen, expected)
        assert not torch.equal(chosen, torch.as_tensor(boards[2].reshape(1, 16), dtype=torch.float32))
        q = got.detach().clone().requires_grad_(True)
        (q * got).sum().backward()
        assert agent.target_model.offset.grad is None
        assert not torch.is_inference(got)


def test_replay_target_sync_weights_and_priorities_match():
    rng = np.random.default_rng(6)
    boards = []
    for _ in range(24):
        board = np.zeros(16, dtype=np.int64)
        index = rng.choice(16, size=5, replace=False)
        board[index] = rng.choice([2, 4, 8, 16, 2 ** 15, 2 ** 16, 2 ** 17], size=5)
        boards.append(board)

    def play(inference):
        agents = [
            _small_agent(td_forward=forward, td_inference_mode=inference,
                         afterstate_expand="vectorized", track_replay_age=True,
                         memory_capacity=24, batch_size=8, seed=21)
            for forward in ("reference", "fast")
        ]
        for agent in agents:
            for step, board in enumerate(boards):
                agent.set_env_step(step)
                agent.remember(board, board, step % 6 == 0)
        for step in range(4):
            stats = [agent.replay() for agent in agents]
            assert not torch.is_inference_mode_enabled()
            for key, value in stats[0].items():
                assert stats[1][key] == value
            for left, right in zip(agents[0].model.parameters(), agents[1].model.parameters()):
                assert torch.equal(left, right)
            for left, right in zip(agents[0].target_model.parameters(), agents[1].target_model.parameters()):
                assert torch.equal(left, right)
            assert np.array_equal(agents[0].memory.tree, agents[1].memory.tree)
            assert agents[0].rng.bit_generator.state == agents[1].rng.bit_generator.state
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in agents[1].model.parameters())
            assert all(p.grad is None for p in agents[1].target_model.parameters())
            if step % 2 == 1:
                for agent in agents:
                    agent.update_target_model()
                for left, right in zip(agents[0].target_model.parameters(),
                                       agents[1].target_model.parameters()):
                    assert torch.equal(left, right)
        return agents

    for inference in (False, True):
        play(inference)


def test_encoded_entry_and_checkpoint_names_shapes(tmp_path):
    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", seed=4,
                                          td_forward="fast", td_inference_mode=True))
    keys = list(agent.model.state_dict())
    shapes = {key: tuple(value.shape) for key, value in agent.model.state_dict().items()}
    assert keys and all("encoded" not in key for key in keys)
    raw = torch.tensor([
        [2, 0, 0, 0] + [0] * 12,
        [2 ** 16, 2 ** 8, 0, 0] + [0] * 12,
        [2 ** 17, 2, 2, 0] + [0] * 12,
    ], dtype=torch.float32)
    with torch.no_grad():
        plain = agent.model(raw)
        encoded = agent.model(log2_onehot_torch(raw.to(torch.int64)), encoded=True)
    assert torch.equal(plain, encoded)
    assert list(agent.model.state_dict()) == keys
    assert {key: tuple(value.shape) for key, value in agent.model.state_dict().items()} == shapes

    path = tmp_path / "agent.pt"
    agent.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["config"]["td_forward"] == "fast"
    assert payload["config"]["td_inference_mode"] is True
    assert set(payload["state_dict"]) == set(keys)
    for key, value in payload["state_dict"].items():
        assert tuple(value.shape) == shapes[key]
    del payload["config"]["td_inference_mode"], payload["config"]["td_forward"]
    torch.save(payload, path)
    agent.config.td_forward = "fast"
    agent.config.td_inference_mode = True
    agent.load(path)
    assert agent.config.td_forward == "reference" and agent.config.td_inference_mode is False
    loaded = load_checkpoint(path)
    assert loaded.config.td_forward == "reference" and loaded.config.td_inference_mode is False
    assert list(loaded.model.state_dict()) == keys
    for key, value in loaded.model.state_dict().items():
        assert tuple(value.shape) == shapes[key]
        assert torch.equal(value, agent.model.state_dict()[key])


class _FakeAgent:
    def __init__(self, state_size, action_size, config=None):
        self.config = config
        self.memory = type("Memory", (), {"size": 0})()


def _install_bench_fakes(monkeypatch):
    threads = {"n": None}
    seen = []

    def set_threads(count):
        threads["n"] = count

    def fill(agent, mode, target_capacity, seed=None):
        seen.append({
            "expand": agent.config.afterstate_expand,
            "mode": mode,
            "tdinf": agent.config.td_inference_mode,
            "tdfwd": agent.config.td_forward,
            "threads": threads["n"],
        })
        return 0

    def measure_b(agent, seconds):
        seen.append({
            "measure": "B",
            "expand": agent.config.afterstate_expand,
            "tdinf": agent.config.td_inference_mode,
            "tdfwd": agent.config.td_forward,
            "threads": threads["n"],
        })
        return {"updates": 1, "elapsed_sec": 0.01, "updates_per_sec": 1.0}

    monkeypatch.setattr(bench, "AfterstateAgent", _FakeAgent)
    monkeypatch.setattr(bench.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(bench.torch, "set_num_interop_threads",
                        lambda count: (_ for _ in ()).throw(AssertionError("預設不該設定 interop")))
    monkeypatch.setattr(bench, "fill_replay_buffer", fill)
    monkeypatch.setattr(bench, "bench_measure_b", measure_b)
    return seen


def test_bench_variants_apply_to_agent_and_abba_order(monkeypatch):
    seen = _install_bench_fakes(monkeypatch)
    args = [
        "--variant", "base=expand:reference,threads:1,tdinf:0,tdfwd:reference",
        "--variant", "fast=expand:vectorized,threads:2,tdinf:1,tdfwd:fast",
        "--rounds", "4",
        "--order", "abba",
        "--measures", "B",
        "--seconds", "0.01",
        "--seed", "3",
        "--threads", "1",
    ]
    assert bench.main(args) == 0
    fills = [item for item in seen if "measure" not in item]
    measures = [item for item in seen if item.get("measure") == "B"]
    base = {"expand": "reference", "mode": "reference", "tdinf": False, "tdfwd": "reference", "threads": 1}
    fast = {"expand": "vectorized", "mode": "vectorized", "tdinf": True, "tdfwd": "fast", "threads": 2}
    assert fills == [base, fast, fast, base, base, fast, fast, base]
    assert [(item["threads"], item["expand"], item["tdinf"], item["tdfwd"]) for item in measures] == [
        (1, "reference", False, "reference"),
        (2, "vectorized", True, "fast"),
        (2, "vectorized", True, "fast"),
        (1, "reference", False, "reference"),
        (1, "reference", False, "reference"),
        (2, "vectorized", True, "fast"),
        (2, "vectorized", True, "fast"),
        (1, "reference", False, "reference"),
    ]

    seen.clear()
    assert bench.main([
        "--variant", "base=expand:reference,threads:1,tdinf:false,tdfwd:reference",
        "--variant", "fast=expand:vectorized,threads:2,tdinf:true,tdfwd:fast",
        "--rounds", "2",
        "--order", "abab",
        "--measures", "B",
        "--seconds", "0.01",
        "--threads", "1",
    ]) == 0
    fills = [item for item in seen if "measure" not in item]
    assert fills == [base, fast, base, fast]

    seen.clear()
    assert bench.main([
        "--variant", "only=expand:vectorized,threads:3",
        "--rounds", "1",
        "--measures", "B",
        "--seconds", "0.01",
    ]) == 0
    fills = [item for item in seen if "measure" not in item]
    assert fills == [{
        "expand": "vectorized", "mode": "vectorized",
        "tdinf": False, "tdfwd": "reference", "threads": 3,
    }]


def test_bench_interop_is_set_once_before_agents_or_errors(monkeypatch):
    events = []

    def set_interop(count):
        events.append(("interop", count))

    def set_threads(count):
        events.append(("intra", count))

    monkeypatch.setattr(bench, "AfterstateAgent", _FakeAgent)
    monkeypatch.setattr(bench.torch, "set_num_interop_threads", set_interop)
    monkeypatch.setattr(bench.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(bench, "fill_replay_buffer", lambda *args, **kwargs: events.append(("fill",)) or 0)
    monkeypatch.setattr(bench, "bench_measure_b", lambda *args, **kwargs: {
        "updates": 1, "elapsed_sec": 0.01, "updates_per_sec": 1.0})
    assert bench.main([
        "--interop-threads", "1",
        "--expand", "reference",
        "--measures", "B",
        "--rounds", "1",
        "--seconds", "0.01",
        "--threads", "2",
    ]) == 0
    assert events[0] == ("interop", 1)
    assert ("intra", 2) in events
    assert events.index(("interop", 1)) < events.index(("fill",))

    def boom(count):
        raise RuntimeError("cannot set number of interop threads after parallel work has started")

    monkeypatch.setattr(bench.torch, "set_num_interop_threads", boom)
    assert bench.main([
        "--interop-threads", "1",
        "--expand", "reference",
        "--measures", "B",
        "--seconds", "0.01",
    ]) == 1
    with pytest.raises(SystemExit):
        bench.main(["--interop-threads", "0", "--expand", "reference", "--seconds", "0.01"])
