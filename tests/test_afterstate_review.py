"""Independent correctness review tests for Experiment #1 (afterstate V, subtask 1e).

Covers all 8 required verification checks from the research plan:
1. Toy terminal target: s_{t+1} terminal -> target is 0, regardless of V nets.
2. Reward not double counted: target = r(s_{t+1}, a*) + gamma * V_target(after*)
   hand-computed on a fixed batch vs replay()'s internal target.
3. Double action selection: a* uses ONLINE net, value uses TARGET net (disagreeing stub nets).
4. Mask / no-op: illegal actions never selected in act / greedy_action / target even if
   V(after) is huge; afterstate of illegal action never enters the target.
5. RNG isolation: pure_move / afterstate computation never consumes board RNG during
   act, greedy_action, and replay.
6. Save/load/evaluator consistency: identical greedy actions before/after save/load;
   eval.evaluate deterministic across two runs and matches manual greedy_action rollout.
7. Gamma placement: V target uses config.gamma; changing gamma changes target as hand-computed.
8. Defaults untouched: DQNConfig() defaults unchanged, legacy and onehot_dueling checkpoints load.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afterstate import afterstates, batch_afterstates, is_terminal
from afterstate_agent import AfterstateAgent
from afterstate_policy import afterstate_scores, greedy_actions, merge_reward, select_action
from config import DQNConfig
from dueling import OneHotDuelingDQN
import eval as eval_mod
from fast_board import FastBoard
from RL import DDQNAgent, DQN, load_checkpoint
from value_net import OneHotValueNet


# Checkerboard terminal board: no zeros and no adjacent equal tiles.
TERMINAL_A = np.array(
    [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ],
    dtype=np.int64,
)

# Second distinct terminal board with larger tiles.
TERMINAL_B = np.array(
    [
        [8, 16, 32, 64],
        [128, 256, 512, 1024],
        [2, 4, 8, 16],
        [32, 64, 128, 256],
    ],
    dtype=np.int64,
)

# Non-terminal board where:
# - action 0 (left): merges [4,4,0,0] -> [8,0,0,0] (merge=8) and [2,2,0,0] -> [4,0,0,0] (merge=4), total merge=12
# - action 1 (down): merges col 0 [4,2,0,0]->no merge, slides down, total merge=0, legal=True
# - action 2 (right): merges row 0 (8) and row 1 (4), total merge=12, legal=True
# - action 3 (up): col 0 is [4,2,0,0], col 1 is [4,2,0,0], no slide/merge -> ILLEGAL!
BOARD_MIXED = np.array(
    [
        [4, 4, 0, 0],
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.int64,
)

# Non-terminal board where all 4 actions are legal and yield different merge scores:
# Row 0: [0, 2, 2, 0] -> left/right merge 2+2=4; up/down slide without merging if col has no match,
# Let's include a vertical merge too:
# [0, 2, 2, 0]
# [0, 0, 8, 0]
# [0, 0, 8, 0]
# [0, 0, 0, 0]
# - left (0): merges 2+2=4 (merge=4), legal=True
# - down (1): merges 8+8=16 (merge=16), legal=True
# - right (2): merges 2+2=4 (merge=4), legal=True
# - up (3): merges 8+8=16 (merge=16), legal=True
BOARD_ALL_LEGAL = np.array(
    [
        [0, 2, 2, 0],
        [0, 0, 8, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.int64,
)


class StubValueNet(nn.Module):
    """Stub V(afterstate) network with a board->value lookup table and call recording."""

    def __init__(self, table: dict[tuple[int, ...], float] | None = None, default: float = 0.0, forbidden: set[tuple[int, ...]] | None = None):
        super().__init__()
        self.table = {tuple(int(x) for x in k): float(v) for k, v in (table or {}).items()}
        self.default = float(default)
        self.forbidden = {tuple(int(x) for x in k) for k in (forbidden or set())}
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.queried_boards: list[tuple[int, ...]] = []
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        rows = x.detach().cpu().numpy().astype(np.int64).reshape(-1, 16)
        out = []
        for row in rows:
            key = tuple(row.tolist())
            if key in self.forbidden:
                raise AssertionError(f"Forbidden (illegal) afterstate entered network forward: {key}")
            self.queried_boards.append(key)
            out.append(self.table.get(key, self.default))
        return torch.tensor(out, dtype=torch.float32) * self.scale


def _make_stub_agent(online_net: nn.Module, target_net: nn.Module, **cfg_kwargs) -> AfterstateAgent:
    defaults = dict(arch="onehot_afterstate", memory_capacity=16, batch_size=2, gamma=0.99, lr=0.0, seed=42)
    defaults.update(cfg_kwargs)
    agent = AfterstateAgent(16, 4, DQNConfig(**defaults))
    agent.model = online_net
    agent.target_model = target_net
    agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=defaults["lr"])
    return agent


# ---------------------------------------------------------------------------
# Check 1: Toy terminal target: s_{t+1} terminal -> target exactly 0,
#          regardless of V nets.
# ---------------------------------------------------------------------------
def test_check1_terminal_target_is_strictly_zero_regardless_of_v_nets():
    assert is_terminal(TERMINAL_A)
    assert is_terminal(TERMINAL_B)

    for default_v in (0.0, 1e6, -1e6, 42.5):
        online = StubValueNet(default=default_v)
        target = StubValueNet(default=-default_v if default_v != 0 else 999.0)
        agent = _make_stub_agent(online, target, batch_size=4)

        # Case A: done=True on both terminal and non-terminal boards
        # Case B: done=False on terminal boards (structural finality must still yield 0)
        next_states = torch.as_tensor(
            np.stack([TERMINAL_A, TERMINAL_B, BOARD_MIXED, TERMINAL_A]).reshape(4, 16),
            dtype=torch.float32,
        )
        dones = torch.tensor([True, False, True, False], dtype=torch.bool)
        targets = agent._targets(next_states, dones)

        assert targets.shape == (4,)
        assert torch.all(targets == 0.0), f"Expected all-zero terminal targets, got {targets}"
        # Neither online nor target net should be evaluated on terminal boards
        assert online.forward_calls == 0
        assert target.forward_calls == 0

        # Also verify through full replay() on a batch of terminal transitions
        dummy_after = afterstates(BOARD_MIXED)[0][0]
        agent.remember(dummy_after, TERMINAL_A, True)
        agent.remember(dummy_after, TERMINAL_B, False)  # done=False but board is terminal
        agent.remember(dummy_after, BOARD_MIXED, True)  # non-terminal board but done=True
        agent.remember(dummy_after, TERMINAL_A, False)

        captured_targets = []
        orig_targets = agent._targets

        def spy_targets(ns, d):
            t = orig_targets(ns, d)
            captured_targets.append(t.clone())
            return t

        agent._targets = spy_targets
        stats = agent.replay(batch_size=4)
        assert len(captured_targets) == 1
        assert torch.all(captured_targets[0] == 0.0)
        assert stats["target_mean"] == 0.0


# ---------------------------------------------------------------------------
# Check 2: Reward not double counted:
#          target = r(s_{t+1}, a*) + gamma * V_target(after*)
#          computed by hand on a fixed batch and compared to replay()'s internal target.
# ---------------------------------------------------------------------------
def test_check2_reward_not_double_counted_hand_computed_vs_replay():
    gamma = 0.75
    # Prepare two transitions:
    # Transition 0: big merge on step t (e.g. merging two 512s -> 1024) led to afterstate x_0,
    # then spawn gave s_{t+1} = BOARD_MIXED.
    # Transition 1: another afterstate x_1, spawn gave s_{t+1} = BOARD_ALL_LEGAL.
    x_0 = np.array([1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    x_1 = np.array([512, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    after_mixed, merge_mixed, legal_mixed = afterstates(BOARD_MIXED)
    after_all, merge_all, legal_all = afterstates(BOARD_ALL_LEGAL)

    # Assign specific values in online and target stub nets
    online_table = {
        tuple(x_0): 1.25,
        tuple(x_1): -0.5,
        # For BOARD_MIXED (legal: 0, 1, 2; merge: [12, 0, 12]):
        tuple(after_mixed[0]): 2.0,   # score_0 = log2(13)/16 + 0.75 * 2.0 = 0.231277 + 1.5 = 1.731277
        tuple(after_mixed[1]): 2.1,   # score_1 = 0.0 + 0.75 * 2.1 = 1.575
        tuple(after_mixed[2]): 1.0,   # score_2 = log2(13)/16 + 0.75 * 1.0 = 0.981277
        # -> Online picks a* = 0 for BOARD_MIXED
        # For BOARD_ALL_LEGAL (legal: 0, 1, 2, 3; merge: [4, 16, 4, 16]):
        tuple(after_all[0]): 0.5,     # score_0 = log2(5)/16 + 0.75 * 0.5 = 0.14512 + 0.375 = 0.52012
        tuple(after_all[1]): 0.2,     # score_1 = log2(17)/16 + 0.75 * 0.2 = 0.255466 + 0.15 = 0.405466
        tuple(after_all[2]): 3.0,     # score_2 = log2(5)/16 + 0.75 * 3.0 = 0.14512 + 2.25 = 2.39512
        tuple(after_all[3]): 1.0,     # score_3 = log2(17)/16 + 0.75 * 1.0 = 0.255466 + 0.75 = 1.005466
        # -> Online picks a* = 2 for BOARD_ALL_LEGAL
    }
    target_table = {
        tuple(after_mixed[0]): 10.0,
        tuple(after_mixed[1]): -99.0,
        tuple(after_mixed[2]): -99.0,
        tuple(after_all[0]): -50.0,
        tuple(after_all[1]): -50.0,
        tuple(after_all[2]): 4.0,
        tuple(after_all[3]): -50.0,
    }

    online_net = StubValueNet(online_table, default=0.0)
    target_net = StubValueNet(target_table, default=0.0)
    agent = _make_stub_agent(online_net, target_net, gamma=gamma, batch_size=2)

    # Hand-compute targets:
    # Target for BOARD_MIXED (a* = 0, merge = 12, V_target = 10.0):
    r_mixed_star = math.log2(1.0 + 12.0) / 16.0
    hand_target_mixed = r_mixed_star + gamma * 10.0

    # Target for BOARD_ALL_LEGAL (a* = 2, merge = 4, V_target = 4.0):
    r_all_star = math.log2(1.0 + 4.0) / 16.0
    hand_target_all = r_all_star + gamma * 4.0

    # Store both transitions in memory
    agent.remember(x_0, BOARD_MIXED, False)
    agent.remember(x_1, BOARD_ALL_LEGAL, False)

    # Capture internal targets and states during replay()
    captured = {}
    orig_targets = agent._targets

    def spy_targets(next_states, dones):
        t = orig_targets(next_states, dones)
        captured["next_states"] = next_states.detach().cpu().numpy().astype(np.int64)
        captured["targets"] = t.detach().cpu().numpy()
        return t

    agent._targets = spy_targets
    stats = agent.replay(batch_size=2)

    # Verify each sampled element's internal target matches hand computation
    for ns_row, actual_target in zip(captured["next_states"], captured["targets"]):
        if np.array_equal(ns_row, BOARD_MIXED.reshape(16)):
            assert actual_target == pytest.approx(hand_target_mixed, rel=1e-6)
            # Confirm reward is NOT double-counted (not 2*r + gamma*V) and NOT omitted (not gamma*V)
            assert actual_target != pytest.approx(2 * r_mixed_star + gamma * 10.0, rel=1e-3)
            assert actual_target != pytest.approx(gamma * 10.0, rel=1e-3)
            # Confirm r(s_t, a_t) from x_0 (which had a 1024 merge) is nowhere in the target
            r_prev = math.log2(1.0 + 1024.0) / 16.0
            assert actual_target != pytest.approx(r_prev + hand_target_mixed, rel=1e-3)
        elif np.array_equal(ns_row, BOARD_ALL_LEGAL.reshape(16)):
            assert actual_target == pytest.approx(hand_target_all, rel=1e-6)
            assert actual_target != pytest.approx(2 * r_all_star + gamma * 4.0, rel=1e-3)
            assert actual_target != pytest.approx(gamma * 4.0, rel=1e-3)
        else:
            pytest.fail(f"Unexpected next_state row: {ns_row}")

    assert stats["target_mean"] == pytest.approx(float(np.mean(captured["targets"])), rel=1e-6)


# ---------------------------------------------------------------------------
# Check 3: Double action selection: a* uses ONLINE net, value uses TARGET net
#          (stub nets disagreeing).
# ---------------------------------------------------------------------------
def test_check3_double_action_selection_with_disagreeing_stub_nets():
    gamma = 0.8
    after, merge, legal = afterstates(BOARD_ALL_LEGAL)
    assert legal.all()
    r = merge_reward(merge, mode="log")

    # Online net strongly prefers action 1 (down), whereas Target net strongly prefers action 2 (right):
    # Online scores:
    #   a=0: r[0] + 0.8 * (-10)
    #   a=1: r[1] + 0.8 * (+50)  <-- online argmax a* = 1
    #   a=2: r[2] + 0.8 * (-20)
    #   a=3: r[3] + 0.8 * (-30)
    # Target V values:
    #   a=0: +100.0
    #   a=1: -7.5                <-- target V evaluated at a* = 1
    #   a=2: +9999.0             <-- standard max_a(r + gamma*V_target) would pick a=2!
    #   a=3: +500.0
    online_net = StubValueNet(
        {
            tuple(after[0]): -10.0,
            tuple(after[1]): 50.0,
            tuple(after[2]): -20.0,
            tuple(after[3]): -30.0,
        }
    )
    target_net = StubValueNet(
        {
            tuple(after[0]): 100.0,
            tuple(after[1]): -7.5,
            tuple(after[2]): 9999.0,
            tuple(after[3]): 500.0,
        }
    )
    agent = _make_stub_agent(online_net, target_net, gamma=gamma)

    ns = torch.as_tensor(BOARD_ALL_LEGAL.reshape(1, 16), dtype=torch.float32)
    target_val = agent._targets(ns, torch.tensor([False]))

    expected_double = float(r[1]) + gamma * (-7.5)
    wrong_max_target = float(r[2]) + gamma * 9999.0
    wrong_pure_online = float(r[1]) + gamma * 50.0

    assert target_val.item() == pytest.approx(expected_double, rel=1e-6)
    assert target_val.item() != pytest.approx(wrong_max_target, rel=1e-2)
    assert target_val.item() != pytest.approx(wrong_pure_online, rel=1e-2)

    # Target net must only be queried on the single chosen afterstate after[1]
    assert target_net.queried_boards == [tuple(after[1])]


# ---------------------------------------------------------------------------
# Check 4: Mask / no-op: illegal actions never selected in act / greedy_action / target,
#          even if V(after) is huge; afterstate of illegal action never enters the target.
# ---------------------------------------------------------------------------
def test_check4_illegal_actions_masked_and_never_enter_target():
    # Board with only ONE legal action:
    # All tiles at bottom-right corner [0..0, 2]:
    # - left (0): legal (moves 2 to col 0)
    # - down (1): illegal (already on bottom row)
    # - right (2): illegal (already on rightmost col)
    # - up (3): legal (moves 2 to top row)
    # Wait: let's also construct a board with 3 illegal moves and 1 legal move!
    # E.g. top-row full of distinct non-mergeable tiles, or bottom-right filled:
    # Consider:
    # [2, 4, 8, 16]
    # [0, 0, 0, 0]
    # [0, 0, 0, 0]
    # [0, 0, 0, 0]
    # Here:
    # - left (0): row 0 is [2, 4, 8, 16], no zeros and no adjacent equal -> unchanged (ILLEGAL)
    # - down (1): slides row 0 to row 3 -> changed (LEGAL)
    # - right (2): row 0 is [2, 4, 8, 16] -> unchanged (ILLEGAL)
    # - up (3): already at row 0 -> unchanged (ILLEGAL)
    one_legal_grid = np.array(
        [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    after, merge, legal = afterstates(one_legal_grid)
    assert legal.tolist() == [False, True, False, False]
    unchanged_key = tuple(one_legal_grid.reshape(16).tolist())
    legal_after_key = tuple(after[1].tolist())
    assert unchanged_key != legal_after_key

    # Give the illegal afterstate (unchanged_key) a colossal value +1e12,
    # and the only legal afterstate a very negative value -500.0.
    # Also test with `forbidden={unchanged_key}` on target_net to prove it never enters target_net!
    online_net = StubValueNet(
        table={unchanged_key: 1e12, legal_after_key: -500.0},
        default=1e12,
        forbidden={unchanged_key},
    )
    target_net = StubValueNet(
        table={unchanged_key: 1e12, legal_after_key: -250.0},
        default=1e12,
        forbidden={unchanged_key},
    )
    agent = _make_stub_agent(online_net, target_net, gamma=0.99, epsilon_start=0.0)
    fb = FastBoard.from_grid(one_legal_grid)
    invalid = fb.get_invalid_moves()
    assert invalid == [0, 2, 3]

    # 1. greedy_action picks the sole legal action (1)
    assert agent.greedy_action(one_legal_grid, invalid) == 1
    assert greedy_actions(online_net, one_legal_grid, gamma=0.99)[0] == 1

    # 2. act with epsilon=0.0 and epsilon=1.0 never picks illegal actions {0, 2, 3}
    agent.epsilon = 0.0
    assert agent.act(one_legal_grid, invalid) == 1
    agent.epsilon = 1.0
    for _ in range(50):
        assert agent.act(one_legal_grid, invalid) == 1

    # 3. _targets and replay() evaluate ONLY legal_after_key (-250.0) and never unchanged_key (1e12)
    ns = torch.as_tensor(one_legal_grid.reshape(1, 16), dtype=torch.float32)
    t = agent._targets(ns, torch.tensor([False]))
    assert t.item() == pytest.approx(0.99 * (-250.0), rel=1e-6)
    assert unchanged_key not in online_net.queried_boards
    assert unchanged_key not in target_net.queried_boards


# ---------------------------------------------------------------------------
# Check 5: pure_move / afterstate computation never consumes the board RNG
#          during act, greedy_action, and replay.
# ---------------------------------------------------------------------------
class _RngTrap(random.Random):
    """Random subclass that fails if any random generation method is called."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.armed = False

    def random(self):
        if self.armed:
            raise AssertionError("Board RNG .random() was consumed!")
        return super().random()

    def choice(self, seq):
        if self.armed:
            raise AssertionError("Board RNG .choice() was consumed!")
        return super().choice(seq)

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        if self.armed:
            raise AssertionError("Board RNG .choices() was consumed!")
        return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)


def test_check5_board_rng_never_consumed_in_pure_move_act_greedy_action_or_replay():
    trap_rng = _RngTrap(98765)
    board = FastBoard(rng=trap_rng)
    board.move("left")
    rng_state_before = trap_rng.getstate()
    trap_rng.armed = True

    agent = AfterstateAgent(16, 4, DQNConfig(arch="onehot_afterstate", batch_size=2, seed=55))

    # 1. pure_move and get_invalid_moves and afterstates
    for a in range(4):
        board.pure_move(a)
    invalid = board.get_invalid_moves()
    after, _, _ = afterstates(board.grid)
    batch_afterstates([board.grid, BOARD_MIXED])

    # 2. greedy_action must not touch board RNG OR agent.rng
    agent_rng_before = agent.rng.bit_generator.state
    g_act = agent.greedy_action(board.grid, invalid)
    assert g_act not in invalid
    assert agent.rng.bit_generator.state == agent_rng_before

    # 3. act (both greedy and exploratory) must not touch board RNG
    agent.epsilon = 0.0
    a0 = agent.act(board.grid, invalid)
    assert a0 not in invalid
    agent.epsilon = 1.0
    a1 = agent.act(board.grid, invalid)
    assert a1 not in invalid

    # 4. remember and replay must not touch board RNG
    agent.remember(after[a0], BOARD_MIXED, False)
    agent.remember(after[a1], BOARD_ALL_LEGAL, False)
    stats = agent.replay(batch_size=2)
    assert not stats["nan"]

    assert trap_rng.getstate() == rng_state_before


# ---------------------------------------------------------------------------
# Check 6: Save/load/evaluator consistency:
#          - same greedy actions before and after save/load
#          - eval.evaluate with checkpoint is deterministic across two runs
#          - equals stepping greedy_action manually for 1 game.
# ---------------------------------------------------------------------------
def test_check6_save_load_and_eval_determinism_and_manual_rollout_equivalence(tmp_path):
    agent = AfterstateAgent(
        16, 4, DQNConfig(arch="onehot_afterstate", gamma=0.95, batch_size=2, seed=314)
    )
    # Perform a couple of replay updates so weights are non-trivial
    after_m, _, _ = afterstates(BOARD_MIXED)
    agent.remember(after_m[0], BOARD_ALL_LEGAL, False)
    agent.remember(after_m[1], BOARD_MIXED, False)
    agent.replay(batch_size=2)

    ckpt_path = tmp_path / "afterstate_review_ckpt.pt"
    agent.save(ckpt_path)

    # Collect diverse boards and check greedy_action before/after save/load
    sample_boards = []
    b = FastBoard(rng=random.Random(101))
    for _ in range(25):
        inv = b.get_invalid_moves()
        if len(inv) == 4:
            break
        sample_boards.append(( [row[:] for row in b.grid], list(inv) ))
        act = agent.greedy_action(b.grid, inv)
        b.move(FastBoard.all_moves[act])

    before_actions = [agent.greedy_action(g, inv) for g, inv in sample_boards]

    # Reload via agent.load() and via RL.load_checkpoint()
    agent.load(ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    loaded.model.eval()

    after_load_actions = [agent.greedy_action(g, inv) for g, inv in sample_boards]
    checkpoint_actions = [loaded.greedy_action(g, inv) for g, inv in sample_boards]
    assert before_actions == after_load_actions == checkpoint_actions

    # Run eval.evaluate twice on the saved checkpoint
    eval_run1 = eval_mod.evaluate(
        checkpoint=ckpt_path, policy="greedy", games=2, seed_start=5000, board="fast", max_steps=200
    )
    eval_run2 = eval_mod.evaluate(
        checkpoint=ckpt_path, policy="greedy", games=2, seed_start=5000, board="fast", max_steps=200
    )
    assert eval_run1["per_game"] == eval_run2["per_game"]
    assert eval_run1["mean"] == eval_run2["mean"]
    assert eval_run1["median"] == eval_run2["median"]
    assert eval_run1["max"] == eval_run2["max"]
    assert eval_run1["max_tile_distribution"] == eval_run2["max_tile_distribution"]

    # Step greedy_action manually for game 0 (seed=5000) and compare against eval.evaluate
    seed = 5000
    manual_board = FastBoard(rng=random.Random(seed))
    manual_steps = 0
    while not manual_board.done and manual_steps < 200:
        inv = manual_board.get_invalid_moves()
        if len(inv) == 4:
            break
        act = loaded.greedy_action(manual_board.grid, inv)
        assert act not in inv
        manual_board.move(FastBoard.all_moves[act])
        manual_steps += 1

    game0 = eval_run1["per_game"][0]
    assert game0["seed"] == seed
    assert game0["score"] == int(manual_board.score_v2)
    assert game0["max_tile"] == int(np.max(manual_board.grid))
    assert game0["steps"] == manual_steps


# ---------------------------------------------------------------------------
# Check 7: Gamma placement: V target uses config.gamma; changing gamma changes
#          target as hand-computed.
# ---------------------------------------------------------------------------
def test_check7_gamma_placement_in_selection_and_target_evaluation():
    after, merge, legal = afterstates(BOARD_MIXED)
    # Legal actions on BOARD_MIXED are 0 (merge=12), 1 (merge=0), 2 (merge=12)
    assert legal.tolist() == [True, True, True, False]
    r0 = math.log2(1.0 + 12.0) / 16.0  # ~0.2312774
    r1 = 0.0

    # Part A: Fixed action selection (only action 0 has positive values), vary gamma
    online_fixed = StubValueNet({tuple(after[0]): 5.0, tuple(after[1]): -10.0, tuple(after[2]): -10.0})
    target_fixed = StubValueNet({tuple(after[0]): 8.0, tuple(after[1]): -10.0, tuple(after[2]): -10.0})

    ns = torch.as_tensor(BOARD_MIXED.reshape(1, 16), dtype=torch.float32)
    for gamma in (0.0, 0.25, 0.5, 0.9, 0.99):
        agent = _make_stub_agent(online_fixed, target_fixed, gamma=gamma, batch_size=1)
        hand_target = r0 + gamma * 8.0
        assert agent._targets(ns, torch.tensor([False])).item() == pytest.approx(hand_target, rel=1e-6)
        agent.remember(after[0], BOARD_MIXED, False)
        stats = agent.replay(batch_size=1)
        assert stats["target_mean"] == pytest.approx(hand_target, rel=1e-6)

    # Part B: Changing gamma flips online action choice a* between action 0 (high immediate r)
    # and action 1 (zero immediate r, higher V_online), jumping the target to V_target(after[1])!
    # Online: V(after[0]) = 0.0, V(after[1]) = 0.5, V(after[2]) = -10.0
    #   At gamma = 0.2: score(0) = r0 + 0 = 0.23128 > score(1) = 0 + 0.2 * 0.5 = 0.10 -> picks a*=0
    #   At gamma = 0.8: score(0) = r0 + 0 = 0.23128 < score(1) = 0 + 0.8 * 0.5 = 0.40 -> picks a*=1
    online_flip = StubValueNet({tuple(after[0]): 0.0, tuple(after[1]): 0.5, tuple(after[2]): -10.0})
    target_flip = StubValueNet({tuple(after[0]): 3.0, tuple(after[1]): 20.0, tuple(after[2]): -10.0})

    agent_low_gamma = _make_stub_agent(online_flip, target_flip, gamma=0.2, batch_size=1)
    assert agent_low_gamma.greedy_action(BOARD_MIXED, [3]) == 0
    assert agent_low_gamma._targets(ns, torch.tensor([False])).item() == pytest.approx(
        r0 + 0.2 * 3.0, rel=1e-6
    )

    agent_high_gamma = _make_stub_agent(online_flip, target_flip, gamma=0.8, batch_size=1)
    assert agent_high_gamma.greedy_action(BOARD_MIXED, [3]) == 1
    assert agent_high_gamma._targets(ns, torch.tensor([False])).item() == pytest.approx(
        r1 + 0.8 * 20.0, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Check 8: Defaults untouched: DQNConfig() defaults unchanged,
#          legacy and onehot_dueling checkpoints still load.
# ---------------------------------------------------------------------------
def test_check8_defaults_untouched_and_legacy_dueling_checkpoints_load(tmp_path):
    cfg = DQNConfig()
    expected_defaults = {
        "gamma": 0.99,
        "lr": 1e-4,
        "memory_capacity": 6000,
        "batch_size": 128,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "per_beta_anneal_steps": 100_000,
        "per_eps": 1e-5,
        "abs_error_upper": 1.0,
        "epsilon_start": 0.15,
        "epsilon_min": 0.001,
        "epsilon_decay": 0.999,
        "epsilon_schedule": "update_exp",
        "epsilon_decay_steps": 200_000,
        "target_update_episodes": 50,
        "grad_clip": 1.0,
        "seed": None,
        "arch": "legacy",
        "n_step": 1,
    }
    actual = asdict(cfg)
    # Pre-existing fields keep their values; fields added by the other experiment
    # branches (merged in exp/integrate) must sit at their OFF defaults.
    assert {k: actual[k] for k in expected_defaults} == expected_defaults
    off_defaults = dict(per_max_priority='scan', track_replay_age=False, reward_mode='log',
                        reward_scale=256.0, restart_frac=0.0, restart_pool_size=5000,
                        epsilon_decay_start_episode=0)
    assert {k: v for k, v in actual.items() if k not in expected_defaults} == off_defaults

    # Default DDQNAgent builds legacy DQN and round-trips raw state_dict checkpoints
    legacy_agent = DDQNAgent(16, 4)
    assert isinstance(legacy_agent.model, DQN)
    legacy_ckpt = tmp_path / "legacy.pt"
    legacy_agent.save(legacy_ckpt)
    raw_payload = torch.load(legacy_ckpt, map_location="cpu", weights_only=True)
    assert "format" not in raw_payload
    loaded_legacy = load_checkpoint(legacy_ckpt)
    assert isinstance(loaded_legacy, DDQNAgent) and not isinstance(loaded_legacy, AfterstateAgent)
    assert loaded_legacy.config.arch == "legacy"
    for k, v in legacy_agent.model.state_dict().items():
        assert torch.equal(v, loaded_legacy.model.state_dict()[k])

    # onehot_dueling DDQNAgent builds OneHotDuelingDQN and round-trips dqn2048-v2 checkpoints
    dueling_agent = DDQNAgent(16, 4, DQNConfig(arch="onehot_dueling", seed=12))
    assert isinstance(dueling_agent.model, OneHotDuelingDQN)
    dueling_ckpt = tmp_path / "dueling.pt"
    dueling_agent.save(dueling_ckpt)
    loaded_dueling = load_checkpoint(dueling_ckpt)
    assert isinstance(loaded_dueling, DDQNAgent) and not isinstance(loaded_dueling, AfterstateAgent)
    assert loaded_dueling.config.arch == "onehot_dueling"
    for k, v in dueling_agent.model.state_dict().items():
        assert torch.equal(v, loaded_dueling.model.state_dict()[k])
