#!/usr/bin/env python3
"""Micro-benchmark script comparing afterstate expand implementations.

Measures:
  A. steps/s: Pure acting loop without replay
  B. updates/s: Pure replay loop with prefilled buffer
  C. mixed steps/s: Acting interleaved with replay at 0.45 ratio
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure worktree root is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from afterstate import afterstates
from afterstate_agent import AfterstateAgent
from config import DQNConfig
from fast_board import FastBoard


def check_vectorized_api_support() -> Tuple[bool, str]:
    """Check whether the vectorized afterstate expand API is available."""
    import afterstate

    # 1. DQNConfig.afterstate_expand
    if not hasattr(DQNConfig, "afterstate_expand"):
        return False, "config.DQNConfig 缺少 'afterstate_expand' 欄位"

    # 2. afterstate.py has batch_afterstates_vectorized or EXPAND_MODES
    has_vec = hasattr(afterstate, "batch_afterstates_vectorized")
    modes = getattr(afterstate, "EXPAND_MODES", ())
    if not has_vec and "vectorized" not in modes:
        return False, "afterstate 模組缺少 'batch_afterstates_vectorized' 或 EXPAND_MODES"

    # 3. Instantiate agent and check select_action signature
    try:
        cfg = DQNConfig(arch="onehot_afterstate", afterstate_expand="vectorized")
        _ = AfterstateAgent(16, 4, config=cfg)
        import inspect
        import afterstate_policy
        sig = inspect.signature(afterstate_policy.select_action)
        if "return_after" not in sig.parameters or "expand" not in sig.parameters:
            return False, "afterstate_policy.select_action 缺少 'return_after' 或 'expand' 參數"
    except Exception as exc:
        return False, f"初始化 vectorized agent 失敗 ({exc})"

    return True, ""


def create_agent(mode: str, seed: Optional[int] = None) -> AfterstateAgent:
    """Instantiate an AfterstateAgent for the given expand mode."""
    kwargs: Dict[str, Any] = {"arch": "onehot_afterstate"}
    if seed is not None:
        kwargs["seed"] = seed

    if mode == "vectorized":
        ok, reason = check_vectorized_api_support()
        if not ok:
            raise RuntimeError(f"API 尚未實作: {reason}")
        kwargs["afterstate_expand"] = "vectorized"
    elif hasattr(DQNConfig, "afterstate_expand"):
        kwargs["afterstate_expand"] = "reference"

    config = DQNConfig(**kwargs)
    return AfterstateAgent(16, 4, config=config)


def run_acting_step(
    agent: AfterstateAgent,
    board: FastBoard,
    state: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, FastBoard, bool, int]:
    """Execute a single acting step faithfully reproducing main.py loop."""
    invalid = board.get_invalid_moves()
    if len(invalid) == 4:
        return state, board, True, 0

    action = agent.act(state, invalid)
    if mode == "vectorized":
        after_state = getattr(agent, "last_after_state", None)
        if after_state is None:
            raise RuntimeError("API 尚未實作: agent.last_after_state 為 None 或未定義")
    else:
        after_state = afterstates(board.grid)[0][action]

    next_grid, _reward, done = board.move(board.all_moves[action])
    next_state = torch.as_tensor(next_grid, dtype=torch.float32)
    next_invalid = board.get_invalid_moves()
    done = bool(done or len(next_invalid) == 4)

    agent.set_env_step(agent.env_step + 1)
    agent.remember(after_state, next_state, done)

    return next_state, board, done, 1


def bench_measure_a(
    agent: AfterstateAgent,
    mode: str,
    seconds: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure A: Pure acting loop steps/s."""
    episode = 0
    board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
    state = torch.as_tensor(board.state, dtype=torch.float32)
    steps = 0

    # Quick 5-step warmup
    for _ in range(5):
        next_state, board, done, _ = run_acting_step(agent, board, state, mode)
        if done:
            episode += 1
            board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
            state = torch.as_tensor(board.state, dtype=torch.float32)
        else:
            state = next_state

    start_time = time.perf_counter()
    while True:
        if time.perf_counter() - start_time >= seconds:
            break
        next_state, board, done, stepped = run_acting_step(agent, board, state, mode)
        steps += stepped
        if done:
            episode += 1
            board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
            state = torch.as_tensor(board.state, dtype=torch.float32)
        else:
            state = next_state

    elapsed = time.perf_counter() - start_time
    rate = steps / elapsed if elapsed > 0 else 0.0
    return {
        "steps": steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": rate,
    }


def fill_replay_buffer(
    agent: AfterstateAgent,
    mode: str,
    target_capacity: int,
    seed: Optional[int] = None,
) -> int:
    """Fill agent replay buffer to target capacity using acting loop without timing."""
    episode = 100_000
    board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
    state = torch.as_tensor(board.state, dtype=torch.float32)
    steps_added = 0

    while agent.memory.size < target_capacity:
        next_state, board, done, stepped = run_acting_step(agent, board, state, mode)
        steps_added += stepped
        if done:
            episode += 1
            board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
            state = torch.as_tensor(board.state, dtype=torch.float32)
        else:
            state = next_state

    return steps_added


def bench_measure_b(
    agent: AfterstateAgent,
    seconds: float,
) -> Dict[str, Any]:
    """Measure B: Pure replay updates/s with prefilled buffer."""
    # 2 warmup replays
    for _ in range(2):
        agent.replay()

    start_time = time.perf_counter()
    updates = 0
    while True:
        if time.perf_counter() - start_time >= seconds:
            break
        agent.replay()
        updates += 1

    elapsed = time.perf_counter() - start_time
    rate = updates / elapsed if elapsed > 0 else 0.0
    return {
        "updates": updates,
        "elapsed_sec": elapsed,
        "updates_per_sec": rate,
    }


def bench_measure_c(
    agent: AfterstateAgent,
    mode: str,
    seconds: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure C: Mixed acting and replay at ratio 0.45."""
    agent.epsilon = agent.config.epsilon_start
    episode = 200_000
    board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
    state = torch.as_tensor(board.state, dtype=torch.float32)

    replay_acc = 0.0
    env_steps = 0
    updates = 0

    start_time = time.perf_counter()
    while True:
        if time.perf_counter() - start_time >= seconds:
            break
        next_state, board, done, stepped = run_acting_step(agent, board, state, mode)
        if stepped > 0:
            env_steps += stepped
            replay_acc += 0.45
            while replay_acc >= 1.0:
                agent.replay()
                replay_acc -= 1.0
                updates += 1

        if done:
            episode += 1
            board = FastBoard(rng=random.Random(None if seed is None else seed + episode))
            state = torch.as_tensor(board.state, dtype=torch.float32)
        else:
            state = next_state

    elapsed = time.perf_counter() - start_time
    env_rate = env_steps / elapsed if elapsed > 0 else 0.0
    upd_rate = updates / elapsed if elapsed > 0 else 0.0
    return {
        "env_steps": env_steps,
        "updates": updates,
        "elapsed_sec": elapsed,
        "env_steps_per_sec": env_rate,
        "updates_per_sec": upd_rate,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="2048 Afterstate Expand Micro-benchmark (reference vs vectorized)"
    )
    parser.add_argument(
        "--expand",
        nargs="+",
        choices=["reference", "vectorized", "both"],
        default=["reference", "vectorized"],
        help="Expand mode(s) to benchmark (reference, vectorized, or both; default: both)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Per-measurement duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of PyTorch threads (default: 1)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to write JSON benchmark results",
    )
    args = parser.parse_args(argv)

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")

    torch.set_num_threads(args.threads)

    # Resolve modes preserving order
    modes: List[str] = []
    for item in args.expand:
        if item == "both":
            for m in ["reference", "vectorized"]:
                if m not in modes:
                    modes.append(m)
        elif item not in modes:
            modes.append(item)

    loadavg_start = os.getloadavg()
    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for mode in modes:
        if mode == "vectorized":
            ok, reason = check_vectorized_api_support()
            if not ok:
                msg = f"API 尚未實作: {reason}"
                print(f"\n[vectorized] {msg}", file=sys.stderr, flush=True)
                errors["vectorized"] = msg
                # If only vectorized was requested, exit with error
                if len(modes) == 1:
                    print(f"Error: {msg}", file=sys.stderr)
                    return 1
                continue

        print(f"\n--- Running Benchmark for mode: {mode} (seed={args.seed}) ---", flush=True)
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        agent = create_agent(mode, seed=args.seed)

        print(f"[{mode}] Measuring A: acting steps/s ({args.seconds:.1f}s)...", flush=True)
        res_a = bench_measure_a(agent, mode, args.seconds, seed=args.seed)
        print(f"  -> {res_a['steps_per_sec']:.1f} steps/s ({res_a['steps']} steps in {res_a['elapsed_sec']:.2f}s)", flush=True)

        print(f"[{mode}] Filling buffer to capacity ({agent.config.memory_capacity} transitions)...", flush=True)
        t_fill_start = time.perf_counter()
        added = fill_replay_buffer(agent, mode, agent.config.memory_capacity, seed=args.seed)
        t_fill_el = time.perf_counter() - t_fill_start
        print(f"  -> Buffer filled (size={agent.memory.size}, added {added} in {t_fill_el:.2f}s)", flush=True)

        print(f"[{mode}] Measuring B: replay updates/s ({args.seconds:.1f}s)...", flush=True)
        res_b = bench_measure_b(agent, args.seconds)
        print(f"  -> {res_b['updates_per_sec']:.1f} updates/s ({res_b['updates']} updates in {res_b['elapsed_sec']:.2f}s)", flush=True)

        print(f"[{mode}] Measuring C: mixed training ({args.seconds:.1f}s, replay_ratio=0.45)...", flush=True)
        res_c = bench_measure_c(agent, mode, args.seconds, seed=args.seed)
        print(
            f"  -> {res_c['env_steps_per_sec']:.1f} env steps/s, {res_c['updates_per_sec']:.1f} updates/s "
            f"({res_c['env_steps']} steps, {res_c['updates']} updates in {res_c['elapsed_sec']:.2f}s)",
            flush=True,
        )

        results[mode] = {
            "measure_a": res_a,
            "measure_b": res_b,
            "measure_c": res_c,
        }

    loadavg_end = os.getloadavg()
    torch_threads = torch.get_num_threads()

    # Pretty print summary table
    print("\n" + "=" * 80)
    print("2048 Afterstate Expand Microbenchmark Results")
    print(f"Torch threads: {torch_threads} | Duration: {args.seconds:.1f}s/phase | Seed: {args.seed}")
    print(
        f"Loadavg: start=({loadavg_start[0]:.2f}, {loadavg_start[1]:.2f}, {loadavg_start[2]:.2f}) "
        f"end=({loadavg_end[0]:.2f}, {loadavg_end[1]:.2f}, {loadavg_end[2]:.2f})"
    )
    print("=" * 80)
    print(
        f"| {'Mode':<12} | {'A: Act (steps/s)':<18} | {'B: Replay (upd/s)':<18} | "
        f"{'C: Mixed (steps/s)':<19} | {'C: Mixed (upd/s)':<17} |"
    )
    print(f"|{'-'*14}|{'-'*20}|{'-'*20}|{'-'*21}|{'-'*19}|")

    for mode in modes:
        if mode in results:
            r = results[mode]
            a_s = f"{r['measure_a']['steps_per_sec']:,.1f}"
            b_s = f"{r['measure_b']['updates_per_sec']:,.2f}"
            c_step = f"{r['measure_c']['env_steps_per_sec']:,.1f}"
            c_upd = f"{r['measure_c']['updates_per_sec']:,.2f}"
            print(f"| {mode:<12} | {a_s:>18} | {b_s:>18} | {c_step:>19} | {c_upd:>17} |")
        elif mode in errors:
            err_msg = "API 尚未實作"
            print(f"| {mode:<12} | {err_msg:>18} | {err_msg:>18} | {err_msg:>19} | {err_msg:>17} |")

    print("-" * 80)

    # Compute ratios if both reference and vectorized succeeded
    ratios: Optional[Dict[str, float]] = None
    if "reference" in results and "vectorized" in results:
        ref = results["reference"]
        vec = results["vectorized"]
        ratio_a = vec["measure_a"]["steps_per_sec"] / max(ref["measure_a"]["steps_per_sec"], 1e-9)
        ratio_b = vec["measure_b"]["updates_per_sec"] / max(ref["measure_b"]["updates_per_sec"], 1e-9)
        ratio_c_step = vec["measure_c"]["env_steps_per_sec"] / max(ref["measure_c"]["env_steps_per_sec"], 1e-9)
        ratio_c_upd = vec["measure_c"]["updates_per_sec"] / max(ref["measure_c"]["updates_per_sec"], 1e-9)

        ratios = {
            "act_steps_ratio": ratio_a,
            "replay_updates_ratio": ratio_b,
            "mixed_env_steps_ratio": ratio_c_step,
            "mixed_updates_ratio": ratio_c_upd,
        }
        print("Ratio (vectorized vs reference):")
        print(f"  Measure A (acting steps/s):        {ratio_a:.2f}x")
        print(f"  Measure B (replay updates/s):      {ratio_b:.2f}x")
        print(f"  Measure C (mixed env steps/s):     {ratio_c_step:.2f}x")
        print(f"  Measure C (mixed updates/s):       {ratio_c_upd:.2f}x")
    elif "vectorized" in modes and "vectorized" not in results:
        print("Ratio (vectorized vs reference): N/A (vectorized 模式 API 尚未實作)")
    print("=" * 80)

    if args.json:
        json_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "expand": modes,
                "seconds": args.seconds,
                "seed": args.seed,
                "threads": args.threads,
            },
            "environment": {
                "torch_threads": torch_threads,
                "loadavg_start": list(loadavg_start),
                "loadavg_end": list(loadavg_end),
            },
            "results": results,
            "errors": errors,
            "ratios": ratios,
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark results to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
