#!/usr/bin/env python3
"""Micro-benchmark script comparing afterstate expand implementations and variants.

Measures:
  A. steps/s: Pure acting loop without replay
  B. updates/s: Pure replay loop with prefilled buffer
  C. mixed steps/s: Acting interleaved with replay at 0.45 ratio
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import statistics
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

TAIPEI_TZ = timezone(timedelta(hours=8))


def get_taipei_time() -> str:
    """Return current timestamp string in Taipei time (UTC+8)."""
    return datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S CST")


@dataclass
class Variant:
    name: str
    expand: str = "reference"
    threads: int = 1
    tdinf: Optional[bool] = None
    tdfwd: Optional[str] = None
    raw_str: str = ""


def parse_variant(spec: str, default_threads: int = 1) -> Variant:
    """Parse a variant CLI spec of format NAME=key:value,key:value."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"無效的 variant 格式: '{spec}'。格式應為 NAME=key:value[,key:value...]"
        )
    name, conf_str = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Variant 名稱不能為空: '{spec}'")

    expand = "reference"
    threads = default_threads
    tdinf: Optional[bool] = None
    tdfwd: Optional[str] = None

    if conf_str.strip():
        pairs = conf_str.split(",")
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise argparse.ArgumentTypeError(
                    f"無效的鍵值對 '{pair}' (variant '{name}')。格式應為 key:value"
                )
            k, v = pair.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k == "expand":
                if v not in ("reference", "vectorized"):
                    raise argparse.ArgumentTypeError(
                        f"無效的 expand 設定 '{v}' (variant '{name}')。僅支援 'reference' 或 'vectorized'"
                    )
                expand = v
            elif k == "threads":
                try:
                    threads = int(v)
                    if threads < 1:
                        raise ValueError()
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"無效的 threads 設定 '{v}' (variant '{name}')。必須為正整數 (>= 1)"
                    )
            elif k == "tdinf":
                if v in ("0", "false", "False"):
                    tdinf = False
                elif v in ("1", "true", "True"):
                    tdinf = True
                else:
                    raise argparse.ArgumentTypeError(
                        f"無效的 tdinf 設定 '{v}' (variant '{name}')。僅支援 0 或 1"
                    )
            elif k == "tdfwd":
                if v not in ("reference", "fast"):
                    raise argparse.ArgumentTypeError(
                        f"無效的 tdfwd 設定 '{v}' (variant '{name}')。僅支援 'reference' 或 'fast'"
                    )
                tdfwd = v
            else:
                raise argparse.ArgumentTypeError(
                    f"不支援的 variant key '{k}' (variant '{name}')。支援鍵值: expand, threads, tdinf, tdfwd"
                )

    return Variant(name=name, expand=expand, threads=threads, tdinf=tdinf, tdfwd=tdfwd, raw_str=spec)


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


def create_agent_for_variant(variant: Variant, seed: Optional[int] = None) -> AfterstateAgent:
    """Instantiate an AfterstateAgent for the given variant with attribute validation."""
    kwargs: Dict[str, Any] = {"arch": "onehot_afterstate"}
    if seed is not None:
        kwargs["seed"] = seed

    # 1. afterstate_expand
    if not hasattr(DQNConfig, "afterstate_expand"):
        raise RuntimeError("config.DQNConfig 缺少 'afterstate_expand' 欄位（相關功能尚未在 config.py 實作）")
    if variant.expand == "vectorized":
        ok, reason = check_vectorized_api_support()
        if not ok:
            raise RuntimeError(f"vectorized API 尚未實作: {reason}")
    kwargs["afterstate_expand"] = variant.expand

    # 2. td_inference_mode
    if variant.tdinf is not None:
        if not hasattr(DQNConfig, "td_inference_mode"):
            raise RuntimeError("config.DQNConfig 缺少 'td_inference_mode' 欄位（相關功能尚未在 config.py 實作）")
        kwargs["td_inference_mode"] = variant.tdinf

    # 3. td_forward
    if variant.tdfwd is not None:
        if not hasattr(DQNConfig, "td_forward"):
            raise RuntimeError("config.DQNConfig 缺少 'td_forward' 欄位（相關功能尚未在 config.py 實作）")
        kwargs["td_forward"] = variant.tdfwd

    config = DQNConfig(**kwargs)

    # Verification using getattr/setattr
    if hasattr(DQNConfig, "afterstate_expand"):
        if getattr(config, "afterstate_expand", None) != variant.expand:
            raise RuntimeError(
                f"DQNConfig.afterstate_expand 設定值不符: 預期 {variant.expand}, 實際 {getattr(config, 'afterstate_expand', None)}"
            )
    if variant.tdinf is not None and hasattr(DQNConfig, "td_inference_mode"):
        if getattr(config, "td_inference_mode", None) != variant.tdinf:
            raise RuntimeError(
                f"DQNConfig.td_inference_mode 設定值不符: 預期 {variant.tdinf}, 實際 {getattr(config, 'td_inference_mode', None)}"
            )
    if variant.tdfwd is not None and hasattr(DQNConfig, "td_forward"):
        if getattr(config, "td_forward", None) != variant.tdfwd:
            raise RuntimeError(
                f"DQNConfig.td_forward 設定值不符: 預期 {variant.tdfwd}, 實際 {getattr(config, 'td_forward', None)}"
            )

    return AfterstateAgent(16, 4, config=config)


def create_agent(mode: str, seed: Optional[int] = None) -> AfterstateAgent:
    """Instantiate an AfterstateAgent for backward compatibility."""
    return create_agent_for_variant(Variant(name=mode, expand=mode), seed=seed)


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


def _print_round_table(
    variants: List[Variant],
    round_results: Dict[str, Any],
    round_ratios: Dict[str, Dict[str, float]],
    measures: List[str],
) -> None:
    """Print results table for a single round."""
    has_a = "A" in measures
    has_b = "B" in measures
    has_c = "C" in measures

    headers = ["Variant", "Threads"]
    if has_a:
        headers.append("A: Act (steps/s)")
    if has_b:
        headers.append("B: Replay (upd/s)")
    if has_c:
        headers.extend(["C: Steps/s", "C: Upd/s", "Ratio (Steps)", "Ratio (Upd)"])

    header_str = " | ".join(f"{h:<15}" for h in headers)
    div_str = "-+-".join("-" * 15 for _ in headers)
    print(header_str)
    print(div_str)

    for v in variants:
        row: List[str] = [f"{v.name:<15}", f"{v.threads:<15}"]
        if v.name not in round_results:
            row.extend(["ERR"] * (len(headers) - 2))
            print(" | ".join(row))
            continue

        res = round_results[v.name]
        ratios = round_ratios.get(v.name, {})

        if has_a:
            a_val = f"{res.get('measure_a', {}).get('steps_per_sec', 0.0):,.1f}"
            row.append(f"{a_val:>15}")
        if has_b:
            b_val = f"{res.get('measure_b', {}).get('updates_per_sec', 0.0):,.2f}"
            row.append(f"{b_val:>15}")
        if has_c:
            c_steps = f"{res.get('measure_c', {}).get('env_steps_per_sec', 0.0):,.1f}"
            c_upd = f"{res.get('measure_c', {}).get('updates_per_sec', 0.0):,.2f}"
            r_step = f"{ratios.get('env_steps_ratio', 1.0):.2f}x"
            r_upd = f"{ratios.get('updates_ratio', 1.0):.2f}x"
            row.extend([f"{c_steps:>15}", f"{c_upd:>15}", f"{r_step:>15}", f"{r_upd:>15}"])

        print(" | ".join(row))
    print(div_str)


def _print_final_summary(
    variants: List[Variant],
    baseline_name: str,
    summary_variants: Dict[str, Any],
    measures: List[str],
) -> None:
    """Print multi-round aggregated summary."""
    has_c = "C" in measures
    has_a = "A" in measures
    has_b = "B" in measures

    for v in variants:
        v_sum = summary_variants.get(v.name, {})
        print(f"\nVariant: '{v.name}' (expand={v.expand}, threads={v.threads}, tdinf={v.tdinf}, tdfwd={v.tdfwd})")
        if v.name == baseline_name:
            print("  (Baseline)")
            continue

        if has_c:
            c_steps_list = v_sum.get("env_steps_ratio_list", [])
            c_steps_med = v_sum.get("env_steps_ratio_median")
            c_steps_str = "[" + ", ".join(f"{x:.2f}x" for x in c_steps_list) + "]"
            med_step_str = f"{c_steps_med:.2f}x" if c_steps_med is not None else "N/A"
            print(f"  Measure C env steps/s ratio: {c_steps_str} -> Median: {med_step_str}")

            c_upd_list = v_sum.get("updates_ratio_list", [])
            c_upd_med = v_sum.get("updates_ratio_median")
            c_upd_str = "[" + ", ".join(f"{x:.2f}x" for x in c_upd_list) + "]"
            med_upd_str = f"{c_upd_med:.2f}x" if c_upd_med is not None else "N/A"
            print(f"  Measure C updates/s ratio:   {c_upd_str} -> Median: {med_upd_str}")

        if has_a and "act_steps_ratio_list" in v_sum:
            a_list = v_sum.get("act_steps_ratio_list", [])
            a_med = v_sum.get("act_steps_ratio_median")
            a_str = "[" + ", ".join(f"{x:.2f}x" for x in a_list) + "]"
            med_a_str = f"{a_med:.2f}x" if a_med is not None else "N/A"
            print(f"  Measure A acting ratio:      {a_str} -> Median: {med_a_str}")

        if has_b and "replay_updates_ratio_list" in v_sum:
            b_list = v_sum.get("replay_updates_ratio_list", [])
            b_med = v_sum.get("replay_updates_ratio_median")
            b_str = "[" + ", ".join(f"{x:.2f}x" for x in b_list) + "]"
            med_b_str = f"{b_med:.2f}x" if b_med is not None else "N/A"
            print(f"  Measure B replay ratio:      {b_str} -> Median: {med_b_str}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="2048 Afterstate Expand Micro-benchmark (reference vs vectorized / variants)"
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Define a variant: NAME=key:value,key:value (supported keys: expand, threads, tdinf, tdfwd; repeatable)",
    )
    parser.add_argument(
        "--expand",
        nargs="+",
        choices=["reference", "vectorized", "both"],
        default=["reference", "vectorized"],
        help="Legacy: expand mode(s) to benchmark (reference, vectorized, or both; default: both)",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        default=None,
        help="Measures to run (A, B, C; default: C when --variant is used, A B C for legacy --expand)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of benchmark rounds (default: 1)",
    )
    parser.add_argument(
        "--order",
        choices=["abba", "abab"],
        default="abba",
        help="Execution order across rounds: abba (reverse on even rounds) or abab (default: abba)",
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
        help="Default number of PyTorch threads (default: 1)",
    )
    parser.add_argument(
        "--interop-threads",
        type=int,
        default=None,
        help="Number of PyTorch interop threads (must be set once before any torch work; default: None)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to write JSON benchmark results",
    )
    args = parser.parse_args(argv)

    if args.threads < 1:
        parser.error("--threads 必須 >= 1")
    if args.rounds < 1:
        parser.error("--rounds 必須 >= 1")
    if args.seconds <= 0:
        parser.error("--seconds 必須 > 0")

    # Set interop threads early before any computation
    if args.interop_threads is not None:
        if args.interop_threads < 1:
            parser.error("--interop-threads 必須 >= 1")
        try:
            torch.set_num_interop_threads(args.interop_threads)
        except RuntimeError as exc:
            print(f"Error: 無法設定 interop threads: {exc}", file=sys.stderr)
            return 1

    # Default thread count
    torch.set_num_threads(args.threads)

    # Resolve variants
    if args.variant:
        variants: List[Variant] = []
        for var_spec in args.variant:
            try:
                v = parse_variant(var_spec, default_threads=args.threads)
                variants.append(v)
            except argparse.ArgumentTypeError as err:
                parser.error(str(err))
    else:
        # Backward compatibility with --expand
        modes: List[str] = []
        for item in args.expand:
            if item == "both":
                for m in ["reference", "vectorized"]:
                    if m not in modes:
                        modes.append(m)
            elif item not in modes:
                modes.append(item)
        variants = [
            Variant(name=m, expand=m, threads=args.threads, raw_str=f"expand:{m}")
            for m in modes
        ]

    # Check for duplicate variant names
    variant_names = [v.name for v in variants]
    if len(variant_names) != len(set(variant_names)):
        parser.error(f"Variant 名稱重複: {variant_names}")

    # Resolve measures
    if args.measures is None:
        if args.variant:
            measures = ["C"]
        else:
            measures = ["A", "B", "C"]
    else:
        measures = []
        for item in args.measures:
            for part in item.replace(",", " ").split():
                part_up = part.upper()
                if part_up not in ("A", "B", "C"):
                    parser.error(f"無效的 measure: '{part}'。僅支援 A, B, C")
                if part_up not in measures:
                    measures.append(part_up)
        if not measures:
            measures = ["C"]

    # Pre-validate variants
    for v in variants:
        if v.expand == "vectorized":
            ok, reason = check_vectorized_api_support()
            if not ok:
                msg = f"API 尚未實作 (vectorized): {reason}"
                print(f"Error in variant '{v.name}': {msg}", file=sys.stderr)
                return 1
        if v.tdinf is not None and not hasattr(DQNConfig, "td_inference_mode"):
            print(
                f"Error in variant '{v.name}': config.DQNConfig 缺少 'td_inference_mode' 欄位（相關功能尚未在 config.py 實作）",
                file=sys.stderr,
            )
            return 1
        if v.tdfwd is not None and not hasattr(DQNConfig, "td_forward"):
            print(
                f"Error in variant '{v.name}': config.DQNConfig 缺少 'td_forward' 欄位（相關功能尚未在 config.py 實作）",
                file=sys.stderr,
            )
            return 1

    baseline = variants[0]
    baseline_name = baseline.name

    all_rounds_data: List[Dict[str, Any]] = []

    for round_idx in range(args.rounds):
        round_num = round_idx + 1
        if args.order == "abba" and (round_num % 2 == 0):
            round_variants = list(reversed(variants))
        else:
            round_variants = list(variants)

        round_start_time = get_taipei_time()
        round_start_load = os.getloadavg()

        print("\n" + "=" * 80)
        print(
            f"--- Round {round_num}/{args.rounds} [Start: {round_start_time} | "
            f"loadavg: {round_start_load[0]:.2f}, {round_start_load[1]:.2f}, {round_start_load[2]:.2f}] ---"
        )
        print(f"Variant Order: {[v.name for v in round_variants]}")
        print("=" * 80)

        round_results: Dict[str, Any] = {}
        round_errors: Dict[str, str] = {}

        # Round seed: identical across variants in this round
        round_seed = args.seed if args.seed is not None else (1000 + round_idx * 100)

        for variant in round_variants:
            print(
                f"\n>>> Running Variant: '{variant.name}' "
                f"(expand={variant.expand}, threads={variant.threads}, tdinf={variant.tdinf}, tdfwd={variant.tdfwd}, seed={round_seed})",
                flush=True,
            )
            torch.set_num_threads(variant.threads)
            random.seed(round_seed)
            np.random.seed(round_seed)
            torch.manual_seed(round_seed)

            try:
                agent = create_agent_for_variant(variant, seed=round_seed)
            except Exception as exc:
                print(f"[{variant.name}] 初始化 Agent 失敗: {exc}", file=sys.stderr, flush=True)
                round_errors[variant.name] = str(exc)
                continue

            # Prefill buffer untimed
            print(
                f"[{variant.name}] Filling buffer to capacity ({agent.config.memory_capacity} transitions)...",
                flush=True,
            )
            t_fill_start = time.perf_counter()
            added = fill_replay_buffer(agent, variant.expand, agent.config.memory_capacity, seed=round_seed)
            t_fill_el = time.perf_counter() - t_fill_start
            print(
                f"  -> Buffer filled (size={agent.memory.size}, added {added} in {t_fill_el:.2f}s)",
                flush=True,
            )

            v_res: Dict[str, Any] = {}
            if "A" in measures:
                print(f"[{variant.name}] Measuring A: acting steps/s ({args.seconds:.1f}s)...", flush=True)
                res_a = bench_measure_a(agent, variant.expand, args.seconds, seed=round_seed)
                print(
                    f"  -> {res_a['steps_per_sec']:.1f} steps/s ({res_a['steps']} steps in {res_a['elapsed_sec']:.2f}s)",
                    flush=True,
                )
                v_res["measure_a"] = res_a

            if "B" in measures:
                print(f"[{variant.name}] Measuring B: replay updates/s ({args.seconds:.1f}s)...", flush=True)
                res_b = bench_measure_b(agent, args.seconds)
                print(
                    f"  -> {res_b['updates_per_sec']:.1f} updates/s ({res_b['updates']} updates in {res_b['elapsed_sec']:.2f}s)",
                    flush=True,
                )
                v_res["measure_b"] = res_b

            if "C" in measures:
                print(
                    f"[{variant.name}] Measuring C: mixed training ({args.seconds:.1f}s, replay_ratio=0.45)...",
                    flush=True,
                )
                res_c = bench_measure_c(agent, variant.expand, args.seconds, seed=round_seed)
                print(
                    f"  -> {res_c['env_steps_per_sec']:.1f} env steps/s, {res_c['updates_per_sec']:.1f} updates/s "
                    f"({res_c['env_steps']} steps, {res_c['updates']} updates in {res_c['elapsed_sec']:.2f}s)",
                    flush=True,
                )
                v_res["measure_c"] = res_c

            round_results[variant.name] = v_res

        round_end_time = get_taipei_time()
        round_end_load = os.getloadavg()
        print(
            f"\n--- End of Round {round_num}/{args.rounds} [End: {round_end_time} | "
            f"loadavg: {round_end_load[0]:.2f}, {round_end_load[1]:.2f}, {round_end_load[2]:.2f}] ---"
        )

        # Compute ratios vs baseline for this round
        round_ratios: Dict[str, Dict[str, float]] = {}
        if baseline_name in round_results:
            base_data = round_results[baseline_name]
            for v in variants:
                if v.name in round_results:
                    v_data = round_results[v.name]
                    v_ratio: Dict[str, float] = {}
                    if "A" in measures and "measure_a" in v_data and "measure_a" in base_data:
                        v_ratio["act_steps_ratio"] = v_data["measure_a"]["steps_per_sec"] / max(
                            base_data["measure_a"]["steps_per_sec"], 1e-9
                        )
                    if "B" in measures and "measure_b" in v_data and "measure_b" in base_data:
                        v_ratio["replay_updates_ratio"] = v_data["measure_b"]["updates_per_sec"] / max(
                            base_data["measure_b"]["updates_per_sec"], 1e-9
                        )
                    if "C" in measures and "measure_c" in v_data and "measure_c" in base_data:
                        v_ratio["env_steps_ratio"] = v_data["measure_c"]["env_steps_per_sec"] / max(
                            base_data["measure_c"]["env_steps_per_sec"], 1e-9
                        )
                        v_ratio["updates_ratio"] = v_data["measure_c"]["updates_per_sec"] / max(
                            base_data["measure_c"]["updates_per_sec"], 1e-9
                        )
                    round_ratios[v.name] = v_ratio

        # Print round table
        print(f"\n[Round {round_num}/{args.rounds} Summary Table (Baseline: '{baseline_name}')]")
        _print_round_table(variants, round_results, round_ratios, measures)

        all_rounds_data.append({
            "round": round_num,
            "order": [v.name for v in round_variants],
            "start_time": round_start_time,
            "end_time": round_end_time,
            "loadavg_start": list(round_start_load),
            "loadavg_end": list(round_end_load),
            "results": round_results,
            "errors": round_errors,
            "ratios_vs_baseline": round_ratios,
        })

    # Overall Summary across rounds
    summary_variants: Dict[str, Any] = {}
    for v in variants:
        summary_variants[v.name] = {}
        for m_key in ["env_steps_ratio", "updates_ratio", "act_steps_ratio", "replay_updates_ratio"]:
            r_list = [
                r["ratios_vs_baseline"][v.name][m_key]
                for r in all_rounds_data
                if v.name in r.get("ratios_vs_baseline", {}) and m_key in r["ratios_vs_baseline"][v.name]
            ]
            if r_list:
                summary_variants[v.name][f"{m_key}_list"] = r_list
                summary_variants[v.name][f"{m_key}_median"] = float(statistics.median(r_list))

    print("\n" + "=" * 80)
    print(f"2048 Microbenchmark Final Summary ({args.rounds} round(s), Baseline: '{baseline_name}')")
    print(f"Measures: {','.join(measures)} | Order: {args.order} | Duration: {args.seconds:.1f}s/phase")
    print("=" * 80)
    _print_final_summary(variants, baseline_name, summary_variants, measures)
    print("=" * 80)

    if args.json:
        json_data = {
            "timestamp": datetime.now(TAIPEI_TZ).isoformat(),
            "config": {
                "variants": [
                    {
                        "name": v.name,
                        "expand": v.expand,
                        "threads": v.threads,
                        "tdinf": v.tdinf,
                        "tdfwd": v.tdfwd,
                    }
                    for v in variants
                ],
                "baseline": baseline_name,
                "measures": measures,
                "rounds": args.rounds,
                "order": args.order,
                "seconds": args.seconds,
                "seed": args.seed,
                "interop_threads": args.interop_threads,
            },
            "rounds": all_rounds_data,
            "summary": summary_variants,
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved benchmark results to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
