"""1-thread micro-benchmark comparing onehot_dueling vs onehot_conv2x2 architectures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conv2x2 import OneHotConv2x2DQN
from dueling import OneHotDuelingDQN


def _make_boards(batch_size: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    exps = rng.integers(0, 16, size=(batch_size, 16))
    vals = np.where(exps == 0, 0, 1 << exps).astype(np.float32)
    return torch.from_numpy(vals)


def bench_interleaved_forward_batch1(
    models: list[tuple[str, torch.nn.Module]], x1: torch.Tensor, seconds_per_model: float
) -> dict[str, tuple[float, float]]:
    for _, m in models:
        m.eval()
    elapsed_map = {name: 0.0 for name, _ in models}
    ops_map = {name: 0 for name, _ in models}
    total_budget = seconds_per_model * len(models)

    with torch.no_grad():
        for _, m in models:
            for _ in range(20):
                _ = m(x1)
        wall_start = time.perf_counter()
        while (time.perf_counter() - wall_start) < total_budget:
            for name, m in models:
                t0 = time.perf_counter()
                for _ in range(20):
                    _ = m(x1)
                elapsed_map[name] += time.perf_counter() - t0
                ops_map[name] += 20

    out = {}
    for name, _ in models:
        el = elapsed_map[name]
        ops = ops_map[name]
        out[name] = ((el / ops) * 1000.0, ops / el)
    return out


def bench_interleaved_update_batch(
    models: list[tuple[str, torch.nn.Module]], x_batch: torch.Tensor, seconds_per_model: float
) -> dict[str, tuple[float, float]]:
    optimizers = {}
    for name, m in models:
        m.train()
        optimizers[name] = torch.optim.RMSprop(m.parameters(), lr=1e-4)
    target = torch.zeros((x_batch.shape[0], 4), dtype=torch.float32)

    for name, m in models:
        opt = optimizers[name]
        for _ in range(5):
            opt.zero_grad()
            q = m(x_batch)
            loss = (q - target).square().mean()
            loss.backward()
            opt.step()

    elapsed_map = {name: 0.0 for name, _ in models}
    upd_map = {name: 0 for name, _ in models}
    total_budget = seconds_per_model * len(models)

    wall_start = time.perf_counter()
    while (time.perf_counter() - wall_start) < total_budget:
        for name, m in models:
            opt = optimizers[name]
            t0 = time.perf_counter()
            opt.zero_grad()
            q = m(x_batch)
            loss = (q - target).square().mean()
            loss.backward()
            opt.step()
            elapsed_map[name] += time.perf_counter() - t0
            upd_map[name] += 1

    out = {}
    for name, _ in models:
        el = elapsed_map[name]
        upds = upd_map[name]
        out[name] = ((el / upds) * 1000.0, upds / el)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch intra-op threads (default: 1)")
    parser.add_argument("--seconds", type=float, default=3.0, help="Seconds per model per phase (default: 3.0)")
    parser.add_argument("--batch-size", type=int, default=128, help="Update batch size (default: 128)")
    args = parser.parse_args(argv)

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    x1 = _make_boards(1, seed=10)
    x_batch = _make_boards(args.batch_size, seed=20)

    models: list[tuple[str, torch.nn.Module]] = [
        ("onehot_dueling", OneHotDuelingDQN()),
        ("onehot_conv2x2", OneHotConv2x2DQN()),
    ]

    act_res = bench_interleaved_forward_batch1(models, x1, args.seconds)
    upd_res = bench_interleaved_update_batch(models, x_batch, args.seconds)

    results = []
    for name, model in models:
        params = sum(p.numel() for p in model.parameters())
        act_ms, act_s = act_res[name]
        upd_ms, upd_s = upd_res[name]
        results.append((name, params, act_ms, act_s, upd_ms, upd_s))

    print(f"Threads: {torch.get_num_threads()} | Phase duration: {args.seconds:.1f}s/model | Update batch: {args.batch_size}")
    print("| Architecture | Params | Act B=1 (ms/op) | Act B=1 (ops/s) | Update B=128 (ms/op) | Updates/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, params, act_ms, act_s, upd_ms, upd_s in results:
        print(f"| {name} | {params:,} | {act_ms:.3f} | {act_s:,.1f} | {upd_ms:.3f} | {upd_s:,.2f} |")

    if len(results) == 2:
        d_act_ms, c_act_ms = results[0][2], results[1][2]
        d_upd_s, c_upd_s = results[0][5], results[1][5]
        print(
            f"\nRatio (onehot_conv2x2 vs onehot_dueling): "
            f"params={results[1][1] / results[0][1]:.2f}x, "
            f"act_speedup={d_act_ms / c_act_ms:.2f}x, "
            f"update_speedup={c_upd_s / d_upd_s:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
