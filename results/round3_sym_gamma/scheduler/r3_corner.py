"""ROUND3 角落診斷（CORNER_A 定義）：用固定 worktree 的 eval 程式重播 eval 的 100 局，計算角落指標。

只當診斷：本檔與 corner_metrics 都在訓練 worktree 之外，訓練程式不會 import。
用法：python r3_corner.py <code_dir> <corner_dir> <checkpoint> <eval_json> <out_json> [games] [seed_start] [max_steps]
"""
import json, os, random, statistics, sys
from pathlib import Path


def main():
    code, corner_dir, ckpt, eval_json, out = sys.argv[1:6]
    games = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    seed_start = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    max_steps = int(sys.argv[8]) if len(sys.argv) > 8 else 5000
    sys.path.insert(0, corner_dir)
    sys.path.insert(0, code)
    os.chdir(code)
    import numpy as np, torch
    from corner_metrics import game_metrics, spearman
    from eval import load_agent, make_board, ALL_MOVES
    torch.set_num_threads(1)
    agent = load_agent(ckpt)
    rows = []
    for seed in range(seed_start, seed_start + games):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        b = make_board('fast', seed)
        boards, steps = [], 0
        while not b.done and steps < max_steps:
            invalid = b.get_invalid_moves()
            if len(invalid) == 4:
                break
            act = int(agent.greedy_action(b.grid, invalid))
            b.move(ALL_MOVES[act]); steps += 1
            boards.append(np.array(b.grid).reshape(16).astype(int).tolist())
        rows.append(dict(seed=seed, score=int(b.score_v2), max_tile=int(np.max(b.grid)), steps=steps,
                         **game_metrics(boards)))
    off = json.loads(Path(eval_json).read_text()).get("per_game", [])
    offm = {r["seed"]: r for r in off}
    match = bool(off) and all(r["seed"] in offm and all(offm[r["seed"]][k] == r[k] for k in ("score", "max_tile", "steps"))
                              for r in rows)
    keys = ["corner_rate", "corner_rate_256", "edge_rate_256", "center_rate_256", "leaves", "leaves_per100",
            "leaves_256", "longest_streak", "final_mono_snake", "final_mono_rowcol"]
    summ = {}
    for k in keys:
        xs = [r[k] for r in rows if r[k] is not None]
        summ[k] = statistics.mean(xs) if xs else None
    h = [r for r in rows if r["corner_rate_256"] is not None]
    summ["spearman_cr256_score"] = spearman([r["corner_rate_256"] for r in h], [r["score"] for r in h]) if len(h) > 2 else None
    summ["spearman_n"] = len(h)
    tmp = Path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(diagnostic_only=True, checkpoint=ckpt, eval=eval_json, eval_match=match,
                                   summary=summ, games=rows), indent=1))
    tmp.replace(out)


if __name__ == "__main__":
    main()
