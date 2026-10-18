"""S 修訂 3 旗標一致性檢查：用固定 worktree 的 main.py parser 解析 S（旗標關閉，即拿掉 --value-skip l1）與對照的實際訓練指令。
比較兩層：(1) 解析後的 CLI 參數，未指定（None）的欄位先換成 DQNConfig 的預設值再比；
(2) 照 main.train 的同一套規則建出的有效 DQNConfig（dataclass 全欄位）。
只排除 minutes、save_every、save_as、log；value_skip 兩邊都必須是 none。
用法：python flagcheck.py <code_dir> <json: {"S_s101": {"s_argv": [...], "ref_script": "..."} ...}>；輸出 JSON。"""
import json, shlex, sys
from pathlib import Path

EXCLUDE = {"minutes", "save_every", "save_as", "log"}


def train_argv(script):
    toks = shlex.split(script)
    i = next(i for i, t in enumerate(toks) if t.endswith("main.py"))
    out = []
    for t in toks[i + 1:]:
        if t in (">", ">>", "&&", "||", ";", "|", "2>&1") or t.startswith(">"):
            break
        out.append(t)
    return out


FIELD = {"gamma": "gamma", "memory_capacity": "memory_capacity", "per_max_priority": "per_max_priority",
         "reward_mode": "reward_mode", "reward_scale": "reward_scale", "eps_schedule": "epsilon_schedule",
         "eps_start": "epsilon_start", "eps_min": "epsilon_min", "eps_decay": "epsilon_decay",
         "eps_decay_start_episode": "epsilon_decay_start_episode", "eps_decay_steps": "epsilon_decay_steps"}


def norm(ns):
    from config import DQNConfig
    d = DQNConfig()
    ns = dict(ns)
    for arg, field in FIELD.items():
        if ns.get(arg) is None:
            ns[arg] = getattr(d, field)
    return ns


def effective(ns):
    """與 main.train 相同的 config_args 建法（norm 後已無 None，所以全部明確傳入）。"""
    import dataclasses
    from config import DQNConfig
    kw = dict(seed=ns["seed"], arch=ns["arch"], value_skip=ns["value_skip"], n_step=ns["n_step"],
              restart_frac=ns["restart_frac"], restart_pool_size=ns["restart_pool_size"],
              track_replay_age=bool(ns.get("track_replay_age")))
    for arg, field in FIELD.items():
        kw[field] = ns[arg]
    return {k: v for k, v in dataclasses.asdict(DQNConfig(**kw)).items()}


def main():
    code = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(code))
    import main as M  # noqa: E402
    spec = json.loads(Path(sys.argv[2]).read_text())
    res = {}
    for job, s in spec.items():
        s_off = [a for a in s["s_argv"]]
        if "--value-skip" in s_off:
            k = s_off.index("--value-skip")
            del s_off[k:k + 2]
        a = norm(vars(M.parser().parse_args(s_off)))
        ref_argv = train_argv(s["ref_script"])
        b = norm(vars(M.parser().parse_args(ref_argv)))
        keys = sorted((set(a) | set(b)) - EXCLUDE)
        diff = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
        ca, cb = effective(a), effective(b)
        cdiff = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
        res[job] = dict(ok=not diff and not cdiff and a.get("value_skip") == "none" and b.get("value_skip") == "none",
                        diff=diff, config_diff=cdiff, config_fields=len(ca),
                        excluded={k: [a.get(k), b.get(k)] for k in sorted(EXCLUDE)}, compared=len(keys),
                        ref_argv=ref_argv)
    print(json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
