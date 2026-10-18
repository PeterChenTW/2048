"""ROUND3 旗標核對（沿用 S flagcheck 的作法，改用固定 worktree 的 parser 與新欄位）。

對每個 job 檢查兩件事：
 (1) arm 的實際 train argv 和同 seed 的 C argv：CLI namespace 與有效 DQNConfig 只能在 PREREG 指定的欄位上不同，
     而且值必須等於登記值（G: reward_mode=raw, gamma=1.0, value_rescale=h；Y: sym_aug=random；GY 兩者）。
 (2) 同 seed 的 C argv 和對照實際 script（meta.json）：CLI（None 換成 DQNConfig 預設）與有效 DQNConfig 全欄位相同，
     只排除 minutes、save_every、save_as、log。
用法：python r3_flagcheck.py <code_dir> <spec.json>；輸出 JSON。
"""
import dataclasses, json, shlex, sys
from pathlib import Path

EXCLUDE = {"minutes", "save_every", "save_as", "log"}
FIELD = {"gamma": "gamma", "memory_capacity": "memory_capacity", "per_max_priority": "per_max_priority",
         "reward_mode": "reward_mode", "reward_scale": "reward_scale", "eps_schedule": "epsilon_schedule",
         "eps_start": "epsilon_start", "eps_min": "epsilon_min", "eps_decay": "epsilon_decay",
         "eps_decay_start_episode": "epsilon_decay_start_episode", "eps_decay_steps": "epsilon_decay_steps",
         "value_rescale_eps": "value_rescale_eps"}
CLI_OF = {v: k for k, v in FIELD.items()}


def train_argv(script):
    toks = shlex.split(script)
    i = next(i for i, t in enumerate(toks) if t.endswith("main.py"))
    out = []
    for t in toks[i + 1:]:
        if t in (">", ">>", "&&", "||", ";", "|", "2>&1") or t.startswith(">"):
            break
        out.append(t)
    return out


def norm(ns):
    from config import DQNConfig
    d = DQNConfig()
    ns = dict(ns)
    for arg, field in FIELD.items():
        if ns.get(arg) is None:
            ns[arg] = getattr(d, field)
    return ns


def effective(ns):
    """與固定 worktree main.train 相同的 config_args 規則。"""
    from config import DQNConfig
    kw = dict(seed=ns["seed"], arch=ns["arch"], value_skip=ns["value_skip"], n_step=ns["n_step"],
              restart_frac=ns["restart_frac"], restart_pool_size=ns["restart_pool_size"],
              track_replay_age=bool(ns.get("track_replay_age")),
              sym_aug=ns["sym_aug"], value_rescale=ns["value_rescale"])
    for arg, field in FIELD.items():
        kw[field] = ns[arg]
    return dataclasses.asdict(DQNConfig(**kw))


def diff(a, b, skip=()):
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)}


def main():
    code = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(code))
    import main as M  # noqa: E402
    spec = json.loads(Path(sys.argv[2]).read_text())
    parse = lambda argv: norm(vars(M.parser().parse_args(argv)))
    res = {}
    for job, s in spec.items():
        arm, c = parse(s["arm_argv"]), parse(s["c_argv"])
        ref_argv = train_argv(s["ref_script"])
        ref = parse(ref_argv)
        ea, ec, er = effective(arm), effective(c), effective(ref)
        expected = s["expected"]
        arm_cfg = diff(ea, ec)
        arm_cli = diff(arm, c, EXCLUDE)
        bad_arm = {k: v for k, v in arm_cfg.items() if k not in expected or v[0] != expected[k]}
        missing = [k for k in expected if k not in arm_cfg]
        bad_cli = {k: v for k, v in arm_cli.items() if k not in expected and CLI_OF.get(k, k) not in expected}
        c_cli, c_cfg = diff(c, ref, EXCLUDE), diff(ec, er)
        ok = (not bad_arm and not missing and not bad_cli and not c_cli and not c_cfg
              and ea["value_skip"] == "none" and er["value_skip"] == "none")
        res[job] = dict(ok=ok, arm_config_diff=arm_cfg, unexpected_arm_diff=bad_arm, missing_arm_diff=missing,
                        unexpected_arm_cli=bad_cli, control_cli_diff=c_cli, control_config_diff=c_cfg,
                        config_fields=len(ea), ref_argv=ref_argv)
    print(json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
