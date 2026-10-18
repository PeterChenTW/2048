#!/usr/bin/env python3
"""S 正式訓練自動接續排程器；獨立複製參考邏輯，不 import experiments。

所有真實訓練/eval 只在固定 worktree 執行。S2_DRY_RUN=1 禁止所有 launch、
快照寫入、發佈及里程碑；使用獨立 dry-run state，僅產生本地狀態。
"""
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import statistics
import tempfile
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TAIPEI = ZoneInfo("Asia/Taipei")
EXP = Path("/workspace/rl2048-review/experiments")
RESULTS = "results/round2_skip"
BRANCH = "exp/round2-skip"
GAP = 60.0


def now():
    return datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S")


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fmt(value, spec=".1f"):
    return "待資料" if value is None else format(value, spec)


def snap_label(m):
    return f"t{int(m):04d}"


class Scheduler:
    def __init__(self):
        self.cfg = json.loads(Path(os.environ.get("S2_CONFIG", HERE / "s_config.json")).read_text())
        c = self.cfg
        self.dry_run = os.environ.get("S2_DRY_RUN") == "1"
        self.state_path = Path(os.environ.get("S2_STATE", HERE / "s_state.json"))
        self.runs = Path(os.environ.get("S2_RUNS", c["runs_dir"]))
        self.code = Path(os.environ.get("S2_CODE", c["code_dir"]))
        self.repo = Path(os.environ.get("S2_REPO", c["repo"]))
        self.status_path = Path(os.environ.get("S2_STATUS", HERE / "S_STATUS.md"))
        self.log_path = Path(os.environ.get("S2_LOG", HERE / "s_run.log"))
        self.miles = Path(os.environ.get("S2_MILESTONES", EXP / "milestones.json"))
        # 真實模式必須與 pair_run 使用相同 flock；測試覆寫 miles 後自動隔離鎖。
        default_lock = EXP / ".milestones.lock" if self.miles == EXP / "milestones.json" else self.miles.parent / ".milestones.lock"
        self.miles_lock = Path(os.environ.get("S2_MILESTONES_LOCK", default_lock))
        self.quad_state = Path(os.environ.get("S2_QUAD_STATE", c["start_requires"]["quad_state"]))
        self.publish_enabled = not self.dry_run and os.environ.get("S2_NO_PUBLISH") != "1" and os.environ.get("S2_PUBLISH", "1") != "0"
        self.controls = c["controls"]
        controls_root = os.environ.get("S2_CONTROLS_ROOT")
        if controls_root:
            for control in self.controls.values():
                control["dir"] = str(Path(controls_root) / control["name"])
                if control.get("load"):
                    control["load"]["state"] = str(Path(controls_root) / "pair_state.json")
        self.py = c["python"]
        self.jobs = {j["name"]: j for j in c["jobs"]}
        self.batches = {str(b["id"]): b for b in c["batches"]}

    def log(self, text):
        line = f"[{now()} 台北] {text}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def now_epoch(self):
        return time.time()

    def process_args(self):
        """讀 /proc 的 argv；不把 shell 提示詞或 pgrep 本身誤認為訓練。"""
        result = []
        for p in Path("/proc").iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = [s.decode(errors="replace") for s in (p / "cmdline").read_bytes().split(b"\0") if s]
                if args:
                    result.append(args)
            except OSError:
                pass
        return result

    def machine_busy(self):
        # 工具的隔離 PID namespace 只看得到這次 exec，不能當成全機空閒。
        if not self.process_scope_ok():
            return True
        for args in self.process_args():
            for i, arg in enumerate(args):
                if Path(arg).name == "eval.py":
                    return True
                if Path(arg).name == "main.py" and i + 1 < len(args) and args[i + 1] == "train":
                    return True
        return False

    def process_scope_ok(self):
        try:
            init_name = Path("/proc/1/comm").read_text().strip()
            return not init_name.startswith("codex")
        except OSError:
            return False

    def job_alive(self, name):
        needle = str(self.runs / name) + "/"
        return any(needle in " ".join(args) for args in self.process_args())

    def run(self, args, **kwargs):
        # 管理/驗證指令維持低優先級；訓練/eval 的 nice 由 launch_command 控制。
        return subprocess.run(["nice", "-n", "19", *map(str, args)], capture_output=True, text=True,
                              timeout=kwargs.pop("timeout", 45), **kwargs)

    def eval_args(self, checkpoint, output):
        e = self.cfg["eval"]
        return ["timeout", "--signal=TERM", "--kill-after=30s", str(e["quota_min"]) + "m", self.py,
                "eval.py", "--checkpoint", str(checkpoint), "--policy", e["policy"], "--games", str(e["games"]),
                "--seed-start", str(e["seed_start"]), "--board", e["board"], "--max-steps", str(e["max_steps"]), "--json", str(output)]

    def script(self, job):
        d = self.runs / job["name"]
        c = self.cfg
        train = [self.py, "main.py", "train", *c["common"], *c["arm_flags"][job["arm"]], "--seed", str(job["seed"]),
                 "--threads", str(c["threads"]), "--minutes", str(c["minutes"]), "--save-every", "1",
                 "--save-as", str(d / "ckpt"), "--log", str(d / "train.csv")]
        q = lambda p: shlex.quote(str(p))
        return (f"{shlex.join(train)} > {q(d / 'train.out')} 2>&1 && date +%s > {q(d / 'eval_start')} && "
                f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {shlex.join(self.eval_args(d / 'ckpt', d / 'eval.json'))} "
                f"> {q(d / 'eval.out')} 2>&1 && date +%s > {q(d / 'eval_end')}")

    def launch_command(self, job):
        batch = self.batches[str(job["batch"])]
        return ["setsid", "nohup", "taskset", "-c", batch["cpus"], "nice", "-n", str(self.cfg["nice"]), "bash", "-c", self.script(job)]

    def spawn(self, args, output, threads=1):
        if self.dry_run:
            raise RuntimeError("dry-run 禁止 launch")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        with Path(output).open("a") as out:
            return subprocess.Popen(args, cwd=self.code, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def milestone(self, text):
        if self.dry_run:
            return
        self.miles_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.miles_lock.open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            ms = load(self.miles, [])
            if not isinstance(ms, list):
                raise ValueError("里程碑不是陣列，停止覆寫")
            if not any(m.get("text") == text for m in ms):
                ms.append(dict(time=datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M"), text=text, reported=False))
                save_json(self.miles, ms)
                self.log("里程碑：" + text)

    def alert_once(self, st, job, text):
        if self.dry_run:
            return
        kind = text.split("：")[0]
        key = f"{job}|{kind}"
        seen = st.setdefault("alerted", [])
        if key in seen:
            return
        # 先查共用檔，避免 state 寫入前當機導致數字改變後再發一次。
        prefix = f"round2 S {job} 異常 — {kind}："
        self.miles_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.miles_lock.open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            ms = load(self.miles, [])
            if not isinstance(ms, list):
                raise ValueError("里程碑不是陣列，停止覆寫")
            if not any(m.get("text", "").startswith(prefix) for m in ms):
                ms.append(dict(time=datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M"),
                               text=f"round2 S {job} 異常 — {text}", reported=False))
                save_json(self.miles, ms)
        seen.append(key)

    @staticmethod
    def valid_eval(ev):
        try:
            values = [ev["mean"], ev["median"], ev["reach_rates"]["1024"], ev["reach_rates"]["2048"]]
            return (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
                    and min(values[:2]) >= 0 and all(0 <= v <= 1 for v in values[2:]))
        except (KeyError, TypeError):
            return False

    def control_ready(self, seed):
        control = self.controls[str(seed)]
        d = Path(control["dir"])
        return self.valid_eval(load(d / control["eval"])) and all((d / f).is_file() for f in control["require"])

    def flag_spec(self):
        spec, waiting = {}, {}
        for seed, control in self.controls.items():
            name = f"S_s{seed}"
            meta = load(Path(control["dir"]) / "meta.json")
            ref = meta.get("script") if isinstance(meta, dict) else None
            if not isinstance(ref, str) or not ref.strip():
                waiting[seed] = dict(ok=False, status="待對照 meta", diff={}, config_diff={})
                continue
            # 從實際 script 取 train argv；parser 只吃 train 子命令起的參數。
            tokens = shlex.split(self.script(self.jobs[name]))
            start = tokens.index("main.py") + 1
            end = tokens.index(">", start)
            spec[name] = dict(s_argv=tokens[start:end], ref_script=ref)
        return spec, waiting

    @staticmethod
    def flag_summary(result):
        parts = []
        for label in ("diff", "config_diff"):
            for key, values in (result.get(label) or {}).items():
                parts.append(f"{label}.{key}={values}")
        return "；".join(parts) or result.get("error", "檢查未通過")

    def check_flags(self, st, force=False):
        t = self.now_epoch()
        cached = st.get("flags", {})
        if not force and 0 <= t - cached.get("checked_epoch", -1e30) < 600:
            results = cached["jobs"]
        else:
            spec, results = self.flag_spec()
            if spec:
                try:
                    # 暫存規格不落 experiments；dry-run 也會唯讀解析 parser/meta。
                    with tempfile.TemporaryDirectory(prefix="s_flags_") as temp:
                        path = Path(temp) / "spec.json"
                        save_json(path, spec)
                        proc = self.run([self.py, HERE / "flagcheck.py", self.code, path], timeout=120,
                                        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                                 OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
                    if proc.returncode:
                        raise ValueError("flagcheck 執行失敗")
                    checked = json.loads(proc.stdout)
                    for name in spec:
                        result = checked[name]
                        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                            raise ValueError("flagcheck 結果格式錯誤")
                        results[str(self.jobs[name]["seed"])] = dict(result, status="一致" if result["ok"] else "不一致")
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                    for name in spec:
                        results[str(self.jobs[name]["seed"])] = dict(ok=False, status="不一致", diff={},
                                                                    config_diff={}, error="flagcheck 執行或結果無效")
            st["flags"] = dict(checked_epoch=t, jobs=results)
        for seed, result in results.items():
            if result["status"] == "不一致":
                self.alert_once(st, f"S_s{seed}", f"旗標不一致：{seed} {self.flag_summary(result)}")
        return results

    def ready(self, st):
        t = self.now_epoch()
        req = self.cfg["start_requires"]
        busy = self.machine_busy()
        idle = st.setdefault("idle", {"count": 0})
        if busy:
            idle.update(count=0, last_epoch=None)
        elif idle.get("last_epoch") is None or t - idle["last_epoch"] > 3 * req["idle_interval_s"]:
            idle.update(count=1, last_epoch=t)
        elif t - idle["last_epoch"] >= req["idle_interval_s"]:
            idle.update(count=min(req["idle_checks"], idle["count"] + 1), last_epoch=t)
        w = dict(quad_done=bool((load(self.quad_state, {}) or {}).get("done")), busy=busy,
                 time_ok=t >= datetime.fromisoformat(req["not_before"]).timestamp(),
                 idle_count=idle["count"], idle_ok=idle["count"] >= req["idle_checks"], checked=now())
        w["control_ready"] = all(self.control_ready(seed) for seed in req["control_ready"])
        w["flags"] = self.check_flags(st)
        w["flags_ok"] = all(w["flags"].get(seed, {}).get("ok", False) for seed in self.controls)
        w["process_scope_ok"] = self.process_scope_ok()
        st["wait"] = w
        return w

    def launch(self, batch, st):
        if self.dry_run:
            return
        commit_result = self.run(["git", "-C", self.code, "rev-parse", "HEAD"])
        commit = commit_result.stdout.strip()
        if commit_result.returncode or commit != self.cfg["expected_commit"]:
            raise RuntimeError("固定程式 commit 不符，禁止開跑")
        flags = self.check_flags(st, force=True)
        st.setdefault("wait", {}).update(flags=flags, flags_ok=all(r["ok"] for r in flags.values()))
        if not all(flags.get(seed, {}).get("ok", False) for seed in self.controls):
            raise RuntimeError("旗標一致性未通過，禁止開跑")
        if not all(self.control_ready(seed) for seed in self.cfg["start_requires"]["control_ready"]):
            raise RuntimeError("對照 t180 尚未完成，禁止開跑")
        # 不覆蓋任何既有 job，也不對中斷的 job 自動重跑。
        if any((self.runs / n).exists() for n in batch["jobs"]):
            raise RuntimeError("批次 run 目錄已存在，禁止自動重跑")
        b = st.setdefault("batches", {}).setdefault(str(batch["id"]), {})
        b.update(started=now(), load_sum=0.0, load_n=0, busy_sum=0.0, busy_n=0, pids={})
        st.setdefault("started", now())
        save_json(self.state_path, st)  # 先保留批次；部分啟動失敗也不自動重跑。
        for name in batch["jobs"]:
            job = self.jobs[name]
            d = self.runs / name
            d.mkdir(parents=True, exist_ok=False)
            meta = dict(job=job, commit=commit, prereg=self.cfg["prereg"], cpus=batch["cpus"], nice=self.cfg["nice"],
                        threads=self.cfg["threads"], minutes=self.cfg["minutes"], snapshots=self.cfg["snapshots"],
                        started=now() + " Taipei", script=self.script(job))
            save_json(d / "meta.json", meta)
            p = self.spawn(self.launch_command(job), d / "launch.out", self.cfg["threads"])
            meta["launcher_pid"] = p.pid
            save_json(d / "meta.json", meta)
            b["pids"][name] = p.pid
            st.setdefault("pids", {})[name] = p.pid
            save_json(self.state_path, st)
        self.log(f"批次 {batch['id']} 已啟動：{batch['jobs']}")
        st.pop("launch_error", None)

    def read_metrics(self, d, eval_path, require):
        ev, tr = load(d / eval_path), load(d / "train.json")
        if not (self.valid_eval(ev) and isinstance(tr, dict) and tr
                and all((d / f).is_file() for f in require)):
            return None
        # 中斷產生的 JSON 不是完整 t180 資料。
        if tr.get("interrupted") or tr.get("stopped_reason") == "nonfinite_replay":
            return None
        try:
            m = dict(mean=ev["mean"], median=ev["median"], r1024=ev["reach_rates"]["1024"], r2048=ev["reach_rates"]["2048"],
                     updates=tr.get("updates"), env_steps=tr.get("env_steps"), rr=tr.get("replay_ratio_actual"),
                     episodes=tr.get("episodes"), nan=tr.get("nan_count"), active_s=tr.get("active_s"),
                     ups=tr["updates"] / tr["active_s"] if tr.get("active_s") and tr.get("updates") is not None else None,
                     eval_min=ev.get("elapsed_seconds", 0) / 60)
            # 非有限診斷值不寫入 state，也不能冒充完整結果。
            if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in m.values()):
                return None
            try:
                m["eval_min"] = (float((d / "eval_end").read_text()) - float((d / "eval_start").read_text())) / 60
            except (ValueError, OSError):
                pass
            return m
        except (KeyError, TypeError, ZeroDivisionError):
            return None

    def metrics(self, name, st=None):
        m = self.read_metrics(self.runs / name, "eval.json", ["eval_end"])
        if m:
            b = (st or {}).get("batches", {}).get("1", {})
            m.update(load1=b.get("mean_load1"), busy=b.get("mean_busy_cores"))
        return m

    def control_metrics(self, seed):
        control = self.controls[str(seed)]
        d = Path(control["dir"])
        m = self.read_metrics(d, control["eval"], control["require"])
        if m:
            cfg = control.get("load")
            state = load(cfg["state"], {}) if cfg else {}
            pairs = state.get("pairs", {}) if isinstance(state, dict) else {}
            b = pairs.get(cfg["pair"], {}) if cfg and isinstance(pairs, dict) else {}
            b = b if isinstance(b, dict) else {}
            def average(mean, total, count):
                value = b.get(mean)
                if value is None:
                    numerator, denominator = b.get(total), b.get(count)
                    if (isinstance(numerator, (int, float)) and isinstance(denominator, (int, float))
                            and denominator > 0):
                        value = numerator / denominator
                return value if isinstance(value, (int, float)) and math.isfinite(value) else None
            m.update(load1=average("mean_load1", "load_sum", "load_n"),
                     busy=average("mean_busy_cores", "busy_sum", "busy_n"), load_note=control.get("load_note"))
        return m

    def verdicts(self, st):
        rows = []
        R = self.cfg["rules"]
        for seed in (101, 202, 303):
            s, c = self.metrics(f"S_s{seed}", st), self.control_metrics(seed)
            if not (s and c):
                rows.append(dict(seed=seed, pending=True))
                continue
            ratio = s["mean"] / c["mean"] if c["mean"] > 0 else None
            dpp = 100 * (s["r1024"] - c["r1024"])
            div = lambda a, b: a / b if a is not None and b is not None and b > 0 else None
            # 修訂 2：三個 S/C 比值只作必報診斷，不影響採用與否。
            diag = dict(updates=div(s["updates"], c["updates"]), env_steps=div(s["env_steps"], c["env_steps"]),
                        ups=div(s["ups"], c["ups"]))
            score_ok = ratio is not None and ratio >= R["ratio"] and dpp >= R["pp_1024"] - 1e-9
            rows.append(dict(seed=seed, pending=False, S=s, C=c, ratio=ratio, dpp=dpp, diag=diag,
                             score_ok=score_ok, ok=score_ok,
                             abs_S=s["mean"] > R["abs_mean"] and s["r1024"] >= R["abs_r1024"],
                             abs_C=c["mean"] > R["abs_mean"] and c["r1024"] >= R["abs_r1024"]))
        pending = any(r["pending"] for r in rows)
        paired = [dict(seed=r["seed"], mean_diff=r["S"]["mean"] - r["C"]["mean"], ratio=r["ratio"], dpp=r["dpp"])
                  for r in rows if not r["pending"]]
        spread = {}
        for key in ("mean_diff", "ratio", "dpp"):
            values = [r[key] for r in paired if r[key] is not None]
            spread[key] = dict(sd=statistics.stdev(values), min=min(values), max=max(values)) if len(values) == 3 else None
        return dict(rows=rows, paired=paired, spread=spread, pending=pending, n_pass=sum(r.get("ok", False) for r in rows),
                    adopt=not pending and all(r["ok"] for r in rows),
                    absolute=not pending and all(r["abs_S"] for r in rows))

    def anomalies(self, job):
        d = self.runs / job["name"]
        if not (d / "meta.json").exists():
            return []
        alive = self.job_alive(job["name"])
        a = []
        csvp = d / "train.csv"
        if alive and not (d / "train.json").exists() and csvp.exists():
            age = (self.now_epoch() - csvp.stat().st_mtime) / 60
            if age > self.cfg["rules"]["stall_min"]:
                a.append(f"卡住：train.csv {age:.0f} 分鐘沒更新")
        if not alive and not (d / "eval_end").exists():
            a.append("中斷：程序不在、也沒有完成 eval（不自動重跑，請判斷）")
        rows = []
        if csvp.exists():
            with csvp.open() as f:
                rows = list(csv.DictReader(f))
        def finite(v):
            try:
                return math.isfinite(float(v))
            except (ValueError, TypeError):
                return False
        bad = [r.get("episode", "?") for r in rows if r.get("nan_count") not in (None, "", "0", "0.0") or
               not all(finite(r[k]) for k in ("loss", "q_mean", "td_abs_mean") if r.get(k) not in (None, ""))]
        tr = load(d / "train.json", {}) or {}
        out = ""
        for f in ("train.out", "eval.out"):
            if (d / f).exists():
                with (d / f).open("rb") as h:
                    h.seek(max(0, (d / f).stat().st_size - 20000))
                    out += h.read().decode(errors="replace")
        if bad or tr.get("nan_count", 0) or "Traceback" in out or "FloatingPointError" in out:
            a.append(f"NaN／例外：episodes {bad[:5]}；train 非有限值 {tr.get('nan_count', 0)}")
        if tr.get("interrupted"):
            a.append("中斷：train.json 記錄 interrupted（不自動重跑）")
        if len(rows) >= 400:
            sc = [float(r["score_v2"]) for r in rows if finite(r.get("score_v2"))]
            roll = [sum(sc[i - 200:i]) / 200 for i in range(200, len(sc) + 1, 50)]
            if roll and roll[-1] < 0.75 * max(roll):
                a.append(f"訓練分數警示：最近 200 局平均 {roll[-1]:,.0f}，低於最高 {max(roll):,.0f} 的 75%")
        return a

    def cpu_times(self):
        lines = Path("/proc/stat").read_text().splitlines()
        v = [float(x) for x in lines[0].split()[1:9]]  # guest 已包含在 user/nice，不重複加。
        return sum(v), v[3] + v[4], sum(1 for line in lines if line.startswith("cpu") and line[3:4].isdigit())

    def sample_load(self, b, training):
        tot, idle, ncpu = self.cpu_times()
        prev = b.get("cpu_prev")
        b["cpu_prev"] = [tot, idle] if training else None
        if training:
            b["load_sum"] = b.get("load_sum", 0) + os.getloadavg()[0]
            b["load_n"] = b.get("load_n", 0) + 1
            if prev and tot > prev[0]:
                busy = max(0, min(ncpu, ncpu * (1 - (idle - prev[1]) / (tot - prev[0]))))
                b["busy_sum"] = b.get("busy_sum", 0) + busy
                b["busy_n"] = b.get("busy_n", 0) + 1
            b["machine_cores"] = ncpu
        b["mean_load1"] = b.get("load_sum", 0) / b["load_n"] if b.get("load_n") else None
        b["mean_busy_cores"] = b.get("busy_sum", 0) / b["busy_n"] if b.get("busy_n") else None

    def snap_tick(self, js, d, training, batch):
        # 與 pair_run 相同的 monotonic／60秒 gap 累積規則，重新開機不沿用舊時鐘。
        mono = time.monotonic()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if training and js.get("boot_id") == boot and js.get("last_mono") is not None:
            dt = mono - js["last_mono"]
            if 0 < dt <= GAP:
                js["active_s"] = js.get("active_s", 0) + dt
        js.update(last_mono=mono if training else None, boot_id=boot)
        if self.dry_run:
            return
        taken = js.setdefault("taken", [])
        for m in self.cfg["snapshots"]:
            if not (training and m not in taken and js.get("active_s", 0) >= 60 * m and (d / "ckpt").exists()):
                continue
            (d / "snapshots").mkdir(exist_ok=True)
            dst = d / "snapshots" / f"{snap_label(m)}.pt"
            valid = False
            for _ in range(5):
                shutil.copyfile(d / "ckpt", dst)
                r = self.run(["taskset", "-c", batch["cpus"], self.py, "-c",
                              f"import torch;torch.load({str(dst)!r},weights_only=False)"], timeout=15,
                             env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                if r.returncode == 0:
                    valid = True
                    break
                time.sleep(2)
            if not valid:
                dst.unlink(missing_ok=True)
                self.log(f"{d.name} {snap_label(m)} 快照驗證失敗，下次 tick 重試")
                continue
            taken.append(m)
            js.setdefault("meta", {})[str(m)] = dict(active_s=js.get("active_s", 0), taken=now())
        for m in taken:
            out = d / "evals" / f"{snap_label(m)}.json"
            key = str(m)
            if out.exists() or any(str(out) in " ".join(a) for a in self.process_args()):
                continue
            tries = js.setdefault("eval_tries", {})
            if tries.get(key, 0) >= 3:
                continue
            (d / "evals").mkdir(exist_ok=True)
            tries[key] = tries.get(key, 0) + 1
            args = ["setsid", "nohup", "taskset", "-c", batch["cpus"], "nice", "-n", str(self.cfg["nice"]),
                    *self.eval_args(d / "snapshots" / f"{snap_label(m)}.pt", out)]
            self.spawn(args, d / "evals" / f"{snap_label(m)}.out")
            self.log(f"{d.name} 探索性 {snap_label(m)} eval 已啟動")

    def finish_text(self, st):
        vs = self.verdicts(st)
        texts = []
        for v in vs["rows"]:
            if v["pending"]:
                texts.append(f"s{v['seed']} 待資料（不通過）")
                continue
            texts.append(f"s{v['seed']} ×{fmt(v['ratio'], '.2f')}/{v['dpp']:+.1f}pp 分數{'過' if v['score_ok'] else '不過'}"
                         f"（診斷 S/C：updates ×{fmt(v['diag']['updates'], '.2f')}、env_steps ×{fmt(v['diag']['env_steps'], '.2f')}、updates/s ×{fmt(v['diag']['ups'], '.2f')}）")
        spread = "；".join(f"{k} SD {fmt(v['sd'] if v else None, '.3f')}" for k, v in vs["spread"].items())
        b = st.get("batches", {}).get("1", {})
        return (f"round2 S 完成：{'；'.join(texts)} → {'採用 S' if vs['adopt'] else '不採用'}（{vs['n_pass']}/3）；"
                f"配對差：{spread}；S 平均 load1 {fmt(b.get('mean_load1'))}／忙碌核 {fmt(b.get('mean_busy_cores'))}；"
                "對照不同時段、負載不同，只並列不校正")

    def snapshots_finished(self, st):
        """已取得快照的 eval 需收尾後才能結束監控及發佈。"""
        finished = True
        for b in st.get("batches", {}).values():
            for name, js in b.get("snap", {}).items():
                for mark in js.get("taken", []):
                    out = self.runs / name / "evals" / f"{snap_label(mark)}.json"
                    if load(out):
                        continue
                    running = any(str(out) in " ".join(a) for a in self.process_args())
                    if running or js.get("eval_tries", {}).get(str(mark), 0) < 3:
                        finished = False
                    else:
                        self.alert_once(st, name, f"快照失敗：{snap_label(mark)} eval 已嘗試3次；不影響t180判定")
        return finished

    def tick(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.with_suffix(".lock").open("a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return load(self.state_path, {})
            st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"batches": {}, "pids": {}}
            if not isinstance(st, dict) or not isinstance(st.get("batches"), dict):
                raise ValueError("state 格式損壞，停止以免重複開跑")
            alerts = []
            w = self.ready(st)
            for key, batch in self.batches.items():
                b = st["batches"].get(key)
                if b is None:
                    ok = (w["quad_done"] and w["time_ok"] and w["idle_ok"] and not w["busy"]
                          and w["control_ready"] and w["flags_ok"])
                    if ok and not self.dry_run:
                        try:
                            self.launch(batch, st)
                        except Exception as exc:
                            text = f"啟動失敗：{type(exc).__name__}（檢查固定 commit、run 目錄與本地 log；不自動重跑已啟動 job）"
                            self.log(text)
                            self.alert_once(st, f"批次{key}", text)
                            st["launch_error"] = text
                            alerts.append((f"批次{key}", text))
                    break  # 每個 tick 最多一批；新程序會佔用機器。
                active = {n: self.job_alive(n) and not (self.runs / n / "train.json").exists() for n in batch["jobs"]}
                self.sample_load(b, any(active.values()))
                for name in batch["jobs"]:
                    if not self.dry_run:
                        self.snap_tick(b.setdefault("snap", {}).setdefault(name, {}), self.runs / name, active[name], batch)
                    for text in self.anomalies(self.jobs[name]):
                        alerts.append((name, text))
                        self.alert_once(st, name, text)
                if all(self.metrics(n) for n in batch["jobs"]):
                    b.setdefault("done", now())
            if all(st["batches"].get(k, {}).get("done") for k in self.batches) and not self.verdicts(st)["pending"] and self.snapshots_finished(st):
                if not st.get("done"):
                    text = st.setdefault("completion_text", self.finish_text(st))
                    self.milestone(text)
                    st["done"] = now()
            output_state = self.state_path.with_name(self.state_path.stem + ".dry_run.json") if self.dry_run else self.state_path
            save_json(output_state, st)
            self.write_fragment(st, alerts)
            if self.publish_enabled:
                self.publish(st)
                save_json(self.state_path, st)
            return st

    def write_fragment(self, st, alerts):
        L = [f"# 第二輪 S 正式訓練狀態（更新：{now()} 台北）", "", f"事前登記：{self.cfg['prereg']}。",
             "只有一批：S_s101、S_s202、S_s303 共用 CPU 0–7；每 job 2 threads、nice 10、180 分鐘 active。",
             "S = C 加 `--value-skip l1`；對照使用既有同 seed t180 軌跡；t90／120／150 僅供探索。", ""]
        if self.dry_run:
            L.append("**DRY-RUN：只寫本地狀態，禁止 launch／快照／publish／里程碑。**")
        w = st.get("wait", {})
        yn = lambda v: "是" if v else "否"
        if not st.get("started"):
            L += ["狀態：**等待中**", f"- R quad 已完成（只讀 done）：{yn(w.get('quad_done'))}",
                  f"- 機器仍有 main.py train／eval.py：{yn(w.get('busy')) if w.get('process_scope_ok') else '無法確認（保守視為忙碌）'}",
                  f"- 能確認全機程序：{yn(w.get('process_scope_ok'))}（隔離程序空間視為忙碌，保持等待）",
                  f"- 連續兩次空閒（間隔 ≥30秒）：{w.get('idle_count', 0)}/2；滿足：{yn(w.get('idle_ok'))}",
                  f"- 台北時間 ≥ 2026-10-18 20:30：{yn(w.get('time_ok'))}",
                  f"- C_s303 t180 eval 已存在（有效 JSON 且 eval_end 存在）：{yn(w.get('control_ready'))}"]
        else:
            L.append("狀態：**" + ("全部完成" if st.get("done") else "進行中") + "**")
        flags = w.get("flags", st.get("flags", {}).get("jobs", {}))
        L += ["- 旗標一致性："]
        for seed in self.controls:
            result = flags.get(seed, {})
            status = result.get("status", "待核對")
            if status == "不一致":
                status += "：" + self.flag_summary(result)
            L.append(f"  - s{seed}：{status}")
        if not st.get("started"):
            L.append("上述條件全部滿足且三個 seed 旗標一致，才會開跑批次 1；commit 核對後再次強制檢查旗標。")

        def table(headers, rows):
            # 同一組欄位生成標題／分隔／資料列，避免缺資料時表格錯位。
            L.extend(["", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
            for row in rows:
                assert len(row) == len(headers)
                cells = [str(x).replace("|", "\\|").replace("\n", " ") for x in row]
                L.append("| " + " | ".join(cells) + " |")

        batch_rows = []
        for key, batch in self.batches.items():
            b = st.get("batches", {}).get(key, {})
            status = "完成" if b.get("done") else "進行中" if b.get("started") else "等待中"
            batch_rows.append([key, batch["cpus"], status, fmt(b.get("mean_load1")), fmt(b.get("mean_busy_cores"))])
        table(["批次", "CPU", "狀態", "平均 load1", "平均忙碌核（整機）"], batch_rows)
        for name in self.jobs:
            d = self.runs / name
            alive = self.job_alive(name)
            status = ("完成" if self.metrics(name) else "eval 中" if (d / "train.json").exists() and alive
                      else "訓練中" if alive else "中斷／待判斷" if (d / "meta.json").exists() else "未開跑")
            L.append(f"- {name}：{status}")
        L += ["", "判定（S_PREREG 修訂 2＋修訂 3）：每 seed mean 比值 ≥1.20 且 Δ≥1024 ≥+10pp（等時間分數門）；三個 seed 全過才採用。缺資料不視為通過。",
              "S/C 的 updates、env_steps、updates/s 比值為必報診斷，不影響判定。絕對目標另列：mean >15,000 且 ≥1024 ≥80%；三種子平均僅供探索。"]
        vs = self.verdicts(st)
        verdict_rows, side_rows = [], []
        yes = lambda x: "過" if x else "不過"
        for v in vs["rows"]:
            seed = v["seed"]
            label = self.controls[str(seed)]["name"] + " t180"
            if v["pending"]:
                verdict_rows.append([seed, label, "待資料", "待資料", "待資料", "待資料", "不通過（待資料）",
                                     "待資料", "待資料", "待資料", "待資料"])
            else:
                s, c = v["S"], v["C"]
                verdict_rows.append([seed, label, f"{c['mean']:,.0f}／{c['r1024']:.0%}", f"{s['mean']:,.0f}／{s['r1024']:.0%}",
                                     "×" + fmt(v["ratio"], ".2f"), f"{v['dpp']:+.1f}pp", yes(v["ok"]),
                                     "×" + fmt(v["diag"]["updates"], ".3f"), "×" + fmt(v["diag"]["env_steps"], ".3f"),
                                     "×" + fmt(v["diag"]["ups"], ".3f"), yes(v["abs_C"]) + "／" + yes(v["abs_S"])])
            for name, m in ((f"S_s{seed}", self.metrics(f"S_s{seed}", st)), (label, self.control_metrics(seed))):
                if m:
                    note = m.get("load_note")
                    side_rows.append([seed, name, note or fmt(m["load1"]), note or fmt(m["busy"]),
                                      fmt(m["updates"], ","), fmt(m["env_steps"], ","), fmt(m["ups"]),
                                      fmt(m["episodes"], ","), fmt(m["active_s"])])
                else:
                    note = self.controls[str(seed)].get("load_note") if name == label else None
                    side_rows.append([seed, name, note or "待資料", note or "待資料", *(["待資料"] * 5)])
        table(["seed", "對照（預定終點）", "C mean／≥1024", "S mean／≥1024", "S/C", "Δ≥1024", "分數門＝判定",
               "診斷 S/C updates", "診斷 S/C env_steps", "診斷 S/C updates/s", "絕對目標 C／S"], verdict_rows)
        L += ["", "S 與對照並列："]
        table(["seed", "job", "load1", "忙碌核", "updates", "env_steps", "updates/s", "episodes", "active_s"], side_rows)
        L += ["", "每 seed 配對差（S−C）："]
        paired = {v["seed"]: v for v in vs["paired"]}
        table(["seed", "mean 差（S−C）", "mean 比值（S/C）", "≥1024 差（pp）"],
              [[seed, fmt(paired.get(seed, {}).get("mean_diff"), "+.1f"),
                fmt(paired.get(seed, {}).get("ratio"), ".3f"), fmt(paired.get(seed, {}).get("dpp"), "+.1f")]
               for seed in (101, 202, 303)])
        table(["跨 seed 統計", "樣本 SD（n−1）", "min", "max"],
              [[k, *[fmt(v[x] if v else None, ".3f") for x in ("sd", "min", "max")]] for k, v in vs["spread"].items()])
        L += ["", "3 個 seed＝3 條確定性軌跡（n=3），seed 間變異是唯一的變異估計；沒有 seed 內重複。跨 seed SD／範圍只在三個值都有時列出。"]
        if not vs["pending"]:
            avg = {arm: {k: sum(v[arm][k] for v in vs["rows"]) / 3 for k in ("mean", "r1024")} for arm in ("C", "S")}
            L += ["", f"三種子平均（探索性，不作判定）：C {avg['C']['mean']:,.0f}／{avg['C']['r1024']:.1%}；S {avg['S']['mean']:,.0f}／{avg['S']['r1024']:.1%}。",
                  f"**正式判定：{'採用 S' if vs['adopt'] else '不採用'}（{vs['n_pass']}/3 seed 通過）；絕對目標：{'穩健達標' if vs['absolute'] else '尚未穩健達標'}。**"]
        other = []
        for seed in self.controls:
            for name, m in ((f"S_s{seed}", self.metrics(f"S_s{seed}", st)),
                            (self.controls[seed]["name"], self.control_metrics(seed))):
                if m:
                    other.append([name, fmt(m["median"], ",.0f"), fmt(m["r2048"], ".1%"), fmt(m["rr"], ".3f"), fmt(m["eval_min"]), fmt(m["nan"], ",")])
        table(["job", "median", "≥2048", "實際 replay ratio", "eval 分鐘", "非有限值數"], other)
        L += ["", "探索性快照（只列 S，不參與判定；正式為 t180）："]
        snapshots = []
        for name in self.jobs:
            cells = []
            for mark in self.cfg["snapshots"]:
                ev = load(self.runs / name / "evals" / f"{snap_label(mark)}.json")
                cells.append(f"{ev['mean']:,.0f}／{ev['reach_rates']['1024']:.0%}" if self.valid_eval(ev) else "待資料")
            snapshots.append([name, *cells])
        table(["job", *[f"t{m}" for m in self.cfg["snapshots"]]], snapshots)
        L += ["", "結果只能歸因到「整套 S 設定在 180 分鐘 active 預算下（含負載條件）」；env steps、updates、ε／target 節奏也會隨之改變。",
              "對照不同時段、負載不同，只並列不校正。",
              "啟動異常：" + st["launch_error"] if st.get("launch_error") else "",
              "目前異常：" + ("；".join(f"{n}：{a}" for n, a in alerts) if alerts else "無"), ""]
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text("\n".join(L))

    def control_files(self):
        for control in self.controls.values():
            d = Path(control["dir"])
            for relative in (control["eval"], "train.json", "meta.json"):
                source = d / relative
                if source.is_file():
                    yield source, Path("controls") / control["name"] / relative

    def publication_sig(self, st):
        events = {"started": st.get("started"), "done": st.get("done"), "alerts": st.get("alerted", []),
                  "batches": {k: {x: b.get(x) for x in ("started", "done")} for k, b in st.get("batches", {}).items()}}
        files = {}
        for name in self.jobs:
            d = self.runs / name
            for f in [d / n for n in ("meta.json", "train.json", "eval.json")] + sorted((d / "evals").glob("*.json")):
                if f.is_file():
                    files[str(f.relative_to(self.runs))] = hashlib.sha256(f.read_bytes()).hexdigest()
        for source, relative in self.control_files():
            files[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        events["files"] = files
        return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()

    def publish(self, st):
        if not self.publish_enabled or self.dry_run or not st.get("started"):
            return False
        sig = self.publication_sig(st)
        if st.get("published_sig") == sig and not st.get("publish_pending"):
            return True
        st["publish_pending"] = True
        # 鎖只能放 skip-run；絕不寫 experiments/.publish.lock。
        with self.state_path.with_suffix(".publish.lock").open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                git = lambda *a: self.run(["git", "-C", self.repo, *a])
                gd = git("rev-parse", "--absolute-git-dir")
                if gd.returncode or (Path(gd.stdout.strip()) / "index.lock").exists():
                    return False
                branch = git("branch", "--show-current")
                if branch.returncode or branch.stdout.strip() != BRANCH:
                    self.log("發佈延後：結果 worktree 分支不符")
                    return False
                dest = self.repo / RESULTS
                dest.mkdir(parents=True, exist_ok=True)
                self.write_fragment(st, []) if not self.status_path.exists() else None
                shutil.copyfile(self.status_path, dest / "S_RESULTS.md")
                for name in self.jobs:
                    d = self.runs / name
                    for f in [d / n for n in ("meta.json", "train.json", "eval.json")] + sorted((d / "evals").glob("*.json")):
                        if f.exists():
                            target = dest / "runs" / name / f.relative_to(d)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(f, target)
                for source, relative in self.control_files():
                    target = dest / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                if (Path(gd.stdout.strip()) / "index.lock").exists():
                    return False
                add = git("add", RESULTS)
                if add.returncode:
                    return False
                # commit -- <path> 不夾帶別人的 staged 檔；只檢查指定目錄。
                diff = git("diff", "--cached", "--quiet", "--", RESULTS)
                if diff.returncode not in (0, 1):
                    return False
                if diff.returncode == 1:
                    if git("commit", "-m", "第二輪 S：更新正式訓練結果", "--", RESULTS).returncode:
                        return False
                # 即使沒有新 diff 仍要重試上次失敗的 push。不 merge、不 force。
                rc = git("push", "origin", BRANCH).returncode
                st["publish_pending"] = rc != 0
                if rc == 0:
                    st["published"] = now()
                    st["published_sig"] = sig
                self.log("結果發佈成功" if rc == 0 else "push 未成功，下次 tick 重試（不 force、不 merge）")
                return rc == 0
            except (OSError, subprocess.SubprocessError):
                st["publish_pending"] = True
                self.log("發佈失敗，下次 tick 重試")
                return False

    def loop(self):
        if self.dry_run:
            self.tick()
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.with_suffix(".loop.lock").open("a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("同一 state 的 loop 已在執行")
            while True:
                st = self.tick()
                if st.get("done") and (not self.publish_enabled or not st.get("publish_pending", True)):
                    self.log("全部完成，排程器結束")
                    return
                time.sleep(self.cfg.get("interval_s", 30))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("tick", "loop"):
        raise SystemExit("用法：s_run.py tick|loop；安全預覽：S2_DRY_RUN=1")
    s = Scheduler()
    try:
        result = getattr(s, sys.argv[1])()
        if result is not None:
            print(s.status_path.read_text(), end="")
    except Exception as exc:
        s.log(f"排程器異常：{type(exc).__name__}；停止此程序，請檢查 state／本地 log")
        s.milestone(f"round2 S 排程器 異常 — 例外：{type(exc).__name__}；程序停止，請檢查本地 log")
        raise


if __name__ == "__main__":
    main()
