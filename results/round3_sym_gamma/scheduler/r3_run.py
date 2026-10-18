#!/usr/bin/env python3
"""ROUND3（G／Y／GY）自動排程器：等 S 完成與機器空閒，依序開跑 3 批，收 eval／角落診斷，發佈 ROUND3_RESULTS.md。

用法：r3_run.py tick|loop；安全預覽：R3_DRY_RUN=1（不 launch、不快照、不發佈、不寫里程碑）。
風格與規則沿用 skip-run/s_run.py；不 import 它，也不修改任何訓練 worktree。
"""
import fcntl, hashlib, json, math, os, shlex, shutil, statistics, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TAIPEI = ZoneInfo("Asia/Taipei")
EXP = Path("/workspace/rl2048-review/experiments")
GAP = 60.0
BUSY_SCRIPTS = {"eval.py", "r3_corner.py", "run_corner.py"}


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
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n")
    tmp.replace(path)


def fmt(v, spec=".1f"):
    return "—" if v is None else format(v, spec)


def snap_label(m):
    return f"t{int(m):04d}"


def sd(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else None


class Scheduler:
    def __init__(self):
        env = os.environ.get
        self.cfg = json.loads(Path(env("R3_CONFIG", HERE / "r3_config.json")).read_text())
        c = self.cfg
        self.dry_run = env("R3_DRY_RUN") == "1"
        self.state_path = Path(env("R3_STATE", HERE / "r3_state.json"))
        self.runs = Path(env("R3_RUNS", c["runs_dir"]))
        self.code = Path(env("R3_CODE", c["code_dir"]))
        self.repo = Path(env("R3_REPO", c["repo"]))
        self.status_path = Path(env("R3_STATUS", HERE / "R3_STATUS.md"))
        self.log_path = Path(env("R3_LOG", HERE / "r3_run.log"))
        self.miles = Path(env("R3_MILESTONES", EXP / "milestones.json"))
        default_lock = EXP / ".milestones.lock" if self.miles == EXP / "milestones.json" else self.miles.parent / ".milestones.lock"
        self.miles_lock = Path(env("R3_MILESTONES_LOCK", default_lock))
        self.s_runs = Path(env("R3_S_RUNS", c["start_requires"]["s_runs"]))
        self.publish_enabled = not self.dry_run and env("R3_NO_PUBLISH") != "1"
        self.controls = c["controls"]
        root = env("R3_CONTROLS_ROOT")
        if root:
            for ctl in self.controls.values():
                ctl["dir"] = str(Path(root) / ctl["name"])
                if ctl.get("load"):
                    ctl["load"]["state"] = str(Path(root) / Path(ctl["load"]["state"]).name)
        self.py = c["python"]
        self.jobs = {j["name"]: j for j in c["jobs"]}
        self.batches = [b for b in sorted(c["batches"], key=lambda b: b["id"])]

    # ---------- utilities ----------
    def log(self, text):
        line = f"[{now()} 台北] {text}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def now_epoch(self):
        return time.time()

    def loadavg(self):
        return os.getloadavg()[0]

    def process_args(self):
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

    def process_scope_ok(self):
        try:
            return not Path("/proc/1/comm").read_text().strip().startswith("codex")
        except OSError:
            return False

    def machine_busy(self):
        if not self.process_scope_ok():
            return True
        for args in self.process_args():
            for i, arg in enumerate(args):
                name = Path(arg).name
                if name in BUSY_SCRIPTS:
                    return True
                if name == "main.py" and i + 1 < len(args) and args[i + 1] == "train":
                    return True
        return False

    def job_alive(self, name):
        needle = str(self.runs / name) + "/"
        return any(needle in " ".join(a) for a in self.process_args())

    def run(self, args, **kw):
        return subprocess.run(["nice", "-n", "19", *map(str, args)], capture_output=True, text=True,
                              timeout=kw.pop("timeout", 45), **kw)

    # ---------- commands ----------
    def eval_args(self, ckpt, out):
        e = self.cfg["eval"]
        return ["timeout", "--signal=TERM", "--kill-after=30s", f"{e['quota_min']}m", self.py, "eval.py",
                "--checkpoint", str(ckpt), "--policy", e["policy"], "--games", str(e["games"]),
                "--seed-start", str(e["seed_start"]), "--board", e["board"], "--max-steps", str(e["max_steps"]),
                "--json", str(out)]

    def corner_args(self, ckpt, eval_json, out):
        e = self.cfg["eval"]
        return ["timeout", "--signal=TERM", "--kill-after=30s", f"{e['corner_quota_min']}m", self.py,
                str(HERE / "r3_corner.py"), str(self.code), self.cfg["corner_dir"], str(ckpt), str(eval_json), str(out),
                str(e["games"]), str(e["seed_start"]), str(e["max_steps"])]

    def train_argv(self, job, arm=None):
        d = self.runs / job["name"]
        c = self.cfg
        arm = job["arm"] if arm is None else arm
        return [self.py, "main.py", "train", *c["common"], *c["arm_flags"][arm], "--seed", str(job["seed"]),
                "--threads", str(c["threads"]), "--minutes", str(c["minutes"]), "--save-every", "1",
                "--save-as", str(d / "ckpt"), "--log", str(d / "train.csv")]

    def script(self, job):
        d = self.runs / job["name"]
        q = lambda p: shlex.quote(str(p))
        th = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
        return (f"{shlex.join(self.train_argv(job))} > {q(d / 'train.out')} 2>&1 && date +%s > {q(d / 'eval_start')} && "
                f"{th} {shlex.join(self.eval_args(d / 'ckpt', d / 'eval.json'))} > {q(d / 'eval.out')} 2>&1 && "
                f"date +%s > {q(d / 'eval_end')} && "
                f"{th} {shlex.join(self.corner_args(d / 'ckpt', d / 'eval.json', d / 'corner.json'))} > {q(d / 'corner.out')} 2>&1")

    def batch_of(self, job):
        return next(b for b in self.batches if b["id"] == job["batch"])

    def launch_command(self, job):
        return ["setsid", "nohup", "taskset", "-c", self.batch_of(job)["cpus"], "nice", "-n", str(self.cfg["nice"]),
                "bash", "-c", self.script(job)]

    def spawn(self, args, output, threads=1):
        if self.dry_run:
            raise RuntimeError("dry-run 禁止 launch")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        with Path(output).open("a") as out:
            return subprocess.Popen(args, cwd=self.code, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)

    # ---------- milestones ----------
    def _append_milestone(self, text, prefix=None):
        if self.dry_run or not self.cfg.get("milestones", True):
            return
        self.miles_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.miles_lock.open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            ms = load(self.miles, [])
            if not isinstance(ms, list):
                raise ValueError("里程碑不是陣列，停止覆寫")
            key = prefix or text
            if not any(m.get("text", "").startswith(key) for m in ms):
                ms.append(dict(time=datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M"), text=text, reported=False))
                save_json(self.miles, ms)
                self.log("里程碑：" + text)

    def milestone(self, text):
        self._append_milestone(text)

    def alert_once(self, st, job, text):
        if self.dry_run:
            return
        kind = text.split("：")[0]
        key = f"{job}|{kind}"
        seen = st.setdefault("alerted", [])
        if key in seen:
            return
        prefix = f"round3 {job} 異常 — {kind}："
        self._append_milestone(f"round3 {job} 異常 — {text}", prefix=prefix)
        seen.append(key)

    # ---------- readiness ----------
    @staticmethod
    def valid_eval(ev):
        try:
            v = [ev["mean"], ev["median"], ev["reach_rates"]["1024"], ev["reach_rates"]["2048"]]
            return (all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)
                    and min(v[:2]) >= 0 and all(0 <= x <= 1 for x in v[2:]))
        except (KeyError, TypeError):
            return False

    def s_done(self):
        out = {}
        for name in self.cfg["start_requires"]["s_jobs"]:
            d = self.s_runs / name
            out[name] = self.valid_eval(load(d / "eval.json")) and (d / "eval_end").is_file()
        return out

    def control_ready(self, seed):
        ctl = self.controls[str(seed)]
        d = Path(ctl["dir"])
        return self.valid_eval(load(d / ctl["eval"])) and all((d / f).is_file() for f in ctl["require"])

    def flag_spec(self, names):
        spec, waiting = {}, {}
        for name in names:
            job = self.jobs[name]
            ctl = self.controls[str(job["seed"])]
            meta = load(Path(ctl["dir"]) / "meta.json")
            ref = meta.get("script") if isinstance(meta, dict) else None
            if not isinstance(ref, str) or not ref.strip():
                waiting[name] = dict(ok=False, status="待對照 meta")
                continue
            tokens = shlex.split(self.script(job))
            start = tokens.index("main.py") + 1
            end = tokens.index(">", start)
            spec[name] = dict(arm_argv=tokens[start:end], c_argv=self.train_argv(job, "C")[2:],
                              ref_script=ref, expected=self.cfg["expected_config_diff"][job["arm"]])
        return spec, waiting

    def check_flags(self, st, force=False):
        t = self.now_epoch()
        cached = st.get("flags", {})
        names = list(self.jobs)
        complete = isinstance(cached.get("jobs"), dict) and set(cached["jobs"]) == set(names)
        if not force and complete and 0 <= t - cached.get("checked_epoch", -1e30) < 600:
            return cached["jobs"]
        spec, results = self.flag_spec(names)
        if spec:
            try:
                with tempfile.TemporaryDirectory(prefix="r3_flags_") as tmp:
                    path = Path(tmp) / "spec.json"
                    save_json(path, spec)
                    proc = self.run([self.py, HERE / "r3_flagcheck.py", self.code, path], timeout=120,
                                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                             OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
                if proc.returncode:
                    raise ValueError("flagcheck 執行失敗")
                checked = json.loads(proc.stdout)
                for name in spec:
                    r = checked[name]
                    if not isinstance(r, dict) or type(r.get("ok")) is not bool:
                        raise ValueError("flagcheck 結果格式錯誤")
                    results[name] = dict(r, status="一致" if r["ok"] else "不一致")
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                for name in spec:
                    results[name] = dict(ok=False, status="不一致", error="flagcheck 執行或結果無效")
        st["flags"] = dict(checked_epoch=t, checked=now(), jobs=results)
        for name, r in results.items():
            if r.get("status") == "不一致":
                self.alert_once(st, name, f"旗標不一致：{name} {self.flag_summary(r)}")
        return results

    @staticmethod
    def flag_summary(r):
        parts = []
        for label in ("unexpected_arm_diff", "missing_arm_diff", "unexpected_arm_cli", "control_cli_diff", "control_config_diff"):
            v = r.get(label)
            if v:
                parts.append(f"{label}={v}")
        return "；".join(parts) or r.get("error", "檢查未通過")

    def ready(self, st):
        t = self.now_epoch()
        req = self.cfg["start_requires"]
        busy = self.machine_busy()
        load1 = self.loadavg()
        quiet = not busy and load1 < req["max_load1"]
        idle = st.setdefault("idle", {"count": 0})
        if not quiet:
            idle.update(count=0, last_epoch=None)
        elif idle.get("last_epoch") is None or t - idle["last_epoch"] > 3 * req["idle_interval_s"]:
            idle.update(count=1, last_epoch=t)
        elif t - idle["last_epoch"] >= req["idle_interval_s"]:
            idle.update(count=min(req["idle_checks"], idle["count"] + 1), last_epoch=t)
        s = self.s_done()
        w = dict(s_jobs=s, s_done=all(s.values()), busy=busy, load1=load1, load_ok=load1 < req["max_load1"],
                 idle_count=idle["count"], idle_ok=idle["count"] >= req["idle_checks"],
                 process_scope_ok=self.process_scope_ok(), checked=now())
        w["controls_ready"] = {seed: self.control_ready(seed) for seed in req["control_ready"]}
        w["control_ready"] = all(w["controls_ready"].values())
        # 只在其他條件已滿足、或尚無快取時才跑 flagcheck（等待期間不做多餘運算）。
        gate_open = w["s_done"] and w["control_ready"]
        if gate_open or "flags" not in st:
            w["flags"] = self.check_flags(st)
        else:
            w["flags"] = st["flags"]["jobs"]
        w["flags_ok"] = all(w["flags"].get(n, {}).get("ok", False) for n in self.jobs)
        if w["s_done"] and w["control_ready"]:
            st.setdefault("gate_open_epoch", t)
        st["wait"] = w
        return w

    # ---------- launch ----------
    def launch(self, batch, st):
        if self.dry_run:
            return
        cr = self.run(["git", "-C", self.code, "rev-parse", "HEAD"])
        commit = cr.stdout.strip()
        if cr.returncode or commit != self.cfg["expected_commit"]:
            raise RuntimeError("固定程式 commit 不符，禁止開跑")
        status = self.run(["git", "-C", self.code, "status", "--porcelain", "--untracked-files=no"])
        if status.returncode or status.stdout.strip():
            raise RuntimeError("固定 worktree 有未 commit 的修改，禁止開跑")
        flags = self.check_flags(st, force=True)
        st.setdefault("wait", {}).update(flags=flags, flags_ok=all(r.get("ok") for r in flags.values()))
        if not all(flags.get(n, {}).get("ok", False) for n in self.jobs):
            raise RuntimeError("旗標一致性未通過，禁止開跑")
        if not all(self.control_ready(s) for s in self.cfg["start_requires"]["control_ready"]):
            raise RuntimeError("對照 t180 尚未完成，禁止開跑")
        if any((self.runs / n).exists() for n in batch["jobs"]):
            raise RuntimeError("批次 run 目錄已存在，禁止自動重跑")
        b = st.setdefault("batches", {}).setdefault(str(batch["id"]), {})
        b.update(started=now(), started_epoch=self.now_epoch(), load_sum=0.0, load_n=0, busy_sum=0.0, busy_n=0, pids={})
        st.setdefault("started", now())
        save_json(self.state_path, st)
        for name in batch["jobs"]:
            job = self.jobs[name]
            d = self.runs / name
            d.mkdir(parents=True, exist_ok=False)
            meta = dict(job=job, commit=commit, prereg=self.cfg["prereg"], cpus=batch["cpus"], nice=self.cfg["nice"],
                        threads=self.cfg["threads"], minutes=self.cfg["minutes"], snapshots=self.cfg["snapshots"],
                        started=now() + " Taipei", script=self.script(job), flagcheck=flags.get(name))
            save_json(d / "meta.json", meta)
            p = self.spawn(self.launch_command(job), d / "launch.out", self.cfg["threads"])
            meta["launcher_pid"] = p.pid
            save_json(d / "meta.json", meta)
            b["pids"][name] = p.pid
            save_json(self.state_path, st)
        self.log(f"批次 {batch['id']} 已啟動：{batch['jobs']}")
        st.pop("launch_error", None)

    # ---------- metrics ----------
    def read_metrics(self, d, eval_path, require):
        ev, tr = load(d / eval_path), load(d / "train.json")
        if not (self.valid_eval(ev) and isinstance(tr, dict) and tr and all((d / f).is_file() for f in require)):
            return None
        if tr.get("interrupted") or tr.get("stopped_reason") == "nonfinite_replay":
            return None
        try:
            m = dict(mean=ev["mean"], median=ev["median"], r1024=ev["reach_rates"]["1024"], r2048=ev["reach_rates"]["2048"],
                     updates=tr.get("updates"), env_steps=tr.get("env_steps"), episodes=tr.get("episodes"),
                     nan=tr.get("nan_count"), active_s=tr.get("active_s"),
                     ups=tr["updates"] / tr["active_s"] if tr.get("active_s") and tr.get("updates") is not None else None)
        except (KeyError, TypeError, ZeroDivisionError):
            return None
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in m.values()):
            return None
        return m

    def metrics(self, name, st=None):
        m = self.read_metrics(self.runs / name, "eval.json", ["eval_end"])
        if m:
            b = (st or {}).get("batches", {}).get(str(self.jobs[name]["batch"]), {})
            m.update(load1=b.get("mean_load1"), busy=b.get("mean_busy_cores"))
            c = load(self.runs / name / "corner.json")
            m["corner"] = c.get("summary") if isinstance(c, dict) else None
        return m

    def control_metrics(self, seed):
        ctl = self.controls[str(seed)]
        m = self.read_metrics(Path(ctl["dir"]), ctl["eval"], ctl["require"])
        if m:
            cfg = ctl.get("load")
            state = load(cfg["state"], {}) if cfg else {}
            b = {}
            if cfg and isinstance(state, dict):
                pairs = state.get("pairs", {})
                b = pairs.get(cfg["pair"], {}) if isinstance(pairs, dict) else {}
            b = b if isinstance(b, dict) else {}

            def avg(mean, total, count):
                v = b.get(mean)
                if v is None and isinstance(b.get(total), (int, float)) and isinstance(b.get(count), (int, float)) and b[count] > 0:
                    v = b[total] / b[count]
                return v if isinstance(v, (int, float)) and math.isfinite(v) else None
            m.update(load1=avg("mean_load1", "load_sum", "load_n"), busy=avg("mean_busy_cores", "busy_sum", "busy_n"),
                     load_note=ctl.get("load_note"))
        return m

    def verdicts(self, st):
        R = self.cfg["rules"]
        arms = {}
        for arm in ("G", "Y", "GY"):
            rows = []
            for seed in self.cfg["formal_seeds"]:
                name = f"{arm}_s{seed}"
                a, c = self.metrics(name, st), self.control_metrics(seed)
                row = dict(job=name, seed=seed, arm=a, ctl=c, pending=not (a and c))
                if not row["pending"]:
                    row["ratio"] = a["mean"] / c["mean"] if c["mean"] else None
                    row["dmean"] = a["mean"] - c["mean"]
                    row["dpp"] = 100 * (a["r1024"] - c["r1024"])
                    row["pass"] = bool(row["ratio"] is not None and row["ratio"] >= R["ratio"] and row["dpp"] >= R["pp_1024"])
                    row["abs_ok"] = a["mean"] > R["abs_mean"] and a["r1024"] >= R["abs_r1024"]
                    rr = lambda k: (a[k] / c[k]) if a.get(k) and c.get(k) else None
                    row["diag"] = dict(updates=rr("updates"), env_steps=rr("env_steps"), ups=rr("ups"))
                    row["bias_note"] = self.bias_note(row)
                rows.append(row)
            done = [r for r in rows if not r["pending"]]
            spread = {}
            for k in ("dmean", "ratio", "dpp"):
                xs = [r[k] for r in done if r.get(k) is not None]
                spread[k] = dict(mean=statistics.mean(xs), sd=sd(xs), min=min(xs), max=max(xs)) if len(xs) == len(rows) else None
            arms[arm] = dict(rows=rows, n_pass=sum(1 for r in done if r["pass"]), adopt=len(done) == len(rows) and all(r["pass"] for r in done),
                             all_pending=not done, spread=spread)
        inter = []
        for seed in self.cfg["formal_seeds"]:
            vals = {arm: self.metrics(f"{arm}_s{seed}", st) for arm in ("G", "Y", "GY")}
            c = self.control_metrics(seed)
            if c and all(vals.values()):
                inter.append(dict(seed=seed, mean=vals["GY"]["mean"] - vals["G"]["mean"] - vals["Y"]["mean"] + c["mean"],
                                  pp=100 * (vals["GY"]["r1024"] - vals["G"]["r1024"] - vals["Y"]["r1024"] + c["r1024"])))
        return dict(arms=arms, interaction=inter, all_pending=all(v["all_pending"] for v in arms.values()))

    def bias_note(self, row):
        if row["seed"] not in self.cfg["compute_bias_seeds"] or row.get("pass"):
            return ""
        ups, ref = (row.get("ctl") or {}).get("ups"), self.cfg["compute_ref"]["ups"]
        rel = f"（對照 updates/s 為參考值的 ×{ups / ref:.2f}）" if ups else ""
        return self.cfg["compute_bias_flag"] + rel

    def anomalies(self, name, alive):
        d = self.runs / name
        out = []
        if not d.exists():
            return out
        tr = load(d / "train.json")
        if isinstance(tr, dict) and tr.get("nan_count"):
            out.append(f"NaN：{name} nan_count={tr['nan_count']}")
        if isinstance(tr, dict) and (tr.get("interrupted") or tr.get("stopped_reason") == "nonfinite_replay"):
            out.append(f"中斷：{name} 訓練中斷（{tr.get('stopped_reason')}）；不自動重跑")
        if not alive and not (d / "train.json").exists():
            out.append(f"中斷：{name} 訓練程序已不存在且沒有 train.json；不自動重跑")
        if not alive and (d / "train.json").exists() and not (d / "eval_end").exists():
            out.append(f"eval 失敗：{name} 沒有 eval_end；不自動重跑")
        csv = d / "train.csv"
        if alive and not (d / "train.json").exists() and csv.exists():
            age = (self.now_epoch() - csv.stat().st_mtime) / 60
            if age > self.cfg["rules"]["stall_min"]:
                out.append(f"停滯：{name} train.csv {age:.0f} 分鐘沒有更新")
        return out

    def cpu_times(self):
        lines = Path("/proc/stat").read_text().splitlines()
        v = [float(x) for x in lines[0].split()[1:9]]
        return sum(v), v[3] + v[4], sum(1 for l in lines if l.startswith("cpu") and l[3:4].isdigit())

    def sample_load(self, b, training):
        tot, idle, ncpu = self.cpu_times()
        prev = b.get("cpu_prev")
        b["cpu_prev"] = [tot, idle] if training else None
        if training:
            b["load_sum"] = b.get("load_sum", 0) + self.loadavg()
            b["load_n"] = b.get("load_n", 0) + 1
            if prev and tot > prev[0]:
                busy = max(0, min(ncpu, ncpu * (1 - (idle - prev[1]) / (tot - prev[0]))))
                b["busy_sum"] = b.get("busy_sum", 0) + busy
                b["busy_n"] = b.get("busy_n", 0) + 1
        b["mean_load1"] = b["load_sum"] / b["load_n"] if b.get("load_n") else None
        b["mean_busy_cores"] = b["busy_sum"] / b["busy_n"] if b.get("busy_n") else None

    def snap_tick(self, js, d, training, batch):
        mono = time.monotonic()
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            boot = "?"
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
            ok = False
            for _ in range(5):
                shutil.copyfile(d / "ckpt", dst)
                r = self.run(["taskset", "-c", batch["cpus"], self.py, "-c", f"import torch;torch.load({str(dst)!r},weights_only=True)"],
                             timeout=30, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                if r.returncode == 0:
                    ok = True
                    break
                time.sleep(2)
            if not ok:
                self.log(f"{d.name} {snap_label(m)} 快照驗證失敗，下次 tick 重試")
                continue
            taken.append(m)
            js.setdefault("meta", {})[str(m)] = dict(active_s=js.get("active_s", 0), taken=now())
        for m in taken:
            out = d / "evals" / f"{snap_label(m)}.json"
            corner = d / "evals" / f"{snap_label(m)}.corner.json"
            if (out.exists() and corner.exists()) or any(str(out) in " ".join(a) for a in self.process_args()):
                continue
            tries = js.setdefault("eval_tries", {})
            if tries.get(str(m), 0) >= 3:
                continue
            (d / "evals").mkdir(exist_ok=True)
            tries[str(m)] = tries.get(str(m), 0) + 1
            snap = d / "snapshots" / f"{snap_label(m)}.pt"
            th = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
            parts = [] if out.exists() else [f"{th} {shlex.join(self.eval_args(snap, out))}"]
            parts.append(f"{th} {shlex.join(self.corner_args(snap, out, corner))}")
            args = ["setsid", "nohup", "taskset", "-c", batch["cpus"], "nice", "-n", str(self.cfg["nice"]),
                    "bash", "-c", " && ".join(parts)]
            self.spawn(args, d / "evals" / f"{snap_label(m)}.out")
            self.log(f"{d.name} 探索性 {snap_label(m)} eval＋角落診斷已啟動")

    def snapshots_finished(self, st, batch_ids=None):
        finished = True
        for key, b in st.get("batches", {}).items():
            if batch_ids is not None and key not in batch_ids:
                continue
            for name, js in b.get("snap", {}).items():
                for m in js.get("taken", []):
                    out = self.runs / name / "evals" / f"{snap_label(m)}.json"
                    corner = self.runs / name / "evals" / f"{snap_label(m)}.corner.json"
                    if load(out) and load(corner):
                        continue
                    running = any(str(out) in " ".join(a) for a in self.process_args())
                    if running or js.get("eval_tries", {}).get(str(m), 0) < 3:
                        finished = False
                    else:
                        self.alert_once(st, name, f"快照失敗：{snap_label(m)} eval／角落已嘗試3次；不影響t180判定")
        return finished

    def corner_finished(self, name):
        d = self.runs / name
        return (d / "corner.json").exists() or not self.job_alive(name)

    # ---------- main tick ----------
    def tick(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.with_suffix(".lock").open("a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return load(self.state_path, {})
            st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"batches": {}}
            if not isinstance(st, dict) or not isinstance(st.get("batches"), dict):
                raise ValueError("state 格式損壞，停止以免重複開跑")
            alerts = []
            w = self.ready(st)
            for batch in self.batches:
                key = str(batch["id"])
                b = st["batches"].get(key)
                if b is None:
                    prev_ok = all(st["batches"].get(str(p["id"]), {}).get("done") for p in self.batches if p["id"] < batch["id"])
                    ok = (prev_ok and w["s_done"] and w["control_ready"] and w["idle_ok"] and not w["busy"]
                          and w["load_ok"] and w["flags_ok"])
                    if prev_ok and w["s_done"] and w["control_ready"] and not ok:
                        since = st.get(f"wait_since_{key}") or self.now_epoch()
                        st[f"wait_since_{key}"] = since
                        if (self.now_epoch() - since) / 60 > self.cfg["start_requires"]["delay_alert_min"]:
                            self.alert_once(st, f"批次{key}", f"開跑延遲：條件已滿足以外仍等待超過 {self.cfg['start_requires']['delay_alert_min']} 分鐘（busy={w['busy']}、load1={w['load1']:.1f}、flags_ok={w['flags_ok']}）")
                    if ok and not self.dry_run:
                        try:
                            self.launch(batch, st)
                        except Exception as exc:
                            text = f"啟動失敗：{type(exc).__name__} {exc}（不自動重跑已啟動 job）"
                            self.log(text)
                            self.alert_once(st, f"批次{key}", text)
                            st["launch_error"] = text
                            alerts.append((f"批次{key}", text))
                    break  # 每 tick 最多一批；未開跑的批次之後的批次也不看
                active = {n: self.job_alive(n) and not (self.runs / n / "train.json").exists() for n in batch["jobs"]}
                self.sample_load(b, any(active.values()))
                for name in batch["jobs"]:
                    if not self.dry_run:
                        self.snap_tick(b.setdefault("snap", {}).setdefault(name, {}), self.runs / name, active[name], batch)
                    for text in self.anomalies(name, self.job_alive(name)):
                        alerts.append((name, text))
                        self.alert_once(st, name, text)
                if (not b.get("done") and all(self.metrics(n) for n in batch["jobs"])
                        and all(self.corner_finished(n) for n in batch["jobs"]) and self.snapshots_finished(st, {key})):
                    b["done"] = now()
                    self.milestone(self.batch_text(st, batch))
            if all(st["batches"].get(str(b["id"]), {}).get("done") for b in self.batches) and not st.get("done"):
                st["completion_text"] = self.finish_text(st)
                self.milestone(st["completion_text"])
                st["done"] = now()
            save_json(self.state_path.with_name(self.state_path.stem + ".dry_run.json") if self.dry_run else self.state_path, st)
            self.write_status(st, alerts)
            if self.publish_enabled:
                self.publish(st)
                save_json(self.state_path, st)
            return st

    def row_text(self, r):
        if r["pending"]:
            return f"{r['job']} 待資料（不通過）"
        return (f"{r['job']} ×{fmt(r['ratio'], '.2f')}/{r['dpp']:+.0f}pp {'過' if r['pass'] else '不過'}"
                f"（updates/s ×{fmt(r['diag']['ups'], '.2f')}）" + ("；" + r["bias_note"] if r["bias_note"] else ""))

    def batch_text(self, st, batch):
        vs = self.verdicts(st)
        rows = [r for a in vs["arms"].values() for r in a["rows"] if r["job"] in batch["jobs"]]
        b = st["batches"].get(str(batch["id"]), {})
        return (f"round3 第{batch['id']}批（s{batch['seed']}）完成：" + "；".join(self.row_text(r) for r in rows)
                + f"；批次 load1 {fmt(b.get('mean_load1'))}／忙碌核 {fmt(b.get('mean_busy_cores'))}")

    def finish_text(self, st):
        vs = self.verdicts(st)
        parts = []
        for arm, v in vs["arms"].items():
            sp = v["spread"].get("ratio")
            parts.append(f"{arm} {'採用' if v['adopt'] else '不採用'}（{v['n_pass']}/3；mean 比值平均 ×{fmt(sp and sp['mean'], '.2f')}、SD {fmt(sp and sp['sd'], '.2f')}）")
        inter = "、".join(f"s{i['seed']} {i['mean']:+.0f}/{i['pp']:+.0f}pp" for i in vs["interaction"]) or "—"
        return "round3 完成：" + "；".join(parts) + f"；交互作用（探索性）{inter}；詳見 ROUND3_RESULTS.md"

    # ---------- status / results ----------
    def write_status(self, st, alerts):
        yn = lambda v: "是" if v else "否"
        w = st.get("wait", {})
        L = [f"# ROUND3（G／Y／GY）狀態與結果（更新：{now()} 台北）", "", f"事前登記：{self.cfg['prereg']}。",
             "3 批 × 3 個 job（每批同一個 seed 的 G、Y、GY），每個 job 2 threads、nice 10、CPU 0–5、180 分鐘 active；順序 s303 → s202 → s101。",
             "G = C＋`--reward-mode raw --gamma 1.0 --value-rescale h`；Y = C＋`--sym-aug random`；GY = 兩者都開。對照為既有同 seed 的 t180 C 軌跡。t90／120／150 只供探索。", ""]
        if self.dry_run:
            L.append("**DRY-RUN：只寫本地狀態，禁止 launch／快照／publish／里程碑。**")
        if not st.get("started"):
            L += ["狀態：**等待中**",
                  *[f"- {n} t180 eval（有效 eval.json＋eval_end）：{yn(v)}" for n, v in w.get("s_jobs", {}).items()],
                  f"- S 全部完成：{yn(w.get('s_done'))}",
                  *[f"- 對照 C_s{s} t180 eval 已存在：{yn(v)}" for s, v in w.get("controls_ready", {}).items()],
                  f"- 機器有 main.py train／eval.py／角落診斷：{yn(w.get('busy'))}；load1 {fmt(w.get('load1'), '.2f')}（需 < {self.cfg['start_requires']['max_load1']}）",
                  f"- 連續空閒（間隔 ≥30 秒）：{w.get('idle_count', 0)}/2",
                  f"- 旗標核對全部一致：{yn(w.get('flags_ok'))}（檢查時間 {st.get('flags', {}).get('checked', '—')}）"]
        else:
            L.append("狀態：**" + ("全部完成" if st.get("done") else "進行中") + "**")
        L += ["", "## 批次", "", "| 批 | seed | jobs | 開始 | 完成 | 平均 load1 | 忙碌核 |", "|---|---|---|---|---|---|---|"]
        for b in self.batches:
            s = st.get("batches", {}).get(str(b["id"]), {})
            L.append(f"| {b['id']} | {b['seed']} | {'、'.join(b['jobs'])} | {s.get('started', '—')} | {s.get('done', '—')} | {fmt(s.get('mean_load1'), '.2f')} | {fmt(s.get('mean_busy_cores'), '.2f')} |")
        flags = st.get("flags", {}).get("jobs", {})
        L += ["", "## 旗標核對", "", "| job | 狀態 | arm 與 C 的有效設定差 |", "|---|---|---|"]
        for n in self.jobs:
            r = flags.get(n, {})
            L.append(f"| {n} | {r.get('status', '—')} | {r.get('arm_config_diff', '—')} |")
        vs = self.verdicts(st)
        L += ["", "## t180 正式判定（每個 seed 都要 mean ≥×1.20 且 ≥1024 +10pp；三個 seed 全過才採用該 arm）", ""]
        for arm, v in vs["arms"].items():
            L += [f"### {arm}：{'採用' if v['adopt'] else ('待資料' if v['all_pending'] else '不採用')}（{v['n_pass']}/3）", "",
                  "| seed | arm mean／≥1024 | 對照 mean／≥1024 | 比值 | Δmean | Δ≥1024 | 分數門 | 絕對目標 | updates ×| env_steps ×| updates/s ×| arm load1／忙碌核 | 對照 load1 | 註記 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in v["rows"]:
                if r["pending"]:
                    L.append(f"| {r['seed']} | 待資料 | {'有' if r['ctl'] else '待資料'} | | | | 不通過（缺資料） | | | | | | | |")
                    continue
                a, c, dg = r["arm"], r["ctl"], r["diag"]
                note = "；".join(x for x in [r["bias_note"], c.get("load_note") or ""] if x)
                L.append(f"| {r['seed']} | {a['mean']:.0f}／{100*a['r1024']:.0f}% | {c['mean']:.0f}／{100*c['r1024']:.0f}% | ×{r['ratio']:.2f} | {r['dmean']:+.0f} | {r['dpp']:+.0f}pp | "
                         f"{'過' if r['pass'] else '不過'} | {'達標' if r['abs_ok'] else '未達'} | {fmt(dg['updates'], '.2f')} | {fmt(dg['env_steps'], '.2f')} | {fmt(dg['ups'], '.2f')} | "
                         f"{fmt(a.get('load1'))}／{fmt(a.get('busy'))} | {fmt(c.get('load1'))} | {note} |")
            sp = v["spread"]
            if all(sp.values()):
                L.append("")
                L.append("配對差（arm − C，n=3 條確定性軌跡，樣本 SD n−1）：" + "；".join(
                    f"{k} 平均 {sp[k]['mean']:.2f}、SD {fmt(sp[k]['sd'], '.2f')}、範圍 [{sp[k]['min']:.2f}, {sp[k]['max']:.2f}]" for k in ("dmean", "ratio", "dpp")))
            L.append("")
        L += ["## 交互作用（探索性）：I = GY − G − Y + C", "", "| seed | I(mean) | I(≥1024, pp) |", "|---|---|---|"]
        for i in vs["interaction"]:
            L.append(f"| {i['seed']} | {i['mean']:+.0f} | {i['pp']:+.1f} |")
        if vs["interaction"]:
            L.append(f"| 平均 | {statistics.mean(i['mean'] for i in vs['interaction']):+.0f} | {statistics.mean(i['pp'] for i in vs['interaction']):+.1f} |")
        L += ["", "## 角落診斷（CORNER_A 定義；只當診斷，訓練程式讀不到）", "",
              "| job | ckpt | mean | corner_rate | ≥256 corner | center≥256 | 移開/100步 | 最長連續 | 終局 mono_snake | ρ(cr256,分數) | 與 eval 一致 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in self.jobs:
            d = self.runs / n
            items = [(snap_label(m), d / "evals" / f"{snap_label(m)}.json", d / "evals" / f"{snap_label(m)}.corner.json") for m in self.cfg["snapshots"]]
            items.append(("t0180", d / "eval.json", d / "corner.json"))
            for lab, evp, cp in items:
                c = load(cp)
                ev = load(evp)
                if not isinstance(c, dict):
                    continue
                s = c["summary"]
                L.append(f"| {n} | {lab}{'（正式）' if lab == 't0180' else '（探索）'} | {fmt(ev and ev.get('mean'), '.0f')} | {fmt(s['corner_rate'], '.3f')} | {fmt(s['corner_rate_256'], '.3f')} | "
                         f"{fmt(s['center_rate_256'], '.3f')} | {fmt(s['leaves_per100'], '.2f')} | {fmt(s['longest_streak'], '.1f')} | {fmt(s['final_mono_snake'], '.3f')} | "
                         f"{fmt(s['spearman_cr256_score'], '.2f')} | {c.get('eval_match')} |")
        L += ["", "## 探索性快照（mean／≥1024）", "", "| job | t090 | t120 | t150 | t180 |", "|---|---|---|---|---|"]
        for n in self.jobs:
            d = self.runs / n
            cells = []
            for p in [d / "evals" / f"{snap_label(m)}.json" for m in self.cfg["snapshots"]] + [d / "eval.json"]:
                ev = load(p)
                cells.append(f"{ev['mean']:.0f}／{100*ev['reach_rates']['1024']:.0f}%" if self.valid_eval(ev) else "—")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L += ["", "## 註記", "",
              "- 對照在不同時段、不同負載下跑，只並列不校正；結果只能歸因於整套 arm 設定在 180 分鐘 active 預算下（含負載條件）的表現。",
              f"- s101／s202 的對照（LONG）updates/s 為 8 核滿載 C 的 ×1.19／×1.15；arm 在這兩個 seed 沒過時會加註「{self.cfg['compute_bias_flag']}」，判定不變。",
              "- s303 的對照 C_s303 在 15:56–16:03（台北）期間機器降速。",
              "- G 的 loss／q 在 h 空間，不能和 C 比；只比 eval。"]
        if alerts or st.get("alerted"):
            L += ["", "## 異常", ""] + [f"- {a}" for a in st.get("alerted", [])]
        if st.get("launch_error"):
            L.append(f"- 啟動錯誤：{st['launch_error']}")
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_name(self.status_path.name + ".tmp")
        tmp.write_text("\n".join(L) + "\n")
        tmp.replace(self.status_path)

    def result_files(self):
        for name in self.jobs:
            d = self.runs / name
            for f in [d / n for n in ("meta.json", "train.json", "eval.json", "corner.json")] + sorted((d / "evals").glob("*.json")):
                if f.is_file():
                    yield f, Path("runs") / name / f.relative_to(d)
        for ctl in self.controls.values():
            d = Path(ctl["dir"])
            for rel in (ctl["eval"], "train.json", "meta.json"):
                if (d / rel).is_file():
                    yield d / rel, Path("controls") / ctl["name"] / rel

    def publication_sig(self, st):
        ev = {"started": st.get("started"), "done": st.get("done"), "alerts": st.get("alerted", []),
              "batches": {k: {x: b.get(x) for x in ("started", "done")} for k, b in st.get("batches", {}).items()}}
        ev["files"] = {str(r): hashlib.sha256(s.read_bytes()).hexdigest() for s, r in self.result_files()}
        return hashlib.sha256(json.dumps(ev, sort_keys=True).encode()).hexdigest()

    def publish(self, st):
        if not self.publish_enabled or self.dry_run or not st.get("started"):
            return False
        sig = self.publication_sig(st)
        if st.get("published_sig") == sig and not st.get("publish_pending"):
            return True
        st["publish_pending"] = True
        sub, branch = self.cfg["results_subdir"], self.cfg["branch"]
        with self.state_path.with_suffix(".publish.lock").open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                git = lambda *a: self.run(["git", "-C", self.repo, *a], timeout=90)
                gd = git("rev-parse", "--absolute-git-dir")
                if gd.returncode or (Path(gd.stdout.strip()) / "index.lock").exists():
                    return False
                cur = git("branch", "--show-current")
                if cur.returncode or cur.stdout.strip() != branch:
                    self.log("發佈延後：結果 worktree 分支不符")
                    return False
                dest = self.repo / sub
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.status_path, dest / "ROUND3_RESULTS.md")
                for src, rel in self.result_files():
                    t = dest / rel
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, t)
                if git("add", "--", sub).returncode:
                    return False
                diff = git("diff", "--cached", "--quiet", "--", sub)
                if diff.returncode not in (0, 1):
                    return False
                if diff.returncode == 1 and git("commit", "-m", "ROUND3：更新結果（自動）", "--", sub).returncode:
                    return False
                rc = git("push", "origin", branch).returncode  # 不 force、不 merge
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
        raise SystemExit("用法：r3_run.py tick|loop；安全預覽：R3_DRY_RUN=1")
    s = Scheduler()
    try:
        getattr(s, sys.argv[1])()
    except Exception as exc:
        s.log(f"排程器異常：{type(exc).__name__}: {exc}；停止此程序")
        s.milestone(f"round3 排程器 異常 — 例外：{type(exc).__name__}；程序停止，請檢查 round3-run/r3_run.log")
        raise


if __name__ == "__main__":
    main()
