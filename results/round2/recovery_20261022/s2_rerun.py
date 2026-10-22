#!/usr/bin/env python3
"""n-tuple seed 2 gated auto-rerun (PM 2026-10-18 17:28 Taipei). Read-only towards every other job.
Phases: WAITING_GATE -> TRAINING -> WAITING_EVAL -> EVALUATING -> DONE  (or STOPPED_THROUGHPUT / FAILED).
Gate (v2, PM 10-22 15:xx): Round 3 all 3 batches (G/Y/GY x s303/s202/s101) complete incl. final eval
       (r3_state.json done, or every job has eval_end+eval.json+corner.json), no R3 train/eval/corner process left,
       machine idle (load1 < idle_load1 for every sample in the last idle_wall_s), no other eval / n-tuple job.
       State migration: STOPPED_THROUGHPUT attempt 1 (v1 gate) -> archived, attempt 2 WAITING_GATE.
       (v1 gate = R3 batch 1 running with baseline; kept in s2_rerun.py.bak_20261022b.)
Train: nice 19, taskset -c 7, `./train 2 3600 rerun_s2/snapshots` (same binary/PREREG/snapshot points as seed 1).
Guard: every minute, each arm's env steps per active-second over trailing 300 active-s (5-min MA) vs 15-min baseline;
       an arm below 0.95x baseline on 3 consecutive non-excluded checks -> kill seed 2, STOPPED_THROUGHPUT.
       Checks whose window touched another job's eval are excluded (logged).
Eval : 1,000 greedy games, eval seeds 1000-1999, corner metrics + 50-game Python cross-check, one at a time,
       CPU 7, nice 19, 1 thread; only starts when no other eval is running.
Usage: s2_rerun.py run | status | selftest-gate   (env S2_TEST_CONFIG=<json> overrides CFG for tests)"""
import csv, datetime as dt, fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NT = Path("/workspace/rl2048-review/corner/ntuple")
CFG = {
    "r3_runs": "/workspace/rl2048-review/round3-run/runs",
    "r3_state": "/workspace/rl2048-review/round3-run/r3_state.json",
    "r3_jobs": [f"{a}_s{s}" for s in (303, 202, 101) for a in ("G", "Y", "GY")],
    "gate_version": 2,
    "idle_load1": 2.0,
    "idle_wall_s": 600,                # load1 < idle_load1 for every sample over this many wall seconds
    "other_run_globs": ["/workspace/rl2048-review/experiments/runs/*", "/workspace/rl2048-review/experiments/runs/round2/*",
                        "/workspace/rl2048-review/skip-run/runs/*", "/workspace/rl2048-review/round3-run/runs/*"],
    "busy_scripts": ["eval.py", "r3_corner.py", "run_corner.py"],
    "eval_marker_max_age_s": 2400,     # eval_start without newer eval_end counts as running for up to 40 min
    "poll_s": 60,
    "baseline_active_s": 900,
    "baseline_quiet_wall_s": 900,      # no foreign eval seen in this many wall seconds before start
    "csv_fresh_s": 180,
    "guard_window_active_s": 300,
    "guard_start_after_s": 300,
    "guard_drop": 0.05,
    "guard_consecutive": 3,
    "guard_exclude_wall_s": 360,
    "freeze_gap_s": 180,
    "max_attempts": 3,
    "cpu": "7",
    "train_cmd": ["./train", "2", "3600"],          # + outdir
    "train_seconds": 3600,
    "snaps": [5, 15, 30, 60, 120, 240, 480, 900, 1800, 3600],
    "eval_cmd": ["./nteval", "{snap}", "1000", "1000", "{csv}", "{dump}", "50"],
    "check_cmd": ["python3", "check_metrics.py", "{csv}", "{dump}"],
    "workdir": str(NT),
    "out": str(HERE),
    "results_md": str(NT / "NTUPLE_RESULTS.md"),
    "seed1_summary": str(NT / "eval" / "summary.json"),
    "quiet_window": ["17:45", "18:00"],
    "dry_run": False,
    "summarize": str(NT / "summarize.py"),
    # 10-22 PM：R3 排程器／run 不存在超過寬限時間 → 記錄閘門異常並寫里程碑回報，不無聲無限等
    "r3_sched_script": "r3_run.py",
    "r3_missing_grace_s": 1800,
    "milestones": "/workspace/rl2048-review/experiments/milestones.json",
}
if os.environ.get("S2_TEST_CONFIG"):
    CFG.update(json.load(open(os.environ["S2_TEST_CONFIG"])))
OUT = Path(CFG["out"]); WD = Path(CFG["workdir"])
STATE = OUT / "s2_state.json"; STATUS = OUT / "S2_STATUS.md"; LOG = OUT / "s2_rerun.log"; MON = OUT / "s2_monitor_v2.csv"
SNAPDIR = OUT / "snapshots"; EVALDIR = OUT / "eval"; LOGDIR = OUT / "logs"
BEGIN, END = "<!-- S2_RERUN_BEGIN -->", "<!-- S2_RERUN_END -->"

def now(): return time.time()
def ts(t=None): return dt.datetime.fromtimestamp(t or now()).strftime("%Y-%m-%d %H:%M:%S")
def log(msg):
    with open(LOG, "a") as f: f.write(f"{ts()} {msg}\n")
def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"phase": "WAITING_GATE", "attempt": 1, "created": ts(), "events": [], "samples": []}
def save_state(st):
    tmp = STATE.with_suffix(".tmp"); json.dump(st, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, STATE)
def event(st, msg):
    st.setdefault("events", []).append(f"{ts()} {msg}"); log(msg)

# ---------- observation (read-only) ----------
def proc_args():
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try:
            a = [s.decode(errors="replace") for s in (p / "cmdline").read_bytes().split(b"\0") if s]
            if a: out.append((int(p.name), a))
        except OSError: pass
    return out

def foreign_evals(procs=None):
    """List of descriptions of other jobs' evals currently running (process scan + run-dir markers)."""
    found = []
    for pid, a in (procs if procs is not None else proc_args()):
        for x in a:
            if Path(x).name in CFG["busy_scripts"]:
                ck = next((a[i + 1] for i, y in enumerate(a) if y in ("--checkpoint", "--json") and i + 1 < len(a)), "")
                found.append(f"proc {pid} {Path(x).name} {ck}"); break
    import glob
    t = now()
    for g in CFG["other_run_globs"]:
        for d in glob.glob(g):
            s, e = Path(d) / "eval_start", Path(d) / "eval_end"
            try:
                if s.is_file() and (not e.is_file() or e.stat().st_mtime < s.stat().st_mtime) and t - s.stat().st_mtime < CFG["eval_marker_max_age_s"]:
                    found.append(f"marker {d}/eval_start")
            except OSError: pass
    return found

def train_procs(procs):
    """main.py train processes -> {log_path: pid}"""
    res = {}
    for pid, a in procs:
        for i, x in enumerate(a):
            if Path(x).name == "main.py" and i + 1 < len(a) and a[i + 1] == "train" and "--log" in a:
                res[a[a.index("--log") + 1]] = pid
    return res

def read_train_csv(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try: rows.append((float(r["elapsed_s"]), int(r["steps"])))
                except (KeyError, ValueError, TypeError): pass
    except OSError: pass
    return rows

def rate(rows, win):
    """env steps per active-second over the trailing `win` active seconds; None if not enough data."""
    if not rows or rows[-1][0] < win: return None
    T = rows[-1][0]
    return sum(s for e, s in rows if e > T - win) / win

def r3_arm_info(procs):
    tp = train_procs(procs); info = {}
    for j in CFG["r3_jobs"]:
        p = Path(CFG["r3_runs"]) / j / "train.csv"
        rows = read_train_csv(p)
        try: age = now() - p.stat().st_mtime
        except OSError: age = None
        d = Path(CFG["r3_runs"]) / j
        info[j] = dict(done=all((d / x).is_file() for x in ("eval_end", "eval.json", "corner.json")), alive=str(p) in tp, age=age, elapsed=rows[-1][0] if rows else None,
                       base=rate(rows, CFG["baseline_active_s"]), ma=rate(rows, CFG["guard_window_active_s"]))
    return info

def other_train_info(procs):
    r3 = {str(Path(CFG["r3_runs"]) / j / "train.csv") for j in CFG["r3_jobs"]}
    out = {}
    for lp, pid in train_procs(procs).items():
        if lp in r3: continue
        out[Path(lp).parent.name] = rate(read_train_csv(lp), CFG["guard_window_active_s"])
    return out

def own_ntuple_running(procs, mine=()):
    res = []
    for pid, a in procs:
        if pid in mine: continue
        n = Path(a[0]).name
        if n == "nteval" or (n == "train" and len(a) >= 3 and a[1].isdigit()):
            res.append(pid)
    return res

def in_quiet_window():
    a, b = CFG["quiet_window"]; h = dt.datetime.now().strftime("%H:%M"); return a <= h < b

def sample(st, phase):
    procs = proc_args(); fe = foreign_evals(procs); arms = r3_arm_info(procs); oth = other_train_info(procs)
    t = now(); s = dict(t=t, fe=fe, arms={k: v["ma"] for k, v in arms.items()}, other=oth, load1=float(open("/proc/loadavg").read().split()[0]))
    st.setdefault("samples", []).append(s); st["samples"] = st["samples"][-400:]
    new = not MON.exists()
    with open(MON, "a") as f:
        if new: f.write("time,phase,load1," + ",".join(f"{j}_ma300" for j in CFG["r3_jobs"]) + ",foreign_evals,other_jobs_ma300\n")
        f.write(f"{ts(t)},{phase},{s['load1']}," + ",".join("" if s['arms'][j] is None else f"{s['arms'][j]:.1f}" for j in CFG["r3_jobs"]) +
                f",{len(fe)},\"{';'.join(f'{k}={v:.1f}' if v else f'{k}=NA' for k, v in oth.items())}\"\n")
    return procs, fe, arms, s

def r3_presence(procs):
    """(排程器在跑?, 任一 R3 run 目錄存在?, 任一 R3 訓練 process 在?)"""
    sched = any(Path(x).name == CFG["r3_sched_script"] for _, a in procs for x in a[:3])
    dirs = any((Path(CFG["r3_runs"]) / j).is_dir() for j in CFG["r3_jobs"])
    return sched, dirs

def r3_complete(arms):
    """R3 三批（含最終 eval＋corner）全部完成？以 r3_state.json done 或每個 job 的 eval_end/eval.json/corner.json 為準。"""
    try: sd = bool(json.load(open(CFG["r3_state"])).get("done"))
    except Exception: sd = False
    return sd or all(a.get("done") for a in arms.values()), sd

def r3_procs(procs):
    """仍在跑、和 R3 有關的 process（訓練、eval、corner、bash -c 鏈）pid 清單；排程器本身不算。"""
    root = str(Path(CFG["r3_runs"]))
    return [pid for pid, a in procs if any(root in x for x in a) and not any(Path(x).name == CFG["r3_sched_script"] for x in a[:3])]

def milestone(text):
    mp = Path(CFG["milestones"])
    try:
        lk = open(str(mp.parent / ".milestones.lock"), "a"); fcntl.flock(lk, fcntl.LOCK_EX)
        ms = json.load(open(mp)) if mp.exists() else []
        if isinstance(ms, list) and not any(m.get("text") == text for m in ms):
            ms.append(dict(time=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), text=text, reported=False))
            tmp = mp.with_name(mp.name + ".tmp"); tmp.write_text(json.dumps(ms, ensure_ascii=False, indent=2) + "\n"); os.replace(tmp, mp)
        lk.close()
    except Exception as e:
        log(f"里程碑寫入失敗：{e}")

def check_r3_missing(st, procs, arms):
    """R3 排程器不在且沒有任何 R3 訓練在跑：超過寬限即標記 GATE_ANOMALY 並回報一次；恢復後自動清除。"""
    sched, dirs = r3_presence(procs)
    alive_any = any(a["alive"] for a in arms.values())
    t = now()
    if sched or alive_any or r3_complete(arms)[0]:
        if st.get("gate_anomaly"):
            event(st, "R3 已恢復（排程器或訓練在跑），清除閘門異常")
        st.pop("r3_missing_since", None); st.pop("gate_anomaly", None)
        return None
    since = st.setdefault("r3_missing_since", t)
    if t - since < CFG["r3_missing_grace_s"]:
        return None
    if not st.get("gate_anomaly"):
        why = "R3 排程器（" + CFG["r3_sched_script"] + "）不在" + ("、R3 run 目錄不存在" if not dirs else "、R3 訓練 process 不在")
        st["gate_anomaly"] = dict(since=ts(since), flagged=ts(t), why=why)
        event(st, f"閘門異常：{why}，已持續 {(t-since)/60:.0f} 分鐘；seed 2 閘門不可能通過，需人工判斷")
        milestone(f"n-tuple seed 2 異常 — 閘門等待不存在的 R3：{why}（自 {ts(since)}）")
    return st["gate_anomaly"]

def gate(st, procs, fe, arms):
    """v2 閘門：R3 全部完成 + R3 process 都不在 + 機器空閒。returns (ok, reasons)"""
    why = []
    comp, sd = r3_complete(arms)
    if not comp:
        nd = [j for j, a in arms.items() if not a.get("done")]
        why.append(f"R3 尚未全部完成（含最終 eval／corner）：未完成 {len(nd)}/{len(arms)} 個 job：{'、'.join(nd)}")
    rp = r3_procs(procs)
    if rp: why.append(f"R3 相關 process 還在：{rp[:6]}")
    if fe: why.append("他人 eval 進行中：" + "；".join(fe[:3]))
    t = now(); recent = [s for s in st.get("samples", []) if t - s["t"] <= CFG["idle_wall_s"]]
    if any(s.get("load1", 99) >= CFG["idle_load1"] for s in recent):
        why.append(f"機器未空閒：最近 {CFG['idle_wall_s']//60} 分鐘 load1 最高 {max(s.get('load1', 99) for s in recent):.2f}（需全部 < {CFG['idle_load1']}）")
    elif not recent or t - min(s["t"] for s in recent) < CFG["idle_wall_s"] - 2 * CFG["poll_s"]:
        why.append(f"空閒觀察時間不足 {CFG['idle_wall_s']//60} 分鐘")
    if any(s["fe"] for s in recent): why.append(f"最近 {CFG['idle_wall_s']//60} 分鐘內有他人 eval")
    ot = own_ntuple_running(procs)
    if ot: why.append(f"已有其他 n-tuple train/nteval 在跑：{ot}")
    if in_quiet_window(): why.append("17:45–18:00 靜默時段")
    return (not why), why

def guard_check(st, arms):
    """returns (stop, note). Uses state['guard'] counters."""
    g = st.setdefault("guard", {"breach": {}, "excluded": 0, "checks": 0, "min_ratio": {}})
    t = now()
    if t - st["train_start"] < CFG["guard_start_after_s"]: return False, "warmup"
    excl = any(s["fe"] for s in st["samples"] if t - s["t"] <= CFG["guard_exclude_wall_s"])
    g["checks"] += 1
    notes = []
    for j, a in arms.items():
        b = st["baseline"].get(j)
        if not b or a["ma"] is None or not a["alive"]:
            notes.append(f"{j}=NA"); g["breach"][j] = 0; continue
        r = a["ma"] / b; notes.append(f"{j}={a['ma']:.1f}({(r-1)*100:+.1f}%)")
        if excl: continue
        g["min_ratio"][j] = min(g["min_ratio"].get(j, 9), r)
        g["breach"][j] = g["breach"].get(j, 0) + 1 if r < 1 - CFG["guard_drop"] else 0
    if excl: g["excluded"] += 1; return False, "EXCLUDED(他人 eval) " + " ".join(notes)
    bad = [j for j, n in g["breach"].items() if n >= CFG["guard_consecutive"]]
    return bool(bad), ("BREACH " + ",".join(bad) + " " if bad else "ok ") + " ".join(notes)

# ---------- actions ----------
def alive(pid):
    if not pid or pid < 0: return False
    try: os.kill(pid, 0)
    except OSError: return False
    try: return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError: return False

def start_train(st):
    for d in (SNAPDIR, LOGDIR): d.mkdir(parents=True, exist_ok=True)
    cmd = ["nice", "-n", "19", "taskset", "-c", CFG["cpu"], "env", "OMP_NUM_THREADS=1", *CFG["train_cmd"], str(SNAPDIR)]
    err = open(LOGDIR / "train_s2.err", "w"); outf = open(LOGDIR / "train_s2.out", "w")
    p = subprocess.Popen(cmd, cwd=WD, stdin=subprocess.DEVNULL, stdout=outf, stderr=err, start_new_session=True)
    st.update(phase="TRAINING", train_pid=p.pid, train_start=now(), train_cmd=" ".join(cmd))
    event(st, f"開跑 seed 2（attempt {st['attempt']}）pid {p.pid}：{' '.join(cmd)}（cwd {WD}）")

def train_finished_ok():
    try: e = (LOGDIR / "train_s2.err").read_text()
    except OSError: return False
    return "END train_s=" in e and (SNAPDIR / f"s2_t{CFG['snaps'][-1]:05d}.bin").is_file()

def archive_attempt(st, why):
    d = OUT / "attempts" / f"attempt{st['attempt']}_{dt.datetime.now().strftime('%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    for x in (SNAPDIR, LOGDIR):
        if x.exists(): shutil.move(str(x), str(d / x.name))
    event(st, f"attempt {st['attempt']} 作廢（{why}），部分輸出移到 {d}")
    st["attempt"] += 1

def kill_train(st):
    pid = st.get("train_pid")
    if pid and alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not alive(pid): break
            time.sleep(0.5)
        if alive(pid): os.kill(pid, signal.SIGKILL)

def run_evals(st):
    EVALDIR.mkdir(parents=True, exist_ok=True)
    st.setdefault("evals", {})
    for t in CFG["snaps"]:
        name = f"s2_t{t:05d}"; csvp = EVALDIR / f"{name}.csv"; dump = EVALDIR / f"dump_{name}.txt"; snap = SNAPDIR / f"{name}.bin"
        if csvp.exists() and not dump.exists() and st["evals"].get(name, {}).get("ok"): continue
        while True:
            fe = foreign_evals()
            if not fe and not in_quiet_window(): break
            st["phase"] = "WAITING_EVAL"; st["wait_reason"] = "；".join(fe[:3]) or "17:45–18:00"; save_state(st); write_status(st)
            time.sleep(CFG["poll_s"])
        st["phase"] = "EVALUATING"; st["wait_reason"] = ""; save_state(st); write_status(st)
        sub = lambda c: [x.format(snap=snap, csv=csvp, dump=dump) for x in c]
        pre = ["nice", "-n", "19", "taskset", "-c", CFG["cpu"], "env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
        t0 = now()
        r1 = subprocess.run(pre + sub(CFG["eval_cmd"]), cwd=WD, capture_output=True, text=True)
        r2 = subprocess.run(pre + sub(CFG["check_cmd"]), cwd=WD, capture_output=True, text=True) if r1.returncode == 0 else None
        ok = r1.returncode == 0 and r2 is not None and r2.returncode == 0
        overlap = foreign_evals()
        st["evals"][name] = dict(ok=ok, secs=round(now() - t0, 1), check=(r2.stdout.strip().splitlines() or [""])[-1] if r2 else r1.stderr[-300:],
                                 overlap=overlap)
        if ok: dump.unlink(missing_ok=True)
        event(st, f"eval {name} {'OK' if ok else 'FAIL'} {now()-t0:.1f}s {st['evals'][name]['check']}" + (f"；結束時有他人 eval：{overlap}" if overlap else ""))
        save_state(st)
        if not ok: return False
    return True

# ---------- results ----------
def lab(t): return f"{t}s" if t < 60 else f"{t // 60}m"
def write_results(st):
    r = subprocess.run(["nice", "-n", "19", "python3", CFG["summarize"]], cwd=OUT, capture_output=True, text=True)
    if r.returncode != 0: event(st, "summarize 失敗：" + r.stderr[-300:]); return False
    table = r.stdout.strip()
    S2 = json.load(open(EVALDIR / "summary.json")); S1 = json.load(open(CFG["seed1_summary"]))
    err = (LOGDIR / "train_s2.err").read_text().splitlines(); endl = next((l for l in err if l.startswith("END")), "")
    kv = dict(x.split("=") for x in endl.split()[1:]) if endl else {}
    T = float(kv.get("train_s", "nan")); G = int(kv.get("games", 0)); U = int(kv.get("updates", 0))
    t60 = S2.get(f"s2_t{CFG['snaps'][-1]:05d}", {}); s160 = S1.get("s1_t03600", {})
    passed = t60.get("mean_score", 0) > 15000 and t60.get("reach1024", 0) >= 0.8
    g = st.get("guard", {}); mr = g.get("min_ratio", {})
    L = [BEGIN, "", f"## 5. seed 2 補跑（PM 10-18 17:28 核准；結果由 `rerun_s2/s2_rerun.py` 自動寫入，{ts()} 台北）", "",
         "### 5.1 執行紀錄",
         f"- **seed 2 補跑**：CPU {CFG['cpu']}，nice 19，單執行緒，同一個 `train` 執行檔、同 PREREG（§1–§4）、同 10 個快照點。"
         f"{ts(st['train_start'])} 開跑（attempt {st['attempt']}），{ts(st.get('train_end', now()))} 結束。訓練時間 {T:,.2f} 秒，共 {G:,} 局、{U/1e8:.1f} 億次 TD 更新，平均 {U/T/1e4:.0f} 萬次更新/秒、{G/T:.0f} 局/秒。",
         f"- 快照在 `rerun_s2/snapshots/`，eval 在 `rerun_s2/eval/`；先前被吞吐守門停掉的 attempt 歸檔在 `rerun_s2/attempts/`，未覆蓋。",
         f"- 開跑閘門（v2，PM 10-22）：Round 3 三批（{len(CFG['r3_jobs'])} 個 job）全部完成含最終 eval、R3 process 都不在、開跑前 {CFG['idle_wall_s']//60} 分鐘 load1 皆 < {CFG['idle_load1']}、沒有他人 eval。",
         "- baseline（env steps／active 秒）：" + ("、".join(f"{j} {b:.1f}" for j, b in st.get("baseline", {}).items()) or "無（開跑時 R3 已結束）") + "。",
         f"- 吞吐守門（每分鐘、5 分鐘移動平均，任一 arm 連續 {CFG['guard_consecutive']} 次 < baseline×{1-CFG['guard_drop']:.2f} 就停）：共 {g.get('checks', 0)} 次檢查、{g.get('excluded', 0)} 次因他人 eval 排除，"
         "最低比值 " + "、".join(f"{j} {(v-1)*100:+.1f}%" for j, v in mr.items()) + "；未觸發。",
         "- 訓練期間碰到的他人 eval：" + ("；".join(st.get("train_overlaps", [])[:10]) or "無") + "。",
         "- Eval：" + "；".join(f"{k} {v['secs']}s {v['check']}" for k, v in st.get("evals", {}).items()) + "。" +
         (" eval 結束時偵測到他人 eval：" + "；".join(f"{k}" for k, v in st.get("evals", {}).items() if v.get("overlap")) if any(v.get("overlap") for v in st.get("evals", {}).values()) else " 全程沒有和他人 eval 重疊。"),
         *([f"- 異常／備註：{'；'.join(st['anomalies'])}"] if st.get("anomalies") else []), "",
         "### 5.2 正式判定（t60，seed 2，1,000 局）", "",
         "| 指標 | seed 2 t60 | seed 1 t60 | 專案目標 | seed 2 結果 |", "|---|---|---|---|---|",
         f"| 平均分 | **{t60.get('mean_score',0):,.0f}** | {s160.get('mean_score',0):,.0f} | > 15,000 | {'✅' if t60.get('mean_score',0) > 15000 else '❌'} |",
         f"| 中位數 | {t60.get('median_score',0):,.0f} | {s160.get('median_score',0):,.0f} | — | |",
         *[f"| 到 {k} | {'**' if k == 1024 else ''}{100*t60.get(f'reach{k}',0):.1f}%{'**' if k == 1024 else ''} | {100*s160.get(f'reach{k}',0):.1f}% | {'≥ 80%' if k == 1024 else '—'} | {('✅' if t60.get('reach1024',0) >= 0.8 else '❌') if k == 1024 else ''} |" for k in (1024, 2048, 4096, 8192)],
         f"| 最高分 | {t60.get('max_score',0):,.0f} | {s160.get('max_score',0):,.0f} | — | |",
         f"| 角落占比（全部步數） | {100*(t60.get('corner_rate') or 0):.1f}% | {100*(s160.get('corner_rate') or 0):.1f}% | 只是診斷 | |",
         f"| 角落占比（最大方塊 ≥256 之後） | {100*(t60.get('corner_rate_256') or 0):.1f}% | {100*(s160.get('corner_rate_256') or 0):.1f}% | 只是診斷 | |", "",
         f"**seed 2 {'達標' if passed else '未達標'}。**（每個 seed 各自判定，不合併。）", "",
         "### 5.3 seed 2 補跑各快照（格式同 §3）", "", table, "",
         "### 5.4 兩個 seed 比較（seed 2 補跑 − seed 1）", "",
         "| 快照 | seed1 平均 | seed2 平均 | 差 | seed1 ≥1024 | seed2 ≥1024 | seed1 ≥2048 | seed2 ≥2048 | seed1 corner≥256 | seed2 corner≥256 | 差(pp) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in CFG["snaps"]:
        a, b = S1.get(f"s1_t{t:05d}"), S2.get(f"s2_t{t:05d}")
        if not a or not b: continue
        L.append(f"| {lab(t)} | {a['mean_score']:,.0f} | {b['mean_score']:,.0f} | {(b['mean_score']/a['mean_score']-1)*100:+.1f}% | {100*a['reach1024']:.1f}% | {100*b['reach1024']:.1f}% | "
                 f"{100*a['reach2048']:.1f}% | {100*b['reach2048']:.1f}% | {a['corner_rate_256']:.3f} | {b['corner_rate_256']:.3f} | {100*(b['corner_rate_256']-a['corner_rate_256']):+.1f} |")
    L += ["", END, ""]
    md = Path(CFG["results_md"]); txt = md.read_text() if md.exists() else ""
    if BEGIN in txt and END in txt:
        txt = txt[:txt.index(BEGIN)] + "\n".join(L) + txt[txt.index(END) + len(END):].lstrip("\n")
    else:
        txt = txt.rstrip("\n") + "\n\n" + "\n".join(L)
    tmp = md.with_suffix(".md.tmp"); tmp.write_text(txt); os.replace(tmp, md)
    st["result"] = dict(t60_mean=t60.get("mean_score"), t60_r1024=t60.get("reach1024"), t60_r2048=t60.get("reach2048"),
                        t60_cr256=t60.get("corner_rate_256"), passed=passed, games=G, updates=U)
    event(st, f"NTUPLE_RESULTS.md §5 已寫入：t60 mean {t60.get('mean_score',0):,.0f}、≥1024 {100*t60.get('reach1024',0):.1f}% → {'達標' if passed else '未達標'}")
    return True

# ---------- status ----------
RESTART = ("nohup setsid nice -n 19 bash /workspace/rl2048-review/corner/ntuple/rerun_s2/s2_keepalive.sh "
           ">> /workspace/rl2048-review/corner/ntuple/rerun_s2/s2_keepalive.out 2>&1 < /dev/null &")
def write_status(st, why=None):
    s = st.get("samples", [])[-1:] or [{}]; s = s[0]
    L = [f"# S2_STATUS：n-tuple seed 2 補跑（更新：{ts()} 台北）", "",
         f"狀態：**{st['phase']}**（attempt {st.get('attempt', 1)}）" + ("　**GATE_ANOMALY**" if st.get("gate_anomaly") else ""), ""]
    if st["phase"] == "WAITING_GATE" and why: L += ["開跑閘門尚未通過：", *[f"- {w}" for w in why], ""]
    if st.get("wait_reason"): L += [f"等待原因：{st['wait_reason']}", ""]
    if st.get("baseline"): L += ["baseline（env steps／active 秒，開跑前 15 分鐘）：" + "、".join(f"{j} {b:.1f}" for j, b in st["baseline"].items()), ""]
    if st.get("train_start"): L += [f"seed 2 開跑：{ts(st['train_start'])}；pid {st.get('train_pid')}" + (f"；結束 {ts(st['train_end'])}" if st.get("train_end") else ""), ""]
    if st.get("last_guard"): L += [f"最近一次守門：{st['last_guard']}", ""]
    if s: L += [f"最近取樣 {ts(s.get('t'))}：load1 {s.get('load1')}；R3 arm 5 分鐘 MA " + "、".join(f"{k} {'NA' if v is None else f'{v:.1f}'}" for k, v in s.get("arms", {}).items()) +
                "；其他訓練 " + ("、".join(f"{k} {'NA' if v is None else f'{v:.1f}'}" for k, v in s.get("other", {}).items()) or "無") + f"；他人 eval {len(s.get('fe', []))} 個", ""]
    if st.get("evals"): L += ["Eval：", *[f"- {k}：{'OK' if v['ok'] else 'FAIL'} {v['secs']}s {v['check']}" for k, v in st["evals"].items()], ""]
    if st.get("result"): L += [f"結果：t60 mean {st['result']['t60_mean']:,.0f}、≥1024 {100*st['result']['t60_r1024']:.1f}% → {'達標' if st['result']['passed'] else '未達標'}（詳見 NTUPLE_RESULTS.md §5）", ""]
    L += ["最近事件：", *[f"- {e}" for e in st.get("events", [])[-12:]], "",
          "## 指令", f"- 看狀態：`cat {STATUS}`；`tail -n 30 {LOG}`；每分鐘吞吐 `{MON}`",
          f"- （重新）啟動／box 凍結後恢復（有 flock，重複啟動會被擋；會自動接手還活著的訓練，或作廢中斷的 attempt 重來）：`{RESTART}`",
          f"- 取消：`kill $(cat {OUT}/s2_keepalive.pid) $(cat {OUT}/s2_rerun.pid)`；訓練 pid 見上（只在要停 seed 2 時 kill）",
          f"- 輸出：快照 `{SNAPDIR}`、eval `{EVALDIR}`、訓練 log `{LOGDIR}`；結果寫進 `{CFG['results_md']}` 的 §5（標記 S2_RERUN_BEGIN/END）"]
    tmp = STATUS.with_suffix(".tmp"); tmp.write_text("\n".join(L) + "\n"); os.replace(tmp, STATUS)

# ---------- main loop ----------
def main_loop():
    st = load_state(); last = None
    st.pop("wait_reason", None); st.pop("_adopt_checked", None)
    if st["phase"] == "STOPPED_THROUGHPUT" and st.get("gate_version", 1) < CFG["gate_version"]:
        # PM 10-22：attempt 1 被 v1 守門停掉 → 作廢歸檔，改用 v2 閘門（R3 全完成＋機器空閒）重新等待
        st.setdefault("history", []).append({k: st.get(k) for k in ("attempt", "phase", "train_start", "train_end", "train_pid", "baseline", "guard", "last_guard", "train_overlaps")})
        archive_attempt(st, "v1 吞吐守門停止；改用 v2 閘門")
        for k in ("baseline", "guard", "last_guard", "train_overlaps", "train_pid", "train_start", "train_end", "train_cmd", "r3_missing_since", "gate_anomaly"): st.pop(k, None)
        st["phase"] = "WAITING_GATE"; st["gate_version"] = CFG["gate_version"]
        event(st, f"閘門改為 v2（R3 三批全部完成含最終 eval、R3 process 都不在、load1 < {CFG['idle_load1']} 持續 {CFG['idle_wall_s']//60} 分鐘），等待開跑 attempt {st['attempt']}")
        save_state(st)
    st.setdefault("gate_version", CFG["gate_version"])
    if st["phase"] in ("DONE", "STOPPED_THROUGHPUT", "FAILED"):
        write_status(st); log(f"已是終態 {st['phase']}，不動作"); return 0
    event(st, f"s2_rerun 啟動 pid {os.getpid()}，phase {st['phase']}")
    while True:
        t = now()
        if last and t - last > CFG["freeze_gap_s"]:
            event(st, f"偵測到 {t-last:.0f}s 的取樣空檔（box 可能凍結）")
            if st["phase"] == "TRAINING": st.setdefault("anomalies", []).append(f"{ts()} 訓練中有 {t-last:.0f}s 取樣空檔（train 用 CLOCK_MONOTONIC）")
        last = t
        ph = st["phase"]
        procs, fe, arms, s = sample(st, ph)
        if ph == "WAITING_GATE":
            if st["attempt"] > CFG["max_attempts"]:
                st["phase"] = "FAILED"; event(st, "超過最大 attempt 次數"); save_state(st); write_status(st); return 0
            ok, why = gate(st, procs, fe, arms)
            ga = check_r3_missing(st, procs, arms)
            if ga: why = [f"⚠ 閘門異常：{ga['why']}（自 {ga['since']}；已寫里程碑回報）", *why]
            if ok:
                st["baseline"] = {j: a["base"] for j, a in arms.items() if a["alive"] and a["base"]}  # v2：R3 已結束 → 通常為空，守門無對象
                event(st, f"閘門通過（v2）：R3 全部完成、無 R3 process、load1 {s['load1']}；baseline " + ("、".join(f"{j} {b:.1f}" for j, b in st["baseline"].items()) or "無（R3 已不在，守門無可比對象）"))
                if CFG["dry_run"]:
                    st["phase"] = "DRY_RUN_GATE_OPEN"; save_state(st); write_status(st); return 0
                start_train(st); st["guard"] = {"breach": {}, "excluded": 0, "checks": 0, "min_ratio": {}}; st["train_overlaps"] = []
            save_state(st); write_status(st, None if ok else why)
        elif ph == "TRAINING":
            pid = st["train_pid"]
            if alive(pid) and not st.get("_adopt_checked"):
                try: ca = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except OSError: ca = []
                if not (any(Path(x.decode()).name == Path(CFG["train_cmd"][0]).name for x in ca[:2]) and str(SNAPDIR).encode() in ca):
                    event(st, f"pid {pid} 不是 seed 2 訓練（pid 重用），視為中斷"); pid = st["train_pid"] = -1
                else: event(st, f"確認 seed 2 訓練 pid {pid}（接手或首次）")
                st["_adopt_checked"] = True
            if fe:
                for x in fe:
                    if x not in st["train_overlaps"]: st["train_overlaps"].append(x); event(st, f"訓練中碰到他人 eval：{x}")
            if not alive(pid):
                if train_finished_ok():
                    st["train_end"] = now(); st["phase"] = "WAITING_EVAL"; event(st, "seed 2 訓練完成：" + (LOGDIR / "train_s2.err").read_text().strip().splitlines()[-1])
                else:
                    archive_attempt(st, "訓練 process 中斷"); st["phase"] = "WAITING_GATE"
            else:
                stop, note = guard_check(st, arms); st["last_guard"] = f"{ts()} {note}"; log("guard " + note)
                if stop:
                    kill_train(st); st["phase"] = "STOPPED_THROUGHPUT"; st["train_end"] = now()
                    event(st, f"吞吐守門觸發，已停掉 seed 2：{note}"); save_state(st); write_status(st); return 0
            save_state(st); write_status(st)
        elif ph in ("WAITING_EVAL", "EVALUATING"):
            if not run_evals(st):
                st["phase"] = "FAILED"; event(st, "eval 失敗"); save_state(st); write_status(st); return 0
            st["phase"] = "WRITING"; save_state(st)
            if write_results(st):
                st["phase"] = "DONE"; event(st, "DONE")
            else:
                st["phase"] = "FAILED"
            save_state(st); write_status(st); return 0
        elif ph == "WRITING":
            st["phase"] = "EVALUATING"; continue
        time.sleep(max(1, CFG["poll_s"] - (now() - t)))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "status":
        print(STATUS.read_text() if STATUS.exists() else "no status"); sys.exit(0)
    lk = open(OUT / "s2_rerun.lock", "w")
    try: fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: print("s2_rerun 已在執行（lock 被占用）"); sys.exit(3)
    (OUT / "s2_rerun.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    try: sys.exit(main_loop())
    except SystemExit: raise
    except Exception as e:
        import traceback; log("例外：" + traceback.format_exc().replace("\n", " | ")); sys.exit(1)
