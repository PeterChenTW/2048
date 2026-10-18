"""ROUND3 排程器測試：所有輸出在 tmp_path；Popen／git／flagcheck 都被替換，不啟動訓練或 eval。
旗標整合測試以唯讀方式使用固定 worktree 的 parser（nice 19，不訓練）。"""
import json, os, shlex, subprocess, sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_run  # noqa: E402

CFG = json.loads((HERE / "r3_config.json").read_text())
PY = CFG["python"]


def good_eval(mean=10000.0, r1024=0.5):
    return dict(mean=mean, median=mean, max=mean * 2, reach_rates={"512": 0.9, "1024": r1024, "2048": 0.0, "4096": 0.0},
                per_game=[])


def good_train(updates=300000, active_s=10800.0, env_steps=700000):
    return dict(updates=updates, env_steps=env_steps, active_s=active_s, episodes=1500, nan_count=0, interrupted=False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in dict(R3_STATE=tmp_path / "state.json", R3_RUNS=tmp_path / "runs", R3_STATUS=tmp_path / "STATUS.md",
                     R3_LOG=tmp_path / "log", R3_MILESTONES=tmp_path / "miles.json", R3_S_RUNS=tmp_path / "s_runs",
                     R3_CONTROLS_ROOT=tmp_path / "controls", R3_REPO=tmp_path / "repo", R3_NO_PUBLISH="1").items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("R3_DRY_RUN", raising=False)
    root = tmp_path / "controls"
    for seed, ctl in CFG["controls"].items():
        d = root / ctl["name"]
        (d / Path(ctl["eval"]).parent).mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(dict(script="python main.py train --seed " + seed)))
    return tmp_path


class Fake:
    def __init__(self, tmp, monkeypatch, flags_ok=True, commit=None, porcelain=""):
        self.t = 1_000_000.0
        self.busy_args = []
        self.load1 = 0.5
        self.spawned = []
        self.git = []
        self.flag_calls = 0
        self.flags_ok = flags_ok
        self.commit = commit or CFG["expected_commit"]
        self.porcelain = porcelain
        S = r3_run.Scheduler
        monkeypatch.setattr(S, "now_epoch", lambda s: self.t)
        monkeypatch.setattr(S, "loadavg", lambda s: self.load1)
        monkeypatch.setattr(S, "process_args", lambda s: list(self.busy_args))
        monkeypatch.setattr(S, "process_scope_ok", lambda s: True)
        monkeypatch.setattr(S, "spawn", lambda s, args, out, threads=1: self._spawn(args))
        monkeypatch.setattr(S, "run", lambda s, args, **kw: self._run(list(map(str, args))))
        monkeypatch.setattr(r3_run.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Popen")))

    def _spawn(self, args):
        self.spawned.append(args)
        return type("P", (), {"pid": 4242 + len(self.spawned)})()

    def _run(self, args):
        if args[0] == "git":
            self.git.append(args)
            if "rev-parse" in args and "HEAD" in args:
                return subprocess.CompletedProcess(args, 0, stdout=self.commit + "\n", stderr="")
            if "status" in args:
                return subprocess.CompletedProcess(args, 0, stdout=self.porcelain, stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if any(a.endswith("r3_flagcheck.py") for a in args):
            self.flag_calls += 1
            spec = json.loads(Path(args[-1]).read_text())
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({n: dict(ok=self.flags_ok, arm_config_diff={}) for n in spec}), stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def s_finish(tmp, names=("S_s101", "S_s202", "S_s303", "S_s404")):
    for n in names:
        d = tmp / "s_runs" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "eval.json").write_text(json.dumps(good_eval()))
        (d / "eval_end").write_text("1")


def control_finish(tmp, seeds=("101", "202", "303"), mean=10000.0, r1024=0.5, ups_active=10800.0):
    for seed in seeds:
        ctl = CFG["controls"][seed]
        d = tmp / "controls" / ctl["name"]
        (d / ctl["eval"]).write_text(json.dumps(good_eval(mean, r1024)))
        (d / "train.json").write_text(json.dumps(good_train(active_s=ups_active)))
        for f in ctl["require"]:
            (d / f).write_text("1")


def tick(fake, dt=31):
    st = r3_run.Scheduler().tick()
    fake.t += dt
    return st


def test_waits_until_s_done_and_never_launches_early(env, monkeypatch):
    f = Fake(env, monkeypatch)
    control_finish(env)
    for _ in range(5):
        st = tick(f)
    assert not f.spawned and not st.get("started")
    assert "等待中" in (env / "STATUS.md").read_text()
    s_finish(env, names=("S_s101", "S_s202", "S_s303"))  # S_s404 尚未完成
    for _ in range(3):
        tick(f)
    assert not f.spawned


def test_waits_for_control_s303(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env)
    for _ in range(4):
        tick(f)
    assert not f.spawned


def test_launch_batch1_after_two_idle_checks(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    tick(f)
    assert not f.spawned  # 第一次空閒只計 1/2
    st = tick(f)
    assert len(f.spawned) == 3 and st["batches"]["1"]["started"]
    names = {Path(shlex.split(a[-1])[shlex.split(a[-1]).index("--save-as") + 1]).parent.name for a in f.spawned}
    assert names == {"G_s303", "Y_s303", "GY_s303"}
    for a in f.spawned:
        assert a[:8] == ["setsid", "nohup", "taskset", "-c", "0-5", "nice", "-n", "10"]
        script = a[-1]
        assert "--threads 2 --minutes 180" in script and "eval.py" in script and "r3_corner.py" in script
        assert script.index("eval_end") < script.index("r3_corner.py")
    g = next(a[-1] for a in f.spawned if "/G_s303/" in a[-1])
    assert "--reward-mode log --gamma 0.99" in g and g.index("--reward-mode raw --gamma 1.0 --value-rescale h") > g.index("--budget-clock active")
    assert "--sym-aug" not in g
    y = next(a[-1] for a in f.spawned if "/Y_s303/" in a[-1])
    assert "--sym-aug random" in y and "--value-rescale" not in y and "raw" not in y
    meta = json.loads((env / "runs" / "G_s303" / "meta.json").read_text())
    assert meta["commit"] == CFG["expected_commit"] and meta["launcher_pid"]
    for _ in range(3):
        tick(f)
    assert len(f.spawned) == 3  # 不會重複開跑，第二批也不會提前


def test_busy_and_load_reset_idle(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    tick(f)
    f.busy_args = [["python", "/x/eval.py", "--checkpoint", "c"]]
    tick(f)
    f.busy_args = []
    tick(f)
    assert not f.spawned
    f.load1 = 3.5
    tick(f); tick(f)
    assert not f.spawned
    f.load1 = 0.2
    f.busy_args = [["/venv/python", "main.py", "train", "--seed", "1"]]
    tick(f); tick(f)
    assert not f.spawned
    f.busy_args = []
    tick(f); tick(f)
    assert len(f.spawned) == 3


def test_idle_checks_must_be_30s_apart(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    tick(f, dt=5); tick(f, dt=5); tick(f, dt=5)
    assert not f.spawned


def test_flag_mismatch_blocks(env, monkeypatch):
    f = Fake(env, monkeypatch, flags_ok=False)
    s_finish(env); control_finish(env)
    for _ in range(4):
        tick(f)
    assert not f.spawned
    ms = json.loads((env / "miles.json").read_text())
    assert any("旗標不一致" in m["text"] for m in ms)


def test_commit_mismatch_and_dirty_worktree_block(env, monkeypatch):
    for kw in (dict(commit="deadbeef"), dict(porcelain=" M main.py\n")):
        f = Fake(env, monkeypatch, **kw)
        s_finish(env); control_finish(env)
        st_path = env / "state.json"
        st_path.unlink(missing_ok=True)
        for _ in range(3):
            st = tick(f)
        assert not f.spawned and "啟動失敗" in st.get("launch_error", "")


def test_existing_run_dir_blocks(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    (env / "runs" / "Y_s303").mkdir(parents=True)
    for _ in range(3):
        st = tick(f)
    assert not f.spawned and st.get("launch_error")


def test_dry_run_never_launches(env, monkeypatch):
    monkeypatch.setenv("R3_DRY_RUN", "1")
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    for _ in range(4):
        tick(f)
    assert not f.spawned and not (env / "state.json").exists() and (env / "state.dry_run.json").exists()


def finish_job(env, name, mean, r1024, updates=300000):
    d = env / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(json.dumps(good_eval(mean, r1024)))
    (d / "train.json").write_text(json.dumps(good_train(updates=updates)))
    (d / "eval_end").write_text("1")
    (d / "corner.json").write_text(json.dumps(dict(summary=dict(corner_rate=0.1, corner_rate_256=0.1, center_rate_256=0.4,
                                                                 leaves_per100=2.0, longest_streak=20, final_mono_snake=0.6,
                                                                 spearman_cr256_score=0.3), eval_match=True)))


def test_batches_chain_in_order_and_verdicts(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    tick(f); tick(f)
    assert len(f.spawned) == 3
    # 批 1 跑完前不會開第 2 批
    for n in ("G_s303", "Y_s303"):
        finish_job(env, n, 13000, 0.65)
    tick(f); tick(f)
    assert len(f.spawned) == 3
    finish_job(env, "GY_s303", 11000, 0.55)
    st = tick(f)
    assert st["batches"]["1"].get("done")
    tick(f); st = tick(f)
    assert len(f.spawned) == 6 and st["batches"]["2"]["started"]
    assert all("_s202/" in a[-1] for a in f.spawned[3:])
    for n in ("G_s202", "Y_s202", "GY_s202"):
        finish_job(env, n, 12500, 0.62, updates=270000)
    for _ in range(3):
        tick(f)
    assert len(f.spawned) == 9 and all("_s101/" in a[-1] for a in f.spawned[6:])
    finish_job(env, "G_s101", 12100, 0.61)
    finish_job(env, "Y_s101", 11000, 0.70)  # ratio 1.10：不過
    finish_job(env, "GY_s101", 13000, 0.66)
    st = tick(f)
    assert st.get("done")
    vs = r3_run.Scheduler().verdicts(st)
    assert vs["arms"]["G"]["adopt"] and vs["arms"]["G"]["n_pass"] == 3
    assert not vs["arms"]["Y"]["adopt"] and vs["arms"]["Y"]["n_pass"] == 2
    y101 = next(r for r in vs["arms"]["Y"]["rows"] if r["seed"] == "101")
    assert not y101["pass"] and "可能受運算量偏差影響" in y101["bias_note"]
    gy303 = next(r for r in vs["arms"]["GY"]["rows"] if r["seed"] == "303")
    assert not gy303["pass"]  # ×1.10
    g202 = next(r for r in vs["arms"]["G"]["rows"] if r["seed"] == "202")
    assert g202["bias_note"] == "" and abs(g202["diag"]["updates"] - 0.9) < 1e-9
    sp = vs["arms"]["G"]["spread"]["ratio"]
    assert abs(sp["mean"] - (1.3 + 1.25 + 1.21) / 3) < 1e-9 and sp["sd"] > 0
    i303 = next(i for i in vs["interaction"] if i["seed"] == "303")
    assert abs(i303["mean"] - (11000 - 13000 - 13000 + 10000)) < 1e-9
    ms = json.loads((env / "miles.json").read_text())
    assert sum("round3 第" in m["text"] for m in ms) == 3 and any(m["text"].startswith("round3 完成") for m in ms)
    text = (env / "STATUS.md").read_text()
    assert "交互作用" in text and "角落診斷" in text and "配對差" in text
    # 每個 arm–C 比較都附 updates/s 比值與偏離註記（PM 16:43）
    for arm in ("G", "Y", "GY"):
        sec = text.split(f"### {arm}：")[1].split("###")[0].split("\n## ")[0]
        assert "updates/s ×" in sec and "偏離紀錄（PM 10-18 16:43 核准）" in sec and "運算量校正的參考" in sec
        for line in [l for l in sec.splitlines() if l.startswith("| ") and l[2:5].strip().isdigit()]:
            assert line.split("|")[11].strip() not in ("", "—")  # updates/s 比值欄有數字


def test_missing_data_is_not_pass(env, monkeypatch):
    f = Fake(env, monkeypatch)
    control_finish(env)
    vs = r3_run.Scheduler().verdicts({"batches": {}})
    assert all(not v["adopt"] and v["all_pending"] for v in vs["arms"].values())


def test_publish_commits_path_and_pushes_without_force(env, monkeypatch):
    monkeypatch.delenv("R3_NO_PUBLISH")
    f = Fake(env, monkeypatch)
    (env / "repo").mkdir()
    orig = f._run
    def run(args):
        if args[0] == "git" and "branch" in args:
            return subprocess.CompletedProcess(args, 0, stdout=CFG["branch"] + "\n", stderr="")
        if args[0] == "git" and "--absolute-git-dir" in args:
            return subprocess.CompletedProcess(args, 0, stdout=str(env / "gitdir") + "\n", stderr="")
        if args[0] == "git" and "diff" in args:
            f.git.append(args)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return orig(args)
    monkeypatch.setattr(r3_run.Scheduler, "run", lambda s, a, **kw: run(list(map(str, a))))
    s_finish(env); control_finish(env)
    tick(f)
    assert not any("push" in g for g in f.git)  # 未開跑不發佈
    tick(f)
    assert any("push" in g for g in f.git)
    push = next(g for g in f.git if "push" in g)
    assert push[-2:] == ["origin", CFG["branch"]] and not any("force" in x or x == "-f" for x in push)
    commit = next(g for g in f.git if "commit" in g)
    assert commit[-2:] == ["--", CFG["results_subdir"]]
    assert (env / "repo" / CFG["results_subdir"] / "ROUND3_RESULTS.md").exists()


def test_anomaly_dead_job(env, monkeypatch):
    f = Fake(env, monkeypatch)
    s_finish(env); control_finish(env)
    tick(f); tick(f)
    st = tick(f)  # 程序不存在（process_args 為空）且沒有 train.json
    ms = json.loads((env / "miles.json").read_text())
    assert any("中斷" in m["text"] for m in ms)


# ---------- 旗標整合（唯讀 parser） ----------

def _flagcheck(tmp, spec):
    p = tmp / "spec.json"
    p.write_text(json.dumps(spec))
    out = subprocess.run(["nice", "-n", "19", PY, str(HERE / "r3_flagcheck.py"), CFG["code_dir"], str(p)],
                         capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _spec(job, arm_argv_mutator=None):
    s = r3_run.Scheduler.__new__(r3_run.Scheduler)
    s.cfg = CFG; s.runs = Path("/tmp/r3x"); s.py = PY; s.code = Path(CFG["code_dir"])
    j = next(x for x in CFG["jobs"] if x["name"] == job)
    s.batches = CFG["batches"]
    arm = s.train_argv(j)[2:]
    if arm_argv_mutator:
        arm = arm_argv_mutator(arm)
    ref = json.loads(Path(CFG["controls"][str(j["seed"])]["dir"], "meta.json").read_text())["script"]
    return {job: dict(arm_argv=arm, c_argv=s.train_argv(j, "C")[2:], ref_script=ref,
                      expected=CFG["expected_config_diff"][j["arm"]])}


@pytest.mark.parametrize("job", ["G_s303", "Y_s202", "GY_s101"])
def test_flagcheck_real_parser_ok(tmp_path, job):
    r = _flagcheck(tmp_path, _spec(job))[job]
    assert r["ok"], r
    assert set(r["arm_config_diff"]) == set(CFG["expected_config_diff"][job.split("_")[0]])


def test_flagcheck_detects_extra_and_missing(tmp_path):
    extra = _flagcheck(tmp_path, _spec("Y_s303", lambda a: a + ["--replay-ratio", "0.5"]))["Y_s303"]
    assert not extra["ok"] and "replay_ratio" in json.dumps(extra["unexpected_arm_cli"])
    missing = _flagcheck(tmp_path, _spec("G_s303", lambda a: [x for x in a if x not in ("--value-rescale", "h")]))["G_s303"]
    assert not missing["ok"] and "value_rescale" in missing["missing_arm_diff"]
    wrong = _flagcheck(tmp_path, _spec("G_s202", lambda a: a + ["--gamma", "0.999"]))["G_s202"]
    assert not wrong["ok"]
