"""S 排程器獨立測試。

輸出路徑全部落在 tmp_path。Popen 與 git 都被替換；旗標整合測試只唯讀執行固定 parser，
不會啟動訓練、eval，也不會對真實 git 發佈。
"""
import json
import math
import shlex
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_run  # noqa: E402

REAL_RUN = s_run.subprocess.run
REAL_POPEN = s_run.subprocess.Popen

CONFIG = Path("/workspace/rl2048-review/skip-run/s_config.json")
NOT_BEFORE = datetime.fromisoformat("2026-10-18T20:30:00+08:00").timestamp()
RESULTS = "results/round2_skip"
BRANCH = "exp/round2-skip"

# 修訂 1 的共同旗標（--seed／--threads／--minutes／--save-every 由排程器接在後面）。
COMMON = [
    "--arch", "onehot_afterstate",
    "--from-scratch",
    "--n-step", "1",
    "--memory-capacity", "6000",
    "--replay-ratio", "0.45",
    "--per-max-priority", "tree",
    "--reward-mode", "log",
    "--gamma", "0.99",
    "--eps-schedule", "episode_exp",
    "--eps-start", "0.15",
    "--eps-min", "0.001",
    "--eps-decay", "0.999",
    "--eps-decay-start-episode", "128",
    "--restart-frac", "0",
    "--budget-clock", "active",
]


class Proc:
    def __init__(self, pid):
        self.pid = pid


class GitBox:
    def __init__(self):
        self.calls = []
        self.popen_calls = []
        self.head = ""
        self.branch = BRANCH
        self.gitdir = None
        self.diff_code = 1
        self.push_code = 0
        self.diff_seq = None
        self.push_seq = None
        self.flag_diff = False

    def reset(self, gitdir):
        self.calls = []
        self.popen_calls = []
        self.head = "c0dc1c3ce4200d4316d006aacb66cfb1e1256f00"
        self.branch = BRANCH
        self.gitdir = gitdir
        self.diff_code = 1
        self.push_code = 0
        self.diff_seq = None
        self.push_seq = None
        self.flag_diff = False


GIT = GitBox()


@pytest.fixture(autouse=True)
def block_real_subprocesses(monkeypatch, tmp_path):
    """禁止真實 Popen／git／torch。管理指令若被呼叫，只記錄參數。"""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    GIT.reset(gitdir)

    def fake_run(args, **kwargs):
        argv = [str(a) for a in args]
        GIT.calls.append(argv)
        cmd = argv[3:] if argv[:3] == ["nice", "-n", "19"] else argv
        stdout = ""
        code = 0
        if cmd[:1] == ["git"] and len(cmd) >= 4 and cmd[1] == "-C":
            rest = cmd[3:]
            if rest[:2] == ["rev-parse", "HEAD"]:
                stdout = GIT.head + "\n"
            elif rest[:2] == ["rev-parse", "--absolute-git-dir"]:
                stdout = str(GIT.gitdir) + "\n"
            elif rest[:1] == ["branch"]:
                stdout = GIT.branch + "\n"
            elif rest[:3] == ["diff", "--cached", "--quiet"]:
                code = GIT.diff_seq.pop(0) if GIT.diff_seq else GIT.diff_code
            elif rest[:1] == ["push"]:
                code = GIT.push_seq.pop(0) if GIT.push_seq else GIT.push_code
            elif rest[:1] in ("add", "commit"):
                code = 0
        elif len(cmd) > 1 and Path(cmd[1]).name == "flagcheck.py":
            spec = json.loads(Path(cmd[-1]).read_text())
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            stdout = json.dumps({name: dict(ok=not GIT.flag_diff,
                                           diff={"gamma": [0.99, 0.98]} if GIT.flag_diff else {},
                                           config_diff={"gamma": [0.99, 0.98]} if GIT.flag_diff else {}) for name in spec})
        elif any("torch.load" in a for a in cmd):
            code = 0
        return s_run.subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")

    def fake_popen(args, **kwargs):
        pid = 5000 + len(GIT.popen_calls)
        stdout = kwargs.get("stdout")
        env = kwargs.get("env") or {}
        GIT.popen_calls.append({
            "args": [str(a) for a in args],
            "cwd": kwargs.get("cwd"),
            "env": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if k in env},
            "stdout_name": getattr(stdout, "name", None),
            "stdin": kwargs.get("stdin"),
            "stderr": kwargs.get("stderr"),
            "start_new_session": kwargs.get("start_new_session"),
            "pid": pid,
        })
        return Proc(pid)

    monkeypatch.setattr(s_run.subprocess, "run", fake_run)
    monkeypatch.setattr(s_run.subprocess, "Popen", fake_popen)


def make_scheduler(tmp_path, monkeypatch, *, dry_run=False, publish=False, quad="done", stub_host=True):
    runs = tmp_path / "runs"
    repo = tmp_path / "repo"
    code = tmp_path / "code"
    for folder in (runs, repo, code):
        folder.mkdir()
    quad_path = tmp_path / "quad_state.json"
    if quad == "done":
        quad_path.write_text(json.dumps({"done": "2026-10-18 21:00:00"}) + "\n")
    elif quad == "missing":
        pass
    elif quad == "empty":
        quad_path.write_text("{}\n")
    elif quad == "false":
        quad_path.write_text(json.dumps({"done": False}) + "\n")
    elif quad == "null":
        quad_path.write_text(json.dumps({"done": None}) + "\n")
    else:
        quad_path.write_text(quad)
    env = {
        "S2_CONTROLS_ROOT": str(tmp_path / "controls"),
        "S2_CONFIG": str(CONFIG),
        "S2_STATE": str(tmp_path / "s_state.json"),
        "S2_RUNS": str(runs),
        "S2_MILESTONES": str(tmp_path / "milestones.json"),
        "S2_MILESTONES_LOCK": str(tmp_path / ".milestones.lock"),
        "S2_CODE": str(code),
        "S2_REPO": str(repo),
        "S2_STATUS": str(tmp_path / "S_STATUS.md"),
        "S2_LOG": str(tmp_path / "s_run.log"),
        "S2_QUAD_STATE": str(quad_path),
        "S2_DRY_RUN": "1" if dry_run else "0",
        "S2_NO_PUBLISH": "0" if publish else "1",
        "S2_PUBLISH": "1" if publish else "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sched = s_run.Scheduler()
    GIT.head = sched.cfg["expected_commit"]
    sched._busy = False
    if stub_host:
        sched.machine_busy = lambda: sched._busy
        sched.job_alive = lambda name: False
        sched.process_args = lambda: []
    assert sched.runs == runs
    assert sched.repo == repo
    assert sched.code == code
    assert sched.state_path == tmp_path / "s_state.json"
    assert sched.miles == tmp_path / "milestones.json"
    assert sched.status_path == tmp_path / "S_STATUS.md"
    assert sched.log_path == tmp_path / "s_run.log"
    assert sched.quad_state == quad_path
    assert sched.dry_run is dry_run
    assert str(tmp_path) in str(sched.miles_lock)
    for seed, control in sched.controls.items():
        plant(sched, control["name"], mean=10000, r1024=0.5, updates=1000)
        script = sched.script(sched.jobs[f"S_s{seed}"]).replace("--value-skip l1 ", "")
        (Path(control["dir"]) / "meta.json").write_text(json.dumps({"script": script}))
    return sched


def tick_at(sched, when):
    sched.now_epoch = lambda when=when: when
    return sched.tick()


def assert_not_launched(sched, state):
    assert GIT.popen_calls == []
    assert state.get("pids", {}) == {}
    assert "1" not in state.get("batches", {})
    assert list(sched.runs.iterdir()) == []


def git_cmds():
    found = []
    for args in GIT.calls:
        if args[:3] == ["nice", "-n", "19"] and len(args) >= 4 and args[3] == "git":
            found.append(args[3:])
    return found


def split_script(script):
    parts = [part.strip() for part in script.split("&&")]
    assert len(parts) == 4, script
    train = shlex.split(parts[0].split(" > ", 1)[0])
    eval_tokens = shlex.split(parts[2].split(" > ", 1)[0])
    assert eval_tokens[:2] == ["OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
    return parts, train, eval_tokens[2:]


def expected_train(sched, job):
    dest = sched.runs / job["name"]
    flags = list(COMMON)
    if job["arm"] == "S":
        flags += ["--value-skip", "l1"]
    else:
        assert job["arm"] == "C"
    return [
        sched.cfg["python"], "main.py", "train", *flags,
        "--seed", str(job["seed"]),
        "--threads", "2",
        "--minutes", "180",
        "--save-every", "1",
        "--save-as", str(dest / "ckpt"),
        "--log", str(dest / "train.csv"),
    ]


def expected_eval(sched, job):
    dest = sched.runs / job["name"]
    return [
        "timeout", "--signal=TERM", "--kill-after=30s", "30m",
        sched.cfg["python"], "eval.py",
        "--checkpoint", str(dest / "ckpt"),
        "--policy", "greedy",
        "--games", "100",
        "--seed-start", "1000",
        "--board", "fast",
        "--max-steps", "5000",
        "--json", str(dest / "eval.json"),
    ]


def assert_launch(call, sched, job, cpus):
    args = call["args"]
    script = sched.script(job)
    assert args == ["setsid", "nohup", "taskset", "-c", cpus, "nice", "-n", "10", "bash", "-c", script]
    assert args == sched.launch_command(job)
    parts, train, eval_argv = split_script(args[-1])
    dest = sched.runs / job["name"]
    assert train == expected_train(sched, job)
    assert eval_argv == expected_eval(sched, job)
    assert parts[0].endswith(f"> {shlex.quote(str(dest / 'train.out'))} 2>&1")
    assert parts[1] == f"date +%s > {shlex.quote(str(dest / 'eval_start'))}"
    assert parts[2].endswith(f"> {shlex.quote(str(dest / 'eval.out'))} 2>&1")
    assert parts[3] == f"date +%s > {shlex.quote(str(dest / 'eval_end'))}"
    if job["arm"] == "S":
        assert train.count("--value-skip") == 1
        assert train[train.index("--value-skip") + 1] == "l1"
        assert "--value-skip" not in eval_argv
    else:
        assert "--value-skip" not in train
        assert "value-skip" not in args[-1]
    assert call["cwd"] == sched.code
    assert call["start_new_session"] is True
    assert call["stdin"] is s_run.subprocess.DEVNULL
    assert call["stderr"] is s_run.subprocess.STDOUT
    assert call["env"] == {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    assert call["stdout_name"] == str(dest / "launch.out")
    meta = json.loads((dest / "meta.json").read_text())
    assert meta["launcher_pid"] == call["pid"]
    assert meta["cpus"] == cpus
    assert meta["nice"] == 10
    assert meta["script"] == script
    assert meta["commit"] == sched.cfg["expected_commit"]
    assert meta["job"]["seed"] == job["seed"]
    assert meta["minutes"] == 180
    assert meta["threads"] == 2


def plant(sched, name, *, mean, r1024, updates, write_eval=True, write_train=True, write_end=True, extra_train=None, drop=()):
    control = next((c for c in sched.controls.values() if c["name"] == name), None)
    dest = Path(control["dir"]) if control else sched.runs / name
    eval_path = dest / control["eval"] if control else dest / "eval.json"
    dest.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    if write_eval:
        payload = {
            "mean": mean,
            "median": mean / 2,
            "reach_rates": {"1024": r1024, "2048": 0.01},
            "elapsed_seconds": 120,
        }
        for key in drop:
            if key == "r2048":
                del payload["reach_rates"]["2048"]
            elif key == "r1024":
                del payload["reach_rates"]["1024"]
            else:
                payload.pop(key, None)
        eval_path.write_text(json.dumps(payload))
    if write_train:
        train = {
            "updates": updates,
            "env_steps": 5000,
            "replay_ratio_actual": 0.45,
            "episodes": 80,
            "nan_count": 0,
            "active_s": 10800,
        }
        if extra_train:
            train.update(extra_train)
        if "updates" in drop:
            train.pop("updates", None)
        (dest / "train.json").write_text(json.dumps(train))
    if write_end:
        (dest / "eval_end").write_text("200\n")
        (dest / "eval_start").write_text("80\n")


def plant_seed(sched, seed, *, s_mean, c_mean, s_r, c_r, s_up, c_up, **kwargs):
    plant(sched, f"S_s{seed}", mean=s_mean, r1024=s_r, updates=s_up, **kwargs)
    plant(sched, sched.controls[str(seed)]["name"], mean=c_mean, r1024=c_r, updates=c_up, **kwargs)


def row_by_seed(result, seed):
    return next(row for row in result["rows"] if row["seed"] == seed)


def test_machine_busy_只認訓練與評估程序(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch, stub_host=False)
    # 隔離的 PID namespace 看不到全機時一律視為忙碌，避免誤判空閒後開跑。
    sched.process_scope_ok = lambda: False
    sched.process_args = lambda: []
    assert sched.machine_busy() is True

    sched.process_scope_ok = lambda: True
    sched.process_args = lambda: [["/usr/bin/python", "/opt/app/main.py", "train", "--seed", "101"]]
    assert sched.machine_busy() is True
    sched.process_args = lambda: [["/usr/bin/python", "/opt/app/eval.py", "--checkpoint", "ckpt"]]
    assert sched.machine_busy() is True
    sched.process_args = lambda: [["/usr/bin/python", "main.py", "play"]]
    assert sched.machine_busy() is False
    sched.process_args = lambda: [["bash", "-lc", "echo main.py train"]]
    assert sched.machine_busy() is False
    sched.process_args = lambda: []
    assert sched.machine_busy() is False


@pytest.mark.parametrize("quad", ["missing", "empty", "false", "null"])
def test_r_state未完成不開跑(tmp_path, monkeypatch, quad):
    sched = make_scheduler(tmp_path, monkeypatch, quad=quad)
    first = tick_at(sched, NOT_BEFORE)
    assert first["wait"]["quad_done"] is False
    assert first["wait"]["time_ok"] is True
    assert first["wait"]["busy"] is False
    assert first["wait"]["idle_count"] == 1
    second = tick_at(sched, NOT_BEFORE + 30)
    assert second["wait"]["idle_ok"] is True
    assert second["wait"]["quad_done"] is False
    assert_not_launched(sched, second)


def test_時間未到20點30不開跑(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    first = tick_at(sched, NOT_BEFORE - 31)
    assert first["wait"]["time_ok"] is False
    assert first["wait"]["quad_done"] is True
    second = tick_at(sched, NOT_BEFORE - 1)
    assert second["wait"]["time_ok"] is False
    assert second["wait"]["idle_ok"] is True
    assert second["wait"]["busy"] is False
    assert_not_launched(sched, second)


def test_機器忙碌不開跑且空閒計數歸零(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    sched._busy = False
    warmed = tick_at(sched, NOT_BEFORE)
    assert warmed["wait"]["idle_count"] == 1
    sched._busy = True
    blocked = tick_at(sched, NOT_BEFORE + 30)
    assert blocked["wait"]["busy"] is True
    assert blocked["wait"]["idle_count"] == 0
    assert blocked["wait"]["idle_ok"] is False
    assert blocked["wait"]["time_ok"] is True
    assert blocked["wait"]["quad_done"] is True
    again = tick_at(sched, NOT_BEFORE + 90)
    assert again["wait"]["idle_count"] == 0
    assert_not_launched(sched, again)


def test_兩次空閒間隔不足30秒不開跑(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    first = tick_at(sched, NOT_BEFORE)
    assert first["wait"]["idle_count"] == 1
    second = tick_at(sched, NOT_BEFORE + 29)
    assert second["wait"]["idle_count"] == 1
    assert second["wait"]["idle_ok"] is False
    assert second["wait"]["time_ok"] is True
    assert second["wait"]["quad_done"] is True
    assert second["wait"]["busy"] is False
    assert_not_launched(sched, second)
    # 間隔超過 90 秒會把連續空閒重算，不能把兩次不相鄰的空閒當成已滿足。
    reset = tick_at(sched, NOT_BEFORE + 29 + 91)
    assert reset["wait"]["idle_count"] == 1
    assert reset["wait"]["idle_ok"] is False
    assert_not_launched(sched, reset)


def test_條件滿足後單批三個S命令逐一對照(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    assert sched.cfg["common"] == COMMON
    assert sched.cfg["arm_flags"] == {"S": ["--value-skip", "l1"], "C": []}
    assert list(sched.batches) == ["1"]
    assert list(sched.jobs) == ["S_s101", "S_s202", "S_s303"]
    early = tick_at(sched, NOT_BEFORE)
    assert_not_launched(sched, early)
    state = tick_at(sched, NOT_BEFORE + 30)
    assert state["wait"]["idle_ok"] and state["wait"]["control_ready"] and state["wait"]["flags_ok"]
    names = ["S_s101", "S_s202", "S_s303"]
    assert list(state["pids"]) == names
    assert list(state["batches"]["1"]["pids"]) == names
    assert len(GIT.popen_calls) == 3
    for call, name, seed in zip(GIT.popen_calls, names, (101, 202, 303)):
        job = sched.jobs[name]
        assert job["batch"] == 1 and job["seed"] == seed and job["arm"] == "S"
        assert_launch(call, sched, job, "0-7")
    assert list(state["batches"]) == ["1"]
    # 就算有 C 的旗標定義，也不產生任何 C job；旗標檢查以 S 實際 train argv 為準。
    c_job = dict(name="C_test", arm="C", seed=101, batch=1)
    assert "--value-skip" not in split_script(sched.script(c_job))[1]
    assert len(flag_calls()) == 2  # ready 與 commit 核對後強制重算。


def test_判定門檻剛好達到才採用(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    for seed in (101, 202, 303):
        plant_seed(sched, seed, s_mean=12000, c_mean=10000, s_r=0.6, c_r=0.5, s_up=850, c_up=1000)
    result = sched.verdicts({})
    assert result["pending"] is False
    assert result["n_pass"] == 3
    assert result["adopt"] is True
    assert [row["seed"] for row in result["rows"]] == [101, 202, 303]
    for row in result["rows"]:
        assert row["pending"] is False
        assert row["ratio"] == pytest.approx(1.2)
        assert row["dpp"] == pytest.approx(10)
        assert row["diag"]["updates"] == pytest.approx(0.85)
        assert row["diag"]["env_steps"] == pytest.approx(1.0)
        assert row["diag"]["ups"] == pytest.approx(0.85)
        assert row["score_ok"] is True
        assert row["ok"] is True
        assert "cost_ok" not in row


def test_修訂2_成本比值只診斷不影響判定(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    assert "cost_guard" not in sched.cfg["rules"]
    # updates 只有 C 的 30%、env_steps 一半：分數門過就採用。
    for seed in (101, 202, 303):
        plant(sched, f"S_s{seed}", mean=13000, r1024=0.7, updates=300, extra_train={"env_steps": 2500})
        plant(sched, sched.controls[str(seed)]["name"], mean=10000, r1024=0.5, updates=1000)
    result = sched.verdicts({})
    assert result["adopt"] is True and result["n_pass"] == 3
    for row in result["rows"]:
        assert row["diag"]["updates"] == pytest.approx(0.3)
        assert row["diag"]["env_steps"] == pytest.approx(0.5)
        assert row["diag"]["ups"] == pytest.approx(0.3)
    text = sched.finish_text({"batches": {}})
    assert "成本門" not in text and "updates ×0.30" in text and "env_steps ×0.50" in text and "updates/s ×0.30" in text
    assert "採用 S（3/3）" in text
    assert "配對差" in text and "mean_diff SD" in text
    # 比值很高也救不了分數門不過。
    for seed in (101, 202, 303):
        plant(sched, f"S_s{seed}", mean=11999, r1024=0.7, updates=5000)
    result = sched.verdicts({})
    assert result["adopt"] is False and result["n_pass"] == 0
    assert all(row["diag"]["updates"] == pytest.approx(5.0) for row in result["rows"])


def test_修訂2_狀態表輸出三個診斷比值(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    for seed in (101, 202, 303):
        plant(sched, f"S_s{seed}", mean=13000, r1024=0.7, updates=600, extra_train={"env_steps": 4000})
        plant(sched, sched.controls[str(seed)]["name"], mean=10000, r1024=0.5, updates=1000)
    sched.write_fragment({"started": "x", "batches": {}}, [])
    md = sched.status_path.read_text()
    assert "成本保護門" not in md and "0.85" not in md
    assert "修訂 2" in md
    lines = [l for l in md.splitlines() if l.startswith("| seed") or l.startswith("| 101") or l.startswith("|---|---|---|---|---|---|---|---|---|")]
    header = next(l for l in md.splitlines() if l.startswith("| seed"))
    row = next(l for l in md.splitlines() if l.startswith("| 101"))
    assert header.count("|") == row.count("|")
    assert "診斷 S/C updates |" in header and "診斷 S/C env_steps" in header and "診斷 S/C updates/s" in header
    assert "×0.600 | ×0.800 | ×0.600" in row


def test_判定低於ratio或pp不採用(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    plant_seed(sched, 101, s_mean=11999, c_mean=10000, s_r=0.6, c_r=0.5, s_up=850, c_up=1000)
    plant_seed(sched, 202, s_mean=12000, c_mean=10000, s_r=0.59, c_r=0.5, s_up=850, c_up=1000)
    plant_seed(sched, 303, s_mean=12000, c_mean=10000, s_r=0.6, c_r=0.5, s_up=500, c_up=1000)
    result = sched.verdicts({})
    ratio_row = row_by_seed(result, 101)
    pp_row = row_by_seed(result, 202)
    low_up_row = row_by_seed(result, 303)
    assert ratio_row["ratio"] < 1.2
    assert ratio_row["score_ok"] is False and ratio_row["ok"] is False
    assert pp_row["dpp"] < 10
    assert pp_row["ratio"] == pytest.approx(1.2)
    assert pp_row["score_ok"] is False and pp_row["ok"] is False
    # 修訂 2：updates 只有 0.5 倍仍然通過。
    assert low_up_row["diag"]["updates"] == pytest.approx(0.5)
    assert low_up_row["score_ok"] is True and low_up_row["ok"] is True
    assert result["pending"] is False
    assert result["n_pass"] == 1
    assert result["adopt"] is False


def test_判定缺資料或無法計算不算通過(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    plant_seed(sched, 101, s_mean=20000, c_mean=10000, s_r=0.9, c_r=0.5, s_up=1000, c_up=1000, write_end=False)
    plant(sched, "S_s202", mean=20000, r1024=0.9, updates=1000, write_train=False)
    plant(sched, "E1-AS-LONG_s202", mean=10000, r1024=0.5, updates=1000)
    plant(sched, "S_s303", mean=20000, r1024=0.9, updates=1000, write_eval=False)
    plant(sched, "C_s303", mean=10000, r1024=0.5, updates=1000)
    missing = sched.verdicts({})
    assert all(row["pending"] is True for row in missing["rows"])
    assert missing["pending"] is True
    assert missing["n_pass"] == 0
    assert missing["adopt"] is False
    assert all("ok" not in row for row in missing["rows"])

    for seed in (101, 202, 303):
        plant_seed(sched, seed, s_mean=12000, c_mean=10000, s_r=0.6, c_r=0.5, s_up=850, c_up=1000)
    # S 的 updates 缺欄位：診斷比值為 None，但不影響分數判定。
    plant(sched, "S_s101", mean=12000, r1024=0.6, updates=850, drop=("updates",))
    # C 的 mean 為 0，比值無法計算 → 不通過。
    plant(sched, "S_s202", mean=12000, r1024=0.6, updates=850)
    plant(sched, "E1-AS-LONG_s202", mean=0, r1024=0.5, updates=1000)
    # C 的 updates 為 0：診斷比值為 None，分數門照常判定。
    plant(sched, "S_s303", mean=12000, r1024=0.6, updates=850)
    plant(sched, "C_s303", mean=10000, r1024=0.5, updates=0)
    partial = sched.verdicts({})
    assert partial["pending"] is False
    no_updates = row_by_seed(partial, 101)
    zero_mean = row_by_seed(partial, 202)
    zero_up = row_by_seed(partial, 303)
    assert no_updates["pending"] is False
    assert no_updates["diag"]["updates"] is None and no_updates["diag"]["ups"] is None
    assert no_updates["ok"] is True
    assert zero_mean["ratio"] is None
    assert zero_mean["score_ok"] is False and zero_mean["ok"] is False
    assert zero_up["diag"]["updates"] is None and zero_up["diag"]["ups"] is None
    assert zero_up["diag"]["env_steps"] == pytest.approx(1.0)
    assert zero_up["ok"] is True
    assert partial["n_pass"] == 2
    assert partial["adopt"] is False
    sched.write_fragment({"started": "x", "batches": {}}, [])  # None 比值也能輸出

    plant(sched, "S_s101", mean=12000, r1024=0.6, updates=850, drop=("r2048",))
    broken = sched.verdicts({})
    assert row_by_seed(broken, 101)["pending"] is True
    assert broken["pending"] is True
    assert broken["adopt"] is False


def test_里程碑相同文字只寫一次(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    sched.milestone("round2 S 完成：同一段文字")
    sched.milestone("round2 S 完成：同一段文字")
    sched.milestone("round2 S 完成：另一段文字")
    saved = json.loads(sched.miles.read_text())
    assert [item["text"] for item in saved] == [
        "round2 S 完成：同一段文字",
        "round2 S 完成：另一段文字",
    ]
    assert all(item["reported"] is False for item in saved)
    assert all(len(item["time"]) == 16 for item in saved)


def test_同job同異常種類只告警一次(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    state = {}
    sched.alert_once(state, "S_s101", "卡住：train.csv 12 分鐘沒更新")
    sched.alert_once(state, "S_s101", "卡住：train.csv 40 分鐘沒更新")
    sched.alert_once(state, "S_s101", "NaN／例外：episodes ['3']；train 非有限值 1")
    sched.alert_once(state, "S_s101", "NaN／例外：episodes ['3', '9']；train 非有限值 4")
    sched.alert_once(state, "S_s202", "卡住：train.csv 15 分鐘沒更新")
    saved = json.loads(sched.miles.read_text())
    assert [item["text"] for item in saved] == [
        "round2 S S_s101 異常 — 卡住：train.csv 12 分鐘沒更新",
        "round2 S S_s101 異常 — NaN／例外：episodes ['3']；train 非有限值 1",
        "round2 S S_s202 異常 — 卡住：train.csv 15 分鐘沒更新",
    ]
    assert state["alerted"] == ["S_s101|卡住", "S_s101|NaN／例外", "S_s202|卡住"]
    # state 遺失後仍看里程碑前綴，數字再變也不再寫一筆。
    fresh = {}
    sched.alert_once(fresh, "S_s101", "卡住：train.csv 99 分鐘沒更新")
    assert [item["text"] for item in json.loads(sched.miles.read_text())] == [item["text"] for item in saved]
    assert fresh["alerted"] == ["S_s101|卡住"]


def _arm_results(sched, name):
    dest = sched.runs / name
    dest.mkdir(parents=True)
    (dest / "meta.json").write_text(json.dumps({"name": name}))
    (dest / "train.json").write_text(json.dumps({"updates": 1}))
    (dest / "eval.json").write_text(json.dumps({"mean": 1}))
    (dest / "train.csv").write_text("episode,score_v2\n1,1\n")
    (dest / "ckpt").write_bytes(b"ckpt")
    (dest / "evals").mkdir()
    (dest / "evals" / "t0090.json").write_text(json.dumps({"mean": 2, "reach_rates": {"1024": 0.1}}))
    (dest / "evals" / "note.txt").write_text("不是 json，不應複製\n")


def test_publish只提交round2_skip且index_lock會延後(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch, publish=True)
    _arm_results(sched, "S_s101")
    sched.status_path.write_text("狀態片段\n")
    (sched.repo / "SHOULD_NOT_COMMIT.txt").write_text("別人已暫存的內容\n")
    state = {"started": "2026-10-18 21:00:00", "batches": {}, "pids": {}}
    (GIT.gitdir / "index.lock").write_text("locked")
    assert sched.publish(dict(state)) is False
    assert git_cmds() == [["git", "-C", str(sched.repo), "rev-parse", "--absolute-git-dir"]]
    assert not (sched.repo / RESULTS).exists()
    assert "published" not in state

    GIT.calls.clear()
    (GIT.gitdir / "index.lock").unlink()
    assert sched.publish(state) is True
    commands = git_cmds()
    assert commands[0] == ["git", "-C", str(sched.repo), "rev-parse", "--absolute-git-dir"]
    assert commands[1] == ["git", "-C", str(sched.repo), "branch", "--show-current"]
    assert commands[2] == ["git", "-C", str(sched.repo), "add", RESULTS]
    assert commands[3] == ["git", "-C", str(sched.repo), "diff", "--cached", "--quiet", "--", RESULTS]
    commit = commands[4]
    assert commit[:6] == ["git", "-C", str(sched.repo), "commit", "-m", "第二輪 S：更新正式訓練結果"]
    assert commit[6:] == ["--", RESULTS]
    assert "-a" not in commit and "--all" not in commit
    assert commands[5] == ["git", "-C", str(sched.repo), "push", "origin", BRANCH]
    assert len(commands) == 6
    flat = [" ".join(cmd) for cmd in commands]
    assert all("SHOULD_NOT_COMMIT" not in line for line in flat)
    assert all("add -A" not in line and "add ." not in line for line in flat)
    assert all("merge" not in cmd for cmd in commands)
    assert all("--force" not in cmd and "-f" not in cmd for cmd in commands)
    copied = sched.repo / RESULTS / "runs" / "S_s101"
    assert (sched.repo / RESULTS / "S_RESULTS.md").read_text() == "狀態片段\n"
    assert json.loads((copied / "meta.json").read_text())["name"] == "S_s101"
    assert (copied / "train.json").exists()
    assert (copied / "eval.json").exists()
    assert (copied / "evals" / "t0090.json").exists()
    assert not (copied / "train.csv").exists()
    assert not (copied / "ckpt").exists()
    assert not (copied / "evals" / "note.txt").exists()
    assert state["publish_pending"] is False
    assert state.get("published")


def test_push失敗下次仍重試(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch, publish=True)
    sched.status_path.write_text("狀態\n")
    state = {"started": "2026-10-18 21:00:00", "batches": {}, "pids": {}}
    GIT.diff_seq = [1, 0]
    GIT.push_seq = [1, 0]
    assert sched.publish(state) is False
    assert state["publish_pending"] is True
    assert "published" not in state
    first = git_cmds()
    assert first[-1][3] == "push"
    assert any(cmd[3] == "commit" for cmd in first)
    GIT.calls.clear()
    assert sched.publish(state) is True
    second = git_cmds()
    assert [cmd[3] for cmd in second] == ["rev-parse", "branch", "add", "diff", "push"]
    assert second[-1] == ["git", "-C", str(sched.repo), "push", "origin", BRANCH]
    assert state["publish_pending"] is False
    assert state.get("published")


def test_dry_run不launch不publish不寫里程碑(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch, dry_run=True, publish=True)
    opened = NOT_BEFORE + 500
    seeded = {
        "batches": {},
        "pids": {},
        "idle": {"count": 1, "last_epoch": opened - 30},
    }
    raw = json.dumps(seeded)
    sched.state_path.write_text(raw)
    (sched.runs / "S_s101").mkdir()
    (sched.runs / "S_s101" / "ckpt").write_bytes(b"ckpt")
    state = tick_at(sched, opened)
    assert state["wait"]["quad_done"] is True
    assert state["wait"]["time_ok"] is True
    assert state["wait"]["idle_ok"] is True
    assert state["wait"]["busy"] is False
    assert "1" not in state.get("batches", {})
    assert GIT.popen_calls == []
    assert git_cmds() == []
    assert sched.state_path.read_text() == raw
    assert (tmp_path / "s_state.dry_run.json").exists()
    assert not sched.miles.exists()
    assert not (sched.repo / RESULTS).exists()
    assert "DRY-RUN" in sched.status_path.read_text()
    sched.milestone("round2 S 完成：不該出現")
    sched.alert_once({}, "S_s101", "卡住：train.csv 12 分鐘沒更新")
    assert sched.publish({"started": "yes", "batches": {}, "pids": {}}) is False
    assert not sched.miles.exists()
    assert git_cmds() == []
    with pytest.raises(RuntimeError, match="dry-run"):
        sched.spawn(["true"], tmp_path / "launch.out")
    js = {"active_s": 150 * 60}
    sched.snap_tick(js, sched.runs / "S_s101", True, {"cpus": "0-7", "id": 1})
    assert not (sched.runs / "S_s101" / "snapshots").exists()
    assert GIT.popen_calls == []
    assert "taken" not in js


def test_快照複製並沿用該批次cpus與nice(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    payload = b"checkpoint-bytes"
    plan = [("S_s101", "0-7", 150), ("S_s303", "0-7", 90)]
    for name, cpus, minutes in plan:
        dest = sched.runs / name
        dest.mkdir(parents=True)
        (dest / "ckpt").write_bytes(payload)
        js = {"active_s": minutes * 60}
        sched.snap_tick(js, dest, True, {"cpus": cpus, "id": 1 if cpus == "0-7" else 2})
        marks = [90, 120, 150] if minutes == 150 else [90]
        assert js["taken"] == marks
        for mark in marks:
            copied = dest / "snapshots" / f"t{mark:04d}.pt"
            assert copied.read_bytes() == payload
    torch_cmds = [args for args in GIT.calls if any("torch.load" in part for part in args)]
    popen = GIT.popen_calls
    assert len(torch_cmds) == 4
    assert len(popen) == 4
    expected = [
        ("S_s101", "0-7", 90),
        ("S_s101", "0-7", 120),
        ("S_s101", "0-7", 150),
        ("S_s303", "0-7", 90),
    ]
    for args, call, (name, cpus, mark) in zip(torch_cmds, popen, expected):
        dest = sched.runs / name
        shot = dest / "snapshots" / f"t{mark:04d}.pt"
        out = dest / "evals" / f"t{mark:04d}.json"
        assert args[:5] == ["nice", "-n", "19", "taskset", "-c"]
        assert args[5] == cpus
        assert args[6] == sched.cfg["python"]
        assert "torch.load" in args[-1]
        assert str(shot) in args[-1]
        assert call["args"][:7] == ["setsid", "nohup", "taskset", "-c", cpus, "nice", "-n"]
        assert call["args"][7] == "10"
        assert call["args"][8:] == expected_eval_snapshot(sched, shot, out)
        assert call["cwd"] == sched.code
        assert call["env"]["OMP_NUM_THREADS"] == "1"
        assert call["stdout_name"] == str(dest / "evals" / f"t{mark:04d}.out")


def expected_eval_snapshot(sched, checkpoint, output):
    return [
        "timeout", "--signal=TERM", "--kill-after=30s", "30m",
        sched.cfg["python"], "eval.py",
        "--checkpoint", str(checkpoint),
        "--policy", "greedy",
        "--games", "100",
        "--seed-start", "1000",
        "--board", "fast",
        "--max-steps", "5000",
        "--json", str(output),
    ]


def flag_calls():
    return [a for a in GIT.calls if any(Path(part).name == "flagcheck.py" for part in a)]


@pytest.mark.parametrize("missing", ["eval.json", "eval_end", "invalid", "nonfinite", "partial"])
def test_C303對照未就緒不開跑(tmp_path, monkeypatch, missing):
    sched = make_scheduler(tmp_path, monkeypatch)
    d = Path(sched.controls["303"]["dir"])
    if missing in ("eval.json", "eval_end"):
        (d / missing).unlink()
    elif missing == "invalid":
        (d / "eval.json").write_text("{broken")
    else:
        (d / "eval.json").write_text('{"mean": NaN}' if missing == "nonfinite" else '{"mean": 1000}')
    tick_at(sched, NOT_BEFORE)
    state = tick_at(sched, NOT_BEFORE + 30)
    assert state["wait"]["control_ready"] is False
    assert state["wait"]["flags_ok"] is True
    assert_not_launched(sched, state)


def test_旗標不一致不開跑且只告警一次(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    GIT.flag_diff = True
    tick_at(sched, NOT_BEFORE)
    state = tick_at(sched, NOT_BEFORE + 30)
    assert state["wait"]["flags_ok"] is False
    assert_not_launched(sched, state)
    messages = json.loads(sched.miles.read_text())
    assert len(messages) == 3
    assert all("旗標不一致：" in m["text"] and "gamma" in m["text"] for m in messages)
    assert all(r["status"] == "不一致" for r in state["wait"]["flags"].values())
    assert "不一致：diff.gamma" in sched.status_path.read_text()


def test_對照meta缺少不開跑(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    (Path(sched.controls["303"]["dir"]) / "meta.json").unlink()
    tick_at(sched, NOT_BEFORE)
    state = tick_at(sched, NOT_BEFORE + 30)
    assert state["wait"]["control_ready"] is True
    assert state["wait"]["flags_ok"] is False
    assert state["wait"]["flags"]["303"]["status"] == "待對照 meta"
    assert_not_launched(sched, state)
    assert "s303：待對照 meta" in sched.status_path.read_text()
    assert not sched.miles.exists()


def test_旗標快取600秒且開跑前強制重算攔截(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    first = tick_at(sched, NOT_BEFORE)
    assert first["wait"]["flags_ok"] is True and len(flag_calls()) == 1
    # 快取仍為一致，但 commit 核對後必須真的重查。
    GIT.flag_diff = True
    second = tick_at(sched, NOT_BEFORE + 30)
    assert len(flag_calls()) == 2
    assert second["wait"]["flags_ok"] is False
    assert "啟動失敗" in second["launch_error"]
    assert_not_launched(sched, second)
    assert any("啟動失敗" in m["text"] for m in json.loads(sched.miles.read_text()))
    sched.now_epoch = lambda: NOT_BEFORE + 629
    assert sched.check_flags(second)["101"]["ok"] is False
    assert len(flag_calls()) == 2
    GIT.flag_diff = False
    sched.now_epoch = lambda: NOT_BEFORE + 630
    assert all(r["ok"] for r in sched.check_flags(second).values())
    assert len(flag_calls()) == 3
    # 指令依序為初次旗標、commit、強制旗標；管理程序都 nice 19。
    first_three = GIT.calls[:3]
    assert Path(first_three[0][4]).name == "flagcheck.py"
    assert first_three[1][3:5] == ["git", "-C"]
    assert Path(first_three[2][4]).name == "flagcheck.py"
    assert all(a[:3] == ["nice", "-n", "19"] for a in GIT.calls)


@pytest.mark.parametrize("gamma,ok", [(None, True), ("0.98", False)])
def test_flagcheck真實parser核對(tmp_path, monkeypatch, gamma, ok):
    sched = make_scheduler(tmp_path, monkeypatch, dry_run=True)
    sched.code = Path("/workspace/rl2048-review/wt-skip-run")
    for seed, c in sched.controls.items():
        argv = expected_train(sched, sched.jobs[f"S_s{seed}"])
        index = argv.index("--value-skip")
        del argv[index:index + 2]
        # 只排除的四個欄位允許不同，其餘必須真正由 parser 比對。
        argv[argv.index("--minutes") + 1] = "999"
        argv[argv.index("--save-every") + 1] = "17"
        argv[argv.index("--save-as") + 1] = "/tmp/另一份 checkpoint"
        argv[argv.index("--log") + 1] = "/tmp/另一份 log"
        if gamma:
            argv += ["--gamma", gamma]
        script = shlex.join(argv) + " > /tmp/fake-train.out 2>&1 && date"
        (Path(c["dir"]) / "meta.json").write_text(json.dumps({"script": script}))
    monkeypatch.setattr(s_run.subprocess, "run", REAL_RUN)
    monkeypatch.setattr(s_run.subprocess, "Popen", REAL_POPEN)
    results = sched.check_flags({})
    assert all(r["ok"] is ok for r in results.values())
    for r in results.values():
        assert r["status"] == ("一致" if ok else "不一致")
        assert r["config_fields"] > 10
        if not ok:
            assert r["diff"]["gamma"] == [0.99, 0.98]
            assert r["config_diff"]["gamma"] == [0.99, 0.98]
    assert GIT.popen_calls == [] and not sched.miles.exists()


@pytest.mark.parametrize("pair,expected", [
    ({"mean_load1": 7.5, "mean_busy_cores": 6.25, "load_sum": 90, "load_n": 2}, (7.5, 6.25)),
    ({"load_sum": 21, "load_n": 3, "busy_sum": 15, "busy_n": 3}, (7, 5)),
    ({"mean_load1": None, "mean_busy_cores": None, "load_sum": 12, "load_n": 3, "busy_sum": 6, "busy_n": 3}, (4, 2)),
    ({"mean_load1": 0, "mean_busy_cores": 0, "load_sum": 12, "load_n": 3}, (0, 0)),
    ({}, (None, None)),
    ({"load_n": 2, "busy_n": 3}, (None, None)),
    ({"load_sum": 1, "load_n": 0, "busy_sum": 2, "busy_n": 0}, (None, None)),
])
def test_control_metrics與pair_state負載(tmp_path, monkeypatch, pair, expected):
    sched = make_scheduler(tmp_path, monkeypatch)
    root = tmp_path / "controls"
    s_run.save_json(root / "pair_state.json", {"pairs": {"303": pair, "101": {"mean_load1": 99}}})
    c = sched.control_metrics(303)
    assert (c["load1"], c["busy"]) == expected
    assert c["updates"] == 1000 and c["env_steps"] == 5000 and c["ups"] == pytest.approx(1000 / 10800)
    assert c["episodes"] == 80 and c["active_s"] == 10800
    assert c["mean"] == 10000 and c["median"] == 5000 and c["r1024"] == 0.5 and c["r2048"] == 0.01
    assert c["rr"] == 0.45 and c["nan"] == 0
    assert sched.controls["303"]["load"]["state"] == str(root / "pair_state.json")
    for seed in (101, 202):
        d = Path(sched.controls[str(seed)]["dir"])
        (d / "eval_end").unlink()  # LONG 不要求 eval_end；只讀指定 t180。
        (d / "eval.json").write_text('{"mean": 99999}')
        m = sched.control_metrics(seed)
        assert m["mean"] == 10000 and m["load1"] is None and m["busy"] is None
        assert "未記錄" in m["load_note"] and "CPU 4–7" in m["load_note"]
    plant(sched, "S_s101", mean=13000, r1024=0.7, updates=600)
    m = sched.metrics("S_s101", {"batches": {"1": {"mean_load1": 8.1, "mean_busy_cores": 6.3}}})
    assert m["load1"] == 8.1 and m["busy"] == 6.3


@pytest.mark.parametrize("file", ["train.json", "evals/t0180.json"])
def test_LONG缺失或無效資料不算完成(tmp_path, monkeypatch, file):
    sched = make_scheduler(tmp_path, monkeypatch)
    path = Path(sched.controls["101"]["dir"]) / file
    path.unlink()
    assert sched.control_metrics(101) is None
    path.write_text('[]')
    assert sched.control_metrics(101) is None
    path.write_text('{}')
    assert sched.control_metrics(101) is None


def test_配對差樣本SD與範圍手算(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    # mean 差 [1000,2000,3000] → SD=1000；ratio [1.1,1.2,1.3] → SD=.1；dpp [10,20,30] → SD=10。
    for i, seed in enumerate((101, 202, 303), 1):
        plant_seed(sched, seed, s_mean=10000 + i * 1000, c_mean=10000,
                   s_r=0.5 + i * 0.1, c_r=0.5, s_up=600, c_up=1000)
    result = sched.verdicts({})
    assert [r["mean_diff"] for r in result["paired"]] == [1000, 2000, 3000]
    for key, numbers in {"mean_diff": (1000, 1000, 3000), "ratio": (0.1, 1.1, 1.3), "dpp": (10, 10, 30)}.items():
        assert [result["spread"][key][k] for k in ("sd", "min", "max")] == pytest.approx(numbers)
    assert "mean_diff SD 1000.000" in sched.finish_text({})
    (sched.runs / "S_s303" / "eval_end").unlink()
    partial = sched.verdicts({})
    assert len(partial["paired"]) == 2 and all(v is None for v in partial["spread"].values())


def assert_markdown_tables(md):
    tables, current = [], []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    assert len(tables) >= 7
    for table in tables:
        assert len({line.count("|") for line in table}) == 1, table


@pytest.mark.parametrize("complete", [False, True])
def test_報表並列配對差表及欄數一致(tmp_path, monkeypatch, complete):
    sched = make_scheduler(tmp_path, monkeypatch)
    if complete:
        for seed in (101, 202, 303):
            plant_seed(sched, seed, s_mean=13000, c_mean=10000, s_r=0.7, c_r=0.5, s_up=600, c_up=1000)
    sched.write_fragment({"batches": {}}, [])
    md = sched.status_path.read_text()
    assert_markdown_tables(md)
    assert "S 與對照並列" in md and "每 seed 配對差（S−C）" in md
    assert "3 個 seed＝3 條確定性軌跡（n=3），seed 間變異是唯一的變異估計" in md
    assert "對照不同時段、負載不同，只並列不校正" in md
    assert "E1-AS-LONG_s101 t180" in md and "C_s303 t180" in md
    assert "未記錄（extra 排程器未取樣" in md
    assert "兩批" not in md and "skipctl" not in md and "同時段 C" not in md
    snapshot = md.split("探索性快照（只列 S")[1]
    assert "LONG" not in snapshot and "C_s303" not in snapshot


def test_publish複製controls並納入變動偵測(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch, publish=True)
    sched.status_path.write_text("狀態\n")
    state = {"started": "x", "batches": {}}
    before = {str(f): f.read_bytes() for f, _ in sched.control_files()}
    assert sched.publish(state)
    signature = sched.publication_sig(state)
    for control in sched.controls.values():
        for file in (control["eval"], "train.json", "meta.json"):
            source = Path(control["dir"]) / file
            target = sched.repo / RESULTS / "controls" / control["name"] / file
            assert target.read_bytes() == before[str(source)] == source.read_bytes()
    c_meta = Path(sched.controls["101"]["dir"]) / "meta.json"
    c_meta.write_text(c_meta.read_text() + "\n")
    assert sched.publication_sig(state) != signature
    assert all(cmd[3] not in ("add", "commit") or RESULTS in cmd for cmd in git_cmds())


def test_完成等待三個對照及快照收尾且只檢查S異常(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    for seed in (101, 202, 303):
        plant_seed(sched, seed, s_mean=13000, c_mean=10000, s_r=0.7, c_r=0.5, s_up=600, c_up=1000)
    state = {"started": "x", "batches": {"1": {"started": "x"}}, "pids": {}}
    s_run.save_json(sched.state_path, state)
    calls = []
    sched.anomalies = lambda job: calls.append(job["name"]) or []
    c_train = Path(sched.controls["303"]["dir"]) / "train.json"
    payload = c_train.read_text()
    c_train.unlink()
    first = tick_at(sched, NOT_BEFORE)
    assert first["batches"]["1"]["done"] and not first.get("done")
    c_train.write_text(payload)
    sched.snapshots_finished = lambda st: False
    assert not tick_at(sched, NOT_BEFORE + 30).get("done")
    sched.snapshots_finished = lambda st: True
    done = tick_at(sched, NOT_BEFORE + 60)
    assert done["done"] and "採用 S" in done["completion_text"]
    assert set(calls) == {"S_s101", "S_s202", "S_s303"}
