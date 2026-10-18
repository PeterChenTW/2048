"""S 排程器獨立測試。

路徑全部落在 tmp_path。subprocess.Popen 與 subprocess.run 都被替換，
不會啟動訓練、eval，也不會對真實 git 發佈。
"""
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_run  # noqa: E402

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

    def reset(self, gitdir):
        self.calls = []
        self.popen_calls = []
        self.head = "28a8f8bc54639ea0982742015909b9d36854fd09"
        self.branch = BRANCH
        self.gitdir = gitdir
        self.diff_code = 1
        self.push_code = 0
        self.diff_seq = None
        self.push_seq = None


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
    dest = sched.runs / name
    dest.mkdir(parents=True, exist_ok=True)
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
        (dest / "eval.json").write_text(json.dumps(payload))
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
    plant(sched, f"C_s{seed}-skipctl", mean=c_mean, r1024=c_r, updates=c_up, **kwargs)


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


def test_條件滿足後批次1四個命令逐一對照(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    assert sched.cfg["common"] == COMMON
    assert sched.cfg["arm_flags"]["S"] == ["--value-skip", "l1"]
    assert sched.cfg["arm_flags"]["C"] == []
    assert sched.cfg["nice"] == 10
    assert sched.cfg["threads"] == 2
    assert sched.cfg["minutes"] == 180
    early = tick_at(sched, NOT_BEFORE)
    assert early["wait"]["time_ok"] is True
    assert early["wait"]["idle_count"] == 1
    assert_not_launched(sched, early)
    state = tick_at(sched, NOT_BEFORE + 30)
    assert state["wait"]["idle_ok"] is True
    names = ["S_s101", "S_s202", "C_s101-skipctl", "C_s202-skipctl"]
    assert list(state["pids"]) == names
    assert list(state["batches"]["1"]["pids"]) == names
    assert len(GIT.popen_calls) == 4
    for call, name in zip(GIT.popen_calls, names):
        job = sched.jobs[name]
        assert job["batch"] == 1
        assert_launch(call, sched, job, "0-7")
    scripts = [call["args"][-1] for call in GIT.popen_calls]
    assert scripts[0].count("--value-skip l1") == 1
    assert scripts[1].count("--value-skip l1") == 1
    assert "value-skip" not in scripts[2]
    assert "value-skip" not in scripts[3]
    assert "--seed 101" in scripts[0] and "--seed 101" in scripts[2]
    assert "--seed 202" in scripts[1] and "--seed 202" in scripts[3]
    # 批次 1 開跑後，同一個 tick 不再啟動批次 2。
    assert "2" not in state["batches"]
    assert not (sched.runs / "S_s303").exists()


def _prepare_batch1(sched, *, done, ends):
    names = ["S_s101", "S_s202", "C_s101-skipctl", "C_s202-skipctl"]
    batch = {"started": "2026-10-19 00:00:00", "pids": {}, "load_sum": 0, "load_n": 0, "busy_sum": 0, "busy_n": 0}
    if done:
        batch["done"] = "2026-10-19 00:30:00"
    state = {"batches": {"1": batch}, "pids": {}, "started": "2026-10-19 00:00:00"}
    sched.state_path.write_text(json.dumps(state))
    for index, name in enumerate(names):
        dest = sched.runs / name
        dest.mkdir(parents=True)
        if index < ends:
            (dest / "eval_end").write_text("1\n")


def test_批次2在四個eval完成且機器空閒時用cpu0到3開跑(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    _prepare_batch1(sched, done=True, ends=4)
    # 批次 2 不以 20:30 或連續兩次空閒為門檻。
    state = tick_at(sched, NOT_BEFORE - 10_000)
    assert state["wait"]["time_ok"] is False
    names = ["S_s303", "C_s303-skipctl"]
    assert list(state["batches"]["2"]["pids"]) == names
    assert len(GIT.popen_calls) == 2
    for call, name in zip(GIT.popen_calls, names):
        job = sched.jobs[name]
        assert job["seed"] == 303
        assert job["batch"] == 2
        assert_launch(call, sched, job, "0-3")
    assert "--value-skip l1" in GIT.popen_calls[0]["args"][-1]
    assert "value-skip" not in GIT.popen_calls[1]["args"][-1]


@pytest.mark.parametrize("done,ends,busy", [(False, 4, False), (True, 3, False), (True, 4, True)])
def test_批次2條件不足不開跑(tmp_path, monkeypatch, done, ends, busy):
    sched = make_scheduler(tmp_path, monkeypatch)
    _prepare_batch1(sched, done=done, ends=ends)
    sched._busy = busy
    state = tick_at(sched, NOT_BEFORE + 100)
    assert "2" not in state["batches"]
    assert GIT.popen_calls == []
    assert not (sched.runs / "S_s303").exists()
    assert not (sched.runs / "C_s303-skipctl").exists()


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
        assert row["cost_ratio"] == pytest.approx(0.85)
        assert row["score_ok"] is True
        assert row["cost_ok"] is True
        assert row["ok"] is True


def test_判定低於ratio或pp或成本門不採用(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    plant_seed(sched, 101, s_mean=11999, c_mean=10000, s_r=0.6, c_r=0.5, s_up=850, c_up=1000)
    plant_seed(sched, 202, s_mean=12000, c_mean=10000, s_r=0.59, c_r=0.5, s_up=850, c_up=1000)
    plant_seed(sched, 303, s_mean=12000, c_mean=10000, s_r=0.6, c_r=0.5, s_up=849, c_up=1000)
    result = sched.verdicts({})
    ratio_row = row_by_seed(result, 101)
    pp_row = row_by_seed(result, 202)
    cost_row = row_by_seed(result, 303)
    assert ratio_row["ratio"] < 1.2
    assert ratio_row["score_ok"] is False
    assert ratio_row["cost_ok"] is True
    assert ratio_row["ok"] is False
    assert pp_row["dpp"] < 10
    assert pp_row["ratio"] == pytest.approx(1.2)
    assert pp_row["score_ok"] is False
    assert pp_row["cost_ok"] is True
    assert pp_row["ok"] is False
    assert cost_row["score_ok"] is True
    assert cost_row["cost_ratio"] < 0.85
    assert cost_row["cost_ok"] is False
    assert cost_row["ok"] is False
    assert result["pending"] is False
    assert result["n_pass"] == 0
    assert result["adopt"] is False


def test_判定缺資料或無法計算不算通過(tmp_path, monkeypatch):
    sched = make_scheduler(tmp_path, monkeypatch)
    plant_seed(sched, 101, s_mean=20000, c_mean=10000, s_r=0.9, c_r=0.5, s_up=1000, c_up=1000, write_end=False)
    plant(sched, "S_s202", mean=20000, r1024=0.9, updates=1000, write_train=False)
    plant(sched, "C_s202-skipctl", mean=10000, r1024=0.5, updates=1000)
    plant(sched, "S_s303", mean=20000, r1024=0.9, updates=1000, write_eval=False)
    plant(sched, "C_s303-skipctl", mean=10000, r1024=0.5, updates=1000)
    missing = sched.verdicts({})
    assert all(row["pending"] is True for row in missing["rows"])
    assert missing["pending"] is True
    assert missing["n_pass"] == 0
    assert missing["adopt"] is False
    assert all("ok" not in row for row in missing["rows"])

    for seed in (101, 202, 303):
        plant_seed(sched, seed, s_mean=12000, c_mean=10000, s_r=0.6, c_r=0.5, s_up=850, c_up=1000)
    # 有 eval／train，但 S 的 updates 缺欄位：不是 pending，成本門失敗。
    plant(sched, "S_s101", mean=12000, r1024=0.6, updates=850, drop=("updates",))
    # C 的 mean 為 0，比值無法計算。
    plant(sched, "S_s202", mean=12000, r1024=0.6, updates=850)
    plant(sched, "C_s202-skipctl", mean=0, r1024=0.5, updates=1000)
    # C 的 updates 為 0，成本比無法計算。分數門仍可過，綜合判定不過。
    plant(sched, "S_s303", mean=12000, r1024=0.6, updates=850)
    plant(sched, "C_s303-skipctl", mean=10000, r1024=0.5, updates=0)
    partial = sched.verdicts({})
    assert partial["pending"] is False
    assert partial["adopt"] is False
    assert partial["n_pass"] == 0
    no_updates = row_by_seed(partial, 101)
    zero_mean = row_by_seed(partial, 202)
    zero_cost = row_by_seed(partial, 303)
    assert no_updates["pending"] is False
    assert no_updates["cost_ratio"] is None
    assert no_updates["cost_ok"] is False
    assert no_updates["score_ok"] is True
    assert zero_mean["ratio"] is None
    assert zero_mean["score_ok"] is False
    assert zero_mean["pending"] is False
    assert zero_cost["cost_ratio"] is None
    assert zero_cost["cost_ok"] is False
    assert zero_cost["score_ok"] is True

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
    sched.alert_once(state, "C_s101-skipctl", "卡住：train.csv 15 分鐘沒更新")
    saved = json.loads(sched.miles.read_text())
    assert [item["text"] for item in saved] == [
        "round2 S S_s101 異常 — 卡住：train.csv 12 分鐘沒更新",
        "round2 S S_s101 異常 — NaN／例外：episodes ['3']；train 非有限值 1",
        "round2 S C_s101-skipctl 異常 — 卡住：train.csv 15 分鐘沒更新",
    ]
    assert state["alerted"] == ["S_s101|卡住", "S_s101|NaN／例外", "C_s101-skipctl|卡住"]
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
    plan = [("S_s101", "0-7", 150), ("C_s303-skipctl", "0-3", 90)]
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
        ("C_s303-skipctl", "0-3", 90),
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
