"""seed 2 閘門異常（R3 不存在）測試；不碰真實狀態。執行：python3 test/test_gate_anomaly.py"""
import importlib.util, json, os, sys, tempfile
from pathlib import Path
d = Path(tempfile.mkdtemp(prefix="s2ga_"))
cfg = dict(out=str(d / "out"), r3_runs=str(d / "runs"), r3_state=str(d / "nostate.json"), milestones=str(d / "m.json"), r3_missing_grace_s=100)
(d / "out").mkdir(); (d / "c.json").write_text(json.dumps(cfg))
os.environ["S2_TEST_CONFIG"] = str(d / "c.json")
spec = importlib.util.spec_from_file_location("s2", Path(__file__).resolve().parents[1] / "s2_rerun.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
T = [1000.0]; m.now = lambda: T[0]
dead = {j: dict(alive=False) for j in m.CFG["r3_jobs"]}
fails = 0
def ok(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name); fails += (not cond)
st = {"events": []}
ok("寬限內不標記", m.check_r3_missing(st, [], dead) is None and "r3_missing_since" in st)
T[0] += 101
ga = m.check_r3_missing(st, [], dead)
ok("超過寬限標記異常", ga and "不在" in ga["why"] and "run 目錄不存在" in ga["why"])
ms = json.load(open(cfg["milestones"]))
ok("寫一筆里程碑", len(ms) == 1 and "seed 2 異常" in ms[0]["text"])
T[0] += 500; m.check_r3_missing(st, [], dead)
ok("不重複回報", len(json.load(open(cfg["milestones"]))) == 1)
ok("排程器在跑時清除", m.check_r3_missing(st, [(1, ["python", "/x/r3_run.py", "loop"])], dead) is None and "gate_anomaly" not in st)
alive = dict(dead); alive["G_s303"] = dict(alive=True)
st2 = {"events": []}; T[0] += 10000
ok("R3 訓練在跑不算異常", m.check_r3_missing(st2, [], alive) is None and "r3_missing_since" not in st2)
st3 = {"events": []}; T[0] += 10000
done = {j: dict(alive=False, done=True) for j in m.CFG["r3_jobs"]}
ok("R3 全部完成不算異常", m.check_r3_missing(st3, [], done) is None and "r3_missing_since" not in st3)
part = dict(done); part["GY_s101"] = dict(alive=False, done=False)
st4 = {"events": []}; m.check_r3_missing(st4, [], part); T[0] += 101
ok("只差一個 job 未完成仍會標記", m.check_r3_missing(st4, [], part) is not None)
ok("r3_jobs 為三批 9 個", len(m.CFG["r3_jobs"]) == 9)
print("FAILS", fails); sys.exit(1 if fails else 0)
