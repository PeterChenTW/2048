#!/usr/bin/env python3
"""PM 10-18 20:07 復原方案（10-22 執行）：C_s404 從頭重跑，設定與原 quad launch 完全相同（同 script()/cpus/nice/threads/commit）。
舊目錄已改名為 C_s404_interrupted。只啟動 C_s404 一個 job；snapshots/eval 仍由 quad_run.py loop 接手。"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quad_run as Qr
P, Q = Qr.P, Qr.Q
job = next(j for j in Q['jobs'] if j['name'] == 'C_s404')
d = P.RUNS / 'C_s404'
assert (P.RUNS / 'C_s404_interrupted').is_dir(), '舊目錄需先改名'
commit = subprocess.run(['git', '-C', str(P.CODE), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert commit.startswith('497bad4'), commit
s = Qr.script(job)
old = P.load(P.RUNS / 'C_s404_interrupted' / 'meta.json')
assert old['script'] == s, '與中斷 run 的 script 不同，停止'
env = dict(__import__('os').environ, OMP_NUM_THREADS=str(P.CFG['threads']), MKL_NUM_THREADS=str(P.CFG['threads']))
d.mkdir(parents=True, exist_ok=False)
p = subprocess.Popen(['setsid', 'nohup', 'taskset', '-c', Q['cpus'], 'nice', '-n', str(Q['nice']), 'bash', '-c', s],
                     cwd=P.CODE, env=env, stdin=subprocess.DEVNULL, stdout=open(d / 'launch.out', 'w'),
                     stderr=subprocess.STDOUT, start_new_session=True)
P.save_json(d / 'meta.json', dict(job=job, commit=commit, prereg=Q['prereg'] + '；PM 10-18 20:07 復原方案：從頭重跑（舊 run 改名 C_s404_interrupted）',
                                  cpus=Q['cpus'], nice=Q['nice'], threads=P.CFG['threads'], minutes=Q.get('minutes', P.CFG['minutes']),
                                  snapshots=Q.get('snapshots'), started=P.now() + ' Taipei', script=s, launcher_pid=p.pid, rerun_of='C_s404_interrupted'))
print('launched', p.pid, P.now())
