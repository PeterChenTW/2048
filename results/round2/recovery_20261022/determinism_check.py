#!/usr/bin/env python3
"""復原方案步驟 3：決定性比對（只讀、nice 19、單執行緒）。新 run vs *_interrupted：
(a) 前 N 局 train.csv 的 score_v2/max_tile/steps 完全一致？（loss 等浮點欄位另報最大差）
(b) t90 eval（mean/median/≥1024/≥2048）一致？不一致只標記、不停任何訓練。
結果寫 logs/DETERMINISM_20261022.md；--watch 會等到所有 t90 eval 出現（或 6 小時）。"""
import csv, json, sys, time
from pathlib import Path
R = Path('/workspace/rl2048-review')
PAIRS = {'C_s404': R / 'experiments/runs/round2'} | {f'S_s{s}': R / 'skip-run/runs' for s in (101, 202, 303, 404)}
OUT = R / 'experiments/logs/DETERMINISM_20261022.md'
N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 100
def rows(p):
    try: return list(csv.DictReader(open(p)))
    except OSError: return []
def ev(p):
    try:
        e = json.load(open(p)); return dict(mean=e['mean'], median=e['median'], r1024=e['reach_rates']['1024'], r2048=e['reach_rates']['2048'])
    except Exception: return None
def check():
    L = [f'# 決定性比對（更新 {time.strftime("%Y-%m-%d %H:%M:%S")} 台北）', '', f'前 N={N} 局；整數欄位 score_v2/max_tile/steps 需完全一致；不一致只標記。', '',
         '| run | 比較局數 | 整數欄位 | 第一個不一致局 | loss 最大相對差 | t90 eval 新 | t90 eval 舊 | t90 |', '|---|---|---|---|---|---|---|---|']
    done = True
    for name, base in PAIRS.items():
        a, b = rows(base / name / 'train.csv'), rows(base / f'{name}_interrupted' / 'train.csv')
        n = min(N, len(a), len(b)); first = None; lmax = 0.0
        for i in range(n):
            if any(a[i][k] != b[i][k] for k in ('score_v2', 'max_tile', 'steps')) and first is None: first = i + 1
            try:
                x, y = float(a[i]['loss']), float(b[i]['loss']); lmax = max(lmax, abs(x - y) / max(abs(y), 1e-12))
            except (ValueError, KeyError): pass
        int_ok = '待資料' if n == 0 else ('一致' if first is None else '**不一致**')
        e1, e0 = ev(base / name / 'evals/t0090.json'), ev(base / f'{name}_interrupted' / 'evals/t0090.json')
        f = lambda e: '待資料' if not e else f"{e['mean']:,.0f}／{e['r1024']:.0%}"
        t90 = '待資料' if not (e1 and e0) else ('一致' if e1 == e0 else '**不一致（標記，不停）**')
        if not e1 or n < N: done = False
        L.append(f'| {name} | {n} | {int_ok} | {first or "—"} | {lmax:.2e} | {f(e1)} | {f(e0)} | {t90} |')
    L += ['', '註：active budget clock 下快照依 active 秒取，若各局整數一致但 t90 快照落在不同局數，eval 可能不同；'
          '.venv 於 10-22 重建（torch 2.14.1+cpu，原版本未記錄），浮點差可能來自版本差異。']
    OUT.write_text('\n'.join(L) + '\n'); return done
if '--watch' in sys.argv:
    t0 = time.time()
    while not check() and time.time() - t0 < 6 * 3600: time.sleep(300)
else:
    check(); print(OUT.read_text())
