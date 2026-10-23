"""Summarize expectimax eval CSVs. Mean diff vs d=1: paired by eval seed (same seeds 1000-1999),
95% CI = mean(diff) +/- 1.96*sd/sqrt(n). Also Welch-style unpaired CI shown for reference."""
import csv, json, math, os, statistics as st, sys
O = os.path.join(os.path.dirname(__file__), '..', 'results', 'ntuple', 'expectimax')
def load(p):
    with open(p) as f: return list(csv.DictReader(f))
def ci(xs):
    m = st.mean(xs); h = 1.96 * st.stdev(xs) / math.sqrt(len(xs)); return m, m - h, m + h
out = {}; rows = []
for s in (1, 2):
    base = {r['seed']: float(r['score']) for r in load(f'{O}/s{s}_d1.csv')}
    base3 = {r['seed']: float(r['score']) for r in load(f'{O}/s{s}_d3.csv')}
    for d in (1, 2, 3, 4, 5):
        p = f'{O}/s{s}_d{d}.csv'
        if not os.path.exists(p): continue
        R = load(p); sc = [float(r['score']) for r in R]; mt = [int(r['max_tile']) for r in R]
        steps = [int(r['steps']) for r in R]
        cr = st.mean(float(r['corner_rate']) for r in R)  # per-game mean, same as NTUPLE_RESULTS.md
        b = [int(r['boards_256']) for r in R]
        cr256 = st.mean(float(r['corner_rate_256']) for r in R if r['corner_rate_256'])
        tp = f'{O}/timing/s{s}_d{d}.csv'
        ms = None
        if not os.path.exists(tp) and d >= 4: tp = p  # d4/d5: ms/step from 4-proc eval run itself (nice 19)
        if os.path.exists(tp):
            T = load(tp); ms = 1000 * sum(float(r['game_sec']) for r in T) / sum(int(r['steps']) for r in T)
        diff = [float(r['score']) - base[r['seed']] for r in R]
        m, lo, hi = ci(diff) if d > 1 else (0, 0, 0)
        d3d = [float(r['score']) - base3[r['seed']] for r in R]
        m3, lo3, hi3 = ci(d3d) if d > 3 else (0, 0, 0)
        p4 = f'{O}/s{s}_d4.csv'
        m4 = lo4 = hi4 = None
        if d == 5 and os.path.exists(p4):
            b4 = {r['seed']: float(r['score']) for r in load(p4)}
            d4d = [float(r['score']) - b4[r['seed']] for r in R if r['seed'] in b4]
            m4, lo4, hi4 = ci(d4d)
        e = dict(diff_vs_d4=m4, diff4_ci=(lo4, hi4), diff_vs_d3=m3, diff3_ci=(lo3, hi3), seed=s, depth=d, n=len(R), mean=st.mean(sc), mean_ci=ci(sc)[1:], median=st.median(sc),
                 **{f'p{t}': sum(x >= t for x in mt) / len(R) for t in (2048, 4096, 8192, 16384, 32768)},
                 corner_all=cr, corner_256=cr256, ms_per_step=ms, diff_vs_d1=m, diff_ci=(lo, hi), max_score=max(sc))
        out[f's{s}_d{d}'] = e
        rows.append(e)
json.dump(out, open(f'{O}/summary.json', 'w'), indent=1)
pct = lambda x: f'{100*x:.1f}%'
print('| seed | d | 局數 | 平均 (95% CI) | 中位數 | ≥2048 | ≥4096 | ≥8192 | ≥16384 | ≥32768 | 角落（全部／≥256後） | ms/步 | 平均差 vs d=1（配對 95% CI） | 平均差 vs d=3（同 seed 配對 95% CI） | 平均差 vs d=4（共同 seed 配對 95% CI） |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for e in rows:
    dd = '—' if e['depth'] == 1 else f"{e['diff_vs_d1']:+,.0f} [{e['diff_ci'][0]:+,.0f}, {e['diff_ci'][1]:+,.0f}]"
    d3s = '—' if e['depth'] < 4 else f"{e['diff_vs_d3']:+,.0f} [{e['diff3_ci'][0]:+,.0f}, {e['diff3_ci'][1]:+,.0f}]"
    d4s = '—' if e['diff_vs_d4'] is None else f"{e['diff_vs_d4']:+,.0f} [{e['diff4_ci'][0]:+,.0f}, {e['diff4_ci'][1]:+,.0f}]"
    ms = f"{e['ms_per_step']:.4f}" if e['ms_per_step'] else '—'
    print(f"| {e['seed']} | {e['depth']} | {e['n']} | {e['mean']:,.0f} [{e['mean_ci'][0]:,.0f}, {e['mean_ci'][1]:,.0f}] | {e['median']:,.0f} | {pct(e['p2048'])} | {pct(e['p4096'])} | {pct(e['p8192'])} | {pct(e['p16384'])} | {pct(e['p32768'])} | {pct(e['corner_all'])}／{pct(e['corner_256'])} | {ms} | {dd} | {d3s} | {d4s} |")
