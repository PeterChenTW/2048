"""Aggregate eval/*.csv (C eval output) -> eval/summary.json + markdown tables. Diagnostic only."""
import csv, json, glob, os, re, statistics as st, sys
sys.path.insert(0, '/workspace/rl2048-review/corner')
from corner_metrics import spearman
M = ['corner_rate', 'corner_rate_256', 'edge_rate_256', 'center_rate_256', 'leaves', 'leaves_256', 'leaves_per100', 'leaves_256_per100',
     'longest_streak', 'longest_streak_frac', 'final_mono_snake', 'final_mono_rowcol', 'mean_mono_snake']
def num(x): return None if x == '' else float(x)
S = {}
for f in sorted(glob.glob('eval/*.csv')):
    name = os.path.basename(f)[:-4]
    g = [{k: (num(v) if k not in ('seed',) else int(v)) for k, v in r.items()} for r in csv.DictReader(open(f))]
    if not g: continue
    sc = [r['score'] for r in g]; n = len(g)
    s = dict(games=n, mean_score=st.mean(sc), median_score=st.median(sc), sd_score=st.pstdev(sc), mean_steps=st.mean(r['steps'] for r in g),
             max_score=max(sc), **{f'reach{t}': sum(r['max_tile'] >= t for r in g) / n for t in (1024, 2048, 4096, 8192, 16384)},
             games_reaching_256=sum(r['corner_rate_256'] is not None for r in g))
    for k in M:
        xs = [r[k] for r in g if r[k] is not None]; s[k] = st.mean(xs) if xs else None
    h = [r for r in g if r['corner_rate_256'] is not None]
    s['spearman_cr256_score'] = spearman([r['corner_rate_256'] for r in h], [r['score'] for r in h]) if len(h) > 2 else None
    m = re.match(r's(\d)_t(\d+)', name); s['seed'] = int(m.group(1)) if m else None; s['train_s'] = int(m.group(2)) if m else None
    S[name] = s
json.dump(S, open('eval/summary.json', 'w'), indent=1)
f0 = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
def lab(t): return f'{t}s' if t < 60 else f'{t // 60}m'
print('| 模型 | 局數 | 平均 | 中位數 | ≥1024 | ≥2048 | ≥4096 | ≥8192 | 步數 | corner_rate | corner≥256 | edge≥256 | center≥256 | 移開/局 | 移開≥256/局 | 移開/100步 | 移開≥256/100步 | 最長連續 | 最長/局長 | 終局 mono_snake | 終局 mono_rc | mean_mono | ρ(cr256,分數) |')
print('|' + '---|' * 24)
order = sorted(S, key=lambda k: (S[k]['seed'] or 0, S[k]['train_s'] or -1))
for k in order:
    s = S[k]; nm = 'random' if s['seed'] is None else f"seed{s['seed']} {lab(s['train_s'])}"
    print(f"| {nm} | {s['games']} | {s['mean_score']:,.0f} | {s['median_score']:,.0f} | " + ' | '.join(f"{100*s[f'reach{t}']:.1f}%" for t in (1024, 2048, 4096, 8192)) +
          f" | {s['mean_steps']:,.0f} | {f0(s['corner_rate'])} | {f0(s['corner_rate_256'])} | {f0(s['edge_rate_256'])} | {f0(s['center_rate_256'])} | {f0(s['leaves'],1)} | {f0(s['leaves_256'],1)} | {f0(s['leaves_per100'],2)} | {f0(s['leaves_256_per100'],2)} | {f0(s['longest_streak'],0)} | {f0(s['longest_streak_frac'])} | {f0(s['final_mono_snake'])} | {f0(s['final_mono_rowcol'])} | {f0(s['mean_mono_snake'])} | {f0(s['spearman_cr256_score'],2)} |")
