"""Build the step-4 final results table (results/step4/RESULTS.md + final_table.csv) from step4/eval_results.csv."""
import csv, json, sys
from pathlib import Path
S4 = Path('/workspace/rl2048-review/step4'); OUT = Path('/workspace/rl2048-review/2048/results/step4')
rows = list(csv.DictReader((S4 / 'eval_results.csv').open()))
for r in rows:
    for k in ('minutes', 'mean', 'median', 'max', 'reach_512', 'reach_1024', 'reach_2048', 'env_steps', 'updates', 'replay_ratio_actual'):
        r[k] = float(r[k])
final = {}
for r in rows:
    if r['run'] not in final or r['minutes'] > final[r['run']]['minutes']:
        final[r['run']] = r
best = {}
for r in rows:
    if r['run'] not in best or r['mean'] > best[r['run']]['mean']:
        best[r['run']] = r
BASE, TGT_MEAN, TGT_1024 = 7737.52, 15000, 0.80
M = ['mean', 'median', 'max', 'reach_512', 'reach_1024', 'reach_2048']
def fmt(k, v):
    return f"{v:.0%}" if k.startswith('reach') else f"{v:,.0f}" if k != 'mean' else f"{v:,.2f}"
L = ['# Step 4 results: 3 h active training, eval.py 100 games (seeds 1000-1099, fast board)', '',
     f'Target: mean > 15,000 and reach-1024 >= 80%. Baseline (old checkpoint, 63k episodes): mean 7,737.52, reach-1024 18%.', '',
     '## Per run, final snapshot', '',
     '| run | t (min) | env steps | updates | updates/step | ' + ' | '.join(M) + ' | target met | vs baseline |',
     '|' + '---|' * (len(M) + 7)]
for n in sorted(final):
    r = final[n]
    met = 'yes' if r['mean'] > TGT_MEAN and r['reach_1024'] >= TGT_1024 else 'no'
    L.append(f"| {n} | {r['minutes']:.0f} | {r['env_steps']:,.0f} | {r['updates']:,.0f} | {r['replay_ratio_actual']:.3f} | "
             + ' | '.join(fmt(k, r[k]) for k in M) + f" | {met} | {r['mean'] - BASE:+,.0f} |")
for title, src in (('final snapshot', final), ('best snapshot per run (by mean)', best)):
    L += ['', f'## Per arm, {title}: mean across seeds [min, max]', '', '| arm | ' + ' | '.join(M) + ' |', '|' + '---|' * (len(M) + 1)]
    for arm in ('A', 'C'):
        rr = [src[n] for n in sorted(src) if src[n]['arm'] == arm]
        cells = []
        for k in M:
            v = [r[k] for r in rr]
            cells.append(f"{fmt(k, sum(v)/len(v))} [{fmt(k, min(v))}, {fmt(k, max(v))}]")
        L.append(f"| {arm} | " + ' | '.join(cells) + ' |')
    if src is best:
        L.append(''); L.append('Best snapshot times: ' + ', '.join(f"{n} t{best[n]['minutes']:.0f}" for n in sorted(best)))
L += ['', '## Eval history (every 30 min)', '', '| run | t | mean | median | max | 512 | 1024 | 2048 |', '|---|---|---|---|---|---|---|---|']
for r in sorted(rows, key=lambda r: (r['run'], r['minutes'])):
    L.append(f"| {r['run']} | {r['minutes']:.0f} | {r['mean']:,.2f} | {r['median']:,.0f} | {r['max']:,.0f} | {r['reach_512']:.0%} | {r['reach_1024']:.0%} | {r['reach_2048']:.0%} |")
L += ['', 'Curves: step4_eval_curves.png, step4_train_curves.png. Checkpoints (not committed): /workspace/rl2048-review/step4/runs/<run>/{final.pt,snapshots/t*.pt,state*}.']
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
import shutil; shutil.copy(S4 / 'eval_results.csv', OUT / 'eval_results.csv')
print('\n'.join(L))
