import json, glob, re
S = json.load(open('/workspace/rl2048-review/experiments/wt-ntuple-expectimax/results/ntuple/expectimax/summary.json'))
games = [('1', 1813, '中位數局', '兩種深度都落在各自中位數附近'),
         ('1', 1713, '搜尋反而輸的局', 'd=3 只到第 1 四分位，比 d=1 低'),
         ('2', 1523, '中位數局（seed 2）', 'seed 2 網路，兩邊都接近中位數'),
         ('1', 1364, '最佳局', '1000 局中 d=3 的最高分')]
G = []
for net, e, title, note in games:
    g = dict(net=net, eval=e, title=title, note=note)
    for d in (1, 3):
        tok = open(f'traj_s{net}_e{e}_d{d}.txt').read().split()[1:]
        g[f'b{d}'] = ''.join(t.split(':')[0] for t in tok)
        g[f's{d}'] = [int(t.split(':')[1]) for t in tok]
    G.append(g)
summ = {k: dict(mean=v['mean'], lo=v['mean_ci'][0], hi=v['mean_ci'][1], median=v['median'],
                p=[v['p2048'], v['p4096'], v['p8192'], v['p16384']], ms=v['ms_per_step'],
                diff=v['diff_vs_d1'], dlo=v['diff_ci'][0], dhi=v['diff_ci'][1]) for k, v in S.items()}
html = open('template.html').read().replace('/*DATA*/', 'const GAMES=' + json.dumps(G, separators=(',', ':')) + ';const SUMM=' + json.dumps(summ) + ';')
open('expectimax_demo.html', 'w').write(html)
print(len(html))
