"""Read-only: env steps per active-second for C_s303/R_s303/6H_s101 in wall-clock periods, using monitor.csv (time->elapsed map)."""
import csv, datetime as dt, sys
RUNS = {'C_s303': ('/workspace/rl2048-review/experiments/runs/round2/C_s303/train.csv', 3),
        'R_s303': ('/workspace/rl2048-review/experiments/runs/round2/R_s303/train.csv', 5),
        '6H_s101': ('/workspace/rl2048-review/experiments/runs/E1-AS-6H_s101/train.csv', 7)}
mon = [r for r in csv.reader(open('logs/monitor.csv')) if r and r[0] != 'time']
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
P = [('開跑前 15 分', '16:04:45', '16:19:45'), ('兩個 seed 同跑', '16:19:45', '16:26:23'), ('只剩 seed 1', '16:26:23', '17:21:43'), ('訓練結束後', '17:21:43', sys.argv[1] if len(sys.argv) > 1 else '17:28:00')]
for name, (path, col) in RUNS.items():
    rows = [(float(r['elapsed_s']), int(r['steps'])) for r in csv.DictReader(open(path))]
    pts = [(sec(m[0]), float(m[col])) for m in mon]
    # extrapolate before monitor start using slope 1 (elapsed advances ~ wall while active)
    def el(t):
        if t <= pts[0][0]: return pts[0][1] - (pts[0][0] - t)
        for (a, ea), (b, eb) in zip(pts, pts[1:]):
            if a <= t <= b: return ea + (eb - ea) * (t - a) / (b - a)
        return pts[-1][1] + (t - pts[-1][0])
    out = []
    for pn, a, b in P:
        e0, e1 = el(sec(a)), min(el(sec(b)), rows[-1][0])
        st = sum(s for e, s in rows if e0 < e <= e1)
        out.append(f'{pn} {st / (e1 - e0):.1f}')
    print(name, ' | '.join(out))
