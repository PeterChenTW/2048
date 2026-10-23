"""Logs load1 and C_s303/R_s303/6H env-steps/s (from train.csv, trailing 180 s of elapsed_s) every 60 s. Read-only."""
import csv, time, sys, datetime
RUNS = {'C_s303': '/workspace/rl2048-review/experiments/runs/round2/C_s303/train.csv',
        'R_s303': '/workspace/rl2048-review/experiments/runs/round2/R_s303/train.csv',
        '6H_s101': '/workspace/rl2048-review/experiments/runs/E1-AS-6H_s101/train.csv'}
def rate(path, win=180):
    try:
        rows = list(csv.DictReader(open(path)))
        T = float(rows[-1]['elapsed_s']); s = [r for r in rows if float(r['elapsed_s']) > T - win]
        return sum(int(r['steps']) for r in s) / win, T
    except Exception:
        return float('nan'), float('nan')
out = open(sys.argv[1], 'a'); n = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
out.write('time,load1,' + ','.join(f'{k}_steps_per_s,{k}_elapsed' for k in RUNS) + '\n'); out.flush()
for i in range(n):
    l1 = open('/proc/loadavg').read().split()[0]
    vals = []
    for k, p in RUNS.items():
        r, T = rate(p); vals += [f'{r:.1f}', f'{T:.0f}']
    out.write(datetime.datetime.now().strftime('%H:%M:%S') + f',{l1},' + ','.join(vals) + '\n'); out.flush()
    time.sleep(60)
