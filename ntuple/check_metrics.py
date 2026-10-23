"""Cross-check C eval corner metrics against corner/corner_metrics.py (Python reference)."""
import sys, csv, math
sys.path.insert(0, '/workspace/rl2048-review/corner')
from corner_metrics import game_metrics
csv_path, dump_path = sys.argv[1], sys.argv[2]
rows = {int(r['seed']): r for r in csv.DictReader(open(csv_path))}
bad = n = 0
for line in open(dump_path):
    p = line.split(); seed = int(p[0])
    boards = []
    for h in p[1:]:
        x = int(h, 16); boards.append([(1 << ((x >> (4 * i)) & 15)) if (x >> (4 * i)) & 15 else 0 for i in range(16)])
    m = game_metrics(boards); r = rows[seed]; n += 1
    assert int(r['steps']) == len(boards) and int(r['max_tile']) == max(boards[-1])
    for k, v in m.items():
        if k == 'n_boards': continue
        c = r[k]
        if v is None: ok = c == ''
        else: ok = c != '' and abs(float(c) - v) < 1e-9
        if not ok: bad += 1; print('MISMATCH', seed, k, v, c)
print(f'checked {n} games, {len(m)-1} metrics each, mismatches={bad}')
sys.exit(1 if bad else 0)
