import json, glob, statistics as st, re
rows = {}
out = []
for f in sorted(glob.glob('r*_[AB].json'), key=lambda x: (int(re.findall(r'r(\d+)_', x)[0]), x)):
    d = json.load(open(f)); r = int(re.findall(r'r(\d+)_', f)[0]); proc = f[-6]
    for rd in d['rounds']:
        res = {k: v['measure_c'] for k, v in rd['results'].items()}
        base = 't1' if 't1' in res else ('t1_i1' if 't1_i1' in res else ('ref' if 'ref' in res else 'ref_i1'))
        for k, v in res.items():
            rows.setdefault(k, []).append((r, v['env_steps_per_sec'], v['updates_per_sec'], res[base]['env_steps_per_sec'], res[base]['updates_per_sec'], base, rd['loadavg_start'][0], rd['loadavg_end'][0], rd['start_time'][11:19]))
print('| variant | round | time | load start/end | env steps/s | updates/s | baseline | ratio steps | ratio upd |')
print('|---|---|---|---|---|---|---|---|---|')
summ = {}
for k, lst in rows.items():
    for r, s, u, bs, bu, base, l0, l1, t in lst:
        print(f'| {k} | {r} | {t} | {l0:.2f}/{l1:.2f} | {s:.1f} | {u:.2f} | {base} | {s/bs:.3f} | {u/bu:.3f} |')
        summ.setdefault(k, []).append((s/bs, u/bu))
print()
print('| variant | n rounds | median ratio env steps/s | median ratio updates/s | min–max steps |')
print('|---|---|---|---|---|')
for k, v in summ.items():
    a = [x[0] for x in v]; b = [x[1] for x in v]
    print(f'| {k} | {len(v)} | {st.median(a):.3f} | {st.median(b):.3f} | {min(a):.3f}–{max(a):.3f} |')
