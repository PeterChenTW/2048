"""Idempotent step-4 supervisor. Safe to run any number of times (flock-protected).

1. (re)start every run that is not finished (no done.json) and not alive, with --resume;
2. eval every new 30-min snapshot (eval.py, 100 games, seeds 1000-1099, fast board) in a separate,
   niced 1-thread process, store JSON per snapshot and rebuild step4/eval_results.csv;
3. regenerate learning-curve PNGs into the repo's results/step4/;
4. rewrite the STEP4 STATUS block at the top of /workspace/rl2048-review/PROGRESS.md.
State lives under /workspace/rl2048-review/step4/.
"""
import csv, fcntl, json, os, re, subprocess, sys, time
from pathlib import Path

BASE = Path('/workspace/rl2048-review')
REPO = BASE / '2048'
CTL = BASE / 'wt-ctl'          # checkout of e1a034d (arm A code)
PY = str(BASE / '.venv/bin/python')
S4 = Path(os.environ.get('STEP4_DIR', str(BASE / 'step4')))
OUT = Path(os.environ.get('STEP4_OUT', str(REPO / 'results/step4')))
PROGRESS = Path(os.environ.get('STEP4_PROGRESS', str(BASE / 'PROGRESS.md')))
HOURS = float(os.environ.get('STEP4_HOURS', '3'))
SNAP = os.environ.get('STEP4_SNAPSHOT_MIN', '30')
RUNS = [
    dict(name='A_s201', arm='A', seed=201), dict(name='A_s202', arm='A', seed=202),
    dict(name='C_s201', arm='C', seed=201), dict(name='C_s202', arm='C', seed=202),
]
C_RATIO = 0.78


def log(msg):
    S4.mkdir(parents=True, exist_ok=True)
    with (S4 / 'supervisor.log').open('a') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def alive(run):
    pidf = S4 / 'runs' / run['name'] / 'pid'
    if not pidf.exists():
        return None
    pid = int(pidf.read_text().strip() or 0)
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='ignore')
    except OSError:
        return None
    return pid if str(S4 / 'runs' / run['name']) in cmd else None


def ensure_ctl():
    if not (CTL / 'RL.py').exists():
        subprocess.run(['git', '-C', str(REPO), 'worktree', 'prune'], check=False)
        subprocess.run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', '-f', str(CTL), 'e1a034d'], check=True)
        log('recreated wt-ctl at e1a034d')


def command(run):
    rd = str(S4 / 'runs' / run['name'])
    if run['arm'] == 'A':
        return [PY, str(REPO / 'tools/step4_train_control.py'), '--code-dir', str(CTL), '--run-dir', rd,
                '--seed', str(run['seed']), '--threads', '2', '--hours', str(HOURS), '--snapshot-minutes', SNAP, '--resume']
    return [PY, str(REPO / 'tools/step4_train.py'), '--run-dir', rd, '--seed', str(run['seed']),
            '--threads', '2', '--hours', str(HOURS), '--replay-ratio', str(C_RATIO),
            '--arch', 'onehot_dueling', '--n-step', '1', '--eps-start', '0.15', '--eps-min', '0.01',
            '--eps-decay-steps', '400000', '--snapshot-minutes', SNAP, '--resume']


def start_missing():
    started = []
    for run in RUNS:
        rd = S4 / 'runs' / run['name']; rd.mkdir(parents=True, exist_ok=True)
        if (rd / 'done.json').exists() or alive(run):
            continue
        if run['arm'] == 'A':
            ensure_ctl()
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        with (rd / 'run.log').open('a') as out:
            out.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); out.flush()
            p = subprocess.Popen(command(run), cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (rd / 'pid').write_text(str(p.pid))
        started.append(f"{run['name']}:{p.pid}")
        log(f"started {run['name']} pid {p.pid}")
    return started


def eval_snapshots():
    ev = S4 / 'evals'; ev.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        snaps = sorted((S4 / 'runs' / run['name'] / 'snapshots').glob('t*.pt'))
        for snap in snaps:
            out = ev / f"{run['name']}_{snap.stem}.json"
            if out.exists():
                continue
            cwd = CTL if run['arm'] == 'A' else REPO
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            tmp = out.with_suffix('.tmp')
            r = subprocess.run(['nice', '-n', '10', PY, 'eval.py', '--checkpoint', str(snap), '--games', '100',
                                '--seed-start', '1000', '--board', 'fast', '--json', str(tmp)],
                               cwd=str(cwd), env=env, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0 or not tmp.exists():
                log(f'eval failed {snap}: {r.stderr[-500:]}')
                continue
            data = json.loads(tmp.read_text())
            meta = json.loads(snap.with_suffix('.json').read_text())
            data.pop('games_detail', None)
            summary = {k: v for k, v in data.items() if not isinstance(v, list)}
            summary.update(run=run['name'], arm=run['arm'], seed=run['seed'], snapshot=str(snap), meta=meta)
            out.write_text(json.dumps(summary, indent=1)); tmp.unlink()
            log(f"eval {run['name']} {snap.stem}: mean {data['mean']}")
    rows = []
    for f in sorted(ev.glob('*.json')):
        d = json.loads(f.read_text()); m = d['meta']; rr = d['reach_rates']
        rows.append(dict(run=d['run'], arm=d['arm'], seed=d['seed'], minutes=m['minutes'],
                         active_s=round(m['active_s'], 1), wall_s=round(m['wall_s'], 1), episode=m['episode'],
                         env_steps=m['env_steps'], updates=m['updates'],
                         replay_ratio_actual=round(m['replay_ratio_actual'], 4), epsilon=round(m['epsilon'], 5),
                         mean=d['mean'], median=d['median'], max=d['max'], reach_512=rr['512'],
                         reach_1024=rr['1024'], reach_2048=rr['2048'], reach_4096=rr.get('4096', 0)))
    rows.sort(key=lambda r: (r['run'], r['minutes']))
    if rows:
        with (S4 / 'eval_results.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def plots(rows):
    import numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    colors = {'A_s201': 'tab:blue', 'A_s202': 'tab:cyan', 'C_s201': 'tab:red', 'C_s202': 'tab:orange'}
    labels = {'A': 'A control (e1a034d)', 'C': 'C onehot+dueling, ratio 0.78'}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for run in RUNS:
        rr = [r for r in rows if r['run'] == run['name']]
        if not rr:
            continue
        lab = f"{labels[run['arm']]} seed {run['seed']}"
        ax[0].plot([r['active_s'] / 3600 for r in rr], [r['mean'] for r in rr], 'o-', color=colors[run['name']], label=lab)
        ax[1].plot([r['env_steps'] for r in rr], [r['mean'] for r in rr], 'o-', color=colors[run['name']], label=lab)
        ax[2].plot([r['active_s'] / 3600 for r in rr], [100 * r['reach_1024'] for r in rr], 'o-', color=colors[run['name']], label=lab)
    for a_ in ax[:2]:
        a_.axhline(7737.52, color='gray', ls='--', label='baseline 7,737.52'); a_.axhline(15000, color='green', ls=':', label='target mean 15,000')
        a_.set_ylabel('eval mean score (100 games)')
    ax[2].axhline(80, color='green', ls=':', label='target 80%'); ax[2].axhline(18, color='gray', ls='--', label='baseline 18%')
    ax[2].set_ylabel('% games reaching 1024')
    ax[0].set_xlabel('active training hours'); ax[1].set_xlabel('environment steps'); ax[2].set_xlabel('active training hours')
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle('Step 4: eval every 30 min (eval.py, seeds 1000-1099)'); fig.tight_layout()
    fig.savefig(OUT / 'step4_eval_curves.png', dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    for run in RUNS:
        f = S4 / 'runs' / run['name'] / 'train.csv'
        if not f.exists():
            continue
        d = list(csv.DictReader(f.open()))
        if len(d) < 2:
            continue
        sc = np.array([float(r['score_v2']) for r in d]); st = np.array([float(r['env_steps']) for r in d])
        at = np.array([float(r['active_s']) for r in d]) / 3600
        c = np.cumsum(np.insert(sc, 0, 0)); w = 200
        ma = np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1)) for i in range(len(sc))])
        lab = f"{labels[run['arm']]} seed {run['seed']}"
        ax[0].plot(at, ma, color=colors[run['name']], label=lab); ax[1].plot(st, ma, color=colors[run['name']], label=lab)
    ax[0].set_xlabel('active training hours'); ax[1].set_xlabel('environment steps')
    for a_ in ax:
        a_.set_ylabel('training score, moving average of 200 episodes'); a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle('Step 4: training score (epsilon-greedy) moving average'); fig.tight_layout()
    fig.savefig(OUT / 'step4_train_curves.png', dpi=110); plt.close(fig)


def usage_summary():
    try:
        out = subprocess.run(['usage-check'], capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        return f'usage-check failed: {exc}'
    keep = []
    for line in out.splitlines():
        m = re.search(r'([\d.]+% left)', line)
        if m:
            keep.append(re.sub(r'\s+', ' ', line.split(m.group(1))[0].strip()) + ' ' + m.group(1))
    return '; '.join(keep)


def status(rows, started):
    lines = [f"## STEP4 STATUS (updated {time.strftime('%Y-%m-%d %H:%M')} Taipei)"]
    procs = 0
    lines.append('| run | arm | seed | state | PID | active min | wall min | episodes | env steps | updates/step | latest eval (t, mean, median, max, 512/1024/2048) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for run in RUNS:
        rd = S4 / 'runs' / run['name']; pid = alive(run)
        procs += bool(pid)
        st = 'DONE' if (rd / 'done.json').exists() else ('RUNNING' if pid else 'NOT RUNNING')
        last = None
        f = rd / 'train.csv'
        if f.exists():
            with f.open() as fh:
                for last in csv.DictReader(fh):
                    pass
        ev = [r for r in rows if r['run'] == run['name']]
        e = ev[-1] if ev else None
        evs = (f"t{e['minutes']}m: {e['mean']:.0f}, {e['median']:.0f}, {e['max']}, "
               f"{e['reach_512']:.0%}/{e['reach_1024']:.0%}/{e['reach_2048']:.0%}") if e else 'none yet'
        lines.append(f"| {run['name']} | {run['arm']} | {run['seed']} | {st} | {pid or '-'} | "
                     f"{float(last['active_s'])/60:.1f} | {float(last['wall_s'])/60:.1f} | {last['episode']} | {last['env_steps']} | {last['replay_ratio_actual']} | {evs} |"
                     if last else f"| {run['name']} | {run['arm']} | {run['seed']} | {st} | {pid or '-'} | 0 | 0 | 0 | 0 | - | {evs} |")
    lines.append(f"- training processes running: {procs}; subagents running: 0 (eval runs inside the supervisor, niced, 1 thread)")
    if started:
        lines.append(f"- (re)started this pass: {', '.join(started)}")
    lines.append(f"- eval history: {S4 / 'eval_results.csv'}; PNGs: {OUT / 'step4_eval_curves.png'}, {OUT / 'step4_train_curves.png'}")
    lines.append(f"- usage: {usage_summary()}")
    block = '\n'.join(lines) + '\n'
    text = PROGRESS.read_text() if PROGRESS.exists() else ''
    pat = re.compile(r'<!-- STEP4 STATUS BEGIN -->.*?<!-- STEP4 STATUS END -->\n', re.S)
    new = f'<!-- STEP4 STATUS BEGIN -->\n{block}<!-- STEP4 STATUS END -->\n'
    if pat.search(text):
        text = pat.sub(lambda m: new, text, count=1)
    else:
        # insert after the STEP4 START / LANES header block if present, else at top
        marker = '<!-- STEP4 HEADER END -->\n'
        text = text.replace(marker, marker + new, 1) if marker in text else new + text
    PROGRESS.write_text(text)


def main():
    S4.mkdir(parents=True, exist_ok=True)
    with (S4 / 'supervisor.lock').open('w') as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('another supervisor pass is running'); return
        started = start_missing()
        rows = eval_snapshots()
        try:
            plots(rows)
        except Exception as exc:
            log(f'plot failed: {exc}')
        status(rows, started)
        done = all((S4 / 'runs' / r['name'] / 'done.json').exists() for r in RUNS)
        pending = sum(1 for r in RUNS for s in (S4 / 'runs' / r['name'] / 'snapshots').glob('t*.pt')
                      if not (S4 / 'evals' / f"{r['name']}_{s.stem}.json").exists())
        print(json.dumps(dict(started=started, evals=len(rows), all_done=done, pending_evals=pending)))


if __name__ == '__main__':
    main()
