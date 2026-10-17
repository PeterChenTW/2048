"""Step-4 long run for the new agent (arm C: onehot_dueling, step-based epsilon, fixed replay ratio).

Resumable: full state (model, target, optimizer, replay memory + priorities, beta, epsilon, counters,
RNG states, replay-scheduler credit, active/wall time) is saved atomically to <run-dir>/state.pt every
--state-minutes of ACTIVE time and at every snapshot. Eval snapshots (model only, v2 format) are written to
<run-dir>/snapshots/t<minutes>.pt every --snapshot-minutes of active time. The budget (--hours) counts only
active time (gaps > 60 s between steps are not counted). Re-running with --resume continues exactly.
Writes <run-dir>/done.json when the budget is used up.
"""
import argparse, csv, dataclasses, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from config import DQNConfig
from fast_board import FastBoard
from RL import DDQNAgent
from nstep import NStepAccumulator
from step4_utils import ActiveClock, ReplayScheduler, atomic_save

FIELDS = ['episode', 'score_v2', 'max_tile', 'steps', 'env_steps', 'updates', 'replay_ratio_actual',
          'epsilon', 'beta', 'loss', 'q_mean', 'active_s', 'wall_s']


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--hours', type=float, default=3.0)
    p.add_argument('--replay-ratio', type=float, default=0.78)
    p.add_argument('--snapshot-minutes', type=float, default=30.0)
    p.add_argument('--state-minutes', type=float, default=10.0)
    p.add_argument('--arch', default='onehot_dueling')
    p.add_argument('--n-step', type=int, default=1)
    p.add_argument('--eps-start', type=float, default=0.15)
    p.add_argument('--eps-min', type=float, default=0.01)
    p.add_argument('--eps-decay-steps', type=int, default=400_000)
    p.add_argument('--resume', action='store_true', help='continue from <run-dir>/state.pt if present')
    return p.parse_args(argv)


def capture(agent, ctr, sched, clock, wall_s, cfg):
    m = agent.memory
    return dict(format='step4-state-v1', config=dataclasses.asdict(cfg),
                model=agent.model.state_dict(), target=agent.target_model.state_dict(),
                optimizer=agent.optimizer.state_dict(),
                mem_tree=m.tree.copy(), mem_data=m.data.copy(), mem_pointer=m.pointer, mem_size=m.size,
                beta=agent.beta, replay_calls=agent.replay_calls, epsilon=agent.epsilon,
                agent_rng=agent.rng.bit_generator.state, rng_py=random.getstate(),
                rng_np=np.random.get_state(), rng_torch=torch.get_rng_state(),
                sched_credit=sched.credit, active_s=clock.active_s, skipped_s=clock.skipped_s,
                wall_s=wall_s, **ctr)


def restore(agent, st):
    agent.model.load_state_dict(st['model']); agent.target_model.load_state_dict(st['target'])
    agent.optimizer.load_state_dict(st['optimizer'])
    m = agent.memory
    m.tree[:] = st['mem_tree']; m.data[:] = st['mem_data']; m.pointer = st['mem_pointer']; m.size = st['mem_size']
    agent.beta = st['beta']; agent.replay_calls = st['replay_calls']; agent.epsilon = st['epsilon']
    agent.rng.bit_generator.state = st['agent_rng']
    random.setstate(st['rng_py']); np.random.set_state(st['rng_np']); torch.set_rng_state(st['rng_torch'])


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    run = Path(a.run_dir); (run / 'snapshots').mkdir(parents=True, exist_ok=True)
    cfg = DQNConfig(seed=a.seed, arch=a.arch, n_step=a.n_step, epsilon_schedule='linear_steps',
                    epsilon_start=a.eps_start, epsilon_min=a.eps_min, epsilon_decay_steps=a.eps_decay_steps)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    agent = DDQNAgent(16, 4, config=cfg)
    ctr = dict(episode=0, env_steps=0, updates=0, next_snapshot=1, next_state=1, nan_count=0)
    sched = ReplayScheduler(a.replay_ratio)
    clock = ActiveClock()
    wall_prev = 0.0
    state_path = run / 'state.pt'
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        restore(agent, st)
        for k in ctr:
            ctr[k] = st[k]
        sched.credit = st['sched_credit']; clock.active_s = st['active_s']; clock.skipped_s = st['skipped_s']
        wall_prev = st['wall_s']
        print(f"Resumed from {state_path}: episode {ctr['episode']} steps {ctr['env_steps']} "
              f"active {clock.active_s/60:.1f} min", flush=True)
    else:
        print('Starting from scratch', flush=True)
    seg_start = time.monotonic()
    wall = lambda: wall_prev + time.monotonic() - seg_start
    budget = a.hours * 3600
    log = run / 'train.csv'
    # Truncate rows past the restored episode (episodes after the last state save are re-run).
    if log.exists():
        rows = list(csv.DictReader(log.open()))
        rows = [r for r in rows if int(r['episode']) <= ctr['episode']]
    else:
        rows = []
    with log.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()

        def save_state():
            atomic_save(capture(agent, ctr, sched, clock, wall(), cfg), state_path)

        while clock.active_s < budget:
            ctr['episode'] += 1
            ep = ctr['episode']
            board = FastBoard(rng=random.Random(a.seed * 1_000_003 + ep))
            state = torch.as_tensor(board.state, dtype=torch.float32)
            acc = NStepAccumulator(cfg.n_step, cfg.gamma)
            steps, losses, qs = 0, [], []
            while True:
                invalid = board.get_invalid_moves()
                if len(invalid) == 4:
                    break
                action = agent.act(state, invalid)
                grid, reward, done = board.move(board.all_moves[action])
                nxt = torch.as_tensor(grid, dtype=torch.float32)
                ninv = board.get_invalid_moves()
                done = bool(done or len(ninv) == 4)
                mask = np.ones(4, dtype=bool); mask[ninv] = False
                for t in acc.push(state, action, reward, nxt, done, mask):
                    inv = np.flatnonzero(~np.asarray(t.next_mask, dtype=bool))
                    opts = {} if cfg.n_step == 1 else {'discount': t.discount}
                    agent.remember(t.state, t.action, t.reward, t.next_state, t.done, inv, **opts)
                state = nxt; steps += 1; ctr['env_steps'] += 1
                agent.set_env_step(ctr['env_steps'])
                due = sched.step() if agent.memory.size >= cfg.batch_size else 0
                for _ in range(due):
                    try:
                        s = agent.replay()
                    except (FloatingPointError, RuntimeError) as exc:
                        ctr['nan_count'] += 1
                        print(f'Non-finite replay: {exc}', flush=True)
                        save_state()
                        raise SystemExit(2)
                    ctr['updates'] += 1; losses.append(s['loss']); qs.append(s['q_mean'])
                clock.tick()
                if done:
                    break
            if ep % cfg.target_update_episodes == 0:
                agent.update_target_model()
            w.writerow(dict(episode=ep, score_v2=int(board.score_v2), max_tile=int(board.state.max()),
                            steps=steps, env_steps=ctr['env_steps'], updates=ctr['updates'],
                            replay_ratio_actual=round(ctr['updates'] / max(1, ctr['env_steps']), 5),
                            epsilon=round(agent.epsilon, 5), beta=round(agent.beta, 5),
                            loss=float(np.mean(losses)) if losses else '', q_mean=float(np.mean(qs)) if qs else '',
                            active_s=round(clock.active_s, 1), wall_s=round(wall(), 1)))
            fh.flush()
            if ep % 100 == 0:
                print(f"ep {ep} score {board.score_v2} steps {ctr['env_steps']} upd {ctr['updates']} "
                      f"eps {agent.epsilon:.4f} active {clock.active_s/60:.1f}m", flush=True)
            snap_due = clock.active_s >= ctr['next_snapshot'] * a.snapshot_minutes * 60
            if snap_due:
                mins = int(round(ctr['next_snapshot'] * a.snapshot_minutes))
                ctr['next_snapshot'] += 1
                agent.save(str(run / 'snapshots' / f't{mins:04d}.pt'))
                meta = dict(minutes=mins, active_s=clock.active_s, wall_s=wall(), episode=ep,
                            env_steps=ctr['env_steps'], updates=ctr['updates'],
                            replay_ratio_actual=ctr['updates'] / max(1, ctr['env_steps']),
                            epsilon=agent.epsilon)
                (run / 'snapshots' / f't{mins:04d}.json').write_text(json.dumps(meta))
            if snap_due or clock.active_s >= ctr['next_state'] * a.state_minutes * 60:
                while clock.active_s >= ctr['next_state'] * a.state_minutes * 60:
                    ctr['next_state'] += 1
                save_state()
    save_state()
    agent.save(str(run / 'final.pt'))
    done = dict(seed=a.seed, arch=a.arch, n_step=a.n_step, replay_ratio_requested=a.replay_ratio,
                replay_ratio_actual=ctr['updates'] / max(1, ctr['env_steps']), episodes=ctr['episode'],
                env_steps=ctr['env_steps'], updates=ctr['updates'], active_s=clock.active_s,
                skipped_freeze_s=clock.skipped_s, wall_s=wall(), nan_count=ctr['nan_count'],
                eps_start=a.eps_start, eps_min=a.eps_min, eps_decay_steps=a.eps_decay_steps)
    (run / 'done.json').write_text(json.dumps(done, indent=2))
    print(json.dumps(done, indent=2), flush=True)


if __name__ == '__main__':
    main()
