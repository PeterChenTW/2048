"""Step-4 long run for arm A (FAIR CONTROL): the ORIGINAL e1a034d agent (ReLU Q, max target, gamma 0.9,
128 single-sample updates after every episode once episode > 128, eps 0.15*0.999 per replay call),
from scratch on FastBoard. Learning loop identical to tools/control_train_e1a034d.py; this adds an
active-time budget, 30-min eval snapshots and full-state resume. Run with --code-dir pointing to a
checkout of e1a034d (it imports RL.py / fast_board.py from there)."""
import argparse, csv, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--hours', type=float, default=3.0)
    p.add_argument('--snapshot-minutes', type=float, default=30.0)
    p.add_argument('--state-minutes', type=float, default=10.0)
    p.add_argument('--resume', action='store_true')
    return p.parse_args(argv)


FIELDS = ['episode', 'score_v2', 'max_tile', 'steps', 'env_steps', 'updates', 'replay_ratio_actual',
          'epsilon', 'loss', 'active_s', 'wall_s']


def main(argv=None):
    a = parse(argv)
    sys.path.insert(0, str(Path(a.code_dir).resolve()))
    import numpy as np
    import torch
    from RL import DDQNAgent  # original agent from e1a034d
    from fast_board import FastBoard
    sys.path.insert(0, str(HERE))
    from step4_utils import ActiveClock, atomic_save
    torch.set_num_threads(a.threads)
    run = Path(a.run_dir); (run / 'snapshots').mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed); random.seed(a.seed)
    agent = DDQNAgent(16, 4)
    ctr = dict(episode=0, env_steps=0, updates=0, next_snapshot=1, next_state=1)
    clock = ActiveClock(); wall_prev = 0.0
    sp = run / 'state.pt'
    if a.resume and sp.exists():
        st = torch.load(sp, map_location='cpu', weights_only=False)
        agent.model.load_state_dict(st['model']); agent.target_model.load_state_dict(st['target'])
        agent.optimizer.load_state_dict(st['optimizer']); agent.memory = st['memory']
        agent.epsilon = st['epsilon']; agent.beta = st['beta']
        random.setstate(st['rng_py']); np.random.set_state(st['rng_np']); torch.set_rng_state(st['rng_torch'])
        for k in ctr:
            ctr[k] = st[k]
        clock.active_s = st['active_s']; clock.skipped_s = st['skipped_s']; wall_prev = st['wall_s']
        print(f"Resumed: episode {ctr['episode']} active {clock.active_s/60:.1f} min", flush=True)
    else:
        print('Starting from scratch (original e1a034d agent)', flush=True)
    seg = time.monotonic()
    wall = lambda: wall_prev + time.monotonic() - seg

    def save_state():
        atomic_save(dict(format='step4-control-state-v1', model=agent.model.state_dict(),
                         target=agent.target_model.state_dict(), optimizer=agent.optimizer.state_dict(),
                         memory=agent.memory, epsilon=agent.epsilon, beta=agent.beta,
                         rng_py=random.getstate(), rng_np=np.random.get_state(),
                         rng_torch=torch.get_rng_state(), active_s=clock.active_s,
                         skipped_s=clock.skipped_s, wall_s=wall(), **ctr), sp)

    log = run / 'train.csv'
    rows = [r for r in csv.DictReader(log.open()) if int(r['episode']) <= ctr['episode']] if log.exists() else []
    budget = a.hours * 3600
    with log.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        while clock.active_s < budget:
            e = ctr['episode']  # 0-based like the original loop
            env = FastBoard(rng=random.Random(a.seed * 1_000_003 + e))
            state = torch.from_numpy(env.state).float()
            steps = 0
            while True:
                action = agent.act(state, env.get_invalid_moves())
                ns, reward, done = env.move(env.all_moves[action])
                ns = torch.from_numpy(np.asarray(ns)).float()
                agent.remember(state, action, torch.tensor(reward).float(), ns, done)
                state = ns; steps += 1
                clock.tick()
                if done or len(env.get_invalid_moves()) == 4:
                    break
            ctr['env_steps'] += steps
            loss = ''
            if e > 128:
                agent.replay(128)
                ctr['updates'] += 128
                loss = agent.losses[-1]
                if not np.isfinite(loss):
                    print('non-finite loss', flush=True); save_state(); raise SystemExit(2)
                clock.tick()
            if e % 50 == 0:
                agent.update_target_model()
            ctr['episode'] += 1
            w.writerow(dict(episode=e + 1, score_v2=int(env.score_v2), max_tile=int(np.max(env.state)),
                            steps=steps, env_steps=ctr['env_steps'], updates=ctr['updates'],
                            replay_ratio_actual=round(ctr['updates'] / max(1, ctr['env_steps']), 5),
                            epsilon=round(agent.epsilon, 5), loss=loss,
                            active_s=round(clock.active_s, 1), wall_s=round(wall(), 1)))
            fh.flush()
            if ctr['episode'] % 100 == 0:
                print(f"ep {ctr['episode']} score {env.score_v2} steps {ctr['env_steps']} "
                      f"eps {agent.epsilon:.4f} active {clock.active_s/60:.1f}m", flush=True)
            snap_due = clock.active_s >= ctr['next_snapshot'] * a.snapshot_minutes * 60
            if snap_due:
                mins = int(round(ctr['next_snapshot'] * a.snapshot_minutes)); ctr['next_snapshot'] += 1
                agent.save(str(run / 'snapshots' / f't{mins:04d}.pt'))
                (run / 'snapshots' / f't{mins:04d}.json').write_text(json.dumps(dict(
                    minutes=mins, active_s=clock.active_s, wall_s=wall(), episode=ctr['episode'],
                    env_steps=ctr['env_steps'], updates=ctr['updates'],
                    replay_ratio_actual=ctr['updates'] / max(1, ctr['env_steps']), epsilon=agent.epsilon)))
            if snap_due or clock.active_s >= ctr['next_state'] * a.state_minutes * 60:
                while clock.active_s >= ctr['next_state'] * a.state_minutes * 60:
                    ctr['next_state'] += 1
                save_state()
    save_state()
    agent.save(str(run / 'final.pt'))
    done = dict(seed=a.seed, arm='A_control_e1a034d', episodes=ctr['episode'], env_steps=ctr['env_steps'],
                updates=ctr['updates'], replay_ratio_actual=ctr['updates'] / max(1, ctr['env_steps']),
                active_s=clock.active_s, skipped_freeze_s=clock.skipped_s, wall_s=wall(),
                final_epsilon=agent.epsilon)
    (run / 'done.json').write_text(json.dumps(done, indent=2))
    print(json.dumps(done, indent=2), flush=True)


if __name__ == '__main__':
    main()
