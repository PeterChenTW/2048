"""FAIR CONTROL driver: trains the ORIGINAL pre-fix agent (RL.py at e1a034d: ReLU Q output,
max-target (non-double), gamma 0.9, per-sample replay of 128 after every episode, eps 0.15*0.999/replay)
FROM SCRATCH on FastBoard (rule-identical to the legacy Board, see tests/test_fast_board_equivalence.py).
The learning loop is copied from the original main.py train_DDQN; only env (FastBoard), seeding,
time budget and CSV logging are added. Must be run from a checkout of e1a034d."""
import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RL import DDQNAgent  # original agent
from fast_board import FastBoard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=15)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--save-as', required=True)
    p.add_argument('--log', required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    np.random.seed(a.seed); torch.manual_seed(a.seed); random.seed(a.seed)
    agent = DDQNAgent(16, 4)
    start = time.monotonic(); deadline = start + a.minutes * 60
    env_steps = updates = 0
    rows = []
    with open(a.log, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['episode', 'score_v2', 'max_tile', 'steps', 'env_steps',
                                           'loss', 'epsilon', 'elapsed_s'])
        w.writeheader()
        e = 0
        while time.monotonic() < deadline:
            env = FastBoard(rng=random.Random(a.seed * 1_000_003 + e))
            state = torch.from_numpy(env.state).float()
            steps = 0
            while True:
                action = agent.act(state, env.get_invalid_moves())
                next_state, reward, done = env.move(env.all_moves[action])
                next_state = torch.from_numpy(np.asarray(next_state)).float()
                reward = torch.tensor(reward).float()
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if done or len(env.get_invalid_moves()) == 4:
                    break
            env_steps += steps
            loss = None
            if e > 128:
                agent.replay(128)
                updates += 128
                loss = agent.losses[-1]
            if e % 50 == 0:
                agent.update_target_model()
            row = dict(episode=e + 1, score_v2=int(env.score_v2), max_tile=int(np.max(env.state)),
                       steps=steps, env_steps=env_steps, loss=loss, epsilon=agent.epsilon,
                       elapsed_s=round(time.monotonic() - start, 2))
            w.writerow(row); fh.flush(); rows.append(row)
            if (e + 1) % 100 == 0:
                print(f"episode {e+1} score {env.score_v2} eps {agent.epsilon:.4f} loss {loss}", flush=True)
            e += 1
    agent.save(a.save_as)
    losses = [r['loss'] for r in rows if r['loss'] is not None]
    summ = dict(arm='control_e1a034d', episodes=len(rows), env_steps=env_steps, updates=updates,
                duration_s=time.monotonic() - start,
                score_last_100_mean=float(np.mean([r['score_v2'] for r in rows[-100:]])),
                loss_first_50_mean=float(np.mean(losses[:50])) if losses else None,
                loss_last_50_mean=float(np.mean(losses[-50:])) if losses else None,
                nan_count=int(sum(not np.isfinite(l) for l in losses)), checkpoint=a.save_as, seed=a.seed)
    Path(a.log).with_suffix('.json').write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
