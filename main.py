"""Train a DDQN agent on FastBoard, or launch the optional pygame GUI."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

DEFAULT_CHECKPOINT = 'model/nn_prioritize_replay_no_invalid_gamma_099_2'
FIELDS = ['episode', 'score_v2', 'max_tile', 'steps', 'epsilon', 'beta',
          'loss', 'q_mean', 'q_min', 'q_max', 'td_abs_mean', 'is_weight_mean',
          'nan_count', 'elapsed_s']


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='train using the fast CPU board')
    train.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    train.add_argument('--from-scratch', action='store_true')
    train.add_argument('--arch', choices=('legacy', 'onehot_dueling'), default='legacy')
    train.add_argument('--n-step', type=positive_int, default=1)
    train.add_argument('--save-as', help='output checkpoint (default: model/dqn_revamp_<timestamp>)')
    train.add_argument('--episodes', type=positive_int, default=1_000_000)
    train.add_argument('--minutes', type=nonnegative_float)
    train.add_argument('--gamma', type=nonnegative_float)
    train.add_argument('--reward-mode', choices=('log', 'linear'), default=None)
    train.add_argument('--reward-scale', type=positive_float, default=None)
    train.add_argument('--eps-schedule', choices=('update_exp', 'linear_steps'))
    train.add_argument('--eps-start', type=nonnegative_float)
    train.add_argument('--eps-min', type=nonnegative_float)
    train.add_argument('--eps-decay-steps', type=positive_int)
    train.add_argument('--seed', type=int)
    train.add_argument('--log', help='episode CSV (default: results/train_<output name>.csv)')
    train.add_argument('--threads', type=positive_int, default=1)
    train.add_argument('--replay-every', type=positive_int, default=4)
    train.add_argument('--replay-ratio', type=nonnegative_float,
                       help='gradient updates per env step (e.g. 0.78); overrides --replay-every')
    train.add_argument('--memory-capacity', type=positive_int,
                       help='replay memory capacity (default: DQNConfig default 6000)')
    train.add_argument('--per-max-priority', choices=('scan', 'tree'), default=None,
                       help='PER max priority strategy (default: scan)')
    train.add_argument('--track-replay-age', action='store_true',
                       help='track replay age in env steps and updates')
    train.add_argument('--save-every', type=positive_int, default=250)
    train.add_argument('--plot', action='store_true')
    play = commands.add_parser('play', help='launch the optional pygame GUI')
    play.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    return root


def finite(value):
    return value is not None and math.isfinite(float(value))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return statistics.fmean(values) if values else None


def train(args):
    import numpy as np
    import torch
    from config import DQNConfig
    from fast_board import FastBoard
    from RL import DDQNAgent
    from nstep import NStepAccumulator
    from rewards import shape_reward

    torch.set_num_threads(args.threads)
    config_args = {'seed': args.seed, 'arch': args.arch, 'n_step': args.n_step}
    if args.gamma is not None:
        config_args['gamma'] = args.gamma
    if args.memory_capacity is not None:
        config_args['memory_capacity'] = args.memory_capacity
    if args.per_max_priority is not None:
        config_args['per_max_priority'] = args.per_max_priority
    if args.track_replay_age:
        config_args['track_replay_age'] = True
    if args.reward_mode is not None:
        config_args['reward_mode'] = args.reward_mode
    if args.reward_scale is not None:
        config_args['reward_scale'] = args.reward_scale
    for argument, field in [('eps_schedule', 'epsilon_schedule'),
                            ('eps_start', 'epsilon_start'), ('eps_min', 'epsilon_min'),
                            ('eps_decay_steps', 'epsilon_decay_steps')]:
        value = getattr(args, argument)
        if value is not None:
            config_args[field] = value
    config = DQNConfig(**config_args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    agent = DDQNAgent(16, 4, config=config)
    if not args.from_scratch:
        try:
            agent.load(args.checkpoint)
        except ValueError as exc:
            if config.arch != 'legacy':
                raise ValueError('--arch onehot_dueling requires --from-scratch or '
                                 '--checkpoint pointing to a v2 checkpoint of the same arch') from exc
            raise
        print(f'Loaded checkpoint: {args.checkpoint}', flush=True)
    else:
        print('Starting from scratch', flush=True)

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.save_as or f'model/dqn_revamp_{stamp}')
    log = Path(args.log or f'results/train_{output.name}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.minutes * 60 if args.minutes is not None else math.inf
    env_steps = updates = entries = nan_count = 0
    rows, all_stats = [], []
    interrupted = False
    stopped_reason = None
    stop = False
    accumulator = None
    from step4_utils import ReplayScheduler
    scheduler = ReplayScheduler(args.replay_ratio) if args.replay_ratio else None

    def remember_emitted(transitions):
        for transition in transitions:
            invalid = np.flatnonzero(~np.asarray(transition.next_mask, dtype=bool))
            # Preserve the legacy one-step call contract, including wrappers.
            options = {} if config.n_step == 1 else {'discount': transition.discount}
            agent.remember(transition.state, transition.action, transition.reward,
                           transition.next_state, transition.done, invalid, **options)

    csv_fields = list(FIELDS)
    if config.track_replay_age:
        csv_fields.extend(['age_steps_mean', 'age_updates_mean'])
    with log.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        handle.flush()
        try:
            for episode in range(1, args.episodes + 1):
                if time.monotonic() >= deadline:
                    break
                board = FastBoard(rng=random.Random(None if args.seed is None else args.seed + episode - 1))
                state = torch.as_tensor(board.state, dtype=torch.float32)
                accumulator = NStepAccumulator(config.n_step, config.gamma)
                steps = 0
                episode_stats = []
                episode_nan = 0
                while True:
                    invalid = board.get_invalid_moves()
                    if len(invalid) == 4 or time.monotonic() >= deadline:
                        break
                    action = agent.act(state, invalid)
                    next_grid, reward, done = board.move(board.all_moves[action])
                    if config.reward_mode != 'log':
                        spawned = (reward != -1)
                        reward = shape_reward(board.get_score, spawned,
                                              mode=config.reward_mode,
                                              scale=config.reward_scale)
                    next_state = torch.as_tensor(next_grid, dtype=torch.float32)
                    next_invalid = board.get_invalid_moves()
                    done = bool(done or len(next_invalid) == 4)
                    next_mask = np.ones(4, dtype=bool)
                    next_mask[next_invalid] = False
                    remember_emitted(accumulator.push(state, action, reward, next_state,
                                                       done, next_mask))
                    state = next_state
                    steps += 1
                    env_steps += 1
                    agent.set_env_step(env_steps)
                    entries = agent.memory.size
                    if scheduler is not None:
                        due = scheduler.step() if entries >= config.batch_size else 0
                    else:
                        due = int(entries >= config.batch_size and env_steps % args.replay_every == 0)
                    for _ in range(due):
                        try:
                            stats = agent.replay()
                        except (FloatingPointError, RuntimeError) as exc:
                            if isinstance(exc, RuntimeError) and not any(
                                    marker in str(exc).lower() for marker in ('non-finite', 'nonfinite')):
                                raise
                            stats = {'nan': True, 'error': str(exc)}
                        else:
                            updates += 1
                        episode_stats.append(stats)
                        all_stats.append(stats)
                        if stats.get('nan', False) or not finite(stats.get('loss')):
                            episode_nan += 1
                            nan_count += 1
                            stopped_reason = 'nonfinite_replay'
                            print(f'Non-finite replay at episode {episode}, step {env_steps}: {stats}', flush=True)
                            stop = True
                            break
                    if done or stop:
                        break
                remember_emitted(accumulator.flush())
                row = dict(episode=episode, score_v2=int(board.score_v2),
                           max_tile=int(board.state.max()), steps=steps,
                           epsilon=float(agent.epsilon), beta=float(agent.beta),
                           nan_count=episode_nan, elapsed_s=time.monotonic() - started)
                stat_keys = ['loss', 'q_mean', 'td_abs_mean', 'is_weight_mean']
                if config.track_replay_age:
                    stat_keys.extend(['age_steps_mean', 'age_updates_mean'])
                for key in stat_keys:
                    row[key] = mean(s.get(key) for s in episode_stats)
                for key, aggregate in [('q_min', min), ('q_max', max)]:
                    values = [float(s[key]) for s in episode_stats if finite(s.get(key))]
                    row[key] = aggregate(values) if values else None
                writer.writerow({k: v if not isinstance(v, float) or finite(v) else None for k, v in row.items()})
                handle.flush()
                rows.append(row)
                if episode % config.target_update_episodes == 0:
                    agent.update_target_model()
                if episode % args.save_every == 0:
                    agent.save(str(output))
                if episode % 20 == 0:
                    print(f'Episode {episode}: score={board.score_v2} steps={steps} '
                          f'epsilon={agent.epsilon:.5f} loss={row["loss"]}', flush=True)
                if stop:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print('Interrupted; saving checkpoint.', flush=True)
        finally:
            if accumulator is not None:
                remember_emitted(accumulator.flush())
            agent.save(str(output))

    losses = [s.get('loss') for s in all_stats]
    q_ranges = {}
    for key in ('q_mean', 'q_min', 'q_max'):
        values = [float(s[key]) for s in all_stats if finite(s.get(key))]
        q_ranges[key] = {'min': min(values) if values else None,
                         'max': max(values) if values else None}
    summary = dict(arch=config.arch, n_step=config.n_step,
                   memory_capacity=config.memory_capacity,
                   per_max_priority=config.per_max_priority,
                   track_replay_age=config.track_replay_age,
                   episodes=len(rows),
                   gamma=config.gamma, reward_mode=config.reward_mode,
                   reward_scale=config.reward_scale,
                   epsilon_schedule=config.epsilon_schedule,
                   epsilon_start=config.epsilon_start, epsilon_min=config.epsilon_min,
                   epsilon_decay_steps=config.epsilon_decay_steps,
                   env_steps=env_steps, updates=updates,
                   replay_ratio_requested=args.replay_ratio,
                   replay_ratio_actual=(updates / env_steps) if env_steps else None,
                   duration_s=time.monotonic() - started,
                   score_last_100_mean=mean(r['score_v2'] for r in rows[-100:]),
                   loss_first_50_mean=mean(losses[:50]), loss_last_50_mean=mean(losses[-50:]),
                   q_stats_ranges=q_ranges, nan_count=nan_count, stopped_reason=stopped_reason,
                   interrupted=interrupted,
                   checkpoint=str(output), loaded_checkpoint=None if args.from_scratch else args.checkpoint)
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    log.with_suffix('.json').write_text(serialized + '\n')
    print(serialized, flush=True)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('Plot skipped: install matplotlib to use --plot.')
        else:
            fig, axes = plt.subplots(2, 1)
            axes[0].plot([r['episode'] for r in rows], [r['score_v2'] for r in rows])
            axes[0].set_ylabel('Score')
            axes[1].plot([r['episode'] for r in rows], [r['loss'] for r in rows])
            axes[1].set_ylabel('Mean loss')
            axes[1].set_xlabel('Episode')
            fig.tight_layout()
            fig.savefig(log.with_suffix('.png'))
            plt.close(fig)
    return 2 if nan_count else 0


def play(args):
    import pygame
    from game_2048 import Game

    pygame.init()
    try:
        pygame.display.set_mode((410, 410))
        pygame.display.set_caption('2048')
        # Game prepends model/ to its model_name argument.
        game = Game(4, role='AI', model_name=os.path.relpath(args.checkpoint, 'model'))
        game.draw()
        game.run()
    finally:
        pygame.quit()
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    return train(args) if args.command == 'train' else play(args)


if __name__ == '__main__':
    raise SystemExit(main())
