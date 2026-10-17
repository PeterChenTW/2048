"""Post-hoc Q / target / TD diagnostics; never updates checkpoint weights.

Run explicitly with --checkpoint; this tool adds no training-time behavior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RL import load_checkpoint
from fast_board import FastBoard
from nstep import NStepAccumulator, double_dqn_targets
from rewards import shape_reward


NOTES = [
    'Target weights are not saved: checkpoint network is BOTH online and target; '
    'these TD errors do not reconstruct historical training TD errors.',
    'Training replay in RL.py uses PER-weighted mean squared TD loss (not Huber), '
    'RMSprop, and clip_grad_norm_ (default threshold 1.0).',
    'Gradient diagnostics use uniformly sampled minibatches and unweighted mean '
    'squared TD loss, without optimizer steps. They do not reproduce PER sampling '
    'or importance weights. Fraction clipped uses threshold 1.0; the checkpoint '
    'threshold and its corresponding fraction are also recorded.',
    'States come from fresh epsilon-greedy FastBoard episodes, not the training '
    'replay buffer or restart pool. Each checkpoint induces its own state distribution.',
    'Returns honor checkpoint n_step and gamma; reward statistics describe individual '
    'environment rewards. The final partial episode is truncated at --states and '
    'bootstraps normally. Complete-episode and partial-episode statistics are separate.',
    'max_a Q(s) is over all four actions (unmasked); action selection and bootstrap '
    'targets are masked to legal actions.',
]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError('diagnostic values must be nonempty and finite')
    result = {f'p{p}': float(np.percentile(values, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    result.update(mean=float(values.mean()), std=float(values.std()))
    return result


def collect_transitions(agent, states=4000, seed=0, epsilon=0.05):
    """Return replay-style n-step transitions, episode records, and raw step rewards."""
    action_rng = random.Random(seed + 1_000_000)
    transitions, episodes, rewards = [], [], []
    while len(rewards) < states:
        episode_seed = seed + len(episodes)
        board = FastBoard(rng=random.Random(episode_seed))
        accumulator = NStepAccumulator(agent.config.n_step, agent.config.gamma)
        steps = 0
        done = False
        while len(rewards) < states:
            invalid = board.get_invalid_moves()
            if len(invalid) == 4:
                done = True
                break
            state = board.state.astype(np.float32).reshape(16).copy()
            valid = [a for a in range(4) if a not in invalid]
            if action_rng.random() < epsilon:
                action = action_rng.choice(valid)
            else:
                with torch.no_grad():
                    q = agent.model(torch.from_numpy(state)).reshape(4)
                    q[invalid] = -torch.inf
                    action = int(q.argmax().item())
            next_grid, env_reward, done = board.move(action)
            reward = float(env_reward)
            if agent.config.reward_mode != 'log':
                reward = shape_reward(board.get_score, env_reward != -1,
                                      mode=agent.config.reward_mode,
                                      scale=agent.config.reward_scale)
            next_mask = np.ones(4, dtype=bool)
            next_mask[board.get_invalid_moves()] = False
            done = bool(done or not next_mask.any())
            transitions.extend(accumulator.push(
                state, action, reward, next_grid.astype(np.float32).reshape(16).copy(),
                done, next_mask))
            rewards.append(reward)
            steps += 1
            if done:
                break
        transitions.extend(accumulator.flush())
        episodes.append(dict(seed=episode_seed, score=int(board.score_v2), steps=steps,
                             complete=done))
    return transitions, episodes, rewards


def batch_terms(model, transitions):
    """Use the training target helper with detached bootstrap and TD = target - Q."""
    states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    q_all = model(states).reshape(len(transitions), 4)
    q = q_all.gather(1, actions[:, None]).squeeze(1)
    target = double_dqn_targets(
        model, model, rewards, next_states, [t.done for t in transitions],
        np.stack([t.next_mask for t in transitions]), [t.discount for t in transitions])
    return dict(q=q, q_max=q_all.max(dim=1).values, target=target, td=target - q)


def analyze_checkpoint(checkpoint, *, label, states=4000, seed=0, epsilon=0.05,
                       batch=128, grad_batches=20):
    torch.set_num_threads(1)
    started = time.perf_counter()
    agent = load_checkpoint(checkpoint)
    result = dict(label=label, checkpoint=str(Path(checkpoint).resolve()),
                  config=asdict(agent.config), architecture=agent.config.arch)
    if agent.config.arch == 'onehot_afterstate':
        result.update(status='skipped', reason='onehot_afterstate predicts V(afterstate); '
                      'AfterstateAgent target semantics are not implemented by this Q-only tool.')
        return result
    agent.model.cpu().eval()
    transitions, episodes, rewards = collect_transitions(agent, states, seed, epsilon)
    values = {key: [] for key in ('q', 'q_max', 'target', 'td')}
    with torch.no_grad():
        for start in range(0, len(transitions), batch):
            terms = batch_terms(agent.model, transitions[start:start + batch])
            for key, value in terms.items():
                values[key].extend(value.cpu().tolist())
    values['td_abs'] = np.abs(values['td'])
    values['reward'] = rewards
    metrics = {key: summarize(value) for key, value in values.items()}

    rng = np.random.default_rng(seed)
    norms = []
    for _ in range(grad_batches):
        indices = rng.choice(len(transitions), size=batch, replace=len(transitions) < batch)
        terms = batch_terms(agent.model, [transitions[int(i)] for i in indices])
        loss = terms['td'].square().mean()
        gradients = torch.autograd.grad(loss, tuple(agent.model.parameters()))
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g.detach()) for g in gradients]))
        norms.append(float(norm.item()))
    metrics['grad_norm'] = summarize(norms)
    complete = [episode for episode in episodes if episode['complete']]
    result.update(
        status='ok', states=len(transitions), env_steps=len(rewards), optimizer_updates=0,
        seed=seed, epsilon=epsilon, batch=batch, grad_batches=grad_batches,
        uniform_sampling_replacement=len(transitions) < batch,
        metrics=metrics, fraction_abs_td_gt_1=float(np.mean(values['td_abs'] > 1)),
        fraction_clipped=float(np.mean(np.asarray(norms) > 1.0)),
        grad_clip=agent.config.grad_clip,
        fraction_above_config_grad_clip=float(np.mean(np.asarray(norms) > agent.config.grad_clip)),
        grad_norms=norms,
        episodes=dict(records=episodes, count=len(episodes), complete_count=len(complete),
                      mean_score=float(np.mean([e['score'] for e in episodes])),
                      mean_steps=float(np.mean([e['steps'] for e in episodes])),
                      complete_mean_score=float(np.mean([e['score'] for e in complete])) if complete else None,
                      complete_mean_steps=float(np.mean([e['steps'] for e in complete])) if complete else None),
        elapsed_seconds=time.perf_counter() - started)
    return result


def markdown_report(report):
    lines = ['# Checkpoint Q / target / TD diagnostics', '',
             *[f'- {note}' for note in report['notes']], '',
             '| label | reward_mode | gamma | Q p5/p50/p95 | |TD| p50/p95/p99 | '
             'frac |TD|>1 | grad-norm p50/p95 | frac clipped | reward p50/p95 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    # Escape literal pipes in column names so Markdown keeps nine columns.
    lines[-2] = lines[-2].replace('|TD|', '\\|TD\\|')
    def quantiles(metric, ps):
        return '/'.join(f'{metric[f"p{p}"]:.5g}' for p in ps)
    for label, result in report['results'].items():
        safe_label = label.replace('|', '\\|').replace('\n', ' ')
        if result['status'] != 'ok':
            continue
        config, metrics = result['config'], result['metrics']
        lines.append(
            f'| {safe_label} | {config["reward_mode"]} | {config["gamma"]:g} | '
            f'{quantiles(metrics["q"], (5, 50, 95))} | '
            f'{quantiles(metrics["td_abs"], (50, 95, 99))} | '
            f'{result["fraction_abs_td_gt_1"]:.1%} | '
            f'{quantiles(metrics["grad_norm"], (50, 95))} | '
            f'{result["fraction_clipped"]:.1%} | {quantiles(metrics["reward"], (50, 95))} |')
    lines.extend(['', 'Episode scores use score_v2 (merge points). Means across all collected '
                  'episode segments include the final partial episode; completed means exclude it.', ''])
    for label, result in report['results'].items():
        if result['status'] != 'ok':
            lines.append(f'- {label}: skipped — {result["reason"]}')
            continue
        episodes = result['episodes']
        complete_score = episodes['complete_mean_score']
        complete_steps = episodes['complete_mean_steps']
        completed = (f'{complete_score:.2f} / {complete_steps:.2f}'
                     if complete_score is not None else 'n/a (no completed episodes)')
        lines.append(f'- {label}: {result["env_steps"]} steps; {result["states"]} transitions; '
                     f'{episodes["complete_count"]} complete / {episodes["count"]} collected episodes; '
                     f'mean score / steps {episodes["mean_score"]:.2f} / {episodes["mean_steps"]:.2f}; '
                     f'completed mean score / steps {completed}; '
                     f'{result["grad_batches"]} gradient batches; '
                     f'{result["optimizer_updates"]} optimizer updates; all metrics finite; '
                     f'{result["elapsed_seconds"]:.1f}s.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', required=True, type=Path)
    parser.add_argument('--label', nargs='+', action='extend', help='one label per checkpoint, in order')
    parser.add_argument('--states', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--grad-batches', type=int, default=20)
    parser.add_argument('--threads', type=int, choices=(1,), default=1)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--md', type=Path)
    args = parser.parse_args(argv)
    if min(args.states, args.batch, args.grad_batches) < 1:
        parser.error('--states, --batch and --grad-batches must be positive')
    if not 0 <= args.epsilon <= 1:
        parser.error('--epsilon must be in [0, 1]')
    labels = args.label or [str(path) for path in args.checkpoint]
    if len(labels) != len(args.checkpoint) or len(set(labels)) != len(labels):
        parser.error('provide one unique --label per checkpoint')
    torch.set_num_threads(1)
    report = dict(notes=NOTES, results={})
    for checkpoint, label in zip(args.checkpoint, labels):
        result = analyze_checkpoint(checkpoint, label=label, states=args.states, seed=args.seed,
                                    epsilon=args.epsilon, batch=args.batch,
                                    grad_batches=args.grad_batches)
        report['results'][label] = result
        print(f'{label}: {result["status"]}' + (
            f', {result["env_steps"]} steps, 0 updates, finite diagnostics, '
            f'{result["elapsed_seconds"]:.1f}s' if result['status'] == 'ok'
            else f' — {result["reason"]}'), file=sys.stderr, flush=True)
    markdown = markdown_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.md:
        args.md.write_text(markdown)
    print(markdown)
    return report


if __name__ == '__main__':
    main()
