"""Plot learning curves of all comparison arms: score vs episode and vs env steps (+ moving average), and loss."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = [('control_e1a034d', 'A control (original e1a034d code)'),
        ('step2_legacy', 'B step-2 fixed, legacy net'),
        ('s3_onehot_dueling', 'C step-3 one-hot + dueling'),
        ('s3_onehot_dueling_n3', 'D step-3 one-hot + dueling + n-step 3')]


def load(path):
    rows = list(csv.DictReader(open(path)))
    ep = np.array([int(r['episode']) for r in rows])
    score = np.array([float(r['score_v2']) for r in rows])
    steps = np.cumsum([int(r['steps']) for r in rows])
    loss = np.array([float(r['loss']) if r.get('loss') not in (None, '', 'None') else np.nan for r in rows])
    return ep, steps, score, loss


def ma(x, w=100):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main(d=Path('results/compare'), w=100):
    data = {k: load(d / f'train_{k}.csv') for k, _ in ARMS if (d / f'train_{k}.csv').exists()}
    for xkey, xlabel, fname in [(0, 'episode', 'learning_curve_episodes.png'),
                                (1, 'environment steps', 'learning_curve_steps.png')]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for k, label in ARMS:
            if k not in data:
                continue
            x = data[k][xkey]
            line, = ax.plot(x, data[k][2], alpha=0.12, linewidth=0.6)
            ax.plot(x, ma(data[k][2], w), color=line.get_color(), linewidth=2, label=f'{label} (MA{w})')
        ax.set_xlabel(xlabel); ax.set_ylabel('score (score_v2)')
        ax.set_title(f'2048 DQN 15-min comparison, seed 101: score vs {xlabel}')
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(d / fname, dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, label in ARMS:
        if k not in data:
            continue
        loss = data[k][3]; m = np.isfinite(loss)
        ax.plot(data[k][1][m], ma(loss[m], 50), linewidth=1.5, label=label)
    ax.set_yscale('log'); ax.set_xlabel('environment steps'); ax.set_ylabel('loss (MA50 of per-episode mean)')
    ax.set_title('Training loss'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(d / 'loss_curve.png', dpi=120); plt.close(fig)
    print('wrote', [str(d / f) for f in ('learning_curve_episodes.png', 'learning_curve_steps.png', 'loss_curve.png')])


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/compare'))
