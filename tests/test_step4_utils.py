from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step4_utils import ActiveClock, ReplayScheduler, atomic_save


@pytest.mark.parametrize('ratio,steps,expected', [(0.78, 100, 78), (0.78, 171216, 133548),
                                                  (0.25, 8, 2), (1.0, 5, 5), (2.5, 4, 10)])
def test_replay_ratio_counts(ratio, steps, expected):
    s = ReplayScheduler(ratio)
    assert sum(s.step() for _ in range(steps)) == expected


def test_replay_ratio_spread_and_resume():
    s = ReplayScheduler(0.5)
    assert [s.step() for _ in range(6)] == [0, 1, 0, 1, 0, 1]
    a = ReplayScheduler(0.78)
    first = [a.step() for _ in range(37)]
    b = ReplayScheduler(0.78, credit=a.credit)
    rest = [b.step() for _ in range(63)]
    assert sum(first) + sum(rest) == 78


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_replay_ratio_rejects(bad):
    with pytest.raises(ValueError):
        ReplayScheduler(bad)


def test_active_clock_skips_freeze():
    t = [0.0]
    c = ActiveClock(active_s=10.0, max_gap_s=60, now=lambda: t[0])
    for dt in (1.0, 2.0, 3600.0, 4.0):
        t[0] += dt
        c.tick()
    assert c.active_s == pytest.approx(17.0)
    assert c.skipped_s == pytest.approx(3600.0)


def test_atomic_save(tmp_path):
    import torch
    p = tmp_path / 'x.pt'
    atomic_save({'a': 1}, p)
    assert torch.load(p) == {'a': 1}
    assert not (tmp_path / 'x.pt.tmp').exists()


def test_main_replay_ratio_knob(tmp_path):
    import json, subprocess
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / 'r.csv'
    r = subprocess.run([sys.executable, 'main.py', 'train', '--from-scratch', '--episodes', '20',
                        '--seed', '3', '--replay-ratio', '0.78', '--save-as', str(tmp_path / 'm'),
                        '--log', str(log)], cwd=repo, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    s = json.loads(log.with_suffix('.json').read_text())
    assert s['replay_ratio_requested'] == 0.78
    # updates start once the buffer holds batch_size (128) transitions
    expected = int(0.78 * (s['env_steps'] - 127) + 1e-9)
    assert abs(s['updates'] - expected) <= 1
    assert s['replay_ratio_actual'] == pytest.approx(s['updates'] / s['env_steps'])
