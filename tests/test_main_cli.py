"""Exercise the real training command without GUI or plotting dependencies."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / 'model/nn_prioritize_replay_no_invalid_gamma_099_2'


def run_train(tmp_path, *options):
    checkpoint = tmp_path / 'checkpoints' / 'trained.pt'
    log = tmp_path / 'logs' / 'train.csv'
    result = subprocess.run(
        [sys.executable, 'main.py', 'train', '--save-as', str(checkpoint),
         '--log', str(log), '--seed', '0', *options],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['nan_count'] == 0
    assert all(int(row['nan_count']) == 0 for row in rows)
    for row in rows:
        assert all(math.isfinite(float(value)) for value in row.values() if value)
    assert summary['env_steps'] == sum(int(row['steps']) for row in rows)
    if summary['updates']:
        assert math.isfinite(summary['loss_first_50_mean'])
        assert math.isfinite(summary['loss_last_50_mean'])
    assert summary['episodes'] == len(rows)
    assert summary['checkpoint'] == str(checkpoint)
    return result, rows, summary


def test_train_from_scratch(tmp_path):
    result, rows, summary = run_train(
        tmp_path, '--from-scratch', '--episodes', '3',
        '--checkpoint', str(tmp_path / 'must-not-be-loaded.pt'),
    )
    assert len(rows) == 3
    assert summary['updates'] > 0
    assert summary['loaded_checkpoint'] is None
    assert 'Starting from scratch' in result.stdout


def test_default_train_loads_old_checkpoint(tmp_path):
    assert DEFAULT_CHECKPOINT.is_file()
    before = DEFAULT_CHECKPOINT.read_bytes()
    result, rows, summary = run_train(tmp_path, '--episodes', '1')
    assert len(rows) == 1
    assert 'Loaded checkpoint: model/nn_prioritize_replay_no_invalid_gamma_099_2' in result.stdout
    assert summary['loaded_checkpoint'] == 'model/nn_prioritize_replay_no_invalid_gamma_099_2'
    assert DEFAULT_CHECKPOINT.read_bytes() == before


def test_help_without_optional_dependencies():
    result = subprocess.run([sys.executable, 'main.py', '--help'], cwd=REPO,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'train' in result.stdout and 'play' in result.stdout


@pytest.mark.parametrize('failure', ['returned_nan', 'floating_point', 'gradient_norm', 'nan_reward'])
def test_nonfinite_replay_is_logged_and_stops(tmp_path, monkeypatch, capsys, failure):
    import importlib.util
    import config
    import RL

    spec = importlib.util.spec_from_file_location('training_main', REPO / 'main.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original_config = config.DQNConfig
    monkeypatch.setattr(config, 'DQNConfig',
                        lambda **kwargs: original_config(batch_size=1, **kwargs))
    if failure == 'returned_nan':
        monkeypatch.setattr(RL.DDQNAgent, 'replay',
                            lambda self: {'loss': float('nan'), 'nan': True})
    elif failure == 'nan_reward':
        original_remember = RL.DDQNAgent.remember

        def remember_nan(self, state, action, reward, next_state, done, next_invalid):
            return original_remember(self, state, action, float('nan'), next_state, done, next_invalid)

        monkeypatch.setattr(RL.DDQNAgent, 'remember', remember_nan)
    else:
        def raise_nonfinite(self):
            if failure == 'floating_point':
                raise FloatingPointError('nonfinite replay statistics')
            raise RuntimeError('The total norm of order 2.0 for gradients from `parameters` is '
                               'non-finite, so it cannot be clipped.')

        monkeypatch.setattr(RL.DDQNAgent, 'replay', raise_nonfinite)
    log = tmp_path / 'failure.csv'
    checkpoint = tmp_path / 'failure.pt'
    code = entry.main(['train', '--from-scratch', '--episodes', '3',
                       '--replay-every', '1', '--seed', '0',
                       '--save-as', str(checkpoint), '--log', str(log)])
    assert code == 2
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
    assert 'Non-finite replay at episode 1, step 1' in capsys.readouterr().out
    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['nan_count'] == 1
    assert summary['stopped_reason'] == 'nonfinite_replay'
    assert summary['updates'] == (1 if failure == 'returned_nan' else 0)
    assert summary['env_steps'] == 1
    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]['nan_count'] == '1'
    assert rows[0]['loss'] == ''
