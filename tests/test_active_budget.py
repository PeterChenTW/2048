"""Tests for active-time budget clock in main.py train."""
import csv
import json
import math
from pathlib import Path
import time

import pytest

import main
import step4_utils


def test_cli_flags_parsing_and_defaults():
    p = main.parser()

    # Defaults
    args = p.parse_args(['train'])
    assert args.budget_clock == 'wall'
    assert args.active_max_gap_s == 60.0

    # Custom valid values
    args = p.parse_args(['train', '--budget-clock', 'active', '--active-max-gap-s', '12.5'])
    assert args.budget_clock == 'active'
    assert args.active_max_gap_s == 12.5

    args = p.parse_args(['train', '--budget-clock', 'wall', '--active-max-gap-s', '0.5'])
    assert args.budget_clock == 'wall'
    assert args.active_max_gap_s == 0.5


@pytest.mark.parametrize('bad_clock', ['invalid', 'cpu', '', 'ACTIVE', 'Wall'])
def test_cli_rejects_bad_budget_clock(bad_clock):
    p = main.parser()
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--budget-clock', bad_clock])


@pytest.mark.parametrize('bad_gap', ['0', '-1', '-0.01', 'nan', 'inf', '-inf', 'abc'])
def test_cli_rejects_bad_active_max_gap(bad_gap):
    p = main.parser()
    with pytest.raises(SystemExit):
        p.parse_args(['train', '--active-max-gap-s', bad_gap])


def test_default_wall_mode_preserves_csv_and_json_keys(tmp_path):
    checkpoint = tmp_path / 'wall_test.pt'
    log = tmp_path / 'wall_test.csv'
    ret = main.main(['train', '--from-scratch', '--episodes', '3', '--seed', '42',
                     '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0

    # CSV headers must match standard FIELDS exactly
    with log.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == main.FIELDS

    # JSON summary must not contain budget_clock, active_s, or skipped_s
    summary = json.loads(log.with_suffix('.json').read_text())
    assert 'budget_clock' not in summary
    assert 'active_s' not in summary
    assert 'skipped_s' not in summary
    assert summary['episodes'] == 3


def test_explicit_wall_mode_preserves_csv_and_json_keys(tmp_path):
    checkpoint = tmp_path / 'wall_explicit.pt'
    log = tmp_path / 'wall_explicit.csv'
    ret = main.main(['train', '--from-scratch', '--budget-clock', 'wall',
                     '--episodes', '3', '--seed', '42',
                     '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0

    with log.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == main.FIELDS

    summary = json.loads(log.with_suffix('.json').read_text())
    assert 'budget_clock' not in summary
    assert 'active_s' not in summary
    assert 'skipped_s' not in summary
    assert summary['episodes'] == 3


def test_active_clock_without_minutes_runs_episodes_and_records_stats(tmp_path):
    checkpoint = tmp_path / 'active_no_minutes.pt'
    log = tmp_path / 'active_no_minutes.csv'
    ret = main.main(['train', '--from-scratch', '--budget-clock', 'active',
                     '--episodes', '3', '--seed', '42',
                     '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0

    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['budget_clock'] == 'active'
    assert 'active_s' in summary
    assert 'skipped_s' in summary
    assert summary['episodes'] == 3


def test_active_clock_simulated_freeze_continues_past_wall_deadline(tmp_path, monkeypatch):
    """Proves (a): simulated 10-minute freeze gap is not counted so training continues past wall deadline."""
    current_time = [1000.0]
    call_count = [0]
    freeze_injected = [False]

    def fake_monotonic():
        call_count[0] += 1
        # Inject a 10-minute (600s) freeze jump after a few ticks
        if call_count[0] == 15 and not freeze_injected[0]:
            current_time[0] += 600.0
            freeze_injected[0] = True
        else:
            current_time[0] += 0.5
        return current_time[0]

    monkeypatch.setattr(time, 'monotonic', fake_monotonic)
    monkeypatch.setattr(main.time, 'monotonic', fake_monotonic)
    monkeypatch.setattr(step4_utils.time, 'monotonic', fake_monotonic)

    checkpoint = tmp_path / 'freeze_test.pt'
    log = tmp_path / 'freeze_test.csv'

    # Budget is 0.1 minutes = 6.0 seconds.
    # Without active clock, at call 15 time jumps by 600s, exceeding the 6s wall deadline immediately.
    # With active clock, the 600s gap is skipped (> max_gap_s 60.0), so training continues
    # until active_s reaches 6.0s!
    ret = main.main(['train', '--from-scratch', '--budget-clock', 'active',
                     '--active-max-gap-s', '60.0', '--minutes', '0.1',
                     '--seed', '7', '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0
    assert freeze_injected[0], "Freeze was not injected"

    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['budget_clock'] == 'active'
    assert summary['skipped_s'] >= 600.0
    assert summary['active_s'] >= 6.0
    # Wall clock duration must reflect the 600s freeze
    assert summary['duration_s'] >= 600.0
    # Training continued past the freeze, so env_steps must be greater than steps at freeze
    assert summary['env_steps'] > 5


def test_active_clock_normal_ticks_stop_at_budget(tmp_path, monkeypatch):
    """Proves (b): normal ticks stop at the budget."""
    current_time = [500.0]

    def fake_monotonic():
        current_time[0] += 0.2
        return current_time[0]

    monkeypatch.setattr(time, 'monotonic', fake_monotonic)
    monkeypatch.setattr(main.time, 'monotonic', fake_monotonic)
    monkeypatch.setattr(step4_utils.time, 'monotonic', fake_monotonic)

    checkpoint = tmp_path / 'normal_ticks.pt'
    log = tmp_path / 'normal_ticks.csv'

    # Budget = 0.05 minutes = 3.0 seconds
    ret = main.main(['train', '--from-scratch', '--budget-clock', 'active',
                     '--minutes', '0.05', '--seed', '13',
                     '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0

    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['budget_clock'] == 'active'
    assert summary['skipped_s'] == 0.0
    assert summary['active_s'] >= 3.0
    # Stops shortly after reaching budget (e.g. within 1-2 ticks)
    assert summary['active_s'] < 5.0


def test_real_tiny_run_active_clock(tmp_path):
    """Real tiny run with --budget-clock active --minutes 0.05 terminates and writes active_s."""
    checkpoint = tmp_path / 'real_tiny.pt'
    log = tmp_path / 'real_tiny.csv'

    ret = main.main(['train', '--from-scratch', '--arch', 'onehot_dueling',
                     '--budget-clock', 'active', '--minutes', '0.05',
                     '--threads', '1', '--seed', '101',
                     '--save-as', str(checkpoint), '--log', str(log)])
    assert ret == 0
    assert checkpoint.is_file()
    assert log.is_file()

    summary = json.loads(log.with_suffix('.json').read_text())
    assert summary['budget_clock'] == 'active'
    assert 'active_s' in summary
    assert 'skipped_s' in summary
    assert summary['active_s'] >= 3.0  # 0.05 minutes = 3.0 seconds
    assert summary['skipped_s'] == 0.0
    assert summary['nan_count'] == 0
