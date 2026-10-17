"""Restart curriculum isolation, collection rules, and real training integration."""
import csv
import io
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from config import DQNConfig
from fast_board import FastBoard
from restart_pool import RestartPool, collect_episode


def grid(tile, marker=0):
    return [[tile, marker, 0, 0], [0] * 4, [0] * 4, [0] * 4]


def test_ring_replacement_and_defensive_copies():
    pool = RestartPool(2, random.Random(17))
    original = grid(256)
    pool.add(original)
    original[0][0] = 4096
    assert pool.sample() == grid(256)
    sampled = pool.sample()
    sampled[0][0] = 8192
    assert pool.sample() == grid(256)
    pool.add(grid(512))
    pool.add(grid(1024))
    assert len(pool) == 2
    assert {pool.sample()[0][0] for _ in range(100)} == {512, 1024}
    pool.add(grid(2048))
    assert {pool.sample()[0][0] for _ in range(100)} == {1024, 2048}


def test_empty_pool_and_invalid_capacity():
    rng = random.Random(5)
    before = rng.getstate()
    pool = RestartPool(5, rng)
    assert len(pool) == 0
    assert pool.sample() is None
    assert rng.getstate() == before
    for capacity in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            RestartPool(capacity, rng)


def test_collection_crossings_and_midpoint():
    tiles = [128, 256, 512, 512, 512, 512, 1024, 1024, 1024]
    trajectory = [grid(tile, 2 ** i) for i, tile in enumerate(tiles)]
    selected = collect_episode(trajectory)
    assert selected == [trajectory[2], trajectory[6], trajectory[5]]
    selected[0][0][0] = 8192
    assert trajectory[2][0][0] == 512
    assert collect_episode([]) == []
    assert collect_episode([grid(2), grid(256)]) == []


def test_collection_does_not_repeat_starting_milestones():
    trajectory = [grid(512), grid(512, 2), grid(1024, 4), grid(1024, 8)]
    assert collect_episode(trajectory) == [trajectory[2], trajectory[1]]
    trajectory = [grid(1024), grid(1024, 2), grid(1024, 4)]
    assert collect_episode(trajectory) == [trajectory[1]]
    trajectory = [grid(256), grid(512)]
    assert collect_episode(trajectory) == [trajectory[1], trajectory[1]]


def test_from_grid_does_not_consume_rng():
    rng = random.Random(42)
    before = rng.getstate()
    source = grid(512, 512)
    board = FastBoard.from_grid(source, rng=rng)
    assert rng.getstate() == before
    assert board.grid == source
    assert board.score == board.score_v2 == board.turns == 0
    assert board.rng is rng
    source[0][0] = 0
    assert board.grid[0][0] == 512


def read_training(log):
    with log.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return rows, json.loads(log.with_suffix('.json').read_text())


# Frozen from the unmodified training driver at 0f541e0, seed=5, episodes=3.
# Wall-clock columns are deliberately excluded; every other CSV value is checked.
BASELINE_CSV = '''episode,score_v2,max_tile,steps,epsilon,beta,loss,q_mean,q_min,q_max,td_abs_mean,is_weight_mean,nan_count
1,576,64,81,0.15,0.4,,,,,,,0
2,1164,128,119,0.1471755052296599,0.400108,0.038907810972121204,0.30054710001537677,-0.6952866315841675,2.9516255855560303,0.20466995317684977,0.43206616004587256,0
3,512,32,78,0.14440419559737094,0.400222,0.008891003672033548,0.265921278219474,0.008049093186855316,1.0141748189926147,0.11819660898886229,0.46014588080519286,0
'''
BASELINE_KEYS = {
    'arch', 'n_step', 'episodes', 'epsilon_schedule', 'epsilon_start', 'epsilon_min',
    'epsilon_decay_steps', 'env_steps', 'updates', 'replay_ratio_requested',
    'replay_ratio_actual', 'duration_s', 'score_last_100_mean', 'loss_first_50_mean',
    'loss_last_50_mean', 'q_stats_ranges', 'nan_count', 'stopped_reason', 'interrupted',
    'checkpoint', 'loaded_checkpoint',
}


@pytest.mark.parametrize('options', [[], ['--restart-frac', '0', '--restart-pool-size', '1']])
def test_off_matches_original_seeded_training(tmp_path, monkeypatch, options):
    import restart_pool

    def forbidden(*args, **kwargs):
        pytest.fail('disabled restart curriculum must not initialize or collect the pool')

    monkeypatch.setattr(restart_pool, 'RestartPool', forbidden)
    monkeypatch.setattr(restart_pool, 'collect_episode', forbidden)
    log = tmp_path / 'off.csv'
    assert main.main(['train', '--from-scratch', '--episodes', '3', '--seed', '5',
                      '--threads', '1', '--save-as', str(tmp_path / 'off.pt'),
                      '--log', str(log), *options]) == 0
    rows, summary = read_training(log)
    # Other experiment lanes add summary metadata; retain every original key.
    assert BASELINE_KEYS <= set(summary)
    assert set(rows[0]) == set(main.FIELDS)
    expected = list(csv.DictReader(io.StringIO(BASELINE_CSV)))
    for actual, reference in zip(rows, expected, strict=True):
        actual.pop('elapsed_s')
        assert actual == reference
    assert summary['env_steps'] == 278
    assert summary['updates'] == 38
    assert summary['nan_count'] == 0


@pytest.mark.parametrize('source', ['legal', 'empty', 'terminal'])
def test_forced_restart_or_fallback(tmp_path, monkeypatch, source):
    import RL
    import restart_pool

    starting_grid = grid(512, 512)
    terminal_grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    real_pool = RestartPool
    pools = []

    def prefilled_pool(capacity, rng):
        pool = real_pool(capacity, rng)
        if source != 'empty':
            pool.add(starting_grid if source == 'legal' else terminal_grid)
        pools.append(pool)
        return pool

    monkeypatch.setattr(restart_pool, 'RestartPool', prefilled_pool)
    initial_states = []
    moved_states = []
    original_move = FastBoard.move

    def legal_action(self, state, invalid):
        if len(moved_states) % 3 == 0:
            initial_states.append(state.numpy().copy())
        return next(action for action in range(4) if action not in invalid)

    def three_moves(self, direction):
        assert self.pure_move(direction)[2]
        result = original_move(self, direction)
        moved_states.append(self.grid)
        # Bound real board episodes to three legal moves, without a clock dependency.
        return result[0], result[1], self.turns == 3

    monkeypatch.setattr(RL.DDQNAgent, 'act', legal_action)
    monkeypatch.setattr(FastBoard, 'move', three_moves)
    log = tmp_path / 'enabled.csv'
    assert main.main(['train', '--from-scratch', '--episodes', '2', '--seed', '5',
                      '--threads', '1', '--restart-frac', '1', '--restart-pool-size', '2',
                      '--save-as', str(tmp_path / 'enabled.pt'), '--log', str(log)]) == 0
    rows, summary = read_training(log)
    assert len(initial_states) == 2
    assert len(moved_states) == summary['env_steps'] == 6
    assert summary['fresh_env_steps'] + summary['restart_env_steps'] == 6
    assert summary['pool_size_final'] == len(pools[0])
    assert all(int(row['steps']) == 3 for row in rows)
    if source == 'legal':
        assert np.array_equal(initial_states[0], starting_grid)
        assert [row['start'] for row in rows] == ['restart', 'restart']
        assert int(rows[0]['start_max_tile']) == 512
        assert int(rows[0]['score_v2']) == 1024  # Only the post-restart merge.
        assert summary['restart_episodes'] == 2
        assert summary['fresh_episodes'] == summary['fresh_env_steps'] == 0
        assert summary['restart_env_steps'] == 6
        assert summary['late_stage_fraction'] == 1.0
        assert summary['pool_adds'] == 3  # First 1024 + midpoint, then midpoint.
        assert summary['score_last_100_fresh_mean'] is None
    else:
        for i, initial in enumerate(initial_states):
            assert np.array_equal(initial, FastBoard(rng=random.Random(5 + i)).state)
        assert [row['start'] for row in rows] == ['fresh', 'fresh']
        assert summary['fresh_episodes'] == 2
        assert summary['restart_episodes'] == summary['restart_env_steps'] == 0
        assert summary['fresh_env_steps'] == 6
        assert summary['late_stage_fraction'] == summary['pool_adds'] == 0
        assert summary['score_last_100_fresh_mean'] == np.mean([int(r['score_v2']) for r in rows])


@pytest.mark.parametrize('options', [['--restart-frac', '-0.1'], ['--restart-frac', '1.01'],
                                  ['--restart-frac', 'nan'], ['--restart-frac', 'inf'],
                                  ['--restart-pool-size', '0']])
def test_cli_rejects_invalid_restart_options(options):
    with pytest.raises(SystemExit):
        main.parser().parse_args(['train', *options])


def test_config_and_cli_defaults():
    config = DQNConfig()
    args = main.parser().parse_args(['train'])
    assert config.restart_frac == args.restart_frac == 0.0
    assert config.restart_pool_size == args.restart_pool_size == 5000
