"""Small end-to-end evaluations, including the real pretrained policy."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import DEFAULT_CHECKPOINT, evaluate, main


def assert_sane(result, games, max_steps):
    assert result['games'] == games
    assert len(result['per_game']) == games
    assert result['elapsed_seconds'] >= 0
    assert 0 <= result['mean'] <= result['max']
    assert 0 <= result['median'] <= result['max']
    assert sum(result['max_tile_distribution'].values()) == games
    assert all(0 <= rate <= 1 for rate in result['reach_rates'].values())
    assert list(result['reach_rates'].values()) == sorted(result['reach_rates'].values(), reverse=True)
    for i, record in enumerate(result['per_game']):
        assert record['seed'] == 1000 + i
        assert record['score'] >= 0
        assert 0 < record['steps'] <= max_steps
        assert record['max_tile'] >= 2
        assert record['max_tile'] & (record['max_tile'] - 1) == 0


@pytest.mark.parametrize('board', ['old', 'fast'])
def test_random(board):
    result = evaluate(policy='random', games=3, board=board)
    assert_sane(result, 3, 5000)
    repeated = evaluate(policy='random', games=3, board=board)
    assert result['per_game'] == repeated['per_game']


def test_greedy_fast():
    pytest.importorskip('fast_board')
    checkpoint = Path(__file__).resolve().parents[1] / DEFAULT_CHECKPOINT
    result = evaluate(checkpoint=checkpoint, games=2, board='fast')
    assert_sane(result, 2, 5000)


def test_json_and_step_limit(tmp_path, capsys):
    import json
    output = tmp_path / 'evaluation.json'
    result = main(['--policy', 'random', '--games', '1', '--max-steps', '1', '--json', str(output)])
    assert result['per_game'][0]['steps'] == 1
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out)['games'] == 1
