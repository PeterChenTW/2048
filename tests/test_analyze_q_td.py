"""Post-hoc diagnostics must use the same rewards and masked targets as training."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DQNConfig
from fast_board import FastBoard
from nstep import NStepTransition, double_dqn_targets
from RL import DDQNAgent
from tools.analyze_q_td import analyze_checkpoint, batch_terms, collect_transitions, reward_profile


@pytest.mark.parametrize('mode', ['log', 'linear'])
def test_cli_finite_reports_and_exact_rewards(tmp_path, mode):
    torch.set_num_threads(1)
    config = DQNConfig(arch='onehot_dueling', seed=7, reward_mode=mode,
                       reward_scale=128, memory_capacity=32)
    agent = DDQNAgent(16, 4, config)
    checkpoint = tmp_path / 'checkpoint'
    agent.save(checkpoint)
    output_json, output_md = tmp_path / 'result.json', tmp_path / 'result.md'
    proc = subprocess.run(
        [sys.executable, 'tools/analyze_q_td.py', '--checkpoint', str(checkpoint),
         '--label', mode, '--states', '200', '--batch', '32', '--threads', '1',
         '--json', str(output_json), '--md', str(output_md)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output_md.is_file() and output_json.is_file()
    result = json.loads(output_json.read_text())['results'][mode]
    assert result['states'] == result['env_steps'] == 200
    assert result['optimizer_updates'] == 0
    assert result['grad_batches'] == len(result['grad_norms']) == 20
    for metric in result['metrics'].values():
        assert set(metric) == {'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'mean', 'std'}
        assert all(math.isfinite(value) for value in metric.values())
    for key in ('fraction_abs_td_gt_1', 'fraction_clipped'):
        assert 0 <= result[key] <= 1
    assert result['fraction_clipped'] == np.mean(np.asarray(result['grad_norms']) > 1)
    markdown = output_md.read_text()
    assert f'| {mode} | {mode} |' in markdown
    assert 'not Huber' in markdown and 'BOTH online and target' in markdown

    transitions, episodes, rewards = collect_transitions(agent, states=200)
    assert sum(episode['steps'] for episode in episodes) == len(transitions) == 200
    assert any(t.reward > 0 for t in transitions)
    for transition, reward in zip(transitions, rewards):
        board = FastBoard.from_grid(transition.state)
        _, merge, changed = board.pure_move(transition.action)
        assert changed
        _, env_reward, _ = board.move(transition.action)
        expected = env_reward if mode == 'log' else (merge / config.reward_scale if env_reward != -1 else -1)
        assert transition.reward == reward == expected


def test_handmade_targets_td_and_unclipped_mse_gradient():
    torch.set_num_threads(1)
    model = torch.nn.Linear(16, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(4, 16) / 64)
    transitions = [
        NStepTransition(np.ones(16), 0, 0.25, np.full(16, 2), False,
                        np.array([True, False, True, False]), 0.9),
        NStepTransition(np.zeros(16), 3, -1, np.full(16, 4), True,
                        np.zeros(4, dtype=bool), 0.81),
    ]
    terms = batch_terms(model, transitions)
    q = model(torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32))
    target = double_dqn_targets(
        model, model, [0.25, -1], torch.tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32),
        [False, True], np.stack([t.next_mask for t in transitions]), [0.9, 0.81])
    torch.testing.assert_close(terms['target'], target)
    torch.testing.assert_close(terms['td'], target - q[[0, 1], [0, 3]])
    assert terms['target'][1].item() == -1
    assert not terms['target'].requires_grad
    # Analytic mean-MSE gradient (2 / batch) * (Q - target) * input.
    gradient, = torch.autograd.grad(terms['td'].square().mean(), model.weight)
    expected = torch.zeros_like(gradient)
    expected[0] = -terms['td'][0].detach()
    torch.testing.assert_close(gradient, expected)


def test_nstep_collection_flush_and_analysis_preserve_weights(tmp_path):
    torch.set_num_threads(1)
    agent = DDQNAgent(16, 4, DQNConfig(arch='onehot_conv2x2', n_step=3, gamma=0.9, seed=9))
    transitions, _, _ = collect_transitions(agent, states=12)
    assert len(transitions) == 12
    assert transitions[0].discount == pytest.approx(0.9 ** 3)
    assert transitions[-1].discount == pytest.approx(0.9)
    assert not transitions[-1].done  # State budget truncation bootstraps.
    checkpoint = tmp_path / 'conv'
    agent.save(checkpoint)
    before = checkpoint.read_bytes()
    result = analyze_checkpoint(checkpoint, label='conv', states=12, batch=4, grad_batches=2)
    assert result['status'] == 'ok'
    assert checkpoint.read_bytes() == before


def test_afterstate_is_clearly_skipped(tmp_path):
    torch.set_num_threads(1)
    from afterstate_agent import AfterstateAgent
    agent = AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate'))
    checkpoint = tmp_path / 'afterstate'
    agent.save(checkpoint)
    result = analyze_checkpoint(checkpoint, label='afterstate', states=2, batch=2, grad_batches=1)
    assert result['status'] == 'skipped'
    assert 'V(afterstate)' in result['reason']


@pytest.mark.parametrize('mode', ['log', 'linear'])
def test_reward_profile_matches_training_formulas(mode):
    torch.set_num_threads(1)
    config = DQNConfig(arch='onehot_dueling', seed=3, reward_mode=mode, reward_scale=256,
                       memory_capacity=32)
    agent = DDQNAgent(16, 4, config)
    agent.model.eval()
    prof = reward_profile(agent, games=2, seed=5, max_steps=150)
    assert prof['epsilon'] == 0.0 and prof['mode'] == mode and 0 < prof['steps'] <= 300
    for key in ('own', 'as_log', 'as_linear'):
        st = prof[key]
        assert {'p10', 'p50', 'p90', 'p99', 'mean', 'frac_zero', 'frac_neg'} <= set(st)
        assert all(math.isfinite(v) for v in st.values())
        assert 0 <= st['frac_zero'] <= 1 and st['p10'] <= st['p50'] <= st['p90'] <= st['p99']
    assert prof['own'] == prof['as_log' if mode == 'log' else 'as_linear']
    # zero-merge steps get reward 0 under both formulas, so the zero fraction is formula independent
    assert prof['as_log']['frac_zero'] == prof['as_linear']['frac_zero']
    assert abs(sum(prof['merge_score_fraction'].values()) - 1) < 1e-9
    assert prof['as_log']['frac_zero'] == pytest.approx(prof['merge_score_fraction']['0-0'])


def test_reward_formula_reference_values():
    from rewards import shape_reward
    assert shape_reward(4, True, 'linear', 256) == pytest.approx(0.015625)
    assert shape_reward(4, True, 'log') == pytest.approx(math.log2(5) / 16)  # 0.1451
    assert shape_reward(1024, True, 'linear', 256) == pytest.approx(4.0)
    assert shape_reward(1024, True, 'log') == pytest.approx(math.log2(1025) / 16)  # 0.6251
    assert shape_reward(0, True, 'log') == shape_reward(0, True, 'linear') == 0
    assert shape_reward(8, False, 'log') == shape_reward(8, False, 'linear') == -1.0
