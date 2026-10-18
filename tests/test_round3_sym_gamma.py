"""Round 3: D4 replay symmetry augmentation, raw-score reward, gamma=1, value rescaling h."""
import random

import numpy as np
import pytest
import torch

from afterstate import afterstates
from afterstate_agent import AfterstateAgent
from afterstate_policy import afterstate_scores
from config import DQNConfig
from fast_board import FastBoard
import main
from RL import DDQNAgent, load_checkpoint
from rewards import merge_rewards, shape_reward
import symmetry as S
from value_rescale import h, h_inv


def played_boards(n=60, seed=7):
    """Realistic boards from a seeded random policy (both early and crowded)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = FastBoard(rng=random.Random(rng.randrange(10**6)))
        while not b.done:
            inv = b.get_invalid_moves()
            if len(inv) == 4:
                break
            out.append(np.array(b.grid).reshape(16).astype(np.int64))
            b.move(b.all_moves[rng.choice([a for a in range(4) if a not in inv])])
    return out[::max(1, len(out) // n)][:n]


BOARDS = played_boards()


# ---------------- symmetry ----------------

def test_d4_is_a_group_of_order_8():
    perms = {tuple(p) for p in S.PERMS}
    assert len(perms) == 8 and tuple(range(16)) == tuple(S.PERMS[0])
    for a in S.PERMS:
        assert any(tuple(a[b]) == tuple(range(16)) for b in S.PERMS)  # inverse exists
        for b in S.PERMS:
            assert tuple(a[b]) in perms  # closure
    assert any(tuple(a[b]) != tuple(b[a]) for a in S.PERMS for b in S.PERMS)  # non-abelian (D4)
    assert sorted(len({tuple(np.arange(16)[_pow(p, k)]) for k in range(1, 5)}) for p in S.PERMS) == \
        [1, 2, 2, 2, 2, 2, 4, 4]  # element orders of D4: e, 5 involutions, 2 rotations of order 4


def _pow(p, k):
    q = np.arange(16)
    for _ in range(k):
        q = q[p]
    return q


def test_action_maps_are_permutations_and_identity():
    assert S.ACTION_MAPS[0].tolist() == [0, 1, 2, 3]
    for m in S.ACTION_MAPS:
        assert sorted(m.tolist()) == [0, 1, 2, 3]


@pytest.mark.parametrize('g', range(8))
def test_moves_commute_with_symmetry_legality_and_merge_score(g):
    for s in BOARDS:
        after, merge, legal = afterstates(s)
        gafter, gmerge, glegal = afterstates(S.apply_np(s, g))
        for a in range(4):
            b = S.ACTION_MAPS[g][a]
            assert glegal[b] == legal[a]
            assert gmerge[b] == merge[a]
            if legal[a]:
                assert np.array_equal(gafter[b], S.apply_np(after[a], g))


def test_apply_batch_matches_numpy_and_preserves_tiles():
    x = torch.as_tensor(np.stack(BOARDS[:16]), dtype=torch.float32)
    ids = np.arange(16) % 8
    y = S.apply_batch(x, ids)
    for i in range(16):
        assert np.array_equal(y[i].numpy(), S.apply_np(x[i].numpy(), ids[i]))
        assert sorted(y[i].tolist()) == sorted(x[i].tolist())


class SymValue(torch.nn.Module):
    """A D4-invariant toy value: sum of log tiles + 0.1 * corner-agnostic pair term."""
    def forward(self, x):
        x = x.reshape(-1, 16).float()
        lg = torch.log2(x.clamp(min=1))
        return lg.sum(1) + 0.1 * (lg ** 2).sum(1)


@pytest.mark.parametrize('rescale', ['none', 'h'])
def test_scores_invariant_under_symmetry(rescale):
    opts = dict(gamma=0.99, reward_mode='log') if rescale == 'none' else \
        dict(gamma=1.0, reward_mode='raw', value_rescale='h', value_rescale_eps=1e-3)
    m = SymValue()
    for s in BOARDS[:20]:
        base = afterstate_scores(m, s, **opts)
        for g in range(8):
            r = afterstate_scores(m, S.apply_np(s, g), **opts)
            for a in range(4):
                b = S.ACTION_MAPS[g][a]
                assert r.legal[0, b] == base.legal[0, a]
                if base.legal[0, a]:
                    assert torch.isclose(r.scores[0, b], base.scores[0, a], rtol=0, atol=1e-5)


def _filled_agent(**kw):
    cfg = DQNConfig(arch='onehot_afterstate', seed=11, batch_size=16, **kw)
    agent = AfterstateAgent(16, 4, cfg)
    for s in BOARDS[:40]:
        after, _, legal = afterstates(s)
        a = int(np.flatnonzero(legal)[0])
        agent.remember(after[a], s, False)
    return agent


def test_targets_invariant_with_symmetric_net():
    agent = _filled_agent()
    agent.model = SymValue(); agent.target_model = SymValue()
    nxt = torch.as_tensor(np.stack(BOARDS[:24]), dtype=torch.float32)
    dones = torch.zeros(24, dtype=torch.bool)
    base = agent._targets(nxt, dones)
    for g in range(8):
        t = agent._targets(S.apply_batch(nxt, np.full(24, g)), dones)
        assert torch.allclose(t, base, atol=1e-5)


def test_sym_off_has_no_extra_rng_and_on_keeps_main_rng_stream():
    off, on = _filled_agent(), _filled_agent(sym_aug='random')
    assert off.sym_rng is None and on.sym_rng is not None
    on.model.load_state_dict(off.model.state_dict()); on.target_model.load_state_dict(off.target_model.state_dict())
    off.replay(); on.replay()
    assert off.rng.bit_generator.state == on.rng.bit_generator.state  # PER sampling stream unchanged


def test_sym_on_replay_finite_and_actually_transforms(monkeypatch):
    agent = _filled_agent(sym_aug='random')
    seen = []
    orig = S.apply_batch
    monkeypatch.setattr(S, 'apply_batch', lambda b, ids: (seen.append(np.asarray(ids).copy()), orig(b, ids))[1])
    for _ in range(5):
        st = agent.replay()
        assert np.isfinite(st['loss'])
    ids = np.concatenate(seen)
    assert len(seen) == 10 and len(set(ids.tolist())) > 1  # states and next_states, varied elements


# ---------------- raw reward, gamma=1, rescaling ----------------

def test_h_and_inverse_are_inverse_and_odd():
    x = torch.cat([torch.linspace(-1e6, 1e6, 20001, dtype=torch.float64),
                   torch.logspace(-6, 6, 500, dtype=torch.float64)])
    for eps in (1e-3, 1e-2):
        assert torch.allclose(h_inv(h(x, eps), eps), x, rtol=1e-9, atol=1e-8)
        assert torch.allclose(h(h_inv(x / 1e3, eps), eps), x / 1e3, rtol=1e-9, atol=1e-8)
        assert torch.equal(h(-x, eps), -h(x, eps))
    assert h(torch.tensor(0.0)).item() == 0.0 and h_inv(torch.tensor(0.0)).item() == 0.0
    assert torch.all(torch.diff(h(torch.sort(x).values)) >= 0)
    # compresses a 2048-scale return to a small range
    assert 190 < h(torch.tensor(30000.0)).item() < 210  # sqrt(30001)-1 + 30


def test_raw_reward_mode():
    assert merge_rewards(np.array([0, 4, 2048]), 'raw').tolist() == [0.0, 4.0, 2048.0]
    assert shape_reward(36, True, mode='raw') == 36.0 and shape_reward(36, False, mode='raw') == -1.0


def test_afterstate_scores_apply_h_inverse():
    m = SymValue(); s = BOARDS[30]
    after, merge, legal = afterstates(s)
    r = afterstate_scores(m, s, gamma=1.0, reward_mode='raw', value_rescale='h', value_rescale_eps=1e-3)
    for a in np.flatnonzero(legal):
        v = m(torch.as_tensor(after[a], dtype=torch.float32))[0]
        assert torch.isclose(r.scores[0, a], merge[a] + h_inv(v, 1e-3), atol=1e-3)
    with pytest.raises(ValueError):
        afterstate_scores(m, s, value_rescale='bogus')


def test_gamma1_h_targets_and_terminal_truncation():
    agent = _filled_agent(gamma=1.0, reward_mode='raw', value_rescale='h')
    terminal = np.array([2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2])
    assert not afterstates(terminal)[2].any()
    nxt = torch.as_tensor(np.stack([BOARDS[5], BOARDS[50], BOARDS[50], terminal]), dtype=torch.float32)
    dones = torch.tensor([False, False, True, False])
    t = agent._targets(nxt, dones)
    assert t[2].item() == 0.0  # done: no bootstrap past game over even at gamma=1
    assert t[3].item() == 0.0  # no legal move: terminal, value 0
    for i in (0, 1):
        res = afterstate_scores(agent.model, nxt[i], **agent._reward_options())
        a = int(res.scores[0].argmax())
        v = agent.target_model(torch.as_tensor(res.after[0, a], dtype=torch.float32).reshape(1, 16))[0]
        expect = h(torch.tensor(float(res.rewards[0, a])) + 1.0 * h_inv(v.detach(), 1e-3), 1e-3)
        assert torch.isclose(t[i], expect, atol=1e-4)
    assert np.isfinite(agent.replay()['loss'])


def test_validation_and_defaults():
    cfg = DQNConfig()
    assert (cfg.sym_aug, cfg.value_rescale, cfg.value_rescale_eps) == ('none', 'none', 1e-3)
    AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', gamma=1.0))
    for bad in (dict(sym_aug='all'), dict(value_rescale='x'), dict(value_rescale_eps=0.0), dict(gamma=1.5)):
        with pytest.raises(ValueError):
            AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', **bad))
    with pytest.raises(ValueError):
        DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling', sym_aug='random'))
    with pytest.raises(ValueError):
        DDQNAgent(16, 4, DQNConfig(arch='onehot_dueling', value_rescale='h'))


def test_checkpoint_config_keys(tmp_path):
    off = AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate'))
    off.save(tmp_path / 'off.pt')
    cfg_off = torch.load(tmp_path / 'off.pt', weights_only=True)['config']
    assert not {'sym_aug', 'value_rescale', 'value_rescale_eps'} & set(cfg_off)
    on = AfterstateAgent(16, 4, DQNConfig(arch='onehot_afterstate', gamma=1.0, reward_mode='raw',
                                          value_rescale='h', sym_aug='random'))
    on.save(tmp_path / 'on.pt')
    back = load_checkpoint(tmp_path / 'on.pt')
    assert (back.config.value_rescale, back.config.sym_aug, back.config.gamma) == ('h', 'random', 1.0)
    assert back._reward_options()['value_rescale'] == 'h'
    s = BOARDS[10]
    assert back.greedy_action(s.reshape(4, 4), [a for a in range(4) if not afterstates(s)[2][a]]) in range(4)


def test_cli_flags_parse():
    p = main.build_parser() if hasattr(main, 'build_parser') else main.parser()
    a = p.parse_args(['train', '--arch', 'onehot_afterstate', '--sym-aug', 'random', '--reward-mode', 'raw',
                      '--gamma', '1.0', '--value-rescale', 'h', '--value-rescale-eps', '0.01'])
    assert (a.sym_aug, a.reward_mode, a.gamma, a.value_rescale, a.value_rescale_eps) == ('random', 'raw', 1.0, 'h', 0.01)
    d = p.parse_args(['train'])
    assert (d.sym_aug, d.value_rescale, d.value_rescale_eps) == ('none', 'none', None)
