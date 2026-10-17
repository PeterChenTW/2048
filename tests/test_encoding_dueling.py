"""Unit tests for encoding.py and dueling.py."""

from __future__ import annotations

import io
import math
import numpy as np
import pytest
import torch

from dueling import DuelingHead, OneHotDuelingDQN
from encoding import NUM_CHANNELS, log2_onehot, log2_onehot_torch


def test_num_channels_constant() -> None:
    assert NUM_CHANNELS == 16


def test_encoding_known_boards() -> None:
    # Empty board -> channel 0 all ones, other channels zero
    empty = np.zeros((4, 4), dtype=np.float32)
    enc_empty = log2_onehot(empty)
    assert enc_empty.shape == (1, 16, 4, 4)
    assert enc_empty.dtype == np.float32
    assert np.all(enc_empty[0, 0] == 1.0)
    assert np.all(enc_empty[0, 1:] == 0.0)

    # Specific tiles: 0, 2, 4, 2048, 32768, 65536 (clamped to ch 15), 131072 (clamped to ch 15)
    board = np.zeros((4, 4), dtype=np.float32)
    board[0, 0] = 0
    board[0, 1] = 2        # 2^1 -> channel 1
    board[0, 2] = 4        # 2^2 -> channel 2
    board[0, 3] = 2048     # 2^11 -> channel 11
    board[1, 0] = 32768    # 2^15 -> channel 15
    board[1, 1] = 65536    # 2^16 -> clamped to channel 15
    board[1, 2] = 131072   # 2^17 -> clamped to channel 15

    for fn in (log2_onehot, lambda b: log2_onehot_torch(torch.tensor(b)).numpy()):
        enc = fn(board)
        assert enc.shape == (1, 16, 4, 4)
        assert enc.dtype == np.float32
        # Every cell has exactly one active channel
        assert np.all(enc.sum(axis=1) == 1.0)
        assert enc[0, 0, 0, 0] == 1.0
        assert enc[0, 1, 0, 1] == 1.0
        assert enc[0, 2, 0, 2] == 1.0
        assert enc[0, 11, 0, 3] == 1.0
        assert enc[0, 15, 1, 0] == 1.0
        assert enc[0, 15, 1, 1] == 1.0
        assert enc[0, 15, 1, 2] == 1.0


def test_encoding_all_input_shapes() -> None:
    rng = np.random.default_rng(42)
    exponents = rng.integers(0, 17, size=(5, 4, 4))
    boards = np.where(exponents == 0, 0, 1 << exponents).astype(np.float32)

    single_4x4 = boards[0]
    single_16 = boards[0].reshape(16)
    batch_n16 = boards.reshape(5, 16)
    batch_n44 = boards

    # NumPy shapes
    out_4x4 = log2_onehot(single_4x4)
    out_16 = log2_onehot(single_16)
    out_n16 = log2_onehot(batch_n16)
    out_n44 = log2_onehot(batch_n44)

    assert out_4x4.shape == (1, 16, 4, 4)
    assert out_16.shape == (1, 16, 4, 4)
    assert out_n16.shape == (5, 16, 4, 4)
    assert out_n44.shape == (5, 16, 4, 4)
    assert np.array_equal(out_4x4, out_16)
    assert np.array_equal(out_n16, out_n44)
    assert np.array_equal(out_4x4[0], out_n44[0])

    # Also test N=4 batch of (4, 16) vs single (4, 4) disambiguation
    out_4_16 = log2_onehot(boards[:4].reshape(4, 16))
    assert out_4_16.shape == (4, 16, 4, 4)

    # Torch shapes and float dtypes
    for dtype in (torch.float32, torch.float64):
        t_4x4 = log2_onehot_torch(torch.tensor(single_4x4, dtype=dtype))
        t_16 = log2_onehot_torch(torch.tensor(single_16, dtype=dtype))
        t_n16 = log2_onehot_torch(torch.tensor(batch_n16, dtype=dtype))
        t_n44 = log2_onehot_torch(torch.tensor(batch_n44, dtype=dtype))

        assert t_4x4.shape == (1, 16, 4, 4)
        assert t_16.shape == (1, 16, 4, 4)
        assert t_n16.shape == (5, 16, 4, 4)
        assert t_n44.shape == (5, 16, 4, 4)
        assert t_4x4.dtype == torch.float32
        assert not t_4x4.requires_grad
        assert torch.equal(t_4x4, t_16)
        assert torch.equal(t_n16, t_n44)


def test_numpy_torch_equality_1000_random_boards() -> None:
    rng = np.random.default_rng(12345)
    exponents = rng.integers(0, 18, size=(1000, 4, 4))
    boards = np.where(exponents == 0, 0, 1 << exponents).astype(np.float32)

    np_encoded = log2_onehot(boards)
    torch_encoded = log2_onehot_torch(torch.from_numpy(boards)).numpy()

    assert np_encoded.shape == (1000, 16, 4, 4)
    assert np.array_equal(np_encoded, torch_encoded)


@pytest.mark.parametrize(
    "bad_val",
    [-2.0, -1.0, 1.0, 3.0, 5.0, 6.0, 7.0, 10.0, 12.0, 15.0, 2.5, 0.5, np.nan, np.inf],
)
def test_encoding_invalid_values_raise(bad_val: float) -> None:
    board = np.zeros((4, 4), dtype=np.float32)
    board[2, 1] = bad_val
    with pytest.raises(ValueError):
        log2_onehot(board)
    with pytest.raises(ValueError):
        log2_onehot_torch(torch.tensor(board, dtype=torch.float32))


@pytest.mark.parametrize(
    "bad_shape",
    [(15,), (17,), (3, 4), (4, 5), (0, 16), (2, 3, 4), (1, 1, 4, 4)],
)
def test_encoding_invalid_shapes_raise(bad_shape: tuple[int, ...]) -> None:
    arr = np.zeros(bad_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        log2_onehot(arr)
    with pytest.raises(ValueError):
        log2_onehot_torch(torch.zeros(bad_shape, dtype=torch.float32))


def test_dueling_identity() -> None:
    torch.manual_seed(0)
    head = DuelingHead(in_features=64, n_actions=4, hidden=32)
    x = torch.randn(8, 64)

    v, a = head.forward_streams(x)
    q = head(x)

    # 1. Q - V has zero mean over actions
    diff = q - v
    assert torch.allclose(diff.mean(dim=1), torch.zeros(8), atol=1e-6)

    # 2. Shifting A by an arbitrary per-sample or scalar constant leaves Q unchanged
    const_shift = torch.tensor([[5.0], [-12.3], [0.0], [100.0], [-0.5], [3.14], [7.7], [-42.0]])
    q_shifted_combine = head.combine(v, a + const_shift)
    assert torch.allclose(q, q_shifted_combine, atol=1e-5)

    # Also test shifting the final advantage layer bias in-place
    with torch.no_grad():
        head.advantage_stream[-1].bias.add_(17.5)
    q_after_bias_shift = head(x)
    assert torch.allclose(q, q_after_bias_shift, atol=1e-5)


def test_onehot_dueling_dqn_shapes_and_negative_q() -> None:
    torch.manual_seed(7)
    model = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    assert OneHotDuelingDQN.architecture == "onehot_dueling_v1"
    assert model.architecture == "onehot_dueling_v1"

    board = torch.tensor(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [0, 0, 2, 4],
        ],
        dtype=torch.float32,
    )

    q_44 = model(board)
    q_16 = model(board.reshape(16))
    q_b16 = model(board.reshape(1, 16).repeat(6, 1))
    q_b44 = model(board.unsqueeze(0).repeat(6, 1, 1))

    assert q_44.shape == (1, 4)
    assert q_16.shape == (1, 4)
    assert q_b16.shape == (6, 4)
    assert q_b44.shape == (6, 4)

    # Verify Q can be negative (no ReLU on output)
    assert (q_44 < 0).any(), f"Expected negative Q values, got {q_44}"

    # Also verify with explicitly negative value stream bias that all Q values can be negative
    with torch.no_grad():
        model.head.value_stream[-1].bias.fill_(-100.0)
        q_all_neg = model(board)
    assert torch.all(q_all_neg < -50.0)


def test_gradients_flow_to_all_parameters() -> None:
    torch.manual_seed(99)
    model = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    boards = torch.tensor(
        [
            [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            [2, 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536],
        ],
        dtype=torch.float32,
    )
    q = model(boards)
    # Weighted sum loss with distinct action weights so both V and A receive non-zero gradients
    weights = torch.tensor([[1.0, 2.0, -0.5, 0.25], [-1.0, 0.5, 1.5, -0.25]])
    loss = (q * weights).sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} received no gradient"
        assert torch.any(param.grad != 0), f"Parameter {name} has all-zero gradient"


def test_batch_vs_single_consistency() -> None:
    torch.manual_seed(101)
    model = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    model.eval()

    rng = np.random.default_rng(2024)
    exps = rng.integers(0, 16, size=(8, 4, 4))
    boards = torch.from_numpy(np.where(exps == 0, 0, 1 << exps).astype(np.float32))

    with torch.no_grad():
        batch_q = model(boards)
        single_qs = torch.cat([model(boards[i]) for i in range(8)], dim=0)
        single_flat_qs = torch.cat([model(boards[i].reshape(16)) for i in range(8)], dim=0)

    assert torch.allclose(batch_q, single_qs, atol=1e-5)
    assert torch.allclose(batch_q, single_flat_qs, atol=1e-5)


def test_param_count_bound() -> None:
    model = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    total_params = sum(p.numel() for p in model.parameters())
    assert 50_000 <= total_params <= 1_500_000, f"Unexpected param count: {total_params}"


def test_state_dict_save_load_roundtrip() -> None:
    torch.manual_seed(555)
    model1 = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    model1.eval()

    board = torch.tensor([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 2, 4, 8], dtype=torch.float32)
    with torch.no_grad():
        expected_q = model1(board)

    buf = io.BytesIO()
    torch.save(model1.state_dict(), buf)
    buf.seek(0)

    model2 = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    model2.load_state_dict(torch.load(buf, weights_only=True))
    model2.eval()

    with torch.no_grad():
        loaded_q = model2(board)

    assert torch.equal(expected_q, loaded_q)


def test_dueling_combine_hand_computed() -> None:
    v = torch.tensor([[10.0]])
    a = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    q = DuelingHead.combine(v, a)
    expected = torch.tensor([[7.0, 9.0, 11.0, 13.0]])
    assert torch.allclose(q, expected)
    assert q.shape == (1, 4)


def test_dueling_combine_1d_value_per_sample() -> None:
    v = torch.tensor([10.0, 20.0, 30.0, 40.0])  # (N,) with N=4
    a = torch.zeros((4, 4))
    q = DuelingHead.combine(v, a)

    # Per-sample addition ensures sample i gets v[i] across all actions:
    expected = torch.tensor([
        [10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0, 30.0],
        [40.0, 40.0, 40.0, 40.0],
    ])
    assert torch.allclose(q, expected)
    # Ensure it did NOT do per-action addition (where row 0 would be [10, 20, 30, 40])
    assert not torch.allclose(q[0], v)

    # Check non-zero advantage stream matches 2D value.unsqueeze(-1)
    a_nonzero = torch.tensor([
        [1.0, 3.0, 5.0, 7.0],
        [2.0, 4.0, 6.0, 8.0],
        [-1.0, 0.0, 1.0, 2.0],
        [0.0, 5.0, -5.0, 0.0],
    ])
    q_1d = DuelingHead.combine(v, a_nonzero)
    q_2d = DuelingHead.combine(v.unsqueeze(-1), a_nonzero)
    assert torch.allclose(q_1d, q_2d)


def test_dueling_combine_shape_validation() -> None:
    # Batch dimension mismatch
    with pytest.raises(ValueError):
        DuelingHead.combine(torch.tensor([[1.0], [2.0]]), torch.zeros((3, 4)))

    # 2D value trailing dimension != 1
    with pytest.raises(ValueError):
        DuelingHead.combine(torch.zeros((4, 2)), torch.zeros((4, 4)))

    # 3D value
    with pytest.raises(ValueError):
        DuelingHead.combine(torch.zeros((4, 1, 1)), torch.zeros((4, 4)))

    # Incompatible advantage dimension
    with pytest.raises(ValueError):
        DuelingHead.combine(torch.zeros((4, 1)), torch.zeros(4))


@pytest.mark.parametrize("k", range(1, 16))
def test_tile_maps_to_channel_k(k: int) -> None:
    tile = 2 ** k
    expected_channel = int(math.log2(tile))
    assert expected_channel == k

    board_np = np.zeros((4, 4), dtype=np.int64)
    board_np[1, 2] = tile

    enc_np = log2_onehot(board_np)
    assert enc_np.shape == (1, 16, 4, 4)
    assert enc_np[0, expected_channel, 1, 2] == 1.0
    assert enc_np[0, :, 1, 2].sum() == 1.0
    assert enc_np[0, 0, 0, 0] == 1.0

    board_torch = torch.from_numpy(board_np)
    enc_torch = log2_onehot_torch(board_torch)
    assert enc_torch.shape == (1, 16, 4, 4)
    assert enc_torch[0, expected_channel, 1, 2].item() == 1.0
    assert enc_torch[0, :, 1, 2].sum().item() == 1.0
    assert enc_torch[0, 0, 0, 0].item() == 1.0


def test_onehot_dueling_dqn_int64_input() -> None:
    torch.manual_seed(42)
    model = OneHotDuelingDQN(channels=128, hidden=256, n_actions=4)
    model.eval()

    board_int64 = torch.tensor(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [0, 0, 2, 4],
        ],
        dtype=torch.int64,
    )
    with torch.no_grad():
        out_int = model(board_int64)
        out_float = model(board_int64.to(dtype=torch.float32))

    assert out_int.shape == (1, 4)
    assert out_int.dtype == torch.float32
    assert torch.equal(out_int, out_float)

    # Also test batch dimension with int64
    batch_int64 = board_int64.unsqueeze(0).repeat(4, 1, 1)
    with torch.no_grad():
        out_batch = model(batch_int64)
    assert out_batch.shape == (4, 4)
    assert torch.allclose(out_batch, out_int.repeat(4, 1))
