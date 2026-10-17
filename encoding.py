"""Log2 one-hot board state encoding for 2048."""

from __future__ import annotations

import numpy as np
import torch

NUM_CHANNELS: int = 16
_MAX_CLAMP_TILE: int = 1 << (NUM_CHANNELS - 1)  # 2^15 = 32768


def _reshape_numpy_board(arr: np.ndarray) -> np.ndarray:
    """Validate and reshape input board(s) to (N, 4, 4)."""
    if arr.ndim == 1:
        if arr.shape != (16,):
            raise ValueError(f"Invalid 1D board shape {arr.shape}; expected (16,).")
        return arr.reshape(1, 4, 4)
    if arr.ndim == 2:
        if arr.shape == (4, 4):
            return arr.reshape(1, 4, 4)
        if arr.shape[0] >= 1 and arr.shape[1] == 16:
            return arr.reshape(arr.shape[0], 4, 4)
        raise ValueError(f"Invalid 2D board shape {arr.shape}; expected (4, 4) or (N, 16).")
    if arr.ndim == 3:
        if arr.shape[0] >= 1 and arr.shape[1:] == (4, 4):
            return arr
        raise ValueError(f"Invalid 3D board shape {arr.shape}; expected (N, 4, 4).")
    raise ValueError(f"Invalid board ndim {arr.ndim}; expected 1, 2, or 3.")


def _reshape_torch_board(x: torch.Tensor) -> torch.Tensor:
    """Validate and reshape input tensor board(s) to (N, 4, 4)."""
    if x.ndim == 1:
        if x.shape != (16,):
            raise ValueError(f"Invalid 1D board shape {tuple(x.shape)}; expected (16,).")
        return x.reshape(1, 4, 4)
    if x.ndim == 2:
        if x.shape == (4, 4):
            return x.reshape(1, 4, 4)
        if x.shape[0] >= 1 and x.shape[1] == 16:
            return x.reshape(x.shape[0], 4, 4)
        raise ValueError(f"Invalid 2D board shape {tuple(x.shape)}; expected (4, 4) or (N, 16).")
    if x.ndim == 3:
        if x.shape[0] >= 1 and x.shape[1:] == (4, 4):
            return x
        raise ValueError(f"Invalid 3D board shape {tuple(x.shape)}; expected (N, 4, 4).")
    raise ValueError(f"Invalid board ndim {x.ndim}; expected 1, 2, or 3.")


def log2_onehot(board: np.ndarray | list | torch.Tensor) -> np.ndarray:
    """Encode raw 2048 board(s) into a (N, 16, 4, 4) float32 one-hot array.

    Channel 0 encodes empty cells (0).
    Channel k encodes tile 2^k for k=1..15, with tiles >= 2^15 clamped to channel 15.
    Raises ValueError on invalid shapes, negative values, or non-power-of-two nonzero tiles.
    """
    if isinstance(board, torch.Tensor):
        arr = board.detach().cpu().numpy()
    else:
        arr = np.asarray(board)

    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"Unsupported board dtype {arr.dtype}.")

    arr = _reshape_numpy_board(arr)

    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError("Board contains non-finite values.")
        if np.any(arr < 0) or np.any(np.floor(arr) != arr):
            raise ValueError("Board values must be non-negative integers.")
    else:
        if np.any(arr < 0):
            raise ValueError("Board values must be non-negative.")

    int_arr = arr.astype(np.int64, copy=False)
    nonzero = int_arr != 0
    if np.any((int_arr < 2) & nonzero) or np.any(((int_arr & (int_arr - 1)) != 0) & nonzero):
        raise ValueError("Nonzero board values must be powers of two >= 2.")

    clamped = np.where(nonzero, np.minimum(int_arr, _MAX_CLAMP_TILE), 1)
    channels = np.where(nonzero, np.rint(np.log2(clamped)).astype(np.int64), 0)

    out = np.zeros((int_arr.shape[0], NUM_CHANNELS, 4, 4), dtype=np.float32)
    np.put_along_axis(out, channels[:, None, :, :], np.float32(1.0), axis=1)
    return out


@torch.no_grad()
def log2_onehot_torch(x: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch equivalent of log2_onehot returning (N, 16, 4, 4) float32 tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if not (x.is_floating_point() or x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError(f"Unsupported tensor dtype {x.dtype}.")

    x = _reshape_torch_board(x)

    if x.is_floating_point():
        if not torch.all(torch.isfinite(x)):
            raise ValueError("Board tensor contains non-finite values.")
        if torch.any(x < 0) or torch.any(torch.floor(x) != x):
            raise ValueError("Board tensor values must be non-negative integers.")
    else:
        if torch.any(x < 0):
            raise ValueError("Board tensor values must be non-negative.")

    xi = x.to(dtype=torch.int64)
    nonzero = xi != 0
    if torch.any((xi < 2) & nonzero) or torch.any(((xi & (xi - 1)) != 0) & nonzero):
        raise ValueError("Nonzero board tensor values must be powers of two >= 2.")

    clamped = torch.where(nonzero, torch.clamp(xi, max=_MAX_CLAMP_TILE), torch.ones_like(xi))
    channels = torch.where(
        nonzero,
        torch.round(torch.log2(clamped.to(dtype=torch.float32))).to(dtype=torch.int64),
        torch.zeros_like(xi),
    )

    out = torch.zeros((xi.shape[0], NUM_CHANNELS, 4, 4), dtype=torch.float32, device=x.device)
    out.scatter_(1, channels.unsqueeze(1), 1.0)
    return out
