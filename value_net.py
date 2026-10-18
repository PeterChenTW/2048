"""Scalar afterstate value network V(x) for 2048 (same trunk as the Q baseline, scalar head)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dueling import OneHotDuelingDQN
from encoding import log2_onehot_torch


class _L1SkipDirectionalTrunk(OneHotDuelingDQN):
    """Directional V trunk with post-ReLU L1 features appended after L2."""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        onehot = log2_onehot_torch(x)
        onehot = onehot.to(device=self.conv_h.weight.device, dtype=self.conv_h.weight.dtype)
        h = F.relu(self.conv_h(onehot))
        v = F.relu(self.conv_v(onehot))
        hh = F.relu(self.conv_hh(h)).flatten(start_dim=1)
        hv = F.relu(self.conv_hv(h)).flatten(start_dim=1)
        vh = F.relu(self.conv_vh(v)).flatten(start_dim=1)
        vv = F.relu(self.conv_vv(v)).flatten(start_dim=1)
        return torch.cat([hh, hv, vh, vv, h.flatten(1), v.flatten(1)], dim=1)


class OneHotValueNet(nn.Module):
    """V(afterstate) network: shared conv trunk + Linear-ReLU-Linear(1) head.

    ``trunk='directional'`` composes the exact trunk of :class:`dueling.OneHotDuelingDQN`
    (its Q head is replaced by ``nn.Identity`` so no unused parameters remain).

    EXTENSION POINT: add new trunks in ``_build_trunk``. A trunk must be an ``nn.Module``
    exposing ``forward_features(raw_boards) -> (N, F)`` and the int attribute
    ``trunk_features`` set here. ``trunk='conv2x2'`` (experiment #4) is not yet implemented.
    """

    architecture: str = "onehot_afterstate_v1"

    def __init__(self, channels: int = 128, hidden: int = 256, trunk: str = "directional",
                 value_skip: str = "none") -> None:
        super().__init__()
        if value_skip not in ("none", "l1"):
            raise ValueError(f"unknown value_skip: {value_skip!r}")
        if value_skip == "l1" and trunk != "directional":
            raise ValueError("value_skip='l1' requires trunk='directional'")
        self.value_skip = value_skip
        if value_skip == "l1":
            self.architecture = "onehot_afterstate_skip_v1"
        self.channels = channels
        self.hidden = hidden
        self.trunk_name = trunk
        self.trunk, trunk_features = self._build_trunk(trunk, channels, hidden, value_skip)
        self.trunk_features = trunk_features

        self.head = nn.Sequential(
            nn.Linear(trunk_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    @staticmethod
    def _build_trunk(trunk: str, channels: int, hidden: int,
                     value_skip: str = "none") -> tuple[nn.Module, int]:
        if trunk == "directional":
            # The none path retains the original modules, initialization order and RNG use.
            network = OneHotDuelingDQN if value_skip == "none" else _L1SkipDirectionalTrunk
            net = network(channels=channels, hidden=hidden)
            features = net.head.in_features
            if value_skip == "l1":
                features += 24 * channels
            net.head = nn.Identity()  # drop the unused Q head (keeps trunk param names intact)
            return net, features
        if trunk == "conv2x2":
            # TODO(experiment #4): connect the 2x2-conv trunk from the other branch here.
            raise NotImplementedError("trunk='conv2x2' is not implemented yet (experiment #4).")
        raise ValueError(f"Unknown trunk {trunk!r}; expected 'directional' or 'conv2x2'.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Raw tile values (N,16)/(N,4,4)/(16,) -> flattened trunk features (N, F)."""
        return self.trunk.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return V of shape (N,) float32 (N=1 for a single 16-vector board)."""
        return self.head(self.forward_features(x)).squeeze(-1)
