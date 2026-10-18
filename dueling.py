"""Dueling DQN head and one-hot convolutional Dueling DQN architecture for 2048."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_CHANNELS, log2_onehot_torch


class DuelingHead(nn.Module):
    """Dueling network head computing Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)."""

    def __init__(self, in_features: int, n_actions: int = 4, hidden: int = 256) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_actions = n_actions
        self.hidden = hidden

        self.value_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # Keep initial value magnitude smaller than advantage spread so initial Q-values
        # naturally span both negative and positive values while preserving gradient flow.
        with torch.no_grad():
            self.value_stream[-1].weight.mul_(0.1)
            self.value_stream[-1].bias.zero_()
            self.advantage_stream[-1].bias.zero_()

    @staticmethod
    def combine(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Combine state value (N, 1) and action advantages (N, A) into Q-values (N, A)."""
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or advantage.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors (or 1D for value), got value.ndim={value.ndim}, advantage.ndim={advantage.ndim}"
            )
        if value.shape[0] != advantage.shape[0]:
            raise ValueError(
                f"Batch dimension mismatch: value has {value.shape[0]} samples, advantage has {advantage.shape[0]} samples"
            )
        if value.shape[-1] != 1:
            raise ValueError(
                f"Value tensor must have shape (N, 1), got shape {tuple(value.shape)}"
            )
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def forward_streams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (value, advantage) tensors before aggregation."""
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v, a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, a = self.forward_streams(x)
        return self.combine(v, a)


class OneHotDuelingDQN(nn.Module):
    """2048 Dueling DQN with log2 one-hot encoding and directional (1,2)/(2,1) conv trunk."""

    architecture: str = "onehot_dueling_v1"

    def __init__(self, channels: int = 128, hidden: int = 256, n_actions: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.n_actions = n_actions

        c2 = max(1, channels // 2)

        # First directional conv layer on (N, 16, 4, 4)
        self.conv_h = nn.Conv2d(NUM_CHANNELS, channels, kernel_size=(1, 2))  # -> (N, C, 4, 3)
        self.conv_v = nn.Conv2d(NUM_CHANNELS, channels, kernel_size=(2, 1))  # -> (N, C, 3, 4)

        # Second directional conv layer on each branch
        self.conv_hh = nn.Conv2d(channels, c2, kernel_size=(1, 2))  # -> (N, c2, 4, 2) = 8 * c2
        self.conv_hv = nn.Conv2d(channels, c2, kernel_size=(2, 1))  # -> (N, c2, 3, 3) = 9 * c2
        self.conv_vh = nn.Conv2d(channels, c2, kernel_size=(1, 2))  # -> (N, c2, 3, 3) = 9 * c2
        self.conv_vv = nn.Conv2d(channels, c2, kernel_size=(2, 1))  # -> (N, c2, 2, 4) = 8 * c2

        trunk_features = (8 + 9 + 9 + 8) * c2  # 34 * c2
        self.head = DuelingHead(in_features=trunk_features, n_actions=n_actions, hidden=hidden)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw board states and extract flattened convolutional trunk features."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        onehot = log2_onehot_torch(x)
        return self.forward_onehot_features(onehot)

    def forward_onehot_features(self, onehot: torch.Tensor) -> torch.Tensor:
        """Extract the same trunk features from an already validated encoding."""
        onehot = onehot.to(device=self.conv_h.weight.device, dtype=self.conv_h.weight.dtype)

        h = F.relu(self.conv_h(onehot))
        v = F.relu(self.conv_v(onehot))

        hh = F.relu(self.conv_hh(h)).flatten(start_dim=1)
        hv = F.relu(self.conv_hv(h)).flatten(start_dim=1)
        vh = F.relu(self.conv_vh(v)).flatten(start_dim=1)
        vv = F.relu(self.conv_vv(v)).flatten(start_dim=1)

        return torch.cat([hh, hv, vh, vv], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)
