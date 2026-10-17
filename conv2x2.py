"""Lightweight 2x2 convolutional Dueling DQN architecture for 2048."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dueling import DuelingHead
from encoding import NUM_CHANNELS, log2_onehot_torch


class OneHotConv2x2DQN(nn.Module):
    """2048 Dueling DQN with log2 one-hot encoding and 2x2 convolutional trunk."""

    architecture: str = "onehot_conv2x2_v1"

    def __init__(self, hidden: int = 256, n_actions: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc = nn.Linear(128 * 2 * 2, 256)
        self.head = DuelingHead(in_features=256, n_actions=n_actions, hidden=hidden)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw board states and extract 256-d convolutional trunk features."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        onehot = log2_onehot_torch(x)
        onehot = onehot.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        h = F.relu(self.conv1(onehot))
        h = F.relu(self.conv2(h))
        flat = h.flatten(start_dim=1)
        return F.relu(self.fc(flat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)
