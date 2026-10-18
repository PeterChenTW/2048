"""Invertible value rescaling h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x (Pohlen et al., 2018).

Used with raw-score rewards and gamma up to 1.0: the network predicts h(V); targets
are h(r + gamma * h^{-1}(V_target)). Closed-form inverse (Pohlen et al., Prop. A.2):
h^{-1}(y) = sign(y) * (((sqrt(1 + 4 eps (|y| + 1 + eps)) - 1) / (2 eps))^2 - 1).
"""
import torch

DEFAULT_EPS = 1e-3


def h(x, eps=DEFAULT_EPS):
    x = torch.as_tensor(x)
    return torch.sign(x) * (torch.sqrt(x.abs() + 1) - 1) + eps * x


def h_inv(y, eps=DEFAULT_EPS):
    y = torch.as_tensor(y)
    if eps == 0:
        return torch.sign(y) * ((y.abs() + 1) ** 2 - 1)
    root = (torch.sqrt(1 + 4 * eps * (y.abs() + 1 + eps)) - 1) / (2 * eps)
    return torch.sign(y) * (root ** 2 - 1)
