"""Networks: a small CNN over stacked frames, a tanh-Gaussian actor and a DroQ
critic ensemble (dropout + LayerNorm Q-heads)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
HL_GAUSS_BINS, HL_GAUSS_MIN, HL_GAUSS_MAX = 141, -150.0, 200.0


class Encoder(nn.Module):
    """(B, stack, H, W) uint8 -> (B, feature_dim). Stride-2 convs: 64 -> 4 pixels."""

    def __init__(self, stack: int, frame_size: int, channels: list[int], feature_dim: int):
        super().__init__()
        layers, c_in, size = [], stack, frame_size
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.ReLU()]
            c_in, size = c, math.ceil(size / 2)
        self.convs = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c_in * size * size, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.float() / 255.0 - 0.5
        return self.head(self.convs(x).flatten(1))


class NoEncoder(nn.Module):
    """State mode has no camera: (B, stack, 0, 0) -> (B, 0) features."""

    feature_dim = 0

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.new_zeros((frames.shape[0], 0), dtype=torch.float32)


def make_encoder(env: dict, model: dict) -> tuple[nn.Module, int]:
    """The image encoder and its feature size; no encoder in state mode."""
    if env.get("observation", "image") == "state":
        return NoEncoder(), 0
    feat = model["hidden_dim"]
    return Encoder(env["frame_stack"], env["frame_size"], model["encoder_channels"], feat), feat


def mlp(n_in: int, hidden: int, n_out: int, dropout: float = 0.0, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(2):
        layers.append(nn.Linear(n_in, hidden))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        n_in = hidden
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, feature_dim: int, state_dim: int, action_dim: int, hidden: int,
                 layer_norm: bool = True):
        super().__init__()
        self.net = mlp(feature_dim + state_dim, hidden, 2 * action_dim, layer_norm=layer_norm)

    def forward(self, features, state, deterministic: bool = False):
        """Returns (action in [-1, 1], log-probability)."""
        mu, log_std = self.net(torch.cat([features, state], -1)).chunk(2, -1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_prob


class Critic(nn.Module):
    """Ensemble of Q-heads; each head uses dropout and LayerNorm (DroQ)."""

    def __init__(self, feature_dim: int, state_dim: int, action_dim: int, hidden: int,
                 num_critics: int, dropout: float, *, q_bins: int = HL_GAUSS_BINS,
                 q_min: float = HL_GAUSS_MIN, q_max: float = HL_GAUSS_MAX, hl_sigma_ratio: float = 0.75):
        super().__init__()
        n_in = feature_dim + state_dim + action_dim
        self.heads = nn.ModuleList(mlp(n_in, hidden, q_bins, dropout, layer_norm=True) for _ in range(num_critics))
        self.distributional = q_bins > 1
        if self.distributional:
            self.q_min, self.q_max = q_min, q_max
            width = (q_max - q_min) / (q_bins - 1)
            self.register_buffer("support", torch.linspace(q_min, q_max, q_bins), persistent=False)
            self.register_buffer(
                "bin_edges", torch.linspace(q_min - width / 2, q_max + width / 2, q_bins + 1),
                persistent=False,
            )
            self.cdf_scale = 1 / (math.sqrt(2) * hl_sigma_ratio * width)

    def forward(self, features, state, action) -> torch.Tensor:
        x = torch.cat([features, state, action], -1)
        output = torch.stack([h(x) for h in self.heads])
        return output if self.distributional else output.squeeze(-1)

    def expected_q(self, output: torch.Tensor) -> torch.Tensor:
        return (output.softmax(-1) * self.support).sum(-1) if self.distributional else output

    def project(self, target: torch.Tensor) -> torch.Tensor:
        """HL-Gauss mass in each bin, normalized over the finite support."""
        clipped = target.clamp(self.q_min, self.q_max)
        cdf = torch.erf((self.bin_edges - clipped.unsqueeze(-1)) * self.cdf_scale)
        mass = cdf[..., 1:] - cdf[..., :-1]
        return mass / (cdf[..., -1:] - cdf[..., :1])


def random_shift(frames: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ augmentation: shift each image by up to `pad` pixels (edge padded)."""
    b, c, h, w = frames.shape
    x = F.pad(frames.float(), (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=frames.device)[:h]
    base_x = arange.unsqueeze(0).repeat(h, 1)
    base = torch.stack([base_x, base_x.t()], -1).unsqueeze(0).repeat(b, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=frames.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)
