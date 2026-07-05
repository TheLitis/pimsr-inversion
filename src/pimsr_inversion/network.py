"""Multi-task inversion network.

Encoder ingests the concatenated observable vector (MT apparent
resistivity + phase across periods, gravity anomaly profile) and decodes:

  * ``log_rho``   -- log10 resistivity on the fixed depth grid
  * ``density``   -- density contrast on the fixed depth grid
  * ``log_sigma`` -- per-cell heteroscedastic log-variance (aleatoric
    uncertainty, Kendall & Gal 2017)
  * ``scenario``  -- scenario-class logits (background / conductor / void /
    dense body)

A 1-D conv decoder over the depth axis enforces spatial coherence of the
recovered profiles instead of predicting each cell independently.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class PimsrNet(nn.Module):
    """Observables -> (log_rho, density, log_sigma, scenario logits)."""

    def __init__(
        self,
        n_obs: int,
        n_depth: int = 64,
        n_scenarios: int = 4,
        width: int = 256,
        conv_channels: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.encoder = nn.Sequential(
            nn.Linear(n_obs, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, conv_channels * n_depth),
        )
        self.conv_channels = conv_channels
        blocks = [ResidualBlock1d(conv_channels, dilation=2**i) for i in range(n_blocks)]
        self.decoder = nn.Sequential(*blocks)
        self.head_rho = nn.Conv1d(conv_channels, 1, 1)
        self.head_density = nn.Conv1d(conv_channels, 1, 1)
        self.head_log_sigma = nn.Conv1d(conv_channels, 2, 1)
        self.head_scenario = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_channels, n_scenarios),
        )

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(obs).view(-1, self.conv_channels, self.n_depth)
        z = self.decoder(z)
        log_sigma = self.head_log_sigma(z).clamp(-6.0, 4.0)
        return {
            "log_rho": self.head_rho(z).squeeze(1),
            "density": self.head_density(z).squeeze(1),
            "log_sigma_rho": log_sigma[:, 0],
            "log_sigma_density": log_sigma[:, 1],
            "scenario_logits": self.head_scenario(z),
        }
