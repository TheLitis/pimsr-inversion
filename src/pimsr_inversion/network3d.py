"""Memory-conscious 3D inversion baseline for one NVIDIA A100.

Input is a survey cube ``(B, C, frequency, y_station, x_station)`` and output
is a log-resistivity volume.  The network uses checkpointable residual blocks;
production resolution is selected by :class:`Model3DConfig`, not hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ["Model3DConfig", "PimsrNet3D", "estimate_activation_bytes"]


@dataclass(frozen=True)
class Model3DConfig:
    name: str
    crop: tuple[int, int, int]
    width: int
    batch_size: int
    precision: str

    @classmethod
    def preset(cls, name: str) -> "Model3DConfig":
        presets = {
            "local-8gb": cls(name, (24, 16, 24), 16, 1, "fp16"),
            "fallback-24gb": cls(name, (40, 32, 48), 24, 1, "fp16"),
            "a100-40gb": cls(name, (48, 48, 64), 32, 2, "bf16"),
            "a100-80gb": cls(name, (64, 64, 96), 40, 2, "bf16"),
        }
        if name not in presets:
            raise ValueError(f"unknown 3D preset: {name}")
        return presets[name]


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = min(8, cout)
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PimsrNet3D(nn.Module):
    def __init__(self, in_channels=4, width=32, checkpoint_blocks=True):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.enc = _Block(in_channels, width)
        self.mid = _Block(width, 2 * width)
        self.dec = _Block(2 * width + width, width)
        self.pool = nn.MaxPool3d(2)
        self.rho = nn.Conv3d(width, 1, 1)
        self.log_sigma = nn.Conv3d(width, 1, 1)

    def _run(self, block, x):
        if self.checkpoint_blocks and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x, output_shape=None):
        e = self._run(self.enc, x)
        m = self._run(self.mid, self.pool(e))
        up = nn.functional.interpolate(m, size=e.shape[-3:], mode="trilinear", align_corners=False)
        d = self._run(self.dec, torch.cat((up, e), dim=1))
        if output_shape is not None:
            d = nn.functional.interpolate(d, size=output_shape, mode="trilinear", align_corners=False)
        return {
            "log_rho": self.rho(d).squeeze(1),
            "log_sigma_rho": self.log_sigma(d).squeeze(1).clamp(-10, 6),
        }


def estimate_activation_bytes(config: Model3DConfig, training_factor: float = 10.0) -> int:
    """Conservative planning estimate; measured peak memory remains authoritative."""
    voxels = config.batch_size * config.crop[0] * config.crop[1] * config.crop[2]
    scalar_bytes = 2 if config.precision in {"fp16", "bf16"} else 4
    return int(voxels * config.width * scalar_bytes * training_factor)
