"""Conv-2D profile inversion network.

Maps a pseudo-section of MT observables -- shape (B, 2, n_freq, n_stations)
with channels (log10 apparent resistivity, phase/45) -- to a 2-D resistivity
section on the (depth, x) grid, plus a heteroscedastic sigma head and a
scenario classification head.

Architecture: a compact U-Net-style encoder-decoder. The encoder sees the
(frequency x station) pseudo-section; the bottleneck is reshaped and decoded
onto the (depth x x-grid) output raster. Frequency roughly maps to depth
(skin-depth relation), which the decoder learns to warp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PimsrNet2D"]


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


class PimsrNet2D(nn.Module):
    """Pseudo-section -> resistivity section with uncertainty."""

    def __init__(
        self,
        n_freq: int = 24,
        n_stations: int = 16,
        n_depth: int = 48,
        n_x: int = 64,
        n_scenarios: int = 5,
        width: int = 48,
    ) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.n_x = n_x
        w = width

        self.enc1 = _block(2, w)
        self.enc2 = _block(w, 2 * w)
        self.enc3 = _block(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)

        # bottleneck operates at (n_freq/4, n_stations/4)
        self.mid = _block(4 * w, 4 * w)

        # decoder upsamples directly to the output raster resolution
        self.dec2 = _block(4 * w + 2 * w, 2 * w)
        self.dec1 = _block(2 * w + w, w)

        self.head_rho = nn.Conv2d(w, 1, 1)
        self.head_sigma = nn.Conv2d(w, 1, 1)
        self.head_scen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, n_scenarios)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(e3)

        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # warp from pseudo-section raster to the physical (depth, x) raster
        out = F.interpolate(
            d1, size=(self.n_depth, self.n_x), mode="bilinear", align_corners=False
        )
        return {
            "log_rho": self.head_rho(out).squeeze(1),
            "log_sigma_rho": self.head_sigma(out).squeeze(1).clamp(-10.0, 6.0),
            "scenario_logits": self.head_scen(m),
        }
