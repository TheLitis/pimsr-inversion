"""Differentiable 2D TE-mode MT forward for physics fine-tuning.

Wraps :class:`pimsr_forward.mt2d.MT2DForward`'s SimPEG simulation in a
``torch.autograd.Function``: the forward pass runs the real 2D solve
(``dpred``), the backward pass uses the adjoint (``Jtvec``) to pull data
gradients back to the conductivity model, then chains analytically
through the nearest-neighbour section-to-mesh map and the
``sigma = 10**(-log10_rho)`` transform onto the network's output section.

This gives real-profile fine-tuning a physics signal that "sees" lateral
structure: the per-column 1D loss treats galvanic/2D distortion as
information (the row-I adapter failure), while the 2D forward correctly
attributes it to off-column resistivity.

Cost: one 2D solve + one adjoint per profile per step (~2-3 s on CPU for
8 frequencies x 16 stations), so use ~100-200 steps, not 600.

SimPEG is required (``pimsr-forward[mt2d]``).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

__all__ = ["Physics2DLoss"]

_LN10 = float(np.log(10.0))


class _MT2DResponseFn(torch.autograd.Function):
    """log10(rho_a) and phase from a section, differentiable via Jtvec."""

    @staticmethod
    def forward(ctx, log_rho: torch.Tensor, ph2d: "Physics2DLoss"):
        lr_np = log_rho.detach().cpu().numpy().astype(float)
        sigma = ph2d._sigma_from_log_rho(lr_np)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = ph2d._sim.dpred(sigma)
        d = data.reshape(ph2d.n_freq, 2, ph2d.n_station)
        rho_a = np.maximum(d[:, 0, :], 1e-12)
        phase = d[:, 1, :] + 180.0  # SimPEG xy TE -> first quadrant
        ctx.ph2d = ph2d
        ctx.sigma = sigma
        ctx.rho_a = rho_a
        ctx.lr_shape = log_rho.shape
        ctx.device = log_rho.device
        out = np.stack([np.log10(rho_a), phase])  # (2, n_freq, n_station)
        return torch.from_numpy(out.astype(np.float64)).to(log_rho.device)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ph2d = ctx.ph2d
        g = grad_out.detach().cpu().numpy().astype(float)
        # d log10(rho)/d rho = 1/(rho ln10); phase gradient passes through
        v = np.empty((ph2d.n_freq, 2, ph2d.n_station))
        v[:, 0, :] = g[0] / (ctx.rho_a * _LN10)
        v[:, 1, :] = g[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grad_sigma = ph2d._sim.Jtvec(ctx.sigma, v.ravel())
        # chain: sigma_cell = 10**(-lr[iz, ix]) => d sigma/d lr = -ln10*sigma
        grad_lr = np.zeros(ctx.lr_shape, dtype=float).ravel()
        np.add.at(
            grad_lr,
            ph2d._flat_idx,
            grad_sigma[ph2d._active_idx] * (-_LN10) * ctx.sigma[ph2d._active_idx],
        )
        grad = torch.from_numpy(grad_lr.reshape(ctx.lr_shape)).to(ctx.device)
        return grad.to(grad_out.dtype), None


class Physics2DLoss:
    """Shift-invariant chi-squared against observed TE curves through the
    true 2D forward.

    Parameters
    ----------
    station_cols : column index (0..n_x-1) of each observed station on the
        network's section grid.
    x_grid, depth_grid : the section grid the network predicts on.
    obs_periods : observed period vector (s). Simulator frequencies are the
        subset of its own valid band (0.1..100 s) nearest to observations.
    sigma_lr, sigma_ph : observation errors, log10 rho / degrees.
    """

    def __init__(
        self,
        station_cols: np.ndarray,
        x_grid: np.ndarray,
        depth_grid: np.ndarray,
        obs_periods: np.ndarray,
        sigma_lr: float = 0.1,
        sigma_ph: float = 5.0,
    ) -> None:
        from pimsr_forward.mt2d import MT2DForward

        station_cols = np.asarray(station_cols, int)
        self.x_grid = np.asarray(x_grid, float)
        self.depth_grid = np.asarray(depth_grid, float)
        self.sigma_lr = float(sigma_lr)
        self.sigma_ph = float(sigma_ph)

        # simulator periods: nearest observed period per default band point,
        # deduplicated; remember which observed index each one matches
        band = np.logspace(-1, 2, 8)  # s, within the validated mesh band
        obs_periods = np.asarray(obs_periods, float)
        idx = sorted({int(np.argmin(np.abs(obs_periods - p))) for p in band})
        self.obs_period_idx = np.asarray(idx, int)
        freqs = 1.0 / obs_periods[self.obs_period_idx]

        self._fwd = MT2DForward(
            frequencies=freqs, station_x=self.x_grid[station_cols]
        )
        self._sim = self._fwd._sim
        self.n_freq = len(freqs)
        self.n_station = len(station_cols)

        # precompute mesh-cell -> section-cell index map (nearest neighbour,
        # identical to MT2DForward.sigma_from_section)
        m = self._fwd._m
        cc = m.active_cc
        ix = np.clip(np.searchsorted(self.x_grid, cc[:, 0]), 0, len(self.x_grid) - 1)
        iz = np.clip(
            np.searchsorted(self.depth_grid, -cc[:, 1]), 0, len(self.depth_grid) - 1
        )
        self._active_idx = m.active_idx
        self._flat_idx = iz * len(self.x_grid) + ix
        self._air_sigma = 1e-8
        self._n_cells = m.mesh.n_cells

    def _sigma_from_log_rho(self, log_rho: np.ndarray) -> np.ndarray:
        sigma = np.full(self._n_cells, self._air_sigma)
        sigma[self._active_idx] = 10.0 ** (-log_rho.ravel()[self._flat_idx])
        return sigma

    def response(self, log_rho: torch.Tensor) -> torch.Tensor:
        """(2, n_freq, n_station): [log10 rho_a, phase_deg], differentiable."""
        return _MT2DResponseFn.apply(log_rho, self)

    def misfit(
        self,
        log_rho: torch.Tensor,
        obs_lr: torch.Tensor,
        obs_ph: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shift-invariant chi2. Observations are (n_periods_obs,
        n_station) on the full observed period grid; the simulator subset
        ``obs_period_idx`` is compared."""
        pred = self.response(log_rho)
        sel = torch.as_tensor(self.obs_period_idx, device=obs_lr.device)
        o_lr = obs_lr.index_select(0, sel)
        o_ph = obs_ph.index_select(0, sel)
        m = obs_mask.index_select(0, sel).to(pred.dtype)
        p_lr, p_ph = pred[0].to(o_lr.dtype), pred[1].to(o_ph.dtype)
        # per-station static shift: remove masked-mean log-rho offset
        w = m.sum(dim=0).clamp(min=1.0)
        off = ((o_lr - p_lr) * m).sum(dim=0) / w
        r_lr = (o_lr - p_lr - off[None, :]) * m / self.sigma_lr
        r_ph = (o_ph - p_ph) * m / self.sigma_ph
        n = m.sum().clamp(min=1.0)
        return (r_lr.square().sum() + r_ph.square().sum()) / (2.0 * n)
