"""Differentiable 1D MT forward modeling in torch.

Re-implements the exact Wait impedance recursion from ``pimsr_forward.mt1d``
with torch complex tensors so the data misfit of a *predicted* resistivity
profile can be back-propagated into the network (physics-informed loss).

The predicted profile lives on the fixed log-depth grid (64 nodes). We treat
each grid cell as a layer whose thickness is the spacing between grid nodes,
terminated by a half-space with the deepest node's resistivity.
"""

from __future__ import annotations

import math

import torch

MU0 = 4.0e-7 * math.pi


def grid_to_layers(depth_grid: torch.Tensor) -> torch.Tensor:
    """Layer thicknesses (n_grid - 1,) from a monotone depth grid (n_grid,)."""
    return depth_grid[1:] - depth_grid[:-1]


def mt1d_impedance_torch(
    log10_res: torch.Tensor,
    thicknesses: torch.Tensor,
    periods: torch.Tensor,
) -> torch.Tensor:
    """Batched surface impedance.

    Parameters
    ----------
    log10_res : (B, n_layers) log10 resistivity per layer; last layer is the
        terminating half-space.
    thicknesses : (n_layers - 1,) shared layer thicknesses, m.
    periods : (n_periods,) s.

    Returns
    -------
    Z : (B, n_periods) complex64/128 surface impedance.
    """
    res = torch.pow(10.0, log10_res.to(torch.float64))  # (B, L)
    sigma = 1.0 / res
    omega = 2.0 * math.pi / periods.to(torch.float64)  # (P,)

    # k = sqrt(i omega mu0 sigma): (B, P, L)
    iomega = 1j * omega.view(1, -1, 1) * MU0
    k = torch.sqrt(iomega * sigma.unsqueeze(1))
    z_intr = iomega / k

    z = z_intr[..., -1]
    n_layers = res.shape[1]
    for j in range(n_layers - 2, -1, -1):
        zj = z_intr[..., j]
        t = torch.tanh(k[..., j] * thicknesses[j])
        z = zj * (z + zj * t) / (zj + z * t)
    return z


def mt1d_response_torch(
    log10_res: torch.Tensor,
    thicknesses: torch.Tensor,
    periods: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(log10 apparent resistivity, phase deg), each (B, n_periods), float64."""
    Z = mt1d_impedance_torch(log10_res, thicknesses, periods)
    omega = 2.0 * math.pi / periods.to(torch.float64)
    rho_app = Z.abs().square() / (omega * MU0)
    phase = torch.rad2deg(torch.atan2(Z.imag, Z.real))
    return torch.log10(rho_app), phase


class MTPhysicsLoss(torch.nn.Module):
    """Chi^2-style misfit between forward-modeled prediction and observed MT.

    Static shift makes the *level* of log10(rho_a) unreliable in the field, so
    the apparent-resistivity term is compared after removing the per-sample
    mean offset (phase is immune to static shift and enters directly).
    """

    def __init__(
        self,
        depth_grid: torch.Tensor,
        periods: torch.Tensor,
        rho_sigma: float = 0.03 / math.log(10.0),
        phase_sigma_deg: float = 1.0,
        shift_invariant: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("thicknesses", grid_to_layers(depth_grid).to(torch.float64))
        self.register_buffer("periods", periods.to(torch.float64))
        self.rho_sigma = rho_sigma
        self.phase_sigma_deg = phase_sigma_deg
        self.shift_invariant = shift_invariant

    def forward(
        self,
        pred_log10_res: torch.Tensor,
        obs_log10_rho: torch.Tensor,
        obs_phase: torch.Tensor,
    ) -> torch.Tensor:
        sim_rho, sim_phase = mt1d_response_torch(
            pred_log10_res, self.thicknesses, self.periods
        )
        d_rho = sim_rho - obs_log10_rho.to(torch.float64)
        if self.shift_invariant:
            d_rho = d_rho - d_rho.mean(dim=1, keepdim=True)
        d_phase = sim_phase - obs_phase.to(torch.float64)
        chi2 = (d_rho / self.rho_sigma).square().mean() + (
            d_phase / self.phase_sigma_deg
        ).square().mean()
        return chi2.to(pred_log10_res.dtype) * 0.5
