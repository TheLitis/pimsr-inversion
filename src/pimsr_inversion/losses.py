"""Physics-informed multi-task loss.

Total loss =
    heteroscedastic Gaussian NLL on log-resistivity
  + heteroscedastic Gaussian NLL on density contrast
  + cross-entropy on scenario class
  + lambda_phys * data-misfit of the *differentiable* MT forward applied to
    the predicted resistivity profile vs the observed (noisy) responses.

The physics term closes the loop: the network is penalised not only for
deviating from the ground-truth model but for predicting profiles whose
simulated response disagrees with the actual measurement. This is what
makes the method "physics-informed" rather than a pure regression.
"""

from __future__ import annotations

import torch
from torch import nn

from .physics import mt1d_response_torch


def heteroscedastic_nll(
    pred: torch.Tensor, target: torch.Tensor, log_sigma: torch.Tensor
) -> torch.Tensor:
    """Gaussian NLL with learned per-cell variance (Kendall & Gal 2017)."""
    inv_var = torch.exp(-log_sigma)
    return (0.5 * inv_var * (pred - target) ** 2 + 0.5 * log_sigma).mean()


class PimsrLoss(nn.Module):
    def __init__(
        self,
        depth_cell_thickness: torch.Tensor,
        periods: torch.Tensor,
        lambda_density: float = 1.0,
        lambda_scenario: float = 0.3,
        lambda_phys: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("cell_thickness", depth_cell_thickness)
        self.register_buffer("periods", periods)
        self.lambda_density = lambda_density
        self.lambda_scenario = lambda_scenario
        self.lambda_phys = lambda_phys
        self.ce = nn.CrossEntropyLoss()

    def physics_misfit(
        self,
        pred_log_rho: torch.Tensor,
        obs_log_rho_a: torch.Tensor,
        obs_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Normalised misfit between simulated response of the predicted
        profile and the observed MT response."""
        log_rho_a, phase = mt1d_response_torch(
            pred_log_rho, self.cell_thickness, self.periods
        )
        misfit = (log_rho_a - obs_log_rho_a.to(torch.float64)) ** 2 + (
            (phase - obs_phase.to(torch.float64)) / 45.0
        ) ** 2
        return misfit.mean().to(pred_log_rho.dtype)

    def forward(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        l_rho = heteroscedastic_nll(out["log_rho"], batch["log_rho"], out["log_sigma_rho"])
        l_den = heteroscedastic_nll(
            out["density"], batch["density"], out["log_sigma_density"]
        )
        l_scn = self.ce(out["scenario_logits"], batch["scenario"])
        l_phy = self.physics_misfit(
            out["log_rho"], batch["obs_log_rho_a"], batch["obs_phase"]
        )
        total = (
            l_rho
            + self.lambda_density * l_den
            + self.lambda_scenario * l_scn
            + self.lambda_phys * l_phy
        )
        return {
            "total": total,
            "rho_nll": l_rho.detach(),
            "density_nll": l_den.detach(),
            "scenario_ce": l_scn.detach(),
            "physics": l_phy.detach(),
        }
