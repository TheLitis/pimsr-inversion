import numpy as np
import torch

from pimsr_inversion.data import grid_cell_thicknesses
from pimsr_inversion.losses import PimsrLoss, heteroscedastic_nll
from pimsr_inversion.network import PimsrNet


def _fake_batch(bs=4, n_periods=24, n_grav=16, n_depth=64):
    n_obs = 2 * n_periods + n_grav
    return {
        "obs": torch.randn(bs, n_obs),
        "log_rho": torch.randn(bs, n_depth),
        "density": torch.randn(bs, n_depth) * 0.1,
        "scenario": torch.randint(0, 4, (bs,)),
        "obs_log_rho_a": torch.full((bs, n_periods), 2.0),
        "obs_phase": torch.full((bs, n_periods), 45.0),
    }, n_obs


def test_network_output_shapes():
    batch, n_obs = _fake_batch()
    net = PimsrNet(n_obs=n_obs)
    out = net(batch["obs"])
    assert out["log_rho"].shape == (4, 64)
    assert out["density"].shape == (4, 64)
    assert out["log_sigma_rho"].shape == (4, 64)
    assert out["scenario_logits"].shape == (4, 4)


def test_loss_backward_and_components():
    batch, n_obs = _fake_batch()
    net = PimsrNet(n_obs=n_obs)
    depth_grid = np.logspace(1.0, np.log10(6.0e4), 64)
    crit = PimsrLoss(
        depth_cell_thickness=torch.from_numpy(grid_cell_thicknesses(depth_grid)),
        periods=torch.logspace(-3, 4, 24, dtype=torch.float64),
    )
    out = net(batch["obs"])
    losses = crit(out, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    for key in ("rho_nll", "density_nll", "scenario_ce", "physics"):
        assert torch.isfinite(losses[key])


def test_heteroscedastic_nll_penalises_overconfidence():
    pred = torch.zeros(10)
    target = torch.ones(10)  # wrong by 1
    confident = heteroscedastic_nll(pred, target, torch.full((10,), -4.0))
    calibrated = heteroscedastic_nll(pred, target, torch.zeros(10))
    assert confident > calibrated
