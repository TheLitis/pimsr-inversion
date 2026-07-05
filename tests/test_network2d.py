"""Shape, gradient, and loss sanity checks for the 2D network."""

import torch

from pimsr_inversion.network2d import PimsrNet2D
from pimsr_inversion.train2d import _loss


def test_forward_shapes():
    model = PimsrNet2D(n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5)
    x = torch.randn(3, 2, 24, 16)
    out = model(x)
    assert out["log_rho"].shape == (3, 48, 64)
    assert out["log_sigma_rho"].shape == (3, 48, 64)
    assert out["scenario_logits"].shape == (3, 5)


def test_loss_backward():
    model = PimsrNet2D(n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5)
    x = torch.randn(2, 2, 24, 16)
    tgt = torch.randn(2, 48, 64)
    scen = torch.tensor([0, 3])
    loss, parts = _loss(model(x), tgt, scen)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert set(parts) == {"nll", "tv", "ce"}


def test_odd_input_sizes():
    # non-power-of-two pseudo-section dims must not crash the U-Net
    model = PimsrNet2D(n_freq=23, n_stations=13, n_depth=48, n_x=64, n_scenarios=5)
    out = model(torch.randn(1, 2, 23, 13))
    assert out["log_rho"].shape == (1, 48, 64)
