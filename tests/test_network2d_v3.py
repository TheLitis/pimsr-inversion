"""v3 features: 4-channel input, multiscale scenario head, beta-NLL,
checkpoint round-trip via from_checkpoint."""

import torch

from pimsr_inversion.network2d import PimsrNet2D
from pimsr_inversion.train2d import _loss


def test_four_channel_forward():
    model = PimsrNet2D(
        n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5,
        in_channels=4, scen_head="multiscale",
    )
    out = model(torch.randn(2, 4, 24, 16))
    assert out["log_rho"].shape == (2, 48, 64)
    assert out["scenario_logits"].shape == (2, 5)


def test_multiscale_head_gradients():
    model = PimsrNet2D(
        n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5,
        in_channels=4, scen_head="multiscale",
    )
    out = model(torch.randn(2, 4, 24, 16))
    out["scenario_logits"].sum().backward()
    head_grads = [
        p.grad for n, p in model.named_parameters()
        if "head_scen" in n and p.grad is not None
    ]
    assert len(head_grads) > 0
    assert all(torch.isfinite(g).all() for g in head_grads)


def test_beta_nll_finite_and_differs():
    model = PimsrNet2D(n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5)
    x = torch.randn(2, 2, 24, 16)
    tgt = torch.randn(2, 48, 64)
    scen = torch.tensor([0, 3])
    out = model(x)
    l0, _ = _loss(out, tgt, scen, sigma_on=True, beta=0.0)
    lb, _ = _loss(out, tgt, scen, sigma_on=True, beta=0.5)
    assert torch.isfinite(l0) and torch.isfinite(lb)
    # with a non-constant sigma head the reweighting must change the loss
    assert not torch.isclose(l0, lb)


def test_beta_nll_backward():
    model = PimsrNet2D(n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5)
    x = torch.randn(2, 2, 24, 16)
    tgt = torch.randn(2, 48, 64)
    scen = torch.tensor([1, 2])
    loss, _ = _loss(model(x), tgt, scen, sigma_on=True, beta=0.5, sigma_reg=0.05)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_from_checkpoint_roundtrip_v3():
    model = PimsrNet2D(
        n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5,
        in_channels=4, scen_head="multiscale",
    )
    ckpt = {
        "model_state": model.state_dict(),
        "n_freq": 24, "n_stations": 16, "n_depth": 48, "n_x": 64,
        "n_scenarios": 5, "in_channels": 4, "scen_head": "multiscale",
    }
    rebuilt = PimsrNet2D.from_checkpoint(ckpt)
    assert rebuilt.in_channels == 4
    x = torch.randn(1, 4, 24, 16)
    with torch.no_grad():
        a = model(x)["log_rho"]
        b = rebuilt(x)["log_rho"]
    assert torch.allclose(a, b)


def test_from_checkpoint_legacy_defaults():
    # legacy checkpoints carry no in_channels / scen_head keys
    model = PimsrNet2D(n_freq=24, n_stations=16, n_depth=48, n_x=64, n_scenarios=5)
    ckpt = {
        "model_state": model.state_dict(),
        "n_freq": 24, "n_stations": 16, "n_depth": 48, "n_x": 64,
        "n_scenarios": 5,
    }
    rebuilt = PimsrNet2D.from_checkpoint(ckpt)
    assert rebuilt.in_channels == 2
    assert rebuilt.scen_head_kind == "gap"
