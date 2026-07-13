import pytest
import torch

from pimsr_inversion.network3d import Model3DConfig, PimsrNet3D, estimate_activation_bytes


def test_3d_forward_and_backward():
    model = PimsrNet3D(in_channels=4, width=8, checkpoint_blocks=True).train()
    x = torch.randn(2, 4, 8, 6, 8, requires_grad=True)
    out = model(x, output_shape=(10, 7, 9))
    assert out["log_rho"].shape == (2, 10, 7, 9)
    assert out["log_sigma_rho"].shape == (2, 10, 7, 9)
    (out["log_rho"].square().mean() + out["log_sigma_rho"].mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_a100_presets_and_estimate():
    c40 = Model3DConfig.preset("a100-40gb")
    c80 = Model3DConfig.preset("a100-80gb")
    assert c40.precision == c80.precision == "bf16"
    assert estimate_activation_bytes(c80) > estimate_activation_bytes(c40) > 0
    with pytest.raises(ValueError):
        Model3DConfig.preset("unknown")
