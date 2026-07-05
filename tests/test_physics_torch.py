"""The differentiable torch MT forward must agree with the NumPy reference
implementation in pimsr-forward and must be autograd-friendly."""

import numpy as np
import pytest
import torch

from pimsr_inversion.physics import mt1d_response_torch

pimsr_forward = pytest.importorskip("pimsr_forward")


def test_matches_numpy_reference():
    from pimsr_forward.mt1d import mt1d_response

    rho = np.array([100.0, 10.0, 1000.0])
    thick = np.array([500.0, 2000.0])
    periods = np.logspace(-2, 3, 20)

    rho_a_np, phase_np = mt1d_response(rho, thick, periods)

    log_rho_a_t, phase_t = mt1d_response_torch(
        torch.tensor(np.log10(rho)).unsqueeze(0),
        torch.tensor(thick),
        torch.tensor(periods),
    )
    np.testing.assert_allclose(
        log_rho_a_t.squeeze(0).numpy(), np.log10(rho_a_np), rtol=1e-6
    )
    np.testing.assert_allclose(phase_t.squeeze(0).numpy(), phase_np, rtol=1e-6)


def test_gradients_flow():
    rho = torch.full((2, 5), 2.0, dtype=torch.float64, requires_grad=True)
    thick = torch.tensor([100.0, 300.0, 1000.0, 3000.0], dtype=torch.float64)
    periods = torch.logspace(-2, 3, 12, dtype=torch.float64)
    log_rho_a, phase = mt1d_response_torch(rho, thick, periods)
    (log_rho_a.sum() + phase.sum()).backward()
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()
    assert rho.grad.abs().sum() > 0


def test_halfspace_phase_is_45deg():
    rho = torch.log10(torch.tensor([[50.0]], dtype=torch.float64))
    thick = torch.zeros(0, dtype=torch.float64)
    periods = torch.logspace(-1, 2, 8, dtype=torch.float64)
    log_rho_a, phase = mt1d_response_torch(rho, thick, periods)
    np.testing.assert_allclose(log_rho_a.numpy(), np.log10(50.0), rtol=1e-8)
    np.testing.assert_allclose(phase.numpy(), 45.0, rtol=1e-6)
