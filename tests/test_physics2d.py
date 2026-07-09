"""Adjoint-gradient validation for the differentiable 2D forward.

Slow (real SimPEG solves): kept small — one gradient check with a
handful of finite-difference probes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simpeg")

from pimsr_inversion.physics2d import Physics2DLoss  # noqa: E402


@pytest.fixture(scope="module")
def loss():
    n_x, n_z = 64, 48
    x_grid = np.linspace(-12000, 12000, n_x)
    depth_grid = np.logspace(1, np.log10(8000), n_z)
    periods = np.logspace(-2, 3, 24)
    cols = np.array([16, 24, 32, 40, 48])
    return Physics2DLoss(cols, x_grid, depth_grid, periods), (n_z, n_x)


def test_adjoint_gradient_matches_fd(loss):
    ph2d, shape = loss
    rng = np.random.default_rng(3)
    lr = 2.0 + 0.3 * rng.standard_normal(shape)
    x = torch.tensor(lr, requires_grad=True)

    obs_lr = torch.full((24, 5), 2.0, dtype=torch.float64)
    obs_ph = torch.full((24, 5), 45.0, dtype=torch.float64)
    mask = torch.ones(24, 5, dtype=torch.bool)

    f = ph2d.misfit(x, obs_lr, obs_ph, mask)
    f.backward()
    g = x.grad.numpy()
    assert np.isfinite(g).all()
    assert np.abs(g).max() > 0

    # FD probes at the cells with the largest analytic gradient
    flat = np.abs(g).ravel()
    probes = np.argsort(flat)[-3:]
    eps = 1e-4
    for p in probes:
        iz, ix = np.unravel_index(p, shape)
        lp = lr.copy()
        lp[iz, ix] += eps
        fp = ph2d.misfit(torch.tensor(lp), obs_lr, obs_ph, mask).item()
        lm = lr.copy()
        lm[iz, ix] -= eps
        fm = ph2d.misfit(torch.tensor(lm), obs_lr, obs_ph, mask).item()
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - g[iz, ix]) / max(abs(fd), abs(g[iz, ix]), 1e-12)
        assert rel < 0.05, f"cell ({iz},{ix}): fd {fd:.3e} vs adjoint {g[iz, ix]:.3e}"


def test_misfit_is_shift_invariant(loss):
    ph2d, shape = loss
    rng = np.random.default_rng(5)
    lr = torch.tensor(2.0 + 0.3 * rng.standard_normal(shape))
    obs_lr = torch.full((24, 5), 2.0, dtype=torch.float64)
    obs_ph = torch.full((24, 5), 45.0, dtype=torch.float64)
    mask = torch.ones(24, 5, dtype=torch.bool)
    f0 = ph2d.misfit(lr, obs_lr, obs_ph, mask).item()
    f1 = ph2d.misfit(lr, obs_lr + 0.7, obs_ph, mask).item()
    assert abs(f0 - f1) < 1e-8
