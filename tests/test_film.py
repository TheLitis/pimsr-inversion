"""FiLM adapter behaviour on the 2D network."""

import torch

from pimsr_inversion.network2d import PimsrNet2D


def _c_mid(model: PimsrNet2D) -> int:
    return model.mid[3].out_channels


def test_zero_film_is_identity():
    m = PimsrNet2D(in_channels=4, scen_head="multiscale")
    m.eval()
    x = torch.randn(2, 4, 24, 16)
    c = _c_mid(m)
    with torch.no_grad():
        base = m(x)
        filmed = m(x, film=(torch.zeros(c), torch.zeros(c)))
    for k in ("log_rho", "log_sigma_rho", "scenario_logits"):
        assert torch.equal(base[k], filmed[k])


def test_nonzero_film_changes_output():
    m = PimsrNet2D()
    m.eval()
    x = torch.randn(1, 2, 24, 16)
    c = _c_mid(m)
    with torch.no_grad():
        base = m(x)["log_rho"]
        filmed = m(x, film=(torch.full((c,), 0.5), torch.full((c,), 0.2)))["log_rho"]
    assert not torch.equal(base, filmed)


def test_film_gradients_flow():
    m = PimsrNet2D()
    x = torch.randn(1, 2, 24, 16)
    c = _c_mid(m)
    gamma = torch.zeros(c, requires_grad=True)
    beta = torch.zeros(c, requires_grad=True)
    out = m(x, film=(gamma, beta))["log_rho"]
    out.square().mean().backward()
    assert gamma.grad is not None and float(gamma.grad.abs().sum()) > 0
    assert beta.grad is not None and float(beta.grad.abs().sum()) > 0
