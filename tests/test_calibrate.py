"""Temperature scaling sanity: closed-form T recovers a known mis-scale."""

import torch

from pimsr_inversion.calibrate import fit_temperatures
from pimsr_inversion.network import PimsrNet


class _SyntheticDs:
    """Targets drawn as mu + eps with eps ~ N(0, (k*sigma_raw)^2), so the
    NLL-optimal temperature must recover k."""

    def __init__(self, model: PimsrNet, n: int, n_obs: int, k: float, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.obs_t = torch.randn(n, n_obs, generator=g)
        with torch.no_grad():
            out = model(self.obs_t)
        sig_rho = torch.exp(0.5 * out["log_sigma_rho"])
        sig_den = torch.exp(0.5 * out["log_sigma_density"])
        self.rho_t = out["log_rho"] + k * sig_rho * torch.randn(
            out["log_rho"].shape, generator=g
        )
        self.den_t = out["density"] + k * sig_den * torch.randn(
            out["density"].shape, generator=g
        )

    def __len__(self):
        return self.obs_t.shape[0]

    def __getitem__(self, i):
        return {
            "obs": self.obs_t[i],
            "log_rho": self.rho_t[i],
            "density": self.den_t[i],
        }


def test_temperature_recovers_known_scale():
    torch.manual_seed(0)
    model = PimsrNet(n_obs=24, n_depth=16, n_scenarios=5, width=32, conv_channels=8)
    model.eval()
    k = 2.5
    ds = _SyntheticDs(model, n=512, n_obs=24, k=k)
    temps = fit_temperatures(model, ds, batch_size=256)
    assert abs(temps["sigma_temperature_rho"] - k) / k < 0.1
    assert abs(temps["sigma_temperature_density"] - k) / k < 0.1


def test_identity_when_calibrated():
    torch.manual_seed(1)
    model = PimsrNet(n_obs=24, n_depth=16, n_scenarios=5, width=32, conv_channels=8)
    model.eval()
    ds = _SyntheticDs(model, n=512, n_obs=24, k=1.0, seed=3)
    temps = fit_temperatures(model, ds, batch_size=256)
    assert abs(temps["sigma_temperature_rho"] - 1.0) < 0.1
