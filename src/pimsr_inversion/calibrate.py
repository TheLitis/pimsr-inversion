"""Post-hoc uncertainty recalibration via temperature scaling.

The heteroscedastic sigma heads are trained with a Gaussian NLL, but both
multi-task trade-offs and physics fine-tuning can leave them mis-scaled
(observed 1-sigma coverage drifting from the nominal 0.683). Temperature
scaling (Guo et al. 2017, adapted to regression) fixes this with a single
scalar per head fitted on the validation split:

    sigma_cal = T * sigma_raw,   T^2 = mean(z^2),   z = (mu - y) / sigma_raw

which is the closed-form NLL minimiser over T. The temperatures are stored
in the checkpoint so downstream consumers apply them transparently.

Usage:
    python -m pimsr_inversion.calibrate \
        --checkpoint best.pt --val-h5 ds_val.h5 --out best_calibrated.pt
"""

from __future__ import annotations

import argparse
import json

import torch

from .data import DENSITY_SCALE, NormStats, PimsrDataset
from .network import PimsrNet


@torch.no_grad()
def fit_temperatures(
    model: PimsrNet, ds: PimsrDataset, batch_size: int = 1024
) -> dict[str, float]:
    """Closed-form NLL-optimal temperature per sigma head on ``ds``."""
    model.eval()
    z2_rho, z2_den = [], []
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        obs = torch.stack([ds[i]["obs"] for i in idx])
        out = model(obs)
        # heads emit log-variance (see losses.heteroscedastic_nll)
        sig_rho = torch.exp(0.5 * out["log_sigma_rho"])
        sig_den = torch.exp(0.5 * out["log_sigma_density"])
        tgt_rho = torch.stack([ds[i]["log_rho"] for i in idx])
        tgt_den = torch.stack([ds[i]["density"] for i in idx])
        z2_rho.append(((out["log_rho"] - tgt_rho) / sig_rho) ** 2)
        z2_den.append(((out["density"] - tgt_den) / sig_den) ** 2)
    t_rho = float(torch.cat(z2_rho).mean().sqrt())
    t_den = float(torch.cat(z2_den).mean().sqrt())
    return {"sigma_temperature_rho": t_rho, "sigma_temperature_density": t_den}


@torch.no_grad()
def coverage_after(
    model: PimsrNet, ds: PimsrDataset, t_rho: float, batch_size: int = 1024
) -> float:
    """Empirical 1-sigma coverage of the rho head after scaling by ``t_rho``."""
    model.eval()
    hits, total = 0, 0
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        obs = torch.stack([ds[i]["obs"] for i in idx])
        out = model(obs)
        sig = torch.exp(0.5 * out["log_sigma_rho"]) * t_rho
        tgt = torch.stack([ds[i]["log_rho"] for i in idx])
        hits += int(((out["log_rho"] - tgt).abs() <= sig).sum())
        total += tgt.numel()
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = NormStats.from_dict(ckpt["norm_stats"])
    ds = PimsrDataset(args.val_h5, stats)
    model = PimsrNet(
        n_obs=ds.n_obs,
        n_depth=int(ckpt["n_depth"]),
        n_scenarios=int(ckpt.get("n_scenarios", 4)),
    )
    model.load_state_dict(ckpt["model_state"])

    before = coverage_after(model, ds, 1.0)
    temps = fit_temperatures(model, ds)
    after = coverage_after(model, ds, temps["sigma_temperature_rho"])

    ckpt.update(temps)
    ckpt["calibration"] = {
        "val_coverage_1sigma_before": before,
        "val_coverage_1sigma_after": after,
        "nominal": 0.6827,
        "density_scale": DENSITY_SCALE,
    }
    torch.save(ckpt, args.out)
    print(json.dumps({**temps, **ckpt["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
