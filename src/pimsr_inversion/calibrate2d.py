"""Post-hoc sigma recalibration for the 2D network.

Training logs (v3 run 28891525012) show the root cause of the val-NLL
"drift": val RMSE plateaus by epoch ~12 while train loss keeps falling
into memorisation (negative NLL by epoch 43). The sigma head then tracks
the vanishing *train* residuals and under-covers validation data. No
training-time loss (plain NLL, sigma-reg, beta-NLL) can fix this — the
sigma must be re-fitted against residuals the mean network has NOT
memorised.

This module freezes the mean prediction and fits an affine map in
log-sigma space on the validation split:

    log_sigma' = a * log_sigma + b

minimising the Gaussian NLL. Temperature scaling is the a=1 special
case; the extra slope parameter lets over-sharpened sigma fields be
flattened as well as rescaled. Parameters are stored in the checkpoint
as ``sigma_affine2d`` and applied transparently by the benchmarks.

Usage:
    python -m pimsr_inversion.calibrate2d \
        --checkpoint best2d.pt --val-h5 ds2d_val.h5 --out best2d_cal.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .network2d import PimsrNet2D
from .train2d import Section2DDataset


@torch.no_grad()
def _collect(model: PimsrNet2D, ds: Section2DDataset, batch: int = 64):
    """Frozen-model residuals and raw log-sigma over a dataset."""
    model.eval()
    res, ls = [], []
    for start in range(0, len(ds), batch):
        idx = range(start, min(start + batch, len(ds)))
        obs = torch.stack([torch.from_numpy(ds.obs[i]) for i in idx])
        tgt = torch.stack([torch.from_numpy(ds.target[i]) for i in idx])
        out = model(obs)
        res.append(out["log_rho"] - tgt)
        ls.append(out["log_sigma_rho"])
    return torch.cat(res), torch.cat(ls)


def fit_affine(res: torch.Tensor, ls: torch.Tensor) -> dict[str, float]:
    """NLL-optimal affine recalibration of log-sigma (frozen residuals)."""
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = a * ls + b
        nll = 0.5 * (s + res.square() * torch.exp(-s)).mean()
        nll.backward()
        return nll

    opt.step(closure)
    with torch.no_grad():
        s = a * ls + b
        nll = float(0.5 * (s + res.square() * torch.exp(-s)).mean())
    return {"a": float(a.detach()), "b": float(b.detach()), "val_nll": nll}


@torch.no_grad()
def coverage(res: torch.Tensor, ls: torch.Tensor, a: float, b: float) -> float:
    sig = torch.exp(0.5 * (a * ls + b))
    return float((res.abs() <= sig).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PimsrNet2D.from_checkpoint(ckpt)

    # apply the checkpoint's own normalisation, exactly as in training
    ds = Section2DDataset(
        args.val_h5,
        stats={
            "mean": np.asarray(ckpt["stats_mean"], dtype=np.float32),
            "std": np.asarray(ckpt["stats_std"], dtype=np.float32),
        },
    )

    res, ls = _collect(model, ds)
    before_cov = coverage(res, ls, 1.0, 0.0)
    before_nll = float(0.5 * (ls + res.square() * torch.exp(-ls)).mean())

    fit = fit_affine(res, ls)
    after_cov = coverage(res, ls, fit["a"], fit["b"])

    ckpt["sigma_affine2d"] = {"a": fit["a"], "b": fit["b"]}
    ckpt["calibration2d"] = {
        "val_nll_before": before_nll,
        "val_nll_after": fit["val_nll"],
        "val_coverage_1sigma_before": before_cov,
        "val_coverage_1sigma_after": after_cov,
        "nominal": 0.6827,
    }
    torch.save(ckpt, args.out)
    print(json.dumps({**ckpt["sigma_affine2d"], **ckpt["calibration2d"]}, indent=2))


if __name__ == "__main__":
    main()
