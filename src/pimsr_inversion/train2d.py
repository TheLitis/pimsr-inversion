"""Training entry point for the 2D profile inversion network.

Usage:
    pimsr-train2d --train ds2d_train.h5 --val ds2d_val.h5 --out ckpt_dir
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .network2d import PimsrNet2D

__all__ = ["main"]


class Section2DDataset(Dataset):
    """Loads 2D pseudo-section observations and resistivity targets."""

    def __init__(self, path: str, stats: dict | None = None) -> None:
        with h5py.File(path, "r") as f:
            lr = f["obs_mt_log10_rho"][:].astype(np.float32)  # (N, F, S)
            ph = f["obs_mt_phase"][:].astype(np.float32) / 45.0
            self.target = f["target_log10_res"][:].astype(np.float32)  # (N, Z, X)
            self.scenario = f["scenario"][:].astype(np.int64)
        self.obs = np.stack([lr, ph], axis=1)  # (N, 2, F, S)
        if stats is None:
            stats = {
                "mean": self.obs.mean(axis=(0, 2, 3), keepdims=True),
                "std": self.obs.std(axis=(0, 2, 3), keepdims=True) + 1e-6,
            }
        self.stats = stats
        self.obs = (self.obs - stats["mean"]) / stats["std"]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.obs[i]),
            torch.from_numpy(self.target[i]),
            int(self.scenario[i]),
        )


def _loss(out, tgt, scen, *, sigma_on: bool, class_weights=None):
    if sigma_on:
        s = out["log_sigma_rho"]
        fit = 0.5 * (s + (out["log_rho"] - tgt) ** 2 * torch.exp(-s)).mean()
    else:
        # sigma warm-up: plain MSE while the mean head stabilises, so the
        # sigma head cannot absorb early fitting error (the cause of the
        # val-NLL divergence seen after ~epoch 20 in the first 2D run)
        fit = 0.5 * ((out["log_rho"] - tgt) ** 2).mean()
    # total-variation smoothness prior on the predicted section
    p = out["log_rho"]
    tv = (p[:, 1:, :] - p[:, :-1, :]).abs().mean() + (
        p[:, :, 1:] - p[:, :, :-1]
    ).abs().mean()
    ce = F.cross_entropy(out["scenario_logits"], scen, weight=class_weights)
    return fit + 0.05 * tv + 0.1 * ce, {"fit": fit.item(), "tv": tv.item(), "ce": ce.item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--sigma-warmup", type=int, default=15,
        help="epochs of plain MSE before enabling the NLL sigma term",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    train_ds = Section2DDataset(args.train)
    val_ds = Section2DDataset(args.val, stats=train_ds.stats)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    n, _, nf, ns = train_ds.obs.shape
    nz, nx = train_ds.target.shape[1:]
    n_scen = int(train_ds.scenario.max()) + 1

    # inverse-frequency class weights for the scenario head
    counts = np.bincount(train_ds.scenario, minlength=n_scen).astype(np.float64)
    weights = counts.sum() / (n_scen * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"scenario counts: {counts.astype(int).tolist()}", flush=True)
    model = PimsrNet2D(
        n_freq=nf, n_stations=ns, n_depth=nz, n_x=nx, n_scenarios=n_scen
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        sigma_on = epoch >= args.sigma_warmup
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        for obs, tgt, scen in train_dl:
            obs, tgt, scen = obs.to(device), tgt.to(device), scen.to(device)
            opt.zero_grad()
            loss, _ = _loss(
                model(obs), tgt, scen,
                sigma_on=sigma_on, class_weights=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * obs.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss, va_rmse = 0.0, 0.0
        with torch.no_grad():
            for obs, tgt, scen in val_dl:
                obs, tgt, scen = obs.to(device), tgt.to(device), scen.to(device)
                out = model(obs)
                # validation always scores the full NLL objective so that
                # checkpoint selection is comparable across warm-up boundary
                loss, _ = _loss(
                    out, tgt, scen, sigma_on=True, class_weights=class_weights
                )
                va_loss += loss.item() * obs.size(0)
                va_rmse += ((out["log_rho"] - tgt) ** 2).mean().sqrt().item() * obs.size(0)
        va_loss /= len(val_ds)
        va_rmse /= len(val_ds)
        sched.step()

        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
             "val_rmse": va_rmse, "sec": time.time() - t0}
        )
        print(
            f"epoch {epoch}: train {tr_loss:.4f} val {va_loss:.4f} "
            f"rmse {va_rmse:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stats_mean": train_ds.stats["mean"],
                    "stats_std": train_ds.stats["std"],
                    "n_freq": nf, "n_stations": ns,
                    "n_depth": nz, "n_x": nx, "n_scenarios": n_scen,
                    "epoch": epoch, "val_loss": va_loss, "val_rmse": va_rmse,
                },
                out_dir / "best2d.pt",
            )

    (out_dir / "history2d.json").write_text(json.dumps(history, indent=2))
    print(f"best val loss: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
