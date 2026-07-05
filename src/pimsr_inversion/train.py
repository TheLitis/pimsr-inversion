"""Training entry point.

Usage:
    pimsr-train --train ds_train.h5 --val ds_val.h5 --out checkpoints/ \
        --epochs 60 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PimsrDataset, compute_norm_stats, grid_cell_thicknesses
from .losses import PimsrLoss
from .network import PimsrNet


def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["obs"])
            losses = criterion(out, batch)
            bs = batch["obs"].shape[0]
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v) * bs
            correct = (out["scenario_logits"].argmax(1) == batch["scenario"]).sum()
            agg["scenario_acc"] = agg.get("scenario_acc", 0.0) + float(correct)
            rmse = ((out["log_rho"] - batch["log_rho"]) ** 2).mean(1).sqrt().sum()
            agg["log_rho_rmse"] = agg.get("log_rho_rmse", 0.0) + float(rmse)
            n += bs
    return {k: v / n for k, v in agg.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the PIMSR inversion network")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-phys", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_norm_stats(args.train)
    train_ds = PimsrDataset(args.train, stats)
    val_ds = PimsrDataset(args.val, stats)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = PimsrNet(n_obs=train_ds.n_obs, n_depth=train_ds.n_depth).to(device)
    thick = torch.from_numpy(grid_cell_thicknesses(train_ds.depth_grid))
    criterion = PimsrLoss(
        depth_cell_thickness=thick,
        periods=torch.from_numpy(train_ds.periods),
        lambda_phys=args.lambda_phys,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_total = 0.0
        n_batches = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["obs"])
            losses = criterion(out, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_total += float(losses["total"])
            n_batches += 1
        sched.step()

        val = evaluate(model, val_dl, criterion, device)
        row = {
            "epoch": epoch,
            "train_total": train_total / max(n_batches, 1),
            "sec": round(time.time() - t0, 1),
            **{f"val_{k}": round(v, 5) for k, v in val.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val["total"] < best_val:
            best_val = val["total"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_obs": train_ds.n_obs,
                    "n_depth": train_ds.n_depth,
                    "norm_stats": stats.to_dict(),
                    "periods": train_ds.periods.tolist(),
                    "depth_grid": train_ds.depth_grid.tolist(),
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    np.save(out_dir / "depth_grid.npy", train_ds.depth_grid)
    print(f"best val total: {best_val:.5f}")


if __name__ == "__main__":
    main()
