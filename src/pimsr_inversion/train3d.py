"""Resumable mixed-precision training contract for PIMSR 3D."""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .network3d import Model3DConfig, PimsrNet3D, estimate_activation_bytes


class Volume3DDataset(Dataset):
    """Load atomic per-sample HDF5 files without retaining volumes in RAM."""

    def __init__(self, root):
        self.paths = sorted(Path(root).glob("*.h5"))
        if not self.paths:
            raise ValueError(f"no 3D HDF5 samples in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with h5py.File(self.paths[index]) as f:
            rho = np.log10(f["observations/apparent_resistivity"][:].astype(np.float32))
            phase = f["observations/phase"][:].astype(np.float32) / 45.0
            # (F,M,Y,X) -> channels=(M*rho,M*phase), F,Y,X
            obs = np.concatenate((rho.transpose(1, 0, 2, 3), phase.transpose(1, 0, 2, 3)), axis=0)
            target = f["target/log10_resistivity"][:].astype(np.float32)
        return torch.from_numpy(obs), torch.from_numpy(target)


def train(args) -> dict:
    config = Model3DConfig.preset(args.preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Volume3DDataset(args.data)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.workers)
    sample, _ = dataset[0]
    model = PimsrNet3D(sample.shape[0], config.width, checkpoint_blocks=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "fp16")
    start_epoch = 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume) if args.resume else out / "last3d.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1

    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for obs, target in loader:
            obs, target = obs.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
            with amp:
                # Presets cap the production target volume; small smoke datasets
                # retain their native shape. The full survey cube remains input.
                output_shape = tuple(min(a, b) for a, b in zip(target.shape[-3:], config.crop))
                if output_shape != target.shape[-3:]:
                    target = torch.nn.functional.interpolate(
                        target.unsqueeze(1), size=output_shape, mode="trilinear", align_corners=False
                    ).squeeze(1)
                pred = model(obs, output_shape=output_shape)
                residual = pred["log_rho"] - target
                loss = 0.5 * (
                    pred["log_sigma_rho"]
                    + residual.square() * torch.exp(-pred["log_sigma_rho"])
                ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(obs)
        scheduler.step()
        history.append({"epoch": epoch, "loss": total / len(dataset)})
        state = {
            "epoch": epoch, "preset": args.preset, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "history": history,
        }
        torch.save(state, out / "last3d.pt")
    (out / "history3d.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"preset": args.preset, "estimated_activation_bytes": estimate_activation_bytes(config), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", default="a100-40gb", choices=("local-8gb", "fallback-24gb", "a100-40gb", "a100-80gb"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
