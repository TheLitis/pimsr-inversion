"""Self-supervised fine-tuning on real MT transfer functions.

No ground-truth resistivity exists for field stations, so the only training
signal is the physics itself: forward-model the predicted profile and match
the *measured* response (masked to each station's period band, static-shift
invariant). An L2-SP anchor to the pretrained weights prevents catastrophic
forgetting of the synthetic prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import PHASE_SCALE, NormStats
from .network import PimsrNet
from .physics import grid_to_layers, mt1d_response_torch

__all__ = ["finetune"]


def _masked_physics_misfit(
    pred_log10_res: torch.Tensor,
    obs_log_rho_a: torch.Tensor,
    obs_phase: torch.Tensor,
    mask: torch.Tensor,
    thicknesses: torch.Tensor,
    periods: torch.Tensor,
) -> torch.Tensor:
    """Shift-invariant chi^2 restricted to in-band periods per station."""
    sim_lr, sim_ph = mt1d_response_torch(pred_log10_res, thicknesses, periods)
    m = mask.to(torch.float64)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)

    d_lr = sim_lr - obs_log_rho_a.to(torch.float64)
    d_lr = d_lr - (d_lr * m).sum(dim=1, keepdim=True) / n  # static shift out
    d_ph = (sim_ph - obs_phase.to(torch.float64)) / PHASE_SCALE

    per_station = ((d_lr.square() + d_ph.square()) * m).sum(dim=1) / n.squeeze(1)
    return per_station.mean().to(pred_log10_res.dtype)


def finetune(
    checkpoint: str | Path,
    real_npz: str | Path,
    out: str | Path,
    steps: int = 400,
    lr: float = 3.0e-5,
    anchor_weight: float = 1.0,
    device: str | None = None,
) -> dict:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)

    model = PimsrNet(
        n_obs=int(ckpt["n_obs"]),
        n_depth=int(ckpt["n_depth"]),
        n_scenarios=int(ckpt.get("n_scenarios", 4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).train()
    anchor = {k: v.detach().clone() for k, v in model.named_parameters()}

    stats = NormStats.from_dict(ckpt["norm_stats"])
    periods = torch.tensor(np.asarray(ckpt["periods"]), dtype=torch.float64)
    depth_grid = np.asarray(ckpt["depth_grid"])
    thicknesses = torch.tensor(
        grid_to_layers(torch.tensor(depth_grid)).numpy(), dtype=torch.float64
    )

    data = np.load(real_npz, allow_pickle=True)
    lr_a, ph, mask = data["log_rho_a"], data["phase"], data["mask"]
    n_periods = periods.numel()
    grav_fill = stats.obs_mean[2 * n_periods :]
    obs = np.concatenate(
        [lr_a, ph / PHASE_SCALE, np.tile(grav_fill, (lr_a.shape[0], 1))], axis=1
    ).astype(np.float32)
    obs = (obs - stats.obs_mean.astype(np.float32)) / stats.obs_std.astype(np.float32)

    x = torch.from_numpy(obs).to(dev)
    t_lr = torch.from_numpy(lr_a).to(dev)
    t_ph = torch.from_numpy(ph).to(dev)
    t_mask = torch.from_numpy(mask).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        opt.zero_grad()
        out_dict = model(x)
        phys = _masked_physics_misfit(
            out_dict["log_rho"], t_lr, t_ph, t_mask, thicknesses, periods
        )
        reg = sum(
            (p - anchor[k]).square().sum() for k, p in model.named_parameters()
        )
        loss = phys + anchor_weight * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            p, a = float(phys.detach()), float(reg.detach())
            history.append({"step": step, "physics": p, "anchor": a})
            print(f"step {step}: physics={p:.4f} anchor={a:.5f}")

    ckpt["model_state"] = model.state_dict()
    ckpt["finetune"] = {
        "steps": steps,
        "lr": lr,
        "anchor_weight": anchor_weight,
        "history": history,
    }
    torch.save(ckpt, out)
    return {"final_physics": history[-1]["physics"], "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--real-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    args = ap.parse_args()
    result = finetune(
        args.checkpoint,
        args.real_npz,
        args.out,
        steps=args.steps,
        lr=args.lr,
        anchor_weight=args.anchor_weight,
    )
    print(json.dumps({"final_physics": result["final_physics"]}))


if __name__ == "__main__":
    main()
