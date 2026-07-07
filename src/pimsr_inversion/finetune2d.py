"""Self-supervised fine-tuning of the conv-2D net on a real MT profile.

Mirrors the 1D recipe (`finetune.py`) that cut real nRMS by 27 %: the only
training signal is physics consistency — every station column of the
predicted section is re-simulated with the differentiable 1D forward and
matched to the measured transfer function (static-shift invariant, masked to
each station's period band). An L2-SP anchor to the pretrained weights
prevents catastrophic forgetting of the synthetic prior; it matters even
more here because a single profile is one training sample.

Small lateral jitter of the interpolated pseudo-section is applied per step
as a cheap augmentation against overfitting to interpolation artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch

from .network2d import PimsrNet2D
from .physics import mt1d_response_torch

__all__ = ["finetune2d", "build_profile_obs"]

PHASE_SCALE = 45.0


def build_profile_obs(
    emtf_dir: str,
    profile_ids: list[str],
    freqs: np.ndarray,
    station_x: np.ndarray,
    ref_lat_deg: float = 44.6,
) -> dict:
    """Interpolate a station profile onto the model's pseudo-section grid.

    Returns observation tensors on the (n_freq, n_stations) grid plus the
    in-band mask (nearest-station) used by the physics loss.
    """
    from pimsr_benchmarks.emtf import parse_emtf_xml, resample_station

    stations = {}
    for f in glob.glob(f"{emtf_dir}/*.xml"):
        st = parse_emtf_xml(f)
        stations[st.station_id] = st
    profile = [stations[i] for i in profile_ids]

    periods = 1.0 / freqs
    n_f, n_s = len(freqs), len(station_x)

    lon = np.array([s.longitude for s in profile])
    x_km = (lon - lon.min()) * 111.0 * np.cos(np.radians(ref_lat_deg))
    x_model = np.linspace(x_km.min(), x_km.max(), n_s)

    lr_st = np.empty((n_f, len(profile)))
    ph_st = np.empty((n_f, len(profile)))
    mask_st = np.empty((n_f, len(profile)), dtype=bool)
    for j, st in enumerate(profile):
        lr_st[:, j], ph_st[:, j], mask_st[:, j] = resample_station(st, periods)

    lr = np.stack([np.interp(x_model, x_km, lr_st[i]) for i in range(n_f)])
    ph = np.stack([np.interp(x_model, x_km, ph_st[i]) for i in range(n_f)])
    # nearest real station supplies the in-band mask for each model station
    nearest = np.array([int(np.argmin(np.abs(x_km - x))) for x in x_model])
    mask = mask_st[:, nearest]

    out = {
        "lr": lr, "ph": ph, "mask": mask,
        "x_model": x_model, "x_km": x_km, "periods": periods,
    }

    # per-mode observations for v3 4-channel models
    from pimsr_benchmarks.emtf import resample_station_modes

    for mode in ("te", "tm"):
        lr_m = np.empty((n_f, len(profile)))
        ph_m = np.empty((n_f, len(profile)))
        for j, st in enumerate(profile):
            m = resample_station_modes(st, periods)
            lr_m[:, j], ph_m[:, j] = m[f"lr_{mode}"], m[f"ph_{mode}"]
        out[f"lr_{mode}"] = np.stack(
            [np.interp(x_model, x_km, lr_m[i]) for i in range(n_f)]
        )
        out[f"ph_{mode}"] = np.stack(
            [np.interp(x_model, x_km, ph_m[i]) for i in range(n_f)]
        )
    return out


def _physics_misfit(
    section: torch.Tensor,
    obs_lr: torch.Tensor,
    obs_ph: torch.Tensor,
    mask: torch.Tensor,
    col_of_station: torch.Tensor,
    thicknesses: torch.Tensor,
    periods: torch.Tensor,
) -> torch.Tensor:
    """Masked, shift-invariant chi^2 over the station columns of a section.

    section : (n_depth, n_x) predicted log10 resistivity.
    obs_*   : (n_freq, n_stations) measured pseudo-section.
    """
    cols = section[:, col_of_station].T  # (n_stations, n_depth)
    sim_lr, sim_ph = mt1d_response_torch(cols, thicknesses, periods)

    m = mask.T.to(torch.float64)  # (n_stations, n_freq)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)

    d_lr = sim_lr - obs_lr.T.to(torch.float64)
    d_lr = d_lr - (d_lr * m).sum(dim=1, keepdim=True) / n  # static shift out
    d_ph = (sim_ph - obs_ph.T.to(torch.float64)) / PHASE_SCALE

    per_station = ((d_lr.square() + d_ph.square()) * m).sum(dim=1) / n.squeeze(1)
    return per_station.mean().to(section.dtype)


def finetune2d(
    checkpoint: str,
    emtf_dir: str,
    data_h5: str,
    out: str,
    profile_ids: list[str] | None = None,
    steps: int = 200,
    lr: float = 2.0e-5,
    anchor_weight: float = 10.0,
    jitter: float = 0.02,
    device: str | None = None,
) -> dict:
    import h5py

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)

    model = PimsrNet2D.from_checkpoint(ckpt)
    model.to(dev).train()
    anchor = {k: v.detach().clone() for k, v in model.named_parameters()}

    with h5py.File(data_h5, "r") as f:
        freqs = f["frequencies"][:]
        station_x = f["station_x"][:]
        x_grid = f["x_grid"][:]
        depth_grid = f["depth_grid"][:]

    if profile_ids is None:
        profile_ids = ["MTH15", "MTH16", "WYYS1", "WYYS2", "WYYS3", "WYH18", "WYH19"]
    prof = build_profile_obs(emtf_dir, profile_ids, freqs, station_x)

    # model stations sit at fixed fractions of the section width: map each
    # station index to its nearest x-grid column (same layout as training).
    sx_norm = (station_x - station_x.min()) / (station_x.max() - station_x.min())
    xg_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())
    col_of_station = torch.tensor(
        [int(np.argmin(np.abs(xg_norm - s))) for s in sx_norm], dtype=torch.long
    )

    if model.in_channels == 4:
        obs_np = np.stack(
            [prof["lr_te"], prof["ph_te"] / PHASE_SCALE,
             prof["lr_tm"], prof["ph_tm"] / PHASE_SCALE]
        )[None].astype(np.float32)
        # physics target: TE observations (1D column response equals both
        # modes for a layered column; TE keeps continuity with the 1D recipe)
        t_lr_np, t_ph_np = prof["lr_te"], prof["ph_te"]
    else:
        obs_np = np.stack(
            [prof["lr"], prof["ph"] / PHASE_SCALE]
        )[None].astype(np.float32)
        t_lr_np, t_ph_np = prof["lr"], prof["ph"]
    obs_np = (obs_np - ckpt["stats_mean"]) / ckpt["stats_std"]
    x = torch.from_numpy(obs_np).to(dev)

    t_lr = torch.from_numpy(t_lr_np).to(dev)
    t_ph = torch.from_numpy(t_ph_np).to(dev)
    t_mask = torch.from_numpy(prof["mask"]).to(dev)
    thick = torch.tensor(np.diff(depth_grid), dtype=torch.float64)
    periods_t = torch.tensor(prof["periods"], dtype=torch.float64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        opt.zero_grad()
        xin = x + jitter * torch.randn_like(x) if jitter > 0 else x
        out_dict = model(xin)
        phys = _physics_misfit(
            out_dict["log_rho"][0], t_lr, t_ph, t_mask,
            col_of_station, thick, periods_t,
        )
        reg = sum(
            (p - anchor[k]).square().sum() for k, p in model.named_parameters()
        )
        loss = phys + anchor_weight * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            p, a = float(phys.detach()), float(reg.detach())
            history.append({"step": step, "physics": p, "anchor": a})
            print(f"step {step}: physics={p:.4f} anchor={a:.5f}", flush=True)

    ckpt["model_state"] = model.state_dict()
    ckpt["finetune2d"] = {
        "steps": steps, "lr": lr, "anchor_weight": anchor_weight,
        "jitter": jitter, "profile": profile_ids, "history": history,
    }
    torch.save(ckpt, out)
    return {"final_physics": history[-1]["physics"], "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--emtf-dir", required=True)
    ap.add_argument("--data-h5", required=True, help="any 2D dataset split (for grids)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--anchor-weight", type=float, default=10.0)
    ap.add_argument("--jitter", type=float, default=0.02)
    args = ap.parse_args()
    result = finetune2d(
        args.checkpoint, args.emtf_dir, args.data_h5, args.out,
        steps=args.steps, lr=args.lr,
        anchor_weight=args.anchor_weight, jitter=args.jitter,
    )
    print(json.dumps({"final_physics": result["final_physics"]}))


if __name__ == "__main__":
    main()
