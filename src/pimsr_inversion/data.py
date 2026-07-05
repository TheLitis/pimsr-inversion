"""HDF5 dataset loading and feature normalisation.

Reads the files produced by ``pimsr-forward-dataset``:

  observables : obs_mt_log10_rho, obs_mt_phase, obs_gravity
  targets     : target_log10_res, target_density, scenario
  axes        : periods, depth_grid, grav_offsets

Normalisation statistics are computed on the training split and stored with
the checkpoint so that evaluation/real-data inference uses identical scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DENSITY_SCALE = 1000.0  # kg/m^3 -> unitless order-1 values
PHASE_SCALE = 45.0  # degrees; half-space phase, natural unit


@dataclass
class NormStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray

    def to_dict(self) -> dict:
        return {"obs_mean": self.obs_mean.tolist(), "obs_std": self.obs_std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(np.asarray(d["obs_mean"]), np.asarray(d["obs_std"]))


def grid_cell_thicknesses(depth_grid: np.ndarray) -> np.ndarray:
    """Interpret grid nodes as cell centres of a layered model for the
    differentiable forward: n-1 finite thicknesses + implicit half-space."""
    edges = np.sqrt(depth_grid[:-1] * depth_grid[1:])  # geometric midpoints
    edges = np.concatenate([[0.0], edges])
    return np.diff(edges)


def _observable_matrix(f: h5py.File) -> np.ndarray:
    return np.concatenate(
        [
            f["obs_mt_log10_rho"][:],
            f["obs_mt_phase"][:] / PHASE_SCALE,
            f["obs_gravity"][:],
        ],
        axis=1,
    ).astype(np.float32)


def compute_norm_stats(train_path: str | Path) -> NormStats:
    with h5py.File(train_path, "r") as f:
        obs = _observable_matrix(f)
    return NormStats(obs.mean(axis=0), obs.std(axis=0) + 1e-6)


class PimsrDataset(Dataset):
    """In-memory dataset (MVP-1 files are ~100 MB; keep it simple and fast)."""

    def __init__(self, path: str | Path, stats: NormStats) -> None:
        with h5py.File(path, "r") as f:
            obs = _observable_matrix(f)
            self.obs = (obs - stats.obs_mean.astype(np.float32)) / stats.obs_std.astype(
                np.float32
            )
            self.log_rho = f["target_log10_res"][:].astype(np.float32)
            self.density = (
                (f["target_density"][:] - 2670.0) / DENSITY_SCALE
            ).astype(np.float32)
            self.scenario = f["scenario"][:].astype(np.int64)
            # Unnormalised observed MT response for the physics loss.
            self.obs_log_rho_a = f["obs_mt_log10_rho"][:].astype(np.float32)
            self.obs_phase = f["obs_mt_phase"][:].astype(np.float32)
            self.periods = f["periods"][:].astype(np.float64)
            self.depth_grid = f["depth_grid"][:].astype(np.float64)

    @property
    def n_obs(self) -> int:
        return self.obs.shape[1]

    @property
    def n_depth(self) -> int:
        return self.log_rho.shape[1]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.obs[i]),
            "log_rho": torch.from_numpy(self.log_rho[i]),
            "density": torch.from_numpy(self.density[i]),
            "scenario": torch.tensor(self.scenario[i]),
            "obs_log_rho_a": torch.from_numpy(self.obs_log_rho_a[i]),
            "obs_phase": torch.from_numpy(self.obs_phase[i]),
        }
