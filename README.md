# pimsr-inversion

Physics-informed neural inversion for the PIMSR project: maps multi-modal geophysical
observables (MT apparent resistivity + phase, relative gravity profile) to subsurface
property profiles with calibrated uncertainty.

## Architecture

- **Encoder** — per-modality 1D conv towers (MT curves, gravity profile) fused by an MLP trunk.
- **Heads** (multi-task):
  - `log10_res` — heteroscedastic Gaussian head (mean + log-variance per depth node) on the
    fixed 64-node log-depth grid;
  - `density` — same parameterization for bulk density;
  - `scenario` — 5-way classifier (background / aquifer / hydrocarbon / salt / geothermal).
- **Physics-informed loss** — a differentiable torch re-implementation of the exact 1D MT
  impedance recursion (validated against `pimsr-forward` to ~1e-6). The predicted resistivity
  profile is forward-modeled and penalized against the *observed* MT curves, closing the
  data-consistency loop that a pure regression model lacks.

## Loss

```
L = NLL(log10_res) + w_d * NLL(density) + w_s * CE(scenario) + w_p * MT_misfit(pred_profile, obs)
```

Heteroscedastic NLL yields per-depth uncertainty estimates; the MT misfit is normalized by
the sensor error floors from `pimsr-forward.sensors`.

## Training

Runs on the self-hosted GPU runner via the `pimsr-train` workflow in
[TheLitis/Runner](https://github.com/TheLitis/Runner):

```bash
pimsr-train --data-dir pimsr_data --out-dir runs/exp1 --epochs 40 --physics-weight 0.1
```

## Install

```bash
pip install -e .            # + torch installed separately for your CUDA version
```

## License

MIT
