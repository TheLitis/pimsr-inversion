# pimsr-inversion

Physics-informed neural inversion for the **PIMSR** project (Physics-Informed
Multi-modal Subsurface Reconstruction): maps magnetotelluric observables to
subsurface resistivity models with calibrated uncertainty — in milliseconds
instead of the seconds-to-minutes of classical iterative inversion.

**Headline result** (see [pimsr-benchmarks](https://github.com/TheLitis/pimsr-benchmarks)):
on 27 real USArray stations across 5 Yellowstone-region profiles, the 2D model
outperforms **both** production inversion codes on **every** profile —
mean 2D nRMS **4.30** vs 6.06 (Occam2DMT v3.0, Scripps) and 7.09 (ModEM NLCG) —
at roughly four orders of magnitude less compute per profile.

Part of the PIMSR platform:
[pimsr-geogen](https://github.com/TheLitis/pimsr-geogen) ·
[pimsr-forward](https://github.com/TheLitis/pimsr-forward) ·
pimsr-inversion ·
[pimsr-benchmarks](https://github.com/TheLitis/pimsr-benchmarks)

## Models

### 2D (primary): U-Net section inversion

- **Input** — 4-channel TE+TM pseudo-section `(log10 rho_a, phase/45)` per mode,
  24 periods x 16 stations.
- **Output heads** — resistivity section (48 x 64 log-depth grid) with a
  heteroscedastic sigma head, plus a 5-way scenario classifier
  (multiscale avg+max head over bottleneck + finest decoder features).
- **Losses** — beta-NLL (Seitzer 2022, `--beta 0.5`) for calibrated
  uncertainty, TV smoothness, class-weighted CE.
- **Physics fine-tuning** (`pimsr-finetune2d`) — masked shift-invariant
  chi-squared through a differentiable per-column 1D MT forward, with an
  L2-SP anchor; supports multi-profile joint adaptation (`--profiles`).
- **Post-hoc sigma calibration** (`pimsr-calibrate2d`) — affine correction of
  the log-sigma head fitted on the validation split.

### 1D (legacy): multi-task conv net

Per-station MT + gravity curves -> layered resistivity/density profile +
scenario. Kept for the 1D benchmark suite; superseded by the 2D model.

## Training

Runs on a self-hosted GPU runner via GitHub Actions workflows in
[TheLitis/Runner](https://github.com/TheLitis/Runner) — dataset generation
auto-triggers training on completion.

```bash
pimsr-train2d --data-dir pimsr_data2d --out runs/exp1 --epochs 80 \
    --beta 0.5 --scen-head multiscale
pimsr-finetune2d --checkpoint best2d.pt --emtf-dir data/emtf \
    --data-h5 ds2d_test.h5 --out best2d_ft.pt --steps 600 --anchor-weight 3
pimsr-calibrate2d --checkpoint best2d.pt --val-h5 ds2d_val.h5 --out best2d_cal.pt
```

## Install

```bash
pip install -e .            # + torch for your CUDA version
```

## License

MIT
