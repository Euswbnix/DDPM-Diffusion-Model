# DDPM-Diffusion-Model

A **single-GPU, pixel-space CIFAR-10 diffusion speedrun**: start from a faithful
from-scratch DDPM replication (Ho et al. 2020, FID 3.17), then modernize the recipe
one change at a time, measuring **wall-clock time-to-FID** on one consumer GPU
under a locked evaluation protocol. nanoGPT-speedrun spirit, diffusion edition.

- **[PROTOCOL.md](PROTOCOL.md)** — the benchmark rules (metric, clock, allowed moves)
- **[research-directions.md](research-directions.md)** — the July 2026 survey that
  chose this direction (and the alternatives it rejected)

## Status

M0 (infrastructure) in progress. Reference hardware: RTX 5090 32GB.

| Milestone | What | Status |
|---|---|---|
| M0 | Code skeleton, locked eval protocol, smoke tests | 🔨 |
| M1 | Faithful DDPM baseline entry: target FID ≈ 3.17 | — |
| M2 | The ladder: one change per rung, time-to-FID curves | — |
| M3 | Public leaderboard + reproduction scripts | — |

## Leaderboard (protocol v0.1)

| Entry | Time to FID≤5.0 | ≤3.17 | ≤2.5 | ≤2.0 | Final FID (min of 3) | Hardware |
|---|---|---|---|---|---|---|
| DDPM (paper-faithful) | — | — | — | — | — | RTX 5090 |

## Layout

```
unet.py        DDPM U-Net (35.7M, faithful to Ho et al. 2020)
diffusion.py   linear-beta DDPM + DDIM samplers
train.py       training loop; the speedrun clock lives here
ema.py         multi-decay EMA (post-hoc EMA-length selection, disclosed per PROTOCOL)
eval_fid.py    off-clock clean-fid evaluation (record + dev-proxy modes)
sample.py      sample grids for eyeballing
data.py        CIFAR-10 -> [-1,1], hflip
config.py      yaml config over paper-faithful defaults
configs/       run configs (ddpm_cifar10 = M1 baseline, smoke = CI-sized)
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py --config configs/smoke.yaml          # ~1 min end-to-end check
python train.py --config configs/ddpm_cifar10.yaml   # the M1 baseline (~20h on a 5090)
python eval_fid.py runs/ddpm_cifar10_baseline/snap_0800000.pt --ema 0.9999   # record eval
python sample.py runs/ddpm_cifar10_baseline/ckpt.pt --out grid.png
```
