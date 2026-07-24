# DDPM-Diffusion-Model

A **single-GPU, pixel-space CIFAR-10 diffusion speedrun**: start from a faithful
from-scratch DDPM replication (Ho et al. 2020, FID 3.17), then modernize the recipe
one change at a time, measuring **wall-clock time-to-FID** on one consumer GPU
under a locked evaluation protocol. nanoGPT-speedrun spirit, diffusion edition.

- **[PROTOCOL.md](PROTOCOL.md)** — the benchmark rules (metric, clock, allowed moves)
- **[research-directions.md](research-directions.md)** — the July 2026 survey that
  chose this direction (and the alternatives it rejected)

## Status

M1 baseline landed: a paper-faithful DDPM reproduces to legacy-TF FID **3.150**
in **10.98 h** on one RTX 5090. Now building the M2 recipe ladder. Reference
hardware: RTX 5090 32GB.

| Milestone | What | Status |
|---|---|---|
| M0 | Code skeleton, locked eval protocol, smoke tests | ✅ |
| M1 | Faithful DDPM baseline entry: target FID ≈ 3.17 | ✅ FID 3.150 |
| M2 | The ladder: one change per rung, time-to-FID curves | 🔨 |
| M3 | Public leaderboard + reproduction scripts | — |

## Leaderboard (protocol v0.2)

Ranked by legacy-TF FID (min of 3 seeds); clean-FID shown as disclosure.

| Entry | Wall-clock | Final FID (min of 3) | clean-FID | EMA | Hardware |
|---|---|---|---|---|---|
| DDPM (paper-faithful) | 10.98 h | **3.150** | 3.835 | 0.99995 | RTX 5090 |

Baseline detail: 35.75M U-Net, linear-β, ε-prediction, 800k steps, DDPM-1000
ancestral sampling. Per-seed legacy-TF FID {3.150, 3.182, 3.164}. Matches the
paper's 3.17 anchor (community PyTorch reproduction: 3.188). Convergence check:
FID is flat between the 600k and 800k snapshots, so the paper's 800k budget has
~25% slack — a lever for M2.

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
python train.py --config configs/ddpm_cifar10.yaml   # the M1 baseline (~11h on a 5090)
python eval_fid.py runs/ddpm_cifar10_baseline/snap_0800000.pt --ema 0.99995 --seeds 0,1,2   # record eval
python sample.py runs/ddpm_cifar10_baseline/ckpt.pt --out grid.png
```
