# DDPM-Diffusion-Model

A **single-GPU, pixel-space CIFAR-10 diffusion speedrun**: start from a faithful
from-scratch DDPM replication (Ho et al. 2020, FID 3.17), then modernize the recipe
one change at a time, measuring **wall-clock time-to-FID** on one consumer GPU
under a locked evaluation protocol. nanoGPT-speedrun spirit, diffusion edition.

- **[PROTOCOL.md](PROTOCOL.md)** — the benchmark rules (metric, clock, allowed moves)
- **[research-directions.md](research-directions.md)** — the July 2026 survey that
  chose this direction (and the alternatives it rejected)

## Status

M1 baseline landed: a paper-faithful DDPM reproduces to legacy-TF FID **3.150**,
crossing the 3.17 target in **≤8.24 h** on one RTX 5090 (M2 Phase-0 time-to-FID
curve; 50k confirmation pending). Now building the M2 recipe ladder. Reference
hardware: RTX 5090 32GB.

| Milestone | What | Status |
|---|---|---|
| M0 | Code skeleton, locked eval protocol, smoke tests | ✅ |
| M1 | Faithful DDPM baseline entry: target FID ≈ 3.17 | ✅ FID 3.150 |
| M2 | The ladder: one change per rung, time-to-FID curves | 🔨 |
| M3 | Public leaderboard + reproduction scripts | — |

## Leaderboard (protocol v0.2)

Wall-clock training time to reach each legacy-TF FID target (min of 3 seeds).

| Entry | →5.0 | →3.17 | →2.5 | →2.0 | Final FID (min-3) | Hardware |
|---|---|---|---|---|---|---|
| DDPM (paper-faithful) | ~2.4 h † | ≤8.24 h † | not reached | not reached | **3.150** | RTX 5090 |

† Located from a 10k-sample proxy sweep across snapshots (`M2 Phase-0`, zero
retrain); 50k/3-seed record confirmation of the 600k crossing is pending. The
≤8.24 h is high-confidence: the 600k proxy FID (5.132) ties 800k's (5.140), and
800k's true 50k FID is the record 3.150 — so 600k also clears 3.17, i.e. the old
"10.98 h" was the 800k *upper bound*, not the crossing.

Baseline detail: 35.75M U-Net, linear-β, ε-prediction, DDPM-1000 ancestral
sampling, EMA 0.99995. Final min-of-3 legacy-TF FID {3.150, 3.182, 3.164} at 800k;
clean-FID 3.835. Matches the paper's 3.17 anchor (community PyTorch reproduction:
3.188). **2.5/2.0 are unreachable by this recipe** — FID is flat past 600k, so
they require an M2 recipe change (EDM rung), not more training.

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
