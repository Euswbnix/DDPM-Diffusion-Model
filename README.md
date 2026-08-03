# DDPM-Diffusion-Model

A **single-GPU, pixel-space CIFAR-10 diffusion speedrun**: start from a faithful
from-scratch DDPM replication (Ho et al. 2020, FID 3.17), then modernize the recipe
one change at a time, measuring **wall-clock time-to-FID** on one consumer GPU
under a locked evaluation protocol. nanoGPT-speedrun spirit, diffusion edition.

- **[PROTOCOL.md](PROTOCOL.md)** — the benchmark rules (metric, clock, allowed moves)
- **[research-directions.md](research-directions.md)** — the July 2026 survey that
  chose this direction (and the alternatives it rejected)

## Status

M1 baseline landed: a paper-faithful DDPM reaches legacy-TF FID **3.139**
(50k/3-seed, at 700k / 9.61 h) and crosses the 3.17 target in **≤9.61 h** on one
RTX 5090. The 50k confirmation replaced an earlier proxy-based ≤8.24 h estimate:
600k (8.24 h) actually scores 3.191 (misses), 700k (9.61 h) scores 3.139 (clears),
so the crossing sits in (8.24 h, 9.61 h] — and 700k is the best snapshot, training
past it doesn't help (800k is a hair worse at 3.150). Now building the M2 recipe
ladder. Reference hardware: RTX 5090 32GB.

| Milestone | What | Status |
|---|---|---|
| M0 | Code skeleton, locked eval protocol, smoke tests | ✅ |
| M1 | Faithful DDPM baseline entry: target FID ≈ 3.17 | ✅ FID 3.139 |
| M2 | The ladder: one change per rung, time-to-FID curves | 🔨 |
| M3 | Public leaderboard + reproduction scripts | — |

## Leaderboard (protocol v0.2)

Wall-clock training time to reach each legacy-TF FID target (min of 3 seeds).

| Entry | →5.0 | →3.17 | →2.5 | →2.0 | Final FID (min-3) | Hardware |
|---|---|---|---|---|---|---|
| DDPM (paper-faithful) | ~2.4 h † | ≤9.61 h | not reached | not reached | **3.139** | RTX 5090 |

† →5.0 is located from a 10k-sample proxy sweep (`M2 Phase-0`, zero retrain) and is
not yet 50k-confirmed. →3.17 **is** 50k/3-seed confirmed across three snapshots:
600k (8.24 h) scores min-of-3 **3.191** — all three seeds above target, misses;
700k (9.61 h) scores **3.139** — clears; 800k (10.98 h) scores 3.150. So the
crossing is bracketed in (8.24 h, 9.61 h], and 700k is the best snapshot — training
past it doesn't help (800k is marginally worse). This replaced an earlier ≤8.24 h
estimate: the 10k proxy tie (600k 5.132 ≈ 800k 5.140) did **not** survive to the
50k metric — a textbook case of the proxy being a biased *locator*, not a record.

Baseline detail: 35.75M U-Net, linear-β, ε-prediction, DDPM-1000 ancestral
sampling, EMA 0.99995. Best min-of-3 legacy-TF FID {3.1385, 3.2149, 3.2275} =
**3.139** at 700k (9.61 h); clean-FID 3.805. Matches the paper's 3.17 anchor
(community PyTorch reproduction: 3.188). **2.5/2.0 are unreachable by this recipe**
— FID plateaus at ≈3.14 by 700k (600k 3.191 → 700k 3.139 → 800k 3.150, i.e. flat
within seed noise past 700k), so they require an M2 recipe change (EDM rung), not
more training.

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
python eval_fid.py runs/ddpm_cifar10_baseline/snap_0700000.pt --ema 0.99995 --seeds 0,1,2   # record eval (best snapshot)
python sample.py runs/ddpm_cifar10_baseline/ckpt.pt --out grid.png
```
