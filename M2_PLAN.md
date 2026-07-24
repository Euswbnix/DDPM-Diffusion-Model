# M2: The Recipe Ladder

Designed 2026-07-23 (6-component parallel study + sequencing synthesis). Each rung
is a **single attributable change**, measured against the previous *kept* rung (a
reverted rung is not the next baseline). The ranked metric is **training wall-clock
to legacy-TF FID ≤ target**; sampling is off-clock.

## Mechanism classes

- **free-lever** — no training, no new code; bankable from existing M1 artifacts.
- **sampler-offclock** — lowers a *fixed* checkpoint's FID at eval time → an earlier
  snapshot crosses the target → lower time-to-FID with **zero** training.
- **train-dynamics** — changes FID-per-training-second (or seconds-per-step). Requires
  a retrain to measure.

## Phase 0 — bank the free / off-clock wins first (no retrain, fits ~1h shared windows)

These MUST land before any retrain: they **re-baseline the M1 leaderboard row** that
every later rung's delta is measured against. Skipping them overstates every gain ~25%.

- **[0A] time-to-FID curve on existing snapshots.** Re-score `snap_0*.pt` (EMA 0.99995,
  DDPM-1000). *Phase-1 locate* (10k/1-seed, a rank-preserving locator only): bisect
  {300,400,500,600}k for the 3.17 crossing and {150,200,250,300}k for 5.0.
  *Phase-2 confirm* (50k/3-seed, the record): the 1–2 snapshots bracketing each
  crossing. Turns time-to-3.17 from the **10.98h/800k upper bound → true ~8.24h @600k**
  (maybe ~6.9–7.6h @500–550k), adds the missing **time-to-5.0** (~4.8–5.5h @350–400k),
  and records **2.5/2.0 as NOT reached** by M1 (min 3.150).
  *STOP rule:* 3.150 vs 3.17 is inside the ~0.04 seed floor — once two consecutive
  snapshots straddle 3.17 within ±0.04, report the **bracket**, not a spurious step.
- **[0B] EMA-decay disclosure** (folded into 0A, zero extra runs): emit all 4 tracked
  decays' 50k FID at the crossing snapshot (PROTOCOL rule 5); label 0.99995-vs-0.9999
  "tied" if the gap ≤ the seed floor. *Sub-lever C (synthesize a longer 0.999975 decay)
  REJECTED* — a real longer decay needs an ~11h on-clock retrain; EDM2 post-hoc EMA
  won't help on a dataset as small as CIFAR (arXiv:2312.02696).
- **[1] sampler bet on frozen M1 weights** (add an EDM `D(x,σ)` wrapper + Heun /
  stochastic loops; screen with a **DDPM-1000/Heun 10k proxy, never DDIM**). This
  **gates the whole ladder**: deterministic DDIM/Heun cannot beat ~3.5–4.4 on
  ε/linear-beta weights (ODE floor > 3.17), so the only upside is a tuned stochastic
  sampler (~0–0.2 FID, crossing ~550k→~450k). Equally publishable **strong negative**:
  "no off-clock sampler beats stochastic DDPM-1000 → the 3.17 floor is a train-dynamics
  problem," which justifies the retrain ladder below.

## The retrain ladder (ascending cost / uncertainty)

| Rung | Change | Class | GPU cost | Isolates | Keep-if |
|---|---|---|---|---|---|
| 1 | Pure-systems bundle: `Adam(fused=True)`, channels_last audit, single-decay production run. **Math-identical to M1.** | train-dyn (throughput) | ~0.5–1h A/B (+~8h opt confirm) | seconds/step only → 8.24→~7.8–8.0h | imgs/s measurably up **and** loss curve overlays M1 |
| 2 | Batch 128→256, LR sqrt-scaled, **EMA co-swept** (hard confound). Objective/schedule/sampler = M1. | train-dyn (throughput) | ~4–6h proxy, ~18–22h record | imgs/s, steps-to-FID, wall-clock separately | reaches rung-1 FID in strictly less wall-clock; revert if equal-budget FID degrades (past B_crit) |
| 3 | Linear-β → **iDDPM cosine** ᾱ schedule (one buffer swap in `diffusion.py`). Everything else = M1. | train-dyn (schedule) | ~8h proxy, ~25h record | SNR-schedule effect alone | crosses 3.17 at lower wall-clock **and/or** final min-of-3 < prior floor |
| 4 | **Attribution control (dev-proxy, not a record):** ε vs v-const vs v-SNR-matched vs x0. | train-dyn (diagnostic) | ~15–20h proxy | whether ε/v/x0 are one objective under a weighting | keep standalone only if a v/x0 arm actually wins *and* parametrization (not smuggled weighting) is the cause — else **fold into rung 5** |
| 5 | **EDM objective**: preconditioning c_skip/c_out/c_in/c_noise (σ_data=0.5) + λ(σ) weighting + log-normal σ (P_mean −1.2, P_std 1.2). UNet reused. **Sampler held at M1.** | train-dyn (bundle) | ~35–55h incl. 2.0 push | the whole-objective train-dynamics delta | FID-neutral-or-better at matched sampler **and** crosses 3.17 in less wall-clock (target ≤0.75×) |
| 6 | **EDM Heun** (+ optional stochastic) on rung-5 snapshots. Zero retrain. | sampler-offclock | ~10h off-clock | incremental sampler-only gain | pushes a strictly earlier snapshot to each target |
| 7 | **Non-leaky augmentation** (augment-conditioned embedding). | train-dyn | ~8–12h/seed (+~30h record) | the augmentation regularizer alone | min-of-3 improves > the ~0.04 seed floor |

## Attribution firewalls (the benchmark's whole value)

- **Re-baseline first.** Measure every rung against the re-baselined ~8.24h M1, never
  the 10.98h upper bound. Read time-to-FID from each rung's **own** snapshot `elapsed`
  stamps (throughput rungs shift the axis).
- **Seed floor.** legacy-TF gaps < ~0.04 (~1.3% CoV at 3.17) are inconclusive; report
  min over exactly 3 seeds and bracket crossings.
- **EDM rung 5 bundles four coupled sub-levers** (parametrization + continuous-σ +
  λ-weighting + log-normal σ; c_out *defines* the weighting) → honestly **one** rung,
  not four. This is why rung 4's v-pred is a control, not a spent retrain. Hold the
  sampler fixed to separate train-dynamics from the Heun win; keep LR = M1 on the first
  alpha run so EDM's usual lr=1e-3 doesn't confound.
- **Never fold a sampler (Heun/stochastic) gain into a training rung's time-to-FID.**
- **No PROTOCOL bump needed** — rule 4 already permits schedule/loss/optimizer/arch/
  sampler/precision changes; the free-lever only fills existing target rows. A bump is
  required only if a change touches the clock, the ranked FID mode, the reference stats,
  or the 3-seed rule.

**Total:** ~140–220 GPU-wall-hours end-to-end, dominated by rung 5. The *ranked*
metric (on-clock training seconds) is a small fraction; most GPU-h are off-clock
sampling/FID and dev-proxy iteration. Every training rung is resumable → schedules into
the shared ~1h windows, except the >1h 50k-record evals (need negotiated blocks or a
resumable-generate patch).
