# Benchmark Protocol (v0.2, locked 2026-07-22)

The rules of the CIFAR-10 single-GPU pixel-space diffusion speedrun. Any change to
this file bumps the protocol version; records are only comparable within a version.

*v0.1 → v0.2 (pre-first-record, after adversarial review): ranking FID mode set to
`legacy_tensorflow` with `clean` as mandatory disclosure (v0.1 mixed clean-mode
scoring with legacy-TF literature anchors — measured ~0.09 systematic offset);
clock start/end pinned to the implementation; record seeds fixed at exactly 3.*

## Task

Unconditional generation of CIFAR-10 32×32 images, **in pixel space**.

## Metric

- FID via [`clean-fid`](https://github.com/GaParmar/clean-fid), **50,000 generated
  samples** vs the CIFAR-10 **train** split reference statistics
  (`dataset_name="cifar10", dataset_res=32, dataset_split="train"`).
- **Ranking metric: `mode="legacy_tensorflow"`** — the TF-Inception protocol used
  by the literature, so the DDPM 3.17 and EDM 1.97 anchors are directly
  comparable. **`mode="clean"` must also be computed and published** for every
  entry (disclosure; guards against resize-implementation artifacts).
- A record entry generates the 50k set with **exactly 3 distinct sampling seeds**,
  reports all FIDs in both modes, and is ranked by the **minimum** of the three
  legacy-TF scores (EDM convention).

## The clock

- The clock runs **from entry into the training process's `main()` to the
  initiation of the save** of the evaluated checkpoint. It accumulates across
  resume sessions (tracked inside checkpoints).
- **On the clock**: model construction, CUDA context and `torch.compile` warmup,
  data loading, EMA updates, and the write time of all *earlier* checkpoints.
- **Off the clock**: the explicit dataset download/verification pre-pass
  (`ensure_cifar10`, measured and subtracted), and all sample generation / FID
  computation (done afterwards from saved snapshots).
- Record hardware: a **single GPU**. The reference machine is one RTX 5090 32GB;
  entries on other single GPUs are welcome but listed with their hardware.

## Record targets

Wall-clock training time to reach legacy-TF FID ≤ **5.0**, ≤ **3.17** (the DDPM
paper anchor), ≤ **2.5**, ≤ **2.0**.

## Rules

1. **Data**: CIFAR-10 train split (50k images) only. Horizontal flips allowed. No
   external data, no other splits.
2. **No pretrained networks** anywhere in the model or training loss. (Inception /
   feature extractors appear only inside the off-clock FID computation.)
3. Pixel space only: no pretrained autoencoders or latent spaces.
4. Otherwise anything goes: architecture, optimizer, noise schedule, loss weighting,
   sampler and NFE, precision (bf16/fp16/fp32) for training and sampling,
   `torch.compile`. Sampler, NFE, and precision are disclosed in the entry JSON.
5. **EMA**: tracking multiple EMA decay rates during training is allowed (it is
   cheap and on-clock). Evaluating several decays off-clock and reporting the best
   is allowed **but must be disclosed**: report the chosen decay and the FID of
   every tracked decay.
6. Every entry must be reproducible from a config committed to this repo plus a
   documented command line, and must publish its training log (`log.jsonl`).

## Dev proxy (not comparable to records)

For iteration, use the cheap proxy: 10,000 samples, 1 seed, DDIM 100 steps.
Roughly tracks record FID but is biased; never report proxy numbers as records.
