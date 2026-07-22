"""Tiny YAML config loader. Defaults are the paper-faithful DDPM CIFAR-10 setup."""

from types import SimpleNamespace

import yaml

DEFAULTS = dict(
    # model (Ho et al. 2020 CIFAR-10)
    ch=128,
    ch_mult=[1, 2, 2, 2],
    num_res_blocks=2,
    attn_res=[16],
    dropout=0.1,
    img_res=32,
    # diffusion
    T=1000,
    beta_start=1.0e-4,
    beta_end=0.02,
    # training (paper: batch 128, lr 2e-4, 800k steps, warmup 5k, clip 1.0)
    batch_size=128,
    lr=2.0e-4,
    warmup=5000,
    grad_clip=1.0,
    total_steps=800_000,
    ema_decays=[0.999, 0.9995, 0.9999, 0.99995],
    hflip=True,
    # systems
    compile=True,
    amp_bf16=True,
    num_workers=8,
    seed=0,
    # bookkeeping
    log_every=100,
    ckpt_every=20_000,
    snapshot_every=50_000,
    run_dir="runs/dev",
    data_dir="data",
)


def load_config(path: str | None) -> SimpleNamespace:
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        for k, v in overrides.items():
            # PyYAML (YAML 1.1) parses dot-less sci-notation ('2e-4') as a STRING
            if isinstance(DEFAULTS[k], float) and isinstance(v, str):
                overrides[k] = float(v)
            elif isinstance(DEFAULTS[k], int) and not isinstance(DEFAULTS[k], bool) and isinstance(v, str):
                overrides[k] = int(float(v))
        cfg.update(overrides)
    return SimpleNamespace(**cfg)
