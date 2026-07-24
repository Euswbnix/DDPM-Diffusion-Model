"""Off-clock FID evaluation (PROTOCOL.md v0.2).

Record mode: 50k samples x exactly 3 distinct seeds (any sampler), scored vs
CIFAR-10 train stats in BOTH clean-fid modes: legacy_tensorflow (the ranking
metric — comparable to the DDPM 3.17 / EDM 1.97 literature anchors) and
clean (published as disclosure). Reports all FIDs and the min.

Dev proxy: --n 10000 --seeds 0 --sampler ddim --ddim-steps 100

Usage:
    python eval_fid.py runs/x/snap_0400000.pt --ema 0.9999
    python eval_fid.py runs/x/snap_0400000.pt --ema 0.9999 --n 10000 --seeds 0 --sampler ddim
"""

import argparse
import json
import os
import shutil
import time

import torch
from torchvision.utils import save_image

from config import load_config
from diffusion import GaussianDiffusion
from ema import MultiEMA
from unet import UNet


def load_model(ckpt_path: str, ema_decay: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = load_config(None)
    vars(cfg).update(ck["config"])
    model = UNet(
        ch=cfg.ch,
        ch_mult=tuple(cfg.ch_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_res=tuple(cfg.attn_res),
        dropout=cfg.dropout,
        img_res=cfg.img_res,
    ).to(device)
    if ema_decay == "raw":
        if "model" not in ck:
            raise SystemExit(
                "snapshots store EMA weights only; use --ema <decay> or point at ckpt.pt"
            )
        model.load_state_dict(ck["model"])
    else:
        ema = MultiEMA(model, cfg.ema_decays)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model, float(ema_decay))
    model.eval()
    diffusion = GaussianDiffusion(cfg.T, cfg.beta_start, cfg.beta_end).to(device)
    return model, diffusion, cfg, ck


@torch.no_grad()
def generate(model, diffusion, cfg, n, batch, sampler, ddim_steps, seed, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device).manual_seed(seed)
    done = 0
    t_start = time.perf_counter()
    while done < n:
        b = min(batch, n - done)
        shape = (b, 3, cfg.img_res, cfg.img_res)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            if sampler == "ddpm":
                x = diffusion.p_sample_loop(model, shape, device, generator=gen)
            else:
                x = diffusion.ddim_sample_loop(
                    model, shape, device, steps=ddim_steps, generator=gen
                )
        x = (x.float().clamp(-1, 1) + 1) / 2
        for j in range(b):
            save_image(x[j], os.path.join(out_dir, f"{done + j:06d}.png"))
        done += b
        rate = done / (time.perf_counter() - t_start)
        print(f"  seed {seed}: {done}/{n} ({rate:.1f} img/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ema", default="0.9999", help="EMA decay to evaluate, or 'raw'")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--keep-samples", action="store_true")
    args = ap.parse_args()

    from cleanfid import fid  # heavy import; keep it here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, ck = load_model(args.ckpt, args.ema, device)
    seeds = [int(s) for s in args.seeds.split(",")]
    # PROTOCOL v0.2: exactly 3 distinct seeds (min-of-3); sampler is free
    record = args.n == 50_000 and len(seeds) == 3 and len(set(seeds)) == 3

    tag = os.path.basename(args.ckpt).removesuffix(".pt")

    # Per-seed resume: checkpoint each finished seed's FID next to the ckpt, so a
    # mid-run kill only costs the seed in flight. Re-running the same command
    # skips seeds already scored. (runs/ is gitignored, so these stay local.)
    def partial_path(seed):
        return args.ckpt.replace(
            ".pt", f"_fid_ema{args.ema}_{args.sampler}{args.n}_seed{seed}.partial.json"
        )

    scores, scores_clean = {}, {}
    for seed in seeds:
        pp = partial_path(seed)
        if os.path.exists(pp):
            cached = json.load(open(pp))
            scores[seed], scores_clean[seed] = cached["legacy"], cached["clean"]
            print(
                f"seed {seed}: resumed from checkpoint — "
                f"FID {scores[seed]:.4f} (legacy_tf) / {scores_clean[seed]:.4f} (clean)",
                flush=True,
            )
            continue
        out_dir = os.path.join(
            "samples", f"{tag}_ema{args.ema}_{args.sampler}_seed{seed}"
        )
        shutil.rmtree(out_dir, ignore_errors=True)
        try:
            generate(
                model, diffusion, cfg, args.n, args.batch, args.sampler,
                args.ddim_steps, seed, out_dir, device,
            )
            for mode, dst in (("legacy_tensorflow", scores), ("clean", scores_clean)):
                dst[seed] = fid.compute_fid(
                    out_dir,
                    dataset_name="cifar10",
                    dataset_res=32,
                    dataset_split="train",
                    mode=mode,
                )
            # checkpoint this seed atomically before the next one starts
            tmp = pp + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"legacy": scores[seed], "clean": scores_clean[seed]}, f)
            os.replace(tmp, pp)
            print(
                f"seed {seed}: FID {scores[seed]:.4f} (legacy_tf) / "
                f"{scores_clean[seed]:.4f} (clean)",
                flush=True,
            )
        finally:
            if not args.keep_samples:
                shutil.rmtree(out_dir, ignore_errors=True)

    result = dict(
        ckpt=args.ckpt,
        step=ck["step"],
        elapsed_on_clock_sec=ck["elapsed"],
        ema=args.ema,
        n=args.n,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps if args.sampler == "ddim" else None,
        fid_by_seed={str(k): round(v, 4) for k, v in scores.items()},
        fid_min=round(min(scores.values()), 4),
        fid_clean_by_seed={str(k): round(v, 4) for k, v in scores_clean.items()},
        fid_clean_min=round(min(scores_clean.values()), 4),
        record_protocol=record,
    )
    print(json.dumps(result, indent=2))
    # seed tag in the filename so split-seed runs don't overwrite each other
    seed_tag = "-".join(str(s) for s in seeds)
    out_json = args.ckpt.replace(
        ".pt", f"_fid_ema{args.ema}_{args.sampler}{args.n}_seed{seed_tag}.json"
    )
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    # final result persisted — drop the per-seed resume checkpoints
    for seed in seeds:
        try:
            os.remove(partial_path(seed))
        except OSError:
            pass
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
