"""Quick visual check: sample a grid from a checkpoint.

    python sample.py runs/x/ckpt.pt --ema 0.9999 --sampler ddim --out grid.png
"""

import argparse

import torch
from torchvision.utils import save_image

from eval_fid import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ema", default="0.9999")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, diffusion, cfg, _ = load_model(args.ckpt, args.ema, device)
    gen = torch.Generator(device).manual_seed(args.seed)
    shape = (args.n, 3, cfg.img_res, cfg.img_res)
    if args.sampler == "ddpm":
        x = diffusion.p_sample_loop(model, shape, device, generator=gen, progress=True)
    else:
        x = diffusion.ddim_sample_loop(model, shape, device, steps=args.ddim_steps, generator=gen)
    save_image((x.clamp(-1, 1) + 1) / 2, args.out, nrow=8)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
