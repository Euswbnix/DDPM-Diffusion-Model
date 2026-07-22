"""Training loop with the speedrun clock built in.

The clock (PROTOCOL.md v0.2): wall time from entry into main() to the
initiation of each checkpoint save, accumulated across resumes. Model build,
CUDA/compile warmup, data loading and EMA updates are all on-clock; the
explicit dataset download/verification pre-pass and all evaluation are
off-clock. (Resume does not restore RNG or dataloader position — accepted:
the loss is an expectation over fresh (t, eps, batch) draws either way.)

Usage:
    python train.py --config configs/ddpm_cifar10.yaml
    python train.py --config configs/ddpm_cifar10.yaml --resume runs/x/ckpt.pt
"""

import argparse
import json
import os
import time

import torch

from config import load_config
from data import cifar10_loader, ensure_cifar10, infinite
from diffusion import GaussianDiffusion
from ema import MultiEMA
from unet import UNet


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_ckpt(path, model, opt, ema, step, elapsed, cfg):
    tmp = path + ".tmp"
    torch.save(
        dict(
            model=model.state_dict(),
            opt=opt.state_dict(),
            ema=ema.state_dict(),
            step=step,
            elapsed=elapsed,
            config=vars(cfg),
        ),
        tmp,
    )
    os.replace(tmp, path)


def main():
    t0 = time.perf_counter()  # ---- the clock starts at process entry (v0.2)
    off_clock = 0.0

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    t_dl = time.perf_counter()  # dataset download/verify pass is off-clock
    ensure_cifar10(cfg.data_dir)
    off_clock += time.perf_counter() - t_dl

    device = pick_device()
    is_cuda = device.type == "cuda"
    torch.manual_seed(cfg.seed)
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)
    log_f = open(os.path.join(cfg.run_dir, "log.jsonl"), "a")

    model = UNet(
        ch=cfg.ch,
        ch_mult=tuple(cfg.ch_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_res=tuple(cfg.attn_res),
        dropout=cfg.dropout,
        img_res=cfg.img_res,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    diffusion = GaussianDiffusion(cfg.T, cfg.beta_start, cfg.beta_end).to(device)
    ema = MultiEMA(model, cfg.ema_decays)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    step, elapsed_prev = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        step, elapsed_prev = ck["step"], ck["elapsed"]
        print(f"resumed from {args.resume} @ step {step}, elapsed {elapsed_prev:.0f}s")

    if cfg.compile and is_cuda:
        torch._dynamo.config.recompile_limit = 64  # one ResAttnBlock code object, many shapes
        # torch 2.10.0's inductor mix-order-reduction kernel silently corrupted
        # conv-bias grads (NaN within a few steps) under bf16 autocast on sm_120
        # (https://github.com/pytorch/pytorch/issues/190796). Fixed by requiring
        # torch >= 2.11; on 2.10.x set TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 (the
        # old AttnBlock-eager workaround only sidestepped the bad fusion).
        fwd = torch.compile(model)
    else:
        fwd = model

    loader = cifar10_loader(
        cfg.batch_size, cfg.num_workers, cfg.data_dir, cfg.hflip, is_cuda
    )
    batches = infinite(loader)
    print(f"device={device} params={n_params/1e6:.2f}M steps={cfg.total_steps}")

    clock = lambda: elapsed_prev + (time.perf_counter() - t0) - off_clock
    loss_acc = torch.zeros((), device=device)  # GPU-side; no per-step .item() sync
    n_acc, t_log = 0, time.perf_counter()

    while step < cfg.total_steps:
        for g in opt.param_groups:  # linear LR warmup (paper: 5k steps)
            g["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup)

        x = next(batches).to(device, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp_bf16 and is_cuda):
            loss = diffusion.loss(fwd, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(model)
        step += 1
        loss_acc += loss.detach()
        n_acc += 1

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            rec = dict(
                step=step,
                loss=round(loss_acc.item() / n_acc, 5),
                imgs_per_sec=round(cfg.log_every * cfg.batch_size / (now - t_log), 1),
                elapsed=round(clock(), 1),
                lr=opt.param_groups[0]["lr"],
            )
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            loss_acc.zero_()
            n_acc, t_log = 0, now

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            save_ckpt(
                os.path.join(cfg.run_dir, "ckpt.pt"),
                model, opt, ema, step, clock(), cfg,
            )
        if step % cfg.snapshot_every == 0 or step == cfg.total_steps:
            # EMA-only snapshot for off-clock evaluation; clock stamp inside.
            tmp = os.path.join(cfg.run_dir, f"snap_{step:07d}.pt.tmp")
            torch.save(
                dict(ema=ema.state_dict(), step=step, elapsed=clock(), config=vars(cfg)),
                tmp,
            )
            os.replace(tmp, tmp[:-4])

    print(f"done: {step} steps in {clock():.0f}s on-clock")


if __name__ == "__main__":
    main()
