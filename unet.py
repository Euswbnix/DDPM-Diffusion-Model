"""DDPM U-Net, faithful to Ho et al. 2020 (arXiv:2006.11239).

Matches the official TF implementation (hojonathanho/diffusion) for CIFAR-10:
ch=128, ch_mult=(1,2,2,2), 2 res blocks per resolution, self-attention at 16x16,
dropout 0.1, sinusoidal timestep embedding -> 2-layer MLP. ~35.7M parameters.

Known benign deviation from TF: stride-2 conv downsampling uses symmetric
padding=1 (PyTorch) instead of TF 'SAME' asymmetric padding; community PyTorch
reproductions (e.g. tqch/ddpm-torch, FID 3.188) use the same convention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embeddings as in the official DDPM code (and Transformer)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def zero_init(module: nn.Module) -> nn.Module:
    """Official DDPM initializes output layers to zero (init_scale=0.)."""
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions (official DDPM style)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(dim=1)
        # [B, 1, HW, C]; SDPA computes softmax(q k^T / sqrt(C)) v, identical to
        # the official einsum formulation but uses fused kernels.
        q, k, v = (a.transpose(1, 2).unsqueeze(1) for a in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class ResAttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, dropout, attn: bool):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, temb_ch, dropout)
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb):
        return self.attn(self.res(x, temb))


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        ch: int = 128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_res=(16,),
        dropout: float = 0.1,
        img_res: int = 32,
    ):
        super().__init__()
        self.ch = ch
        temb_ch = ch * 4
        self.temb = nn.Sequential(
            nn.Linear(ch, temb_ch), nn.SiLU(), nn.Linear(temb_ch, temb_ch)
        )
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        attn_res = set(attn_res)
        self.down_blocks = nn.ModuleList()
        skip_chs = [ch]
        cur_ch, cur_res = ch, img_res
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResAttnBlock(cur_ch, out_ch, temb_ch, dropout, cur_res in attn_res)
                )
                cur_ch = out_ch
                skip_chs.append(cur_ch)
            if i != len(ch_mult) - 1:
                self.down_blocks.append(Downsample(cur_ch))
                cur_res //= 2
                skip_chs.append(cur_ch)

        self.mid1 = ResAttnBlock(cur_ch, cur_ch, temb_ch, dropout, attn=True)
        self.mid2 = ResAttnBlock(cur_ch, cur_ch, temb_ch, dropout, attn=False)

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResAttnBlock(
                        cur_ch + skip_chs.pop(), out_ch, temb_ch, dropout,
                        cur_res in attn_res,
                    )
                )
                cur_ch = out_ch
            if i != 0:
                self.up_blocks.append(Upsample(cur_ch))
                cur_res *= 2
        assert not skip_chs

        self.norm_out = nn.GroupNorm(32, cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, in_ch, 3, padding=1)

        # Official init: variance_scaling(1, 'fan_avg', 'uniform') = Glorot/Xavier
        # uniform on every dense/conv weight, zero biases; then re-zero the
        # init_scale=0 output layers (PyTorch's default Kaiming-uniform has ~1/3
        # the official variance and nonzero biases — a fidelity gap).
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, ResBlock):
                zero_init(m.conv2)
            elif isinstance(m, AttnBlock):
                zero_init(m.proj)
        zero_init(self.conv_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.conv_in(x)
        hs = [h]
        for blk in self.down_blocks:
            h = blk(h) if isinstance(blk, Downsample) else blk(h, temb)
            hs.append(h)
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)
        for blk in self.up_blocks:
            if isinstance(blk, Upsample):
                h = blk(h)
            else:
                h = blk(torch.cat([h, hs.pop()], dim=1), temb)
        assert not hs
        return self.conv_out(F.silu(self.norm_out(h)))
