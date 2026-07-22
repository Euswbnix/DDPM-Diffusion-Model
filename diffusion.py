"""Gaussian diffusion: linear-beta DDPM (Ho et al. 2020) + DDIM sampler.

Conventions match the official implementation:
- T=1000, betas linear from 1e-4 to 0.02 (computed in float64, stored float32)
- epsilon-prediction, simple MSE loss, t ~ Uniform{0..T-1}
- ancestral sampling clips the predicted x0 to [-1, 1] each step
- "fixedlarge" variance (sigma_t^2 = beta_t) by default, as in the official
  CIFAR-10 FID 3.17 configuration; no noise is added at t=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return buf.gather(0, t).reshape(-1, 1, 1, 1)


class GaussianDiffusion(nn.Module):
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])

        reg = lambda name, val: self.register_buffer(name, val.float(), persistent=False)
        reg("betas", betas)
        reg("sqrt_alphas_bar", alphas_bar.sqrt())
        reg("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())
        reg("sqrt_recip_alphas_bar", (1.0 / alphas_bar).sqrt())
        reg("sqrt_recipm1_alphas_bar", (1.0 / alphas_bar - 1.0).sqrt())
        reg("posterior_var", betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        reg("posterior_mean_coef1", betas * alphas_bar_prev.sqrt() / (1.0 - alphas_bar))
        reg("posterior_mean_coef2", (1.0 - alphas_bar_prev) * alphas.sqrt() / (1.0 - alphas_bar))
        reg("alphas_bar", alphas_bar)

    # ------------------------------------------------------------- training
    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = _extract(self.sqrt_alphas_bar, t) * x0 + _extract(
            self.sqrt_one_minus_alphas_bar, t
        ) * noise
        return F.mse_loss(model(x_t, t), noise)

    # ------------------------------------------------------------- sampling
    def _predict_x0(self, x_t, t, eps, clip: bool):
        x0 = _extract(self.sqrt_recip_alphas_bar, t) * x_t - _extract(
            self.sqrt_recipm1_alphas_bar, t
        ) * eps
        return x0.clamp(-1.0, 1.0) if clip else x0

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape,
        device,
        var_type: str = "fixedlarge",
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Ancestral (DDPM) sampling, T model evaluations."""
        assert var_type in ("fixedlarge", "fixedsmall")
        x = torch.randn(shape, device=device, generator=generator)
        steps = reversed(range(self.T))
        if progress:
            from tqdm import tqdm

            steps = tqdm(steps, total=self.T, desc="ddpm sample")
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            x0 = self._predict_x0(x, t, eps, clip_denoised)
            mean = _extract(self.posterior_mean_coef1, t) * x0 + _extract(
                self.posterior_mean_coef2, t
            ) * x
            if i > 0:
                var = self.betas if var_type == "fixedlarge" else self.posterior_var
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + _extract(var, t).sqrt() * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape,
        device,
        steps: int = 100,
        eta: float = 0.0,
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDIM sampling (Song et al. 2021) on a uniform timestep subsequence."""
        seq = torch.linspace(0, self.T - 1, steps).round().long().unique().tolist()
        x = torch.randn(shape, device=device, generator=generator)
        for i in reversed(range(len(seq))):
            t_val = seq[i]
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps = model(x, t)
            x0 = self._predict_x0(x, t, eps, clip_denoised)
            # re-derive eps from the clipped x0 (as in openai/improved-diffusion)
            eps = (x - _extract(self.sqrt_alphas_bar, t) * x0) / _extract(
                self.sqrt_one_minus_alphas_bar, t
            )
            ab_prev = (
                self.alphas_bar[seq[i - 1]]
                if i > 0
                else torch.ones((), device=device)
            )
            ab_t = self.alphas_bar[t_val]
            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            # clamp: analytically >= 0 for eta <= 1, but float32 sqrt-then-square
            # rounding can push it ~1e-11 negative on big jumps (NaN without this)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * eps
            if eta > 0 and i > 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x
