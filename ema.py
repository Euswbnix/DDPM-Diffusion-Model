"""Multi-decay EMA: track several EMA copies of the model simultaneously.

Rationale (PROTOCOL.md rule 5): FID is markedly sensitive to EMA length
(Karras et al. 2024, EDM2). Tracking k decay rates costs k extra weight copies
per step — cheap, on-clock — and lets every checkpoint be evaluated at
multiple EMA lengths off-clock without EDM2's post-hoc machinery.

Implementation note: updates use torch._foreach_lerp_ over flat tensor lists
cached at init (one fused kernel per decay instead of one kernel per tensor —
~3x faster, verified bit-identical to per-tensor lerp_). The cached source
list aliases the live parameters, which optimizers update in place.
"""

import torch
import torch.nn as nn


class MultiEMA:
    def __init__(self, model: nn.Module, decays=(0.999, 0.9995, 0.9999, 0.99995)):
        self.decays = tuple(decays)
        sd = model.state_dict()
        # this UNet is all-float; fail loudly if that ever changes (e.g. BN stats)
        assert all(v.dtype.is_floating_point for v in sd.values()), (
            "MultiEMA assumes an all-float state_dict"
        )
        self._names = list(sd.keys())
        self._src = [sd[k] for k in self._names]  # aliases live params
        self.shadow = {
            d: [v.detach().clone() for v in self._src] for d in self.decays
        }

    @torch.no_grad()
    def update(self, model: nn.Module = None):
        for d in self.decays:
            torch._foreach_lerp_(self.shadow[d], self._src, 1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module, decay: float):
        model.load_state_dict(dict(zip(self._names, self.shadow[decay])))

    def state_dict(self):
        return {
            repr(d): dict(zip(self._names, self.shadow[d])) for d in self.decays
        }

    def load_state_dict(self, state, device=None):
        for d in self.decays:
            src = state[repr(d)]
            for name, v in zip(self._names, self.shadow[d]):
                v.copy_(src[name].to(v.device))
