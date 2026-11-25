"""在同一份权重上切换语言模型的注意力实现，并统计实际调用次数（证明切换生效）。"""
from __future__ import annotations

import sys

import torch


class AttnSwitch:
    """loaded：加载时的实现（本仓库为 eager）；"sdpa"：torch SDPA。calls 只统计语言模型层里的调用；其余（视觉塔）记在 calls_other。"""

    def __init__(self, ad):
        self.cfgs = {}
        for m in ad.model.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                self.cfgs[id(c)] = c
        self.orig = {k: c._attn_implementation for k, c in self.cfgs.items()}
        self.calls = {"eager": 0, "sdpa": 0}
        self.calls_other = {"eager": 0, "sdpa": 0}
        self._in_lm = False
        mod = sys.modules[type(ad.lm_layers[0].self_attn).__module__]
        eager, sdpa = mod.eager_attention_forward, torch.nn.functional.scaled_dot_product_attention

        def eager_counted(*a, **k):
            self.calls["eager"] += self._in_lm
            self.calls_other["eager"] += not self._in_lm
            return eager(*a, **k)

        def sdpa_counted(*a, **k):
            self.calls["sdpa"] += self._in_lm
            self.calls_other["sdpa"] += not self._in_lm
            return sdpa(*a, **k)

        mod.eager_attention_forward = eager_counted
        torch.nn.functional.scaled_dot_product_attention = sdpa_counted
        for l in ad.lm_layers:
            l.self_attn.register_forward_pre_hook(lambda m, i: setattr(self, "_in_lm", True))
            l.self_attn.register_forward_hook(lambda m, i, o: setattr(self, "_in_lm", False))

    def use(self, impl: str):
        for k, c in self.cfgs.items():
            c._attn_implementation = self.orig[k] if impl == "eager" else "sdpa"
        self.calls.update(eager=0, sdpa=0)
        self.calls_other.update(eager=0, sdpa=0)
