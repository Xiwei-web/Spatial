"""在同一份权重上切换语言模型的注意力实现，并统计实际调用次数（证明切换生效）。"""
from __future__ import annotations

import sys

import torch


def attn_layer_indices(lm_layers) -> list[int]:
    """带 self_attn 子模块的语言模型层下标（升序）。"""
    return [i for i, l in enumerate(lm_layers) if getattr(l, "self_attn", None) is not None]


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
        self.attn_layer_idx = attn_layer_indices(ad.lm_layers)
        if not self.attn_layer_idx:
            raise RuntimeError("语言模型层里没有任何带 self_attn 的层，无法切换 / 统计注意力实现")
        mod = sys.modules[type(ad.lm_layers[self.attn_layer_idx[0]].self_attn).__module__]
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
        for i in self.attn_layer_idx:              # 只挂在带 self_attn 的层上（Qwen3.5 系 3/4 的层是 Gated DeltaNet，没有 self_attn）
            l = ad.lm_layers[i]
            l.self_attn.register_forward_pre_hook(lambda m, i: setattr(self, "_in_lm", True))
            l.self_attn.register_forward_hook(lambda m, i, o: setattr(self, "_in_lm", False))

    def use(self, impl: str):
        for k, c in self.cfgs.items():
            c._attn_implementation = self.orig[k] if impl == "eager" else "sdpa"
        self.calls.update(eager=0, sdpa=0)
        self.calls_other.update(eager=0, sdpa=0)
