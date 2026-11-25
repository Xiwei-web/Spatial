"""与模型家族无关的干预工具：都作用在语言模型的 decoder 层上。

约定（由各 adapter 保证）：
- layers[i] 是第 i 个 decoder 层；其前向输出的第 0 项（或输出本身）是残差流 [B, T, D]
- 标准自注意力层有 `self_attn` 子模块，且 `self_attn.o_proj` 的输入是按 head 拼接的
  [B, T, n_heads * head_dim]（Qwen2/Qwen3/LLaMA/Gemma 系实现均如此；adapter 加载后会做形状校验）
- 自注意力模块以关键字参数 `attention_mask` 接收 4D 掩码 [B, 1, Tq, Tk]（加性浮点或布尔）
所有 hook 都通过上下文管理器注册，退出时移除，不修改模型权重。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch
from torch import nn


def find_attn(layer: nn.Module) -> nn.Module:
    """层内的 softmax 自注意力模块：优先 `self_attn`，否则取唯一一个同时有 q_proj 和 o_proj 的子模块。"""
    m = getattr(layer, "self_attn", None)
    if m is not None and hasattr(m, "o_proj"):
        return m
    cands = [c for _, c in layer.named_children() if hasattr(c, "o_proj") and hasattr(c, "q_proj")]
    if len(cands) != 1:
        raise RuntimeError(f"该层没有可识别的自注意力模块（子模块：{[n for n, _ in layer.named_children()]}）")
    return cands[0]


def _hidden_of(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    if isinstance(output, list):
        return [new] + list(output[1:])
    return new


@contextmanager
def capture_residual(layers: nn.ModuleList, layer_ids: Iterable[int], store: dict):
    """把指定层输出的残差流存进 store[layer_id]（detach 后的克隆）。不改变前向结果。"""
    handles = []
    for i in layer_ids:
        def hook(_m, _a, out, i=i):
            store[i] = _hidden_of(out).detach().clone()
        handles.append(layers[i].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def noop_hooks(layers: nn.ModuleList):
    """在每一层注册“什么都不改”的 hook（层输出 hook + o_proj 输入 hook）。
    用于验证：带 hook 的前向与原模型逐位一致。"""
    handles = []
    for layer in layers:
        handles.append(layer.register_forward_hook(lambda m, a, out: None))
        for _, child in layer.named_children():      # 所有带 o_proj 的混合模块（含非标准层）
            if hasattr(child, "o_proj"):
                handles.append(child.o_proj.register_forward_pre_hook(lambda m, a: None))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_residual(layers: nn.ModuleList, layer_id: int, positions, value: torch.Tensor):
    """activation patching：把第 layer_id 层输出的残差流在 positions 处替换为 value[:, positions]。
    value 形状 [B, T, D]（通常来自另一次运行的 capture_residual）。"""
    pos = torch.as_tensor(list(positions), dtype=torch.long)

    def hook(_m, _a, out):
        h = _hidden_of(out).clone()
        p = pos.to(h.device)
        h[:, p] = value.to(h.device, h.dtype)[:, p]
        return _replace_hidden(out, h)

    handle = layers[layer_id].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def head_output_hook(layers: nn.ModuleList, layer_id: int, n_heads: int,
                     fn: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
                     store: dict | None = None):
    """取到 / 改写单个 head 的输出（o_proj 之前、按 head 拆开的 [B, T, n_heads, head_dim]）。
    fn 接收该张量，返回 None 表示只读，返回张量表示改写。"""
    o_proj = find_attn(layers[layer_id]).o_proj

    def pre_hook(_m, args):
        x = args[0]
        b, t, d = x.shape
        if d % n_heads != 0:
            raise RuntimeError(f"o_proj 输入维度 {d} 不能被 n_heads={n_heads} 整除")
        per_head = x.view(b, t, n_heads, d // n_heads)
        if store is not None:
            store["per_head"] = per_head.detach().clone()
        if fn is not None:
            new = fn(per_head.clone())
            if new is not None:
                return (new.reshape(b, t, d),) + tuple(args[1:])
        return None

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def attention_knockout(layers: nn.ModuleList, layer_ids: Iterable[int], query_pos, key_pos,
                       report: dict | None = None):
    """屏蔽注意力：在指定层禁止 query_pos 处的 token 关注 key_pos 处的 token。
    通过改写传给自注意力模块的 4D attention_mask 实现（布尔掩码置 False，加性掩码置 dtype 最小值）。
    若该模块没有以关键字收到 4D 掩码，直接报错而不是静默跳过。"""
    q = torch.as_tensor(list(query_pos), dtype=torch.long)
    k = torch.as_tensor(list(key_pos), dtype=torch.long)
    handles = []

    def pre_hook(_m, args, kwargs):
        mask = kwargs.get("attention_mask", None)
        if mask is None or mask.dim() != 4:
            raise RuntimeError(
                "attention_knockout: 自注意力模块没有收到 4D attention_mask "
                f"(got {None if mask is None else tuple(mask.shape)})；该实现需要专门适配")
        mask = mask.clone()
        if mask.shape[1] != 1:
            mask = mask.expand(-1, -1, -1, -1).clone()
        qq, kk = q.to(mask.device), k.to(mask.device)
        blocked = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
        mask[:, :, qq[:, None], kk[None, :]] = blocked
        kwargs["attention_mask"] = mask
        if report is not None:
            report["mask_dtype"] = str(mask.dtype)
            report["mask_shape"] = tuple(mask.shape)
        return args, kwargs

    for i in layer_ids:
        attn = find_attn(layers[i])
        handles.append(attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
