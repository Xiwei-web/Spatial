"""干预正确性检查。tests/test_interventions.py（pytest）和 scripts/smoke_test.py 调用的是同一套函数，
判定标准只在这里定义一次：逐位一致用 torch.equal，屏蔽后的注意力权重必须精确等于 0。"""
from __future__ import annotations

import torch

from src import hooks


def _attn_for_layer(attentions, layer_id: int, attn_layers: list[int], num_layers: int):
    """从 outputs.attentions 里取第 layer_id 层的权重。有的实现只给注意力层返回条目，有的给全部层（非注意力层为 None）。"""
    atts = list(attentions)
    if len(atts) == num_layers:
        return atts[layer_id]
    if len(atts) == len(attn_layers):
        return atts[attn_layers.index(layer_id)]
    raise RuntimeError(f"attentions 长度 {len(atts)} 既不等于层数 {num_layers} 也不等于注意力层数 {len(attn_layers)}")


def check_hidden_states(ad, prep) -> dict:
    out = ad.forward(prep, output_hidden_states=True)
    hs = out.hidden_states
    n = len(ad.lm_layers)
    T = out.logits.shape[1]
    ok = len(hs) == n + 1 and all(h.shape[0] == 1 and h.shape[1] == T for h in hs)
    return {"pass": bool(ok), "n_hidden_states": len(hs), "expected": n + 1, "seq_len": T,
            "hidden_size": int(hs[-1].shape[-1]), "all_finite": bool(all(torch.isfinite(h).all() for h in hs))}


def check_attention_weights(ad, prep) -> dict:
    al = ad.attn_layers()
    if not al:
        return {"pass": False, "reason": "没有标准注意力层"}
    out = ad.forward(prep, output_attentions=True)
    n = len(ad.lm_layers)
    T = out.logits.shape[1]
    got, bad = 0, []
    for i in al:
        a = _attn_for_layer(out.attentions, i, al, n)
        if a is None:
            bad.append((i, "None"))
            continue
        if a.shape[-2:] != (T, T):
            bad.append((i, tuple(a.shape)))
            continue
        row = a[0, :, -1].float().sum(-1)        # 最后一个位置对所有 key 的权重和应为 1
        if not torch.allclose(row, torch.ones_like(row), atol=2e-2):
            bad.append((i, f"rowsum {row.min().item():.4f}..{row.max().item():.4f}"))
            continue
        got += 1
    a0 = _attn_for_layer(out.attentions, al[0], al, n)
    return {"pass": got == len(al), "attn_layers": len(al), "ok_layers": got, "bad": bad[:5],
            "n_returned": len(list(out.attentions)), "shape_example": None if a0 is None else tuple(a0.shape)}


def check_noop_identity(ad, prep) -> dict:
    base = ad.forward(prep).logits
    with hooks.noop_hooks(ad.lm_layers):
        hooked = ad.forward(prep).logits
    return {"pass": bool(torch.equal(base, hooked)), "max_abs_diff": float((base.float() - hooked.float()).abs().max())}


def check_self_patch(ad, prep, layer_ids=None) -> dict:
    n = len(ad.lm_layers)
    layer_ids = layer_ids or sorted({0, n // 2, n - 1})
    base = ad.forward(prep).logits
    store = {}
    with hooks.capture_residual(ad.lm_layers, layer_ids, store):
        ad.forward(prep)
    T = base.shape[1]
    res = {}
    for i in layer_ids:
        for tag, pos in (("visual", prep.visual_positions), ("all", range(T))):
            with hooks.patch_residual(ad.lm_layers, i, pos, store[i]):
                patched = ad.forward(prep).logits
            res[f"L{i}_{tag}"] = {"equal": bool(torch.equal(base, patched)),
                                  "max_abs_diff": float((base.float() - patched.float()).abs().max())}
    # 反向对照：patch 进别的值，输出必须变（证明 hook 真的生效）
    i = layer_ids[len(layer_ids) // 2]
    with hooks.patch_residual(ad.lm_layers, i, prep.visual_positions, torch.zeros_like(store[i])):
        changed = not torch.equal(base, ad.forward(prep).logits)
    return {"pass": all(v["equal"] for v in res.values()) and changed, "layers": res, "zero_patch_changes_output": changed,
            "mixers_tested": {i: ad.layer_mixers[i] for i in layer_ids}}


def check_head_hook(ad, prep, layer_id=None, head=0) -> dict:
    al = ad.attn_layers()
    if not al:
        return {"pass": False, "reason": "没有标准注意力层"}
    layer_id = al[len(al) // 2] if layer_id is None else layer_id
    attn = hooks.find_attn(ad.lm_layers[layer_id])
    d_in = attn.o_proj.in_features
    # 该层实际 head 数：优先模块/配置里的值，退回到模型清单
    n_heads = getattr(attn, "num_heads", None) or getattr(getattr(attn, "config", None), "num_attention_heads", None) or ad.n_heads
    base = ad.forward(prep).logits
    store = {}
    with hooks.head_output_hook(ad.lm_layers, layer_id, n_heads, store=store):
        read_only = ad.forward(prep).logits
    with hooks.head_output_hook(ad.lm_layers, layer_id, n_heads, fn=lambda x: x):
        identity = ad.forward(prep).logits

    def zero(x):
        x[:, :, head] = 0
        return x

    with hooks.head_output_hook(ad.lm_layers, layer_id, n_heads, fn=zero):
        ablated = ad.forward(prep).logits
    ph = store["per_head"]
    ok = (torch.equal(base, read_only) and torch.equal(base, identity) and not torch.equal(base, ablated)
          and ph.shape[2] == n_heads and d_in % n_heads == 0)
    return {"pass": bool(ok), "layer": layer_id, "n_heads": int(n_heads), "per_head_shape": tuple(ph.shape),
            "read_only_identical": bool(torch.equal(base, read_only)), "identity_rewrite_identical": bool(torch.equal(base, identity)),
            "zero_head_changes_output": bool(not torch.equal(base, ablated)),
            "max_abs_logit_change_zero_head": float((base.float() - ablated.float()).abs().max())}


def check_knockout(ad, prep, layer_id=None) -> dict:
    al = ad.attn_layers()
    if not al:
        return {"pass": False, "reason": "没有标准注意力层"}
    n = len(ad.lm_layers)
    layer_id = al[len(al) // 2] if layer_id is None else layer_id
    base = ad.forward(prep, output_attentions=True)
    T = base.logits.shape[1]
    q = [T - 1]
    keys = list(prep.visual_positions)
    if ad.layer_mixers[layer_id] == "sliding_attention":
        keys = keys[-16:]        # 滑动窗口层里只有窗口内的 key 原本权重非零，取离末尾最近的若干视觉 token
    a0 = _attn_for_layer(base.attentions, layer_id, al, n)[0, :, q][:, :, keys].float()
    rep = {}
    with hooks.attention_knockout(ad.lm_layers, [layer_id], q, keys, report=rep):
        ko = ad.forward(prep, output_attentions=True)
    a1 = _attn_for_layer(ko.attentions, layer_id, al, n)[0, :, q][:, :, keys].float()
    row = _attn_for_layer(ko.attentions, layer_id, al, n)[0, :, T - 1].float().sum(-1)
    # 其它层不受影响（取另一注意力层的同一位置比较）
    other = [i for i in al if i < layer_id]
    untouched = True
    if other:
        o0 = _attn_for_layer(base.attentions, other[-1], al, n)
        o1 = _attn_for_layer(ko.attentions, other[-1], al, n)
        untouched = bool(torch.equal(o0, o1))
    ok = (float(a1.abs().max()) == 0.0 and float(a0.sum()) > 0 and untouched
          and torch.allclose(row, torch.ones_like(row), atol=2e-2) and not torch.equal(base.logits, ko.logits))
    return {"pass": bool(ok), "layer": layer_id, "mixer": ad.layer_mixers[layer_id], "n_keys": len(keys),
            "attn_mass_before": float(a0.sum(-1).mean()), "max_weight_after": float(a1.abs().max()),
            "rows_still_sum_to_1": bool(torch.allclose(row, torch.ones_like(row), atol=2e-2)),
            "earlier_layer_untouched": untouched, "logits_changed": bool(not torch.equal(base.logits, ko.logits)), **rep}


def describe_nonstandard_layers(ad, prep) -> dict:
    """对非标准层（linear_attention 等）如实记录：有哪些子模块、残差流能否读写、有没有注意力权重。"""
    ns = [i for i, t in enumerate(ad.layer_mixers) if t not in ("full_attention", "sliding_attention")]
    if not ns:
        return {"has_nonstandard": False}
    i = ns[len(ns) // 2]
    layer = ad.lm_layers[i]
    children = {n: type(c).__name__ for n, c in layer.named_children()}
    store = {}
    with hooks.capture_residual(ad.lm_layers, [i], store):
        base = ad.forward(prep).logits
    with hooks.patch_residual(ad.lm_layers, i, prep.visual_positions, store[i]):
        same = torch.equal(base, ad.forward(prep).logits)
    with hooks.patch_residual(ad.lm_layers, i, prep.visual_positions, torch.zeros_like(store[i])):
        changed = not torch.equal(base, ad.forward(prep).logits)
    softmax_attn = True
    try:
        hooks.find_attn(layer)
    except RuntimeError:
        softmax_attn = False
    return {"has_nonstandard": True, "n_nonstandard": len(ns), "example_layer": i, "type": ad.layer_mixers[i],
            "children": children, "residual_capture_ok": True, "self_patch_identical": bool(same),
            "zero_patch_changes_output": bool(changed), "has_softmax_attention_module": softmax_attn,
            "can": "读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出",
            "cannot": "没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析"}


def check_determinism(ad, prep, max_new_tokens=32) -> dict:
    torch.manual_seed(0)
    g1 = ad.generate(prep, max_new_tokens)
    torch.manual_seed(0)
    g2 = ad.generate(prep, max_new_tokens)
    return {"pass": g1["token_ids"] == g2["token_ids"] and bool(torch.equal(g1["first_logits"], g2["first_logits"])),
            "tokens_identical": g1["token_ids"] == g2["token_ids"],
            "first_logits_identical": bool(torch.equal(g1["first_logits"], g2["first_logits"]))}
