"""直接回答选项的 fp32 打分（scripts/synth_run.py、scripts/glm_letter_step_check.py 共用）。
FP32Head：在输出层（lm_head）上挂前向钩子，每一步生成都取最后一个位置的输入（最终归一化后的隐状态），分块按 fp32 重算 logit 与全词表 logsumexp。
模型本身照常用 bf16 生成，钩子不改变解码。
  首步结果 = res（字段 fp32_lse / fp32_logprob / fp32_argmax_logit_check，与第 10 轮冒烟的记录格式相同）；
  每一步的结果在 steps 里（第 k 个元素对应生成的第 k 个 token），final_letter_step() 用它找“最终字母所在的那一步”（planner 第 11 轮第 3 条）。"""
from __future__ import annotations

import torch

CHUNK = 16384


def find_head(model):
    for obj in (model, getattr(model, "language_model", None), getattr(model, "llm", None)):
        if obj is not None and hasattr(obj, "get_output_embeddings") and obj.get_output_embeddings() is not None:
            return obj.get_output_embeddings()
    raise RuntimeError("找不到输出层")


@torch.no_grad()
def fp32_scores(x, W, b, ids: dict):
    """x：隐状态（1 维）；W、b：输出层权重与偏置；ids：{变体: token id}。返回 (lse, {变体: fp32 logprob}, argmax id)。"""
    x = x.float()
    lse_parts, sel, best = [], {}, (-float("inf"), -1)
    for i in range(0, W.shape[0], CHUNK):
        z = W[i:i + CHUNK].float() @ x
        if b is not None:
            z = z + b[i:i + CHUNK].float()
        lse_parts.append(torch.logsumexp(z, 0))
        v, j = torch.max(z, 0)
        if float(v) > best[0]:
            best = (float(v), i + int(j))
        for name, t in ids.items():
            if i <= t < i + CHUNK:
                sel[name] = float(z[t - i])
    lse = float(torch.logsumexp(torch.stack(lse_parts), 0))
    return lse, {v: z - lse for v, z in sel.items()}, best[1]


class FP32Head:
    def __init__(self, ad, variants, all_steps=False):
        self.head = find_head(ad.model)
        self.ids = {}
        for v in variants:
            t = ad.tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:
                self.ids[v] = t[0]
        self.all_steps = all_steps
        self.res, self.steps = None, []
        self.handle = self.head.register_forward_hook(self.hook)

    def reset(self):
        self.res, self.steps = None, []

    @torch.no_grad()
    def hook(self, m, inp, out):
        if self.res is not None and not self.all_steps:
            return
        lse, lp, am = fp32_scores(inp[0][0, -1], m.weight, getattr(m, "bias", None), self.ids)
        if self.res is None:
            self.res = {"fp32_lse": lse, "fp32_logprob": lp, "fp32_argmax_logit_check": float(out[0, -1].float().max())}
        self.steps.append({"lse": lse, "logprob": lp, "argmax_id": am})


def final_letter_step(token_ids, steps, letter_ids: dict):
    """token_ids：生成的 token；steps：FP32Head.steps；letter_ids：{变体: id}（只放字母变体，如 A、" A"、B、" B"）。
    在生成序列里找最后一个字母 token，返回该步的 fp32 打分；找不到返回 None。fp32 的 argmax 与生成的 token 是否一致另记（bf16 与 fp32 可能在近似平局时不同）。"""
    inv = {}
    for v, t in letter_ids.items():
        inv.setdefault(t, v)
    for k in range(len(token_ids) - 1, -1, -1):
        if token_ids[k] in inv:
            if k >= len(steps):
                return {"step": k, "token": inv[token_ids[k]], "error": f"钩子只记了 {len(steps)} 步"}
            s = steps[k]
            return {"step": k, "token": inv[token_ids[k]], "fp32_lse": s["lse"], "fp32_logprob": s["logprob"], "fp32_argmax_is_generated": s["argmax_id"] == token_ids[k],
                    "n_steps_hooked": len(steps)}
    return None
