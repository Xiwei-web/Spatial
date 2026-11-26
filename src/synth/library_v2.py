"""抠图库 v2 的过滤规则（纯函数，可单测；配置见 configs/synth/library_v2.yaml）。框一律为 COCO 的 [x, y, w, h]。"""
from __future__ import annotations

import math

import numpy as np


def bbox_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def rule_a(ann: dict, others: list[dict]) -> bool:
    """候选不是 iscrowd，且外接框与同图其他任何实例（含 iscrowd）的外接框交集面积为 0（边相接不算交集）。"""
    if ann.get("iscrowd", 0):
        return False
    return all(bbox_intersection_area(ann["bbox"], o["bbox"]) == 0 for o in others if o["id"] != ann["id"])


def rule_b(ann: dict, width: int, height: int, min_px: float) -> bool:
    x, y, w, h = ann["bbox"]
    return x >= min_px and y >= min_px and x + w <= width - min_px and y + h <= height - min_px


def fill_ratio(ann: dict) -> float:
    w, h = ann["bbox"][2], ann["bbox"][3]
    return ann["area"] / (w * h) if w * h > 0 else 0.0


def fill_quantiles(pool: dict[str, list[float]], q=(0.10, 0.90)) -> dict[str, list[float]]:
    """pool：{类别: [fill_ratio, …]}（a、b 过滤后的 train 候选）→ {类别: [q_lo, q_hi]}（numpy 线性插值）。"""
    return {c: [float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))] for c, v in pool.items() if len(v)}


def rule_c(ann: dict, category: str, quantiles: dict[str, list[float]]) -> bool:
    if category not in quantiles:
        return False
    lo, hi = quantiles[category]
    return lo <= fill_ratio(ann) <= hi


def rule_d(long_side: float, short_side: float, mask_area: float, cfg: dict) -> dict:
    """返回各子条件是否通过（长边、放大倍数、宽高比）。long/short 为最大连通块紧致框的边长，mask_area 为其像素数。"""
    up = cfg["d_max_sqrt_area_px"] / math.sqrt(mask_area) if mask_area > 0 else float("inf")
    return {"long_side": long_side >= cfg["d_min_long_side_px"], "upscale": up <= cfg["d_max_upscale"],
            "aspect": short_side > 0 and long_side / short_side <= cfg["d_max_aspect"], "upscale_value": up}


def clip_labels(coco80: list[str], confusable: list[str]) -> list[str]:
    out = list(coco80)
    for c in confusable:
        if c not in out:
            out.append(c)
    return out


def clip_decision(probs, labels: list[str], true_label: str, min_prob: float) -> dict:
    """probs：对 labels 的 softmax。真类别必须排第一且概率 >= min_prob；任何标签（含易混标签）排在真类别之前即不通过。"""
    probs = np.asarray(probs, float)
    k = labels.index(true_label)
    rank = int((probs > probs[k]).sum())
    top = int(probs.argmax())
    return {"pass": bool(rank == 0 and probs[k] >= min_prob), "prob_true": float(probs[k]), "rank_true": rank, "top1": labels[top], "top1_prob": float(probs[top])}


def small_tier_scale(mask_area: float, rendered_px: float) -> float:
    """36 px 档检查：把抠图缩放到 sqrt(mask 面积) = rendered_px 所需的缩放系数（< 1 为缩小）。"""
    return rendered_px / math.sqrt(mask_area)
