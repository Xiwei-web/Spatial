"""候选框选择规则（纯函数，CPU）。候选按检测分数降序，元素为 {"box_xyxy": [...], "score": float}。"""
from __future__ import annotations


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_v2(t_cands, r_cands, thr=0.8):
    """v2 的规则 (a)：默认 reference 的最高分候选是对的；target 最高分候选与它 IoU>thr 时 target 换下一个不重合的候选。
    返回 (target 下标, reference 下标, 说明)。"""
    if not t_cands or not r_cands:
        return (0 if t_cands else None), (0 if r_cands else None), "top_score"
    if box_iou(t_cands[0]["box_xyxy"], r_cands[0]["box_xyxy"]) <= thr:
        return 0, 0, "top_score"
    alt = [i for i, c in enumerate(t_cands) if box_iou(c["box_xyxy"], r_cands[0]["box_xyxy"]) <= thr]
    return (alt[0], 0, "target_moved") if alt else (0, 0, "no_alternative")


def select_v3(t_cands, r_cands, thr=0.8):
    """修正后的规则 (a)（用户 2026-11-25）：两者最高分候选高度重合（IoU>thr）时比较各自的最高分，
    分数高的一方保留这个框，分数低的一方改用下一个与保留框不重合（IoU<=thr）的候选；没有这样的候选则保留并标记。
    分数相等时保留 reference（与 v2 行为一致）。"""
    if not t_cands or not r_cands:
        return (0 if t_cands else None), (0 if r_cands else None), "top_score"
    if box_iou(t_cands[0]["box_xyxy"], r_cands[0]["box_xyxy"]) <= thr:
        return 0, 0, "top_score"
    if t_cands[0]["score"] > r_cands[0]["score"]:       # target 分数高：target 保留，reference 让位
        keep = t_cands[0]["box_xyxy"]
        alt = [i for i, c in enumerate(r_cands) if box_iou(c["box_xyxy"], keep) <= thr]
        return (0, alt[0], "reference_moved") if alt else (0, 0, "no_alternative_for_reference")
    keep = r_cands[0]["box_xyxy"]
    alt = [i for i, c in enumerate(t_cands) if box_iou(c["box_xyxy"], keep) <= thr]
    return (alt[0], 0, "target_moved") if alt else (0, 0, "no_alternative_for_target")
