"""框坐标约定标定（只用 dev）：候选约定的事后换算、IoU、按图像 bootstrap 的选择规则。纯函数，不碰模型、不碰 test。

候选（对同一份原始四个数事后换算，不用真值挑）：
  8 种固定约定 = {norm1000, norm1, model, orig} × {xyxy, yxyx}；
  2 种按数值范围的规则（xyxy，不看真值）：range_norm1_else_norm1000 / range_norm1_else_model ——四个数都 ≤ 1 按 0–1 归一，否则按 0–1000 归一 / 模型输入像素。
归一化与模型输入像素都相对“模型看到的视图”：有 view_box（LLaVA-1.5 补成正方形）时相对该正方形，否则相对整张原图；orig 不做任何平移。
IoU：解析失败、x2 ≤ x1 或 y2 ≤ y1（换算后、裁剪前）一律 0；超出图像的先裁剪到 [0, W] × [0, H] 再算（裁剪后无面积也是 0）。出错 / OOM 的条目按 IoU 0 计入分母并单独计数。
越界：换算后、裁剪前任一坐标超出图像（容差 1e-6）；比例以解析成功的条目为分母。

选择规则（planner 第 3 轮，写死在这里，先提交再看任何 IoU）：
  1. 每个模型在所有（提示词, 约定）里取平均 IoU 最高的；平手按固定顺序（提示词 P0 先，约定按 CONVENTIONS 的顺序）。
  2. 第二名 = 其余候选里平均 IoU 最高的，排除与第一名逐条换算结果完全相同的。
  3. 差 = 第一名 − 第二名（逐条配对），按图像 bootstrap 10,000 次（同图的框一起重采样，百分位法，seed 20261125）。
  4. 差的 95% 区间含 0，或最高平均 IoU < 0.3，或 What's Up 与 COCO 各自的第一名不是总第一名（该子集内总第一名的平均 IoU 低于子集最高值，容差 1e-12）→ “未定”。"""
from __future__ import annotations

import numpy as np

CONVENTIONS = [f"{k}_{o}" for k in ("norm1000", "norm1", "model", "orig") for o in ("xyxy", "yxyx")] + ["range_norm1_else_norm1000", "range_norm1_else_model"]
N_BOOT, SEED = 10000, 20261125


def convert(nums, conv: str, orig_wh, model_wh, view_box=None) -> list[float] | None:
    """四个数 → 原图像素 [x1, y1, x2, y2]；nums 为 None 返回 None。"""
    if nums is None:
        return None
    a = [float(v) for v in nums]
    if conv.startswith("range_"):
        conv = "norm1_xyxy" if all(v <= 1 for v in a) else ("norm1000_xyxy" if conv.endswith("norm1000") else "model_xyxy")
    kind, order = conv.split("_")
    if order == "yxyx":
        a = [a[1], a[0], a[3], a[2]]
    W, H = orig_wh
    if kind == "orig":
        return a
    x0, y0, x1, y1 = view_box if view_box is not None else (0, 0, W, H)
    if kind == "norm1000":
        sx, sy = (x1 - x0) / 1000, (y1 - y0) / 1000
    elif kind == "norm1":
        sx, sy = float(x1 - x0), float(y1 - y0)
    else:
        sx, sy = (x1 - x0) / model_wh[0], (y1 - y0) / model_wh[1]
    return [x0 + a[0] * sx, y0 + a[1] * sy, x0 + a[2] * sx, y0 + a[3] * sy]


def iou(pred, truth, orig_wh) -> float:
    if pred is None or pred[2] <= pred[0] or pred[3] <= pred[1]:
        return 0.0
    W, H = orig_wh
    x1, x2 = (min(max(v, 0.0), W) for v in (pred[0], pred[2]))
    y1, y2 = (min(max(v, 0.0), H) for v in (pred[1], pred[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    tx1, ty1, tx2, ty2 = truth
    iw, ih = min(x2, tx2) - max(x1, tx1), min(y2, ty2) - max(y1, ty1)
    inter = max(0.0, iw) * max(0.0, ih)
    return inter / ((x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter)


def out_of_bounds(pred, orig_wh, tol=1e-6) -> bool:
    W, H = orig_wh
    return pred[0] < -tol or pred[1] < -tol or pred[2] > W + tol or pred[3] > H + tol


def build(records: list[dict], prompts: list[str]) -> dict:
    """records：每条 {item_id, prompt_id, parsed, orig_wh, model_wh, view_box, truth, subset, image_key}。
    返回 {(prompt, conv): {"iou": (n,), "boxes": [...]}} 与公共的逐条信息（每个提示词必须覆盖同一批 item，顺序按 item_id）。"""
    by = {p: {r["item_id"]: r for r in records if r["prompt_id"] == p} for p in prompts}
    ids = sorted(by[prompts[0]])
    for p in prompts:
        assert sorted(by[p]) == ids, p
    cand = {}
    for p in prompts:
        for c in CONVENTIONS:
            bx = [convert(by[p][i]["parsed"], c, by[p][i]["orig_wh"], by[p][i]["model_wh"], by[p][i].get("view_box")) for i in ids]
            cand[(p, c)] = {"boxes": bx, "iou": np.array([iou(b, by[p][i]["truth"], by[p][i]["orig_wh"]) for b, i in zip(bx, ids)])}
    r0 = [by[prompts[0]][i] for i in ids]
    return {"ids": ids, "cand": cand, "subset": np.array([r["subset"] for r in r0]), "image": [r["image_key"] for r in r0], "by": by}


def bootstrap_diff(diff: np.ndarray, images: list[str], n_boot=N_BOOT, seed=SEED) -> list[float]:
    uniq = sorted(set(images))
    pos = {u: k for k, u in enumerate(uniq)}
    g = np.array([pos[i] for i in images])
    s = np.bincount(g, weights=diff, minlength=len(uniq))
    n = np.bincount(g, minlength=len(uniq)).astype(float)
    draw = np.random.default_rng(seed).integers(0, len(uniq), (n_boot, len(uniq)))
    m = s[draw].sum(1) / n[draw].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def select(records: list[dict], prompts: list[str]) -> dict:
    d = build(records, prompts)
    cand, ids = d["cand"], d["ids"]
    order = [(p, c) for p in prompts for c in CONVENTIONS]
    mean = {k: float(cand[k]["iou"].mean()) for k in order}
    best = max(order, key=lambda k: mean[k])                                   # max 取最先出现的最大值
    rest = [k for k in order if k != best and cand[k]["boxes"] != cand[best]["boxes"]]
    second = max(rest, key=lambda k: mean[k])
    diff = cand[best]["iou"] - cand[second]["iou"]
    ci = bootstrap_diff(diff, d["image"])
    sub_first, sub_ok = {}, {}
    for sb in sorted(set(d["subset"])):
        m = d["subset"] == sb
        sm = {k: float(cand[k]["iou"][m].mean()) for k in order}
        top = max(order, key=lambda k: sm[k])
        sub_first[sb] = {"first": list(top), "mean_iou": sm[top], "overall_first_mean_iou": sm[best], "n": int(m.sum())}
        sub_ok[sb] = sm[best] >= sm[top] - 1e-12
    reasons = []
    if ci[0] <= 0 <= ci[1]:
        reasons.append("与第二名的差 95% 区间含 0")
    if mean[best] < 0.3:
        reasons.append("最高平均 IoU < 0.3")
    if not all(sub_ok.values()):
        reasons.append("What's Up 与 COCO 的第一名不一致")
    pr = {p: np.array([d["by"][p][i]["parsed"] is None for i in ids]) for p in prompts}
    err = {p: sum(bool(d["by"][p][i].get("error")) for i in ids) for p in prompts}
    oob = {}
    for k in order:
        bx = [b for b, i in zip(cand[k]["boxes"], ids) if b is not None]
        ws = [d["by"][k[0]][i]["orig_wh"] for b, i in zip(cand[k]["boxes"], ids) if b is not None]
        oob[k] = float(np.mean([out_of_bounds(b, w) for b, w in zip(bx, ws)])) if bx else None
    tbl = [{"prompt": p, "convention": c, "mean_iou": mean[(p, c)], "iou_at_0.5": float((cand[(p, c)]["iou"] >= 0.5).mean()), "out_of_bounds": oob[(p, c)],
            **{f"mean_iou_{sb}": float(cand[(p, c)]["iou"][d["subset"] == sb].mean()) for sb in sorted(set(d["subset"]))}} for p, c in order]
    b = cand[best]["iou"]
    return {"n_items": len(ids), "n_images": len(set(d["image"])), "first": list(best), "first_mean_iou": mean[best], "first_iou_at_0.5": float((b >= 0.5).mean()), "first_out_of_bounds": oob[best],
            "second": list(second), "second_mean_iou": mean[second], "diff_mean": float(diff.mean()), "diff_ci95": ci, "subset_first": sub_first, "undecided": bool(reasons), "undecided_reasons": reasons,
            "parse_fail": {p: float(pr[p].mean()) for p in prompts}, "n_error": err, "per_candidate": tbl,
            "first_subset": {sb: {"mean_iou": float(b[d["subset"] == sb].mean()), "iou_at_0.5": float((b[d["subset"] == sb] >= 0.5).mean()), "n": int((d["subset"] == sb).sum())} for sb in sorted(set(d["subset"]))}}
