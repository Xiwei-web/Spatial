"""抠图库 v3 的新增过滤特征（纯函数，CPU；配置 configs/synth/library_v3.yaml）。mask 一律为 bool 的 H×W 数组；COCO 多边形为 [x0, y0, x1, y1, …]。"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

KP = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ankle": 15, "right_ankle": 16}      # COCO 17 点的顺序


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def boundary(m: np.ndarray) -> np.ndarray:
    """mask 的边界像素（8 邻域腐蚀后剩下的差）。"""
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def near_others(others: np.ndarray, r: int) -> np.ndarray:
    """同图其他实例（含 crowd）mask 的并集膨胀 r px。"""
    return ndimage.binary_dilation(others, structure=disk(r)) if others.any() else others


def occ_contact_frac(m: np.ndarray, near: np.ndarray) -> float:
    """规则 a：mask 边界像素里落在 near 范围内的比例。"""
    b = boundary(m)
    n = int(b.sum())
    return float((b & near).sum() / n) if n else 0.0


def _segments(seg) -> list[list[tuple[float, float, float, float]]]:
    out = []
    for poly in seg if isinstance(seg, list) else []:
        p = np.asarray(poly, float).reshape(-1, 2)
        if len(p) < 3:
            continue
        out.append([(p[i, 0], p[i, 1], p[(i + 1) % len(p), 0], p[(i + 1) % len(p), 1]) for i in range(len(p))])
    return out


def _contact(seg, near, step=1.0) -> float:
    x0, y0, x1, y1 = seg
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L / step) + 1)
    xs = np.clip(np.round(np.linspace(x0, x1, n)).astype(int), 0, near.shape[1] - 1)
    ys = np.clip(np.round(np.linspace(y0, y1, n)).astype(int), 0, near.shape[0] - 1)
    return float(near[ys, xs].mean())


def straight_runs(seg, area: float, near: np.ndarray | None, angle_tol_deg: float, min_contact: float) -> float:
    """规则 b：把多边形里相邻、方向差 <= angle_tol_deg 的线段合并成直线段（run）；near 不为 None 时只合并“与其他实例接触”的线段
    （线段上采样点落在 near 内的比例 >= min_contact）。返回最长 run 的长度 / sqrt(mask 面积)。near 为 None 时不要求接触（参考变体）。"""
    best = 0.0
    for segs in _segments(seg):
        ok = [near is None or _contact(s, near) >= min_contact for s in segs]
        ang = [math.atan2(s[3] - s[1], s[2] - s[0]) for s in segs]
        L = [math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs]
        n = len(segs)
        # 从一个不可合并的位置起转一圈，避免把首尾断开的同一条直线算成两段
        start = next((i for i in range(n) if not ok[i]), 0)
        run, prev = 0.0, None
        for k in range(n + 1):
            i = (start + k) % n
            if k < n and ok[i] and (prev is None or abs(math.remainder(ang[i] - ang[prev], 2 * math.pi)) <= math.radians(angle_tol_deg)):
                run += L[i]
            else:
                best = max(best, run)
                run = L[i] if (k < n and ok[i]) else 0.0
            prev = i if (k < n and ok[i]) else None
    return best / math.sqrt(area) if area > 0 else 0.0


def person_visible(kps) -> bool:
    """规则 c：鼻子或双眼至少一个可见（v >= 1），并且双脚踝都可见。"""
    v = lambda k: kps[3 * KP[k] + 2] >= 1      # noqa: E731
    return (v("nose") or v("left_eye") or v("right_eye")) and v("left_ankle") and v("right_ankle")


def n_components(m: np.ndarray, min_frac: float) -> int:
    """规则 d：8 连通的连通块数，只数面积 >= min_frac × 总面积的块（与 SpatialRGPT 去碎块同一口径）。"""
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    a = np.bincount(lab.ravel())[1:]
    return int((a >= min_frac * m.sum()).sum())


def solidity(m: np.ndarray) -> float:
    """规则 d：mask 面积 / 凸包面积（凸包按像素四角点算）。"""
    import cv2
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return 1.0
    pts = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)]).astype(np.float32)
    hull = cv2.convexHull(pts)
    ha = cv2.contourArea(hull)
    return float(m.sum() / ha) if ha > 0 else 1.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def rates(flags, verdicts) -> dict:
    """flags：是否剔除；verdicts：'BAD'/'OK'/'UNSURE'。返回 BAD 召回与 OK 损失（UNSURE 不参与）。"""
    bad = [f for f, v in zip(flags, verdicts) if v == "BAD"]
    ok = [f for f, v in zip(flags, verdicts) if v == "OK"]
    return {"n_bad": len(bad), "n_ok": len(ok), "bad_recall": sum(bad) / len(bad) if bad else None, "ok_loss": sum(ok) / len(ok) if ok else None}
