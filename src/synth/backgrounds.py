"""程序化背景：纯色 / 线性渐变 / 径向渐变 / 低频噪声纹理。本版本不使用自然图像背景。

分两步：`sample_background(rng, type)` 只抽参数（全部可 JSON 序列化，原样写进 metadata），
`render_background(params, size)` 只根据参数画图（不再用随机数）。所以 metadata 里的参数足以重画背景。
颜色各通道取值范围 [30, 225]，避免纯黑 / 纯白。
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

TYPES = ("solid", "linear_gradient", "radial_gradient", "noise")


def _color(rng) -> list[int]:
    return [int(v) for v in rng.integers(30, 226, size=3)]


def sample_background(rng: np.random.Generator, bg_type: str) -> dict:
    if bg_type == "solid":
        return {"type": "solid", "color": _color(rng)}
    if bg_type == "linear_gradient":
        # 沿 angle_deg 方向（0° = 向右，90° = 向下）从 color0 过渡到 color1
        return {"type": "linear_gradient", "color0": _color(rng), "color1": _color(rng),
                "angle_deg": int(rng.integers(0, 360))}
    if bg_type == "radial_gradient":
        # 圆心处 color0，离圆心 radius_frac × 画布边长 及更远处 color1；圆心坐标为画布边长的比例
        return {"type": "radial_gradient", "color0": _color(rng), "color1": _color(rng),
                "center_frac": [round(float(rng.uniform(0.2, 0.8)), 4), round(float(rng.uniform(0.2, 0.8)), 4)],
                "radius_frac": round(float(rng.uniform(0.4, 1.0)), 4)}
    if bg_type == "noise":
        # n×n 的随机网格（0..255）双三次放大到画布大小，作为 color0 → color1 的混合系数
        n = int(rng.integers(3, 9))
        return {"type": "noise", "color0": _color(rng), "color1": _color(rng), "grid_n": n,
                "grid": [[int(v) for v in row] for row in rng.integers(0, 256, size=(n, n))]}
    raise ValueError(f"未知背景类型 {bg_type!r}；可选 {TYPES}")


def _mix(c0, c1, t: np.ndarray) -> np.ndarray:
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    img = c0 + (c1 - c0) * t[..., None]
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)


def render_background(params: dict, size: int) -> np.ndarray:
    """返回 (size, size, 3) uint8。像素 (i, j) 的坐标取像素中心 (i + 0.5, j + 0.5)。"""
    t = params["type"]
    if t == "solid":
        return np.broadcast_to(np.asarray(params["color"], dtype=np.uint8), (size, size, 3)).copy()
    c = np.arange(size, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(c, c)
    if t == "linear_gradient":
        a = math.radians(params["angle_deg"])
        dx, dy = math.cos(a), math.sin(a)
        proj = (xx - size / 2) * dx + (yy - size / 2) * dy
        half = size / 2 * (abs(dx) + abs(dy))          # 画布四角投影的最大绝对值
        return _mix(params["color0"], params["color1"], np.clip(proj / (2 * half) + 0.5, 0.0, 1.0))
    if t == "radial_gradient":
        cx, cy = (v * size for v in params["center_frac"])
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return _mix(params["color0"], params["color1"], np.clip(r / (params["radius_frac"] * size), 0.0, 1.0))
    if t == "noise":
        g = Image.fromarray(np.asarray(params["grid"], dtype=np.uint8))
        up = np.asarray(g.resize((size, size), Image.BICUBIC), dtype=np.float64) / 255.0
        return _mix(params["color0"], params["color1"], up)
    raise ValueError(f"未知背景类型 {t!r}")
