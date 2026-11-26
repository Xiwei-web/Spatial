"""单物体定位检查的图像（configs/synth/synth_v2.yaml 的 single_object_check；Xiwei 第 7 条 (b)）。
只用 dev 抠图；4 个尺寸档各 per_size 张；896 画布、程序背景、无干扰物；物体位置在画布内（离边 >= 16 px）随机。完全由配置 + 序号决定。"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import backgrounds
from . import generator_v1 as g1
from .cutouts import mask_bbox, png_bytes
from .generator import _rle

VERSION = "single_object_v2"


def design(cfg):
    so, sizes, bgs = cfg["single_object_check"], cfg["sizes_px"], cfg["design"]["background"]
    rows = []
    for k, s in enumerate(sizes):
        for i in range(so["per_size"]):
            rows.append({"index": len(rows), "sqrt_area_px": s, "background": bgs[i % len(bgs)]})
    return rows


def generate(cfg, lib, index):
    row = design(cfg)[index]
    C, mg, s = cfg["canvas"], cfg["placement"]["canvas_margin_px"], row["sqrt_area_px"]
    seed = int.from_bytes(hashlib.sha256(f"{VERSION}:{cfg['global_seed']}:{index}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    cats = lib.categories("dev", small36=(s == 36))
    names = sorted(cats)
    c = names[(index * 7 + int(rng.integers(len(names)))) % len(names)]
    cid = cats[c][int(rng.integers(len(cats[c])))]
    o = g1.prepare(lib, cid, s, cfg["alpha_threshold"], cfg["max_upscale"])
    off = (int(rng.integers(mg, C - mg - o.w + 1)), int(rng.integers(mg, C - mg - o.h + 1)))
    bg = backgrounds.sample_background(rng, row["background"])
    img = backgrounds.render_background(bg, C)
    g1._paste(img, o, off)
    m = g1._full(o, off, C)
    png = png_bytes(img, cfg["png_compress_level"])
    sid = f"{VERSION}_{index:04d}"
    meta = {"scene_id": sid, "version": VERSION, "index": index, "side": "dev", "image": f"images/{sid}.png", "canvas_wh": [C, C], "sqrt_area_px": s,
            "object": {"cutout_id": o.cutout_id, "category": o.category, "offset_xy": list(off), "box_xyxy": mask_bbox(m), "centroid_xy": [round(v, 4) for v in g1._cen(o, off)],
                       "area": int(m.sum()), "sqrt_area": round(math.sqrt(m.sum()), 4), "upscale": round(o.scale, 4), "mask_rle": _rle(m)},
            "background": bg, "seed": seed, "image_sha256": hashlib.sha256(png).hexdigest()}
    return meta, png
