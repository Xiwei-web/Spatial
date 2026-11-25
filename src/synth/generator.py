"""参数化合成场景生成器（F4）。

一个“基础场景”= 程序化背景 + 两个主物体（target / reference）+ 若干干扰物，另带两张孪生图：
  position_swap  target 与 reference 互换位置，其余完全相同，关系标签取反；
  identity_swap  位置不变，target 换成另一个类别的物体（同一大小档）。

精确定义（写报告和下游代码都以这里为准）：
  坐标      像素 (i, j) 覆盖 [i, i+1) × [j, j+1)；框 [x1, y1, x2, y2)；质心 = 下标均值 + 0.5；y 轴向下。
  mask      alpha >= alpha_threshold(128) 的像素。「完整 mask」= 物体自己的全部 mask（不管有没有被压住）；
            「可见 mask」= 完整 mask 去掉被上层主物体完整 mask 盖住的部分（上层物体的可见 mask = 完整 mask）。
  大小      缩放后完整 mask 紧致框的长边（像素）= floor(size_frac × 画布宽 + 0.5)。
  间距      两个主物体完整 mask 质心沿关系轴的坐标差的绝对值。物体只能放在整数像素上，而质心一般带小数，
            所以实际间距与要求值最多差 0.5 px（先放 reference，再把 target 放到离目标质心最近的整数位置）。
  抖动      垂直于关系轴的质心坐标差（target − reference），整数要求值，实际值同样最多差 0.5 px。
  关系标签  由实际质心算：horizontal → target 在 reference 的 left / right；vertical → above / below。
  遮挡      occluded_fraction = |下层完整 ∩ 上层完整| / |下层完整|；iou_complete = 两个完整 mask 的 IoU；
            两个可见 mask 按定义不相交，所以 iou_visible 恒为 0（照写，免得下游误会）。
  z 顺序    背景 → 干扰物 → 下层主物体 → 上层主物体。谁在上层由设计表指定（跟物体身份走，位置互换后不变）。

随机性：scene_seed = sha256("<global_seed>:<index>") 的前 8 字节；第 a 次尝试用 default_rng([scene_seed, a])。
设计表（间距 / 大小 / 轴 / 干扰物数 / 背景类型 / 上下层 / 方向）只由配置决定，所以任何一个场景都能单独重生成。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src import paths

from . import backgrounds
from .cutouts import CutoutLibrary, mask_bbox, png_bytes

OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
RELATIONS = {"horizontal": ("left", "right"), "vertical": ("above", "below")}   # (负方向, 正方向)
VARIANTS = ("base", "position_swap", "identity_swap")


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = paths.REPO_ROOT / p
    return yaml.safe_load(open(p))


def scene_seed(global_seed: int, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{global_seed}:{index}".encode()).digest()[:8], "big")


def size_px(frac: float, canvas: int) -> int:
    return int(math.floor(frac * canvas + 0.5))


def _round(x: float) -> int:
    return int(math.floor(x + 0.5))


# =====================================================================
# 设计表
# =====================================================================
def _balanced_assign(strata: dict, levels: list, rng: np.random.Generator, out: list, key: str) -> None:
    """层内随机排列后轮流取各水平；计数器跨层延续，所以每层内和总体都尽量均衡。"""
    counter = int(rng.integers(len(levels)))
    for s in sorted(strata):
        idx = strata[s]
        for i in rng.permutation(len(idx)):
            out[idx[int(i)]][key] = levels[counter % len(levels)]
            counter += 1


_DESIGN_CACHE: dict[str, list[dict]] = {}


def build_design(cfg: dict) -> list[dict]:
    """返回 n_scenes 行设计表。

    间距 × 大小 × 干扰物数 × 轴 的全因子格子按下面的顺序循环：k = index mod n_cells，
    间距 = k mod nD，大小 = k mod nS，干扰物数 = k mod nN（三者两两互素时前 nD·nS·nN 个互不重复），
    轴 = (k' // nN + k // (nD·nS·nN)) mod nA，其中 k' = k mod (nD·nS·nN)。
    这样任何长度的前缀在每个因子上的边际都接近均衡，前 n_cells 个场景恰好覆盖每个格子一次。
    背景类型 / 上层物体 / 方向 用固定种子在层内均衡分配（分层变量分别是 间距、间距×大小、轴×间距）。
    """
    f = cfg["factors"]
    key = json.dumps([cfg["global_seed"], cfg["n_scenes"], cfg["canvas"], f], sort_keys=True)
    if key in _DESIGN_CACHE:
        return _DESIGN_CACHE[key]
    D, S, N, A = f["distance_px"], f["size_frac"], f["n_distractors"], f["axis"]
    tri = len(D) * len(S) * len(N)
    n_cells = tri * len(A)
    rows = []
    for i in range(cfg["n_scenes"]):
        k = i % n_cells
        kk = k % tri
        rows.append({"index": i, "cell": k, "replicate": i // n_cells,
                     "distance_px": D[k % len(D)], "size_frac": S[k % len(S)], "size_px": size_px(S[k % len(S)], cfg["canvas"]),
                     "n_distractors": N[k % len(N)], "axis": A[(kk // len(N) + k // tri) % len(A)]})
    cells = {(r["distance_px"], r["size_frac"], r["n_distractors"], r["axis"]) for r in rows[:n_cells]}
    if len(cells) != min(n_cells, len(rows)):
        raise ValueError("设计表的格子有重复：间距 / 大小 / 干扰物数 的水平个数需要两两互素")
    rng = np.random.default_rng([cfg["global_seed"], 0xF4])

    def strata(fn):
        s: dict = {}
        for r in rows:
            s.setdefault(fn(r), []).append(r["index"])
        return s

    _balanced_assign(strata(lambda r: r["distance_px"]), list(f["background"]), rng, rows, "background")
    _balanced_assign(strata(lambda r: (r["distance_px"], r["size_frac"])), ["target", "reference"], rng, rows, "top")
    _balanced_assign(strata(lambda r: (r["axis"], r["distance_px"])), [-1, 1], rng, rows, "sign")
    for r in rows:
        r["relation"] = RELATIONS[r["axis"]][0 if r["sign"] < 0 else 1]
    _DESIGN_CACHE[key] = rows
    return rows


# =====================================================================
# 物体与摆放
# =====================================================================
@dataclass
class Obj:
    cutout_id: str
    category: str
    source: str
    rgba: np.ndarray        # 紧致裁剪后的 (h, w, 4)
    mask: np.ndarray        # alpha >= 阈值
    cx: float               # 完整 mask 在小图内的质心（下标均值 + 0.5）
    cy: float

    @property
    def w(self) -> int:
        return self.rgba.shape[1]

    @property
    def h(self) -> int:
        return self.rgba.shape[0]


def _prepare(lib: CutoutLibrary, cutout_id: str, long_side: int, thr: int) -> Obj:
    rgba = lib.resized(cutout_id, long_side)
    mask = rgba[..., 3] >= thr
    ys, xs = np.nonzero(mask)
    e = lib.by_id[cutout_id]
    return Obj(cutout_id, e["category"], e["source"], rgba, mask, float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)


def _place(o: Obj, cx: float, cy: float) -> tuple[int, int]:
    """返回左上角整数偏移，使物体质心离 (cx, cy) 最近（每个方向误差 ≤ 0.5 px）。"""
    return _round(cx - o.cx), _round(cy - o.cy)


def _centroid(o: Obj, off: tuple[int, int]) -> tuple[float, float]:
    return off[0] + o.cx, off[1] + o.cy


def _box(o: Obj, off: tuple[int, int]) -> list[int]:
    return [off[0], off[1], off[0] + o.w, off[1] + o.h]


def _boxes_apart(a, b, gap: int) -> bool:
    return a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]


def _paste(canvas: np.ndarray, o: Obj, off: tuple[int, int]) -> None:
    """按 alpha 混合（整数运算，四舍五入），原地修改 canvas。"""
    x, y = off
    region = canvas[y:y + o.h, x:x + o.w].astype(np.uint16)
    a = o.rgba[..., 3:4].astype(np.uint16)
    canvas[y:y + o.h, x:x + o.w] = ((o.rgba[..., :3].astype(np.uint16) * a + region * (255 - a) + 127) // 255).astype(np.uint8)


def _full_mask(o: Obj, off: tuple[int, int], size: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[off[1]:off[1] + o.h, off[0]:off[0] + o.w] = o.mask
    return m


def _rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def _group_of(cfg: dict) -> dict[str, str]:
    g = {}
    for grp in cfg.get("confusable_groups") or []:
        for c in grp:
            g[c] = "|".join(grp)
    return g


class _Retry(Exception):
    pass


def _try_scene(cfg: dict, lib: CutoutLibrary, row: dict, sseed: int, attempt: int):
    C = cfg["canvas"]
    thr = cfg["alpha_threshold"]
    pl = cfg["placement"]
    rng = np.random.default_rng([sseed, attempt])
    cats = lib.categories(cfg["cutout_split"])
    grp = _group_of(cfg)
    n_dis = row["n_distractors"]

    # ---- 1. 类别与切图：按随机顺序扫类别，跳过与已选类别同组的；顺序 = target, reference, 替换 target, 干扰物... ----
    names = sorted(cats)
    chosen, used = [], set()
    for i in rng.permutation(len(names)):
        c = names[int(i)]
        g = grp.get(c, c)
        if g not in used:
            used.add(g)
            chosen.append(c)
        if len(chosen) == 3 + n_dis:
            break
    if len(chosen) < 3 + n_dis:
        raise ValueError(f"切图库（split={cfg['cutout_split']}）里互不混淆的类别不够 {3 + n_dis} 个")
    ids = [cats[c][int(rng.integers(len(cats[c])))] for c in chosen]
    sizes = cfg["factors"]["size_frac"]
    dis_fracs = [sizes[int(rng.integers(len(sizes)))] for _ in range(n_dis)]
    T = _prepare(lib, ids[0], row["size_px"], thr)
    R = _prepare(lib, ids[1], row["size_px"], thr)
    T2 = _prepare(lib, ids[2], row["size_px"], thr)
    dis = [_prepare(lib, i, size_px(fr, C), thr) for i, fr in zip(ids[3:], dis_fracs)]

    # ---- 2. 抖动、背景参数 ----
    d = row["distance_px"]
    J = int(math.floor(min(cfg["jitter"]["max_frac_of_distance"] * d, cfg["jitter"]["max_px"])))
    jit = int(rng.integers(-J, J + 1))
    bg = backgrounds.sample_background(rng, row["background"])
    vx, vy = (row["sign"] * d, jit) if row["axis"] == "horizontal" else (jit, row["sign"] * d)

    # ---- 3. 相对布局（reference 的目标质心先放在原点；整数平移不改变取整结果）----
    r0 = _place(R, 0.0, 0.0)
    cr = _centroid(R, r0)
    t0 = _place(T, cr[0] + vx, cr[1] + vy)
    r1 = _place(R, cr[0] + vx, cr[1] + vy)             # position_swap：reference 去 target 的目标质心处
    cr1 = _centroid(R, r1)
    t1 = _place(T, cr1[0] - vx, cr1[1] - vy)           # target 去 reference 原来的质心处（位移向量取反）
    t2 = _place(T2, cr[0] + vx, cr[1] + vy)            # identity_swap：新 target 的质心放在原 target 的目标质心处
    main = [(R, r0), (T, t0), (R, r1), (T, t1), (T2, t2)]
    boxes = [_box(o, off) for o, off in main]
    m = pl["canvas_margin_px"]
    lo_x, hi_x = m - min(b[0] for b in boxes), C - m - max(b[2] for b in boxes)
    lo_y, hi_y = m - min(b[1] for b in boxes), C - m - max(b[3] for b in boxes)
    if lo_x > hi_x or lo_y > hi_y:
        raise _Retry("pair_does_not_fit")
    # ---- 4. 整对物体在画布上的随机位置（三张图的主物体都完整落在画布内的所有整数平移里均匀抽）----
    g = (int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1)))
    sh = lambda off: (off[0] + g[0], off[1] + g[1])  # noqa: E731
    r0, t0, r1, t1, t2 = sh(r0), sh(t0), sh(r1), sh(t1), sh(t2)
    main_boxes = [_box(o, off) for o, off in [(R, r0), (T, t0), (R, r1), (T, t1), (T2, t2)]]

    # ---- 5. 干扰物：框与所有主物体位置（含孪生图）及其他干扰物相隔 ≥ gap ----
    gap = pl["distractor_gap_px"]
    placed = []
    for o in dis:
        for _ in range(pl["max_distractor_tries"]):
            off = (int(rng.integers(m, C - m - o.w + 1)), int(rng.integers(m, C - m - o.h + 1)))
            b = _box(o, off)
            if all(_boxes_apart(b, x, gap) for x in main_boxes) and all(_boxes_apart(b, _box(p, q), gap) for p, q in placed):
                placed.append((o, off))
                break
        else:
            raise _Retry("distractor_no_room")

    # ---- 6. 三张图 ----
    variants = {"base": (T, t0, R, r0), "position_swap": (T, t1, R, r1), "identity_swap": (T2, t2, R, r0)}
    bg_img = backgrounds.render_background(bg, C)
    for o, off in placed:
        _paste(bg_img, o, off)
    out = {}
    for name, (to, toff, ro, roff) in variants.items():
        mt, mr = _full_mask(to, toff, C), _full_mask(ro, roff, C)
        inter = int((mt & mr).sum())
        top = row["top"]
        lower_area = int((mr if top == "target" else mt).sum())
        occ = inter / lower_area
        if 1.0 - occ < cfg["occlusion"]["min_visible_fraction"]:
            raise _Retry("lower_object_hidden")
        img = bg_img.copy()
        order = [("reference", ro, roff), ("target", to, toff)] if top == "target" else [("target", to, toff), ("reference", ro, roff)]
        for _, o, off in order:
            _paste(img, o, off)
        vis = {"target": mt if top == "target" else mt & ~mr, "reference": mr if top == "reference" else mr & ~mt}
        ctc, crc = _centroid(to, toff), _centroid(ro, roff)
        dx, dy = ctc[0] - crc[0], ctc[1] - crc[1]
        along, perp = (dx, dy) if row["axis"] == "horizontal" else (dy, dx)
        relation = RELATIONS[row["axis"]][0 if along < 0 else 1]
        objects = {}
        for role, o, off, full in (("target", to, toff, mt), ("reference", ro, roff, mr)):
            v = vis[role]
            c = _centroid(o, off)
            objects[role] = {
                "cutout_id": o.cutout_id, "category": o.category, "source": o.source,
                "offset_xy": [off[0], off[1]], "long_side_px": max(o.w, o.h),
                "box_xyxy": _box(o, off), "centroid_xy": [round(c[0], 4), round(c[1], 4)],
                "area": int(full.sum()), "mask_rle": _rle(full),
                "visible_box_xyxy": mask_bbox(v), "visible_area": int(v.sum()), "visible_mask_rle": _rle(v)}
        bt, br = objects["target"]["box_xyxy"], objects["reference"]["box_xyxy"]
        bi = max(0, min(bt[2], br[2]) - max(bt[0], br[0])) * max(0, min(bt[3], br[3]) - max(bt[1], br[1]))
        union = objects["target"]["area"] + objects["reference"]["area"] - inter
        meta = {
            "variant": name,
            "relation": relation, "axis": row["axis"],
            "design": {k: row[k] for k in ("distance_px", "size_frac", "size_px", "axis", "n_distractors", "background", "top", "cell", "replicate")},
            "distance": {"requested_px": d, "along_axis_px": round(abs(along), 4), "error_px": round(abs(along) - d, 4),
                         "jitter_requested_px": jit if name != "position_swap" else -jit, "jitter_max_px": J,
                         "perpendicular_px": round(perp, 4), "euclidean_px": round(math.hypot(dx, dy), 4)},
            "pair_position": {"translation_xy": [g[0], g[1]], "feasible_x": [lo_x, hi_x], "feasible_y": [lo_y, hi_y],
                              "pair_center_xy": [round((ctc[0] + crc[0]) / 2, 4), round((ctc[1] + crc[1]) / 2, 4)]},
            "objects": objects,
            "z_order": [f"distractor_{i}" for i in range(len(placed))] + [o[0] for o in order],
            "occlusion": {"occluded": inter > 0, "top": top, "bottom": "reference" if top == "target" else "target",
                          "intersection_px": inter, "occluded_fraction": round(occ, 6),
                          "iou_complete": round(inter / union, 6), "iou_visible": 0.0,
                          "bbox_iou_complete": round(bi / ((bt[2] - bt[0]) * (bt[3] - bt[1]) + (br[2] - br[0]) * (br[3] - br[1]) - bi), 6)},
            "distractors": [{"cutout_id": o.cutout_id, "category": o.category, "source": o.source, "size_frac": fr,
                             "long_side_px": max(o.w, o.h), "offset_xy": [off[0], off[1]], "box_xyxy": _box(o, off),
                             "centroid_xy": [round(off[0] + o.cx, 4), round(off[1] + o.cy, 4)],
                             "area": int(o.mask.sum()), "mask_rle": _rle(_full_mask(o, off, C))}
                            for (o, off), fr in zip(placed, dis_fracs)],
            "background": bg,
        }
        if name == "identity_swap":
            meta["replaced"] = {"role": "target", "original_cutout_id": T.cutout_id, "original_category": T.category}
        out[name] = (meta, img)
    if out["position_swap"][0]["relation"] != OPPOSITE[out["base"][0]["relation"]] or out["base"][0]["relation"] != row["relation"]:
        raise AssertionError("关系标签与设计不一致（不应发生）")
    return out


def scene_ids(version: str, index: int) -> dict[str, str]:
    return {"base": f"{version}_{index:06d}_base", "position_swap": f"{version}_{index:06d}_pos",
            "identity_swap": f"{version}_{index:06d}_id"}


def generate_scene(cfg: dict, lib: CutoutLibrary, index: int) -> list[dict]:
    """生成第 index 个基础场景及其两张孪生图。返回 3 个 dict：{"meta": ..., "png": bytes, "image": ndarray}。

    只依赖 (cfg, 切图库, index)，与生成顺序、进程数无关。
    """
    row = build_design(cfg)[index]
    sseed = scene_seed(cfg["global_seed"], index)
    reasons = []
    for attempt in range(cfg["placement"]["max_attempts"]):
        try:
            res = _try_scene(cfg, lib, row, sseed, attempt)
            break
        except _Retry as e:
            reasons.append(str(e))
    else:
        raise RuntimeError(f"场景 {index} 重采样 {cfg['placement']['max_attempts']} 次仍失败：{reasons}")
    ids = scene_ids(cfg["version"], index)
    out = []
    for name in VARIANTS:
        body, img = res[name]
        png = png_bytes(img, cfg["png_compress_level"])
        meta = {"scene_id": ids[name], "base_id": ids["base"], "version": cfg["version"], "index": index,
                "twins": {k: v for k, v in ids.items() if k != name},
                "image": f"images/{ids[name]}.png", "canvas_wh": [cfg["canvas"], cfg["canvas"]],
                "image_sha256": hashlib.sha256(png).hexdigest(),
                "pixels_sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                "seed": {"global_seed": cfg["global_seed"], "scene_seed": sseed, "attempt": attempt, "rejected": reasons},
                "cutout_split": cfg["cutout_split"], "cutout_library_sha256": lib.sha256}
        meta.update(body)
        out.append({"meta": meta, "png": png, "image": img})
    return out


def dumps(meta: dict) -> str:
    """metadata 的规范序列化（一行）。"""
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
