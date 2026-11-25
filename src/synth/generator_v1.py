"""参数化合成场景生成器 v1（trial_v1）。与 v0（generator.py）的区别：间距 = 边缘间距；重叠单独成子集；尺寸按 sqrt(mask 面积)；
身份互换对齐质心并保持面积；抠图库按图像划分、场景继承其侧；边距 16 px。

精确定义：
  坐标 / 质心 / 完整 mask / 可见 mask / z 顺序  同 generator.py。
  投影间距 projection_gap_px   两个完整 mask 的框在关系轴上投影区间之间的空隙（整数；相接为 0；重叠为负）。主网格的控制量。
  最短间距 min_distance_px     两个完整 mask 像素之间的最短欧氏距离 − 1（相邻像素为 0；重叠为 0）。
  尺寸       sqrt(完整 mask 面积)，目标 floor(size_frac × 画布 + 0.5)，实际值在 objects.*.sqrt_area。放大倍数 = 缩放系数（相对原图像素）。
  主网格布局  先 / 后两个物体在轴上占 [0, wA) 和 [wA+gap, wA+gap+wB)，垂直方向质心差 = jitter。
  position_swap  主网格：占位区间对调（后者移到 [0, wB)，前者移到 [wB+gap, ...)），间距不变，垂直方向质心对调；
                 重叠子集：两个质心对调，上层物体不变，遮挡比例须仍在同一档。
  identity_swap  target 换成另一类别、sqrt(面积) 相同的抠图，质心对齐原 target；主网格要求与 reference 投影间距 ≥ 0，
                 重叠子集要求遮挡比例仍在同一档。
  遮挡比例    |下层 ∩ 上层| / |下层|。
随机性：scene_seed = sha256("<version>:<global_seed>:<index>") 前 8 字节；第 a 次尝试 default_rng([scene_seed, a])。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from src import paths

from . import backgrounds
from .cutouts import mask_bbox, png_bytes
from .generator import OPPOSITE, RELATIONS, VARIANTS, _balanced_assign, _boxes_apart, _group_of, _Retry, _rle, _round, size_px


class LibraryV1:
    """抠图库 v1 的只读视图；sha256 必须与 splits/synth_library_v1.json 一致。"""

    def __init__(self, root=None, entries=None, arrays=None):
        from src.data.splits import read_split
        self._arrays = dict(arrays or {})
        if entries is None:
            self.root = root or paths.get("data") / "synth" / "cutouts_v1"
            raw = (self.root / "library.jsonl").read_bytes()
            self.sha256 = hashlib.sha256(raw).hexdigest()
            sp = read_split("synth_library_v1")
            if sp["library_sha256"] != self.sha256:
                raise RuntimeError("抠图库 v1 与 splits/synth_library_v1.json 的 sha256 不一致")
            entries = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        else:
            self.root, self.sha256 = None, hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        self.entries = sorted(entries, key=lambda e: e["cutout_id"])
        self.by_id = {e["cutout_id"]: e for e in self.entries}
        self._prep: dict = {}

    def categories(self, side: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in self.entries:
            if e["split"] == side:
                out.setdefault(e["category"], []).append(e["cutout_id"])
        return {k: out[k] for k in sorted(out)}

    def rgba(self, cid: str) -> np.ndarray:
        if cid not in self._arrays:
            self._arrays[cid] = np.asarray(Image.open(self.root / "png" / f"{cid}.png").convert("RGBA"))
        return self._arrays[cid]


@dataclass
class Obj:
    cutout_id: str
    category: str
    rgba: np.ndarray
    mask: np.ndarray
    cx: float
    cy: float
    area: int
    scale: float

    @property
    def w(self):
        return self.rgba.shape[1]

    @property
    def h(self):
        return self.rgba.shape[0]

    def ext(self, axis):
        return self.w if axis == "horizontal" else self.h


def _resize(rgba: np.ndarray, s: float, thr: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    nw, nh = max(1, _round(w * s)), max(1, _round(h * s))
    rgb = Image.fromarray(np.ascontiguousarray(rgba[..., :3])).resize((nw, nh), Image.LANCZOS)
    a = Image.fromarray(np.ascontiguousarray(rgba[..., 3])).resize((nw, nh), Image.BILINEAR)
    out = np.dstack([np.asarray(rgb), np.asarray(a)])
    x1, y1, x2, y2 = mask_bbox(out[..., 3] >= thr)
    return np.ascontiguousarray(out[y1:y2, x1:x2])


def prepare(lib: LibraryV1, cid: str, sqrt_area: int, thr: int, max_upscale: float) -> Obj:
    key = (cid, sqrt_area)
    if key not in lib._prep:
        src = lib.rgba(cid)
        a0 = int((src[..., 3] >= thr).sum())
        s = sqrt_area / math.sqrt(a0)
        best = None
        for _ in range(5):                         # 取整会让面积略偏，最多修正几次缩放系数
            r = _resize(src, s, thr)
            got = math.sqrt(int((r[..., 3] >= thr).sum()))
            if best is None or abs(got - sqrt_area) < abs(best[2] - sqrt_area):
                best = (r, s, got)
            if abs(got - sqrt_area) < 0.2:
                break
            s *= sqrt_area / got
        r, s, _ = best
        m = r[..., 3] >= thr
        ys, xs = np.nonzero(m)
        lib._prep[key] = Obj(cid, lib.by_id[cid]["category"], r, m, float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, int(m.sum()), s)
    o = lib._prep[key]
    if o.scale > max_upscale:
        raise ValueError(f"{cid} 在 sqrt_area={sqrt_area} 时放大 {o.scale:.3f} 倍，超过上限（库的过滤应保证不会发生）")
    return o


def _box(o, off):
    return [off[0], off[1], off[0] + o.w, off[1] + o.h]


def _cen(o, off):
    return off[0] + o.cx, off[1] + o.cy


def _inter(a: Obj, pa, b: Obj, pb) -> int:
    x1, y1 = max(pa[0], pb[0]), max(pa[1], pb[1])
    x2, y2 = min(pa[0] + a.w, pb[0] + b.w), min(pa[1] + a.h, pb[1] + b.h)
    if x1 >= x2 or y1 >= y2:
        return 0
    return int((a.mask[y1 - pa[1]:y2 - pa[1], x1 - pa[0]:x2 - pa[0]] & b.mask[y1 - pb[1]:y2 - pb[1], x1 - pb[0]:x2 - pb[0]]).sum())


def _proj_gap(ba, bb, axis) -> int:
    k = 0 if axis == "horizontal" else 1
    return max(ba[k], bb[k]) - min(ba[k + 2], bb[k + 2])


def _full(o, off, C):
    m = np.zeros((C, C), bool)
    m[off[1]:off[1] + o.h, off[0]:off[0] + o.w] = o.mask
    return m


def _paste(canvas, o, off):
    x, y = off
    reg = canvas[y:y + o.h, x:x + o.w].astype(np.uint16)
    a = o.rgba[..., 3:4].astype(np.uint16)
    canvas[y:y + o.h, x:x + o.w] = ((o.rgba[..., :3].astype(np.uint16) * a + reg * (255 - a) + 127) // 255).astype(np.uint8)


# ===================================================================== 设计表
def levels(cfg) -> list[dict]:
    f = cfg["factors"]
    return [{"subset": "main", "name": f"gap{g}", "gap_px": g, "tier": None} for g in f["gap_px"]] + \
           [{"subset": "overlap", "name": f"occ{int(round(hi * 100))}", "gap_px": None, "tier": [lo, hi]} for lo, hi in f["occlusion_tiers"]]


_CACHE: dict = {}


def build_design(cfg) -> list[dict]:
    """间距档 × 尺寸 × 干扰物数 的循环与 v0 相同（档数 11、尺寸 3、干扰物 4 两两互素）；背景 / 上层 / 方向 / 侧 在层内均衡分配。"""
    f = cfg["factors"]
    key = json.dumps([cfg["version"], cfg["global_seed"], cfg["n_scenes"], cfg["canvas"], f], sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    L, S, N, A = levels(cfg), f["size_frac"], f["n_distractors"], f["axis"]
    tri = len(L) * len(S) * len(N)
    rows = []
    for i in range(cfg["n_scenes"]):
        k = i % (tri * len(A))
        kk = k % tri
        rows.append({"index": i, "cell": k, "replicate": i // (tri * len(A)), "level": L[k % len(L)]["name"], **{x: L[k % len(L)][x] for x in ("subset", "gap_px", "tier")},
                     "size_frac": S[k % len(S)], "sqrt_area_px": size_px(S[k % len(S)], cfg["canvas"]), "n_distractors": N[k % len(N)],
                     "axis": A[(kk // len(N) + k // tri) % len(A)]})
    assert math.gcd(len(L), len(S)) == math.gcd(len(L), len(N)) == math.gcd(len(S), len(N)) == 1
    rng = np.random.default_rng([cfg["global_seed"], 0xF41])

    def strata(fn):
        s: dict = {}
        for r in rows:
            s.setdefault(fn(r), []).append(r["index"])
        return s

    _balanced_assign(strata(lambda r: r["level"]), list(f["background"]), rng, rows, "background")
    _balanced_assign(strata(lambda r: (r["level"], r["size_frac"])), ["target", "reference"], rng, rows, "top")
    _balanced_assign(strata(lambda r: (r["axis"], r["level"])), [-1, 1], rng, rows, "sign")
    _balanced_assign(strata(lambda r: (r["level"], r["axis"])), ["dev", "test"], rng, rows, "side")
    for r in rows:
        r["relation"] = RELATIONS[r["axis"]][0 if r["sign"] < 0 else 1]
    _CACHE[key] = rows
    return rows


# ===================================================================== 一个场景
def _xy(axis, along, perp):
    return (along, perp) if axis == "horizontal" else (perp, along)


def _cc(o, axis):      # (沿轴质心, 垂直质心)，相对左上角
    return (o.cx, o.cy) if axis == "horizontal" else (o.cy, o.cx)


def _at(o, axis, start_along, perp_centroid):
    """沿轴从 start_along 开始、垂直质心最接近 perp_centroid 的整数左上角。"""
    return _xy(axis, start_along, _round(perp_centroid - _cc(o, axis)[1]))


def _occ(lower, pl, upper, pu):
    return _inter(lower, pl, upper, pu) / lower.area


def _layout_main(row, T, R, rng, cfg):
    axis, gap, sg = row["axis"], row["gap_px"], row["sign"]
    first, second = (T, R) if sg < 0 else (R, T)
    d_c = first.ext(axis) - _cc(first, axis)[0] + gap + _cc(second, axis)[0]
    J = int(math.floor(min(cfg["jitter"]["max_frac_of_distance"] * d_c, cfg["jitter"]["max_px"])))
    jit = int(rng.integers(-J, J + 1))
    p1, p2 = 0.0, float(jit)
    o1, o2 = _at(first, axis, 0, p1), _at(second, axis, first.ext(axis) + gap, p2)
    # position_swap：区间对调，垂直质心对调
    s2, s1 = _at(second, axis, 0, p1), _at(first, axis, second.ext(axis) + gap, p2)
    t0, r0 = (o1, o2) if sg < 0 else (o2, o1)
    t1, r1 = (s1, s2) if sg < 0 else (s2, s1)
    return t0, r0, t1, r1, jit, J


def _layout_overlap(row, T, R, rng, cfg):
    axis, sg, (lo, hi) = row["axis"], row["sign"], row["tier"]
    u = float(rng.uniform(lo, hi))
    mx = T.ext(axis) + R.ext(axis)
    ratio = float(rng.uniform(-1, 1))
    cand = []
    for d in range(8, mx):
        J = int(math.floor(min(cfg["jitter"]["max_frac_of_distance"] * d, cfg["jitter"]["max_px"])))
        jit = _round(ratio * J)
        if d < cfg["overlap"]["min_axis_over_perp"] * abs(jit):
            continue
        v = _xy(axis, sg * d, jit)
        r0 = (_round(-R.cx), _round(-R.cy))
        cr = _cen(R, r0)
        t0 = (_round(cr[0] + v[0] - T.cx), _round(cr[1] + v[1] - T.cy))
        ct = _cen(T, t0)
        r1 = (_round(ct[0] - R.cx), _round(ct[1] - R.cy))      # 质心对调
        t1 = (_round(cr[0] - T.cx), _round(cr[1] - T.cy))
        lw = (lambda t, r: _occ(R, r, T, t)) if row["top"] == "target" else (lambda t, r: _occ(T, t, R, r))
        a, b = lw(t0, r0), lw(t1, r1)
        if lo <= a < hi and lo <= b < hi:
            cand.append((abs(a - u), d, t0, r0, t1, r1, jit, J))
        if a == 0 and b == 0 and d > mx / 2:
            break
    if not cand:
        raise _Retry("no_overlap_offset")
    _, _, t0, r0, t1, r1, jit, J = min(cand)
    return t0, r0, t1, r1, jit, J


def _try(cfg, lib, row, sseed, attempt):
    C, thr, axis = cfg["canvas"], cfg["alpha_threshold"], row["axis"]
    mg = cfg["placement"]["canvas_margin_px"]
    rng = np.random.default_rng([sseed, attempt])
    cats = lib.categories(row["side"])
    grp = _group_of(cfg)
    names = sorted(cats)
    order = [names[int(i)] for i in rng.permutation(len(names))]
    chosen, used = [], set()
    for c in order:
        if grp.get(c, c) not in used:
            used.add(grp.get(c, c))
            chosen.append(c)
        if len(chosen) == 2 + row["n_distractors"]:
            break
    if len(chosen) < 2 + row["n_distractors"]:
        raise ValueError(f"{row['side']} 侧互不混淆的类别不够")
    pick = lambda c: cats[c][int(rng.integers(len(cats[c])))]      # noqa: E731
    P = lambda cid, s: prepare(lib, cid, s, thr, cfg["max_upscale"])      # noqa: E731
    T, R = P(pick(chosen[0]), row["sqrt_area_px"]), P(pick(chosen[1]), row["sqrt_area_px"])
    sizes = cfg["factors"]["size_frac"]
    dis = [(P(pick(c), size_px(fr, C)), fr) for c, fr in ((c, sizes[int(rng.integers(len(sizes)))]) for c in chosen[2:])]
    bg = backgrounds.sample_background(rng, row["background"])
    t0, r0, t1, r1, jit, J = (_layout_main if row["subset"] == "main" else _layout_overlap)(row, T, R, rng, cfg)

    # ---- 身份互换：候选 = 其它未用类别（不与 target/reference/干扰物同组）的抠图，随机顺序 ----
    pool = [cid for c in names if grp.get(c, c) not in used for cid in cats[c]]
    if not pool:
        raise ValueError("没有可用于身份互换的类别")
    ct = _cen(T, t0)
    T2 = t2 = None
    for i in rng.permutation(len(pool))[:cfg["placement"]["max_swap_candidates"]]:
        o = P(pool[int(i)], row["sqrt_area_px"])
        off = (_round(ct[0] - o.cx), _round(ct[1] - o.cy))
        if row["subset"] == "main":
            ok = _proj_gap(_box(o, off), _box(R, r0), axis) >= 0
        else:
            f = _occ(R, r0, o, off) if row["top"] == "target" else _occ(o, off, R, r0)
            ok = row["tier"][0] <= f < row["tier"][1]
        if ok:
            T2, t2 = o, off
            break
    if T2 is None:
        raise _Retry("no_identity_candidate")

    main = [(R, r0), (T, t0), (R, r1), (T, t1), (T2, t2)]
    bx = [_box(o, p) for o, p in main]
    lo = [mg - min(b[k] for b in bx) for k in (0, 1)]
    hi = [C - mg - max(b[k + 2] for b in bx) for k in (0, 1)]
    if lo[0] > hi[0] or lo[1] > hi[1]:
        raise _Retry("pair_does_not_fit")
    g = (int(rng.integers(lo[0], hi[0] + 1)), int(rng.integers(lo[1], hi[1] + 1)))
    sh = lambda p: (p[0] + g[0], p[1] + g[1])      # noqa: E731
    r0, t0, r1, t1, t2 = map(sh, (r0, t0, r1, t1, t2))
    main_boxes = [_box(o, p) for o, p in ((R, r0), (T, t0), (R, r1), (T, t1), (T2, t2))]
    placed, gp = [], cfg["placement"]["distractor_gap_px"]
    for o, fr in dis:
        for _ in range(cfg["placement"]["max_distractor_tries"]):
            off = (int(rng.integers(mg, C - mg - o.w + 1)), int(rng.integers(mg, C - mg - o.h + 1)))
            b = _box(o, off)
            if all(_boxes_apart(b, x, gp) for x in main_boxes) and all(_boxes_apart(b, _box(q, w), gp) for q, w, _ in placed):
                placed.append((o, off, fr))
                break
        else:
            raise _Retry("distractor_no_room")
    bg_img = backgrounds.render_background(bg, C)
    for o, off, _ in placed:
        _paste(bg_img, o, off)
    out = {}
    cells = cfg["token_cell_px"]
    for name, (to, tp, ro, rp) in {"base": (T, t0, R, r0), "position_swap": (T, t1, R, r1), "identity_swap": (T2, t2, R, r0)}.items():
        mt, mr = _full(to, tp, C), _full(ro, rp, C)
        inter = int((mt & mr).sum())
        top = row["top"]
        occ = inter / (ro.area if top == "target" else to.area)
        bt, br = _box(to, tp), _box(ro, rp)
        pg = _proj_gap(bt, br, axis)
        md = 0.0 if inter else max(0.0, float(ndimage.distance_transform_edt(~mr)[mt].min()) - 1.0)
        if row["subset"] == "main":
            assert inter == 0 and pg >= 0 and (name == "identity_swap" or pg == row["gap_px"]), (row["index"], name, pg)
        else:
            assert row["tier"][0] <= occ < row["tier"][1], (row["index"], name, occ)
        img = bg_img.copy()
        zo = [("reference", ro, rp), ("target", to, tp)] if top == "target" else [("target", to, tp), ("reference", ro, rp)]
        for _, o, off in zo:
            _paste(img, o, off)
        cA, cB = _cen(to, tp), _cen(ro, rp)
        dx, dy = cA[0] - cB[0], cA[1] - cB[1]
        along, perp = (dx, dy) if axis == "horizontal" else (dy, dx)
        relation = RELATIONS[axis][0 if along < 0 else 1]
        if abs(along) <= abs(perp):
            raise _Retry("axis_not_dominant")
        vis = {"target": mt if top == "target" else mt & ~mr, "reference": mr if top == "reference" else mr & ~mt}
        objs = {}
        for role, o, off, full in (("target", to, tp, mt), ("reference", ro, rp, mr)):
            b = _box(o, off)
            assert min(b[0], b[1]) >= mg and max(b[2], b[3]) <= C - mg
            objs[role] = {"cutout_id": o.cutout_id, "category": o.category, "offset_xy": list(off), "box_xyxy": b, "centroid_xy": [round(v, 4) for v in _cen(o, off)],
                          "area": o.area, "sqrt_area": round(math.sqrt(o.area), 4), "upscale": round(o.scale, 4), "mask_rle": _rle(full),
                          "visible_box_xyxy": mask_bbox(vis[role]), "visible_area": int(vis[role].sum()), "visible_mask_rle": _rle(vis[role])}
        meta = {"variant": name, "side": row["side"], "subset": row["subset"], "level": row["level"], "relation": relation, "axis": axis,
                "design": {k: row[k] for k in ("level", "subset", "gap_px", "tier", "size_frac", "sqrt_area_px", "axis", "n_distractors", "background", "top", "side", "cell", "replicate")},
                "gap": {"requested_px": row["gap_px"], "projection_gap_px": pg, "min_distance_px": round(md, 4),
                        "center_along_px": round(abs(along), 4), "center_perp_px": round(perp, 4), "center_euclidean_px": round(math.hypot(dx, dy), 4),
                        "jitter_px": jit, "jitter_max_px": J,
                        "projection_gap_tokens": {k: round(pg / c, 4) for k, c in cells.items()}, "center_along_tokens": {k: round(abs(along) / c, 4) for k, c in cells.items()}},
                "objects": objs, "z_order": [f"distractor_{i}" for i in range(len(placed))] + [z[0] for z in zo],
                "occlusion": {"occluded": inter > 0, "top": top, "bottom": "reference" if top == "target" else "target", "intersection_px": inter, "occluded_fraction": round(occ, 6)},
                "pair_translation_xy": list(g),
                "distractors": [{"cutout_id": o.cutout_id, "category": o.category, "size_frac": fr, "box_xyxy": _box(o, off), "area": o.area, "upscale": round(o.scale, 4), "mask_rle": _rle(_full(o, off, C))} for o, off, fr in placed],
                "background": bg}
        if name == "identity_swap":
            dev = math.hypot(cA[0] - (t0[0] + T.cx), cA[1] - (t0[1] + T.cy))
            assert dev <= cfg["identity_swap"]["max_centroid_dev_px"]
            meta["identity_swap"] = {"original_cutout_id": T.cutout_id, "original_category": T.category, "centroid_deviation_px": round(dev, 4),
                                     "area_ratio": round(to.area / T.area, 5), "sqrt_area_ratio": round(math.sqrt(to.area / T.area), 5)}
        if name == "position_swap":
            b0 = {"target": _cen(T, t0), "reference": _cen(R, r0)}
            meta["position_swap"] = {"target_minus_base_reference_centroid": [round(cA[k] - b0["reference"][k], 4) for k in (0, 1)],
                                     "reference_minus_base_target_centroid": [round(cB[k] - b0["target"][k], 4) for k in (0, 1)]}
        out[name] = (meta, img)
    if out["position_swap"][0]["relation"] != OPPOSITE[out["base"][0]["relation"]] or out["base"][0]["relation"] != row["relation"] or out["identity_swap"][0]["relation"] != row["relation"]:
        raise _Retry("relation_mismatch")
    return out


def scene_ids(version, index):
    return {"base": f"{version}_{index:06d}_base", "position_swap": f"{version}_{index:06d}_pos", "identity_swap": f"{version}_{index:06d}_id"}


def generate_scene(cfg, lib, index) -> list[dict]:
    row = build_design(cfg)[index]
    sseed = int.from_bytes(hashlib.sha256(f"{cfg['version']}:{cfg['global_seed']}:{index}".encode()).digest()[:8], "big")
    rej: dict[str, int] = {}
    for attempt in range(cfg["placement"]["max_attempts"]):
        try:
            res = _try(cfg, lib, row, sseed, attempt)
            break
        except _Retry as e:
            rej[str(e)] = rej.get(str(e), 0) + 1
    else:
        raise RuntimeError(f"场景 {index}（{row['level']}）重采样 {cfg['placement']['max_attempts']} 次仍失败：{rej}")
    ids = scene_ids(cfg["version"], index)
    out = []
    for name in VARIANTS:
        body, img = res[name]
        png = png_bytes(img, cfg["png_compress_level"])
        out.append({"meta": {"scene_id": ids[name], "base_id": ids["base"], "version": cfg["version"], "index": index, "twins": {k: v for k, v in ids.items() if k != name},
                             "image": f"images/{ids[name]}.png", "canvas_wh": [cfg["canvas"]] * 2, "image_sha256": hashlib.sha256(png).hexdigest(),
                             "seed": {"global_seed": cfg["global_seed"], "scene_seed": sseed, "attempt": attempt, "rejected": rej}, "cutout_library_sha256": lib.sha256, **body}, "png": png})
    return out
