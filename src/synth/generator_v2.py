"""参数化合成场景生成器 v2（configs/synth/synth_v2.yaml）。复用 v1 的抠图准备、粘贴、遮挡与身份互换逻辑，改动：
  主集的控制量 = 质心沿轴距离（按层分层、层内对数均匀），边缘间距是结果（>= 0）；尺寸 4 档（36 px 档只用 small36_ok）；
  身份互换不再重试到满足条件，而是不满足时照常生成并标 identity_swap_flag；侧别按 dev:test = 1:2 分配；每个场景记录各 token 边长下的距离与分档。
  dry_run=True 时只算几何（含 mask 最短距离），不合成图像。"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy import ndimage

from src import paths

from . import backgrounds
from . import generator_v1 as g1
from .cutouts import mask_bbox, png_bytes
from .generator import OPPOSITE, RELATIONS, VARIANTS, _balanced_assign, _boxes_apart, _group_of, _Retry, _rle, _round


class LibraryV3(g1.LibraryV1):
    def __init__(self):
        from src.data.splits import read_split
        sp = read_split("synth_library_v3")
        root = paths.get("data") / "synth" / "cutouts_v3"
        raw = (root / "library_final_r2.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != sp["library_sha256"]:
            raise RuntimeError("抠图库 v3 与 splits/synth_library_v3.json 的 sha256 不一致")
        super().__init__(entries=[json.loads(l) for l in raw.decode().splitlines() if l.strip()])
        self.root, self.sha256 = root, sp["library_sha256"]

    def categories(self, side, small36=False):
        out = {}
        for e in self.entries:
            if e["split"] == side and (not small36 or e["small36_ok"]):
                out.setdefault(e["category"], []).append(e["cutout_id"])
        return {k: out[k] for k in sorted(out)}


def token_edges(cfg):
    b = cfg["token_bins"]
    return [b["base"] * b["ratio"] ** k for k in range(b["n_edges"])]


def token_bin(d_px, px, edges):
    t = d_px / px
    for k in range(len(edges) - 1):
        if edges[k] <= t < edges[k + 1]:
            return k
    return None


_CACHE: dict = {}


def build_design(cfg):
    D = cfg["design"]
    key = json.dumps([cfg["version"], cfg["global_seed"], D], sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    rows = []
    S = D["main_strata_px"]
    for j in range(len(S) - 1):
        lo, hi = S[j], S[j + 1]
        sizes = [s for s in cfg["sizes_px"] if s <= hi] or [min(cfg["sizes_px"])]
        for i in range(D["main_scenes_per_stratum"]):
            rows.append({"subset": "main", "level": f"d{j}", "stratum_px": [lo, hi], "tier": None, "sqrt_area_px": sizes[i % len(sizes)]})
    for lo, hi in D["overlap_tiers"]:
        for i in range(D["overlap_scenes_per_tier"]):
            rows.append({"subset": "overlap", "level": f"occ{int(round(hi * 100))}", "stratum_px": None, "tier": [lo, hi], "sqrt_area_px": cfg["sizes_px"][i % len(cfg["sizes_px"])]})
    n_base_rows = len(rows)
    rng = np.random.default_rng([cfg["global_seed"], 0xF42])
    for i, r in enumerate(rows):
        r["index"] = i
        r["n_distractors"] = D["n_distractors"][(i // 3) % len(D["n_distractors"])]
        r["axis"] = D["axis"][i % 2]

    def strata(fn):
        s = {}
        for r in rows:
            s.setdefault(fn(r), []).append(r["index"])
        return s
    sides = ["dev"] * D["side_ratio"]["dev"] + ["test"] * D["side_ratio"]["test"]
    _balanced_assign(strata(lambda r: (r["level"], r["axis"])), sides, rng, rows, "side")
    _balanced_assign(strata(lambda r: r["level"]), list(D["background"]), rng, rows, "background")
    _balanced_assign(strata(lambda r: (r["level"], r["sqrt_area_px"])), ["target", "reference"], rng, rows, "top")
    _balanced_assign(strata(lambda r: (r["axis"], r["level"])), [-1, 1], rng, rows, "sign")
    # 补场景：按轮次、按层追加，序号接在已有行之后；只在新加的行里均衡分配，已有行不受影响
    for rd in sorted(D.get("extra_scenes") or {}):
        new = []
        for lv, n in sorted(D["extra_scenes"][rd].items()):
            j = int(lv[1:])
            lo, hi = S[j], S[j + 1]
            sizes = [x for x in cfg["sizes_px"] if x <= hi] or [min(cfg["sizes_px"])]
            for i in range(n):
                new.append({"subset": "main", "level": lv, "stratum_px": [lo, hi], "tier": None, "sqrt_area_px": sizes[i % len(sizes)], "extra_round": rd})
        r2 = np.random.default_rng([cfg["global_seed"], 0xF43, int(rd.replace("round", ""))])
        for i, r in enumerate(new):
            r["index"] = len(rows) + i
            r["n_distractors"] = D["n_distractors"][(i // 3) % len(D["n_distractors"])]
            r["axis"] = D["axis"][i % 2]
        rows.extend(new)
        sub = {r["index"]: r for r in new}

        def st(fn):
            s_ = {}
            for r in new:
                s_.setdefault(fn(r), []).append(r["index"])
            return s_
        tmp = [None] * len(rows)
        for r in rows:
            tmp[r["index"]] = r
        _balanced_assign(st(lambda r: (r["level"], r["axis"])), sides, r2, tmp, "side")
        _balanced_assign(st(lambda r: r["level"]), list(D["background"]), r2, tmp, "background")
        _balanced_assign(st(lambda r: (r["level"], r["sqrt_area_px"])), ["target", "reference"], r2, tmp, "top")
        _balanced_assign(st(lambda r: (r["axis"], r["level"])), [-1, 1], r2, tmp, "sign")
    for r in rows:
        r["relation"] = RELATIONS[r["axis"]][0 if r["sign"] < 0 else 1]
    _CACHE[key] = rows
    return rows


def _min_dc(first, second, axis):
    """边缘间距为 0 时的质心沿轴距离（first 在前）。"""
    return first.ext(axis) - g1._cc(first, axis)[0] + g1._cc(second, axis)[0]


def _layout_main(row, T, R, rng, cfg):
    axis, sg = row["axis"], row["sign"]
    first, second = (T, R) if sg < 0 else (R, T)
    lo, hi = row["stratum_px"]
    dmin = _min_dc(first, second, axis)
    a = max(lo, dmin)
    if a >= hi:
        raise _Retry("stratum_unreachable_for_pair")
    d = math.exp(rng.uniform(math.log(a), math.log(hi)))
    gap = _round(d - dmin)
    if gap < 0:
        gap = 0
    J = int(math.floor(min(cfg["jitter"]["max_frac_of_distance"] * d, cfg["jitter"]["max_px"])))
    jit = int(rng.integers(-J, J + 1))
    o1, o2 = g1._at(first, axis, 0, 0.0), g1._at(second, axis, first.ext(axis) + gap, float(jit))
    s2, s1 = g1._at(second, axis, 0, 0.0), g1._at(first, axis, second.ext(axis) + gap, float(jit))      # 位置互换：区间对调（同 v1）
    t0, r0 = (o1, o2) if sg < 0 else (o2, o1)
    t1, r1 = (s1, s2) if sg < 0 else (s2, s1)
    return t0, r0, t1, r1, jit, J, gap


def _try(cfg, lib, row, sseed, attempt, dry_run):
    C, thr, axis = cfg["canvas"], cfg["alpha_threshold"], row["axis"]
    mg = cfg["placement"]["canvas_margin_px"]
    rng = np.random.default_rng([sseed, attempt])
    size = row["sqrt_area_px"]
    if row["subset"] == "main" and attempt >= cfg["placement"]["max_attempts"] // 2:     # 一半尝试失败后换更小的尺寸
        smaller = [s for s in cfg["sizes_px"] if s < size]
        if smaller:
            size = smaller[(attempt - cfg["placement"]["max_attempts"] // 2) % len(smaller)]
    cats = lib.categories(row["side"], small36=(size == 36))
    allc = lib.categories(row["side"])
    grp = _group_of(cfg)
    names = sorted(cats)
    order = [names[int(i)] for i in rng.permutation(len(names))]
    chosen, used = [], set()
    for c in order:
        if grp.get(c, c) not in used:
            used.add(grp.get(c, c)); chosen.append(c)
        if len(chosen) == 2:
            break
    dnames = [c for c in sorted(allc) if grp.get(c, c) not in used]
    dorder = [dnames[int(i)] for i in rng.permutation(len(dnames))]
    for c in dorder:
        if len(chosen) == 2 + row["n_distractors"]:
            break
        if grp.get(c, c) not in used:
            used.add(grp.get(c, c)); chosen.append(c)
    if len(chosen) < 2 + row["n_distractors"]:
        raise ValueError("互不混淆的类别不够")
    P = lambda cid, s: g1.prepare(lib, cid, s, thr, cfg["max_upscale"])      # noqa: E731
    pick = lambda pool, c: pool[c][int(rng.integers(len(pool[c])))]      # noqa: E731
    T, R = P(pick(cats, chosen[0]), size), P(pick(cats, chosen[1]), size)
    sizes = cfg["sizes_px"]
    dis = []
    allowed = [x for x in sizes if x <= size] if cfg["design"].get("distractor_size_rule") == "le_pair" else list(sizes)
    off = int(rng.integers(len(allowed)))
    for i, c in enumerate(chosen[2:]):
        s = allowed[(off + i) % len(allowed)]
        pool = lib.categories(row["side"], small36=(s == 36))
        if c not in pool:                      # 这一类没有 small36_ok 的抠图：换一个允许的更大尺寸档；都不行就用 36 px 档里有的类别
            bigger = [x for x in allowed if x > s]
            if bigger:
                s, pool = bigger[0], allc
            else:
                continue
        dis.append((P(pick(pool, c), s), s))
    gap_set = None
    if row["subset"] == "main":
        t0, r0, t1, r1, jit, J, gap_set = _layout_main(row, T, R, rng, cfg)
    else:
        t0, r0, t1, r1, jit, J = g1._layout_overlap({**row, "top": row["top"]}, T, R, rng, cfg)
    # 身份互换：其它未用类别的同尺寸抠图，质心对齐原 target；先找满足条件的，找不到就用第一个候选并标 flag
    spool = cats if size == 36 else allc
    pool = [cid for c in sorted(spool) if grp.get(c, c) not in used for cid in spool[c]]
    if not pool:
        raise _Retry("no_identity_pool")
    ct = g1._cen(T, t0)
    first_cand, T2, t2 = None, None, None
    for i in rng.permutation(len(pool))[:cfg["placement"]["max_swap_candidates"]]:
        o = P(pool[int(i)], size)
        off = (_round(ct[0] - o.cx), _round(ct[1] - o.cy))
        if row["subset"] == "main":
            pg = g1._proj_gap(g1._box(o, off), g1._box(R, r0), axis)
            ok = g1._inter(o, off, R, r0) == 0 and pg >= 0 and abs(pg - gap_set) <= cfg["identity_swap"]["flag_gap_dev_px"]
        else:
            f = g1._occ(R, r0, o, off) if row["top"] == "target" else g1._occ(o, off, R, r0)
            ok = row["tier"][0] <= f < row["tier"][1]
        if first_cand is None:
            first_cand = (o, off)
        if ok:
            T2, t2 = o, off
            break
    flag = T2 is None
    if flag:
        T2, t2 = first_cand
    main = [(R, r0), (T, t0), (R, r1), (T, t1), (T2, t2)]
    bx = [g1._box(o, p) for o, p in main]
    lo = [mg - min(b[k] for b in bx) for k in (0, 1)]
    hi = [C - mg - max(b[k + 2] for b in bx) for k in (0, 1)]
    if lo[0] > hi[0] or lo[1] > hi[1]:
        raise _Retry("pair_does_not_fit")
    g = (int(rng.integers(lo[0], hi[0] + 1)), int(rng.integers(lo[1], hi[1] + 1)))
    sh = lambda p: (p[0] + g[0], p[1] + g[1])      # noqa: E731
    r0, t0, r1, t1, t2 = map(sh, (r0, t0, r1, t1, t2))
    main_boxes = [g1._box(o, p) for o, p in ((R, r0), (T, t0), (R, r1), (T, t1), (T2, t2))]
    placed, gp = [], cfg["placement"]["distractor_gap_px"]
    for o, s in dis:
        for _ in range(cfg["placement"]["max_distractor_tries"]):
            off = (int(rng.integers(mg, C - mg - o.w + 1)), int(rng.integers(mg, C - mg - o.h + 1)))
            b = g1._box(o, off)
            if all(_boxes_apart(b, x, gp) for x in main_boxes) and all(_boxes_apart(b, g1._box(q, w), gp) for q, w, _ in placed):
                placed.append((o, off, s)); break
        else:
            raise _Retry("distractor_no_room")
    bg = backgrounds.sample_background(rng, row["background"])
    bg_img = None if dry_run else backgrounds.render_background(bg, C)
    if not dry_run:
        for o, off, _ in placed:
            g1._paste(bg_img, o, off)
    edges = token_edges(cfg)
    TP = {k: v["px"] for k, v in cfg["token_px"].items()}
    out = {}
    for name, (to, tp, ro, rp) in {"base": (T, t0, R, r0), "position_swap": (T, t1, R, r1), "identity_swap": (T2, t2, R, r0)}.items():
        mt, mr = g1._full(to, tp, C), g1._full(ro, rp, C)
        inter = int((mt & mr).sum())
        top = row["top"]
        occ = inter / (ro.area if top == "target" else to.area)
        bt, br = g1._box(to, tp), g1._box(ro, rp)
        pg = g1._proj_gap(bt, br, axis)
        md = 0.0 if inter else max(0.0, float(ndimage.distance_transform_edt(~mr)[mt].min()) - 1.0)
        if row["subset"] == "main" and name != "identity_swap":
            assert inter == 0 and pg == gap_set, (row["index"], name, pg, gap_set)
        if row["subset"] == "overlap" and name != "identity_swap":
            assert row["tier"][0] <= occ < row["tier"][1], (row["index"], name, occ)
        cA, cB = g1._cen(to, tp), g1._cen(ro, rp)
        dx, dy = cA[0] - cB[0], cA[1] - cB[1]
        along, perp = (dx, dy) if axis == "horizontal" else (dy, dx)
        if abs(along) <= abs(perp):
            raise _Retry("axis_not_dominant")
        relation = RELATIONS[axis][0 if along < 0 else 1]
        img = None
        zo = [("reference", ro, rp), ("target", to, tp)] if top == "target" else [("target", to, tp), ("reference", ro, rp)]
        if not dry_run:
            img = bg_img.copy()
            for _, o, off in zo:
                g1._paste(img, o, off)
        vis = {"target": mt if top == "target" else mt & ~mr, "reference": mr if top == "reference" else mr & ~mt}
        objs = {}
        for role, o, off, full in (("target", to, tp, mt), ("reference", ro, rp, mr)):
            b = g1._box(o, off)
            assert min(b[0], b[1]) >= mg and max(b[2], b[3]) <= C - mg
            objs[role] = {"cutout_id": o.cutout_id, "category": o.category, "offset_xy": list(off), "box_xyxy": b, "centroid_xy": [round(v, 4) for v in g1._cen(o, off)],
                          "area": o.area, "sqrt_area": round(math.sqrt(o.area), 4), "upscale": round(o.scale, 4),
                          **({} if dry_run else {"mask_rle": _rle(full), "visible_box_xyxy": mask_bbox(vis[role]), "visible_area": int(vis[role].sum()), "visible_mask_rle": _rle(vis[role])})}
        meta = {"variant": name, "side": row["side"], "subset": row["subset"], "level": row["level"], "relation": relation, "axis": axis,
                "design": {**{k: row[k] for k in ("level", "subset", "stratum_px", "tier", "sqrt_area_px", "axis", "n_distractors", "background", "top", "side")}, "extra_round": row.get("extra_round")}, "sqrt_area_px": size,
                "geometry": {"edge_gap_px": pg, "edge_gap_set_px": gap_set, "min_distance_px": round(md, 4), "center_along_px": round(abs(along), 4), "center_perp_px": round(perp, 4),
                             "center_euclidean_px": round(math.hypot(dx, dy), 4), "jitter_px": jit, "jitter_max_px": J,
                             "center_along_tokens": {k: round(abs(along) / p, 4) for k, p in TP.items()}, "token_bin": {k: token_bin(abs(along), p, edges) for k, p in TP.items()}},
                "objects": objs, "z_order": [f"distractor_{i}" for i in range(len(placed))] + [z[0] for z in zo],
                "occlusion": {"occluded": inter > 0, "top": top, "intersection_px": inter, "occluded_fraction": round(occ, 6)}, "pair_translation_xy": list(g),
                "distractors": [{"cutout_id": o.cutout_id, "category": o.category, "sqrt_area_px": s, "box_xyxy": g1._box(o, off)} for o, off, s in placed], "background": bg}
        if name == "identity_swap":
            dev = math.hypot(cA[0] - (t0[0] + T.cx), cA[1] - (t0[1] + T.cy))
            meta["identity_swap"] = {"original_cutout_id": T.cutout_id, "original_category": T.category, "centroid_deviation_px": round(dev, 4),
                                     "sqrt_area_ratio": round(math.sqrt(to.area / T.area), 5), "identity_swap_flag": bool(flag or dev > cfg["identity_swap"]["max_centroid_dev_px"]),
                                     "flag_reason": ("no candidate met the criterion" if flag else ("centroid deviation" if dev > cfg["identity_swap"]["max_centroid_dev_px"] else None))}
        if name == "position_swap":
            b0 = {"target": g1._cen(T, t0), "reference": g1._cen(R, r0)}
            meta["position_swap"] = {"target_minus_base_reference_centroid": [round(cA[k] - b0["reference"][k], 4) for k in (0, 1)],
                                     "reference_minus_base_target_centroid": [round(cB[k] - b0["target"][k], 4) for k in (0, 1)]}
        out[name] = (meta, img)
    if out["position_swap"][0]["relation"] != OPPOSITE[out["base"][0]["relation"]] or out["base"][0]["relation"] != row["relation"]:
        raise _Retry("relation_mismatch")
    return out


def scene_ids(version, index):
    return {"base": f"{version}_{index:06d}_base", "position_swap": f"{version}_{index:06d}_pos", "identity_swap": f"{version}_{index:06d}_id"}


def generate_scene(cfg, lib, index, dry_run=False):
    row = build_design(cfg)[index]
    sseed = int.from_bytes(hashlib.sha256(f"{cfg['version']}:{cfg['global_seed']}:{index}".encode()).digest()[:8], "big")
    rej = {}
    for attempt in range(cfg["placement"]["max_attempts"]):
        try:
            res = _try(cfg, lib, row, sseed, attempt, dry_run)
            break
        except _Retry as e:
            rej[str(e)] = rej.get(str(e), 0) + 1
        except ValueError as e:
            rej["value_error:" + str(e)[:40]] = rej.get("value_error:" + str(e)[:40], 0) + 1
    else:
        return [{"meta": {"index": index, "failed": True, "level": row["level"], "side": row["side"], "rejected": rej}, "png": None}]
    ids = scene_ids(cfg["version"], index)
    out = []
    for name in VARIANTS:
        body, img = res[name]
        png = None if dry_run else png_bytes(img, cfg["png_compress_level"])
        out.append({"meta": {"scene_id": ids[name], "base_id": ids["base"], "version": cfg["version"], "index": index, "image": f"images/{ids[name]}.png",
                             "canvas_wh": [cfg["canvas"]] * 2, "image_sha256": None if png is None else hashlib.sha256(png).hexdigest(),
                             "seed": {"global_seed": cfg["global_seed"], "scene_seed": sseed, "attempt": attempt, "rejected": rej}, "cutout_library_sha256": lib.sha256, **body}, "png": png})
    return out
