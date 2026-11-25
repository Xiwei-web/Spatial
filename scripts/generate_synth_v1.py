"""生成 trial_v1（500 个基础场景 + 各 2 张孪生图）、确定性检查、统计、联系表。CPU，计算节点上跑。
用法：python scripts/generate_synth_v1.py [--workers 8] [--limit N --out-dir 目录]
      python scripts/generate_synth_v1.py --check-cells     用各模型的图像处理器核对 896x896 画布上的 token 网格（不加载权重）
输出：<data>/synth/trial_v1/{images, metadata.jsonl}、manifests/synth_trial_v1.json、<outputs>/synth_review/trial_v1/sheet_*.jpg"""
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import paths
from src.synth import generator_v1 as g1
from src.synth.generator import VARIANTS, load_config

_G: dict = {}


def _init(cfg, out):
    _G.update(cfg=cfg, lib=g1.LibraryV1(), out=out)


def _work(i):
    t = time.perf_counter()
    res = g1.generate_scene(_G["cfg"], _G["lib"], i)
    for r in res:
        (_G["out"] / r["meta"]["image"]).write_bytes(r["png"])
    return i, [json.dumps(r["meta"], ensure_ascii=False, separators=(",", ":")) for r in res], [len(r["png"]) for r in res], time.perf_counter() - t


def dist(v):
    v = np.asarray(v, float)
    return {"n": len(v), "min": round(float(v.min()), 4), "p5": round(float(np.percentile(v, 5)), 4), "median": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4)}


def summarize(metas, cfg):
    base = [m for m in metas if m["variant"] == "base"]
    ids = [m for m in metas if m["variant"] == "identity_swap"]
    pos = [m for m in metas if m["variant"] == "position_swap"]
    L = [x["name"] for x in g1.levels(cfg)]
    cross = {}
    for l in L:
        sub = [m for m in base if m["level"] == l]
        cross[l] = {"n": len(sub), **{f"{k}={v}": sum(m["design"][k] == v for m in sub) for k, vals in (("side", ("dev", "test")), ("axis", cfg["factors"]["axis"]), ("size_frac", cfg["factors"]["size_frac"]), ("n_distractors", cfg["factors"]["n_distractors"])) for v in vals},
                    "center_along_px_median": round(statistics.median(m["gap"]["center_along_px"] for m in sub), 1)}
        if l.startswith("gap"):
            cross[l]["identity_twin_projection_gap"] = dist([m["gap"]["projection_gap_px"] for m in ids if m["level"] == l])
            cross[l]["min_distance_px_base"] = dist([m["gap"]["min_distance_px"] for m in sub])
        else:
            cross[l]["occluded_fraction"] = {v: dist([m["occlusion"]["occluded_fraction"] for m in metas if m["level"] == l and m["variant"] == v]) for v in VARIANTS}
    used = defaultdict(Counter)
    for m in metas:
        for o in m["objects"].values():
            used[m["side"]][o["cutout_id"]] += 1
    lib = {e["cutout_id"]: e for e in g1.LibraryV1().entries}
    per_cat = {}
    for c in sorted({e["category"] for e in lib.values()}):
        per_cat[c] = {s: {"library": sum(e["category"] == c and e["split"] == s for e in lib.values()), "used_as_main": sum(lib[i]["category"] == c for i in used[s]),
                          "main_appearances": sum(n for i, n in used[s].items() if lib[i]["category"] == c)} for s in ("dev", "test")}
    for m in metas:
        assert all(lib[o["cutout_id"]]["split"] == m["side"] for o in list(m["objects"].values()) + m["distractors"])
    ups = [o["upscale"] for m in metas for o in list(m["objects"].values()) + m["distractors"]]
    C, mg = cfg["canvas"], cfg["placement"]["canvas_margin_px"]
    margins = [min(b[0], b[1], C - b[2], C - b[3]) for m in metas for b in [o["box_xyxy"] for o in list(m["objects"].values()) + m["distractors"]]]
    assert min(margins) >= mg
    return {"n_base": len(base), "n_images": len(metas), "by_level": cross, "by_side": dict(Counter(m["side"] for m in base)), "relation": dict(Counter(m["relation"] for m in base)),
            "identity_swap": {"centroid_deviation_px": dist([m["identity_swap"]["centroid_deviation_px"] for m in ids]), "area_ratio": dist([m["identity_swap"]["area_ratio"] for m in ids]),
                              "sqrt_area_ratio": dist([m["identity_swap"]["sqrt_area_ratio"] for m in ids])},
            "position_swap_centroid_shift_px": {k: dist([float(np.hypot(*m["position_swap"][k])) for m in pos if m["subset"] == s]) for s in ("main", "overlap") for k in ("target_minus_base_reference_centroid",)} if False else
            {s: dist([float(np.hypot(*m["position_swap"]["target_minus_base_reference_centroid"])) for m in pos if m["subset"] == s]) for s in ("main", "overlap")},
            "sqrt_area_error_px": dist([o["sqrt_area"] - m["design"]["sqrt_area_px"] for m in metas for o in m["objects"].values()]),
            "upscale": dist(ups), "n_upscale_gt_1.5": sum(u > cfg["max_upscale"] for u in ups), "min_margin_px": min(margins),
            "attempts": dict(sorted(Counter(min(m["seed"]["attempt"], 100) for m in base).items())), "attempt_by_level_median": {l: statistics.median(m["seed"]["attempt"] for m in base if m["level"] == l) for l in L},
            "rejections": {k: sum(m["seed"]["rejected"].get(k, 0) for m in base) for k in sorted({k for m in base for k in m["seed"]["rejected"]})},
            "aspect_along_axis_by_level": {l: round(statistics.median((o["box_xyxy"][2] - o["box_xyxy"][0]) / (o["box_xyxy"][3] - o["box_xyxy"][1]) if m["axis"] == "horizontal" else (o["box_xyxy"][3] - o["box_xyxy"][1]) / (o["box_xyxy"][2] - o["box_xyxy"][0])
                                                                                                   for m in base if m["level"] == l for o in m["objects"].values()), 3) for l in L},
            "per_category": per_cat, "n_categories_used": sum(any(v["used_as_main"] for v in d.values()) for d in per_cat.values())}


COL = {"target": (220, 30, 30), "reference": (30, 80, 230), "distractor": (150, 150, 150)}


def sheets(metas, cfg, data, out, rows=8, n_sheets=8):
    from matplotlib import font_manager
    ft = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    T, CAP, PAD = 448, 330, 6
    by = defaultdict(dict)
    for m in metas:
        by[m["index"]][m["variant"]] = m
    L = [x["name"] for x in g1.levels(cfg)]
    q = {l: sorted(i for i in by if by[i]["base"]["level"] == l) for l in L}
    order, r = [], 0
    while len(order) < rows * n_sheets and any(r < len(v) for v in q.values()):
        order += [q[l][r] for l in L if r < len(q[l])]
        r += 1
    order = order[: rows * n_sheets]
    out.mkdir(parents=True, exist_ok=True)
    for p in range(0, len(order), rows):
        ch = order[p:p + rows]
        sh = Image.new("RGB", (CAP + 3 * (T + PAD) + PAD, 24 + len(ch) * (T + PAD)), (255, 255, 255))
        d = ImageDraw.Draw(sh)
        d.text((6, 4), f"{cfg['version']} sheet {p // rows:02d}: base | position_swap | identity_swap      red=target blue=reference grey=distractor", fill=(0, 0, 0), font=ft)
        for k, i in enumerate(ch):
            y = 24 + k * (T + PAD)
            b = by[i]["base"]
            cap = [b["base_id"], f"{b['side']}  {b['subset']}  level={b['level']}", f"axis={b['axis']}  top={b['occlusion']['top']}", f"size={b['design']['size_frac']} (sqrt area {b['design']['sqrt_area_px']}px)",
                   f"n_dis={b['design']['n_distractors']}  bg={b['design']['background']}", ""]
            for v in VARIANTS:
                m = by[i][v]
                cap += [f"{v}: {m['objects']['target']['category']}", f"   [{m['relation']}] {m['objects']['reference']['category']}",
                        f"   proj gap {m['gap']['projection_gap_px']}  min dist {m['gap']['min_distance_px']:.1f}", f"   center {m['gap']['center_along_px']:.1f}px  occ {m['occlusion']['occluded_fraction']:.3f}"]
            s = by[i]["identity_swap"]["identity_swap"]
            cap += ["", f"id-swap: centroid dev {s['centroid_deviation_px']:.2f}px", f"   area ratio {s['area_ratio']:.3f}"]
            d.multiline_text((6, y + 4), "\n".join(cap), fill=(0, 0, 0), font=ft, spacing=3)
            for c, v in enumerate(VARIANTS):
                m = by[i][v]
                im = Image.open(data / m["image"]).convert("RGB").resize((T, T), Image.LANCZOS)
                dd = ImageDraw.Draw(im)
                for o in m["distractors"]:
                    dd.rectangle([x * T / cfg["canvas"] for x in o["box_xyxy"]], outline=COL["distractor"])
                for role in ("reference", "target"):
                    dd.rectangle([x * T / cfg["canvas"] for x in m["objects"][role]["box_xyxy"]], outline=COL[role], width=2)
                sh.paste(im, (CAP + PAD + c * (T + PAD), y))
        sh.save(out / f"sheet_{p // rows:03d}.jpg", quality=86)
    return len(range(0, len(order), rows))


def check_cells(cfg):
    from transformers import AutoConfig, AutoProcessor
    from src.models.adapter import adapter_spec
    im = Image.new("RGB", (cfg["canvas"],) * 2, (90, 120, 60))
    for n in ("Qwen2.5-VL-7B-Instruct", "RoboBrain2.0-7B", "VST-7B-RL", "GLM-4.6V-Flash", "Qwen3-VL-8B-Instruct", "Qwen3.5-9B"):
        sp = adapter_spec(n)
        path = paths.get(sp["root"]) / n
        pr, cf = AutoProcessor.from_pretrained(path), AutoConfig.from_pretrained(path)
        kw = {"size": dict(sp["image_size"])} if sp.get("image_size") else {"max_pixels": sp["max_pixels"]}
        t = pr.image_processor(images=[im], return_tensors="pt", **kw)["image_grid_thw"][0].tolist()
        m = cf.vision_config.spatial_merge_size
        print(n, "grid", (t[1] // m, t[2] // m), "cell px on canvas", cfg["canvas"] / (t[2] // m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth/trial_v1.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--check-cells", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.check_cells:
        return check_cells(cfg)
    n = a.limit or cfg["n_scenes"]
    data = a.out_dir or paths.get("data") / "synth" / cfg["version"]
    (data / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(cfg, data)) as pool:
        res = sorted(pool.imap_unordered(_work, range(n), chunksize=2))
    wall = time.time() - t0
    text = "".join(l + "\n" for r in res for l in r[1])
    (data / "metadata.jsonl").write_text(text)
    metas = [json.loads(l) for l in text.splitlines()]
    lib = g1.LibraryV1()
    pick = sorted({int(i * (n - 1) / 19) for i in range(20)}) if n >= 20 else list(range(n))
    same = 0
    for i in pick:
        for r, l in zip(g1.generate_scene(cfg, lib, i), res[i][1]):
            same += json.dumps(r["meta"], ensure_ascii=False, separators=(",", ":")) == l and (data / r["meta"]["image"]).read_bytes() == r["png"]
    s = {"config": cfg, "library_sha256": lib.sha256, "metadata_sha256": hashlib.sha256(text.encode()).hexdigest(), "determinism_identical_images": f"{same}/{3 * len(pick)}",
         "wall_seconds": round(wall, 1), "workers": a.workers, "png_bytes_total": sum(sum(r[2]) for r in res), **summarize(metas, cfg)}
    if not a.out_dir:
        json.dump(s, open(paths.REPO_ROOT / "manifests" / f"synth_{cfg['version']}.json", "w"), indent=1, ensure_ascii=False)
    out = (a.out_dir / "sheets") if a.out_dir else paths.get("outputs") / "synth_review" / cfg["version"]
    k = sheets(metas, cfg, data, out)
    print(json.dumps({x: s[x] for x in ("n_base", "by_side", "relation", "determinism_identical_images", "identity_swap", "position_swap_centroid_shift_px", "upscale", "min_margin_px", "rejections", "wall_seconds")}, ensure_ascii=False))
    print("sheets", k, "->", out)


if __name__ == "__main__":
    main()
