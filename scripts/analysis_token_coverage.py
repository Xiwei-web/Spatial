"""F6-2：What's Up 中 target / reference 各覆盖多少个视觉 token（CPU，不跑模型，只用网格映射计算）。
网格来源：
  动态分辨率模型（qwen_thw）：用模型自带的图像处理器在 1280x960 的空白图上实际算出 image_grid_thw（不加载权重），
     在冒烟设置 max_pixels=1003520 以及减半、加倍三种设置下各算一次；并与冒烟记录里的网格核对。
  InternVL 系（固定单视图 448x448）：16x16，直接取冒烟 summary.json 里记录的网格。
覆盖定义：把物体 mask 投到网格上，统计 (a) 与 mask 有任何重叠的格子数；(b) 格子面积≥50% 被 mask 覆盖的格子数。
质心距离：两个 mask 质心在网格坐标下的欧氏距离（单位：token）。
标注来自 F1 的“自动标注”（未经人工验证）；只统计自动质检通过的图像，同时给出全部有 mask 图像的数字。
用法：source scripts/env.sh tf5 && python scripts/analysis_token_coverage.py"""
import json

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mu

from src import paths

W, H = 1280, 960
SMOKE_MAX = 1003520
ad = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "adapters.yaml"))
cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto.yaml"))
rows = [json.loads(l) for l in open(paths.get("data") / "whatsup_annotations" / cfg["version"] / "annotations.jsonl")]
rows = [r for r in rows if all(r["objects"][k]["mask_rle"] for k in ("target", "reference"))]
assert all((r["width"], r["height"]) == (W, H) for r in rows)
masks = [{k: mu.decode(r["objects"][k]["mask_rle"]).astype(np.float32) for k in ("target", "reference")} for r in rows]
cents = [{k: r["objects"][k]["mask_centroid_xy"] for k in ("target", "reference")} for r in rows]
passed = np.array([r["auto_qc"] == "pass" for r in rows])
is_a = np.array([r["dataset"] == "whatsup_a" for r in rows])


def grids_for(name, spec):
    """返回 {设置名: (gh, gw, 说明)}"""
    if spec["kind"] == "internvl_remote":
        s = json.load(open(paths.get("outputs") / "smoke" / name / "summary.json"))
        gh, gw = s["tests"]["6_visual_token_mapping"]["grid_hw"]
        return {"fixed_448_single_view": (gh, gw, "整图压到 448x448，与分辨率设置无关")}
    from transformers import AutoConfig, AutoProcessor
    path = paths.get(spec.get("root", "existing_models")) / name
    proc, conf = AutoProcessor.from_pretrained(path), AutoConfig.from_pretrained(path)
    merge = int(conf.vision_config.spatial_merge_size)
    out = {}
    for tag, mp in (("max_pixels/2", SMOKE_MAX // 2), ("smoke(max_pixels=1003520)", SMOKE_MAX), ("max_pixels*2", SMOKE_MAX * 2)):
        g = proc.image_processor(images=[Image.new("RGB", (W, H))], max_pixels=mp, return_tensors="pt")["image_grid_thw"][0].tolist()
        out[tag] = (g[1] // merge, g[2] // merge, f"max_pixels={mp}")
    try:        # 与冒烟实际记录核对
        s = json.load(open(paths.get("outputs") / "smoke" / name / "summary.json"))["tests"]["6_visual_token_mapping"]["grid_hw"]
        out["smoke(max_pixels=1003520)"] += (f"冒烟记录网格 {s}，{'一致' if list(s) == list(out['smoke(max_pixels=1003520)'][:2]) else '不一致!'}",)
    except Exception as e:
        out["smoke(max_pixels=1003520)"] += (f"无法核对冒烟记录: {e}",)
    return out


_IDX = {}


def cover(m, gh, gw):
    """每个格子被 mask 覆盖的面积比例（格子边界按实数等分，像素按中心归属）。用 bincount 实现。"""
    if (gh, gw) not in _IDX:
        ys = np.minimum((np.arange(H) + 0.5) * gh / H, gh - 1e-6).astype(np.int64)
        xs = np.minimum((np.arange(W) + 0.5) * gw / W, gw - 1e-6).astype(np.int64)
        idx = (ys[:, None] * gw + xs[None, :]).ravel()
        _IDX[(gh, gw)] = (idx, np.bincount(idx, minlength=gh * gw).astype(np.float64))
    idx, cnt = _IDX[(gh, gw)]
    return (np.bincount(idx, weights=m.ravel(), minlength=gh * gw) / cnt).reshape(gh, gw)


def q(v):
    v = np.asarray(v, dtype=float)
    return {"p5": round(float(np.percentile(v, 5)), 1), "median": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}


result = {"annotation": f"{cfg['version']} (automatic, not human-verified)", "image_size": [W, H], "n_images_with_both_masks": len(rows),
          "n_auto_qc_pass": int(passed.sum()), "models": {}}
cache = {}
for name, spec in ad["models"].items():
    if spec.get("group") != 1:
        continue
    result["models"][name] = {}
    for tag, g in grids_for(name, spec).items():
        gh, gw = g[0], g[1]
        if (gh, gw) not in cache:
            st = {"t_any": [], "t_half": [], "r_any": [], "r_half": [], "dist": []}
            for m, c in zip(masks, cents):
                ct, cr = cover(m["target"], gh, gw), cover(m["reference"], gh, gw)
                st["t_any"].append((ct > 0).sum()); st["t_half"].append((ct >= 0.5).sum())
                st["r_any"].append((cr > 0).sum()); st["r_half"].append((cr >= 0.5).sum())
                st["dist"].append(float(np.hypot((c["target"][0] - c["reference"][0]) * gw / W, (c["target"][1] - c["reference"][1]) * gh / H)))
            cache[(gh, gw)] = {k: np.array(v) for k, v in st.items()}
        st = cache[(gh, gw)]
        entry = {"grid_hw": [gh, gw], "n_visual_tokens": gh * gw, "cell_px_in_original": [round(W / gw, 1), round(H / gh, 1)], "notes": list(g[2:])}
        for sel_tag, sel in (("auto_qc_pass", passed), ("all", np.ones(len(rows), bool)), ("auto_qc_pass_A", passed & is_a), ("auto_qc_pass_B", passed & ~is_a)):
            entry[sel_tag] = {"n": int(sel.sum()), "target_tokens_any_overlap": q(st["t_any"][sel]), "target_tokens_ge50pct": q(st["t_half"][sel]),
                              "reference_tokens_any_overlap": q(st["r_any"][sel]), "reference_tokens_ge50pct": q(st["r_half"][sel]),
                              "centroid_distance_tokens": q(st["dist"][sel]),
                              "share_target_with_zero_ge50pct_tokens": round(float((st["t_half"][sel] == 0).mean()), 3)}
        result["models"][name][tag] = entry
        e = entry["auto_qc_pass"]
        print(f"{name:32s} {tag:28s} grid {gh}x{gw:<3d} target(any) {e['target_tokens_any_overlap']} ref(any) {e['reference_tokens_any_overlap']} dist {e['centroid_distance_tokens']}", flush=True)
json.dump(result, open(paths.REPO_ROOT / "manifests" / "whatsup_token_coverage.json", "w"), indent=1, ensure_ascii=False)
