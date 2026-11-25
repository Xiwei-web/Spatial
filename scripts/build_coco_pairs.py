"""构建 COCO-Pairs（任务 F3）：从 COCO val2017 实例标注里取“类别在图中唯一”的物体，两两配对，
记录关系标签和一组连续量（间距、IoU、包含度、拥挤度、投影重叠……），不过滤任何难例。

输出：
  <data>/coco_pairs/pairs.jsonl          每行一条“有向 pair”（target → reference），两个方向互为 twin
  划分：splits/coco_pairs_v2.json（按图像，继承 coco_images_v2），由 scripts/make_joint_coco_split.py 生成，本脚本正式模式下不写划分
  manifests/coco_pairs_summary.json      机器可读的统计 + pairs.jsonl 的 sha256
  docs/parts/F3_coco_pairs.md            分布报告（表格）

用法：
  python scripts/build_coco_pairs.py                 # 全量（请通过 Slurm 跑，见 scripts/submit.sh）
  python scripts/build_coco_pairs.py --force         # 全量，并允许重写已有划分
  python scripts/build_coco_pairs.py --limit 30 --out-dir <临时目录>   # 只取前 30 张图试跑，所有输出写到临时目录

只依赖标注文件，不读图像像素，不做任何模型推理。完全确定性（无时间戳，固定种子）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from src import paths
from src.data import load_dataset
from src.data import coco_pairs as cp

SEED = cp.SEED


# ---------------------------------------------------------------- 小工具
def r6(x: float) -> float:
    return round(float(x), 6)


def ann_to_rle(ann: dict, h: int, w: int) -> dict:
    """COCO segmentation → 压缩 RLE（与 pycocotools.COCO.annToRLE 相同的逻辑）。"""
    seg = ann["segmentation"]
    if isinstance(seg, list):                       # polygon（可能多段）
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):             # 未压缩 RLE
        return mask_utils.frPyObjects(seg, h, w)
    return seg


def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def quantiles(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    qs = [0, 5, 25, 50, 75, 95, 100]
    d = {f"p{q}": r6(np.percentile(a, q)) for q in qs}
    d["mean"] = r6(a.mean())
    return d


def spatial_image_ids(name: str) -> set[int]:
    return {int(Path(s.image_path).stem) for s in load_dataset(name)}


# ---------------------------------------------------------------- 1. 候选物体
def find_candidates(coco: dict, image_ids: list[int]) -> tuple[dict[int, list[dict]], dict]:
    """返回 {image_id: [候选 ann, ...]} 与逐步筛选的漏斗计数。

    规则（依次施加）：
      a) iscrowd == 0；
      b) 该物体的类别在这张图里恰好出现 1 次——统计该图里该类别的**全部**标注（包括 iscrowd=1 的），
         所以“1 个 person 实例 + 1 个 person crowd 区域”不算唯一；
      c) COCO `area` / (width*height) ∈ [1%, 40%]（闭区间）；
      d) 图像短边 min(width, height) ≥ 300。
    """
    funnel = Counter()
    out: dict[int, list[dict]] = {}
    for iid in image_ids:
        im = coco["images"][iid]
        anns = coco["by_image"][iid]
        funnel["annotations_all"] += len(anns)
        cat_count = Counter(a["category_id"] for a in anns)      # 含 crowd
        img_area = im["width"] * im["height"]
        short_ok = min(im["width"], im["height"]) >= cp.MIN_SHORT_SIDE
        cands = []
        for a in anns:
            if a.get("iscrowd", 0):
                continue
            funnel["a_noncrowd"] += 1
            if cat_count[a["category_id"]] != 1:
                continue
            funnel["b_unique_category"] += 1
            frac = a["area"] / img_area
            if not (cp.MIN_AREA_FRAC <= frac <= cp.MAX_AREA_FRAC):
                continue
            funnel["c_area_1_40pct"] += 1
            if not short_ok:
                continue
            funnel["d_short_side_300"] += 1
            cands.append(a)
        if cands:
            out[iid] = sorted(cands, key=lambda a: a["id"])
    return out, dict(funnel)


# ---------------------------------------------------------------- 2-5. pair 记录
def build_records(coco: dict, cands: dict[int, list[dict]], in_one: set[int], in_two: set[int]):
    """对每张有 ≥2 个候选的图，所有无序候选对各生成两条有向记录。同时返回每个候选的解码信息（用于面积核对）。"""
    records, obj_info = [], {}
    n_centroid_tie = 0
    for iid in sorted(cands):
        im = coco["images"][iid]
        h, w = im["height"], im["width"]
        anns_all = coco["by_image"][iid]
        n_all = len(anns_all)
        n_noncrowd = sum(1 for a in anns_all if not a.get("iscrowd", 0))
        objs = {}
        for a in cands[iid]:
            rle = ann_to_rle(a, h, w)
            m = mask_utils.decode(rle)
            cx, cy = cp.mask_centroid(m)
            objs[a["id"]] = {
                "rle": rle,
                "pub": {"ann_id": a["id"], "category": coco["cats"][a["category_id"]], "category_id": a["category_id"],
                        "bbox": [r6(v) for v in cp.xywh_to_xyxy(a["bbox"])], "area": r6(a["area"]),
                        "area_decoded": int(m.sum()), "area_frac": r6(a["area"] / (w * h)),
                        "centroid": [r6(cx), r6(cy)]},
                "c": (cx, cy), "bbox": cp.xywh_to_xyxy(a["bbox"]), "area": float(a["area"]),
            }
            obj_info[a["id"]] = {"image_id": iid, "area": float(a["area"]), "area_decoded": int(m.sum()),
                                 "img_area": w * h, "category": coco["cats"][a["category_id"]]}
        if len(objs) < 2:
            continue
        for ia, ib in combinations(sorted(objs), 2):          # ia < ib（ann_id）
            A, B = objs[ia], objs[ib]
            axis = cp.relation_axis(A["c"], B["c"])           # 对称：两个方向用同一条轴
            tie = (A["c"] == B["c"])
            n_centroid_tie += int(tie)
            sym = {
                "axis": axis,
                "centroid_dist": r6(cp.centroid_distance(A["c"], B["c"])),
                "norm_gap": r6(cp.normalized_gap(A["c"], B["c"], A["bbox"], B["bbox"], axis)),
                "norm_gap_axis": r6(cp.normalized_gap_axis(A["c"], B["c"], A["bbox"], B["bbox"], axis)),
                "bbox_iou": r6(cp.bbox_iou(A["bbox"], B["bbox"])),
                "mask_iou": r6(mask_utils.iou([A["rle"]], [B["rle"]], [0])[0][0]),
                "proj_overlap": bool(cp.projection_overlap(A["bbox"], B["bbox"], axis)),
                "proj_overlap_frac": r6(cp.projection_overlap_frac(A["bbox"], B["bbox"], axis)),
                "n_other_instances": n_all - 2,
                "n_other_instances_noncrowd": n_noncrowd - 2,   # 两个候选本身都是非 crowd
                "n_candidates_in_image": len(objs),
                "centroid_tie": bool(tie),
                "in_coco_spatial_one": iid in in_one,
                "in_coco_spatial_two": iid in in_two,
            }
            for T, R in ((A, B), (B, A)):
                t, ref = T["pub"], R["pub"]
                rec = {
                    "pair_id": cp.make_pair_id(iid, t["ann_id"], ref["ann_id"]),
                    "twin_id": cp.make_pair_id(iid, ref["ann_id"], t["ann_id"]),
                    "image_id": iid, "file_name": im["file_name"], "width": w, "height": h,
                    "target": t, "reference": ref,
                    "relation": cp.relation_label(T["c"], R["c"], target_first_on_tie=t["ann_id"] < ref["ann_id"]),
                    "dx": r6(T["c"][0] - R["c"][0]), "dy": r6(T["c"][1] - R["c"][1]),
                    "size_ratio": r6(T["area"] / R["area"]),
                    "containment_target_in_reference": r6(cp.containment(T["bbox"], R["bbox"])),
                    "containment_reference_in_target": r6(cp.containment(R["bbox"], T["bbox"])),
                }
                rec.update(sym)
                records.append(rec)
    return records, obj_info, n_centroid_tie


def check_records(records: list[dict]):
    """内部一致性：id 唯一、twin 互指、twin 的关系相反、同图 pair 数 = n*(n-1)。"""
    by_id = {r["pair_id"]: r for r in records}
    assert len(by_id) == len(records), "pair_id 不唯一"
    for r in records:
        tw = by_id[r["twin_id"]]
        assert tw["twin_id"] == r["pair_id"]
        assert tw["relation"] == cp.OPPOSITE[r["relation"]], (r["pair_id"], r["relation"], tw["relation"])
        assert tw["target"]["ann_id"] == r["reference"]["ann_id"]
        assert cp.AXIS_OF[r["relation"]] == r["axis"]
        assert r["target"]["category"] != r["reference"]["category"]      # 类别唯一 ⇒ 两个候选必不同类
    per_img = Counter(r["image_id"] for r in records)
    n_c = {r["image_id"]: r["n_candidates_in_image"] for r in records}
    assert all(per_img[i] == n_c[i] * (n_c[i] - 1) for i in per_img)


# ---------------------------------------------------------------- 7. 划分
def image_stratum(recs: list[dict]) -> str:
    """图像级分层键 = (主导关系轴, 候选数档)。

    因为每个无序对都保留两个方向，left 与 right（above 与 below）在任何一张图里数量恒等，
    所以图像级真正能分层的是“水平 / 竖直”：
      主导轴 = 该图无序对里 horizontal 多 → "H"；vertical 多 → "V"；一样多 → "T"。
    候选数档（决定这张图贡献多少 pair）：2 / 3 / 4+。
    """
    und = [r for r in recs if r["target"]["ann_id"] < r["reference"]["ann_id"]]
    n_h = sum(r["axis"] == "horizontal" for r in und)
    n_v = len(und) - n_h
    dom = "H" if n_h > n_v else ("V" if n_v > n_h else "T")
    n = recs[0]["n_candidates_in_image"]
    return f"{dom}/n{n if n < 4 else '4+'}"


def make_split(records: list[dict]) -> dict:
    by_img = defaultdict(list)
    for r in records:
        by_img[r["image_id"]].append(r)
    strata = defaultdict(list)
    for iid, recs in by_img.items():
        strata[image_stratum(recs)].append(iid)
    dev, test, per_stratum = [], [], {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng = random.Random(f"{SEED}:{cp.DATASET}:{key}")      # 每层独立且确定的随机流（同 make_splits.py）
        rng.shuffle(ids)
        h = len(ids) // 2                                       # 奇数时多出的 1 张归 test
        dev += ids[:h]
        test += ids[h:]
        per_stratum[key] = {"n_images": len(ids), "dev": h, "test": len(ids) - h}
    dev.sort(); test.sort()
    assert not set(dev) & set(test) and len(dev) + len(test) == len(by_img)
    return {"dataset": cp.DATASET, "seed": SEED, "unit": "image",
            "stratify_by": "image-level (dominant relation axis H/V/T among the image's undirected pairs) x (number of candidates in image: 2 / 3 / 4+)",
            "n_images": len(by_img), "n_dev_images": len(dev), "n_test_images": len(test),
            "per_stratum": per_stratum, "dev": dev, "test": test}


def split_counts(records: list[dict], split: dict) -> dict:
    out = {}
    for side in ("dev", "test"):
        keep = set(split[side])
        rs = [r for r in records if r["image_id"] in keep]
        out[side] = {
            "n_images": len(keep), "n_directed_pairs": len(rs), "n_undirected_pairs": len(rs) // 2,
            "per_relation": {k: sum(r["relation"] == k for r in rs) for k in cp.RELATIONS},
            "per_gap_bin": {b[0]: sum(cp.bin_label(r["norm_gap"], cp.GAP_BINS) == b[0] for r in rs) for b in cp.GAP_BINS},
            "per_crowding_bin": {b[0]: sum(cp.bin_label(r["n_other_instances"], cp.CROWD_BINS) == b[0] for r in rs) for b in cp.CROWD_BINS},
            "proj_overlap": sum(r["proj_overlap"] for r in rs),
            "bbox_iou_gt0": sum(r["bbox_iou"] > 0 for r in rs),
            "mask_iou_gt0": sum(r["mask_iou"] > 0 for r in rs),
            "in_coco_spatial_one": sum(r["in_coco_spatial_one"] for r in rs),
            "in_coco_spatial_two": sum(r["in_coco_spatial_two"] for r in rs),
        }
    return out


# ---------------------------------------------------------------- 8. 统计
def summarize(coco, image_ids, cands, funnel, records, obj_info, n_tie, in_one, in_two) -> dict:
    n_dir = len(records)
    pair_imgs = sorted({r["image_id"] for r in records})
    s: dict = {"dataset": cp.DATASET, "seed": SEED, "source": "COCO val2017 instances_val2017.json"}
    s["rules"] = {
        "candidate": {
            "iscrowd": 0,
            "unique_category": "该类别在该图的全部标注（含 iscrowd=1）中恰好出现 1 次",
            "area_frac_range_inclusive": [cp.MIN_AREA_FRAC, cp.MAX_AREA_FRAC],
            "area_source": "COCO annotation `area` 字段（不是解码后的像素数；两者差异见 area_check）",
            "min_short_side": cp.MIN_SHORT_SIDE,
        },
        "pairing": "同一张图里所有无序候选对；每对保留两个方向（两条记录，twin_id 互指）",
        "centroid": "解码后的二值 mask 的像素下标均值 + 0.5（像素中心，与 COCO bbox 连续坐标一致）",
        "relation": "|dx|>|dy| → left/right（target 质心 x 更小 → left）；否则 above/below（target 质心 y 更小 → above）",
        "tie": "|dx|==|dy| 归 above/below；质心完全重合时 ann_id 较小的一方记为 above（保证 twin 相反），并置 centroid_tie=true",
        "norm_gap": "质心欧氏距离 / 两个 bbox 沿关系轴长度的均值（水平轴用宽，竖直轴用高）",
        "norm_gap_axis": "质心差在关系轴上的分量 / 同一分母（补充量）",
        "bbox": "COCO bbox 字段，转成 [x1,y1,x2,y2]",
        "size_ratio": "COCO area(target) / COCO area(reference)",
        "containment": "bbox 交集面积 / 自身 bbox 面积；两个方向都记",
        "proj_overlap": "两个 bbox 在关系轴上的投影区间重叠长度 > 0（端点相接不算）",
        "n_other_instances": "该图全部标注数（含 crowd）- 2；n_other_instances_noncrowd 只数 iscrowd=0",
        "filters_on_hard_cases": "无",
    }
    # 漏斗
    s["n_images_scanned"] = len(image_ids)
    s["funnel_objects"] = funnel
    n_c = Counter(len(v) for v in cands.values())
    s["n_candidate_objects"] = sum(len(v) for v in cands.values())
    s["n_images_with_ge1_candidate"] = len(cands)
    s["n_images_with_ge2_candidates"] = sum(1 for v in cands.values() if len(v) >= 2)
    s["images_by_n_candidates"] = {str(k): n_c[k] for k in sorted(n_c)}
    s["images_by_n_candidates"]["0"] = len(image_ids) - len(cands)
    s["n_objects_in_pairs"] = sum(len(v) for v in cands.values() if len(v) >= 2)
    s["n_undirected_pairs"] = n_dir // 2
    s["n_directed_pairs"] = n_dir
    s["n_centroid_ties"] = n_tie
    s["n_abs_dx_eq_abs_dy_undirected"] = sum(abs(r["dx"]) == abs(r["dy"]) for r in records) // 2
    # 面积核对：COCO area vs 解码像素数（全部候选，不是抽样）
    rel, flips = [], 0
    for o in obj_info.values():
        rel.append((o["area_decoded"] - o["area"]) / o["area"])
        f = o["area_decoded"] / o["img_area"]
        flips += int(not (cp.MIN_AREA_FRAC <= f <= cp.MAX_AREA_FRAC))
    a = np.abs(np.asarray(rel)) if rel else np.zeros(0)
    s["area_check"] = {
        "n_objects_checked": len(rel),
        "definition": "(decoded_pixels - coco_area) / coco_area",
        "rel_diff": quantiles(rel),
        "abs_rel_diff_mean": r6(a.mean()) if a.size else None,
        "abs_rel_diff_max": r6(a.max()) if a.size else None,
        "n_abs_rel_diff_gt_1pct": int((a > 0.01).sum()),
        "n_abs_rel_diff_gt_5pct": int((a > 0.05).sum()),
        "n_candidates_outside_1_40pct_if_decoded_area_were_used": flips,
    }
    # 关系 / 分箱
    s["per_relation"] = {k: sum(r["relation"] == k for r in records) for k in cp.RELATIONS}
    s["per_axis_undirected"] = {k: sum(r["axis"] == k for r in records) // 2 for k in ("horizontal", "vertical")}
    gap = Counter(cp.bin_label(r["norm_gap"], cp.GAP_BINS) for r in records)
    gap_ax = Counter(cp.bin_label(r["norm_gap_axis"], cp.GAP_BINS) for r in records)
    s["per_gap_bin"] = {b[0]: gap[b[0]] for b in cp.GAP_BINS}
    s["per_gap_axis_bin"] = {b[0]: gap_ax[b[0]] for b in cp.GAP_BINS}
    crowd = Counter(cp.bin_label(r["n_other_instances"], cp.CROWD_BINS) for r in records)
    crowd_nc = Counter(cp.bin_label(r["n_other_instances_noncrowd"], cp.CROWD_BINS) for r in records)
    s["per_crowding_bin"] = {b[0]: crowd[b[0]] for b in cp.CROWD_BINS}
    s["per_crowding_bin_noncrowd"] = {b[0]: crowd_nc[b[0]] for b in cp.CROWD_BINS}
    crowd_img = Counter(cp.bin_label(len(coco["by_image"][i]) - 2, cp.CROWD_BINS) for i in pair_imgs)
    s["images_per_crowding_bin"] = {b[0]: crowd_img[b[0]] for b in cp.CROWD_BINS}
    x = Counter((r["relation"], cp.bin_label(r["norm_gap"], cp.GAP_BINS)) for r in records)
    s["relation_x_gap_bin"] = {k: {b[0]: x[(k, b[0])] for b in cp.GAP_BINS} for k in cp.RELATIONS}
    xc = Counter((r["relation"], cp.bin_label(r["n_other_instances"], cp.CROWD_BINS)) for r in records)
    s["relation_x_crowding_bin"] = {k: {b[0]: xc[(k, b[0])] for b in cp.CROWD_BINS} for k in cp.RELATIONS}
    xg = Counter((cp.bin_label(r["norm_gap"], cp.GAP_BINS), cp.bin_label(r["n_other_instances"], cp.CROWD_BINS)) for r in records)
    s["gap_bin_x_crowding_bin"] = {g[0]: {b[0]: xg[(g[0], b[0])] for b in cp.CROWD_BINS} for g in cp.GAP_BINS}
    # 歧义 / 重叠
    s["proj_overlap"] = {
        "directed": sum(r["proj_overlap"] for r in records),
        "per_relation": {k: sum(r["proj_overlap"] for r in records if r["relation"] == k) for k in cp.RELATIONS},
        "per_gap_bin": {b[0]: sum(r["proj_overlap"] for r in records if cp.bin_label(r["norm_gap"], cp.GAP_BINS) == b[0]) for b in cp.GAP_BINS},
        "frac_ge_0.5_directed": sum(r["proj_overlap_frac"] >= 0.5 for r in records),
        "frac_eq_1_directed": sum(r["proj_overlap_frac"] >= 1.0 for r in records),
    }
    s["bbox_iou_gt0_directed"] = sum(r["bbox_iou"] > 0 for r in records)
    s["mask_iou_gt0_directed"] = sum(r["mask_iou"] > 0 for r in records)
    s["bbox_iou_ge_0.5_directed"] = sum(r["bbox_iou"] >= 0.5 for r in records)
    s["containment_target_in_reference_ge_0.9_directed"] = sum(r["containment_target_in_reference"] >= 0.9 for r in records)
    s["containment_target_in_reference_eq_1_directed"] = sum(r["containment_target_in_reference"] >= 1.0 for r in records)
    und = [r for r in records if r["target"]["ann_id"] < r["reference"]["ann_id"]]
    s["metric_quantiles_undirected"] = {k: quantiles([r[k] for r in und]) for k in
                                        ("norm_gap", "norm_gap_axis", "centroid_dist", "bbox_iou", "mask_iou",
                                         "proj_overlap_frac", "n_other_instances", "n_other_instances_noncrowd")}
    s["metric_quantiles_directed"] = {k: quantiles([r[k] for r in records]) for k in
                                      ("size_ratio", "containment_target_in_reference")}
    # 类别
    cat_cand = Counter(o["category"] for o in obj_info.values())
    cat_pairobj = Counter()
    seen = set()
    for r in records:
        if r["target"]["ann_id"] not in seen:
            seen.add(r["target"]["ann_id"])
            cat_pairobj[r["target"]["category"]] += 1
    cat_target = Counter(r["target"]["category"] for r in records)
    s["n_distinct_categories_candidates"] = len(cat_cand)
    s["n_distinct_categories_in_pairs"] = len(cat_pairobj)
    s["category_top20"] = [{"category": c, "n_objects_in_pairs": n, "n_directed_pairs_as_target": cat_target[c],
                            "n_candidate_objects": cat_cand[c]}
                           for c, n in sorted(cat_pairobj.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    s["category_objects_in_pairs_all"] = dict(sorted(cat_pairobj.items(), key=lambda kv: (-kv[1], kv[0])))
    cp_cnt = Counter(tuple(sorted((r["target"]["category"], r["reference"]["category"]))) for r in und)
    s["n_distinct_category_pairs_unordered"] = len(cp_cnt)
    s["category_pair_top20_undirected"] = [{"categories": list(k), "n_undirected_pairs": n}
                                           for k, n in sorted(cp_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    # 与 COCO-spatial 的重叠
    pi = set(pair_imgs)
    scanned = set(image_ids)
    s["coco_spatial_overlap"] = {
        "n_coco_spatial_one_images": len(in_one & scanned), "n_coco_spatial_two_images": len(in_two & scanned),
        "n_coco_spatial_either_images": len((in_one | in_two) & scanned),
        "n_pair_images": len(pi),
        "pair_images_in_one": len(pi & in_one), "pair_images_in_two": len(pi & in_two),
        "pair_images_in_either": len(pi & (in_one | in_two)), "pair_images_in_neither": len(pi - in_one - in_two),
        "directed_pairs_in_one": sum(r["in_coco_spatial_one"] for r in records),
        "directed_pairs_in_two": sum(r["in_coco_spatial_two"] for r in records),
        "directed_pairs_in_either": sum(r["in_coco_spatial_one"] or r["in_coco_spatial_two"] for r in records),
        "directed_pairs_in_neither": sum(not (r["in_coco_spatial_one"] or r["in_coco_spatial_two"]) for r in records),
    }
    return s


def write_report(s: dict, f: Path, pairs_path_display: str):
    def pct(n, d):
        return f"{100.0 * n / d:.1f}%" if d else "-"
    nd = s["n_directed_pairs"]
    L = []
    L += ["# F3 COCO-Pairs：构建规则与分布", "",
          "本文件由 `scripts/build_coco_pairs.py` 自动生成，所有数字都是脚本实际算出来的；没有任何模型推理、没有准确率。",
          f"扫描图像数：{s['n_images_scanned']}（COCO val2017 全部图像，未下采样）。种子 {s['seed']}。", "",
          f"- 数据：`{pairs_path_display}`（每行一条有向 pair；sha256 `{s['pairs_jsonl']['sha256']}`，"
          f"{s['pairs_jsonl']['n_lines']} 行，{s['pairs_jsonl']['bytes']} 字节）",
          "- 划分：`splits/coco_pairs_v2.json`（只存图像 id；继承 COCO-spatial 与 COCO-Pairs 的统一图像划分 `splits/coco_images_v2.json`）",
          "- 统计：`manifests/coco_pairs_summary.json`",
          "- 加载：`from src.data.coco_pairs import load_coco_pairs; load_coco_pairs(split=None|'dev'|'test')`", ""]
    L += ["## 1. 规则（含需要解释的地方）", ""]
    r = s["rules"]
    L += ["候选物体（依次施加）：", "",
          "1. `iscrowd == 0`；",
          "2. 类别唯一：该物体的类别在这张图的**全部**标注里恰好出现 1 次（计数时把 `iscrowd=1` 的标注也算进去，"
          "所以“1 个 person + 1 块 person crowd”不算唯一）；",
          f"3. 面积：COCO 标注的 `area` 字段 / (width×height) ∈ [{cp.MIN_AREA_FRAC:.0%}, {cp.MAX_AREA_FRAC:.0%}]（闭区间）。"
          "用的是 COCO `area` 字段而不是解码像素数，两者的差异见下面“面积核对”；",
          f"4. 图像短边 ≥ {cp.MIN_SHORT_SIDE} px。", "",
          "配对与标签：", "",
          f"- 配对：{r['pairing']}。`pair_id = <12 位 image_id>_<target ann_id>_<reference ann_id>`。",
          f"- 质心：{r['centroid']}。",
          f"- 关系：{r['relation']}。",
          f"- 平局：{r['tie']}。实际出现次数：质心完全重合 {s['n_centroid_ties']} 对；|dx|==|dy|（6 位小数下）{s['n_abs_dx_eq_abs_dy_undirected']} 对。",
          f"- `norm_gap`：{r['norm_gap']}。“质心距离”按字面取欧氏距离；另存 `norm_gap_axis`（{r['norm_gap_axis']}）。本文分箱默认用 `norm_gap`。",
          f"- `bbox_iou`：{r['bbox']} 后计算；`mask_iou`：pycocotools 对两个实例 mask 的 RLE 求 IoU。",
          f"- `containment_target_in_reference` / `containment_reference_in_target`：{r['containment']}。",
          f"- `size_ratio`：{r['size_ratio']}。",
          f"- `n_other_instances`：{r['n_other_instances']}。",
          f"- `proj_overlap`：{r['proj_overlap']}；另存 `proj_overlap_frac` = 重叠长度 / 较短的那个投影长度。重叠表示关系本身有歧义。",
          "- **没有过滤任何难例**：重叠、包含、投影重叠、极近的 pair 全部保留，只打标记。",
          "- `caption_true` / `caption_options` 不在 jsonl 里，由加载器按 COCO-spatial 两物体子集的模板生成"
          "（`A photo of a X to the left of a Y` / `A photo of a X above a Y`；第二个选项是相反关系）。", ""]
    L += ["## 2. 候选物体漏斗", ""]
    fn = s["funnel_objects"]
    L += [md_table(["步骤", "剩余标注数"], [
        ["全部标注（含 crowd）", fn.get("annotations_all", 0)],
        ["a) iscrowd=0", fn.get("a_noncrowd", 0)],
        ["b) 类别在图中唯一", fn.get("b_unique_category", 0)],
        ["c) 面积占比 1%–40%", fn.get("c_area_1_40pct", 0)],
        ["d) 图像短边 ≥ 300（= 候选物体）", fn.get("d_short_side_300", 0)],
    ]), ""]
    L += [md_table(["量", "数值"], [
        ["候选物体数", s["n_candidate_objects"]],
        ["至少 1 个候选的图像数", s["n_images_with_ge1_candidate"]],
        ["≥2 个候选的图像数（= 有 pair 的图像）", s["n_images_with_ge2_candidates"]],
        ["进入 pair 的物体数", s["n_objects_in_pairs"]],
        ["无向 pair 数", s["n_undirected_pairs"]],
        ["有向 pair 数（jsonl 行数）", s["n_directed_pairs"]],
        ["候选物体的不同类别数", s["n_distinct_categories_candidates"]],
        ["进入 pair 的不同类别数", s["n_distinct_categories_in_pairs"]],
    ]), ""]
    L += ["每张图的候选物体个数：", "",
          md_table(["候选数", "图像数"], [[k, v] for k, v in sorted(s["images_by_n_candidates"].items(), key=lambda kv: int(kv[0]))]), ""]
    ac = s["area_check"]
    L += ["### 面积核对（COCO `area` 对比 pycocotools 解码像素数）", "",
          f"对**全部** {ac['n_objects_checked']} 个候选物体都解码了 mask（不是抽样）。相对差 = (解码像素数 − COCO area) / COCO area。", "",
          md_table(["统计量", "数值"], [
              ["相对差 最小 / p5 / 中位 / p95 / 最大",
               " / ".join(f"{ac['rel_diff'].get(k, float('nan')):.4f}" for k in ("p0", "p5", "p50", "p95", "p100"))],
              ["abs(相对差) 均值", ac["abs_rel_diff_mean"]], ["abs(相对差) 最大", ac["abs_rel_diff_max"]],
              ["abs(相对差) > 1% 的物体数", ac["n_abs_rel_diff_gt_1pct"]],
              ["abs(相对差) > 5% 的物体数", ac["n_abs_rel_diff_gt_5pct"]],
              ["若改用解码面积，会落到 1%–40% 之外的候选数", ac["n_candidates_outside_1_40pct_if_decoded_area_were_used"]],
          ]), "",
          "筛选阈值和 `size_ratio` 用 COCO `area`；`mask_iou` 与质心用解码后的 mask；每个物体同时记录 `area` 与 `area_decoded`。", ""]
    L += ["## 3. 关系分布（有向 pair）", "",
          md_table(["关系", "有向 pair 数", "占比", "其中投影重叠", "投影重叠占比"],
                   [[k, v, pct(v, nd), s["proj_overlap"]["per_relation"][k], pct(s["proj_overlap"]["per_relation"][k], v)]
                    for k, v in s["per_relation"].items()]), "",
          f"无向 pair 按关系轴：horizontal {s['per_axis_undirected']['horizontal']}，vertical {s['per_axis_undirected']['vertical']}。"
          "因为两个方向都保留，left 与 right、above 与 below 的数量必然相等。", ""]
    L += ["## 4. 归一化间距分箱", "",
          md_table(["norm_gap 区间", "有向", "无向", "占比", "其中投影重叠（有向）", "按 norm_gap_axis 分箱（有向）"],
                   [[b, v, v // 2, pct(v, nd), s["proj_overlap"]["per_gap_bin"][b], s["per_gap_axis_bin"][b]]
                    for b, v in s["per_gap_bin"].items()]), "",
          "区间左闭右开（例如 `0.5-1` 表示 0.5 ≤ gap < 1）。", ""]
    L += ["## 5. 拥挤度分箱（图中其他实例数）", "",
          md_table(["其他实例数", "有向（全部标注）", "无向", "占比", "有向（只数非 crowd）", "图像数（全部标注）"],
                   [[b, v, v // 2, pct(v, nd), s["per_crowding_bin_noncrowd"][b], s["images_per_crowding_bin"][b]]
                    for b, v in s["per_crowding_bin"].items()]), ""]
    gl = [b[0] for b in cp.GAP_BINS]
    L += ["## 6. 关系 × 间距 交叉表（有向 pair）", "",
          md_table(["关系 \\ norm_gap"] + gl + ["合计"],
                   [[k] + [s["relation_x_gap_bin"][k][g] for g in gl] + [s["per_relation"][k]] for k in cp.RELATIONS]
                   + [["合计"] + [s["per_gap_bin"][g] for g in gl] + [nd]]), ""]
    cl = [b[0] for b in cp.CROWD_BINS]
    L += ["### 关系 × 拥挤度（有向 pair）", "",
          md_table(["关系 \\ 其他实例数"] + cl + ["合计"],
                   [[k] + [s["relation_x_crowding_bin"][k][c] for c in cl] + [s["per_relation"][k]] for k in cp.RELATIONS]
                   + [["合计"] + [s["per_crowding_bin"][c] for c in cl] + [nd]]), "",
          "### 间距 × 拥挤度（有向 pair）", "",
          md_table(["norm_gap \\ 其他实例数"] + cl + ["合计"],
                   [[g] + [s["gap_bin_x_crowding_bin"][g][c] for c in cl] + [s["per_gap_bin"][g]] for g in gl]), ""]
    po = s["proj_overlap"]
    L += ["## 7. 歧义与重叠", "",
          md_table(["条件", "有向 pair 数", "无向 pair 数", "占比"], [
              ["关系轴上投影重叠（proj_overlap）", po["directed"], po["directed"] // 2, pct(po["directed"], nd)],
              ["proj_overlap_frac ≥ 0.5", po["frac_ge_0.5_directed"], po["frac_ge_0.5_directed"] // 2, pct(po["frac_ge_0.5_directed"], nd)],
              ["proj_overlap_frac = 1（一个投影完全落在另一个里）", po["frac_eq_1_directed"], po["frac_eq_1_directed"] // 2, pct(po["frac_eq_1_directed"], nd)],
              ["bbox IoU > 0", s["bbox_iou_gt0_directed"], s["bbox_iou_gt0_directed"] // 2, pct(s["bbox_iou_gt0_directed"], nd)],
              ["bbox IoU ≥ 0.5", s["bbox_iou_ge_0.5_directed"], s["bbox_iou_ge_0.5_directed"] // 2, pct(s["bbox_iou_ge_0.5_directed"], nd)],
              ["mask IoU > 0", s["mask_iou_gt0_directed"], s["mask_iou_gt0_directed"] // 2, pct(s["mask_iou_gt0_directed"], nd)],
              ["target bbox ≥90% 落在 reference bbox 内（有向量）", s["containment_target_in_reference_ge_0.9_directed"], "-",
               pct(s["containment_target_in_reference_ge_0.9_directed"], nd)],
              ["target bbox 100% 落在 reference bbox 内（有向量）", s["containment_target_in_reference_eq_1_directed"], "-",
               pct(s["containment_target_in_reference_eq_1_directed"], nd)],
          ]), ""]
    qk = ["p0", "p5", "p25", "p50", "p75", "p95", "p100", "mean"]
    rows = [[k] + [v.get(q, "-") for q in qk] for k, v in s["metric_quantiles_undirected"].items()]
    rows += [[k + "（有向）"] + [v.get(q, "-") for q in qk] for k, v in s["metric_quantiles_directed"].items()]
    L += ["### 连续量的分位数（对称量按无向 pair 统计）", "",
          md_table(["量", "最小", "p5", "p25", "中位", "p75", "p95", "最大", "均值"], rows), ""]
    L += ["## 8. 类别", "",
          f"进入 pair 的不同类别数：{s['n_distinct_categories_in_pairs']}（COCO 共 80 类）；"
          f"不同的无序类别组合数：{s['n_distinct_category_pairs_unordered']}。", "",
          "出现最多的 20 个类别（按进入 pair 的物体数排序）：", "",
          md_table(["类别", "进入 pair 的物体数", "作为 target 的有向 pair 数", "候选物体数"],
                   [[c["category"], c["n_objects_in_pairs"], c["n_directed_pairs_as_target"], c["n_candidate_objects"]]
                    for c in s["category_top20"]]), "",
          "出现最多的 20 个类别组合（无向）：", "",
          md_table(["类别组合", "无向 pair 数"], [[" + ".join(c["categories"]), c["n_undirected_pairs"]]
                                          for c in s["category_pair_top20_undirected"]]), ""]
    o = s["coco_spatial_overlap"]
    L += ["## 9. 与 COCO-spatial 的图像重叠", "",
          f"COCO-spatial 用到的不同图像：one {o['n_coco_spatial_one_images']} 张，two {o['n_coco_spatial_two_images']} 张，"
          f"并集 {o['n_coco_spatial_either_images']} 张。", "",
          md_table(["", "COCO-Pairs 图像数", "有向 pair 数"], [
              ["全部", o["n_pair_images"], nd],
              ["也在 coco_spatial_one", o["pair_images_in_one"], o["directed_pairs_in_one"]],
              ["也在 coco_spatial_two", o["pair_images_in_two"], o["directed_pairs_in_two"]],
              ["在 one 或 two", o["pair_images_in_either"], o["directed_pairs_in_either"]],
              ["两者都不在", o["pair_images_in_neither"], o["directed_pairs_in_neither"]],
          ]), "",
          "每条记录带 `in_coco_spatial_one` / `in_coco_spatial_two` 两个标记；没有因此剔除任何图像。", ""]
    sp = s["split"]
    L += ["## 10. dev / test 划分", "",
          f"按**图像**划分（同一张图的所有 pair 在同一侧），对半，种子 {s['seed']}。分层方法：先给每张图算一个分层键 = "
          "（主导关系轴，候选数档）。主导关系轴：该图的无向 pair 里 horizontal 多记 `H`，vertical 多记 `V`，一样多记 `T`"
          "（left/right、above/below 在每张图里数量恒等，所以图像级能分层的只有水平/竖直）；候选数档：2 / 3 / 4+"
          "（决定这张图贡献多少 pair）。每一层内把图像 id 排序后用 `random.Random(f\"{seed}:coco_pairs:{层键}\")` 打乱，"
          "前一半归 dev，奇数时多出的 1 张归 test。间距和拥挤度没有参与分层，下面的表是事后核对。", "",
          md_table(["层（主导轴/候选数）", "图像数", "dev", "test"],
                   [[k, v["n_images"], v["dev"], v["test"]] for k, v in sp["per_stratum"].items()]), "",
          md_table(["", "dev", "test"],
                   [["图像数", sp["counts"]["dev"]["n_images"], sp["counts"]["test"]["n_images"]],
                    ["无向 pair", sp["counts"]["dev"]["n_undirected_pairs"], sp["counts"]["test"]["n_undirected_pairs"]],
                    ["有向 pair", sp["counts"]["dev"]["n_directed_pairs"], sp["counts"]["test"]["n_directed_pairs"]]]
                   + [[f"关系 {k}", sp["counts"]["dev"]["per_relation"][k], sp["counts"]["test"]["per_relation"][k]] for k in cp.RELATIONS]
                   + [[f"norm_gap {g}", sp["counts"]["dev"]["per_gap_bin"][g], sp["counts"]["test"]["per_gap_bin"][g]] for g in gl]
                   + [[f"其他实例数 {c}", sp["counts"]["dev"]["per_crowding_bin"][c], sp["counts"]["test"]["per_crowding_bin"][c]] for c in cl]
                   + [["投影重叠", sp["counts"]["dev"]["proj_overlap"], sp["counts"]["test"]["proj_overlap"]],
                      ["bbox IoU > 0", sp["counts"]["dev"]["bbox_iou_gt0"], sp["counts"]["test"]["bbox_iou_gt0"]],
                      ["mask IoU > 0", sp["counts"]["dev"]["mask_iou_gt0"], sp["counts"]["test"]["mask_iou_gt0"]],
                      ["也在 coco_spatial_one", sp["counts"]["dev"]["in_coco_spatial_one"], sp["counts"]["test"]["in_coco_spatial_one"]],
                      ["也在 coco_spatial_two", sp["counts"]["dev"]["in_coco_spatial_two"], sp["counts"]["test"]["in_coco_spatial_two"]]]), "",
          "（除前两行外，单位都是有向 pair。）", "",
          "注意：COCO-Pairs 的 dev/test 与 COCO-spatial 自己的 dev/test 是各自独立划分的，同一张图可能在两边落在不同侧；"
          "需要严格隔离时请用 `in_coco_spatial_*` 标记过滤。", ""]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(L))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="只对 --out-dir 试跑有效：允许覆盖试跑目录里的划分")
    ap.add_argument("--limit", type=int, default=None, help="只处理 image_id 最小的 N 张图（试跑用，必须同时给 --out-dir）")
    ap.add_argument("--out-dir", type=Path, default=None, help="把全部输出写到这个目录（试跑用），不碰正式位置")
    args = ap.parse_args()
    if args.limit is not None and args.out_dir is None:
        sys.exit("[error] --limit 只用于试跑，必须同时指定 --out-dir，避免把不完整的结果写到正式位置")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        f_pairs, f_split = args.out_dir / "pairs.jsonl", args.out_dir / "coco_pairs.json"
        f_sum, f_md = args.out_dir / "coco_pairs_summary.json", args.out_dir / "F3_coco_pairs.md"
        pairs_display = str(f_pairs)
    else:
        f_pairs, f_split = cp.pairs_file(), cp.split_file()
        f_sum = paths.REPO_ROOT / "manifests" / "coco_pairs_summary.json"
        f_md = paths.REPO_ROOT / "docs" / "parts" / "F3_coco_pairs.md"
        pairs_display = "<data>/coco_pairs/pairs.jsonl"
        f_pairs.parent.mkdir(parents=True, exist_ok=True)

    coco = cp.load_coco_instances()
    image_ids = sorted(coco["images"])
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    in_one, in_two = spatial_image_ids("coco_spatial_one"), spatial_image_ids("coco_spatial_two")
    print(f"[info] 图像 {len(image_ids)} 张；coco_spatial_one {len(in_one)} 张图，coco_spatial_two {len(in_two)} 张图", flush=True)

    cands, funnel = find_candidates(coco, image_ids)
    records, obj_info, n_tie = build_records(coco, cands, in_one, in_two)
    check_records(records)
    print(f"[info] 候选物体 {sum(len(v) for v in cands.values())}，有向 pair {len(records)}", flush=True)

    # 划分：已存在则不覆盖（除非 --force）；已存在的划分必须与当前图像集合一致，否则报错退出
    pair_imgs = sorted({r["image_id"] for r in records})
    if f_split.exists() and (not args.force or args.out_dir is None):
        split = json.load(open(f_split))
        if sorted(split["dev"] + split["test"]) != pair_imgs:
            sys.exit(f"[error] {f_split} 已存在，但它的图像集合与本次构建不一致；拒绝覆盖。需要重做请运行 scripts/make_joint_coco_split.py --force")
        print(f"[skip] {f_split.name} 已存在且与本次构建一致，沿用（不覆盖）")
        write_split = False
    elif args.out_dir is None:
        split, write_split = None, False        # 正式划分由 make_joint_coco_split.py 生成
    else:
        split = make_split(records)
        write_split = True

    # pairs.jsonl（顺序确定：image_id、无序对、方向）
    with open(f_pairs, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
    raw = f_pairs.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if split is None:
        print(f"[ok] {f_pairs} sha256={sha}；划分文件不存在：先运行 scripts/make_joint_coco_split.py，再重跑本脚本生成统计")
        return

    summary = summarize(coco, image_ids, cands, funnel, records, obj_info, n_tie, in_one, in_two)
    summary["limit"] = args.limit
    summary["pairs_jsonl"] = {"path": pairs_display, "sha256": sha, "bytes": len(raw), "n_lines": len(records)}
    counts = split_counts(records, split)
    if write_split:
        split["counts"] = counts
        split["pairs_jsonl_sha256"] = sha
        json.dump(split, open(f_split, "w"), indent=0)
        print(f"[ok] 写入划分 {f_split}")
    summary["split"] = {"file": f"splits/{f_split.name}", "unit": "image", "stratify_by": split["stratify_by"],
                        "per_stratum": split["per_stratum"], "counts": counts}
    json.dump(summary, open(f_sum, "w"), indent=1, ensure_ascii=False)
    write_report(summary, f_md, pairs_display)
    print(f"[ok] {f_pairs} sha256={sha} bytes={len(raw)} lines={len(records)}")
    print(f"[ok] {f_sum}\n[ok] {f_md}")
    print(json.dumps({k: summary[k] for k in ("n_candidate_objects", "n_images_with_ge2_candidates", "n_undirected_pairs",
                                              "n_directed_pairs", "per_relation", "per_gap_bin", "per_crowding_bin")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
