"""真值框（mask 外接框）与数据集自带框的 IoU 分布：COCO-spatial、COCO-Pairs、SpatialRGPT 深度样本（CPU）。
输出 manifests/truth_box_stats.json"""
import json

import numpy as np

from src import paths
from src.annotate.selection import box_iou
from src.data import load_dataset
from src.data.coco_pairs import load_coco_pairs
from src.data.spatialrgpt import depth_samples

out = {}


def add(name, pairs):
    v = np.array([box_iou(a, b) for a, b, _ in pairs])
    out[name] = {"n_objects": len(v), "min": round(float(v.min()), 3), **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in (1, 5, 50)},
                 **{f"n_lt_{t}": int((v < t).sum()) for t in (0.5, 0.8, 0.9, 0.95)},
                 "iou_lt_0.8": sorted(({"id": i, "iou": round(float(x), 3)} for x, (_, _, i) in zip(v, pairs) if x < 0.8), key=lambda d: d["iou"])}
    print(name, {k: x for k, x in out[name].items() if k != "iou_lt_0.8"}, out[name]["iou_lt_0.8"][:12])


for n in ("coco_spatial_one", "coco_spatial_two"):
    add(n, [(b, s.extra[f"coco_bbox_{r}"], f"{s.sample_id}:{r}") for s in load_dataset(n) for r, b in (("target", s.bbox_target), ("reference", s.bbox_reference)) if b])
seen = {}
for s in load_coco_pairs():
    for r, b in (("target", s.bbox_target), ("reference", s.bbox_reference)):
        seen[s.extra[f"{r}_ann_id"]] = (b, s.extra[f"coco_bbox_{r}"], s.extra[f"{r}_ann_id"])
add("coco_pairs(unique objects)", list(seen.values()))
add("spatialrgpt_depth(used)", [(b, s.extra[f"provided_bbox_{r}"], f"{s.sample_id}:{r}") for s in depth_samples() for r, b in (("target", s.bbox_target), ("reference", s.bbox_reference))])
json.dump(out, open(paths.REPO_ROOT / "manifests" / "truth_box_stats.json", "w"), indent=1, ensure_ascii=False)
