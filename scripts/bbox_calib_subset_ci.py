"""第 4 轮补充（CPU，只读 dev 标定 JSONL）：子集内两个候选的平均 IoU 差的图像级 bootstrap 区间（与 src/bbox_calib.py 同一函数、同一种子），
以及 COCO 子集里模型输入尺寸与原图尺寸之比的分布。用法：python scripts/bbox_calib_subset_ci.py → manifests/bbox_calib_round4_extra.json"""
import json

import numpy as np

from scripts.bbox_calib_table import load
from src import paths
from src.bbox_calib import bootstrap_diff, build

out = {}
for name, sub, a, b in (("Qwen3-VL-8B-Instruct", "coco", ("P0", "norm1000_xyxy"), ("P1", "norm1000_xyxy")),
                        ("RoboBrain2.0-7B", "coco", ("P0", "orig_xyxy"), ("P0", "model_xyxy")),
                        ("RoboBrain2.0-7B", "whatsup", ("P0", "orig_xyxy"), ("P0", "model_xyxy")),
                        ("VST-7B-RL", "coco", ("P0", "model_xyxy"), ("P2", "model_xyxy")),
                        ("VST-7B-RL", "whatsup", ("P0", "model_xyxy"), ("P2", "model_xyxy"))):
    recs, prompts, last = load(name)
    d = build(recs, prompts)
    m = d["subset"] == sub
    diff = (d["cand"][a]["iou"] - d["cand"][b]["iou"])[m]
    img = [k for k, z in zip(d["image"], m) if z]
    out[f"{name}/{sub}/{a[0]},{a[1]} − {b[0]},{b[1]}"] = {"n_items": int(m.sum()), "n_images": len(set(img)), "mean_a": float(d["cand"][a]["iou"][m].mean()), "mean_b": float(d["cand"][b]["iou"][m].mean()),
                                                         "diff_mean": float(diff.mean()), "diff_ci95": bootstrap_diff(diff, img)}
recs, prompts, last = load("RoboBrain2.0-7B")
rat = {"coco": [], "whatsup": []}
seen = set()
for (p, iid), r in last.items():
    if p != "P0" or r["image_key"] in seen:
        continue
    seen.add(r["image_key"])
    (mw, mh), (W, H) = r["model_wh"], r["orig_wh"]
    rat[r["subset"]].append((mw / W, mh / H))
for k, v in rat.items():
    v = np.array(v)
    out[f"RoboBrain2.0-7B/{k}/model_wh ÷ orig_wh（每张图一次）"] = {"n_images": len(v), **{f"{ax}_{q}": float(np.quantile(v[:, i], qq)) for i, ax in enumerate(("w", "h")) for q, qq in (("min", 0), ("median", .5), ("max", 1))},
                                                                 "n_both_exactly_1": int(((v[:, 0] == 1) & (v[:, 1] == 1)).sum()), "max_abs_dev_from_1": float(np.abs(v - 1).max())}
json.dump(out, open(paths.REPO_ROOT / "manifests" / "bbox_calib_round4_extra.json", "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False))
