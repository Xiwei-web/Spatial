"""参考统计（不改变任何质检结果、不写标注目录）：mask 面积占检测框面积的比例。
用于发现“mask 只覆盖物体一部分”或“mask 把空隙也包进去”这类情况。
输出：<outputs>/annotation_review/<version>_mask_fill/{category_table.csv, top40_anomalous.csv, top40.png}，并把表打印出来。
异常程度：|比例 − 该(物体名称)的中位数| / (1.4826 × 该名称的 MAD)；样本数 < 4 或 MAD=0 的名称改用全体的中位数和 MAD。"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import yaml

from src import paths

cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "whatsup_auto_v2.yaml"
cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / cfg_name))
rows = [json.loads(l) for l in open(paths.get("data") / "whatsup_annotations" / cfg["version"] / "annotations.jsonl")]
order = sorted(rows, key=lambda r: (r["auto_qc"] != "review", -len(r["flags"]), r["dataset"], r["sample_id"]))
recs = []
for i, r in enumerate(order):
    for role in ("target", "reference"):
        o = r["objects"][role]
        if not o["mask_area"]:
            continue
        x1, y1, x2, y2 = o["box_xyxy"]
        recs.append({"review_order": i + 1, "dataset": r["dataset"], "sample_id": r["sample_id"], "role": role, "name": o["name"],
                     "fill": o["mask_area"] / max((x2 - x1) * (y2 - y1), 1e-9), "auto_qc": r["auto_qc"], "score": o["score"]})
allv = np.array([x["fill"] for x in recs])
gmed, gmad = float(np.median(allv)), float(np.median(np.abs(allv - np.median(allv))))
by = defaultdict(list)
for x in recs:
    by[(x["dataset"], x["name"])].append(x["fill"])
out = paths.get("outputs") / "annotation_review" / f"{cfg['version']}_mask_fill"
out.mkdir(parents=True, exist_ok=True)
table = []
for (ds, name), v in sorted(by.items()):
    v = np.array(v)
    table.append([ds, name, len(v), round(float(np.min(v)), 3), round(float(np.percentile(v, 5)), 3), round(float(np.median(v)), 3),
                  round(float(np.percentile(v, 95)), 3), round(float(np.max(v)), 3)])
with open(out / "category_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "object_name", "n", "min", "p5", "median", "p95", "max"]); w.writerows(table)
for x in recs:
    v = np.array(by[(x["dataset"], x["name"])])
    med, mad = float(np.median(v)), float(np.median(np.abs(v - np.median(v))))
    if len(v) < 4 or mad == 0:
        med, mad, x["ref"] = gmed, gmad, "global"
    else:
        x["ref"] = "category"
    x["category_median"] = med
    x["robust_z"] = (x["fill"] - med) / (1.4826 * mad)
top = sorted(recs, key=lambda x: -abs(x["robust_z"]))[:40]
with open(out / "top40_anomalous.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "review_order", "dataset", "sample_id", "role", "object_name", "mask_area_over_box_area", "category_median", "robust_z", "reference_distribution", "auto_qc", "score"])
    for k, x in enumerate(top):
        w.writerow([k + 1, x["review_order"], x["dataset"], x["sample_id"], x["role"], x["name"], round(x["fill"], 3), round(x["category_median"], 3),
                    round(x["robust_z"], 2), x["ref"], x["auto_qc"], round(x["score"], 3)])
# 第二份清单：按“与该名称中位数的绝对差”排序。稳健 z 分数会被分布极窄的类别（如 armchair）主导，绝对差更直观。
top_abs = sorted(recs, key=lambda x: -abs(x["fill"] - x["category_median"]))[:40]
with open(out / "top40_by_abs_deviation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "review_order", "dataset", "sample_id", "role", "object_name", "mask_area_over_box_area", "category_median", "abs_deviation", "auto_qc", "score"])
    for k, x in enumerate(top_abs):
        w.writerow([k + 1, x["review_order"], x["dataset"], x["sample_id"], x["role"], x["name"], round(x["fill"], 3), round(x["category_median"], 3),
                    round(x["fill"] - x["category_median"], 3), x["auto_qc"], round(x["score"], 3)])
print(f"n_objects={len(recs)} global median={gmed:.3f} MAD={gmad:.3f}")
print("dataset | name | n | min | p5 | median | p95 | max")
for t in table:
    print(" | ".join(map(str, t)))
print("TOP40")
for k, x in enumerate(top):
    print(k + 1, x["review_order"], x["dataset"][-1], x["sample_id"], x["role"], x["name"], round(x["fill"], 3), "med", round(x["category_median"], 3), "z", round(x["robust_z"], 1), x["auto_qc"])
json.dump({"global_median": gmed, "global_mad": gmad, "n": len(recs)}, open(out / "summary.json", "w"))
