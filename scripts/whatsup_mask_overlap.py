"""统计 v3 中 reference 的 mask 把 target 像素也盖进去的程度（CPU，只读标注，不改任何质检结果）。
质检里的重叠检查是两个 mask 的 IoU>0.5；target 很小而 reference 很大时 IoU 很低，抓不到这种情况。
输出：manifests/whatsup_auto_v3_mask_overlap.csv（交集占 target 面积 > 1% 的图，按比例降序）"""
import csv
import json

from pycocotools import mask as mu

from src import paths

rows = []
for line in open(paths.get("data") / "whatsup_annotations" / "auto_v3" / "annotations.jsonl"):
    r = json.loads(line)
    t, q = (mu.decode(r["objects"][k]["mask_rle"]).astype(bool) for k in ("target", "reference"))
    i = int((t & q).sum())
    if i / t.sum() > 0.01:
        rows.append([round(i / t.sum(), 3), round(i / q.sum(), 4), r["review_order_v2"], r["dataset"], r["sample_id"], r["relation"], r["v3_status"], r["usable"]])
rows.sort(reverse=True)
with open(paths.REPO_ROOT / "manifests" / "whatsup_auto_v3_mask_overlap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["inter_over_target", "inter_over_reference", "order_v2", "dataset", "sample_id", "relation", "v3_status", "usable"])
    w.writerows(rows)
print(len(rows))
