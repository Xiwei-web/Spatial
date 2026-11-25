"""给 What's Up v3 定稿补上默认真值框 box_gt_xyxy（CPU，可重复运行；finalize_whatsup_v3.py 重跑时会自带该字段），
并统计检测框与真值框的 IoU 分布、列出 IoU < 0.8 的样本。只新增字段，不改任何已有字段。
输出：annotations.jsonl（原子替换，summary 里的 sha256 同步更新）、manifests/whatsup_auto_v3_box_iou.{json,csv}"""
import csv
import hashlib
import json
import os

import numpy as np

from src import paths
from src.annotate.masks import add_gt_box
from src.annotate.selection import box_iou

M = paths.REPO_ROOT / "manifests"
F = paths.get("data") / "whatsup_annotations" / "auto_v3" / "annotations.jsonl"
raw = F.read_bytes()
summ = json.load(open(M / "whatsup_auto_v3_summary.json"))
assert summ["finalized"] and hashlib.sha256(raw).hexdigest() == summ["annotations_sha256"]
recs = [json.loads(l) for l in raw.decode().splitlines()]
for r in recs:
    before = json.dumps({k: {a: b for a, b in o.items() if a != "box_gt_xyxy"} for k, o in r["objects"].items()})
    add_gt_box(r)
    assert before == json.dumps({k: {a: b for a, b in o.items() if a != "box_gt_xyxy"} for k, o in r["objects"].items()})
    for o in r["objects"].values():
        assert [float(v) for v in o["box_mask_xyxy"]] == o["box_gt_xyxy"], r["sample_id"]      # 与减之前的外接框一致
new = "".join(json.dumps(r) + "\n" for r in recs).encode()
tmp = F.with_suffix(".tmp")
tmp.write_bytes(new)
os.replace(tmp, F)
summ["annotations_sha256"] = hashlib.sha256(new).hexdigest()
summ["default_truth_box"] = "box_gt_xyxy = 原始 mask（减去 target 之前）的外接框"
json.dump(summ, open(M / "whatsup_auto_v3_summary.json", "w"), indent=1, ensure_ascii=False)
rows = []
for r in recs:
    for role, o in r["objects"].items():
        if r["usable"]:
            rows.append({"iou": box_iou(o["box_det_xyxy"], o["box_gt_xyxy"]), "order_v2": r["review_order_v2"], "dataset": r["dataset"], "sample_id": r["sample_id"],
                         "role": role, "name": o["name"], "box_det_xyxy": [round(v) for v in o["box_det_xyxy"]], "box_gt_xyxy": o["box_gt_xyxy"], "v3_status": r["v3_status"]})
out = {"n_usable_images": len(rows) // 2, "dist": {}}
for ds in ("whatsup_a", "whatsup_b", "all"):
    for role in ("target", "reference", "both"):
        v = np.array([x["iou"] for x in rows if ds in ("all", x["dataset"]) and role in ("both", x["role"])])
        out["dist"][f"{ds}/{role}"] = {"n": len(v), "min": round(float(v.min()), 3), **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in (1, 5, 25, 50)},
                                       **{f"n_lt_{t}": int((v < t).sum()) for t in (0.7, 0.8, 0.9, 0.95)}}
low = sorted((x for x in rows if x["iou"] < 0.8), key=lambda x: x["iou"])
out["iou_lt_0.8"] = low
json.dump(out, open(M / "whatsup_auto_v3_box_iou.json", "w"), indent=1, ensure_ascii=False)
with open(M / "whatsup_auto_v3_box_iou.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda x: x["iou"])[:100])
for k, v in out["dist"].items():
    print(k, v)
for x in low:
    print("IoU<0.8", round(x["iou"], 3), x["order_v2"], x["dataset"][-1], x["sample_id"], x["role"], x["name"], x["box_det_xyxy"], x["box_gt_xyxy"])
