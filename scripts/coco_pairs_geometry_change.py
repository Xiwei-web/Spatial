"""COCO-Pairs 几何量从 COCO 自带框改为 mask 外接框后的变化（CPU）：关系标签、间距档、投影重叠变化的有向对，逐条列出。
输出 manifests/coco_pairs_geometry_change.{json,csv}"""
import csv
import json
from collections import Counter

from src import paths
from src.data import coco_pairs as cp

old = {r["pair_id"]: r for r in cp.read_pair_records(cp.pairs_dir() / "pairs_detbox_20261125.jsonl")}
new = cp.read_pair_records()
assert set(old) == {r["pair_id"] for r in new}
rows = []
for r in new:
    o = old[r["pair_id"]]
    assert all(r[f"{k}_detbox"] == o[k] for k in cp.BOX_FIELDS) and all(r[k] == o[k] for k in o if k not in cp.BOX_FIELDS + ("target", "reference"))
    b0, b1 = cp.bin_label(o["norm_gap"], cp.GAP_BINS), cp.bin_label(r["norm_gap"], cp.GAP_BINS)
    ch = [k for k, c in (("relation", o["relation"] != r["relation"]), ("gap_bin", b0 != b1), ("proj_overlap", o["proj_overlap"] != r["proj_overlap"])) if c]
    if ch:
        rows.append({"pair_id": r["pair_id"], "changed": "+".join(ch), "relation_old": o["relation"], "relation_new": r["relation"], "gap_bin_old": b0, "gap_bin_new": b1,
                     "norm_gap_old": o["norm_gap"], "norm_gap_new": r["norm_gap"], "proj_overlap_old": o["proj_overlap"], "proj_overlap_new": r["proj_overlap"],
                     "target": r["target"]["category"], "reference": r["reference"]["category"]})
with open(paths.REPO_ROOT / "manifests" / "coco_pairs_geometry_change.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
d = [abs(r["norm_gap"] - old[r["pair_id"]]["norm_gap"]) for r in new]
res = {"n_pairs": len(new), "n_relation_changed": sum("relation" in r["changed"] for r in rows), "n_gap_bin_changed": sum("gap_bin" in r["changed"] for r in rows),
       "n_proj_overlap_changed": sum("proj_overlap" in r["changed"] for r in rows), "max_abs_norm_gap_change": max(d), "mean_abs_norm_gap_change": sum(d) / len(d),
       "gap_bin_transitions": dict(Counter(f"{r['gap_bin_old']} -> {r['gap_bin_new']}" for r in rows if "gap_bin" in r["changed"])),
       "n_proj_overlap": {"old": sum(o["proj_overlap"] for o in old.values()), "new": sum(r["proj_overlap"] for r in new)},
       "n_bbox_iou_gt0": {"old": sum(o["bbox_iou"] > 0 for o in old.values()), "new": sum(r["bbox_iou"] > 0 for r in new)},
       "per_gap_bin_new": dict(Counter(cp.bin_label(r["norm_gap"], cp.GAP_BINS) for r in new))}
json.dump(res, open(paths.REPO_ROOT / "manifests" / "coco_pairs_geometry_change.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
