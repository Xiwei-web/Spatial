"""SpatialRGPT 深度样本（96 条，同名物体的配对已跳过）按 F7 的规则全量改写（Xiwei 2026-11-26）。CPU。
规则：src/data/spatialrgpt.py 的 rewrite_record（same_class="skip"），不改规则。原文保留。
输出：<data>/depth/spatialrgpt_derived/depth_rewritten_v1.jsonl（含题目原文与改写；SpatialRGPT 只允许评测、不得再分发，所以不进仓库），
      manifests/spatialrgpt_depth_rewrite_v1.json（只有 id、状态、标记与计数，不含原文）。"""
import collections
import hashlib
import json

from src import paths
from src.data import spatialrgpt as sr

R = {r.id: r for r in sr.load_records()}
S = sr.depth_samples()
out_dir = paths.get("data") / "depth" / "spatialrgpt_derived"
out_dir.mkdir(parents=True, exist_ok=True)
rows, man = [], []
for s in S:
    r = R[s.sample_id]
    w = sr.rewrite_record(r, same_class="skip")
    rec = {"id": r.id, "category": r.category, "classes": r.classes[:2], "target": s.target, "reference": s.reference, "relation": s.relation,
           "relation_source": s.extra["relation_source"], "relation_inferred_from_bare_no": s.extra["relation_inferred_from_bare_no"],
           "question_region_original": r.question_region, "question_mask_original": r.question_mask, "answer_original": r.answer,
           "rewrite_status": w["status"], "question_rewritten": w["question"], "answer_rewritten": w["answer"], "rewrite_notes": w["notes"],
           "bbox_target": s.bbox_target, "bbox_reference": s.bbox_reference,
           "box_metric_exclude_target": s.extra["box_metric_exclude_target"], "box_metric_exclude_reference": s.extra["box_metric_exclude_reference"]}
    rows.append(rec)
    man.append({k: rec[k] for k in ("id", "category", "relation", "relation_source", "rewrite_status", "rewrite_notes", "box_metric_exclude_target", "box_metric_exclude_reference")})
raw = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode()
(out_dir / "depth_rewritten_v1.jsonl").write_bytes(raw)
summ = {"n": len(rows), "rules": "src/data/spatialrgpt.py: rewrite_record(same_class='skip')；同名物体的配对（14 条）已在 depth_samples 里跳过",
        "status": dict(collections.Counter(x["rewrite_status"] for x in rows)), "notes": dict(collections.Counter(n for x in rows for n in x["rewrite_notes"])),
        "n_box_metric_excluded_objects": sum(bool(x["box_metric_exclude_target"]) + bool(x["box_metric_exclude_reference"]) for x in rows),
        "derived_file": "<data>/depth/spatialrgpt_derived/depth_rewritten_v1.jsonl（不进仓库）", "derived_sha256": hashlib.sha256(raw).hexdigest(),
        "license_note": "SpatialRGPT-Bench 只允许评测、不得再分发；本清单不含题目原文", "items": man}
json.dump(summ, open(paths.REPO_ROOT / "manifests" / "spatialrgpt_depth_rewrite_v1.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in summ.items() if k != "items"})
for x in rows[:3]:
    print(x["id"], "|", x["question_region_original"], "→", x["question_rewritten"], "|", x["answer_original"], "→", x["answer_rewritten"])
