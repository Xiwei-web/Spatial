"""检查审核表：final_verdict（没有该列时退回 human_verdict）是否只取允许的值、review_source 是否合法、候选编号格式、各取值数量。
用法：python scripts/validate_review.py <审核表.csv> [--config whatsup_auto_v2.yaml]
允许的 verdict 取值来自配置的 review.allowed_human_verdicts；review_source 取值来自 configs/annotation/object_conventions.yaml。
候选编号（target_pick / reference_pick）：空、1-5 的整数、none；带问号的（如 "3?"、"?"）视为“待确认”，单独计数不算非法。"""
import csv
import re
import sys
from collections import Counter

import yaml

from src import paths

cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "whatsup_auto_v2.yaml"
A = paths.REPO_ROOT / "configs" / "annotation"
allowed = yaml.safe_load(open(A / cfg_name))["review"]["allowed_human_verdicts"]
sources = list(yaml.safe_load(open(A / "object_conventions.yaml"))["review_provenance"])
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
col = "final_verdict" if "final_verdict" in rows[0] else "human_verdict"
vals = [(r.get(col) or "").strip() for r in rows]
bad = [(r["order"], r["sample_id"], v) for r, v in zip(rows, vals) if v and v not in allowed]
empty = sum(1 for v in vals if not v)
print(f"file={sys.argv[1]}\nverdict column={col} rows={len(rows)} filled={len(rows) - empty} empty={empty} invalid={len(bad)}")
print("allowed:", allowed)
print("verdict counts:", dict(Counter(v for v in vals if v)))
rc = 1 if bad or (col == "final_verdict" and empty) else 0
if "review_source" in rows[0]:
    sc = Counter((r.get("review_source") or "").strip() for r in rows)
    bad_src = {k: v for k, v in sc.items() if k not in sources}
    print("review_source counts:", dict(sc), "| invalid:", bad_src)
    rc |= 1 if bad_src else 0
    print("non-ok by source:", dict(Counter(r["review_source"] for r, v in zip(rows, vals) if v != "ok")))
for pc in ("target_pick", "reference_pick"):
    if pc in rows[0]:
        pv = [(r.get(pc) or "").strip() for r in rows]
        unsure = [(r["order"], r["sample_id"], v) for r, v in zip(rows, pv) if "?" in v]
        illegal = [(r["order"], r["sample_id"], v) for r, v in zip(rows, pv) if v and "?" not in v and not re.fullmatch(r"[1-5]|none", v)]
        print(f"{pc}: filled={sum(bool(v) for v in pv)} uncertain(?)={len(unsure)} illegal={len(illegal)}", unsure[:6], illegal[:6])
        rc |= 1 if illegal else 0
for b in bad[:20]:
    print("  invalid verdict:", b)
if col == "human_verdict":
    print("review rows still empty among auto_qc=review:", sum(1 for r, v in zip(rows, vals) if r["auto_qc"] == "review" and not v))
sys.exit(rc)
