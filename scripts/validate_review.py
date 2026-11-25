"""检查人工审核表 review.csv：human_verdict 是否只取允许的值、有多少行未填、各取值的数量。
用法：python scripts/validate_review.py <review.csv> [--config whatsup_auto_v2.yaml]"""
import csv
import sys
from collections import Counter

import yaml

from src import paths

cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "whatsup_auto_v2.yaml"
allowed = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / cfg_name))["review"]["allowed_human_verdicts"]
rows = list(csv.DictReader(open(sys.argv[1])))
vals = [(r.get("human_verdict") or "").strip() for r in rows]
bad = [(r["order"], r["sample_id"], v) for r, v in zip(rows, vals) if v and v not in allowed]
empty = sum(1 for v in vals if not v)
print(f"rows={len(rows)} filled={len(rows) - empty} empty={empty} invalid={len(bad)}")
print("allowed:", allowed)
print("counts:", dict(Counter(v for v in vals if v)))
for b in bad[:20]:
    print("  invalid:", b)
print("review rows still empty among auto_qc=review:", sum(1 for r, v in zip(rows, vals) if r["auto_qc"] == "review" and not v))
sys.exit(1 if bad else 0)
