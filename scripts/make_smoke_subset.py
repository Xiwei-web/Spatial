"""从 What's Up dev 划分里固定抽 20 张（A、B 各 10，按关系轮流取，固定种子）给冒烟测试用。
输出 splits/smoke20.json（进 Git）。这 20 张只用于确认管线跑通，其正确率不是结果。"""
import json
import random
from collections import defaultdict

from src import paths
from src.data import load_split

out = {"note": "smoke test only: accuracy on these 20 images is NOT a result", "seed": 20261125, "items": []}
for ds in ("whatsup_a", "whatsup_b"):
    by_rel = defaultdict(list)
    for s in load_split(ds, "dev"):
        by_rel[s.relation].append(s.sample_id)
    rels = sorted(by_rel)
    for r in rels:
        random.Random(f"smoke:{ds}:{r}").shuffle(by_rel[r])
    picked, i = [], 0
    while len(picked) < 10:
        r = rels[i % len(rels)]
        picked.append(by_rel[r][i // len(rels)])
        i += 1
    out["items"] += [{"dataset": ds, "sample_id": p} for p in sorted(picked)]
f = paths.REPO_ROOT / "splits" / "smoke20.json"
if f.exists():
    print("[skip] smoke20.json 已存在")
else:
    json.dump(out, open(f, "w"), indent=1)
    print("[ok]", len(out["items"]))
