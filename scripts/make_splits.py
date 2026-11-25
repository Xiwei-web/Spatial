"""生成固定的 dev/test 划分：对半、按关系类别分层、固定随机种子。输出 splits/<dataset>.json（进 Git）。
只依赖标注文件，不依赖图像。已存在的划分文件不会被覆盖（除非 --force），保证划分一经提交不再变。
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict

from src import paths
from src.data import DATASETS, load_dataset
from src.data.splits import ACTIVE

SEED = 20261125


def main():
    force = "--force" in sys.argv
    out_dir = paths.REPO_ROOT / "splits"
    out_dir.mkdir(exist_ok=True)
    for name in DATASETS:
        if name in ACTIVE:          # COCO-spatial 的划分由 scripts/make_joint_coco_split.py 统一生成
            print(f"[skip] {name}: 由 make_joint_coco_split.py 生成")
            continue
        f = out_dir / f"{name}.json"
        if f.exists() and not force:
            print(f"[skip] {f.name} 已存在")
            continue
        samples = load_dataset(name)
        by_rel = defaultdict(list)
        for s in samples:
            by_rel[s.relation].append(s.sample_id)
        dev, test = [], []
        for rel in sorted(by_rel):
            ids = sorted(by_rel[rel])
            # 每个 (数据集, 关系) 用独立且确定的随机流
            rng = random.Random(f"{SEED}:{name}:{rel}")
            rng.shuffle(ids)
            h = len(ids) // 2          # 奇数时多出的 1 个归 test
            dev += ids[:h]
            test += ids[h:]
        dev.sort(); test.sort()
        assert not set(dev) & set(test) and len(dev) + len(test) == len(samples) == len({s.sample_id for s in samples})
        ann_sha = hashlib.sha256(json.dumps([s.sample_id for s in samples]).encode()).hexdigest()
        json.dump({"dataset": name, "seed": SEED, "stratify_by": "relation", "n_total": len(samples),
                   "n_dev": len(dev), "n_test": len(test),
                   "per_relation": {r: len(v) for r, v in sorted(by_rel.items())},
                   "sample_ids_sha256": ann_sha, "dev": dev, "test": test},
                  open(f, "w"), indent=0)
        print(f"[ok] {name}: total={len(samples)} dev={len(dev)} test={len(test)} per_rel={ {r: len(v) for r, v in sorted(by_rel.items())} }")


if __name__ == "__main__":
    main()
