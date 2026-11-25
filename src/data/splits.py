"""划分文件的登记与读取。作废的划分文件留在 splits/ 里（带 "status": "deprecated"），任何读取都会被拒绝。"""
from __future__ import annotations

import json

from src import paths

# 数据集 → 现行划分文件；没列出的用 splits/<数据集>.json
ACTIVE = {
    "coco_spatial_one": "coco_spatial_one_v2", "coco_spatial_two": "coco_spatial_two_v2",
    "coco_pairs": "coco_pairs_v2", "synth_cutouts": "synth_cutouts_v2",
    "gqa_spatial_one": "gqa_spatial_one_v2", "gqa_spatial_two": "gqa_spatial_two_v2",
}


def split_path(name: str):
    return paths.REPO_ROOT / "splits" / f"{ACTIVE.get(name, name)}.json"


def read_split(name: str) -> dict:
    f = split_path(name)
    d = json.load(open(f))
    if d.get("status") == "deprecated":
        raise RuntimeError(f"{f.name} 已作废（由 {d.get('superseded_by')} 取代），不得使用")
    return d
