"""What's Up 定稿标注的只读加载（下游统一从这里取）。
状态表述：自动标注，经模型辅助预审和部分人工复核（configs/annotation/object_conventions.yaml）。
- 只返回 usable=True 的样本（剔除更正表里的样本）。
- 未定稿，或 annotations.jsonl 与 manifests 里记录的 sha256 不一致时，直接报错。
- 默认真值框 = 原始 mask（减去 target 之前）的外接框 box_gt_xyxy（用户 2026-11-25）；检测框 box_det_xyxy 只作对照。"""
from __future__ import annotations

import hashlib
import json

from src import paths


def load_annotations(version: str = "auto_v3") -> list[dict]:
    summ = json.load(open(paths.REPO_ROOT / "manifests" / f"whatsup_{version}_summary.json"))
    if not summ.get("finalized"):
        raise RuntimeError(f"{version} 尚未定稿，不得用于下游")
    f = paths.get("data") / "whatsup_annotations" / version / "annotations.jsonl"
    raw = f.read_bytes()
    if hashlib.sha256(raw).hexdigest() != summ["annotations_sha256"]:
        raise RuntimeError(f"{f} 的 sha256 与 manifests 记录不一致")
    return [r for r in map(json.loads, raw.decode().splitlines()) if r["usable"]]


def truth_box(obj: dict, kind: str = "gt") -> list[float]:
    """kind: "gt"（默认真值框：原始 mask 的外接框）或 "det"（检测框，只作对照）。"""
    return obj[{"gt": "box_gt_xyxy", "det": "box_det_xyxy"}[kind]]
