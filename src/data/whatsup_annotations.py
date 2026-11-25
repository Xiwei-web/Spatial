"""What's Up 定稿标注的只读加载（下游统一从这里取）。
状态表述：自动标注，经模型辅助预审和部分人工复核（configs/annotation/object_conventions.yaml）。
- 只返回 usable=True 的样本（剔除更正表里的样本）。
- 未定稿，或 annotations.jsonl 与 manifests 里记录的 sha256 不一致时，直接报错。
- 默认真值框（检测框 / mask 外接框）尚未确定，取框时必须显式指明 kind。"""
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


def truth_box(obj: dict, kind: str) -> list[float]:
    """kind: "det"（检测框）或 "mask"（mask 外接框；reference 为减去 target 之前的 mask 的外接框）。"""
    return obj[{"det": "box_det_xyxy", "mask": "box_mask_xyxy"}[kind]]
