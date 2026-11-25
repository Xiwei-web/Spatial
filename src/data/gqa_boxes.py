"""GQA-spatial 的框：把 caption 里的物体名匹配到 GQA scene graph 的物体上。

匹配规则（保守，按顺序；前一条有候选就不再看后一条）：
  R1 exact      物体的 `name` 与短语完全相同（两边都只做小写 + 去首尾空格）。
  R2 attr+name  仅当 R1 没有任何候选时：把短语切成「前缀词 + 后缀」，后缀与物体 `name` 完全相同，
                且前缀的每个词都在该物体的 `attributes` 里（例如 "green trees" → name "trees" 且带属性 "green"）。
                所有切分位置的候选取并集。
只有恰好 1 个候选时才给框（status = unique）；0 个 → none；多于 1 个 → multiple；
图像不在 scene graph 文件里 → image-not-in-scene-graphs。**不用位置标签（left/right/…）在多个候选里挑。**

没有做单复数、同义词归一化：实测 R1 + R2 之外剩下的未匹配都是同义词/截断（"mobile phone" ↔ "cell phone"、
"meter" ↔ "parking meter"），单复数规则一条都救不回来（见 docs/parts/F2_gqa_boxes.md），所以不加。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src import paths

UNIQUE, NONE, MULTIPLE, NO_IMAGE = "unique", "none", "multiple", "image-not-in-scene-graphs"
NO_FILE = "boxes-file-missing"          # 框文件不存在时加载器用的状态
BOXES_FILENAME = "gqa_spatial_boxes.json"


def _norm(s: str) -> str:
    return s.strip().lower()


def find_candidates(phrase: str, objects: dict[str, dict]) -> tuple[str | None, list[str]]:
    """返回 (命中的规则名, 候选物体 id 列表（按 id 字符串排序，确定性）)。没有候选时规则名为 None。"""
    ph = _norm(phrase)
    cands = sorted(oid for oid, o in objects.items() if _norm(o["name"]) == ph)
    if cands:
        return "exact", cands
    words = ph.split()
    hit = set()
    for k in range(1, len(words)):
        attrs, name = words[:k], " ".join(words[k:])
        for oid, o in objects.items():
            if _norm(o["name"]) == name and all(a in {_norm(x) for x in o.get("attributes", [])} for a in attrs):
                hit.add(oid)
    if hit:
        return "attr+name", sorted(hit)
    return None, []


def xyxy(o: dict) -> list[float]:
    """scene graph 的 x, y, w, h（左上角 + 宽高，原图像素）→ [x1, y1, x2, y2]。"""
    return [float(o["x"]), float(o["y"]), float(o["x"] + o["w"]), float(o["y"] + o["h"])]


def match_phrase(phrase: str, graph: dict | None) -> dict:
    """对一个短语给出匹配记录：status / rule / n_candidates / bbox（仅 unique 时非空）/ 候选框。"""
    if graph is None:
        return {"name": phrase, "status": NO_IMAGE, "rule": None, "n_candidates": 0, "bbox": None,
                "object_id": None, "candidate_boxes": []}
    objs = graph["objects"]
    rule, cands = find_candidates(phrase, objs)
    status = UNIQUE if len(cands) == 1 else (NONE if not cands else MULTIPLE)
    rec = {"name": phrase, "status": status, "rule": rule, "n_candidates": len(cands),
           "bbox": xyxy(objs[cands[0]]) if status == UNIQUE else None,
           "object_id": cands[0] if status == UNIQUE else None,
           "candidate_boxes": [xyxy(objs[c]) for c in cands] if status == MULTIPLE else []}
    if rule == "attr+name":   # 记录同名（不看属性）的物体数，>1 说明是靠属性才唯一
        base = _norm(objs[cands[0]]["name"])
        rec["n_same_name"] = sum(_norm(o["name"]) == base for o in objs.values())
    return rec


def side_of(box: list[float], wh: tuple[int, int], relation: str) -> bool | None:
    """框中心是否落在 caption 所说的那一侧（只对 left/right/top/bottom 有定义，其它返回 None）。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = wh
    return {"left": cx < w / 2, "right": cx > w / 2, "top": cy < h / 2, "bottom": cy > h / 2}.get(relation)


def boxes_file() -> Path | None:
    """优先用仓库里的副本（随代码走），其次用数据目录里的。都没有返回 None。"""
    for f in (paths.REPO_ROOT / "manifests" / BOXES_FILENAME, paths.get("data") / "gqa" / BOXES_FILENAME):
        if f.is_file():
            return f
    return None


@lru_cache(maxsize=1)
def load_boxes() -> dict | None:
    f = boxes_file()
    return None if f is None else json.load(open(f))["samples"]


def lookup(dataset: str, sample_id: str) -> tuple[list[float] | None, list[float] | None, dict]:
    """返回 (bbox_target, bbox_reference, box_status)。box_status = {'target': 状态, 'reference': 状态或 None}。"""
    table = load_boxes()
    if table is None:
        return None, None, {"target": NO_FILE, "reference": NO_FILE}
    rec = table[dataset][sample_id]
    t, r = rec["target"], rec["reference"]
    return (t["bbox"], None if r is None else r["bbox"],
            {"target": t["status"], "reference": None if r is None else r["status"]})
