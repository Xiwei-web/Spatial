"""COCO-Pairs：从 COCO val2017 实例标注里构造的“物体对”数据（用于在拥挤 / 贴近 / 重叠场景下检验结论）。

本文件包含三部分：
1. 纯几何函数（关系标签、归一化间距、包含度、投影重叠等），不依赖任何数据文件，可单测；
2. COCO 标注索引（按 ann_id / image_id）；
3. 加载器 `load_coco_pairs(split=None)`，返回与 src/data/loaders.py 相同的 `Sample`。

数据文件 pairs.jsonl 由 scripts/build_coco_pairs.py 生成，放在 <data>/coco_pairs/ 下（不进仓库）；
划分文件 splits/coco_pairs.json 只存图像 id 列表（按图像划分，同一张图的所有 pair 在同一侧）。

坐标约定：像素 (列 i, 行 j) 覆盖连续区域 [i, i+1) × [j, j+1)，与 COCO bbox 的连续坐标一致；
因此 mask 质心 = 像素下标均值 + 0.5。图像 y 轴向下。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from src import paths

from .loaders import Sample

DATASET = "coco_pairs"
SEED = 20261125

# ---- 候选物体的筛选阈值（闭区间）----
MIN_AREA_FRAC = 0.01     # mask 面积 / 图像面积 下限
MAX_AREA_FRAC = 0.40     # 上限
MIN_SHORT_SIDE = 300     # 图像短边（像素）下限

# ---- 报告用的分箱：(标签, 下界含, 上界不含) ----
GAP_BINS = [("<0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0), ("1-1.5", 1.0, 1.5), ("1.5-2", 1.5, 2.0),
            ("2-3", 2.0, 3.0), ("3-5", 3.0, 5.0), (">=5", 5.0, math.inf)]
CROWD_BINS = [("0", 0, 1), ("1-2", 1, 3), ("3-5", 3, 6), ("6-10", 6, 11), ("11-20", 11, 21), (">20", 21, math.inf)]

RELATIONS = ["left", "right", "above", "below"]
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
AXIS_OF = {"left": "horizontal", "right": "horizontal", "above": "vertical", "below": "vertical"}


def bin_label(value: float, bins) -> str:
    for label, lo, hi in bins:
        if lo <= value < hi:
            return label
    raise ValueError(f"{value} 不在任何分箱里")


# =====================================================================
# 几何函数（bbox 一律为 [x1, y1, x2, y2]，原图像素，连续坐标）
# =====================================================================
def xywh_to_xyxy(b) -> list[float]:
    x, y, w, h = b
    return [float(x), float(y), float(x + w), float(y + h)]


def mask_centroid(mask) -> tuple[float, float]:
    """二值 mask（H×W）的质心 (cx, cy)，像素中心约定（下标均值 + 0.5）。空 mask 报错。"""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("空 mask，无法计算质心")
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


def relation_axis(c_target, c_reference) -> str:
    """|dx| > |dy| → horizontal；否则（含 |dx| == |dy| 的平局）→ vertical。"""
    dx = c_target[0] - c_reference[0]
    dy = c_target[1] - c_reference[1]
    return "horizontal" if abs(dx) > abs(dy) else "vertical"


def relation_label(c_target, c_reference, target_first_on_tie: bool = True) -> str:
    """由两个质心给出 target 相对 reference 的关系标签。

    - |dx| > |dy|：左右。target 质心 x 更小 → "left"，否则 "right"。
    - 否则：上下（图像 y 向下）。target 质心 y 更小（更高）→ "above"，否则 "below"。
    - 平局 1：|dx| == |dy| 归为上下（题面规定的 else 分支）。
    - 平局 2：两个质心完全重合（dx == dy == 0）时方向无定义；为了保证互为 twin 的两条记录标签相反，
      由调用方传入 target_first_on_tie（构建脚本里 = target 的 ann_id 较小）：True → "above"，False → "below"。
    """
    dx = c_target[0] - c_reference[0]
    dy = c_target[1] - c_reference[1]
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    if dy < 0:
        return "above"
    if dy > 0:
        return "below"
    return "above" if target_first_on_tie else "below"


def axis_extent(bbox, axis: str) -> float:
    """bbox 沿关系轴的长度：horizontal → 宽，vertical → 高。"""
    return float(bbox[2] - bbox[0]) if axis == "horizontal" else float(bbox[3] - bbox[1])


def centroid_distance(c_a, c_b) -> float:
    return math.hypot(c_a[0] - c_b[0], c_a[1] - c_b[1])


def normalized_gap(c_target, c_reference, bbox_target, bbox_reference, axis: str) -> float:
    """归一化间距 = 质心欧氏距离 / 两个物体沿关系轴的平均长度（水平轴用 bbox 宽，竖直轴用 bbox 高）。"""
    mean_ext = 0.5 * (axis_extent(bbox_target, axis) + axis_extent(bbox_reference, axis))
    return centroid_distance(c_target, c_reference) / mean_ext


def normalized_gap_axis(c_target, c_reference, bbox_target, bbox_reference, axis: str) -> float:
    """同上，但分子只取质心差在关系轴上的分量（|dx| 或 |dy|）。作为补充量一并记录。"""
    k = 0 if axis == "horizontal" else 1
    mean_ext = 0.5 * (axis_extent(bbox_target, axis) + axis_extent(bbox_reference, axis))
    return abs(c_target[k] - c_reference[k]) / mean_ext


def bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def bbox_iou(a, b) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """a 的 bbox 面积中落在 b 的 bbox 内的比例（0~1）。不对称：containment(a, b) != containment(b, a)。"""
    area = bbox_area(a)
    return bbox_intersection(a, b) / area if area > 0 else 0.0


def projection_overlap_len(a, b, axis: str) -> float:
    """两个 bbox 在关系轴上的投影区间的重叠长度（不重叠为 0）。"""
    k = 0 if axis == "horizontal" else 1
    return max(0.0, min(a[k + 2], b[k + 2]) - max(a[k], b[k]))


def projection_overlap(a, b, axis: str) -> bool:
    """两个 bbox 在关系轴上的投影区间是否重叠（重叠长度严格 > 0；只是端点相接不算）。
    重叠意味着“谁在左/上”本身就有歧义。"""
    return projection_overlap_len(a, b, axis) > 0


def projection_overlap_frac(a, b, axis: str) -> float:
    """投影重叠长度 / 两者中较短的那个投影长度（1.0 表示一个区间完全落在另一个里面）。"""
    m = min(axis_extent(a, axis), axis_extent(b, axis))
    return projection_overlap_len(a, b, axis) / m if m > 0 else 0.0


def make_pair_id(image_id: int, target_ann: int, reference_ann: int) -> str:
    return f"{int(image_id):012d}_{int(target_ann)}_{int(reference_ann)}"


def _article(noun: str) -> str:
    return "an" if noun[0].lower() in "aeiou" else "a"


def make_caption(target: str, reference: str, relation: str) -> str:
    """与 COCO-spatial 两物体子集同一模板（src/data/loaders.py 的 _TWO 能解析）。"""
    t, r = f"{_article(target)} {target}", f"{_article(reference)} {reference}"
    if relation in ("left", "right"):
        return f"A photo of {t} to the {relation} of {r}"
    return f"A photo of {t} {relation} {r}"


# =====================================================================
# COCO 标注索引
# =====================================================================
def coco_ann_file() -> Path:
    return paths.get("data") / "coco" / "annotations" / "instances_val2017.json"


def coco_image_path(file_name: str) -> Path:
    return paths.get("data") / "coco" / "val2017" / file_name


@lru_cache(maxsize=1)
def load_coco_instances() -> dict:
    """返回 {"images": {id: img}, "cats": {id: name}, "anns": {ann_id: ann}, "by_image": {image_id: [ann, ...]}}。"""
    d = json.load(open(coco_ann_file()))
    images = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns, by_image = {}, {i: [] for i in images}
    for a in d["annotations"]:
        anns[a["id"]] = a
        by_image[a["image_id"]].append(a)
    return {"images": images, "cats": cats, "anns": anns, "by_image": by_image}


# =====================================================================
# 加载器
# =====================================================================
def pairs_dir() -> Path:
    return paths.get("data") / "coco_pairs"


def pairs_file() -> Path:
    return pairs_dir() / "pairs.jsonl"


def split_file() -> Path:
    return paths.REPO_ROOT / "splits" / f"{DATASET}.json"


def read_pair_records(f: Path | None = None) -> list[dict]:
    f = pairs_file() if f is None else Path(f)
    with open(f) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# 不放进 Sample.extra 的字段（已经有对应的 Sample 字段）
_NOT_EXTRA = {"pair_id", "file_name", "relation", "target", "reference"}


def record_to_sample(r: dict, anns: dict | None = None) -> Sample:
    """一条 pairs.jsonl 记录 → Sample。mask 按 ann_id 从 COCO 标注里取（polygon），不在 jsonl 里重复存。"""
    t, ref = r["target"], r["reference"]
    extra = {k: v for k, v in r.items() if k not in _NOT_EXTRA}
    for side, o in (("target", t), ("reference", ref)):
        extra[f"{side}_ann_id"] = o["ann_id"]
        extra[f"{side}_area"] = o["area"]
        extra[f"{side}_area_decoded"] = o["area_decoded"]
        extra[f"{side}_area_frac"] = o["area_frac"]
        extra[f"{side}_centroid"] = o["centroid"]
    cap_true = make_caption(t["category"], ref["category"], r["relation"])
    cap_false = make_caption(t["category"], ref["category"], OPPOSITE[r["relation"]])
    return Sample(
        dataset=DATASET, sample_id=r["pair_id"], image_path=str(coco_image_path(r["file_name"])),
        target=t["category"], reference=ref["category"], relation=r["relation"],
        caption_true=cap_true, caption_options=[cap_true, cap_false],
        bbox_target=list(t["bbox"]), bbox_reference=list(ref["bbox"]),
        mask_target=None if anns is None else anns[t["ann_id"]]["segmentation"],
        mask_reference=None if anns is None else anns[ref["ann_id"]]["segmentation"],
        extra=extra,
    )


def load_coco_pairs(split: str | None = None) -> list[Sample]:
    """split ∈ {None, "dev", "test"}。None 返回全部有向 pair；dev/test 按 splits/coco_pairs.json 的图像 id 过滤。
    选择类操作只用 dev，报告数字只用 test。caption_true / caption_options 是按模板生成的（第一个为真，第二个为相反关系）。"""
    assert split in (None, "dev", "test")
    records = read_pair_records()
    if split is not None:
        keep = set(json.load(open(split_file()))[split])
        records = [r for r in records if r["image_id"] in keep]
    anns = load_coco_instances()["anns"]
    return [record_to_sample(r, anns) for r in records]
