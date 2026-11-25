"""mask 之间的运算（纯函数，CPU；RLE 为 COCO 格式、counts 为 str）。"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mu


def intersection_area(rle_a, rle_b) -> int:
    return int(mu.area(mu.merge([rle_a, rle_b], intersect=True)))


def mask_fields(m: np.ndarray) -> dict:
    """由布尔 mask 得到标注里的派生字段。"""
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    ys, xs = np.nonzero(m)
    ok = len(xs) > 0
    return {"mask_rle": r, "mask_area": int(m.sum()), "mask_centroid_xy": [float(xs.mean()), float(ys.mean())] if ok else None,
            "mask_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if ok else None}


def subtract_target_from_reference(rec: dict) -> dict | None:
    """口径：每个 mask 只包含该物体可见的像素；target 放在或挡在 reference 上时，reference 的 mask 不含被 target 覆盖的部分。
    有交集就减（所有关系类型）。只改 reference 的 mask_rle / mask_area / mask_centroid_xy；target 的 mask 与两者的框都不变
    （reference 的 box_mask_xyxy / mask_bbox_xyxy 也保持原 mask 的外接框）。原 mask 存到 reference 的 mask_reference_raw。
    无交集返回 None，否则返回统计。"""
    T, R = rec["objects"]["target"], rec["objects"]["reference"]
    inter = intersection_area(T["mask_rle"], R["mask_rle"])
    if inter == 0:
        return None
    raw = {k: R[k] for k in ("mask_rle", "mask_area", "mask_centroid_xy", "mask_bbox_xyxy")}
    new = mask_fields(mu.decode(R["mask_rle"]).astype(bool) & ~mu.decode(T["mask_rle"]).astype(bool))
    R["mask_reference_raw"] = raw
    R.update({k: new[k] for k in ("mask_rle", "mask_area", "mask_centroid_xy")})
    return {"inter": inter, "raw_area": raw["mask_area"], "new_area": new["mask_area"], "remaining": new["mask_area"] / raw["mask_area"],
            "inter_over_target": inter / T["mask_area"], "bbox_would_change": new["mask_bbox_xyxy"] != raw["mask_bbox_xyxy"]}


def rle_bbox(rle) -> list[float]:
    """mask 的外接框 [x1, y1, x2, y2]（右下为开区间，与 mask_fields 的 mask_bbox_xyxy 同一约定）。"""
    x, y, w, h = (float(v) for v in mu.toBbox(rle))
    return [x, y, x + w, y + h]


def coco_segmentation_bbox(segmentation, height: int, width: int) -> list[float]:
    """COCO segmentation（polygon 列表 / 未压缩 RLE / 压缩 RLE）的 mask 外接框。"""
    if isinstance(segmentation, list):
        rle = mu.merge(mu.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mu.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return rle_bbox(rle)


def add_gt_box(rec: dict) -> None:
    """默认真值框（用户 2026-11-25）：原始 mask（减去 target 之前）的外接框，存为 box_gt_xyxy。检测框 box_det_xyxy 保留作对照。"""
    for o in rec["objects"].values():
        o["box_gt_xyxy"] = rle_bbox((o.get("mask_reference_raw") or o)["mask_rle"])
