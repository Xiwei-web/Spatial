"""SAM 2.1 框提示分割的薄封装（与 scripts/annotate_whatsup.py 中的调用方式相同）。"""
from __future__ import annotations

import numpy as np
import torch
from pycocotools import mask as mu

from src import paths


def load(name: str = "sam2.1-hiera-large", device: str = "cuda"):
    from transformers import Sam2Model, Sam2Processor
    p = paths.get("new_models") / name
    return Sam2Processor.from_pretrained(p), Sam2Model.from_pretrained(p).to(device).eval()


@torch.no_grad()
def segment(proc, model, image, box, multimask_output: bool = False) -> dict:
    """返回与标注文件相同的字段：mask_rle / mask_area / mask_centroid_xy / mask_bbox_xyxy / sam_iou_pred。"""
    W, H = image.size
    dev = next(model.parameters()).device
    inp = proc(images=image, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(dev)
    so = model(**inp, multimask_output=multimask_output)
    m = proc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"])[0]
    m = np.asarray(m).reshape(-1, H, W)[0] > 0
    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    ys, xs = np.nonzero(m)
    return {"mask_rle": rle, "mask_area": int(m.sum()),
            "mask_centroid_xy": [float(xs.mean()), float(ys.mean())] if m.any() else None,
            "mask_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if m.any() else None,
            "sam_iou_pred": float(so.iou_scores.flatten()[0])}
