"""F1 第一步（需要 1 张 GPU）：对 What's Up Subset A/B 每张图的 target、reference 各做一次
GroundingDINO 检测（物体名称来自数据集标注解析结果，不手改）+ SAM 2.1 框提示分割。
只产出原始自动标注，不做质检判断（质检在 scripts/whatsup_qc.py，纯 CPU，可反复重跑）。
输出：<data>/whatsup_annotations/<version>/raw.jsonl
用法：source scripts/env.sh tools && python scripts/annotate_whatsup.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mu

from src import paths
from src.annotate import gdino
from src.data import load_dataset


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedup(cands, thr):
    keep = []
    for c in cands:                       # 已按分数降序
        if all(box_iou(c["box_xyxy"], k["box_xyxy"]) <= thr for k in keep):
            keep.append(c)
    return keep


def main():
    cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto.yaml"))
    out_dir = paths.get("data") / "whatsup_annotations" / cfg["version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = gdino.load(cfg["detector"]["variant"], dev)
    from transformers import Sam2Model, Sam2Processor
    sam_path = paths.get("new_models") / cfg["segmenter"]["name"]
    sam_proc = Sam2Processor.from_pretrained(sam_path)
    sam = Sam2Model.from_pretrained(sam_path).to(dev).eval()
    samples = load_dataset("whatsup_a") + load_dataset("whatsup_b")
    t0 = time.time()
    with open(out_dir / "raw.jsonl", "w") as f:
        for n, s in enumerate(samples):
            image = Image.open(s.image_path).convert("RGB")
            W, H = image.size
            rec = {"dataset": s.dataset, "sample_id": s.sample_id, "relation": s.relation, "width": W, "height": H, "objects": {}}
            for role, name in (("target", s.target), ("reference", s.reference)):
                cands = dedup(gdino.detect(det, image, name, cfg["detector"]["candidate_floor"]), cfg["detector"]["dedup_iou"])
                o = {"name": name, "n_candidates": len(cands), "candidates": cands[:5], "box_xyxy": None, "score": None,
                     "mask_rle": None, "mask_area": None, "mask_centroid_xy": None, "sam_iou_pred": None}
                if cands:
                    box = cands[0]["box_xyxy"]
                    inp = sam_proc(images=image, input_boxes=[[box]], return_tensors="pt").to(dev)
                    with torch.no_grad():
                        so = sam(**inp, multimask_output=cfg["segmenter"]["multimask_output"])
                    m = sam_proc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"])[0]
                    m = np.asarray(m).reshape(-1, H, W)[0] > 0
                    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode()
                    ys, xs = np.nonzero(m)
                    o.update(box_xyxy=box, score=cands[0]["score"], mask_rle=rle, mask_area=int(m.sum()),
                             mask_centroid_xy=[float(xs.mean()), float(ys.mean())] if m.any() else None,
                             mask_bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if m.any() else None,
                             sam_iou_pred=float(so.iou_scores.flatten()[0]))
                rec["objects"][role] = o
            f.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    json.dump({"config": cfg, "detector_load_report": det._load_report, "n": len(samples), "device": dev,
               "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}, open(out_dir / "raw_meta.json", "w"), indent=1)
    print("done", len(samples), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
