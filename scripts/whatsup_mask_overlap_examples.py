"""导出 v3 中 reference mask 覆盖 target 的示例叠加图（CPU，只读标注）。
用法：python scripts/whatsup_mask_overlap_examples.py <sample_id> [sample_id ...]     （Subset A）
每个样本一张图：原图 | 叠加（整图）| 原图局部 | 叠加局部。红=仅 target，蓝=仅 reference，黄=交集；白线=target 轮廓，青线=reference 轮廓。
输出：<outputs>/annotation_review/auto_v3_changes/mask_overlap_examples/<sample_id>.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from pycocotools import mask as mu

from src import paths
from src.data import load_dataset

want = sys.argv[1:]
OUT = paths.get("outputs") / "annotation_review" / "auto_v3_changes" / "mask_overlap_examples"
OUT.mkdir(parents=True, exist_ok=True)
recs = {r["sample_id"]: r for r in map(json.loads, open(paths.get("data") / "whatsup_annotations" / "auto_v3" / "annotations.jsonl")) if r["dataset"] == "whatsup_a"}
img = {s.sample_id: s.image_path for s in load_dataset("whatsup_a")}
for sid in want:
    r = recs[sid]
    T, R = r["objects"]["target"], r["objects"]["reference"]
    t, q = (mu.decode(o["mask_rle"]).astype(bool) for o in (T, R))
    im = np.asarray(Image.open(img[sid]).convert("RGB"))
    tint = np.zeros((*t.shape, 4))
    tint[t & ~q], tint[q & ~t], tint[t & q] = (1, 0, 0, .55), (0, .35, 1, .40), (1, .9, 0, .60)
    fig, ax = plt.subplots(2, 2, figsize=(20, 15.6))
    bx = T["box_xyxy"]
    pad = max(60, 0.6 * max(bx[2] - bx[0], bx[3] - bx[1]))
    for row in (0, 1):
        for col in (0, 1):
            a = ax[row, col]
            a.imshow(im)
            if col == 1:
                a.imshow(tint)
                a.contour(q.astype(float), levels=[.5], colors="cyan", linewidths=1.2)
                a.contour(t.astype(float), levels=[.5], colors="white", linewidths=1.6)
                for o, c in ((R, "#0044ff"), (T, "red")):
                    b = o["box_xyxy"]
                    a.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=1.6, ls="--"))
            if row == 1:
                a.set_xlim(max(0, bx[0] - pad), min(im.shape[1], bx[2] + pad)); a.set_ylim(min(im.shape[0], bx[3] + pad), max(0, bx[1] - pad))
            a.set_title(["full image", "zoom on target"][row] + ["", " + masks"][col], fontsize=12)
            a.axis("off")
    i = int((t & q).sum())
    fig.legend(handles=[Patch(color="red", alpha=.55, label=f"target only ({int((t & ~q).sum())} px)"), Patch(color=(0, .35, 1), alpha=.5, label=f"reference only ({int((q & ~t).sum())} px)"),
                        Patch(color=(1, .9, 0), label=f"intersection ({i} px)"), Patch(fc="gray", ec="white", lw=2, label="white line: target outline"),
                        Patch(fc="gray", ec="cyan", lw=2, label="cyan line: reference outline"), Patch(fc="none", ec="k", ls="--", label="dashed: boxes")],
               loc="lower center", ncol=6, fontsize=12, facecolor="lightgray")
    fig.suptitle(f"order {r['review_order_v2']}  {sid}   target='{T['name']}'  reference='{R['name']}'   intersection = {i / t.sum():.1%} of target, {i / q.sum():.1%} of reference\n"
                 f"[auto annotation v3, read-only]", fontsize=14)
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(OUT / f"{sid}.png", dpi=70)
    plt.close(fig)
    print(OUT / f"{sid}.png", f"target {t.sum()} reference {q.sum()} inter {i}")
