"""重画 v3 待确认图（CPU，只读 proposals.json；不改 confirm.csv、不改标注）。
相对 build_whatsup_v3.py 里的初版：候选只在框角标编号，编号/分数/检测短语列在图下方的图例里，避免标签互相遮挡；标题改用英文避免缺字。
用法：python scripts/render_v3_changes.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as mu

from src import paths
from src.data import load_dataset

P = json.load(open(paths.get("data") / "whatsup_annotations" / "auto_v3" / "proposals.json"))
OUT = paths.get("outputs") / "annotation_review" / "auto_v3_changes"
samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
COL = ["#ff0000", "#ff7f00", "#c0007f", "#00a000", "#8b00ff", "#00a0a0", "#804000", "#606060", "#0044ff", "#00a0ff"]
for k in sorted(P, key=int):
    p = P[k]
    image = np.asarray(Image.open(samples[(p["dataset"], p["sample_id"])].image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.6))
    axes[0].imshow(image)
    if p["case"] != "swapped_boxes":
        rb, cb = p["reference_box"], p["current_box"]
        axes[0].add_patch(Rectangle(rb[:2], rb[2] - rb[0], rb[3] - rb[1], fill=False, ec="#0044ff", lw=1.0))
        axes[0].add_patch(Rectangle(cb[:2], cb[2] - cb[0], cb[3] - cb[1], fill=False, ec="white", lw=1.5, ls=":"))
    corners = [("left", "bottom", 0, 1), ("right", "bottom", 2, 1), ("left", "top", 0, 3), ("right", "top", 2, 3)]
    legend = []
    for i, c in enumerate(p["candidates"]):
        b, col = c["box_xyxy"], COL[i % len(COL)]
        axes[0].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2.2))
        ha, va, xi, yi = corners[i % 4]
        axes[0].text(b[xi], b[yi], c["label"], color="white", fontsize=10, ha=ha, va=va, fontweight="bold", bbox=dict(fc=col, ec="none", pad=1.0))
        legend.append((col, f"{c['label']}  score {c['score']:.2f}  [{c['prompt']}]  {int(b[2] - b[0])}x{int(b[3] - b[1])} px"))
    for i, (col, txt) in enumerate(legend):
        fig.text(0.02 + 0.33 * (i % 3), 0.055 - 0.028 * (i // 3), txt, color=col, fontsize=10, fontweight="bold")
    axes[0].set_title(f"{len(p['candidates'])} candidates (legend below). thin blue = reference box; white dotted = current target box (rejected in review)", fontsize=10)
    vis = image.astype(np.float32) / 255
    ttl = "NO PROPOSAL: needs a point prompt from the reviewer"
    if p.get("proposed_segmentation"):
        m = mu.decode(p["proposed_segmentation"]["mask_rle"]).astype(bool)
        vis[m] = 0.5 * vis[m] + 0.5 * np.array([1, 0, 0])
        ttl = f"proposed pick: {p['proposed']}   (its SAM mask in red)"
    elif p["proposed"]:
        ttl = f"proposed pick: {p['proposed']}   (masks will be re-segmented after confirmation)"
    axes[1].imshow(vis)
    if p["proposed"]:
        for role, lab in p["proposed"].items():
            c = next(x for x in p["candidates"] if x["label"] == lab)
            b, col = c["box_xyxy"], ("red" if role == "target" else "#0044ff")
            axes[1].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2.5))
            axes[1].text(b[0], b[1] - 4, f"{role}: {lab}", color="white", fontsize=10, va="bottom", bbox=dict(fc=col, ec="none", pad=1.2))
    axes[1].set_title(ttl, fontsize=10)
    for a in axes:
        a.axis("off")
    fig.suptitle(f"order {p['order']}   {p['dataset']}:{p['sample_id']}   case={p['case']}   object='{p['name']}'   "
                 f"[auto annotation with model-assisted pre-review and partial human check; v3 PENDING your confirmation]", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.09, wspace=0.02)
    fig.savefig(OUT / f"chg_{p['order']:03d}_{p['dataset'][-1].upper()}_{p['sample_id']}.png", dpi=70)
    plt.close(fig)
print("re-rendered", len(P))
