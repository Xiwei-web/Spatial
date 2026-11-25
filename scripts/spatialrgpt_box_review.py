"""SpatialRGPT：自带框与 mask 外接框 IoU 最低的 20 个物体的联系表（CPU）。两份：深度样本（实际会用的 96 条 = 192 个物体）和全集 2,546 个区域。
每格：物体周围的裁剪；半透明红 + 黄轮廓 = mask，红实线 = mask 外接框（真值），青虚线 = 自带框。输出 <outputs>/data_review/spatialrgpt_box/"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as mu

from src import paths
from src.annotate.masks import rle_bbox
from src.annotate.selection import box_iou
from src.data.spatialrgpt import depth_samples, load_records

OUT = paths.get("outputs") / "data_review" / "spatialrgpt_box"
OUT.mkdir(parents=True, exist_ok=True)
used = {s.sample_id for s in depth_samples()}
objs = []
for r in load_records():
    for k, (b, rle) in enumerate(zip(r.bboxes, r.rles)):
        g = rle_bbox(rle)
        objs.append({"iou": box_iou(b, g), "id": r.id, "k": k, "cls": r.classes[k] if k < len(r.classes) else "?", "cat": r.category, "path": r.image_path, "prov": b, "gt": g, "rle": rle, "src": r.image_source, "depth": r.id in used})
for tag, sub in (("depth_used", [o for o in objs if o["depth"]]), ("all_regions", objs)):
    low = sorted(sub, key=lambda o: o["iou"])[:20]
    json.dump([{k: v for k, v in o.items() if k != "rle"} for o in low], open(OUT / f"lowest20_{tag}.json", "w"), indent=1)
    fig, axes = plt.subplots(4, 5, figsize=(26, 22))
    for ax, o in zip(axes.flat, low):
        im = np.asarray(Image.open(o["path"]).convert("RGB"))
        m = mu.decode(o["rle"]).astype(bool)
        x1, y1 = min(o["prov"][0], o["gt"][0]), min(o["prov"][1], o["gt"][1])
        x2, y2 = max(o["prov"][2], o["gt"][2]), max(o["prov"][3], o["gt"][3])
        p = 0.35 * max(x2 - x1, y2 - y1) + 15
        ax.imshow(im)
        t = np.zeros((*m.shape, 4)); t[m] = (1, 0, 0, .35)
        ax.imshow(t)
        ax.contour(m.astype(float), levels=[.5], colors="yellow", linewidths=1.3)
        for b, c, ls in ((o["gt"], "red", "-"), (o["prov"], "cyan", "--")):
            ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=2.2, ls=ls))
        ax.set_xlim(max(0, x1 - p), min(im.shape[1], x2 + p)); ax.set_ylim(min(im.shape[0], y2 + p), max(0, y1 - p))
        ax.set_title(f"IoU {o['iou']:.2f}  '{o['cls']}'  Region[{o['k']}]\n{o['id']}  {o['cat']}  {o['src']}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"SpatialRGPT-Bench, 20 lowest IoU ({tag}, {len(sub)} objects): red = mask bbox (truth), cyan dashed = provided box, yellow/red = mask", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(OUT / f"lowest20_{tag}.jpg", dpi=60)
    plt.close(fig)
    print(tag, len(sub), [round(o["iou"], 2) for o in low], [o["cls"] for o in low])
