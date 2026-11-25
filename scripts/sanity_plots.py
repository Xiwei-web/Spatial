"""每个数据集固定随机抽 5 个样本，把图像、标签、（有的话）框画出来，存到 outputs/sanity/<dataset>.png 供肉眼检查。"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src import paths
from src.data import DATASETS, load_dataset

out = paths.get("outputs") / "sanity"
out.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    ss = load_dataset(name)
    pick = random.Random(f"sanity:{name}").sample(ss, 5)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    for ax, s in zip(axes, pick):
        try:
            ax.imshow(Image.open(s.image_path).convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"MISSING\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        for box, col, lab in ((s.bbox_target, "red", "target"), (s.bbox_reference, "blue", "reference")):
            if box:
                x1, y1, x2, y2 = box
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=2))
                ax.text(x1, y1, lab, color="white", fontsize=7, bbox=dict(fc=col, ec="none", pad=1))
        ax.set_title(f"{s.sample_id}\ntarget={s.target} | ref={s.reference} | rel={s.relation}\n\"{s.caption_true}\"", fontsize=7.5)
        ax.axis("off")
    fig.suptitle(f"{name}  (n={len(ss)}; red=target box, blue=reference box when available)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    plt.close(fig)
    print("wrote", out / f"{name}.png")
