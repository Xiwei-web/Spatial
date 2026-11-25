"""为待审图像生成“候选框选择图”和空的 corrections.csv（CPU，只读使用已保存的候选，不改动标注目录）。
每张图：target 的前 5 个候选（红色系，编号 T1..T5）和 reference 的前 5 个候选（蓝色系，编号 R1..R5），标分数；
当前 v2 选中的候选用粗实线，其余用细虚线。编号从 1 开始，按检测分数降序。
用法：python scripts/whatsup_candidate_sheets.py [--config whatsup_auto_v2.yaml]
输出：<outputs>/annotation_review/<version>_corrections/{cand_XXX_<数据集>_<样本>.png, corrections.csv, README.txt}"""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from src import paths
from src.data import load_dataset

cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "whatsup_auto_v2.yaml"
cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / cfg_name))
ann = paths.get("data") / "whatsup_annotations" / cfg["version"] / "annotations.jsonl"
rows = [json.loads(l) for l in open(ann)]
# 与 review.csv 相同的排序规则（待审在前、标记多的更靠前），只取待审
order = sorted(rows, key=lambda r: (r["auto_qc"] != "review", -len(r["flags"]), r["dataset"], r["sample_id"]))
todo = [(i + 1, r) for i, r in enumerate(order) if r["auto_qc"] == "review"]
samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
out = paths.get("outputs") / "annotation_review" / f"{cfg['version']}_corrections"
out.mkdir(parents=True, exist_ok=True)
corr = out / "corrections.csv"
if corr.exists():
    print("[!] corrections.csv 已存在，不覆盖；只重画图像")
else:
    with open(corr, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "target_pick", "reference_pick", "note"])
        for i, r in todo:
            w.writerow([f"{r['dataset']}:{r['sample_id']}", "", "", ""])
(out / "README.txt").write_text(
    "corrections.csv 填写说明\n"
    "image_id = <数据集>:<样本 ID>；行的顺序与 review.csv 中待审图像的顺序相同，对应图像文件 cand_<review.csv 的 order>_....png\n"
    "target_pick / reference_pick：填图上的候选编号（1-5，按检测分数降序；T=target 红色，R=reference 蓝色）；都不对填 none；\n"
    "  当前选择已经正确可以留空或填当前编号（图标题里的 current=T?/R? 给出当前选中的编号）。\n"
    "粗实线 = v2 当前选中的候选；细虚线 = 其它候选。只保存了每个物体分数最高的 5 个候选（IoU>0.8 的重复候选已合并）。\n")
T_COL = ["#ff0000", "#ff7f00", "#d4a000", "#c0007f", "#8b0000"]
R_COL = ["#0044ff", "#00a0ff", "#00b894", "#6a3df0", "#003070"]
for i, r in todo:
    s = samples[(r["dataset"], r["sample_id"])]
    im = Image.open(s.image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(12.8, 9.9))
    ax.imshow(im)
    cur = {}
    for role, cols, tag in (("reference", R_COL, "R"), ("target", T_COL, "T")):
        o = r["objects"][role]
        sel = o.get("selected_candidate_rank") or 0
        cur[tag] = sel + 1
        for k, c in enumerate(o["candidates"][:5]):
            x1, y1, x2, y2 = c["box_xyxy"]
            chosen = k == sel
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cols[k], lw=3.2 if chosen else 1.6, ls="-" if chosen else "--"))
            # 标签错开放置，避免互相遮挡：target 标在框左上，reference 标在框右下
            tx, ty, ha, va = (x1, y1 - 4, "left", "bottom") if tag == "T" else (x2, y2 + 4, "right", "top")
            ax.text(tx, ty + (k * 0 if chosen else 0), f"{tag}{k + 1} {c['score']:.2f}", color="white", fontsize=11, ha=ha, va=va,
                    fontweight="bold" if chosen else "normal", bbox=dict(fc=cols[k], ec="none", pad=1.5))
    ax.set_title(f"#{i}  {r['dataset']}:{r['sample_id']}   relation={r['relation']}\n"
                 f"target='{r['objects']['target']['name']}' (red, current=T{cur['T']})   reference='{r['objects']['reference']['name']}' (blue, current=R{cur['R']})"
                 f"   flags: {','.join(r['flags'])}", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"cand_{i:03d}_{r['dataset'][-1].upper()}_{r['sample_id']}.png", dpi=75)
    plt.close(fig)
print("wrote", len(todo), "candidate images and", corr)
