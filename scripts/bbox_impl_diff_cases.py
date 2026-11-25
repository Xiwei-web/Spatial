"""列出“sdpa 参照 vs 修复后 eager”框预测坐标最大差 >= 50 的样本，供用户判断样本本身是否有歧义（CPU；不算正确率）。
输入：<outputs>/smoke/<模型>/default_vs_eager.json；参照框取 What's Up 标注 auto_v3 的检测框。
筛选口径与最初汇报的表相同：两边都能解析出四个数、原始输出数字的最大绝对差 >= 50。
输出：<outputs>/diagnostics/bbox_impl_diff/cases.json、docs/parts/figs/bbox_impl_diff/ 下每例一张图，并刷新 docs/parts/bbox_impl_diff.md 里 CASES 标记之间的表。用法：python scripts/bbox_impl_diff_cases.py"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.prompts import parse_four_numbers, to_original_xyxy

THRESH = 50
G1 = ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "InternVL3-8B", "InternVL3_5-8B", "SenseNova-SI-1.2-InternVL3-8B",
      "SenseNova-SI-1.3-InternVL3-8B", "GLM-4.6V-Flash", "VST-7B-RL", "RoboBrain2.0-7B"]
# 无官方坐标约定的模型：只为画图列出的候选假设，不代表结论（坐标约定标定是后续阶段的事）
HYP = {"SenseNova-SI-1.2-InternVL3-8B": ["norm1_xyxy"], "SenseNova-SI-1.3-InternVL3-8B": ["norm1000_xyxy"],
       "VST-7B-RL": ["abs_model_xyxy", "norm1000_xyxy"]}
OUT = paths.get("outputs") / "diagnostics" / "bbox_impl_diff"
OUT.mkdir(parents=True, exist_ok=True)
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
ann = {(d["dataset"], d["sample_id"]): d for d in map(json.loads, open(paths.get("data") / "whatsup_annotations" / "auto_v3" / "annotations.jsonl"))}
rev = {(r["dataset"], r["sample_id"]): r for r in csv.DictReader(open(paths.REPO_ROOT / "annotations" / "whatsup" / "review_final_v2.csv", encoding="utf-8-sig"))}


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(0, w) * max(0, h)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


cases = []
for m in G1:
    d = json.load(open(paths.get("outputs") / "smoke" / m / "default_vs_eager.json"))["runs"]
    for k, (e, r) in enumerate(zip(d["observed"], d["reference"])):
        if e["task"] != "bbox" or e["tokens"] == r["tokens"]:
            continue
        ne, nr = parse_four_numbers(e["text"]), parse_four_numbers(r["text"])
        if not (ne and nr) or max(abs(x - y) for x, y in zip(ne, nr)) < THRESH:
            continue
        it, role = items[k // 3], ("target", "reference")[k % 3]
        assert it["sample_id"] == e["sample_id"]
        key = (it["dataset"], it["sample_id"])
        s, a, bc = samples[key], ann[key], bbox_convention(m)
        first = next(i for i, (x, y) in enumerate(zip(e["tokens"], r["tokens"])) if x != y)
        convs = [bc["convention"]] if bc["convention"] != "unknown" else HYP[m]
        mwh = next((tuple(x["model_wh"]) for x in map(json.loads, open(paths.get("outputs") / "smoke" / m / "records.jsonl"))
                    if x["sample_id"] == it["sample_id"] and x.get("model_wh")), None)
        boxes = {c: {"eager": to_original_xyxy(ne, c, (a["width"], a["height"]), mwh),
                     "sdpa": to_original_xyxy(nr, c, (a["width"], a["height"]), mwh)} for c in convs}
        o = a["objects"][role]
        cases.append({"model": m, "dataset": it["dataset"], "sample_id": it["sample_id"], "image": str(s.image_path), "role": role,
                      "object": o["name"], "prompt": bc["prompt_template"].format(object=getattr(s, role)),
                      "convention_official": bc["convention"] != "unknown", "eager_text": e["text"], "sdpa_text": r["text"],
                      "eager_nums": ne, "sdpa_nums": nr, "max_coord_diff": max(abs(x - y) for x, y in zip(ne, nr)),
                      "first_diff_token_index": first, "boxes_original_px": boxes,
                      "iou_eager_vs_sdpa": {c: iou(b["eager"], b["sdpa"]) for c, b in boxes.items()},
                      "annotation_v3": {"usable": a["usable"], "box_det": o["box_det_xyxy"], "box_mask": o["box_mask_xyxy"], "score": o["score"],
                                        "other_object": {x["name"]: x["box_det_xyxy"] for n, x in a["objects"].items() if n != role}},
                      "review": {x: rev[key][x] for x in ("order", "final_verdict", "review_source")}})
        c = cases[-1]
        fig, axes = plt.subplots(1, len(convs), figsize=(11 * len(convs), 9), squeeze=False)
        for ax, cv in zip(axes[0], convs):
            ax.imshow(Image.open(s.image_path).convert("RGB"))
            for b, col, ls, lab in ((o["box_det_xyxy"], "lime", "-", "annotation auto_v3"), (boxes[cv]["eager"], "red", "-", "eager (fp32 scoring)"),
                                    (boxes[cv]["sdpa"], "blue", "--", "sdpa")):
                ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2.5, ls=ls, label=lab))
            ax.set_xlim(0, a["width"]); ax.set_ylim(a["height"], 0)
            ax.legend(loc="upper left"); ax.axis("off")
            ax.set_title(f"{m}  '{o['name']}' ({role})  {it['sample_id']}\n{cv} ({'official' if c['convention_official'] else 'HYPOTHESIS, no official convention'})\n"
                         f"eager {ne}\nsdpa {nr}", fontsize=10)
        fig.tight_layout()
        c["figure"] = f"figs/bbox_impl_diff/case{len(cases)}_{m}_{it['sample_id']}_{role}.jpg"      # 相对 docs/parts
        fp = paths.REPO_ROOT / "docs" / "parts" / c["figure"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=60, bbox_inches="tight"); plt.close(fig)
json.dump(cases, open(OUT / "cases.json", "w"), indent=1, ensure_ascii=False)
for c in cases:
    print(c["model"], c["dataset"], c["sample_id"], c["role"], c["eager_nums"], c["sdpa_nums"], "tok", c["first_diff_token_index"],
          {k: round(v, 2) for k, v in c["iou_eager_vs_sdpa"].items()}, {k: {n: [round(x) for x in b] for n, b in v.items()} for k, v in c["boxes_original_px"].items()},
          "ann", [round(x) for x in c["annotation_v3"]["box_det"]], c["review"], "\n   ", c["figure"])

CONV = {"norm1000_xyxy": "0–1000 归一化", "norm1_xyxy": "0–1 归一化", "abs_model_xyxy": "模型图像像素"}
ri = lambda b: "[" + ", ".join(str(round(v)) for v in b) + "]"
rn = lambda b: "[" + ", ".join(f"{v:g}" for v in b) + "]"
rows = ["| # | 模型 | 图像 | 真值物体（角色） | eager 原始输出 | sdpa 原始输出 | 最大差 | 换算约定 | eager 框（原图像素） | sdpa 框（原图像素） | 两框 IoU | 标注框 v3（原图像素） |", "|" + "---|" * 12]
figs = []
for i, c in enumerate(cases, 1):
    for j, (cv, b) in enumerate(c["boxes_original_px"].items()):
        bad = max(map(abs, b["eager"])) > 5000
        rows.append("| " + " | ".join([str(i) if not j else "", c["model"] if not j else "", f"{c['dataset']} / {c['sample_id']}" if not j else "", f"{c['object']}（{c['role']}）" if not j else "",
                    f"`{rn(c['eager_nums'])}`" if not j else "", f"`{rn(c['sdpa_nums'])}`" if not j else "", f"{c['max_coord_diff']:g}" if not j else "",
                    CONV[cv] + ("（官方）" if c["convention_official"] else "（假设，无官方约定）"), "无效（第二个数超出图像）" if bad else ri(b["eager"]), ri(b["sdpa"]),
                    "—" if bad else f"{c['iou_eager_vs_sdpa'][cv]:.2f}", ri(c["annotation_v3"]["box_det"])]) + " |")
    figs.append(f"**#{i} {c['model']}：{c['sample_id']} / {c['object']}**\n\n![case{i}]({c['figure']})\n")
doc = paths.REPO_ROOT / "docs" / "parts" / "bbox_impl_diff.md"
t = doc.read_text()
a, b = "<!-- CASES_BEGIN -->", "<!-- CASES_END -->"
assert t.count(a) == 1 and t.count(b) == 1
doc.write_text(t[: t.index(a) + len(a)] + "\n" + "\n".join(rows) + "\n\n图例：绿 = 标注 auto_v3 检测框，红 = eager（float32 打分，bf16），蓝虚线 = sdpa（bf16）。第 5 条两种换算假设各一幅。\n\n" + "\n".join(figs) + t[t.index(b):])
