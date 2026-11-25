"""F1 第二步（CPU）：对自动标注做质检、出统计、生成人工审核材料。可反复重跑，不改 raw.jsonl。
输出：
  <data>/whatsup_annotations/<version>/annotations.jsonl   每张图：两个物体的框、mask(RLE)、分数、候选数 + 质检标记
  manifests/whatsup_auto_qc_summary.json                    质检统计（进 Git）
  <outputs>/annotation_review/sheet_XXX.png + review.csv   联系表（每页 20 张，待审的排最前）与审核表
用法：python scripts/whatsup_qc.py [--no-sheets]
注意：这是“自动标注”。在 review.csv 的 human_verdict 填完之前，不得称为人工校正/验证。
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter

import numpy as np
import yaml
from pycocotools import mask as mu

from src import paths
from src.data import load_dataset

FLAGS = ["not_detected", "ambiguous_candidates", "low_score", "area_out_of_range", "pair_mask_overlap", "geometry_inconsistent"]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def qc_one(rec: dict, q: dict) -> dict:
    flags = {f: [] for f in FLAGS}
    T, R = rec["objects"]["target"], rec["objects"]["reference"]
    area = rec["width"] * rec["height"]
    for role, o in (("target", T), ("reference", R)):
        if o["box_xyxy"] is None or not o["mask_area"]:
            flags["not_detected"].append(role)
            continue
        c = o["candidates"]
        # 第二个“不同的”候选（与第一候选 IoU<0.5）分数接近
        others = [x for x in c[1:] if box_iou(x["box_xyxy"], c[0]["box_xyxy"]) < 0.5]
        if others and c[0]["score"] - others[0]["score"] < q["ambiguous_margin"]:
            flags["ambiguous_candidates"].append(role)
        if q["low_score"] is not None and o["score"] < q["low_score"]:
            flags["low_score"].append(role)
        frac = o["mask_area"] / area
        if frac < q["min_area_frac"] or frac > q["max_area_frac"]:
            flags["area_out_of_range"].append(role)
    pair_iou = None
    if T["mask_rle"] and R["mask_rle"]:
        pair_iou = float(mu.iou([T["mask_rle"]], [R["mask_rle"]], [0])[0][0])
        if pair_iou > q["max_pair_mask_iou"]:
            flags["pair_mask_overlap"].append("pair")
        (tx, ty), (rx, ry) = T["mask_centroid_xy"], R["mask_centroid_xy"]
        rel = rec["relation"]
        ok = {"left": tx < rx, "right": tx > rx, "on": ty < ry, "under": ty > ry}.get(rel)   # front/behind 不检查
        if ok is False:
            flags["geometry_inconsistent"].append(rel)
    active = [f for f in FLAGS if flags[f]]
    return {"flags": {f: flags[f] for f in active}, "auto_qc": "review" if active else "pass", "pair_mask_iou": pair_iou}


def main():
    cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto.yaml"))
    q = cfg["qc"]
    d = paths.get("data") / "whatsup_annotations" / cfg["version"]
    raw = [json.loads(l) for l in open(d / "raw.jsonl")]
    samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
    assert len(raw) == len(samples) == 820, (len(raw), len(samples))
    rows = []
    for rec in raw:
        r = qc_one(rec, q)
        rec.update(r)
        rec["annotation_status"] = "auto (not human-verified)"
        rows.append(rec)
    with open(d / "annotations.jsonl", "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    # ---- 统计 ----
    def pct(v, ps=(1, 5, 10, 25, 50, 75, 90)):
        return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}
    summ = {"version": cfg["version"], "qc_config": q, "n_images": len(rows),
            "annotations_sha256": hashlib.sha256((d / "annotations.jsonl").read_bytes()).hexdigest(), "by_dataset": {}}
    for ds in ("whatsup_a", "whatsup_b", "all"):
        sub = [r for r in rows if ds == "all" or r["dataset"] == ds]
        sc = {role: [r["objects"][role]["score"] for r in sub if r["objects"][role]["score"] is not None] for role in ("target", "reference")}
        summ["by_dataset"][ds] = {
            "n": len(sub), "n_pass": sum(r["auto_qc"] == "pass" for r in sub), "n_review": sum(r["auto_qc"] == "review" for r in sub),
            "flag_counts_images": dict(Counter(f for r in sub for f in r["flags"])),
            "flag_counts_by_role": dict(Counter(f"{f}:{x}" for r in sub for f, v in r["flags"].items() for x in v)),
            "review_by_relation": dict(Counter(r["relation"] for r in sub if r["auto_qc"] == "review")),
            "score_percentiles": {role: pct(v) for role, v in sc.items()},
            "score_hist_0.1_bins": {role: np.histogram(v, bins=np.linspace(0, 1, 11))[0].tolist() for role, v in sc.items()},
            "n_candidates_hist": {role: dict(sorted(Counter(min(r["objects"][role]["n_candidates"], 5) for r in sub).items())) for role in ("target", "reference")},
            "area_frac_percentiles": {role: pct([r["objects"][role]["mask_area"] / (r["width"] * r["height"]) for r in sub if r["objects"][role]["mask_area"]]) for role in ("target", "reference")}}
    (paths.REPO_ROOT / "manifests").mkdir(exist_ok=True)
    json.dump(summ, open(paths.REPO_ROOT / "manifests" / "whatsup_auto_qc_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ["by_dataset"], indent=1, ensure_ascii=False))
    # ---- 审核表：待审在前（标记多的更靠前），其后是通过的 ----
    order = sorted(rows, key=lambda r: (r["auto_qc"] != "review", -len(r["flags"]), r["dataset"], r["sample_id"]))
    out = paths.get("outputs") / "annotation_review"
    out.mkdir(parents=True, exist_ok=True)
    review = out / "review.csv"
    if review.exists():
        old = list(csv.DictReader(open(review)))
        if any((r.get("human_verdict") or "").strip() or (r.get("note") or "").strip() for r in old):
            review = out / "review.NEW.csv"      # 用户已开始填写：绝不覆盖
            print("[!] review.csv 已有人工填写内容，新表写到 review.NEW.csv")
    with open(review, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "page", "slot", "dataset", "sample_id", "relation", "target", "reference", "auto_qc"] + FLAGS +
                   ["score_target", "score_reference", "n_cand_target", "n_cand_reference", "area_frac_target", "area_frac_reference",
                    "pair_mask_iou", "human_verdict", "note"])
        for i, r in enumerate(order):
            T, R = r["objects"]["target"], r["objects"]["reference"]
            A = r["width"] * r["height"]
            w.writerow([i + 1, i // 20 + 1, i % 20 + 1, r["dataset"], r["sample_id"], r["relation"], T["name"], R["name"], r["auto_qc"]] +
                       ["|".join(r["flags"].get(fl, [])) for fl in FLAGS] +
                       [None if T["score"] is None else round(T["score"], 3), None if R["score"] is None else round(R["score"], 3),
                        T["n_candidates"], R["n_candidates"], None if not T["mask_area"] else round(T["mask_area"] / A, 4),
                        None if not R["mask_area"] else round(R["mask_area"] / A, 4),
                        None if r["pair_mask_iou"] is None else round(r["pair_mask_iou"], 3), "", ""])
    print("review table:", review)
    if "--no-sheets" in sys.argv:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    for p in range(0, len(order), 20):
        fig, axes = plt.subplots(4, 5, figsize=(25, 17))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (j, r) in zip(axes.flat, enumerate(order[p:p + 20])):
            s = samples[(r["dataset"], r["sample_id"])]
            im = np.asarray(Image.open(s.image_path).convert("RGB").resize((640, 480))).astype(np.float32) / 255
            sx, sy = 640 / r["width"], 480 / r["height"]
            for role, col in (("target", (1, 0, 0)), ("reference", (0, 0.3, 1))):
                o = r["objects"][role]
                if o["mask_rle"]:
                    m = np.asarray(Image.fromarray(mu.decode(o["mask_rle"]) * 255).resize((640, 480))) > 127
                    im[m] = 0.55 * im[m] + 0.45 * np.array(col)
            ax.imshow(im)
            for role, col in (("target", "red"), ("reference", "blue")):
                o = r["objects"][role]
                if o["box_xyxy"]:
                    x1, y1, x2, y2 = o["box_xyxy"]
                    ax.add_patch(Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy, fill=False, ec=col, lw=1.5))
                    ax.text(x1 * sx, y1 * sy - 3, f"{role[0].upper()}:{o['name']} {o['score']:.2f}", color="white", fontsize=7,
                            bbox=dict(fc=col, ec="none", pad=0.8))
            fl = ",".join(r["flags"]) or "-"
            ax.set_title(f"#{p + j + 1} [{r['auto_qc'].upper()}] {r['dataset'][-1].upper()}:{r['sample_id']}\nrel={r['relation']} | flags: {fl}",
                         fontsize=8, color="darkred" if r["auto_qc"] == "review" else "black")
        fig.suptitle(f"What's Up AUTO annotation (not human-verified) — page {p // 20 + 1}/{(len(order) + 19) // 20}   red=target  blue=reference", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"sheet_{p // 20 + 1:03d}.png", dpi=70)
        plt.close(fig)
    print("sheets:", out)


if __name__ == "__main__":
    main()
