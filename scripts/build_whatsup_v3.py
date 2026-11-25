"""构建 What's Up 标注 v3（需要 1 张 GPU：SAM 重新分割 + 对 25 张图重新检测）。
v3 = v2 + 审核结论（annotations/whatsup/review_final_v2.csv），配置 configs/annotation/whatsup_auto_v3.yaml。
  - 审核 ok 的 786 张：原样沿用 v2。
  - direct_picks（7 张）：按审核表给出的候选编号改选，改动的物体重新分割。
  - label_check（2 张）：框沿用 v2，标记为“待用户决定标签”（usable=false）。
  - redetect（25 张）：重新检测，出候选图和确认表；在用户确认前 v3 里保留 v2 的值但 usable=false。
输出：
  <data>/whatsup_annotations/auto_v3/annotations.jsonl、proposals.json（25 张的候选与建议）
  <outputs>/annotation_review/auto_v3_changes/{chg_<order>_*.png, confirm.csv, README.txt}
  manifests/whatsup_auto_v3_changelog.csv、manifests/whatsup_auto_v3_geometry_lr.csv、manifests/whatsup_auto_v3_summary.json
在用户确认那 25 张之前，不得用 v3 生成任何下游数据。
用法：source scripts/env.sh tools && python scripts/build_whatsup_v3.py
"""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mu

from src import paths
from src.annotate import gdino, sam
from src.annotate.selection import box_iou
from src.data import load_dataset

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto_v3.yaml"))
RD = CFG["redetect"]
OUT_DATA = paths.get("data") / "whatsup_annotations" / CFG["version"]
OUT_REV = paths.get("outputs") / "annotation_review" / f"{CFG['version']}_changes"
STATUS = "自动标注，经模型辅助预审和部分人工复核"


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def apply_seg(o, seg, box, score, rank, why):
    o.update(seg)
    o.update(box_xyxy=box, box_det_xyxy=box, box_mask_xyxy=seg["mask_bbox_xyxy"], score=score, selected_candidate_rank=rank, selection=why)


def lr_geometry(rec):
    """left/right：两个 mask 质心的水平顺序是否与标签一致。返回 True/False；非 left/right 或缺 mask 返回 None。"""
    T, R = rec["objects"]["target"], rec["objects"]["reference"]
    if rec["relation"] not in ("left", "right") or not T["mask_centroid_xy"] or not R["mask_centroid_xy"]:
        return None
    tx, rx = T["mask_centroid_xy"][0], R["mask_centroid_xy"][0]
    return bool(tx < rx) if rec["relation"] == "left" else bool(tx > rx)


def redetect_candidates(det, image, name, ref_box, W, H):
    """用配置里的多个短语重新检测，排除与 reference 重合 / 覆盖 reference / 近乎整图的框，按分数去重后返回。"""
    cands = []
    for prompt in RD["prompts"].get(name, [name]):
        for c in gdino.detect(det, image, prompt, RD["floor"]):
            b = c["box_xyxy"]
            if box_iou(b, ref_box) > RD["exclude_reference_iou"] or inter(b, ref_box) / max(area(ref_box), 1) > RD["exclude_covering_reference"]:
                continue
            if area(b) > 0.5 * W * H:
                continue
            cands.append({"box_xyxy": b, "score": c["score"], "prompt": prompt, "kind": "detected"})
    cands.sort(key=lambda c: -c["score"])
    keep = []
    for c in cands:
        if all(box_iou(c["box_xyxy"], k["box_xyxy"]) <= 0.8 for k in keep):
            keep.append(c)
    return keep


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REV.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = [json.loads(l) for l in open(paths.get("data") / "whatsup_annotations" / CFG["base_version"] / "annotations.jsonl")]
    by_key = {(r["dataset"], r["sample_id"]): r for r in base}
    review = list(csv.DictReader(open(paths.REPO_ROOT / CFG["review_table"], encoding="utf-8-sig")))
    samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
    case_of = {o: c for c, v in RD["cases"].items() for o in v["orders"]}
    non_ok = {int(v["order"]) for v in review if v["final_verdict"] != "ok"}
    planned = set(CFG["direct_picks"]) | set(CFG["label_check"]) | set(case_of)
    assert non_ok == planned, f"审核表的非 ok 行与配置的处理清单不一致：{sorted(non_ok ^ planned)}"
    sam_proc, sam_model = sam.load(CFG["segmenter"]["name"], dev)
    det = gdino.load(CFG["detector"]["variant"], dev)
    changelog, proposals, out_rows = [], {}, []
    for v in review:
        order, key = int(v["order"]), (v["dataset"], v["sample_id"])
        rec = json.loads(json.dumps(by_key[key]))          # 深拷贝 v2 记录
        rec.update(annotation_version=CFG["version"], annotation_status=STATUS, review_order_v2=order,
                   review_final_verdict_v2=v["final_verdict"], review_source=v["review_source"], usable=True, v3_status="unchanged_from_v2",
                   needs_user_confirmation=[])
        image = Image.open(samples[key].image_path).convert("RGB")
        W, H = image.size
        T, R = rec["objects"]["target"], rec["objects"]["reference"]
        if order in CFG["direct_picks"]:
            tp, rp = int(v["target_pick"]) - 1, int(v["reference_pick"]) - 1
            what = []
            for o, pick, tag in ((T, tp, "target"), (R, rp, "reference")):
                if pick != (o.get("selected_candidate_rank") or 0):
                    old = o.get("selected_candidate_rank") or 0
                    c = o["candidates"][pick]
                    apply_seg(o, sam.segment(sam_proc, sam_model, image, c["box_xyxy"]), c["box_xyxy"], c["score"], pick, "review_pick")
                    what.append(f"{tag}: 候选 {old + 1} → {pick + 1}，重新分割")
            rec["v3_status"] = "changed_by_review_pick"
            changelog.append([order, v["dataset"], v["sample_id"], "改选候选并重新分割", "；".join(what), v["final_verdict"], v["review_source"],
                              "审核表给出的候选编号（3.1 节，用户指定无需再确认）", v["note"]])
        elif order in CFG["label_check"]:
            rec.update(usable=False, v3_status="pending_label_decision")
            rec["needs_user_confirmation"].append("label_contradicts_image")
            changelog.append([order, v["dataset"], v["sample_id"], "标注未改；标记为待定（usable=false）", "框正确，图像中左右位置与文件名/标签相反",
                              v["final_verdict"], v["review_source"], "人工确认 + 原始数据核对（官方 JSON、文件名、压缩包内文件一致，问题在原始数据）", v["note"]])
        elif order in case_of:
            case = case_of[order]
            rec.update(usable=False, v3_status="pending_user_confirmation", pending_case=case)
            rec["needs_user_confirmation"].append("redetect_" + case)
            prop = {"order": order, "dataset": v["dataset"], "sample_id": v["sample_id"], "case": case, "role": "target", "name": T["name"],
                    "current_box": T["box_xyxy"], "reference_box": R["box_xyxy"], "candidates": [], "proposed": None, "proposed_basis": None}
            if case == "swapped_boxes":       # 用已保存的候选：建议按修正后的规则 (a) 对调
                prop["role"] = "both"
                prop["candidates"] = [{"box_xyxy": c["box_xyxy"], "score": c["score"], "prompt": T["name"], "kind": "stored_target", "label": f"T{i + 1}"} for i, c in enumerate(T["candidates"])] \
                    + [{"box_xyxy": c["box_xyxy"], "score": c["score"], "prompt": R["name"], "kind": "stored_reference", "label": f"R{i + 1}"} for i, c in enumerate(R["candidates"])]
                prop["proposed"] = {"target": "T1", "reference": "R2"}
                prop["proposed_basis"] = "修正后的规则 (a)：两者最高分候选重合，target 分数高 → target 保留 T1，reference 改用 R2"
            else:
                cands = redetect_candidates(det, image, T["name"], R["box_xyxy"], W, H)
                extra = []
                if case == "shoes_pair":      # 两只鞋候选框的并集
                    singles = [c for c in cands if area(c["box_xyxy"]) < 0.7 * max(area(x["box_xyxy"]) for x in cands)] or cands
                    pair = None
                    for i in range(len(singles)):
                        for j in range(i + 1, len(singles)):
                            if box_iou(singles[i]["box_xyxy"], singles[j]["box_xyxy"]) < 0.3:
                                pair = (singles[i], singles[j]); break
                        if pair:
                            break
                    if pair:
                        extra.append({"box_xyxy": union_box(pair[0]["box_xyxy"], pair[1]["box_xyxy"]), "score": min(pair[0]["score"], pair[1]["score"]),
                                      "prompt": "union of two shoe boxes", "kind": "union"})
                if case == "plant_with_pot":  # 当前叶片框 与 花盆候选框 的并集
                    cur = T["box_xyxy"]
                    pots = [c for c in cands if c["prompt"] in ("flower pot", "pot") and box_iou(c["box_xyxy"], cur) < 0.5
                            and min(c["box_xyxy"][2], cur[2]) - max(c["box_xyxy"][0], cur[0]) > -0.5 * (cur[2] - cur[0])]
                    if pots:
                        extra.append({"box_xyxy": union_box(cur, pots[0]["box_xyxy"]), "score": pots[0]["score"],
                                      "prompt": f"union of current plant box and '{pots[0]['prompt']}' box", "kind": "union"})
                allc = (extra + cands)[: RD["max_candidates"]]
                for i, c in enumerate(allc):
                    c["label"] = f"C{i + 1}"
                prop["candidates"] = allc
                cur = T["box_xyxy"]
                pick, basis = None, None
                if not allc:
                    basis = "重新检测后仍没有可用候选：需要用户给出点提示位置"
                elif case in ("shoes_pair", "plant_with_pot") and extra:
                    pick, basis = "C1", "并集框（符合口径：" + ("两只鞋" if case == "shoes_pair" else "包含花盆") + "）"
                elif case == "photograph_with_mat":
                    bigger = [c for c in allc if inter(c["box_xyxy"], cur) / max(area(cur), 1) > 0.9 and area(cur) * 1.05 < area(c["box_xyxy"]) < area(cur) * 4]
                    if bigger:
                        best = min(bigger, key=lambda c: area(c["box_xyxy"]))
                        pick, basis = best["label"], "包含当前框且略大的最小候选（口径：包含白色衬纸）"
                    else:
                        pick, basis = allc[0]["label"], "没有找到包含当前框的更大候选；给出分数最高的候选，请人工判断"
                elif case == "bread_body":
                    alt = [c for c in allc if box_iou(c["box_xyxy"], cur) < 0.5]
                    pick, basis = (alt[0]["label"], "与当前框（面包袋尾部）不重合的最高分候选") if alt else (allc[0]["label"], "只有与当前框重合的候选，请人工判断")
                else:
                    pick, basis = allc[0]["label"], "排除与 reference 重合/覆盖的框后分数最高的候选"
                prop["proposed"], prop["proposed_basis"] = ({"target": pick} if pick else None), basis
                if pick:                      # 预先对建议的候选做分割，供确认图显示
                    c = next(x for x in allc if x["label"] == pick)
                    prop["proposed_segmentation"] = sam.segment(sam_proc, sam_model, image, c["box_xyxy"])
            proposals[str(order)] = prop
            changelog.append([order, v["dataset"], v["sample_id"], "待用户确认（v3 中 usable=false，暂保留 v2 的值）",
                              f"重新检测[{case}]：{len(prop['candidates'])} 个候选；建议 {prop['proposed']}（{prop['proposed_basis']}）",
                              v["final_verdict"], v["review_source"], "人工确认" if v["review_source"] == "human_confirmed" else "口径规则", v["note"]])
        ok = lr_geometry(rec)
        rec["lr_geometry_consistent"] = ok
        if ok is False and "lr_order_contradicts_label" not in rec["needs_user_confirmation"]:
            rec["needs_user_confirmation"].append("lr_order_contradicts_label")
        out_rows.append(rec)
    with open(OUT_DATA / "annotations.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    json.dump(proposals, open(OUT_DATA / "proposals.json", "w"), indent=1, ensure_ascii=False)
    M = paths.REPO_ROOT / "manifests"
    with open(M / "whatsup_auto_v3_changelog.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_v2", "dataset", "sample_id", "change", "detail", "review_final_verdict", "review_source", "basis", "review_note"])
        w.writerows(sorted(changelog))
    with open(M / "whatsup_auto_v3_geometry_lr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_v2", "dataset", "sample_id", "relation", "target_centroid_x", "reference_centroid_x", "v3_status", "usable", "review_final_verdict_v2", "review_source", "user_decision"])
        for r in out_rows:
            if r["lr_geometry_consistent"] is False:
                w.writerow([r["review_order_v2"], r["dataset"], r["sample_id"], r["relation"], round(r["objects"]["target"]["mask_centroid_xy"][0]),
                            round(r["objects"]["reference"]["mask_centroid_xy"][0]), r["v3_status"], r["usable"], r["review_final_verdict_v2"], r["review_source"], ""])
    from collections import Counter
    summ = {"version": CFG["version"], "status_wording": STATUS, "n": len(out_rows), "v3_status": dict(Counter(r["v3_status"] for r in out_rows)),
            "usable": sum(r["usable"] for r in out_rows), "not_usable_pending": sum(not r["usable"] for r in out_rows),
            "review_source": dict(Counter(r["review_source"] for r in out_rows)),
            "lr_geometry_inconsistent": [(r["review_order_v2"], r["sample_id"], r["v3_status"]) for r in out_rows if r["lr_geometry_consistent"] is False],
            "needs_user_confirmation": dict(Counter(x for r in out_rows for x in r["needs_user_confirmation"])),
            "proposals_without_candidates": [p["order"] for p in proposals.values() if not p["candidates"]],
            "n_candidates_per_proposal": {k: len(p["candidates"]) for k, p in proposals.items()},
            "annotations_sha256": hashlib.sha256((OUT_DATA / "annotations.jsonl").read_bytes()).hexdigest(),
            "downstream_blocked": "在用户确认 25 张待确认图之前，不得用 v3 生成任何下游数据"}
    json.dump(summ, open(M / "whatsup_auto_v3_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "n_candidates_per_proposal"}, indent=1, ensure_ascii=False))
    # ---- 确认图与确认表 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    COL = ["#ff0000", "#ff7f00", "#c0007f", "#00a000", "#8b00ff", "#00a0a0", "#804000", "#606060", "#0044ff", "#00a0ff"]
    confirm = OUT_REV / "confirm.csv"
    if confirm.exists():
        print("[!] confirm.csv 已存在，不覆盖")
    else:
        with open(confirm, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order_v2", "image_id", "case", "object_name", "n_candidates", "proposed_pick", "proposed_basis", "target_pick", "reference_pick", "note"])
            for k in sorted(proposals, key=int):
                p = proposals[k]
                prop = "" if not p["proposed"] else "/".join(f"{a}={b}" for a, b in p["proposed"].items())
                w.writerow([p["order"], f"{p['dataset']}:{p['sample_id']}", p["case"], p["name"], len(p["candidates"]), prop, p["proposed_basis"], "", "", ""])
    (OUT_REV / "README.txt").write_text(
        "v3 待确认的 25 张图\n"
        "每张图 chg_<order>_*.png：左 = 全部候选（编号 C1..Cn；对调的那张是 T1..、R1..），蓝色细线 = reference 当前框，白色点线 = target 当前（被判错误的）框；\n"
        "右 = 建议的候选及其 SAM mask。\n"
        "confirm.csv：target_pick 填候选编号（如 C2 或 T1）；同意建议可直接填建议的编号；都不对填 none（之后需要你给点提示位置）。\n"
        "reference_pick 只有 order 682（两个框对调）需要填。\n"
        "候选来源写在图上每个候选的标签里（检测短语 / union）。检测下限 0.05；已排除与 reference 重合或覆盖 reference 的框。\n")
    for k in sorted(proposals, key=int):
        p = proposals[k]
        image = np.asarray(Image.open(samples[(p["dataset"], p["sample_id"])].image_path).convert("RGB"))
        fig, axes = plt.subplots(1, 2, figsize=(22, 8.8))
        axes[0].imshow(image)
        if p["case"] != "swapped_boxes":
            rb, cb = p["reference_box"], p["current_box"]
            axes[0].add_patch(Rectangle(rb[:2], rb[2] - rb[0], rb[3] - rb[1], fill=False, ec="#0044ff", lw=1.2))
            axes[0].add_patch(Rectangle(cb[:2], cb[2] - cb[0], cb[3] - cb[1], fill=False, ec="white", lw=1.5, ls=":"))
        for i, c in enumerate(p["candidates"]):
            b = c["box_xyxy"]
            axes[0].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=COL[i % len(COL)], lw=2.2))
            axes[0].text(b[0], b[1] - 4 - 0 * i, f"{c['label']} {c['score']:.2f} [{c['prompt']}]", color="white", fontsize=9, va="bottom",
                         bbox=dict(fc=COL[i % len(COL)], ec="none", pad=1.2))
        axes[0].set_title(f"candidates ({len(p['candidates'])})   blue thin = reference box, white dotted = current (rejected) target box", fontsize=10)
        vis = image.astype(np.float32) / 255
        ttl = "no proposal: needs a point prompt from the reviewer"
        if p.get("proposed_segmentation"):
            m = mu.decode(p["proposed_segmentation"]["mask_rle"]).astype(bool)
            vis[m] = 0.5 * vis[m] + 0.5 * np.array([1, 0, 0])
            ttl = f"proposed: {p['proposed']}  (SAM mask in red)"
        elif p["proposed"]:
            ttl = f"proposed: {p['proposed']}"
        axes[1].imshow(vis)
        if p["proposed"]:
            for role, lab in p["proposed"].items():
                c = next(x for x in p["candidates"] if x["label"] == lab)
                b = c["box_xyxy"]
                col = "red" if role == "target" else "#0044ff"
                axes[1].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2.5))
                axes[1].text(b[0], b[1] - 4, f"{role}: {lab}", color="white", fontsize=10, va="bottom", bbox=dict(fc=col, ec="none", pad=1.2))
        axes[1].set_title(ttl, fontsize=10)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"order {p['order']}  {p['dataset']}:{p['sample_id']}   case={p['case']}   object='{p['name']}'   ({STATUS}; v3 pending confirmation)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_REV / f"chg_{p['order']:03d}_{p['dataset'][-1].upper()}_{p['sample_id']}.png", dpi=70)
        plt.close(fig)
    print("wrote", len(proposals), "confirmation images to", OUT_REV)


if __name__ == "__main__":
    main()
