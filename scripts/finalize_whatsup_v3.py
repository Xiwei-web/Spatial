"""按用户确认的结果更新 What's Up 标注 v3（需要 1 张 GPU 做 SAM 分割）。
输入：annotations/whatsup/confirm_v3.csv（用户确认表，按 target_pick / reference_pick 执行）、
      annotations/whatsup/corrections.yaml（更正表）、auto_v3/annotations_draft.jsonl（build_whatsup_v3.py 的初版）、auto_v3/proposals.json
做的事：
  1. 25 张待确认图：按确认的候选更新框并重新分割。
     8 张 plant：框用确认的并集框；mask = 叶片框（v2 的 plant 框）的 SAM mask ∪ 花盆框（'flower pot' 候选）的 SAM mask。
  2. 更正表里 action=exclude 的样本：usable=false，v3_status=excluded_label_contradicts_image。
  3. 对全部 820 张重算质检标记（初版对改选的图沿用了 v2 的旧标记），并对照检查未改动的图标记不变。
  4. 输出更新后的 annotations.jsonl、变更日志、几何不一致表、8 张 plant 的叠加图一页。
8 张 plant 的 mask 在用户看过叠加图之前 usable=false；用户确认后用 --plants-confirmed 重跑即定稿。
用法：source scripts/env.sh tools && python scripts/finalize_whatsup_v3.py [--plants-confirmed]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mu

from scripts.whatsup_qc import qc_one
from src import paths
from src.annotate import sam
from src.data import load_dataset

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto_v3.yaml"))
D = paths.get("data") / "whatsup_annotations" / CFG["version"]
A = paths.REPO_ROOT / "annotations" / "whatsup"
M = paths.REPO_ROOT / "manifests"
PLANTS_OK = "--plants-confirmed" in sys.argv


def rle_of(m: np.ndarray) -> dict:
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def fields_of(m: np.ndarray, iou_pred) -> dict:
    ys, xs = np.nonzero(m)
    return {"mask_rle": rle_of(m), "mask_area": int(m.sum()), "mask_centroid_xy": [float(xs.mean()), float(ys.mean())],
            "mask_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "sam_iou_pred": iou_pred}


def put(o, seg, cand, rank, why):
    o.update(seg)
    o.update(box_xyxy=cand["box_xyxy"], box_det_xyxy=cand["box_xyxy"], box_mask_xyxy=seg["mask_bbox_xyxy"], score=cand["score"],
             selected_candidate_rank=rank, selection=why)


def geometry_ok(rec):
    T, R = rec["objects"]["target"], rec["objects"]["reference"]
    (tx, ty), (rx, ry) = T["mask_centroid_xy"], R["mask_centroid_xy"]
    return {"left": tx < rx, "right": tx > rx, "on": ty < R["box_xyxy"][3], "under": ty > ry}.get(rec["relation"])


def main():
    draft = [json.loads(l) for l in open(D / "annotations_draft.jsonl")]
    assert len(draft) == 820
    P = json.load(open(D / "proposals.json"))
    confirm = {int(r["order_v2"]): r for r in csv.DictReader(open(A / "confirm_v3.csv", encoding="utf-8-sig"))}
    assert set(map(str, confirm)) == set(P) and len(confirm) == 25
    excl = {(c["dataset"], c["sample_id"]): c for c in yaml.safe_load(open(A / "corrections.yaml"))["corrections"] if c["action"] == "exclude"}
    samples = {(s.dataset, s.sample_id): s for ds in ("whatsup_a", "whatsup_b") for s in load_dataset(ds)}
    if not torch.cuda.is_available():       # 有的节点缺卡（如 gpu-05、gpu-07），不允许悄悄退回 CPU
        sys.exit("CUDA 不可用，退出")
    proc, model = sam.load(CFG["segmenter"]["name"], "cuda")
    log, plant_vis = [], []
    for rec in draft:
        order, key = rec["review_order_v2"], (rec["dataset"], rec["sample_id"])
        T, R = rec["objects"]["target"], rec["objects"]["reference"]
        if key in excl:
            rec.update(usable=False, v3_status="excluded_label_contradicts_image", needs_user_confirmation=[])
            log.append([order, *key, "剔除（usable=false）", excl[key]["reason"], excl[key]["decided_by"]])
        elif order in confirm:
            c, p = confirm[order], P[str(order)]
            tp, rp = c["target_pick"].strip(), c["reference_pick"].strip()
            by = {x["label"]: (i, x) for i, x in enumerate(p["candidates"])}
            assert tp in by, (order, tp)
            image = Image.open(samples[key].image_path).convert("RGB")
            differs = " ；与脚本建议不同" if p["proposed"] != ({"target": tp, "reference": rp} if rp else {"target": tp}) else ""
            basis = "用户确认（按模型预填的确认表执行，2026-11-25）"
            if p["case"] == "swapped_boxes":
                for o, lab in ((T, tp), (R, rp)):
                    k = int(lab[1:]) - 1
                    put(o, sam.segment(proc, model, image, o["candidates"][k]["box_xyxy"]), o["candidates"][k], k, "user_confirmed")
                detail = f"target→{tp}，reference→{rp}，两者重新分割"
            else:
                rank, cand = by[tp]
                T["candidates_v2"], T["candidates"], T["n_candidates"] = T["candidates"], p["candidates"], len(p["candidates"])
                if p["case"] == "plant_with_pot":
                    pot = [x for x in p["candidates"] if x["kind"] == "detected" and x["prompt"] == "flower pot"][0]
                    leaf_box = p["current_box"]
                    s_leaf, s_pot = (sam.segment(proc, model, image, b) for b in (leaf_box, pot["box_xyxy"]))
                    m_leaf, m_pot = (mu.decode(s["mask_rle"]).astype(bool) for s in (s_leaf, s_pot))
                    m = m_leaf | m_pot
                    put(T, fields_of(m, None), cand, rank, "user_confirmed")
                    T["mask_parts"] = {"leaf_box": leaf_box, "pot_box": pot["box_xyxy"], "leaf_mask_rle": s_leaf["mask_rle"], "pot_mask_rle": s_pot["mask_rle"],
                                       "leaf_sam_iou_pred": s_leaf["sam_iou_pred"], "pot_sam_iou_pred": s_pot["sam_iou_pred"]}
                    stat = {"order": order, "sample_id": key[1], "leaf_area": int(m_leaf.sum()), "pot_area": int(m_pot.sum()), "union_area": int(m.sum()),
                            "pot_only_area": int((m_pot & ~m_leaf).sum()), "pot_mask_share_of_pot_box": float(m_pot.sum() / ((pot["box_xyxy"][2] - pot["box_xyxy"][0]) * (pot["box_xyxy"][3] - pot["box_xyxy"][1]))),
                            "pot_sam_iou_pred": s_pot["sam_iou_pred"]}
                    plant_vis.append((rec, image, m_leaf, m_pot, stat))
                    detail = f"框→{tp}（叶片框与花盆框的并集）；mask = 叶片框 mask ∪ 花盆框 mask（花盆独有 {stat['pot_only_area']} px）"
                else:
                    put(T, sam.segment(proc, model, image, cand["box_xyxy"]), cand, rank, "user_confirmed")
                    detail = f"target→{tp}（{cand['prompt']}，{cand['score']:.2f}），重新分割"
            pending = p["case"] == "plant_with_pot" and not PLANTS_OK
            rec.update(usable=not pending, v3_status="pending_plant_mask_confirmation" if pending else "changed_by_user_confirmation",
                       needs_user_confirmation=["plant_mask"] if pending else [])
            log.append([order, *key, "待确认 plant mask（usable=false）" if pending else "按确认更新", detail + differs, basis])
        elif rec["v3_status"] == "changed_by_review_pick":
            log.append([order, *key, "按审核改选并重新分割（初版已做）", f"target 候选 {T['selected_candidate_rank'] + 1}，reference 候选 {R['selected_candidate_rank'] + 1}", "审核表的候选编号"])
    # ---- 全量重算质检 ----
    changed_flags_unchanged_rows, geom = [], []
    for rec in draft:
        old = (rec["flags"], rec["auto_qc"])
        rec.update(qc_one(rec, CFG["qc"]))
        rec["geometry_consistent"] = geometry_ok(rec)
        rec.pop("lr_geometry_consistent", None)
        if rec["v3_status"] == "unchanged_from_v2" and old != (rec["flags"], rec["auto_qc"]):
            changed_flags_unchanged_rows.append((rec["review_order_v2"], old[0], rec["flags"]))
        if rec["geometry_consistent"] is False:
            geom.append(rec)
    assert not changed_flags_unchanged_rows, changed_flags_unchanged_rows[:3]
    with open(D / "annotations.jsonl", "w") as f:
        for rec in draft:
            f.write(json.dumps(rec) + "\n")
    with open(M / "whatsup_auto_v3_changelog.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_v2", "dataset", "sample_id", "change", "detail", "basis"])
        w.writerows(sorted(log))
    with open(M / "whatsup_auto_v3_geometry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_v2", "dataset", "sample_id", "relation", "target_centroid_xy", "reference_centroid_xy", "v3_status", "usable"])
        for r in geom:
            w.writerow([r["review_order_v2"], r["dataset"], r["sample_id"], r["relation"], [round(v) for v in r["objects"]["target"]["mask_centroid_xy"]],
                        [round(v) for v in r["objects"]["reference"]["mask_centroid_xy"]], r["v3_status"], r["usable"]])
    summ = {"version": CFG["version"], "finalized": PLANTS_OK, "status_wording": draft[0]["annotation_status"], "n": len(draft),
            "v3_status": dict(Counter(r["v3_status"] for r in draft)), "usable": sum(r["usable"] for r in draft),
            "usable_by_dataset": dict(Counter(r["dataset"] for r in draft if r["usable"])), "review_source": dict(Counter(r["review_source"] for r in draft)),
            "geometry_inconsistent": [(r["review_order_v2"], r["sample_id"], r["v3_status"], r["usable"]) for r in geom],
            "qc_flags_on_changed_rows": {r["sample_id"]: r["flags"] for r in draft if r["v3_status"] not in ("unchanged_from_v2",) and r["flags"]},
            "plant_mask_stats": [s for *_, s in plant_vis], "annotations_sha256": hashlib.sha256((D / "annotations.jsonl").read_bytes()).hexdigest()}
    json.dump(summ, open(M / "whatsup_auto_v3_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # ---- 8 张 plant 的叠加图（一页）：每张左为原图裁剪，右为叠加；用轮廓线显示 mask 边界，避免颜色与物体接近时看不清 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(4, 4, figsize=(24, 24))
    for k, (rec, image, ml, mp, st) in enumerate(plant_vis):
        o = rec["objects"]["target"]
        x1, y1, x2, y2 = o["box_xyxy"]
        pad = 35
        cx1, cy1, cx2, cy2 = max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(image.width, int(x2 + pad)), min(image.height, int(y2 + pad))
        crop = np.asarray(image)[cy1:cy2, cx1:cx2]
        a, b = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        a.imshow(crop); a.set_title(f"order {st['order']}  {st['sample_id']}  (original crop)", fontsize=12)
        b.imshow(crop)
        L, Q = ml[cy1:cy2, cx1:cx2], mp[cy1:cy2, cx1:cx2]
        tint = np.zeros((*L.shape, 4)); tint[L] = (1, 0, 0, 0.30); tint[Q & ~L] = (0, 0.3, 1, 0.40)
        b.imshow(tint)
        b.contour((L | Q).astype(float), levels=[0.5], colors="yellow", linewidths=2)
        b.add_patch(Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.8))
        pb = o["mask_parts"]["pot_box"]
        b.add_patch(Rectangle((pb[0] - cx1, pb[1] - cy1), pb[2] - pb[0], pb[3] - pb[1], fill=False, ec="cyan", lw=1.5, ls="--"))
        b.set_title(f"pot-only area {st['pot_only_area']} px; pot mask fills {st['pot_mask_share_of_pot_box']:.0%} of pot box", fontsize=12)
        a.axis("off"); b.axis("off")
    fig.legend(handles=[Line2D([], [], color="red", lw=8, alpha=.4, label="leaf-box mask"), Line2D([], [], color="blue", lw=8, alpha=.5, label="added by pot-box mask"),
                        Line2D([], [], color="yellow", lw=2, label="final union mask outline"), Line2D([], [], color="white", lw=2, label="confirmed box"),
                        Line2D([], [], color="cyan", lw=2, ls="--", label="flower-pot box")], loc="lower center", ncol=5, fontsize=14, facecolor="gray")
    fig.suptitle("What's Up v3: 8 plant masks = leaf-box mask U pot-box mask   [auto annotation; PENDING your confirmation]", fontsize=18)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out = paths.get("outputs") / "annotation_review" / "auto_v3_changes" / "plants_overlay.png"
    fig.savefig(out, dpi=75)
    print("wrote", out)


if __name__ == "__main__":
    main()
