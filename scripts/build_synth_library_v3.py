"""合成场景抠图库 v3（configs/synth/library_v3.yaml；新增特征在 src/synth/library_v3.py）。分阶段：
  features   --pool v2|v3   （CPU）规则 a、b（含不限于接触边的参考变体）、c、d 的几何特征 → <OUT>/features_<pool>.jsonl
  clip_extra --pool v2|v3   （CPU）规则 d 的 pile/single 与规则 e 的负类 CLIP → <OUT>/clip_extra_<pool>.jsonl
  sam        --pool v2|v3 [--labeled-only]（GPU 或 CPU）规则 f：SAM 2.1 以 COCO 框为提示的 IoU → <OUT>/sam_<pool>.jsonl
  tune       （CPU）只用 dev 标签在网格上定 T_occ、T_edge、T_sol、T_sam，写 configs/synth/library_v3_thresholds.yaml 与 manifests/synth_library_v3_tune.json；
             达不到目标（dev 上 BAD 召回 >= 0.80 且 OK 损失 <= 0.25）就报告，不放宽
pool=v2：v2 的 cut 阶段通过的抠图（<data>/synth/cutouts_v2/cut.jsonl），用来定阈值。OUT = <data>/synth/cutouts_v3/。"""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src import paths
from src.synth import library_v3 as L3

REPO = paths.REPO_ROOT
CFG = yaml.safe_load(open(REPO / "configs" / "synth" / "library_v3.yaml"))
RU = CFG["rules"]
DATA = paths.get("data")
OUT = DATA / "synth" / "cutouts_v3"
V2 = DATA / "synth" / "cutouts_v2"


def pool_rows(pool):
    src = V2 / "cut.jsonl" if pool == "v2" else OUT / "cut.jsonl"
    return [r for r in map(json.loads, open(src)) if r["cut_status"] == "ok"]


def labels():
    f = REPO / "annotations" / "synth" / "library_v2_prereview.csv"
    return {r["cutout_id"]: r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))}


def load_split_anns(split):
    d = json.load(open(DATA / "coco" / "annotations" / f"instances_{split}2017.json"))
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    kp = {a["id"]: a for a in json.load(open(DATA / "coco" / "annotations" / f"person_keypoints_{split}2017.json"))["annotations"]}
    return {im["id"]: im for im in d["images"]}, by_img, kp


def stage_features(pool):
    from src.synth import cutouts as cu
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pool_rows(pool)
    res = []
    for split in ("train", "val"):
        imgs, by_img, kp = load_split_anns(split)
        for r in [x for x in rows if x["src"] == split]:
            im = imgs[r["image_id"]]
            H, W = im["height"], im["width"]
            anns = by_img[r["image_id"]]
            me = next(a for a in anns if a["id"] == r["ann_id"])
            m_full = cu._coco_ann_mask(me, H, W)
            m, _, _ = cu.largest_component(m_full)
            others = np.zeros((H, W), bool)
            for a in anns:
                if a["id"] != me["id"]:
                    others |= cu._coco_ann_mask(a, H, W)
            near_a = L3.near_others(others, RU["a_occ_contact"]["dilate_px"])
            near_b = L3.near_others(others, RU["b_contact_straight_edge"]["dilate_px"])
            b = RU["b_contact_straight_edge"]
            f = {"cutout_id": r["cutout_id"], "category": r["category"], "src": split,
                 "occ_contact_frac": L3.occ_contact_frac(m, near_a),
                 "contact_straight_run": L3.straight_runs(me["segmentation"], float(m.sum()), near_b, b["angle_tol_deg"], b["min_contact"]),
                 "any_straight_run_ref": L3.straight_runs(me["segmentation"], float(m.sum()), None, b["angle_tol_deg"], b["min_contact"]),
                 "n_components": L3.n_components(m_full, RU["d_pile"]["component_min_frac"]), "solidity": L3.solidity(m),
                 "n_other_instances": len(anns) - 1}
            if r["category"] == "person":
                k = kp.get(me["id"])
                f["person_ok"] = bool(k and k.get("num_keypoints", 0) > 0 and L3.person_visible(k["keypoints"]))
            res.append(f)
    with open(OUT / f"features_{pool}.jsonl", "w") as fo:
        fo.writelines(json.dumps(x) + "\n" for x in res)
    print("features", pool, len(res))


def stage_clip_extra(pool):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from scripts.build_synth_library_v2 import CFG as C2, load_coco, on_gray
    from src.synth import library_v2 as L2
    torch.set_num_threads(8)
    rows = pool_rows(pool)
    png = (V2 if pool == "v2" else OUT) / "png"
    _, cats, _ = load_coco("val")
    base = L2.clip_labels([cats[k] for k in sorted(cats)], C2["clip"]["confusable_labels"])
    art = lambda n: "an" if n[0].lower() in "aeiou" else "a"      # noqa: E731
    path = paths.get("new_models") / C2["clip"]["model"]
    proc, clip = CLIPProcessor.from_pretrained(path), CLIPModel.from_pretrained(path).eval()
    d, e = RU["d_pile"], RU["e_clip_negatives"]
    res = []
    with torch.no_grad():
        for r in rows:
            c = r["category"]
            if c not in d["categories"] and c not in e:
                continue
            img = on_gray(np.asarray(Image.open(png / f"{r['cutout_id']}.png").convert("RGBA")), C2["margin_frac"])
            x = {"cutout_id": r["cutout_id"], "category": c}
            if c in d["categories"]:
                t = [p.format(c=c) for p in d["clip_prompts"]]
                p = clip(**proc(text=t, images=[img], return_tensors="pt", padding=True)).logits_per_image.softmax(-1)[0].numpy()
                x["pile_prob"] = float(p[0])
            if c in e:
                labs = base + list(e[c])
                texts = [C2["clip"]["prompt"].format(article=art(n), name=n) for n in base] + list(e[c])
                p = clip(**proc(text=texts, images=[img], return_tensors="pt", padding=True)).logits_per_image.softmax(-1)[0].numpy()
                top = int(p.argmax())
                x.update(neg_top=labs[top] in e[c], top1=labs[top], top1_prob=float(p[top]), neg_max_prob=float(p[len(base):].max()))
            res.append(x)
    with open(OUT / f"clip_extra_{pool}.jsonl", "w") as fo:
        fo.writelines(json.dumps(x) + "\n" for x in res)
    print("clip_extra", pool, len(res))


def stage_sam(pool, labeled_only):
    import torch
    from PIL import Image
    from src.annotate import sam
    from src.synth import cutouts as cu
    from pycocotools import mask as mu
    rows = pool_rows(pool)
    if labeled_only:
        lab = labels()
        rows = [r for r in rows if r["cutout_id"] in lab]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(8)
    proc, model = sam.load(device=dev)
    out_f = OUT / f"sam_{pool}{'_labeled' if labeled_only else ''}.jsonl"
    done = {json.loads(l)["cutout_id"] for l in open(out_f)} if out_f.is_file() else set()
    anns = {}
    for split in ("train", "val"):
        d = json.load(open(DATA / "coco" / "annotations" / f"instances_{split}2017.json"))
        need = {r["ann_id"] for r in rows if r["src"] == split}
        anns.update({a["id"]: a for a in d["annotations"] if a["id"] in need})
    with open(out_f, "a") as fo:
        for n, r in enumerate(rows):
            if r["cutout_id"] in done:
                continue
            p = (DATA / "coco" / "train2017_subset" / r["file_name"]) if r["src"] == "train" else (DATA / "coco" / "val2017" / r["file_name"])
            img = Image.open(p).convert("RGB")
            x, y, w, h = r["bbox"]
            s = sam.segment(proc, model, img, [x, y, x + w, y + h])
            m_sam = mu.decode(s["mask_rle"]).astype(bool)
            m_coco = cu._coco_ann_mask(anns[r["ann_id"]], img.size[1], img.size[0])
            fo.write(json.dumps({"cutout_id": r["cutout_id"], "sam_iou": L3.mask_iou(m_sam, m_coco), "sam_iou_pred": s["sam_iou_pred"], "device": dev}) + "\n")
            fo.flush()
            if n % 100 == 0:
                print(n, len(rows), flush=True)
    print("sam done", out_f)


def flags_for(f, cx, s, T):
    """单条抠图在给定阈值下每条规则是否剔除。"""
    c = f["category"]
    fl = {"a_occ": f["occ_contact_frac"] > T["T_occ"], "b_edge": f["contact_straight_run"] > T["T_edge"],
          "c_person": c == "person" and not f.get("person_ok", False)}
    d_on = c in RU["d_pile"]["categories"]
    fl["d_pile"] = d_on and (f["n_components"] > 1 or f["solidity"] < T["T_sol"] or (cx is not None and cx.get("pile_prob", 0) > RU["d_pile"]["clip_pile_max_prob"]))
    fl["e_neg"] = bool(cx and cx.get("neg_top"))
    fl["f_sam"] = s is not None and s["sam_iou"] < T["T_sam"]
    return fl


def stage_tune():
    lab = labels()
    F = {x["cutout_id"]: x for x in map(json.loads, open(OUT / "features_v2.jsonl"))}
    CX = {x["cutout_id"]: x for x in map(json.loads, open(OUT / "clip_extra_v2.jsonl"))}
    S = {x["cutout_id"]: x for x in map(json.loads, open(OUT / "sam_v2_labeled.jsonl"))}
    excl = set(CFG["exclude_categories_extra"])
    items = [(cid, l) for cid, l in lab.items() if cid in F and l["verdict"] in ("BAD", "OK")]
    miss = [cid for cid, l in lab.items() if cid not in F or cid not in S]
    assert not miss, f"标签里有 {len(miss)} 个抠图缺特征 / SAM：{miss[:5]}"
    G = CFG["threshold_search"]["grid"]
    def evaluate(T, side, scope):
        sel = [(cid, l) for cid, l in items if l["side"] == side and (scope == "all" or l["category"] not in excl)]
        per = defaultdict(list)
        anyf, ver = [], []
        for cid, l in sel:
            fl = flags_for(F[cid], CX.get(cid), S.get(cid), T)
            if scope == "all" and l["category"] in excl:
                fl = {**fl, "class_excluded": True}
            for k, v in fl.items():
                per[k].append(v)
            anyf.append(any(fl.values())); ver.append(l["verdict"])
        out = {"combined": L3.rates(anyf, ver)}
        for k, v in per.items():
            out[k] = L3.rates(v, ver)
        return out
    best = None
    for vals in itertools.product(G["T_occ"], G["T_edge"], G["T_sol"], G["T_sam"]):
        T = dict(zip(("T_occ", "T_edge", "T_sol", "T_sam"), vals))
        r = evaluate(T, "dev", "primary")["combined"]
        if r["ok_loss"] > CFG["threshold_search"]["max_ok_loss"]:
            continue
        key = (r["bad_recall"], -r["ok_loss"], vals[0], vals[1], -vals[2], -vals[3])      # 召回最高 → OK 损失最低 → 阈值更宽松
        if best is None or key > best[0]:
            best = (key, T, r)
    if best is None:
        res = {"status": "no_threshold_meets_ok_loss", "note": "网格上没有任何组合能让 dev 的 OK 损失 <= 0.25"}
    else:
        T = best[1]
        res = {"status": "meets_target" if best[2]["bad_recall"] >= CFG["threshold_search"]["target_bad_recall"] else "below_target_recall",
               "thresholds": T, "dev_primary": evaluate(T, "dev", "primary"), "test_primary": evaluate(T, "test", "primary"),
               "dev_incl_class_exclusion": evaluate(T, "dev", "all"), "test_incl_class_exclusion": evaluate(T, "test", "all")}
    # 参考：不限于接触边的长直边（不参与定阈值，只报告）
    ref = {}
    for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for side in ("dev", "test"):
            sel = [(cid, l) for cid, l in items if l["side"] == side and l["category"] not in excl]
            ref[f"{side}@{t}"] = L3.rates([F[cid]["any_straight_run_ref"] > t for cid, _ in sel], [l["verdict"] for _, l in sel])
    res["reference_any_straight_run"] = ref
    res["n_labels"] = dict(Counter((l["side"], l["verdict"]) for _, l in items))
    res["bad_codes_dev_missed"] = None
    if best is not None:
        miss_codes = Counter(l["code"] for cid, l in items if l["side"] == "dev" and l["verdict"] == "BAD" and l["category"] not in excl
                             and not any(flags_for(F[cid], CX.get(cid), S.get(cid), best[1]).values()))
        res["bad_codes_dev_missed"] = dict(miss_codes)
        yaml.safe_dump({"note": "scripts/build_synth_library_v3.py tune 在 dev 标签上定的阈值（BAD 正例、OK 负例、UNSURE 不参与）；status 见 manifests/synth_library_v3_tune.json",
                        "status": res["status"], **best[1]}, open(REPO / "configs" / "synth" / "library_v3_thresholds.yaml", "w"), allow_unicode=True, sort_keys=False)
    json.dump(res, open(REPO / "manifests" / "synth_library_v3_tune.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "reference_any_straight_run"}, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--pool", default="v2"); ap.add_argument("--labeled-only", action="store_true")
    A = ap.parse_args()
    {"features": lambda: stage_features(A.pool), "clip_extra": lambda: stage_clip_extra(A.pool), "sam": lambda: stage_sam(A.pool, A.labeled_only), "tune": stage_tune}[A.stage]()
