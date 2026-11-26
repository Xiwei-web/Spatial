"""合成场景抠图库 v2（configs/synth/library_v2.yaml；规则在 src/synth/library_v2.py）。分阶段运行，每阶段的输出是下一阶段的输入：
  select   （CPU，只读标注）规则 a、b、g、泄漏排除；在 a、b 后的 train 池上逐类算 fill 分位数，写 configs/synth/library_v2_fill_quantiles.yaml；
           再按规则 c 与 d（用标注框 / 面积预筛）筛选，train 每类最多 train_cap_per_category 个 → candidates.jsonl、download_list.txt
  download （网络）只下载候选所在的 train2017 图像到 <data>/coco/train2017_subset/
  cut      （CPU）最大连通块抠图，按紧致框复核规则 d；整图感知哈希与评测图像比对 → cut.jsonl、png/
  clip     （CPU）CLIP 正常尺寸检查（e）与 18 px 渲染检查（f）→ clip.jsonl
  finalize （CPU）划分（train 按图像固定种子；val 继承原划分）、每类每侧计数、门槛、library.jsonl、splits/synth_library_v2.json、manifest、联系表
用法：python scripts/build_synth_library_v2.py <stage>"""
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src import paths
from src.data.splits import read_split
from src.synth import library_v2 as L

REPO = paths.REPO_ROOT
CFG = yaml.safe_load(open(REPO / "configs" / "synth" / "library_v2.yaml"))
R = CFG["rules"]
QF = REPO / "configs" / "synth" / "library_v2_fill_quantiles.yaml"
DATA = paths.get("data")
OUT = DATA / "synth" / "cutouts_v2"
TRAIN_IMG = DATA / "coco" / "train2017_subset"
SEED = CFG["seed"]


def h(s) -> str:
    return hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest()


def load_coco(split):
    d = json.load(open(DATA / "coco" / "annotations" / f"instances_{split}2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    return {im["id"]: im for im in d["images"]}, cats, by_img


def gqa_coco_ids() -> set[int]:
    vg = {x["image_id"]: x["coco_id"] for x in json.load(open(DATA / "vg_meta" / "image_data.json"))}
    from src.data import load_split
    out = set()
    for n in ("gqa_spatial_one", "gqa_spatial_two"):
        for side in ("dev", "test"):
            for s in load_split(n, side):
                c = vg[int(Path(s.image_path).stem)]
                if c is not None:
                    out.add(int(c))
    return out


def stage_select():
    OUT.mkdir(parents=True, exist_ok=True)
    excl = CFG["exclude_categories"]
    gqa = gqa_coco_ids()
    hold = set(read_split("coco_holdout")["image_ids"])
    imv2 = read_split("coco_images_v2")
    val_side = {i: s for s in ("dev", "test") for i in imv2[s] + imv2["cutout_source_images_outside_union"][s]}
    funnel = {src: Counter() for src in ("train", "val")}
    pool, rows = defaultdict(list), []
    for src in ("train", "val"):
        imgs, cats, by_img = load_coco(src)
        coco80 = [cats[k] for k in sorted(cats)]
        for iid, anns in by_img.items():
            im = imgs[iid]
            for a in anns:
                c = cats[a["category_id"]]
                f = funnel[src]
                f["0_instances"] += 1
                if c in excl:
                    f["g_category_excluded"] += 1; continue
                if a.get("iscrowd", 0):
                    f["a_iscrowd"] += 1; continue
                if not L.rule_a(a, anns):
                    f["a_bbox_overlap"] += 1; continue
                if not L.rule_b(a, im["width"], im["height"], R["b_min_border_px"]):
                    f["b_border"] += 1; continue
                if src == "train":
                    pool[c].append(L.fill_ratio(a))
                rows.append({"src": src, "image_id": iid, "ann_id": a["id"], "category": c, "bbox": a["bbox"], "area": a["area"], "file_name": im["file_name"],
                             "width": im["width"], "height": im["height"], "fill": L.fill_ratio(a)})
    q = L.fill_quantiles(pool, tuple(R["c_fill_quantiles"]))
    yaml.safe_dump({"note": "由 scripts/build_synth_library_v2.py select 在 a、b 过滤后的 train2017 候选池上逐类计算（numpy 线性插值）；规则 c 用它", "quantiles": q,
                    "n_pool": {c: len(v) for c, v in pool.items()}}, open(QF, "w"), allow_unicode=True, sort_keys=True)
    keep = []
    for r in rows:
        f = funnel[r["src"]]
        if not L.rule_c(r, r["category"], q):
            f["c_fill_quantile"] += 1; continue
        w, hh = r["bbox"][2], r["bbox"][3]
        d = L.rule_d(max(w, hh), min(w, hh), r["area"], R)          # 预筛：用标注框与面积；cut 阶段用最大连通块复核
        if not d["long_side"]:
            f["d_long_side(pre)"] += 1; continue
        if not d["upscale"]:
            f["d_upscale(pre)"] += 1; continue
        if not d["aspect"]:
            f["d_aspect(pre)"] += 1; continue
        if r["src"] == "train" and r["image_id"] in gqa:
            f["leak_gqa_spatial_coco_id"] += 1; continue
        if r["src"] == "val":
            if r["image_id"] in hold:
                f["val_in_holdout"] += 1; continue
            if r["image_id"] not in val_side:
                f["val_image_without_existing_side"] += 1; continue
            r["side"] = val_side[r["image_id"]]
        keep.append(r)
    by_c = defaultdict(list)
    for r in keep:
        if r["src"] == "train":
            by_c[r["category"]].append(r)
    capped = [r for r in keep if r["src"] == "val"]
    for c, v in by_c.items():
        v = sorted(v, key=lambda r: h(f"cap:{r['ann_id']}"))
        capped += v[:CFG["train_cap_per_category"]]
        funnel["train"]["cap_per_category_dropped"] += max(0, len(v) - CFG["train_cap_per_category"])
    with open(OUT / "candidates.jsonl", "w") as fo:
        fo.writelines(json.dumps(r) + "\n" for r in capped)
    files = sorted({r["file_name"] for r in capped if r["src"] == "train"})
    (OUT / "download_list.txt").write_text("".join(f"http://images.cocodataset.org/train2017/{f}\n" for f in files))
    summ = {"funnel": {k: dict(v) for k, v in funnel.items()}, "n_candidates": dict(Counter(r["src"] for r in capped)),
            "n_train_images_to_download": len(files), "n_gqa_coco_ids": len(gqa), "per_category_candidates": {c: dict(Counter(r["src"] for r in capped if r["category"] == c)) for c in sorted({r["category"] for r in capped})}}
    json.dump(summ, open(OUT / "select_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_category_candidates"}, ensure_ascii=False))


def _image_path(r):
    return (TRAIN_IMG / r["file_name"]) if r["src"] == "train" else (DATA / "coco" / "val2017" / r["file_name"])


def phash_img(path, size=64):
    from PIL import Image
    from scipy.fft import dctn
    g = np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), float)
    c = dctn(g, norm="ortho")[:8, :8]
    return (c > np.median(c)).ravel()


def stage_cut():
    from PIL import Image
    from src.synth import cutouts as cu
    from src.data import load_split
    from src.data.coco_pairs import load_coco_pairs
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    # 评测图像（整图感知哈希）：What's Up A/B、COCO-spatial、COCO-Pairs、留出池、SpatialRGPT
    evals = {}
    for n in ("whatsup_a", "whatsup_b", "coco_spatial_one", "coco_spatial_two"):
        for side in ("dev", "test"):
            for s in load_split(n, side):
                evals[s.image_path] = n
    for side in ("dev", "test"):
        for s in load_coco_pairs(side):
            evals[s.image_path] = "coco_pairs"
    for i in read_split("coco_holdout")["image_ids"]:
        evals[str(DATA / "coco" / "val2017" / f"{int(i):012d}.jpg")] = "coco_holdout"
    from src.data.spatialrgpt import load_records
    for r in load_records():
        evals[r.image_path] = "spatialrgpt"
    ek = sorted(evals)
    EH = np.stack([phash_img(p) for p in ek])
    th = CFG["leakage"]["phash_max_hamming"]
    _, cats_t, by_t = load_coco("train")
    _, cats_v, by_v = load_coco("val")
    anns = {**{a["id"]: a for v in by_t.values() for a in v}, **{a["id"]: a for v in by_v.values() for a in v}}
    out, ph_cache = [], {}
    for r in cands:
        p = _image_path(r)
        if not p.is_file():
            out.append({**r, "cut_status": "image_missing"}); continue
        if str(p) not in ph_cache:
            hp = phash_img(p)
            d = (EH != hp).sum(1)
            if str(p) in evals:                 # val 候选：它所在的图本身就是评测图像（v1 起的设计，划分沿用该图），比对时排除它自己
                d[ek.index(str(p))] = 64
            j = int(d.argmin())
            ph_cache[str(p)] = (int(d[j]), evals[ek[j]], ek[j])
        dmin, dset, dpath = ph_cache[str(p)]
        rec = {**r, "phash_min_dist_to_eval": dmin, "phash_nearest_eval_set": dset}
        if r["src"] == "train" and dmin <= th:   # 泄漏比对只针对 train 来源；val 候选另有规则（不在留出池、沿用原划分）
            out.append({**rec, "cut_status": "leak_phash", "phash_nearest_eval_image": Path(dpath).name}); continue
        img = np.asarray(Image.open(p).convert("RGB"))
        m = cu._coco_ann_mask(anns[r["ann_id"]], img.shape[0], img.shape[1])
        big, _, _ = cu.largest_component(m)
        x1, y1, x2, y2 = cu.mask_bbox(big)
        area = int(big.sum())
        d = L.rule_d(max(x2 - x1, y2 - y1), min(x2 - x1, y2 - y1), area, R)
        rec.update(lc_bbox=[x1, y1, x2, y2], lc_area=area, upscale=round(d["upscale_value"], 4), d_long_side=d["long_side"], d_upscale=d["upscale"], d_aspect=d["aspect"])
        if not (d["long_side"] and d["upscale"] and d["aspect"]):
            out.append({**rec, "cut_status": "d_after_largest_component"}); continue
        cid = f"{r['src']}_{r['ann_id']}"
        rgba = cu.make_cutout(img, big, 10 ** 6)
        png = cu.png_bytes(rgba)
        (OUT / "png" / f"{cid}.png").write_bytes(png)
        out.append({**rec, "cut_status": "ok", "cutout_id": cid, "png_sha256": hashlib.sha256(png).hexdigest()})
    with open(OUT / "cut.jsonl", "w") as fo:
        fo.writelines(json.dumps(r) + "\n" for r in out)
    print(Counter(r["cut_status"] for r in out), "eval images hashed", len(ek))


def on_gray(rgba, margin):
    from PIL import Image
    a = rgba.astype(np.uint16)
    hh, w = a.shape[:2]
    S = int(math.ceil(max(hh, w) * (1 + 2 * margin)))
    bg = np.full((S, S, 3), CFG["background_gray"], np.uint16)
    y, x = (S - hh) // 2, (S - w) // 2
    al = a[..., 3:4]
    bg[y:y + hh, x:x + w] = (a[..., :3] * al + bg[y:y + hh, x:x + w] * (255 - al) + 127) // 255
    return Image.fromarray(bg.astype(np.uint8))


def render_small(cid, lc_area):
    from PIL import Image
    from src.synth import cutouts as cu
    rgba = np.asarray(Image.open(OUT / "png" / f"{cid}.png").convert("RGBA"))
    s = L.small_tier_scale(lc_area, CFG["small_tier_check"]["rendered_px"])
    hh, w = rgba.shape[:2]
    small = cu._resize_once(rgba, max(1, round(max(hh, w) * s)))
    return on_gray(small, CFG["margin_frac"])


def stage_clip():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(8)
    rows = [json.loads(l) for l in open(OUT / "cut.jsonl")]
    ok = [r for r in rows if r["cut_status"] == "ok"]
    _, cats, _ = load_coco("val")
    labels = L.clip_labels([cats[k] for k in sorted(cats)], CFG["clip"]["confusable_labels"])
    art = lambda n: "an" if n[0].lower() in "aeiou" else "a"      # noqa: E731
    texts = [CFG["clip"]["prompt"].format(article=art(n), name=n) for n in labels]
    path = paths.get("new_models") / CFG["clip"]["model"]
    proc, clip = CLIPProcessor.from_pretrained(path), CLIPModel.from_pretrained(path).eval()
    res = []
    fo = open(OUT / "clip.jsonl.part", "w")                   # 每批算完就写盘，出错时不丢已算的结果
    with torch.no_grad():
        tin = proc(text=texts, return_tensors="pt", padding=True)
        for i in range(0, len(ok), 32):
            b = ok[i:i + 32]
            ims_n = [on_gray(np.asarray(Image.open(OUT / "png" / f"{r['cutout_id']}.png").convert("RGBA")), CFG["margin_frac"]) for r in b]
            ims_s = [render_small(r["cutout_id"], r["lc_area"]) for r in b]
            pn = clip(**tin, **proc(images=ims_n, return_tensors="pt")).logits_per_image.softmax(-1).numpy()
            ps = clip(**tin, **proc(images=ims_s, return_tensors="pt")).logits_per_image.softmax(-1).numpy()
            for r, a, s in zip(b, pn, ps):
                res.append({"cutout_id": r["cutout_id"], "normal": L.clip_decision(a, labels, r["category"], CFG["clip"]["min_prob"]),
                            "small18": L.clip_decision(s, labels, r["category"], CFG["clip"]["min_prob"])})
                fo.write(json.dumps(res[-1]) + "\n")
            fo.flush()
            if i % 640 == 0:
                print(i, len(ok), flush=True)
    fo.close()
    (OUT / "clip.jsonl.part").rename(OUT / "clip.jsonl")
    print("clip done", len(res), "labels", len(labels))


def stage_finalize():
    rows = {r.get("cutout_id") or f"x{r['ann_id']}": r for r in map(json.loads, open(OUT / "cut.jsonl"))}
    clip = {r["cutout_id"]: r for r in map(json.loads, open(OUT / "clip.jsonl"))}
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    # train 按图像固定种子分 dev/test；val 继承
    timgs = sorted({r["image_id"] for r in cands if r["src"] == "train"})
    order = sorted(timgs, key=lambda i: h(f"train_split:{i}"))
    tside = {i: ("dev" if k < len(order) // 2 else "test") for k, i in enumerate(order)}
    lib, per = [], defaultdict(lambda: defaultdict(Counter))
    for cid, r in rows.items():
        side = r.get("side") or tside.get(r["image_id"])
        c = r["category"]
        per[c][side]["candidates"] += 1
        if r["cut_status"] != "ok":
            per[c][side][r["cut_status"]] += 1; continue
        k = clip[cid]
        if not k["normal"]["pass"]:
            per[c][side]["e_clip"] += 1; continue
        per[c][side]["kept"] += 1
        lib.append({**r, "split": side, "clip_prob_true": round(k["normal"]["prob_true"], 4), "clip_top1": k["normal"]["top1"],
                    "small36_ok": bool(k["small18"]["pass"]), "clip_small18_prob_true": round(k["small18"]["prob_true"], 4), "clip_small18_top1": k["small18"]["top1"]})
    usable = sorted(c for c in per if min(per[c]["dev"]["kept"], per[c]["test"]["kept"]) >= CFG["min_per_side"])
    final = [r for r in lib if r["category"] in usable]
    raw = "".join(json.dumps(r) + "\n" for r in sorted(final, key=lambda r: r["cutout_id"])).encode()
    (OUT / "library.jsonl").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    json.dump({"dataset": "synth_library_v2", "status": "active", "unit": "image（train 按图像固定种子；val 继承 coco_images_v2）", "library_sha256": sha,
               "n_total": len(final), "dev": sorted(r["cutout_id"] for r in final if r["split"] == "dev"), "test": sorted(r["cutout_id"] for r in final if r["split"] == "test"),
               "train_image_side": {str(i): s for i, s in tside.items()}}, open(REPO / "splits" / "synth_library_v2.json", "w"), indent=0)
    small_cats = sorted(c for c in usable if min(sum(r["small36_ok"] for r in final if r["category"] == c and r["split"] == s) for s in ("dev", "test")) >= CFG["min_per_side"])
    man = {"config": CFG, "library_sha256": sha, "n_final": len(final), "n_final_by_side": dict(Counter(r["split"] for r in final)), "n_final_by_src": dict(Counter(r["src"] for r in final)),
           "n_categories_usable": len(usable), "meets_min_categories": len(usable) >= CFG["min_categories"], "categories_usable": usable,
           "n_small36_ok": sum(r["small36_ok"] for r in final), "categories_usable_small36": small_cats,
           "per_category": {c: {s: dict(per[c][s]) for s in ("dev", "test")} for c in sorted(per)},
           "removed_by_rule_after_select": dict(Counter(k for c in per for s in per[c] for k, v in per[c][s].items() for _ in range(v) if k not in ("candidates", "kept"))),
           "select_summary": json.load(open(OUT / "select_summary.json"))}
    json.dump(man, open(REPO / "manifests" / "synth_library_v2.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in man.items() if k not in ("config", "per_category", "select_summary", "categories_usable", "categories_usable_small36")}, ensure_ascii=False))
    sheets(final)


def sheets(final):
    import csv
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    rev = paths.get("outputs") / "synth_review" / "library_v2"
    rev.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    T, COLS = 180, 12
    index, page = [], 0
    by = defaultdict(list)
    for r in final:
        by[r["category"]].append(r)

    def draw(title, items, fn):
        rows_ = (len(items) + COLS - 1) // COLS
        im = Image.new("RGB", (COLS * (T + 6) + 6, 26 + rows_ * (T + 40)), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((6, 5), title, fill=(0, 0, 0), font=font)
        for k, r in enumerate(items):
            y, x = divmod(k, COLS)
            X, Y = 6 + x * (T + 6), 26 + y * (T + 40)
            g = on_gray(np.asarray(Image.open(OUT / "png" / f"{r['cutout_id']}.png").convert("RGBA")), CFG["margin_frac"])
            im.paste(g.resize((T, T), Image.LANCZOS), (X, Y))
            d.text((X, Y + T + 1), r["cutout_id"], fill=(0, 0, 0), font=font)
            d.text((X, Y + T + 17), f"{r['split']} {r['clip_prob_true']:.3f}{'' if r['small36_ok'] else ' no36'}", fill=(0, 0, 160) if r["split"] == "dev" else (160, 0, 0), font=font)
            index.append([fn, y + 1, x + 1, r["cutout_id"], r["category"], r["split"], r["src"], r["clip_prob_true"], r["small36_ok"]])
        im.save(rev / fn, quality=90)

    for c in sorted(by):
        page += 1
        pick = [r for s in ("dev", "test") for r in sorted((r for r in by[c] if r["split"] == s), key=lambda r: h(f"sheet:{r['cutout_id']}"))[:CFG["review"]["per_category_per_side"]]]
        draw(f"{c}: {sum(r['split'] == 'dev' for r in by[c])} dev / {sum(r['split'] == 'test' for r in by[c])} test; up to 12 per side (seeded); number = CLIP prob (normal); no36 = fails 18-px check",
             pick, f"page_{page:03d}_{c.replace(' ', '-')}.jpg")
    low = [r for c in sorted(by) for r in sorted(by[c], key=lambda r: r["clip_prob_true"])[:CFG["review"]["lowest_clip_per_category"]]]
    for k in range(0, len(low), 60):
        draw("lowest CLIP prob per category (3 each)", low[k:k + 60], f"lowest_clip_{k // 60:02d}.jpg")
    with open(rev / "index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category", "side", "source", "clip_prob", "small36_ok"]); w.writerows(index)
    print("sheets", page, "pages +", (len(low) + 59) // 60, "lowest pages")


if __name__ == "__main__":
    {"select": stage_select, "cut": stage_cut, "clip": stage_clip, "finalize": stage_finalize}[sys.argv[1]]()
