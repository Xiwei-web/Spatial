"""合成场景抠图库 v1：从原图重裁 → 几何过滤 → CLIP 可识别性过滤 → 按划分分成 dev / test 两个库 → 每类两侧各 >= 3。
输出：<data>/synth/cutouts_v1/{png/*.png, library.jsonl（只含最终可用）, filter_report.jsonl（全部 1,333 条的过滤记录）}
      splits/synth_library_v1.json、manifests/synth_library_v1.json、<outputs>/synth_review/library_v1/sheet_*.jpg（每类最多 6 个）
用法（计算节点）：source scripts/env.sh tf5 && python scripts/build_synth_library_v1.py"""
import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from src import paths
from src.data.coco_pairs import coco_image_path, load_coco_instances
from src.data.splits import read_split
from src.synth import cutouts as cu

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "synth" / "library_v1.yaml"))
OUT = paths.get("data") / "synth" / "cutouts_v1"
(OUT / "png").mkdir(parents=True, exist_ok=True)
raw = (cu.library_dir() / "library.jsonl").read_bytes()
src = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
sp = read_split("synth_cutouts")
assert sp["library_sha256"] == hashlib.sha256(raw).hexdigest()
side = {i: x for x in ("dev", "test") for i in sp[x]}
coco = load_coco_instances()
names80 = [coco["cats"][k] for k in sorted(coco["cats"])]
assert len(names80) == 80

# ---- 1. 重裁 ----
rows = []
for e in sorted(src, key=lambda e: e["cutout_id"]):
    im = coco["images"][e["src"]["image_id"]]
    m = cu._coco_ann_mask(coco["anns"][e["src"]["ann_id"]], im["height"], im["width"])
    big, _, _ = cu.largest_component(m)
    rgba = cu.make_cutout(np.asarray(Image.open(coco_image_path(im["file_name"])).convert("RGB")), big, 10**6)
    png = cu.png_bytes(rgba)
    (OUT / "png" / f"{e['cutout_id']}.png").write_bytes(png)
    h, w = rgba.shape[:2]
    area = int((rgba[..., 3] >= cu.ALPHA_THR).sum())
    rows.append({"cutout_id": e["cutout_id"], "source": e["source"], "category": e["category"], "split": side[e["cutout_id"]], "src": e["src"],
                 "width": w, "height": h, "alpha_area": area, "long_side": max(w, h), "aspect": round(max(w, h) / min(w, h), 4),
                 "upscale_at_max_size": round(CFG["max_sqrt_area_px"] / math.sqrt(area), 4), "png_sha256": hashlib.sha256(png).hexdigest()})
print("cut", len(rows), flush=True)

# ---- 2. CLIP（对全部 1,333 个都算，便于报告各条过滤各自的影响）----
from transformers import CLIPModel, CLIPProcessor                        # noqa: E402

torch.set_num_threads(8)
path = paths.get("new_models") / CFG["clip"]["model"]
proc, clip = CLIPProcessor.from_pretrained(path), CLIPModel.from_pretrained(path).eval()
art = lambda n: "an" if n[0] in "aeiou" else "a"                        # noqa: E731
texts = [CFG["clip"]["prompt"].format(article=art(n), name=n) for n in names80]


def on_gray(cid):
    a = np.asarray(Image.open(OUT / "png" / f"{cid}.png").convert("RGBA")).astype(np.uint16)
    h, w = a.shape[:2]
    S = int(math.ceil(max(h, w) * (1 + 2 * CFG["margin_frac"])))
    bg = np.full((S, S, 3), CFG["background_gray"], np.uint16)
    y, x = (S - h) // 2, (S - w) // 2
    al = a[..., 3:4]
    bg[y:y + h, x:x + w] = (a[..., :3] * al + bg[y:y + h, x:x + w] * (255 - al) + 127) // 255
    return Image.fromarray(bg.astype(np.uint8))


with torch.no_grad():
    tin = proc(text=texts, return_tensors="pt", padding=True)
    for i in range(0, len(rows), 32):
        b = rows[i:i + 32]
        iin = proc(images=[on_gray(r["cutout_id"]) for r in b], return_tensors="pt")
        lg = clip(**tin, **iin).logits_per_image
        p = lg.softmax(-1)
        for r, q in zip(b, p):
            k = int(q.argmax())
            r["clip_top1"], r["clip_top1_prob"], r["clip_prob_true"] = names80[k], round(float(q[k]), 4), round(float(q[names80.index(r["category"])]), 4)
        if i % 320 == 0:
            print(i, flush=True)
for r in rows:
    r["checks"] = {"long_side": r["long_side"] >= CFG["min_long_side_px"], "upscale": r["upscale_at_max_size"] <= CFG["max_upscale"], "aspect": r["aspect"] <= CFG["max_aspect"],
                   "clip": r["clip_top1"] == r["category"] and r["clip_prob_true"] >= CFG["clip"]["min_prob"]}
    r["pass_all"] = all(r["checks"].values())
cats = sorted({r["category"] for r in rows})
per = {c: {"before": {x: sum(r["category"] == c and r["split"] == x for r in rows) for x in ("dev", "test")},
           "after_geometry": {x: sum(r["category"] == c and r["split"] == x and r["checks"]["long_side"] and r["checks"]["upscale"] and r["checks"]["aspect"] for r in rows) for x in ("dev", "test")},
           "after_all": {x: sum(r["category"] == c and r["split"] == x and r["pass_all"] for r in rows) for x in ("dev", "test")}} for c in cats}
usable = [c for c in cats if min(per[c]["after_all"].values()) >= CFG["min_per_side"]]
excluded = {c: per[c]["after_all"] for c in cats if c not in usable}
final = [r for r in rows if r["pass_all"] and r["category"] in usable]
for r in rows:
    r["in_library"] = r["pass_all"] and r["category"] in usable
with open(OUT / "filter_report.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
lib = "".join(json.dumps(r) + "\n" for r in final).encode()
(OUT / "library.jsonl").write_bytes(lib)
sha = hashlib.sha256(lib).hexdigest()
json.dump({"dataset": "synth_library_v1", "status": "active", "unit": "image (inherited from synth_cutouts_v2 / coco_images_v2)", "n_total": len(final),
           "n_dev": sum(r["split"] == "dev" for r in final), "n_test": sum(r["split"] == "test" for r in final), "n_categories": len(usable),
           "per_category": {c: [per[c]["after_all"]["dev"], per[c]["after_all"]["test"]] for c in usable}, "library_sha256": sha,
           "dev": sorted(r["cutout_id"] for r in final if r["split"] == "dev"), "test": sorted(r["cutout_id"] for r in final if r["split"] == "test")},
          open(paths.REPO_ROOT / "splits" / "synth_library_v1.json", "w"), indent=0)
man = {"config": CFG, "source_library_sha256": sp["library_sha256"], "library_sha256": sha, "clip_texts_example": texts[:3],
       "n_before": len(rows), "n_fail_each_check": {k: sum(not r["checks"][k] for r in rows) for k in ("long_side", "upscale", "aspect", "clip")},
       "n_pass_geometry": sum(r["checks"]["long_side"] and r["checks"]["upscale"] and r["checks"]["aspect"] for r in rows), "n_pass_all": sum(r["pass_all"] for r in rows),
       "n_final": len(final), "n_final_by_side": dict(Counter(r["split"] for r in final)), "n_categories_before": len(cats), "n_categories_usable": len(usable),
       "categories_excluded": excluded, "per_category": per,
       "clip_confusions_top": Counter(f"{r['category']} -> {r['clip_top1']}" for r in rows if not r["checks"]["clip"] and r["clip_top1"] != r["category"]).most_common(25),
       "max_upscale_final": max(r["upscale_at_max_size"] for r in final), "max_aspect_final": max(r["aspect"] for r in final)}
json.dump(man, open(paths.REPO_ROOT / "manifests" / "synth_library_v1.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in man.items() if k not in ("per_category", "config")}, ensure_ascii=False))

# ---- 3. 联系表：每类最多 6 个（dev、test 各最多 3 个，按 cutout_id 固定顺序）----
from matplotlib import font_manager                                     # noqa: E402

font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
rev = paths.get("outputs") / "synth_review" / "library_v1"
rev.mkdir(parents=True, exist_ok=True)
T, LAB, per_sheet = 150, 190, 12
for s in range(0, len(usable), per_sheet):
    chunk = usable[s:s + per_sheet]
    im = Image.new("RGB", (LAB + 6 * (T + 6), 24 + len(chunk) * (T + 26)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((6, 4), "library_v1 kept cutouts on gray; up to 3 dev + 3 test per category; number = CLIP prob of the true class", fill=(0, 0, 0), font=font)
    for k, c in enumerate(chunk):
        y = 24 + k * (T + 26)
        d.multiline_text((6, y + 40), f"{c}\ndev {per[c]['after_all']['dev']}  test {per[c]['after_all']['test']}\n(before {sum(per[c]['before'].values())})", fill=(0, 0, 0), font=font)
        pick = [r for x in ("dev", "test") for r in sorted((r for r in final if r["category"] == c and r["split"] == x), key=lambda r: hashlib.sha256(r["cutout_id"].encode()).hexdigest())[:3]]
        for j, r in enumerate(pick):
            im.paste(on_gray(r["cutout_id"]).resize((T, T), Image.LANCZOS), (LAB + j * (T + 6), y))
            d.text((LAB + j * (T + 6), y + T + 2), f"{r['split']} {r['clip_prob_true']:.2f}", fill=(0, 0, 0), font=font)
    im.save(rev / f"sheet_{s // per_sheet:02d}.jpg", quality=90)
print("library_done", len(final), len(usable))
