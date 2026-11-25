"""library_v1 的全量联系表（923 个，只读）与近重复清单（只列不删）。
用法：python scripts/synth_library_full_sheets.py
输出 <outputs>/synth_review/library_v1_full/：page_XXX_<类别>.jpg（每类另起一页，每页 ≤ 60 格，灰底 128）、index.csv（page,row,col 从 1 计）、
     near_duplicates.csv（全部对）、near_duplicates_cross_split.csv（跨 dev/test）、near_dup_*.jpg（对照图）。
近重复：灰底 64x64 缩略图（灰度）→ 二维 DCT → 左上 8x8 共 64 位，按中位数二值化；对 (i, j) 取 d(原图 i, 原图 j) 与 d(原图 i, 镜像 j)；任一 ≤ 6 就列出（两者都写出）。"""
import csv
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn

from src import paths

CUT = paths.get("data") / "synth" / "cutouts_v1"
man = json.load(open(paths.REPO_ROOT / "manifests" / "synth_library_v1.json"))
raw = (CUT / "library.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == man["library_sha256"]
rows = [json.loads(l) for l in raw.decode().splitlines()]
assert len(rows) == 923
rows.sort(key=lambda r: (r["category"], r["split"], r["cutout_id"]))
OUT = paths.get("outputs") / "synth_review" / "library_v1_full"
OUT.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
GRAY, T, LAB, COLS, ROWS, THR = 128, 200, 36, 10, 6, 6


def on_gray(cid, margin=0.1):
    a = np.asarray(Image.open(CUT / "png" / f"{cid}.png").convert("RGBA")).astype(np.uint16)
    h, w = a.shape[:2]
    S = int(np.ceil(max(h, w) * (1 + 2 * margin)))
    bg = np.full((S, S, 3), GRAY, np.uint16)
    y, x = (S - h) // 2, (S - w) // 2
    al = a[..., 3:4]
    bg[y:y + h, x:x + w] = (a[..., :3] * al + bg[y:y + h, x:x + w] * (255 - al) + 127) // 255
    return Image.fromarray(bg.astype(np.uint8))


def phash(im):
    g = np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), float)
    c = dctn(g, norm="ortho")[:8, :8]
    return (c > np.median(c)).ravel()


imgs = {r["cutout_id"]: on_gray(r["cutout_id"]) for r in rows}
# ---- 联系表 ----
by = defaultdict(list)
for r in rows:
    by[r["category"]].append(r)
index, page = [], 0
for c in sorted(by):
    for s in range(0, len(by[c]), COLS * ROWS):
        part = by[c][s:s + COLS * ROWS]
        page += 1
        nr = (len(part) + COLS - 1) // COLS
        im = Image.new("RGB", (COLS * (T + 6) + 6, 26 + nr * (T + LAB + 6)), (255, 255, 255))
        d = ImageDraw.Draw(im)
        fn = f"page_{page:03d}_{c.replace(' ', '-')}.jpg"
        d.text((6, 5), f"{fn}   {c}: {sum(r['split'] == 'dev' for r in by[c])} dev / {sum(r['split'] == 'test' for r in by[c])} test; cells {s + 1}-{s + len(part)} of {len(by[c])}; number = CLIP prob of the true class", fill=(0, 0, 0), font=font)
        for k, r in enumerate(part):
            y, x = divmod(k, COLS)
            X, Y = 6 + x * (T + 6), 26 + y * (T + LAB + 6)
            im.paste(imgs[r["cutout_id"]].resize((T, T), Image.LANCZOS), (X, Y))
            d.text((X, Y + T + 1), r["cutout_id"], fill=(0, 0, 0), font=font)
            d.text((X, Y + T + 17), f"{r['category']} {r['split']} {r['clip_prob_true']:.4f}", fill=(0, 0, 160) if r["split"] == "dev" else (160, 0, 0), font=font)
            index.append([fn, y + 1, x + 1, r["cutout_id"], r["category"], r["split"], r["clip_prob_true"]])
        assert max(im.size) <= 2400 and len(part) <= 60
        im.save(OUT / fn, quality=90)
assert len(index) == 923 == len({i[3] for i in index})
with open(OUT / "index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category", "side", "clip_prob"]); w.writerows(index)
# ---- 近重复 ----
H = np.stack([phash(imgs[r["cutout_id"]]) for r in rows])
M = np.stack([phash(imgs[r["cutout_id"]].transpose(Image.FLIP_LEFT_RIGHT)) for r in rows])
d0 = (H[:, None, :] != H[None, :, :]).sum(-1)
d1 = (H[:, None, :] != M[None, :, :]).sum(-1)
pairs = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if min(d0[i, j], d1[i, j]) <= THR:
            a, b = rows[i], rows[j]
            pairs.append({"cutout_a": a["cutout_id"], "cutout_b": b["cutout_id"], "category_a": a["category"], "category_b": b["category"], "side_a": a["split"], "side_b": b["split"],
                          "image_a": a["src"]["image_id"], "image_b": b["src"]["image_id"], "d_orig": int(d0[i, j]), "d_mirror": int(d1[i, j]),
                          "kind": "mirror" if d1[i, j] < d0[i, j] else "orig", "cross_split": a["split"] != b["split"], "same_category": a["category"] == b["category"]})
pairs.sort(key=lambda p: (min(p["d_orig"], p["d_mirror"]), p["cutout_a"], p["cutout_b"]))
cols = list(pairs[0]) if pairs else ["cutout_a"]
for fn, sub in (("near_duplicates.csv", pairs), ("near_duplicates_cross_split.csv", [p for p in pairs if p["cross_split"]])):
    with open(OUT / fn, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(sub)
for s in range(0, len(pairs), 30):
    part = pairs[s:s + 30]
    im = Image.new("RGB", (5 * 420 + 6, 6 + ((len(part) + 4) // 5) * 250), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for k, p in enumerate(part):
        y, x = divmod(k, 5)
        X, Y = 6 + x * 420, 6 + y * 250
        for q, c in enumerate((p["cutout_a"], p["cutout_b"])):
            im.paste(imgs[c].resize((200, 200), Image.LANCZOS), (X + q * 203, Y))
        d.text((X, Y + 201), f"{p['cutout_a']} ({p['side_a']}) | {p['cutout_b']} ({p['side_b']})", fill=(160, 0, 0) if p["cross_split"] else (0, 0, 0), font=font)
        d.text((X, Y + 218), f"{p['category_a']} | {p['category_b']}  d_orig={p['d_orig']} d_mirror={p['d_mirror']}", fill=(0, 0, 0), font=font)
    im.save(OUT / f"near_dup_{s // 30:02d}.jpg", quality=90)
allmin = np.minimum(d0, d1)[np.triu_indices(len(rows), 1)]
summ = {"library_sha256": man["library_sha256"], "n_cutouts": len(rows), "n_pages": page, "n_index_rows": len(index), "threshold": THR, "n_pairs_total": int(len(rows) * (len(rows) - 1) / 2), "n_pairs_listed": len(pairs),
        "by_kind": dict(Counter(p["kind"] for p in pairs)), "n_cross_split": sum(p["cross_split"] for p in pairs), "n_cross_split_same_category": sum(p["cross_split"] and p["same_category"] for p in pairs),
        "n_same_category": sum(p["same_category"] for p in pairs), "n_same_source_image": sum(p["image_a"] == p["image_b"] for p in pairs),
        "distance_histogram": dict(sorted(Counter(min(p["d_orig"], p["d_mirror"]) for p in pairs).items())), "n_distinct_cutouts_in_pairs": len({c for p in pairs for c in (p["cutout_a"], p["cutout_b"])}),
        "min_distance_quantiles_all_pairs": {str(q): float(np.quantile(allmin, q)) for q in (0, 0.001, 0.01, 0.5)}}
json.dump(summ, open(paths.REPO_ROOT / "manifests" / "synth_library_v1_near_duplicates.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, ensure_ascii=False))
