"""抠图库 v2 的全量联系表（全部抠图，只读；不做任何剔除）。格式与 build_synth_library_v2.py 的联系表相同：每类另起一页，每页 ≤ 60 格（12 列），
每格标 cutout_id、dev/test、CLIP 概率，“no36” 表示没通过 18 px 检查。输出 <outputs>/synth_review/library_v2_full/{page_*.jpg, index.csv}。
用法（CPU 节点）：python scripts/synth_library_v2_full_sheets.py"""
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from scripts.build_synth_library_v2 import CFG, OUT, on_gray
from src import paths

man = json.load(open(paths.REPO_ROOT / "manifests" / "synth_library_v2.json"))
raw = (OUT / "library.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == man["library_sha256"]
lib = [json.loads(l) for l in raw.decode().splitlines()]
assert len(lib) == man["n_final"]
rev = paths.get("outputs") / "synth_review" / "library_v2_full"
rev.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
T, COLS, PER = 180, 12, 60
by = defaultdict(list)
for r in lib:
    by[r["category"]].append(r)
index, page = [], 0
for c in sorted(by):
    items = sorted(by[c], key=lambda r: (r["split"], r["cutout_id"]))
    for s in range(0, len(items), PER):
        part = items[s:s + PER]
        page += 1
        fn = f"page_{page:03d}_{c.replace(' ', '-')}.jpg"
        rows = (len(part) + COLS - 1) // COLS
        im = Image.new("RGB", (COLS * (T + 6) + 6, 26 + rows * (T + 40)), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((6, 5), f"{fn}  {c}: {sum(r['split'] == 'dev' for r in items)} dev / {sum(r['split'] == 'test' for r in items)} test; cells {s + 1}-{s + len(part)} of {len(items)}; "
               "number = CLIP prob (normal); no36 = fails 18-px check", fill=(0, 0, 0), font=font)
        for k, r in enumerate(part):
            y, x = divmod(k, COLS)
            X, Y = 6 + x * (T + 6), 26 + y * (T + 40)
            g = on_gray(np.asarray(Image.open(OUT / "png" / f"{r['cutout_id']}.png").convert("RGBA")), CFG["margin_frac"])
            im.paste(g.resize((T, T), Image.LANCZOS), (X, Y))
            d.text((X, Y + T + 1), r["cutout_id"], fill=(0, 0, 0), font=font)
            d.text((X, Y + T + 17), f"{r['split']} {r['clip_prob_true']:.3f}{'' if r['small36_ok'] else ' no36'}", fill=(0, 0, 160) if r["split"] == "dev" else (160, 0, 0), font=font)
            index.append([fn, y + 1, x + 1, r["cutout_id"], r["category"], r["split"], r["src"], r["clip_prob_true"], r["small36_ok"]])
        assert max(im.size) <= 2400
        im.save(rev / fn, quality=90)
assert len(index) == len(lib) == len({i[3] for i in index})
with open(rev / "index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category", "side", "source", "clip_prob", "small36_ok"]); w.writerows(index)
print("pages", page, "cells", len(index))
