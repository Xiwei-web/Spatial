"""抠图库 v3 定稿第二轮（planner 2026-11-26）：v3 草稿 − 第一轮定稿剔除（status=excluded；restored 的 2 个恢复）− 第二轮按类复核剔除（85 个）；
两侧达不到各 >= 3 的类别整类排除。旧文件按版本保留：第一轮定稿 library_final.jsonl 不动（= r1），第二轮写 library_final_r2.jsonl；
manifests/synth_library_v3.json 的第一轮内容另存为 manifests/synth_library_v3_r1.json；library_v3_full 联系表先把第一轮的页面复制到 library_v3_full_r1/，再按第二轮重出。CPU。"""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from scripts.build_synth_library_v2 import CFG as C2, on_gray
from src import paths

REPO = paths.REPO_ROOT
V3 = paths.get("data") / "synth" / "cutouts_v3"
man = json.load(open(REPO / "manifests" / "synth_library_v3.json"))
assert man.get("status") == "final" and man["library_file"] == "library_final.jsonl"
r1_file = REPO / "manifests" / "synth_library_v3_r1.json"
if not r1_file.exists():
    shutil.copyfile(REPO / "manifests" / "synth_library_v3.json", r1_file)
raw = (V3 / "library.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == man["draft"]["library_sha256"]
draft = [json.loads(l) for l in raw.decode().splitlines()]
rd = lambda p: list(csv.DictReader(l for l in open(REPO / "annotations" / "synth" / p) if not l.startswith("#")))      # noqa: E731
r1 = rd("library_v3_final_exclusions.csv")
r2 = rd("library_v3_round2_exclusions.csv")
ex1 = {r["cutout_id"] for r in r1 if r["status"] == "excluded"}
restored = [r["cutout_id"] for r in r1 if r["status"] == "restored"]
ex2 = {r["cutout_id"] for r in r2}
assert len(ex1) == 21 and len(restored) == 2 and len(ex2) == 85 and not ex1 & ex2 and "train_586407" not in ex2
r1_final = {json.loads(l)["cutout_id"] for l in open(V3 / "library_final.jsonl")}
assert ex2 <= r1_final and all(c not in r1_final for c in restored)
keep = [r for r in draft if r["cutout_id"] not in ex1 | ex2]
per = defaultdict(Counter)
for r in keep:
    per[r["category"]][r["split"]] += 1
usable = sorted(c for c in per if min(per[c]["dev"], per[c]["test"]) >= C2["min_per_side"])
final = sorted([r for r in keep if r["category"] in usable], key=lambda r: r["cutout_id"])
rawf = "".join(json.dumps(r) + "\n" for r in final).encode()
(V3 / "library_final_r2.jsonl").write_bytes(rawf)
sha = hashlib.sha256(rawf).hexdigest()
small = sorted(c for c in usable if min(sum(r["small36_ok"] for r in final if r["category"] == c and r["split"] == s) for s in ("dev", "test")) >= C2["min_per_side"])
json.dump({"dataset": "synth_library_v3", "status": "final（round2）", "unit": "image（沿用 v2 的划分）", "library_file": "<data>/synth/cutouts_v3/library_final_r2.jsonl", "library_sha256": sha,
           "previous": {"round1_file": "<data>/synth/cutouts_v3/library_final.jsonl", "round1_sha256": man["library_sha256"]},
           "n_total": len(final), "dev": [r["cutout_id"] for r in final if r["split"] == "dev"], "test": [r["cutout_id"] for r in final if r["split"] == "test"]},
          open(REPO / "splits" / "synth_library_v3.json", "w"), indent=0)
r2c = Counter((r["category"], r["side"]) for r in r2)
man.update({
    "status": "final", "round": 2,
    "provenance": "自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；第一轮定稿按 Xiwei 人工抽查剔除 6 个、planner 的 cow 单类复核剔除 17 个（其中 2 个在第二轮恢复）；第二轮按类复核（planner 子代理放大 2 倍逐格看，模型预审，未经人工复核）剔除 85 个",
    "round1": {"library_file": "library_final.jsonl", "library_sha256": man["library_sha256"], "n": man["n_final"], "manifest_copy": "manifests/synth_library_v3_r1.json"},
    "round2": {"exclusions_file": "annotations/synth/library_v3_round2_exclusions.csv", "n_excluded": len(ex2), "restored": restored,
               "restored_reason": "按第二轮的宽标准不该剔除（Xiwei 决定）", "categories_reviewed": ["sheep", "horse", "zebra", "giraffe", "elephant", "bear", "dog", "cat", "bird", "bench", "fire hydrant", "cow"],
               "n_reviewed_cells": 523, "criterion": "只在缺失明显（没头、没腿、直线切口）或背景成片进入 mask 时剔除；蹄下、脚边少量的草不剔除",
               "excluded_by_category": {c: {"dev": r2c[(c, "dev")], "test": r2c[(c, "test")]} for c in sorted({r["category"] for r in r2})}, "kept_despite_subagent": ["train_586407（bear，Xiwei 明确保留）"]},
    "spotcheck_note": "人工抽查（150 个，坏 2.0%）是在两轮剔除之前、对 v3 草稿做的",
    "library_sha256": sha, "library_file": "library_final_r2.jsonl", "n_final": len(final), "n_final_by_side": dict(Counter(r["split"] for r in final)),
    "n_final_by_src": dict(Counter(r["src"] for r in final)), "n_categories_usable": len(usable), "categories_usable": usable,
    "categories_dropped_at_final": {c: dict(per[c]) for c in per if c not in usable}, "n_small36_ok": sum(r["small36_ok"] for r in final),
    "n_categories_small36": len(small), "categories_small36": small, "per_category_final": {c: dict(per[c]) for c in sorted(per)}})
json.dump(man, open(REPO / "manifests" / "synth_library_v3.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: man[k] for k in ("n_final", "n_final_by_side", "n_categories_usable", "categories_dropped_at_final", "n_small36_ok", "n_categories_small36")}, ensure_ascii=False))
print({c: dict(per[c]) for c in sorted({r["category"] for r in r2} | {"cow"})})
out = paths.get("outputs") / "synth_review" / "library_v3_full"
bak = paths.get("outputs") / "synth_review" / "library_v3_full_r1"
if not bak.exists():
    shutil.copytree(out, bak)
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
T, COLS, PER = 180, 12, 60
by = defaultdict(list)
for r in final:
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
        d.text((6, 5), f"FINAL (round2)  {fn}  {c}: {sum(r['split'] == 'dev' for r in items)} dev / {sum(r['split'] == 'test' for r in items)} test; cells {s + 1}-{s + len(part)} of {len(items)}; "
               "number = CLIP prob (normal); no36 = fails 18-px check", fill=(0, 0, 0), font=font)
        for k, r in enumerate(part):
            y, x = divmod(k, COLS)
            X, Y = 6 + x * (T + 6), 26 + y * (T + 40)
            g = on_gray(np.asarray(Image.open(V3 / "png" / f"{r['cutout_id']}.png").convert("RGBA")), C2["margin_frac"])
            im.paste(g.resize((T, T), Image.LANCZOS), (X, Y))
            d.text((X, Y + T + 1), r["cutout_id"], fill=(0, 0, 0), font=font)
            d.text((X, Y + T + 17), f"{r['split']} {r['clip_prob_true']:.3f}{'' if r['small36_ok'] else ' no36'}", fill=(0, 0, 160) if r["split"] == "dev" else (160, 0, 0), font=font)
            index.append([fn, y + 1, x + 1, r["cutout_id"], r["category"], r["split"], r["src"], r["clip_prob_true"], r["small36_ok"]])
        im.save(out / fn, quality=90)
with open(out / "index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category", "side", "source", "clip_prob", "small36_ok"]); w.writerows(index)
(out / "FINAL.txt").write_text(f"library_v3 final round2（{len(final)} 个，{len(usable)} 类，library_sha256 {sha}）；第一轮定稿的页面在 library_v3_full_r1/。\n")
assert len(index) == len(final)
print("pages", page)
