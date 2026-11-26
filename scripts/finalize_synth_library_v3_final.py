"""抠图库 v3 定稿（planner 2026-11-26）：在 v3 草稿（<data>/synth/cutouts_v3/library.jsonl，提交 0bd0f57）上再剔除
annotations/synth/library_v3_final_exclusions.csv 里的 23 个（Xiwei 人工抽查 6 个 + planner 的 cow 单类复核 17 个）；两侧达不到各 >= 3 的类别整类排除。
草稿 library.jsonl 保留不动；定稿写到 library_final.jsonl，更新 splits/synth_library_v3.json 与 manifests/synth_library_v3.json（标 final），并重出 library_v3_full 联系表（标 final）。CPU。"""
import csv
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from scripts.build_synth_library_v2 import CFG as C2, on_gray
from src import paths

REPO = paths.REPO_ROOT
V3 = paths.get("data") / "synth" / "cutouts_v3"
man = json.load(open(REPO / "manifests" / "synth_library_v3.json"))
raw = (V3 / "library.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == man["library_sha256"], "v3 草稿与清单的 sha256 不一致"
draft = [json.loads(l) for l in raw.decode().splitlines()]
ex = {r["cutout_id"]: r for r in csv.DictReader(l for l in open(REPO / "annotations" / "synth" / "library_v3_final_exclusions.csv") if not l.startswith("#"))}
ids = {r["cutout_id"] for r in draft}
assert set(ex) <= ids and len(ex) == 23
keep = [r for r in draft if r["cutout_id"] not in ex]
per = defaultdict(Counter)
for r in keep:
    per[r["category"]][r["split"]] += 1
usable = sorted(c for c in per if min(per[c]["dev"], per[c]["test"]) >= C2["min_per_side"])
dropped = {c: dict(per[c]) for c in per if c not in usable}
final = sorted([r for r in keep if r["category"] in usable], key=lambda r: r["cutout_id"])
rawf = "".join(json.dumps(r) + "\n" for r in final).encode()
(V3 / "library_final.jsonl").write_bytes(rawf)
sha = hashlib.sha256(rawf).hexdigest()
small = sorted(c for c in usable if min(sum(r["small36_ok"] for r in final if r["category"] == c and r["split"] == s) for s in ("dev", "test")) >= C2["min_per_side"])
json.dump({"dataset": "synth_library_v3", "status": "final", "unit": "image（沿用 v2 的划分）", "library_file": "<data>/synth/cutouts_v3/library_final.jsonl", "library_sha256": sha,
           "n_total": len(final), "dev": [r["cutout_id"] for r in final if r["split"] == "dev"], "test": [r["cutout_id"] for r in final if r["split"] == "test"]},
          open(REPO / "splits" / "synth_library_v3.json", "w"), indent=0)
draft_per = {c: dict(Counter(r["split"] for r in draft if r["category"] == c)) for c in sorted({r["category"] for r in draft})}
man.update({
    "status": "final",
    "provenance": "自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；再按 Xiwei 人工抽查剔除 6 个、planner 的 cow 单类复核（模型预审，未经人工复核）剔除 17 个",
    "draft": {"library_sha256": man["library_sha256"], "n": len(draft), "commit": "0bd0f57"},
    "final_exclusions": {"file": "annotations/synth/library_v3_final_exclusions.csv", "n": len(ex), "by_source": dict(Counter(r["source"] for r in ex.values())),
                         "by_category": dict(Counter(r["category"] for r in ex.values()))},
    "spotcheck": {"n": 150, "bad": 3, "unsure": 3, "bad_rate": 0.02, "bad_wilson95": [0.0068, 0.0571], "bad_or_unsure_rate": 0.04, "bad_or_unsure_wilson95": [0.0185, 0.0845],
                  "criterion": "<= 5%（<= 7 个）", "meets": True, "note": "抽查估计的是剔除前草稿的残余比例；定稿时的 23 个剔除是在抽查之后做的"},
    "library_sha256": sha, "library_file": "library_final.jsonl", "n_final": len(final), "n_final_by_side": dict(Counter(r["split"] for r in final)),
    "n_final_by_src": dict(Counter(r["src"] for r in final)), "n_categories_usable": len(usable), "categories_usable": usable,
    "categories_dropped_at_final": dropped, "n_small36_ok": sum(r["small36_ok"] for r in final), "n_categories_small36": len(small), "categories_small36": small,
    "per_category_final": {c: dict(per[c]) for c in sorted(per)}, "per_category_draft": draft_per})
json.dump(man, open(REPO / "manifests" / "synth_library_v3.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: man[k] for k in ("n_final", "n_final_by_side", "n_final_by_src", "n_categories_usable", "categories_dropped_at_final", "n_small36_ok", "n_categories_small36", "final_exclusions")}, ensure_ascii=False))
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
T, COLS, PER = 180, 12, 60
out = paths.get("outputs") / "synth_review" / "library_v3_full"
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
        d.text((6, 5), f"FINAL  {fn}  {c}: {sum(r['split'] == 'dev' for r in items)} dev / {sum(r['split'] == 'test' for r in items)} test; cells {s + 1}-{s + len(part)} of {len(items)}; "
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
(out / "FINAL.txt").write_text(f"library_v3 final（{len(final)} 个，{len(usable)} 类，library_sha256 {sha}）；页面与 index.csv 已按定稿重出。\n")
assert len(index) == len(final)
print("pages", page)
