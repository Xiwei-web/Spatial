"""抠图库 v3 定稿组装（planner 2026-11-26 的口径；CPU，只读 v2 结果，不新增下载）：
  v3 = v2 中 annotations/synth/library_v2_full_prereview.csv 里 exclude=0 的抠图，再减去规则 c（person 关键点）、d（成堆）、e（CLIP 负类）剔除的；
       不用规则 a、b、f（dev 上的理由见 manifests/synth_library_v3_tune.json 与 DATA_REPORT）；规则 d 的凸度阈值按 tune 的结果不用（T_sol = 0），只看连通块数与 CLIP pile；
       两侧达不到各 >= 3 的类别整类排除。
来源写法：自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；人工抽查结果另报。
输出：<data>/synth/cutouts_v3/{library.jsonl, png/}、splits/synth_library_v3.json、manifests/synth_library_v3.json、
      <outputs>/synth_review/library_v3_full/（全量联系表 + index.csv）、<outputs>/synth_review/library_v3_spotcheck/（固定种子随机 150 个，不显示 CLIP 与预审结论）"""
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from scripts.build_synth_library_v2 import CFG as C2, on_gray
from src import paths

REPO = paths.REPO_ROOT
DATA = paths.get("data")
V2, V3 = DATA / "synth" / "cutouts_v2", DATA / "synth" / "cutouts_v3"
C3 = yaml.safe_load(open(REPO / "configs" / "synth" / "library_v3.yaml"))
TH = yaml.safe_load(open(REPO / "configs" / "synth" / "library_v3_thresholds.yaml"))
SEED = 20261125
man2 = json.load(open(REPO / "manifests" / "synth_library_v2.json"))
raw2 = (V2 / "library.jsonl").read_bytes()
assert hashlib.sha256(raw2).hexdigest() == man2["library_sha256"]
lib2 = [json.loads(l) for l in raw2.decode().splitlines()]
lab = {r["cutout_id"]: r for r in csv.DictReader(l for l in open(REPO / "annotations" / "synth" / "library_v2_full_prereview.csv") if not l.startswith("#"))}
assert set(lab) == {r["cutout_id"] for r in lib2}
F = {x["cutout_id"]: x for x in map(json.loads, open(V3 / "features_v2.jsonl"))}
CX = {x["cutout_id"]: x for x in map(json.loads, open(V3 / "clip_extra_v2.jsonl"))}
d_cats = set(C3["rules"]["d_pile"]["categories"])
excl_cls = set(C3["exclude_categories_extra"])


def rule_flags(r):
    f, cx, c = F[r["cutout_id"]], CX.get(r["cutout_id"]), r["category"]
    return {"c_person": c == "person" and not f.get("person_ok", False),
            "d_pile": c in d_cats and (f["n_components"] > 1 or f["solidity"] < TH["T_sol"] or (cx is not None and cx.get("pile_prob", 0) > C3["rules"]["d_pile"]["clip_pile_max_prob"])),
            "e_neg": bool(cx and cx.get("neg_top"))}


cnt = Counter()
keep, per = [], defaultdict(lambda: defaultdict(Counter))
for r in lib2:
    c, s = r["category"], r["split"]
    per[c][s]["v2"] += 1
    L = lab[r["cutout_id"]]
    fl = rule_flags(r)
    if c in excl_cls:
        per[c][s]["class_excluded"] += 1; cnt["class_excluded"] += 1; continue
    if L["exclude"] == "1":
        per[c][s]["prereview_excluded"] += 1; cnt["prereview_excluded"] += 1
        for k, v in fl.items():
            cnt[f"{k}_also_in_prereview"] += v
        continue
    if any(fl.values()):
        per[c][s]["rules_cde_extra"] += 1; cnt["rules_cde_extra"] += 1
        for k, v in fl.items():
            cnt[f"{k}_extra"] += v
        continue
    per[c][s]["kept"] += 1
    keep.append(r)
usable = sorted(c for c in per if min(per[c]["dev"]["kept"], per[c]["test"]["kept"]) >= C2["min_per_side"])
dropped_cls = {c: {s: per[c][s]["kept"] for s in ("dev", "test")} for c in per if c not in usable and c not in excl_cls}
final = sorted([r for r in keep if r["category"] in usable], key=lambda r: r["cutout_id"])
(V3 / "png").mkdir(parents=True, exist_ok=True)
for r in final:
    shutil.copyfile(V2 / "png" / f"{r['cutout_id']}.png", V3 / "png" / f"{r['cutout_id']}.png")
raw = "".join(json.dumps(r) + "\n" for r in final).encode()
(V3 / "library.jsonl").write_bytes(raw)
sha = hashlib.sha256(raw).hexdigest()
json.dump({"dataset": "synth_library_v3", "status": "draft（v3 定稿之前不得用于生成场景）", "unit": "image（沿用 v2 的划分）", "library_sha256": sha, "n_total": len(final),
           "dev": [r["cutout_id"] for r in final if r["split"] == "dev"], "test": [r["cutout_id"] for r in final if r["split"] == "test"]},
          open(REPO / "splits" / "synth_library_v3.json", "w"), indent=0)
small = sorted(c for c in usable if min(sum(r["small36_ok"] for r in final if r["category"] == c and r["split"] == s) for s in ("dev", "test")) >= C2["min_per_side"])
v2_usable = set(man2["categories_usable"])
man = {"provenance": "自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；人工抽查结果另报", "library_sha256": sha, "v2_library_sha256": man2["library_sha256"],
       "n_v2": len(lib2), "n_final": len(final), "n_final_by_side": dict(Counter(r["split"] for r in final)), "n_final_by_src": dict(Counter(r["src"] for r in final)),
       "n_categories_usable": len(usable), "meets_min_categories": len(usable) >= C2["min_categories"], "categories_usable": usable,
       "categories_dropped_lt3_per_side": dropped_cls, "categories_class_excluded": sorted(excl_cls), "categories_lost_vs_v2": sorted(v2_usable - set(usable)),
       "n_small36_ok": sum(r["small36_ok"] for r in final), "n_categories_small36": len(small), "categories_small36": small,
       "removed": dict(cnt), "rule_thresholds": {"T_sol": TH["T_sol"], "pile_prob_max": C3["rules"]["d_pile"]["clip_pile_max_prob"]},
       "rules_not_used": {"a_occ_contact": "v2 候选上不起作用（dev BAD 召回 0）", "b_contact_straight_edge": "v2 候选上不起作用（dev BAD 召回 0）",
                          "f_sam_iou": "每抓到 1 个 BAD 要误删约 1.6 个 OK（dev，T_sam=0.85：召回 0.393、OK 损失 0.138），由逐格预审替代"},
       "per_category": {c: {s: dict(per[c][s]) for s in ("dev", "test")} for c in sorted(per)},
       "per_category_change_vs_v2": {c: {s: per[c][s]["kept"] * (c in usable) - per[c][s]["v2"] * (c in v2_usable) for s in ("dev", "test")} for c in sorted(per)}}
json.dump(man, open(REPO / "manifests" / "synth_library_v3.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in man.items() if k not in ("per_category", "per_category_change_vs_v2", "categories_usable", "categories_small36")}, ensure_ascii=False))

font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
T, COLS, PER = 180, 12, 60


def sheet(items, fn, title, show_scores, out, index, page_name):
    rows = (len(items) + COLS - 1) // COLS
    im = Image.new("RGB", (COLS * (T + 6) + 6, 26 + rows * (T + 40)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((6, 5), title, fill=(0, 0, 0), font=font)
    for k, r in enumerate(items):
        y, x = divmod(k, COLS)
        X, Y = 6 + x * (T + 6), 26 + y * (T + 40)
        g = on_gray(np.asarray(Image.open(V3 / "png" / f"{r['cutout_id']}.png").convert("RGBA")), C2["margin_frac"])
        im.paste(g.resize((T, T), Image.LANCZOS), (X, Y))
        d.text((X, Y + T + 1), r["cutout_id"], fill=(0, 0, 0), font=font)
        if show_scores:
            d.text((X, Y + T + 17), f"{r['split']} {r['clip_prob_true']:.3f}{'' if r['small36_ok'] else ' no36'}", fill=(0, 0, 160) if r["split"] == "dev" else (160, 0, 0), font=font)
            index.append([page_name, y + 1, x + 1, r["cutout_id"], r["category"], r["split"], r["src"], r["clip_prob_true"], r["small36_ok"]])
        else:
            d.text((X, Y + T + 17), r["category"], fill=(0, 0, 0), font=font)
            index.append([page_name, y + 1, x + 1, r["cutout_id"], r["category"]])
    assert max(im.size) <= 2400
    im.save(out / fn, quality=90)


out = paths.get("outputs") / "synth_review" / "library_v3_full"
out.mkdir(parents=True, exist_ok=True)
by = defaultdict(list)
for r in final:
    by[r["category"]].append(r)
index, page = [], 0
for c in sorted(by):
    items = sorted(by[c], key=lambda r: (r["split"], r["cutout_id"]))
    for s in range(0, len(items), PER):
        page += 1
        fn = f"page_{page:03d}_{c.replace(' ', '-')}.jpg"
        sheet(items[s:s + PER], fn, f"{fn}  {c}: {sum(r['split'] == 'dev' for r in items)} dev / {sum(r['split'] == 'test' for r in items)} test; cells {s + 1}-{s + min(PER, len(items) - s)} of {len(items)}; "
              "number = CLIP prob (normal); no36 = fails 18-px check", True, out, index, fn)
with open(out / "index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category", "side", "source", "clip_prob", "small36_ok"]); w.writerows(index)
assert len(index) == len(final)
sp = paths.get("outputs") / "synth_review" / "library_v3_spotcheck"
sp.mkdir(parents=True, exist_ok=True)
pick = random.Random(f"{SEED}:library_v3_spotcheck").sample(final, 150)
idx2 = []
for k in range(0, 150, 60):
    fn = f"spotcheck_{k // 60 + 1:02d}.jpg"
    sheet(pick[k:k + 60], fn, f"{fn}: random sample of 150 cutouts from library_v3 (fixed seed; category shown, no scores, no review labels)", False, sp, idx2, fn)
with open(sp / "index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["page", "row", "col", "cutout_id", "category"]); w.writerows(idx2)
print("full pages", page, "cells", len(index), "| spotcheck", len(idx2))
