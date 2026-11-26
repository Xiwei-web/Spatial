"""GQA-spatial 按图像划分（用户 2026-11-25）。GQA 图像即 Visual Genome 图像：
有 coco_id 且该 COCO 图在统一划分里的，跟随其侧（含抠图来源图像）；其余（没有 coco_id，或 coco_id 不在统一划分里）按固定种子随机分。
随机分按 (在 one, 在 two) 分层对半；层内奇数时多出的 1 张轮流给 test / dev。旧 splits/gqa_spatial_{one,two}.json 保留内容并标记 deprecated。
输出 splits/gqa_images_v2.json、splits/gqa_spatial_{one,two}_v2.json、manifests/gqa_split.json"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import paths
from src.data.splits import read_split
from src.data import load_dataset

SEED = 20261125
S = paths.REPO_ROOT / "splits"
N = ("gqa_spatial_one", "gqa_spatial_two")
if (S / "gqa_images_v2.json").exists() and "--force" not in sys.argv:
    sys.exit("[skip] 已存在")
vg = {x["image_id"]: x for x in json.load(open(paths.get("data") / "vg_meta" / "image_data.json"))}
data = {n: load_dataset(n) for n in N}
img = lambda s: int(Path(s.image_path).stem)      # noqa: E731
imgs = {n: {img(s) for s in v} for n, v in data.items()}
union = sorted(imgs[N[0]] | imgs[N[1]])
missing = [i for i in union if i not in vg]
assert not missing, missing[:5]
C = json.load(open(S / "coco_images_v2.json"))
coco_union = {i: x for x in ("dev", "test") for i in C[x]}
coco_cut = {i: x for x in ("dev", "test") for i in C["cutout_source_images_outside_union"][x]}
holdout = set(read_split("coco_holdout")["image_ids"])
side, why, strata = {}, {}, defaultdict(list)
for i in union:
    cid = vg[i]["coco_id"]
    if cid in coco_union:
        side[i], why[i] = coco_union[cid], "forced_by_coco_union"
    elif cid in coco_cut:
        side[i], why[i] = coco_cut[cid], "forced_by_cutout_source_image"
    else:
        why[i] = "random_no_coco_id" if cid is None else "random_coco_id_not_in_split"
        strata[f"{'one' if i in imgs[N[0]] else '-'}|{'two' if i in imgs[N[1]] else '-'}"].append(i)
# 同一 coco_id 对应多张 GQA 图时必须同侧（随机分的部分按 coco_id 合并）
dup = {k: v for k, v in Counter(vg[i]["coco_id"] for i in union if vg[i]["coco_id"] is not None).items() if v > 1}
extra_to_test = True
for k in sorted(strata):
    ids = sorted(strata[k])
    random.Random(f"{SEED}:gqa_images:{k}").shuffle(ids)
    h = len(ids) // 2
    if len(ids) % 2:
        h += 0 if extra_to_test else 1
        extra_to_test = not extra_to_test
    side.update({i: "dev" for i in ids[:h]})
    side.update({i: "test" for i in ids[h:]})
for cid in dup:
    assert len({side[i] for i in union if vg[i]["coco_id"] == cid}) == 1, cid
meta = {"seed": SEED, "made_by": "scripts/make_gqa_split.py", "status": "active"}
json.dump({"dataset": "gqa_images", **meta, "unit": "image", "n_images": len(union), **{x: sorted(i for i in union if side[i] == x) for x in ("dev", "test")},
           "assignment_reason": dict(Counter(why.values())), "coco_id_of_forced": {str(i): vg[i]["coco_id"] for i in union if why[i].startswith("forced")}}, open(S / "gqa_images_v2.json", "w"), indent=0)
res = {"n_images": len(union), "n_images_by_side": dict(Counter(side.values())), "reason": dict(Counter(why.values())),
       "forced_by_side": dict(Counter(f"{why[i]}:{side[i]}" for i in union if why[i].startswith("forced"))), "n_with_coco_id": sum(vg[i]["coco_id"] is not None for i in union),
       "duplicate_coco_ids": len(dup), "gqa_images_in_coco_holdout": sorted(i for i in union if vg[i]["coco_id"] in holdout),
       "holdout_coco_ids_in_gqa": sorted(vg[i]["coco_id"] for i in union if vg[i]["coco_id"] in holdout), "datasets": {}}
for n, v in data.items():
    d = {x: sorted(s.sample_id for s in v if side[img(s)] == x) for x in ("dev", "test")}
    rel = {x: dict(sorted(Counter(s.relation for s in v if side[img(s)] == x).items())) for x in ("dev", "test")}
    old = json.load(open(S / f"{n}.json"))
    o = {i: x for x in ("dev", "test") for i in old[x]}
    json.dump({"dataset": n, **meta, "unit": "image (inherited from gqa_images_v2)", "n_total": len(v), "n_dev": len(d["dev"]), "n_test": len(d["test"]), "per_relation_by_side": rel, **d},
              open(S / f"{n}_v2.json", "w"), indent=0)
    res["datasets"][n] = {"samples": {x: len(d[x]) for x in d}, "images": {x: len({img(s) for s in v if side[img(s)] == x}) for x in d}, "relations": rel,
                          "samples_changing_side": sum(o[i] != x for x in d for i in d[x])}
    old = {"status": "deprecated", "superseded_by": f"splits/{n}_v2.json", "deprecated_reason": "2026-11-25 起 GQA-spatial 按图像划分（splits/gqa_images_v2.json）",
           **{k: w for k, w in old.items() if k not in ("status", "superseded_by", "deprecated_reason")}}
    json.dump(old, open(S / f"{n}.json", "w"), indent=0, ensure_ascii=False)
json.dump(res, open(paths.REPO_ROOT / "manifests" / "gqa_split.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
