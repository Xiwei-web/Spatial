"""COCO-spatial 与 COCO-Pairs 的统一划分（用户 2026-11-25）：两者图像取并集，按图像一次性分配 dev/test，两个数据集都继承；
合成场景的抠图继承所在图像的划分，来源图像不在并集里的也按图像分配（同一张图的抠图不跨侧）。CPU，种子固定。
分层：并集图像按 (在 spatial_one, 在 spatial_two, 在 coco_pairs, COCO-Pairs 主轴 H/V/T/-，候选物体数 2/3/4+/-)；并集外的抠图来源图像按其抠图中最少见的类别。
每层内按独立随机流打乱后对半；层内张数为奇数时，多出的 1 张轮流给 test / dev（按层的排序），避免总数偏向一侧。
输出（已存在则拒绝覆盖，--force 除外）：
  splits/coco_images_v2.json、splits/{coco_spatial_one,coco_spatial_two,coco_pairs,synth_cutouts}_v2.json
  旧文件 splits/{coco_spatial_one,coco_spatial_two,coco_pairs,synth_cutouts}.json 保留内容，加 status=deprecated 标记
  manifests/joint_coco_split.json"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import paths
from src.data.splits import read_split
from src.data import load_dataset
from src.data.coco_pairs import read_pair_records

SEED = 20261125
S = paths.REPO_ROOT / "splits"
NAMES = ("coco_spatial_one", "coco_spatial_two", "coco_pairs", "synth_cutouts")
if any((S / f"{n}_v2.json").exists() for n in NAMES + ("coco_images",)) and "--force" not in sys.argv:
    sys.exit("[skip] v2 划分已存在，不覆盖（--force 才重做）")

spatial = {n: load_dataset(n) for n in NAMES[:2]}
img_of = lambda s: int(Path(s.image_path).stem)      # noqa: E731
imgs = {n: {img_of(s) for s in v} for n, v in spatial.items()}
pairs = read_pair_records()
pimg = defaultdict(list)
for r in pairs:
    pimg[r["image_id"]].append(r)
imgs["coco_pairs"] = set(pimg)
union = sorted(set().union(*imgs.values()))


def axis(i):
    if i not in pimg:
        return "-"
    h = sum(r["relation"] in ("left", "right") for r in pimg[i])
    v = len(pimg[i]) - h
    return "H" if h > v else "V" if v > h else "T"


def nbin(i):
    """该图的候选物体数（2 / 3 / 4+），使两侧的物体对数接近；与旧 coco_pairs 划分用的分层相同。"""
    if i not in pimg:
        return "-"
    n = pimg[i][0]["n_candidates_in_image"]
    return f"n{n}" if n < 4 else "n4+"


def assign(strata: dict, tag: str) -> dict:
    side, extra_to_test = {}, True
    for key in sorted(strata):
        ids = sorted(strata[key])
        random.Random(f"{SEED}:{tag}:{key}").shuffle(ids)
        h = len(ids) // 2
        if len(ids) % 2:
            h += 0 if extra_to_test else 1
            extra_to_test = not extra_to_test
        side.update({i: "dev" for i in ids[:h]})
        side.update({i: "test" for i in ids[h:]})
    return side


strata = defaultdict(list)
for i in union:
    strata["|".join(["one" if i in imgs["coco_spatial_one"] else "-", "two" if i in imgs["coco_spatial_two"] else "-", "pairs" if i in pimg else "-", axis(i), nbin(i)])].append(i)
side = assign(strata, "joint_coco_images")
per_stratum = {k: {"n": len(v), "dev": sum(side[i] == "dev" for i in v), "test": sum(side[i] == "test" for i in v)} for k, v in sorted(strata.items())}

# 抠图
lib_raw = (paths.get("data") / "synth" / "cutouts" / "library.jsonl").read_bytes()
lib = [json.loads(l) for l in lib_raw.decode().splitlines() if l.strip()]
assert all(e["source"].startswith("coco") for e in lib), Counter(e["source"] for e in lib)
cimg = {e["cutout_id"]: int(e["src"]["image_id"]) for e in lib}
holdout = set(read_split("coco_holdout")["image_ids"])
assert not holdout & (set(union) | set(cimg.values()))
ccount = Counter(e["category"] for e in lib)
by_img = defaultdict(set)
for e in lib:
    by_img[cimg[e["cutout_id"]]].add(e["category"])
out_strata = defaultdict(list)
for i, cats in by_img.items():
    if i not in side:
        out_strata[min(cats, key=lambda c: (ccount[c], c))].append(i)
out_side = assign(out_strata, "cutout_images_outside_union")
all_side = {**side, **out_side}
meta = {"seed": SEED, "made_by": "scripts/make_joint_coco_split.py", "status": "active"}


def dump(name, d):
    json.dump(d, open(S / f"{name}.json", "w"), indent=0, ensure_ascii=False)


ids = lambda sd, x: sorted(i for i, v in sd.items() if v == x)      # noqa: E731
dump("coco_images_v2", {"dataset": "coco_images", **meta, "unit": "image", "n_images": len(union), "stratify_by": "dataset membership x coco_pairs dominant axis x candidates per image (2/3/4+)",
                        "per_stratum": per_stratum, "dev": ids(side, "dev"), "test": ids(side, "test"),
                        "cutout_source_images_outside_union": {"n": len(out_side), "dev": ids(out_side, "dev"), "test": ids(out_side, "test")}})
rel_tab, old_cmp = {}, {}
for n, v in spatial.items():
    d = {x: sorted(s.sample_id for s in v if side[img_of(s)] == x) for x in ("dev", "test")}
    per_rel = {x: dict(sorted(Counter(s.relation for s in v if side[img_of(s)] == x).items())) for x in ("dev", "test")}
    dump(f"{n}_v2", {"dataset": n, **meta, "unit": "image (inherited from coco_images_v2)", "n_total": len(v), "n_dev": len(d["dev"]), "n_test": len(d["test"]),
                     "n_images": {x: len({img_of(s) for s in v if side[img_of(s)] == x}) for x in ("dev", "test")},
                     "per_relation": dict(sorted(Counter(s.relation for s in v).items())), "per_relation_by_side": per_rel, **d})
    rel_tab[n] = per_rel
    old = json.load(open(S / f"{n}.json"))
    o = {i: x for x in ("dev", "test") for i in old[x]}
    img_sides = defaultdict(set)
    for s in v:
        img_sides[img_of(s)].add(o[s.sample_id])
    old_cmp[n] = {"samples_changing_side": sum(o[i] != x for x in ("dev", "test") for i in d[x]), "images_on_both_sides_in_old_split": sum(len(x) == 2 for x in img_sides.values())}
pd = {x: sorted(i for i in pimg if side[i] == x) for x in ("dev", "test")}
counts = {x: {"n_images": len(pd[x]), "n_directed_pairs": sum(len(pimg[i]) for i in pd[x]),
              "per_relation": dict(sorted(Counter(r["relation"] for i in pd[x] for r in pimg[i]).items())),
              "n_proj_overlap": sum(r["proj_overlap"] for i in pd[x] for r in pimg[i])} for x in ("dev", "test")}
dump("coco_pairs_v2", {"dataset": "coco_pairs", **meta, "unit": "image", "n_images": len(pimg), "n_dev_images": len(pd["dev"]), "n_test_images": len(pd["test"]),
                       "stratify_by": "inherited from coco_images_v2", "per_stratum": per_stratum, "counts": counts, **pd})
cd = {x: sorted(c for c, i in cimg.items() if all_side[i] == x) for x in ("dev", "test")}
per_cat = {c: [sum(e["category"] == c and all_side[cimg[e["cutout_id"]]] == x for e in lib) for x in ("dev", "test")] for c in sorted(ccount)}
import hashlib                                                        # noqa: E402
dump("synth_cutouts_v2", {"dataset": "synth_cutouts", **meta, "unit": "image (cutouts inherit their source image's side)", "n_total": len(lib), "n_dev": len(cd["dev"]),
                          "n_test": len(cd["test"]), "n_categories": len(ccount), "per_category": per_cat, "library_sha256": hashlib.sha256(lib_raw).hexdigest(), **cd})
oldc = json.load(open(S / "synth_cutouts.json"))
oc = {i: x for x in ("dev", "test") for i in oldc[x]}
old_imgs = defaultdict(set)
for c, i in cimg.items():
    old_imgs[i].add(oc[c])
old_cmp["synth_cutouts"] = {"cutouts_changing_side": sum(oc[c] != all_side[i] for c, i in cimg.items()), "images_on_both_sides_in_old_split": sum(len(v) == 2 for v in old_imgs.values()),
                            "trial_v0_dev_cutouts_now_test": sum(oc[c] == "dev" and all_side[i] == "test" for c, i in cimg.items())}
op = json.load(open(S / "coco_pairs.json"))
po = {i: x for x in ("dev", "test") for i in op[x]}
old_cmp["coco_pairs"] = {"images_changing_side": sum(po[i] != side[i] for i in pimg)}
zero = {c: v for c, v in per_cat.items() if min(v) == 0}
for n in NAMES:                  # 旧文件：内容不动，只加作废标记
    f = S / f"{n}.json"
    d = json.load(open(f))
    d = {"status": "deprecated", "superseded_by": f"splits/{n}_v2.json", "deprecated_reason": "2026-11-25 起 COCO-spatial / COCO-Pairs / 合成抠图统一按图像划分（splits/coco_images_v2.json）",
         **{k: v for k, v in d.items() if k not in ("status", "superseded_by", "deprecated_reason")}}
    json.dump(d, open(f, "w"), indent=0, ensure_ascii=False)
summ = {"n_union_images": len(union), "dev_images": len(ids(side, "dev")), "test_images": len(ids(side, "test")), "per_stratum": per_stratum, "coco_spatial_relation_by_side": rel_tab,
        "coco_pairs": counts, "cutouts": {"n_dev": len(cd["dev"]), "n_test": len(cd["test"]), "n_in_union_images": sum(i in side for i in cimg.values()),
                                           "categories_missing_on_one_side": zero, "n_categories_both_sides": len(per_cat) - len(zero)}, "vs_old": old_cmp}
json.dump(summ, open(paths.REPO_ROOT / "manifests" / "joint_coco_split.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=None, ensure_ascii=False))
