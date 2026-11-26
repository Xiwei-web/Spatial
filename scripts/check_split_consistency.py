"""跨数据集划分一致性检查（CPU）：把每个样本的图像换成统一的图像键（COCO 图 → coco:<id>；GQA 图有 coco_id → coco:<id>，否则 vg:<id>），
确认同一图像键在所有数据集里只出现在一侧，且各数据集的 dev ∪ test = 全部样本。输出 manifests/split_consistency.json；不一致时退出码非零。"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from src import paths
from src.data import load_dataset, load_split
from src.data.loaders import excluded_samples
from src.data.coco_pairs import load_coco_pairs, read_pair_records
from src.data.splits import read_split

S = paths.REPO_ROOT / "splits"


def run() -> dict:
    vg = {x["image_id"]: x["coco_id"] for x in json.load(open(paths.get("data") / "vg_meta" / "image_data.json"))}
    where = defaultdict(lambda: defaultdict(set))          # key -> side -> {dataset}
    res = {"datasets": {}}

    def add(key, side, ds):
        where[key][side].add(ds)

    for n in ("coco_spatial_one", "coco_spatial_two", "gqa_spatial_one", "gqa_spatial_two", "whatsup_a", "whatsup_b"):
        ids = {}
        for x in ("dev", "test"):
            for s in load_split(n, x):
                assert s.sample_id not in ids
                ids[s.sample_id] = x
                st = Path(s.image_path).stem
                if n.startswith("coco"):
                    key = f"coco:{int(st)}"
                elif n.startswith("gqa"):
                    key = f"coco:{vg[int(st)]}" if vg[int(st)] is not None else f"vg:{int(st)}"
                else:
                    key = f"{n}:{st}"
                add(key, x, n)
        want = {s.sample_id for s in load_dataset(n)} - {i for d, i in excluded_samples() if d == n}
        res["datasets"][n] = {"n_samples": len(ids), "covers_all": set(ids) == want, "n_dev": sum(v == "dev" for v in ids.values()), "n_test": sum(v == "test" for v in ids.values())}
    for x in ("dev", "test"):
        for s in load_coco_pairs(x):
            add(f"coco:{s.extra['image_id']}", x, "coco_pairs")
    sp = read_split("coco_pairs")
    res["datasets"]["coco_pairs"] = {"covers_all": {r["image_id"] for r in read_pair_records()} == set(sp["dev"]) | set(sp["test"])}
    lib = {e["cutout_id"]: e["src"]["image_id"] for e in map(json.loads, open(paths.get("data") / "synth" / "cutouts" / "library.jsonl"))}
    sc = read_split("synth_cutouts")
    for x in ("dev", "test"):
        for c in sc[x]:
            add(f"coco:{lib[c]}", x, "synth_cutouts")
    ho = {f"coco:{i}" for i in read_split("coco_holdout")["image_ids"]}
    bad = {k: {x: sorted(v) for x, v in d.items()} for k, d in where.items() if len(d) > 1}
    res.update(n_image_keys=len(where), n_keys_in_more_than_one_dataset=sum(len(set().union(*d.values())) > 1 for d in where.values()), n_cross_side=len(bad), cross_side=dict(list(bad.items())[:20]),
               holdout_keys_used_by_evaluation={k: sorted(set().union(*where[k].values())) for k in ho if k in where})
    return res


if __name__ == "__main__":
    r = run()
    json.dump(r, open(paths.REPO_ROOT / "manifests" / "split_consistency.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(r, ensure_ascii=False))
    sys.exit(1 if r["n_cross_side"] or not all(d["covers_all"] for d in r["datasets"].values()) else 0)
