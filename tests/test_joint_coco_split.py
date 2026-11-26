"""COCO-spatial / COCO-Pairs / 合成抠图的统一划分：同一张图在所有数据集里必须在同一侧。"""
import json
from pathlib import Path

import pytest

from src import paths
from src.data import load_dataset, load_split
from src.data.splits import read_split

S = paths.REPO_ROOT / "splits"


def test_same_image_same_side_everywhere():
    side = {}

    def put(i, x):
        assert side.setdefault(int(i), x) == x, (i, x)

    im = json.load(open(S / "coco_images_v2.json"))
    assert not set(im["dev"]) & set(im["test"])
    for x in ("dev", "test"):
        for i in im[x] + im["cutout_source_images_outside_union"][x]:
            put(i, x)
        for n in ("coco_spatial_one", "coco_spatial_two"):
            for s in load_split(n, x):
                put(Path(s.image_path).stem, x)
        for i in read_split("coco_pairs")[x]:
            put(i, x)
    lib = paths.get("data") / "synth" / "cutouts" / "library.jsonl"
    if lib.is_file():
        img = {e["cutout_id"]: e["src"]["image_id"] for e in map(json.loads, open(lib))}
        sp = read_split("synth_cutouts")
        assert set(img) == set(sp["dev"]) | set(sp["test"])
        for x in ("dev", "test"):
            for c in sp[x]:
                put(img[c], x)
    assert not set(side) & set(read_split("coco_holdout")["image_ids"])


def test_splits_cover_everything_and_old_ones_are_refused(monkeypatch):
    for n in ("coco_spatial_one", "coco_spatial_two"):
        ids = [s.sample_id for x in ("dev", "test") for s in load_split(n, x)]
        assert len(ids) == len(set(ids)) == len(load_dataset(n))
    import src.data.splits as sp
    for n in ("coco_spatial_one", "coco_spatial_two", "coco_pairs", "synth_cutouts", "gqa_spatial_one", "gqa_spatial_two"):
        monkeypatch.setitem(sp.ACTIVE, n, n)          # 指回旧文件
        with pytest.raises(RuntimeError, match="作废"):
            sp.read_split(n)


def test_no_image_crosses_sides_across_all_datasets():
    """含 GQA（按 Visual Genome 的 coco_id 换成统一图像键）、COCO-Pairs、合成抠图、What's Up。"""
    from scripts.check_split_consistency import run
    if not (paths.get("data") / "vg_meta" / "image_data.json").is_file():
        pytest.skip("VG 图像元数据未下载")
    r = run()
    assert r["n_cross_side"] == 0 and all(d["covers_all"] for d in r["datasets"].values())
