"""框坐标约定标定：清单只含 dev、与任何 test 及 COCO 留出池无共同图像；换算 / IoU / 选择规则的行为。"""
import json
from pathlib import Path

import numpy as np
import pytest

from src import paths
from src.bbox_calib import CONVENTIONS, bootstrap_diff, convert, iou, select

S = paths.REPO_ROOT / "splits"
CAL = S / "bbox_calib_v1.json"


@pytest.fixture(scope="module")
def cal():
    if not CAL.is_file():
        pytest.skip("还没有 splits/bbox_calib_v1.json")
    return json.load(open(CAL))


def test_counts(cal):
    it = cal["items"]
    assert len(it) == len({i["item_id"] for i in it}) == cal["n_items"] == 600
    w = [i for i in it if i["subset"] == "whatsup"]
    assert len(w) == 400 and len({i["image_key"] for i in w}) == 200
    for ds in ("whatsup_a", "whatsup_b"):
        s = {(i["sample_id"], i["relation"]) for i in w if i["dataset"] == ds}
        rels = [r for _, r in s]
        assert len(s) == 100 and all(rels.count(r) == 25 for r in set(rels)) and len(set(rels)) == 4
    c = [i for i in it if i["subset"] == "coco"]
    assert len(c) == 200 and len({i["image_key"] for i in c}) == 100
    for k in {i["image_key"] for i in it}:
        assert sorted(i["role"] for i in it if i["image_key"] == k) == ["reference", "target"]


def test_dev_only_and_no_test_or_holdout_image(cal):
    from src.data import load_split
    keys = lambda n, x: {s.sample_id for s in load_split(n, x)}      # noqa: E731
    for n in ("whatsup_a", "whatsup_b", "coco_spatial_two"):
        dev, test = keys(n, "dev"), keys(n, "test")
        mine = {i["sample_id"] for i in cal["items"] if i["dataset"] == n}
        assert mine and mine <= dev and not mine & test
    # 图像键：What's Up 用 数据集:样本；COCO 用 coco:<image_id>；所有数据集的 test 图像与留出池
    vg = {x["image_id"]: x["coco_id"] for x in json.load(open(paths.get("data") / "vg_meta" / "image_data.json"))}
    test_keys = set()
    for n in ("whatsup_a", "whatsup_b", "coco_spatial_one", "coco_spatial_two", "gqa_spatial_one", "gqa_spatial_two"):
        for s in load_split(n, "test"):
            st = Path(s.image_path).stem
            if n.startswith("coco"):
                test_keys.add(f"coco:{int(st)}")
            elif n.startswith("gqa"):
                test_keys.add(f"coco:{vg[int(st)]}" if vg[int(st)] is not None else f"vg:{int(st)}")
            else:
                test_keys.add(f"{n}:{s.sample_id}")
    from src.data.splits import read_split
    test_keys |= {f"coco:{i}" for i in read_split("coco_pairs")["test"]}
    test_keys |= {f"coco:{i}" for i in json.load(open(S / "coco_images_v2.json"))["test"]}
    hold = {f"coco:{i}" for i in json.load(open(S / "coco_holdout.json"))["image_ids"]}
    mine = {i["image_key"] for i in cal["items"]}
    assert not mine & test_keys and not mine & hold


def test_truth_boxes(cal):
    from src.data.whatsup_annotations import load_annotations
    ann = {(d["dataset"], d["sample_id"]): d for d in load_annotations("auto_v3")}
    for i in cal["items"]:
        x1, y1, x2, y2 = i["truth_xyxy"]
        W, H = i["orig_wh"]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        if i["subset"] == "whatsup":
            o = ann[(i["dataset"], i["sample_id"])]["objects"][i["role"]]
            assert o["box_gt_xyxy"] == i["truth_xyxy"] and o["name"] == i["object"]


def test_convert_all():
    wh, mwh = (1280, 960), (1148, 840)
    b = [100.0, 200.0, 300.0, 400.0]
    assert convert(b, "orig_xyxy", wh, mwh) == b
    assert convert(b, "orig_yxyx", wh, mwh) == [200.0, 100.0, 400.0, 300.0]
    assert np.allclose(convert(b, "norm1000_xyxy", wh, mwh), [128, 192, 384, 384])
    assert np.allclose(convert([0.1, 0.2, 0.3, 0.4], "norm1_xyxy", wh, mwh), [128, 192, 384, 384])
    assert np.allclose(convert(b, "model_xyxy", wh, mwh), [100 / 1148 * 1280, 200 / 840 * 960, 300 / 1148 * 1280, 400 / 840 * 960])
    assert np.allclose(convert(b, "norm1000_xyxy", wh, mwh, (-100, 0, 1380, 1480)), [-100 + 148, 296, -100 + 444, 592])
    assert convert(b, "range_norm1_else_norm1000", wh, mwh) == convert(b, "norm1000_xyxy", wh, mwh)
    assert convert([0, 0.5, 1, 1], "range_norm1_else_model", wh, mwh) == convert([0, 0.5, 1, 1], "norm1_xyxy", wh, mwh)
    assert convert(b, "range_norm1_else_model", wh, mwh) == convert(b, "model_xyxy", wh, mwh)
    assert convert(None, "model_xyxy", wh, mwh) is None
    assert len(CONVENTIONS) == 10


def test_iou_rules():
    wh = (100, 100)
    t = [10, 10, 50, 50]
    assert iou([10, 10, 50, 50], t, wh) == 1.0
    assert iou(None, t, wh) == 0 == iou([50, 10, 50, 50], t, wh) == iou([10, 60, 50, 50], t, wh) == iou([120, 10, 150, 50], t, wh)
    assert iou([-10, 10, 50, 50], t, wh) == pytest.approx(1600 / (2000))      # 裁剪后 [0,10,50,50]


def _rec(iid, parsed, truth, subset="whatsup", p="P0", wh=(1000, 1000), key=None, model=(500, 500)):
    return {"item_id": iid, "prompt_id": p, "parsed": parsed, "orig_wh": wh, "model_wh": model, "view_box": None, "truth": truth, "subset": subset, "image_key": key or iid}


def test_select_picks_the_right_convention():
    rng = np.random.default_rng(0)
    rec = []
    for i in range(60):
        x, y = rng.integers(0, 400, 2)
        t = [float(x) * 2, float(y) * 2, float(x) * 2 + 100, float(y) * 2 + 60]          # 原图 1000，模型 500 → 真值 /2
        pred = [v / 2 for v in t]                                                        # 模型输入像素 xyxy
        rec.append(_rec(f"i{i:03d}", pred, t, subset="whatsup" if i % 2 else "coco"))
    r = select(rec, ["P0"])
    assert r["first"][1] in ("model_xyxy",) and not r["undecided"], r["undecided_reasons"]
    assert r["first_mean_iou"] == pytest.approx(1.0)


def test_identical_conversions_excluded_from_second():
    """所有数 >1 且第一名是 norm1000_xyxy 时，range_norm1_else_norm1000 逐条相同，不能当第二名。"""
    rec = []
    for i in range(40):
        b = [100.0 + 10 * i, 100.0, 200.0 + 10 * i, 220.0]
        rec.append(_rec(f"i{i:03d}", b, [v for v in b], subset="whatsup" if i % 2 else "coco", wh=(1000, 1000)))
    r = select(rec, ["P0"])
    assert r["first"][1] == "norm1000_xyxy" and r["second"][1] != "range_norm1_else_norm1000"


def test_undecided_when_subsets_disagree_or_low_iou():
    rec = []
    for i in range(40):
        t = [100.0, 100.0, 300.0, 300.0]
        rec.append(_rec(f"i{i:03d}", [100.0, 100.0, 300.0, 300.0] if i % 2 else [50.0, 50.0, 150.0, 150.0], t, subset="whatsup" if i % 2 else "coco", wh=(1000, 1000), model=(500, 500)))
    r = select(rec, ["P0"])
    assert r["undecided"] and "What's Up 与 COCO 的第一名不一致" in r["undecided_reasons"]
    low = [_rec(f"i{i:03d}", [1.0, 2.0, 3.0, 4.0], [500.0, 500.0, 900.0, 900.0]) for i in range(40)]
    assert select(low, ["P0"])["undecided"]


def test_bootstrap_groups_by_image():
    a = bootstrap_diff(np.ones(10), [f"im{i // 2}" for i in range(10)])
    assert a == [1.0, 1.0]
    with pytest.raises(AssertionError):
        select([_rec("a", None, [1, 1, 2, 2])] + [_rec("b", None, [1, 1, 2, 2], p="P1")], ["P0", "P1"])


def test_calibrated_block_and_prompts():
    import yaml

    from scripts.bbox_calib_run import prompt_set
    from src.prompts import to_original_xyxy
    cal = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "bbox_conventions.yaml"))["calibrated"]
    assert len(cal) == 16
    for m, c in cal.items():
        if c["prompt_id"] != "pending":
            assert prompt_set(m)[c["prompt_id"]] == c["prompt_template"], m
    assert list(prompt_set("VST-7B-RL")) == ["P0", "P1", "P2"] and prompt_set("VST-7B-RL")["P2"].startswith(prompt_set("VST-7B-RL")["P0"] + " ")
    assert "P2" not in prompt_set("Qwen3-VL-8B-Instruct")
    assert to_original_xyxy([0.1, 0.2, 0.3, 0.4], "range_norm1_else_norm1000", (200, 100)) == to_original_xyxy([0.1, 0.2, 0.3, 0.4], "norm1_xyxy", (200, 100))
    assert to_original_xyxy([0.284, 301, 732, 938], "range_norm1_else_norm1000", (1000, 1000)) == to_original_xyxy([0.284, 301, 732, 938], "norm1000_xyxy", (1000, 1000))
