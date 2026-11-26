"""COCO-Pairs 的单元测试：几何函数用合成的框 / mask；划分完整性测试在划分文件不存在时跳过。"""
import json

import numpy as np
import pytest

from src.data import coco_pairs as cp


def _mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_mask_centroid_pixel_center_convention():
    # 覆盖列 [10,20)、行 [30,50) 的矩形，质心应与同一矩形 bbox 的中心一致
    assert cp.mask_centroid(_mask(100, 100, 10, 30, 20, 50)) == pytest.approx((15.0, 40.0))
    with pytest.raises(ValueError):
        cp.mask_centroid(np.zeros((4, 4), dtype=np.uint8))


def test_relation_label_four_directions():
    ref = (50.0, 50.0)
    assert cp.relation_label((10.0, 55.0), ref) == "left"
    assert cp.relation_label((90.0, 45.0), ref) == "right"
    assert cp.relation_label((55.0, 10.0), ref) == "above"      # y 更小 = 更高
    assert cp.relation_label((45.0, 90.0), ref) == "below"
    assert cp.relation_axis((10.0, 55.0), ref) == "horizontal"
    assert cp.relation_axis((55.0, 10.0), ref) == "vertical"


def test_relation_label_ties():
    ref = (50.0, 50.0)
    # |dx| == |dy| → 归上下
    assert cp.relation_axis((40.0, 40.0), ref) == "vertical"
    assert cp.relation_label((40.0, 40.0), ref) == "above"
    assert cp.relation_label((60.0, 60.0), ref) == "below"
    # 质心完全重合：由 tie 参数决定，两个方向必须相反
    assert cp.relation_label(ref, ref, target_first_on_tie=True) == "above"
    assert cp.relation_label(ref, ref, target_first_on_tie=False) == "below"


@pytest.mark.parametrize("ct,cr", [((10.0, 55.0), (50.0, 50.0)), ((55.0, 10.0), (50.0, 50.0)),
                                   ((40.0, 40.0), (50.0, 50.0)), ((3.5, 7.25), (3.5, 9.0))])
def test_relation_label_twin_is_opposite(ct, cr):
    assert cp.relation_label(cr, ct) == cp.OPPOSITE[cp.relation_label(ct, cr)]
    assert cp.relation_axis(cr, ct) == cp.relation_axis(ct, cr)


def test_normalized_gap():
    a, b = [0, 0, 20, 10], [100, 0, 140, 30]           # 宽 20 / 40，高 10 / 30
    ca, cb = (10.0, 5.0), (120.0, 15.0)
    dist = (110.0 ** 2 + 10.0 ** 2) ** 0.5
    assert cp.normalized_gap(ca, cb, a, b, "horizontal") == pytest.approx(dist / 30.0)   # 平均宽 30
    assert cp.normalized_gap(ca, cb, a, b, "vertical") == pytest.approx(dist / 20.0)     # 平均高 20
    assert cp.normalized_gap_axis(ca, cb, a, b, "horizontal") == pytest.approx(110.0 / 30.0)
    assert cp.normalized_gap_axis(ca, cb, a, b, "vertical") == pytest.approx(10.0 / 20.0)
    # 对称
    assert cp.normalized_gap(cb, ca, b, a, "horizontal") == pytest.approx(cp.normalized_gap(ca, cb, a, b, "horizontal"))


def test_bbox_iou_and_containment():
    big, small = [0, 0, 100, 100], [10, 10, 30, 30]     # small 完全在 big 里
    assert cp.containment(small, big) == pytest.approx(1.0)
    assert cp.containment(big, small) == pytest.approx(400 / 10000)
    assert cp.bbox_iou(big, small) == pytest.approx(400 / 10000)
    a, b = [0, 0, 10, 10], [5, 0, 15, 10]               # 一半重叠
    assert cp.containment(a, b) == pytest.approx(0.5)
    assert cp.containment(b, a) == pytest.approx(0.5)
    assert cp.bbox_iou(a, b) == pytest.approx(50 / 150)
    c = [10, 0, 20, 10]                                 # 与 a 只共边
    assert cp.bbox_iou(a, c) == 0.0 and cp.containment(a, c) == 0.0
    assert cp.bbox_iou(a, [50, 50, 60, 60]) == 0.0
    assert cp.containment([5, 5, 5, 9], a) == 0.0       # 退化框不除零


def test_projection_overlap():
    a, b = [0, 0, 10, 10], [20, 5, 30, 15]              # x 投影分离，y 投影重叠 5
    assert cp.projection_overlap(a, b, "horizontal") is False
    assert cp.projection_overlap(a, b, "vertical") is True
    assert cp.projection_overlap_len(a, b, "vertical") == pytest.approx(5.0)
    assert cp.projection_overlap_frac(a, b, "vertical") == pytest.approx(0.5)
    assert cp.projection_overlap(a, [10, 0, 20, 10], "horizontal") is False     # 端点相接不算重叠
    assert cp.projection_overlap_frac([0, 0, 100, 100], [10, 10, 30, 30], "horizontal") == pytest.approx(1.0)
    assert cp.projection_overlap(b, a, "vertical") == cp.projection_overlap(a, b, "vertical")


def test_helpers():
    assert cp.xywh_to_xyxy([1, 2, 3, 4]) == [1.0, 2.0, 4.0, 6.0]
    assert cp.make_pair_id(632, 5, 7) == "000000000632_5_7"
    assert cp.bin_label(0.0, cp.GAP_BINS) == "<0.5" and cp.bin_label(0.5, cp.GAP_BINS) == "0.5-1"
    assert cp.bin_label(5.0, cp.GAP_BINS) == ">=5" and cp.bin_label(1e9, cp.GAP_BINS) == ">=5"
    assert [cp.bin_label(n, cp.CROWD_BINS) for n in (0, 1, 2, 3, 5, 6, 10, 11, 20, 21)] == \
        ["0", "1-2", "1-2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20", ">20"]
    assert cp.make_caption("apple", "dog", "left") == "A photo of an apple to the left of a dog"
    assert cp.make_caption("dog", "umbrella", "above") == "A photo of a dog above an umbrella"


def test_caption_parses_with_coco_spatial_template():
    from src.data.loaders import _parse_caption
    for rel in cp.RELATIONS:
        assert _parse_caption(cp.make_caption("dining table", "cup", rel)) == ("dining table", "cup", rel)


# ---------------------------------------------------------------- 划分完整性（文件不存在则跳过）
def test_split_integrity():
    if not cp.split_file().is_file():
        pytest.skip("splits/coco_pairs.json 不存在")
    sp = json.load(open(cp.split_file()))
    dev, test = sp["dev"], sp["test"]
    assert sp["seed"] == cp.SEED and sp["unit"] == "image" and sp["status"] == "active"
    assert dev == sorted(set(dev)) and test == sorted(set(test))
    assert not set(dev) & set(test)
    assert len(dev) == sp["n_dev_images"] and len(test) == sp["n_test_images"]
    assert len(dev) + len(test) == sp["n_images"]
    assert abs(len(dev) - len(test)) <= len(sp["per_stratum"])       # 每层最多差 1 张（v2 的层还包含只在 COCO-spatial 的图）
    if not cp.pairs_file().is_file():
        pytest.skip("pairs.jsonl 不存在，只做了划分文件自身的检查")
    recs = cp.read_pair_records()
    by_id = {r["pair_id"]: r for r in recs}
    assert len(by_id) == len(recs)
    assert {r["image_id"] for r in recs} == set(dev) | set(test)     # 每张有 pair 的图恰好在一侧
    side = {i: "dev" for i in dev} | {i: "test" for i in test}
    for r in recs:
        tw = by_id[r["twin_id"]]
        assert tw["twin_id"] == r["pair_id"] and tw["relation"] == cp.OPPOSITE[r["relation"]]
        assert side[tw["image_id"]] == side[r["image_id"]]           # twin 在同一侧
    for s in ("dev", "test"):
        n = sum(side[r["image_id"]] == s for r in recs)
        assert n == sp["counts"][s]["n_directed_pairs"]


def test_holdout_disjoint_from_eval_images():
    from src.data.splits import split_path
    f = split_path("coco_holdout")
    if not (f.is_file() and cp.split_file().is_file()):
        pytest.skip("splits/coco_holdout.json 或 splits/coco_pairs.json 不存在")
    from pathlib import Path

    from src.data import load_dataset
    ho = json.load(open(f))
    ids = ho["image_ids"]
    assert ids == sorted(set(ids)) and len(ids) == ho["counts"]["n_selected"] == len(ho["file_names"])
    sp = json.load(open(cp.split_file()))
    assert not set(ids) & (set(sp["dev"]) | set(sp["test"]))
    for name in ("coco_spatial_one", "coco_spatial_two"):
        assert not set(ids) & {int(Path(s.image_path).stem) for s in load_dataset(name)}
