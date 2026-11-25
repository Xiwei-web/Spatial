import numpy as np
from pycocotools import mask as mu

from src.annotate.masks import mask_fields, subtract_target_from_reference


def _rec(t, r):
    o = {k: {**mask_fields(m), "box_xyxy": [0, 0, 1, 1], "box_mask_xyxy": mask_fields(m)["mask_bbox_xyxy"]} for k, m in (("target", t), ("reference", r))}
    return {"objects": o}


def test_subtraction_changes_only_reference_mask():
    r = np.zeros((20, 30), bool); r[5:15, 5:25] = True
    t = np.zeros((20, 30), bool); t[3:8, 10:14] = True          # 部分在 reference 外
    rec = _rec(t, r)
    before_t = dict(rec["objects"]["target"])
    st = subtract_target_from_reference(rec)
    R = rec["objects"]["reference"]
    assert st["inter"] == 12 and R["mask_area"] == 200 - 12 == int(mu.area(R["mask_rle"]))
    assert R["mask_reference_raw"]["mask_area"] == 200
    assert rec["objects"]["target"] == before_t
    assert R["box_mask_xyxy"] == R["mask_bbox_xyxy"] == [5, 5, 25, 15] and R["box_xyxy"] == [0, 0, 1, 1]
    assert not (mu.decode(R["mask_rle"]).astype(bool) & t).any()


def test_no_intersection_is_noop():
    a = np.zeros((9, 9), bool); a[:3, :3] = True
    b = np.zeros((9, 9), bool); b[5:, 5:] = True
    rec = _rec(a, b)
    assert subtract_target_from_reference(rec) is None and "mask_reference_raw" not in rec["objects"]["reference"]


def test_qc_overlap_flag_default_on_and_can_be_frozen():
    from scripts.whatsup_qc import qc_one
    r = np.zeros((100, 100), bool); r[10:90, 10:90] = True
    t = np.zeros((100, 100), bool); t[40:50, 40:50] = True
    rec = _rec(t, r)
    for k in ("target", "reference"):
        o = rec["objects"][k]
        o.update(box_xyxy=o["mask_bbox_xyxy"], candidates=[{"box_xyxy": o["mask_bbox_xyxy"], "score": 0.9}], score=0.9, selected_candidate_rank=0)
    rec.update(width=100, height=100, dataset="whatsup_a", relation="on")
    q = {"low_score": 0.35, "ambiguous_margin": 0.1, "min_area_frac": 0.0002, "max_area_frac": 0.99, "max_pair_mask_iou": 0.5, "geometry": {}, "min_mask_box_iou": 0.8}
    assert "target_covered_by_reference" in qc_one(rec, q)["flags"]
    assert "target_covered_by_reference" not in qc_one(rec, dict(q, max_inter_over_target=None))["flags"]
    subtract_target_from_reference(rec)
    assert "target_covered_by_reference" not in qc_one(rec, q)["flags"]
