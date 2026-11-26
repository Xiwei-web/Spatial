"""抠图库 v2 过滤规则（手写数据）。"""
import numpy as np
import pytest
import yaml

from src import paths
from src.synth import library_v2 as L


def A(i, bbox, area=None, crowd=0):
    return {"id": i, "bbox": bbox, "area": area if area is not None else bbox[2] * bbox[3] * 0.6, "iscrowd": crowd}


def test_rule_a_bbox_overlap():
    c = A(1, [10, 10, 50, 50])
    assert L.rule_a(c, [c, A(2, [60, 10, 20, 20])])                 # 边相接（x=60）不算交集
    assert not L.rule_a(c, [c, A(2, [59, 10, 20, 20])])
    assert not L.rule_a(c, [c, A(3, [0, 0, 15, 15], crowd=1)])      # iscrowd 实例也算
    assert not L.rule_a(A(4, [10, 10, 5, 5], crowd=1), [])            # 候选本身不能是 iscrowd


def test_rule_b_border():
    assert L.rule_b(A(1, [2, 2, 96, 96]), 100, 100, 2)
    assert not L.rule_b(A(1, [1.9, 2, 50, 50]), 100, 100, 2)
    assert not L.rule_b(A(1, [2, 2, 96.5, 50]), 100, 100, 2)


def test_rule_c_quantiles():
    q = L.fill_quantiles({"cup": list(np.linspace(0.3, 0.9, 61))})
    lo, hi = q["cup"]
    assert lo == pytest.approx(0.36) and hi == pytest.approx(0.84)
    assert L.rule_c(A(1, [0, 0, 10, 10], area=50), "cup", q)
    assert not L.rule_c(A(1, [0, 0, 10, 10], area=35), "cup", q)
    assert not L.rule_c(A(1, [0, 0, 10, 10], area=85), "cup", q)
    assert not L.rule_c(A(1, [0, 0, 10, 10], area=50), "bowl", q)


def test_rule_d():
    cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "synth" / "library_v2.yaml"))["rules"]
    ok = L.rule_d(200, 100, 161 ** 2 / 1.5 ** 2 + 1, cfg)
    assert ok["long_side"] and ok["upscale"] and ok["aspect"]
    assert not L.rule_d(127, 100, 20000, cfg)["long_side"]
    assert not L.rule_d(400, 100, 20000, cfg)["aspect"]
    assert not L.rule_d(200, 100, 161 ** 2 / 1.5 ** 2 - 50, cfg)["upscale"]


def test_clip_decision_and_confusables():
    labels = L.clip_labels(["cup", "toilet", "tv"], ["urinal", "computer monitor", "tv"])
    assert labels == ["cup", "toilet", "tv", "urinal", "computer monitor"]
    assert L.clip_decision([0.1, 0.7, 0.1, 0.05, 0.05], labels, "toilet", 0.5)["pass"]
    d = L.clip_decision([0.05, 0.40, 0.05, 0.45, 0.05], labels, "toilet", 0.5)     # 易混标签 urinal 排在真类别之前
    assert not d["pass"] and d["rank_true"] == 1 and d["top1"] == "urinal"
    assert not L.clip_decision([0.1, 0.45, 0.2, 0.15, 0.1], labels, "toilet", 0.5)["pass"]       # 排第一但概率不足


def test_small_tier_scale():
    assert L.small_tier_scale(36 * 36, 18) == pytest.approx(0.5)


def test_config_excluded_categories_and_thresholds():
    cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "synth" / "library_v2.yaml"))
    assert set(cfg["exclude_categories"]) == {"bicycle", "motorcycle", "donut", "dining table", "banana", "carrot", "broccoli"}
    assert all(cfg["exclude_categories"].values())
    assert cfg["min_categories"] == 30 and cfg["min_per_side"] == 3 and cfg["small_tier_check"]["rendered_px"] == 18
    assert {"urinal", "computer monitor", "trailer", "van", "plate", "wok", "pitcher", "ATV", "wine glass"} <= set(cfg["clip"]["confusable_labels"])


def test_clip_decision_is_json_serializable():
    import json
    d = L.clip_decision(np.array([0.1, 0.7, 0.2]), ["a", "b", "c"], "b", 0.5)
    assert json.loads(json.dumps(d))["pass"] is True
