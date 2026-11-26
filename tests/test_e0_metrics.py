"""E0 test 指标：框指标计算正确；输出里没有任何“框推关系”的量（E1 的 T1 检验在预注册前不得在 test 上算）。"""
import numpy as np
import pytest

from src import e0_metrics as M


def _vl():
    return {v: None for v in ["A", " A", "B", " B", "C", " C", "D", " D", "Yes", "yes", " Yes", " yes", "No", "no", " No", " no"]}


def _recs():
    base = {"model": "Qwen2.5-VL-7B-Instruct", "stage": "test", "answer_modes": {"mcq": "text16", "binary": "text16"}, "bbox_convention": "abs_model_xyxy",
            "orig_wh": [100, 100], "model_wh": [100, 100], "view_box": None, "error": None, "variant_logits": _vl()}
    R = []
    for i in range(4):
        img = f"whatsup_a:s{i}"
        R.append({**base, "dataset": "whatsup_a", "sample_id": f"s{i}", "image_key": img, "task": "box", "role": "target", "box_object": "cup",
                  "parsed_numbers": [10, 10, 30, 30] if i % 2 == 0 else [60, 60, 90, 90], "truth_xyxy": [10, 10, 30, 30], "relation": "left"})
        for rot, opts in enumerate([["left", "right", "on", "under"], ["right", "on", "under", "left"]]):
            R.append({**base, "dataset": "whatsup_a", "sample_id": f"s{i}", "image_key": img, "task": "mcq", "rotation": rot, "options": opts, "relation": "left",
                      "text": "A", "text_raw": "A", "text16": "A", "text16_raw": "A"})
        for asked, ans in (("left", "Yes"), ("right", "No")):
            R.append({**base, "dataset": "whatsup_a", "sample_id": f"s{i}", "image_key": img, "task": "binary", "asked_relation": asked, "relation": "left",
                      "text": ans, "text_raw": ans, "text16": ans, "text16_raw": ans})
    return R


def test_box_metrics_and_no_box_relation_outputs():
    out = M.summarize(_recs())
    d = out["whatsup_a"]
    assert d["iou@0.5"][0] == pytest.approx(0.5) and d["half_correct"][0] == pytest.approx(0.5)
    assert d["mcq_acc"][0] == pytest.approx(0.5)            # rotation 0 答 A=left 对；rotation 1 答 A=right 错
    assert d["binary_pair_acc"][0] == pytest.approx(1.0)
    assert d["center_err_obj_median"] is not None and d["center_err_tok_median"] is not None
    for k in d:
        assert not any(f in k for f in M.FORBIDDEN_KEYS), k
        assert "derived" not in k and "box_to_rel" not in k


def test_module_has_no_box_relation_code():
    """模块里没有任何由预测框推关系、算一致率或列联表的函数 / 代码（FORBIDDEN_KEYS 常量本身除外）。"""
    import inspect
    fns = [n for n, o in inspect.getmembers(M, inspect.isfunction) if o.__module__ == M.__name__]
    assert set(fns) == {"box_record_metrics", "summarize", "impl_diff"}
    body = "\n".join(l for l in inspect.getsource(M).split('"""', 2)[2].splitlines() if not l.startswith("FORBIDDEN_KEYS"))
    for word in ("relation_from_boxes", "box_relation", "contingency", "2x2", "crosstab", "pred_relation"):
        assert word not in body


def test_impl_diff():
    E = _recs()
    S = [dict(r) for r in E]
    S[0]["parsed_numbers"] = [11, 10, 30, 30]; S[0]["text_raw"] = "x"
    d = M.impl_diff(E, S)
    assert d["n_common"] == len(E) and d["box_identical_numbers"] == pytest.approx(0.75) and d["mcq_same_answer"] == 1.0
