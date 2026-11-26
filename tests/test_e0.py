"""E0：冻结校验、解析检查样本、答案推导与判对。"""
import numpy as np
import pytest

from src import e0

BB = "x: 1\ncalibrated:\n  M:\n    prompt_id: P0\n    convention: norm1000_xyxy\n"
E0Y = "answer_modes:\n  M:\n    mcq: text16\n    binary: text16\n"


def test_compare_frozen_ok_and_mismatch():
    cur = e0.blocks_from_text(BB, E0Y)
    h = e0.compare_frozen(cur, e0.blocks_from_text(BB + "y: 2\n", E0Y))          # 块外的改动不影响
    assert h["calibrated_sha"] == e0.block_hash({"M": {"prompt_id": "P0", "convention": "norm1000_xyxy"}})
    with pytest.raises(RuntimeError):
        e0.compare_frozen(cur, e0.blocks_from_text(BB.replace("norm1000_xyxy", "abs_model_xyxy"), E0Y))
    with pytest.raises(RuntimeError):
        e0.compare_frozen(cur, e0.blocks_from_text(BB, E0Y.replace("binary: text16", "binary: text64")))
    with pytest.raises(RuntimeError):
        e0.compare_frozen(cur, e0.blocks_from_text(BB, None))                    # 已提交版本还没有 answer_modes
    with pytest.raises(RuntimeError):
        e0.compare_frozen(e0.blocks_from_text("x: 1\n", None), e0.blocks_from_text("x: 1\n", None))     # 没有 calibrated 块


def test_check_frozen_on_repo_head():
    """当前仓库：工作区里 src/ scripts/ configs/ 干净时通过；否则报错（两种情况都是预期行为）。"""
    import subprocess
    from src import paths
    dirty = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain", "--untracked-files=no", "--", *e0.FROZEN_DIRS], capture_output=True, text=True).stdout.strip()
    if dirty:
        with pytest.raises(RuntimeError):
            e0.check_frozen()
    else:
        assert len(e0.check_frozen()["calibrated_sha"]) == 64


def test_parse_check_samples_dev_only_and_deterministic():
    from src.data import load_split
    for ds in e0.PARSE_CHECK_DATASETS:
        a, b = e0.parse_check_samples(ds), e0.parse_check_samples(ds)
        assert [s.sample_id for s in a] == [s.sample_id for s in b] and len(a) == 50 == len({s.sample_id for s in a})
        if ds != "coco_pairs":
            assert {s.sample_id for s in a} <= {s.sample_id for s in load_split(ds, "dev")}
        from src.prompts import E0_RELATION_OPTIONS
        assert all(s.relation in E0_RELATION_OPTIONS[ds] for s in a)


def _vl(**kw):
    base = {v: None for v in ["A", " A", "B", " B", "C", " C", "D", " D", "Yes", "yes", " Yes", " yes", "No", "no", " No", " no"]}
    base.update(kw)
    return base


def test_answer_and_correct():
    r = {"task": "mcq", "options": ["right", "left", "on", "under"], "relation": "left", "text": "B", "text_raw": "B<|im_end|>", "text16": "B", "text16_raw": "B<|im_end|>",
         "variant_logits": _vl(A=1.0, B=0.5, C=0.2, D=0.1, **{" B": 3.0})}
    assert e0.answer(r, "text16") == "B" and e0.correct(r, "B")
    assert e0.answer(r, "letter_logit") == "B"                               # " B" 的 3.0 最大
    r2 = {**r, "options": ["left", "right"], "text16": "C", "text16_raw": "C", "text": "C", "text_raw": "C"}
    assert e0.answer(r2, "text16") is None                                   # 二选一时 C 不算
    r3 = {**r, "text16": "The mug is", "text16_raw": "The mug is", "text": "The mug is to the right, so the answer is <|begin_of_box|>B<|end_of_box|>",
          "text_raw": "The mug is to the right, so the answer is <|begin_of_box|>B<|end_of_box|>"}
    assert e0.answer(r3, "text16") is None and e0.answer(r3, "text64") == "B"
    b = {"task": "binary", "relation": "left", "asked_relation": "right", "text": "No.", "text_raw": "No.<|im_end|>", "text16": "No.", "text16_raw": "No.",
         "variant_logits": _vl(Yes=2.0, **{" no": 2.5})}
    assert e0.answer(b, "text16") == "no" and e0.correct(b, "no") and not e0.correct(b, "yes")
    assert e0.answer(b, "letter_logit") == "no"
    assert e0.answer({**b, "error": "OOM"}, "text16") is None and not e0.correct(b, None)


def test_bootstrap_groups():
    lo, hi = e0.bootstrap_mean(np.ones(6), ["a", "a", "b", "b", "c", "c"])
    assert lo == hi == 1.0
