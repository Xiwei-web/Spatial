"""GQA 名称匹配规则与 SpatialRGPT 改写函数的单元测试（全部用合成输入，不依赖下载的数据）。"""
from src.data import gqa_boxes as gb
from src.data import spatialrgpt as sr


def _obj(name, attrs=(), x=0, y=0, w=10, h=10):
    return {"name": name, "attributes": list(attrs), "x": x, "y": y, "w": w, "h": h, "relations": []}


def _graph(*objs):
    return {"width": 100, "height": 100, "objects": {str(i): o for i, o in enumerate(objs)}}


# ---------- GQA 名称匹配 ----------
def test_exact_unique_gives_box_xyxy():
    m = gb.match_phrase("purse", _graph(_obj("purse", x=5, y=6, w=10, h=20), _obj("bag")))
    assert (m["status"], m["rule"], m["n_candidates"]) == ("unique", "exact", 1)
    assert m["bbox"] == [5.0, 6.0, 15.0, 26.0]


def test_exact_multiple_gives_no_box():
    m = gb.match_phrase("chair", _graph(_obj("chair"), _obj("chair", x=50)))
    assert (m["status"], m["n_candidates"], m["bbox"]) == ("multiple", 2, None)
    assert len(m["candidate_boxes"]) == 2


def test_none_and_missing_image():
    assert gb.match_phrase("mobile phone", _graph(_obj("cell phone")))["status"] == "none"   # 不做同义词
    assert gb.match_phrase("chair", _graph(_obj("chairs")))["status"] == "none"               # 不做单复数
    assert gb.match_phrase("meter", _graph(_obj("parking meter")))["status"] == "none"        # 不做部分匹配
    assert gb.match_phrase("chair", None)["status"] == "image-not-in-scene-graphs"


def test_attr_plus_name_only_when_exact_has_no_candidate():
    g = _graph(_obj("trees", ["green", "tall"]), _obj("trees", ["black"]), _obj("tree"))
    m = gb.match_phrase("green trees", g)
    assert (m["status"], m["rule"], m["n_same_name"]) == ("unique", "attr+name", 2)
    assert gb.match_phrase("tall green trees", g)["status"] == "unique"       # 多个属性都要有
    assert gb.match_phrase("red trees", g)["status"] == "none"
    # 多词名称：exact 优先，不会被拆成 属性 + 名称
    g2 = _graph(_obj("coffee table"), _obj("table", ["coffee"]))
    m2 = gb.match_phrase("coffee table", g2)
    assert (m2["rule"], m2["object_id"]) == ("exact", "0")
    # 属性 + 多词名称
    m3 = gb.match_phrase("wood coffee table", _graph(_obj("coffee table", ["wood"]), _obj("end table")))
    assert (m3["status"], m3["rule"]) == ("unique", "attr+name")


def test_attr_plus_name_multiple():
    g = _graph(_obj("towel", ["blue"]), _obj("towel", ["blue", "wet"]))
    assert gb.match_phrase("blue towel", g)["status"] == "multiple"


def test_matching_is_case_insensitive_and_deterministic():
    g = _graph(_obj("Purse"), _obj("cup"))
    assert gb.match_phrase(" purse ", g)["status"] == "unique"
    assert gb.find_candidates("chair", _graph(_obj("chair"), _obj("chair"))["objects"]) == ("exact", ["0", "1"])


def test_side_of():
    assert gb.side_of([0, 0, 20, 20], (100, 100), "left") is True
    assert gb.side_of([0, 0, 20, 20], (100, 100), "right") is False
    assert gb.side_of([0, 60, 20, 100], (100, 100), "bottom") is True
    assert gb.side_of([0, 0, 20, 20], (100, 100), "behind") is None


# ---------- SpatialRGPT：答案规范化 ----------
def test_parse_yes_no():
    for a in ("Yes.", "Yes, it is.", "That is True.", "Correct, Region [0] is larger.", "Indeed, Region [0] has ..."):
        assert sr.parse_yes_no(a) == "yes"
    for a in ("No.", "No, it is not.", "That is False.", "Incorrect, ...", "Actually, Region [0] is above.", "In fact, ..."):
        assert sr.parse_yes_no(a) == "no"
    assert sr.parse_yes_no("Region [0] is taller.") is None
    assert sr.parse_yes_no("Nothing to say") is None


def test_depth_relation_depends_on_answer():
    assert sr.depth_relation("behind_predicate", "Yes, Region [0] is behind Region [1].") == ("behind", "answer_yes")
    assert sr.depth_relation("behind_predicate", "No, it is in front of Region [1].") == ("front", "answer_no_explicit_opposite")
    assert sr.depth_relation("behind_predicate", "No, Region [0] is closer to the viewer.") == ("front", "answer_no_explicit_opposite")
    assert sr.depth_relation("behind_predicate", "That is False.") == ("front", "answer_no_implied_opposite")
    assert sr.depth_relation("front_predicate", "That is True.") == ("front", "answer_yes")
    assert sr.depth_relation("front_predicate", "No, Region [0] is further to the viewer.") == ("behind", "answer_no_explicit_opposite")
    assert sr.depth_relation("front_predicate", "No, it is not.") == ("behind", "answer_no_implied_opposite")
    assert sr.depth_relation("front_predicate", "Maybe.") == (None, "unparsed")


# ---------- SpatialRGPT：改写 ----------
def _rw(q, a, classes, **kw):
    n = len(set(sr._REGION.findall(q)))
    names, status = sr.region_names(classes, n, **kw)
    if names is None:
        return status, None, None
    return status, sr.substitute(q, names, classes), sr.substitute(a, names, classes)


def test_rewrite_two_regions_and_multiword():
    st, q, a = _rw("Is Region [0] behind Region [1]?", "No, it is in front of Region [1].", ["night stand", "lamp"])
    assert st == "ok"
    assert q == "Is the night stand behind the lamp?"
    assert a == "No, it is in front of the lamp."


def test_rewrite_sentence_initial_capital_and_one_region():
    st, q, a = _rw("How tall is Region [0]?", "Region [0] is 30.05 inches tall.", ["oven", "stove"])
    assert (st, q, a) == ("ok", "How tall is the oven?", "The oven is 30.05 inches tall.")


def test_rewrite_mask_form_uses_occurrence_order():
    names, _ = sr.region_names(["lamp", "television"], 2)
    assert sr.substitute("Can you confirm if <mask> is smaller than <mask>?", names, ["lamp", "television"]) == \
        "Can you confirm if the lamp is smaller than the television?"
    # 答案里顺序颠倒时按下标取
    assert sr.substitute("No, Region [1] is in front of Region [0].", names, ["lamp", "television"]) == \
        "No, the television is in front of the lamp."


def test_rewrite_same_class_skip_and_position():
    q, a = "Is Region [0] behind Region [1]?", "Yes."
    assert _rw(q, a, ["chair", "chair"])[0] == "ambiguous_same_class"
    kw = dict(bboxes=[[60, 0, 80, 10], [10, 0, 30, 10]], image_w=100, same_class="position")
    st, q2, _ = _rw(q, a, ["chair", "chair"], category="behind_predicate", **kw)
    assert (st, q2) == ("ok_position", "Is the chair on the right behind the chair on the left?")
    # 左右类问题不能用位置区分；两框水平距离不足也不区分
    assert _rw(q, a, ["chair", "chair"], category="left_predicate", **kw)[0] == "ambiguous_same_class"
    near = dict(bboxes=[[10, 0, 30, 10], [14, 0, 34, 10]], image_w=100, same_class="position")
    assert _rw(q, a, ["chair", "chair"], category="behind_predicate", **near)[0] == "ambiguous_same_class"


def test_rewrite_plural_agreement_and_possessive():
    st, q, a = _rw("Is Region [0] bigger than Region [1]?", "Yes, Region [0] is bigger than Region [1].", ["books", "chair"])
    assert q == "Are the books bigger than the chair?"
    assert a == "Yes, the books are bigger than the chair."
    _, q, a = _rw("Does Region [0] have a lesser width compared to Region [1]?",
                  "Incorrect, Region [0]'s width is not less than Region [1]'s.", ["shelves", "blinds"])
    assert q == "Do the shelves have a lesser width compared to the blinds?"
    assert a == "Incorrect, the shelves' width is not less than the blinds'."
    # 占位符不是主语时动词不动
    _, q, a = _rw("What is the height of Region [0]?", "The height of Region [0] is 3 feet.", ["books"])
    assert (q, a) == ("What is the height of the books?", "The height of the books is 3 feet.")
    # 单数但以 s 结尾的类名不受影响
    _, q, _ = _rw("Is Region [0] behind Region [1]?", "Yes.", ["bus", "glass"])
    assert q == "Is the bus behind the glass?"


def test_missing_class_and_articles():
    assert sr.region_names(["chair"], 2) == (None, "missing_class")
    assert [sr.indefinite_article(x) for x in ("oven", "air conditioner", "chair", "utensils", "hour glass")] == \
        ["an", "an", "a", "a", "an"]


def test_spatialrgpt_despeckle_and_box_exclusions():
    import numpy as np
    from pycocotools import mask as mu
    from src.data import spatialrgpt as sr
    m = np.zeros((100, 200), np.uint8, order="F"); m[10:60, 20:120] = 1; m[90, 190] = 1          # 5000 像素的物体 + 1 像素碎块
    rle = mu.encode(m); rle["counts"] = rle["counts"].decode()
    box, info = sr.despeckled_bbox(rle)
    assert box == [20.0, 10.0, 120.0, 60.0] and info["n_specks_removed"] == 1
    m[90:100, 150:200] = 1                                                                       # 500 像素 = 9% → 保留
    rle = mu.encode(m); rle["counts"] = rle["counts"].decode()
    assert sr.despeckled_bbox(rle)[0] == [20.0, 10.0, 200.0, 100.0]
    try:
        S = sr.depth_samples()
    except FileNotFoundError:
        return
    assert len(S) == 96
    ex = {(s.sample_id, r) for s in S for r in ("target", "reference") if s.extra[f"box_metric_exclude_{r}"]}
    assert len(ex) == 6 and len(sr.BOX_METRIC_EXCLUDE) == 6
    changed = sum(s.bbox_target != s.extra["bbox_target_all_pixels"] for s in S) + sum(s.bbox_reference != s.extra["bbox_reference_all_pixels"] for s in S)
    assert changed == 32
