import json

from src import paths
from src.prompts import parse_four_numbers, parse_mcq, relation_mcq, to_original_xyxy


def test_parse_mcq():
    assert parse_mcq("A") == "A"
    assert parse_mcq("B. The mug is to the right") == "B"
    assert parse_mcq("(C)") == "C"
    assert parse_mcq("The answer is D.") == "D"
    assert parse_mcq("<think>maybe A</think>\nB") == "B"
    assert parse_mcq("<think></think><answer>C</answer>") == "C"
    assert parse_mcq("<|begin_of_box|>A<|end_of_box|>") == "A"
    assert parse_mcq("<answer>cat</answer>") is None
    assert parse_mcq("The glasses are located to the right of the armchair") is None
    assert parse_mcq("I cannot tell") is None
    assert parse_mcq("Apple") is None


def test_parse_four_numbers():
    assert parse_four_numbers('{"bbox_2d": [10, 20, 30, 40], "label": "mug"}') == [10, 20, 30, 40]
    assert parse_four_numbers("<box>[[100, 200, 300, 400]]</box>") == [100, 200, 300, 400]
    assert parse_four_numbers("[0.1, 0.2, 0.3, 0.4]") == [0.1, 0.2, 0.3, 0.4]
    assert parse_four_numbers("(12,34),(56,78)") == [12, 34, 56, 78]
    assert parse_four_numbers("<ref>knife</ref><box>213 765 274 843</box>") == [213, 765, 274, 843]
    assert parse_four_numbers("[270, 6") is None
    assert parse_four_numbers('{"bbox_2d":[0.03,0.25,2.29,0.19,0.07,0.19,0.51,0.41,0.51]}') is None
    assert parse_four_numbers("no box here") is None
    assert parse_four_numbers("[1, 2, 3]") is None


def test_coordinate_conventions():
    assert to_original_xyxy([0, 0, 500, 1000], "norm1000_xyxy", (200, 100)) == [0, 0, 100, 100]
    assert to_original_xyxy([0, 0, 500, 1000], "norm1000_yxyx", (200, 100)) == [0, 0, 200, 50]
    assert to_original_xyxy([0, 0, 0.5, 1], "norm1_xyxy", (200, 100)) == [0, 0, 100, 100]
    assert to_original_xyxy([0, 0, 50, 50], "abs_model_xyxy", (200, 100), model_wh=(100, 50)) == [0, 0, 100, 100]


def test_mcq_prompt_has_four_options():
    p, opts = relation_mcq("whatsup_b", "mug", "plate")
    assert len(opts) == 4 and all(f"{L}. " in p for L in "ABCD")


def test_splits_are_disjoint_halves_and_stratified():
    for f in sorted((paths.REPO_ROOT / "splits").glob("*.json")):
        d = json.load(open(f))
        if d.get("status") == "deprecated" or str(d.get("unit", "")).startswith("image"):
            continue      # 作废的划分；按图像继承的划分由 test_joint_coco_split 检查
        if "dev" not in d or "n_total" not in d:     # 只检查按样本分层的划分（coco_pairs 按图像划分，另有专门测试）
            continue
        dev, test = set(d["dev"]), set(d["test"])
        assert not dev & test
        assert len(dev) + len(test) == d["n_total"]
        strata = d.get("per_relation") or d["per_category"]          # 合成场景抠图库按类别分层，其余按关系分层
        assert abs(len(dev) - len(test)) <= len(strata)              # 每个层至多差 1


def test_corrections_are_excluded_from_splits_only():
    """更正表剔除的样本不出现在 load_split 里，但原始数据集仍完整（412 条）。"""
    from src.data import load_dataset, load_split
    from src.data.loaders import excluded_samples
    ex = {i for d, i in excluded_samples() if d == "whatsup_a"}
    assert ex == {"pillow_left_of_chair", "pillow_right_of_chair"}
    ids = {s.sample_id for sp in ("dev", "test") for s in load_split("whatsup_a", sp)}
    assert not ex & ids and len(ids) == 410
    assert len(load_dataset("whatsup_a")) == 412
