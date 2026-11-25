import json

from src import paths
from src.prompts import parse_four_numbers, parse_mcq, relation_mcq, to_original_xyxy


def test_parse_mcq():
    assert parse_mcq("A") == "A"
    assert parse_mcq("B. The mug is to the right") == "B"
    assert parse_mcq("(C)") == "C"
    assert parse_mcq("The answer is D.") == "D"
    assert parse_mcq("<think>maybe A</think>\nB") == "B"
    assert parse_mcq("I cannot tell") is None
    assert parse_mcq("Apple") is None


def test_parse_four_numbers():
    assert parse_four_numbers('{"bbox_2d": [10, 20, 30, 40], "label": "mug"}') == [10, 20, 30, 40]
    assert parse_four_numbers("<box>[[100, 200, 300, 400]]</box>") == [100, 200, 300, 400]
    assert parse_four_numbers("[0.1, 0.2, 0.3, 0.4]") == [0.1, 0.2, 0.3, 0.4]
    assert parse_four_numbers("(12,34),(56,78)") == [12, 34, 56, 78]
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
        if "dev" not in d:
            continue
        dev, test = set(d["dev"]), set(d["test"])
        assert not dev & test
        assert len(dev) + len(test) == d["n_total"]
        assert abs(len(dev) - len(test)) <= len(d["per_relation"])   # 每个关系类别至多差 1
