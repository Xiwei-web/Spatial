"""E0 关系问法的模板与解析（手写字符串，不用任何模型输出）。"""
import pytest

from src.prompts import (AXES, E0_RELATION_OPTIONS, FOUR_WAY, OPPOSITE, answer_mode_order, binary_question, choose_answer_mode, e0_mcq, e0_options,
                         letter_logit_answer, n_choices, parse_mcq, parse_mcq_e0, parse_yes_no, relation_mcq, variant_token_ids, yes_no_logit_answer)


def test_rotation0_equals_whatsup_template():
    for ds in ("whatsup_a", "whatsup_b"):
        assert e0_mcq(ds, "mug", "table", 0) == relation_mcq(ds, "mug", "table")


def test_rotations_are_cyclic_and_cover_each_letter():
    for ds, allowed in E0_RELATION_OPTIONS.items():
        ref = None if ds.endswith("_one") else "chair"
        for rel in allowed:
            n = n_choices(ds)
            seen = [e0_mcq(ds, "cup", ref, r, rel)[1] for r in range(n)]
            assert len(seen[0]) == n and seen[1] == seen[0][1:] + seen[0][:1] and e0_mcq(ds, "cup", ref, n, rel)[1] == seen[0]
            assert rel in seen[0]
            for o in seen[0]:                            # 每个选项在轮换里恰好各占一次每个字母
                assert sorted(s.index(o) for s in seen) == list(range(n))


def test_two_choice_datasets_use_same_axis_only():
    for ds in set(E0_RELATION_OPTIONS) - FOUR_WAY:
        assert n_choices(ds) == 2
        for rel in E0_RELATION_OPTIONS[ds]:
            o = e0_options(ds, rel)
            assert sorted(o) == sorted([rel, OPPOSITE[rel]]) and tuple(o) in AXES      # 固定顺序，不随真关系变
            assert e0_options(ds, OPPOSITE[rel]) == o
    with pytest.raises(ValueError):
        e0_options("gqa_spatial_one", "behind")
    with pytest.raises(ValueError):
        e0_options("coco_pairs", None)
    assert n_choices("whatsup_a") == n_choices("whatsup_b") == 4


def test_two_object_text():
    p, o = e0_mcq("coco_spatial_two", "dog", "car", 1, "left")
    assert p == ("Look at the image. Where is the dog relative to the car?\nA. The dog is to the right of the car.\nB. The dog is to the left of the car.\n"
                 "Answer with the option's letter only.")
    assert o == ["right", "left"]
    p, o = e0_mcq("coco_pairs", "bed", "chair", 0, "below")
    assert p == ("Look at the image. Where is the bed relative to the chair?\nA. The bed is above the chair.\nB. The bed is below the chair.\n"
                 "Answer with the option's letter only.") and o == ["above", "below"]
    p, _ = e0_mcq("gqa_spatial_two", "man", "horse", 0, "behind")
    assert "A. The man is in front of the horse.\nB. The man is behind the horse.\nAnswer" in p


def test_one_object_text():
    p, o = e0_mcq("coco_spatial_one", "bus", None, 0, "top")
    assert p == ("Look at the image. Where is the bus in the image?\nA. The bus is at the top of the image.\nB. The bus is at the bottom of the image.\n"
                 "Answer with the option's letter only.")
    assert e0_mcq("gqa_spatial_one", "purse", None, 1, "right")[1] == ["right", "left"]


def test_binary_text_and_opposites():
    assert binary_question("whatsup_a", "mug", "table", "on") == "Is the mug on the table? Answer yes or no."
    assert binary_question("whatsup_b", "cup", "bowl", "front") == "Is the cup in front of the bowl? Answer yes or no."
    assert binary_question("coco_pairs", "bed", "chair", OPPOSITE["below"]) == "Is the bed above the chair? Answer yes or no."
    assert binary_question("gqa_spatial_one", "tree", None, "top") == "Is the tree at the top of the image? Answer yes or no."
    for ds, opts in E0_RELATION_OPTIONS.items():
        for o in opts:
            assert OPPOSITE[o] in opts and OPPOSITE[OPPOSITE[o]] == o


@pytest.mark.parametrize("text,want", [
    ("Yes", "yes"), ("no.", "no"), ("Yes, the mug is on the table.", "yes"), ("No, it is not.", "no"), ("NO", "no"),
    ("The answer is yes.", "yes"), ("I don't know.", None), ("Nothing is there", None), ("It is not on it", None), ("", None),
    ("<think></think><answer>Yes</answer><|im_end|>", "yes"), ("<answer>No</answer>", "no"),
    ("<|begin_of_box|>No<|end_of_box|>", "no"), ("Looking closely, <|begin_of_box|>yes<|end_of_box|>", "yes"),
    ("<think>no wait, maybe</think>Yes", "yes"), ("Yesterday", None), ("Yes-no", "yes"),
    ("yes or no? no", "no"), ("Yes or No: No, it is not.", "no"), ("Answer yes or no. Yes.", "yes"), ("YES OR NO", None), ("yes  or\nno — yes", "yes"),
])
def test_parse_yes_no(text, want):
    assert parse_yes_no(text) == want


@pytest.mark.parametrize("text,want", [
    ("C", "C"), ("(B)", "B"), ("A.", "A"), ("<answer>D</answer>", "D"), ("<|begin_of_box|>B<|end_of_box|>", "B"),
    ("The cap is positioned behind the headphones, therefore, the answer is <|begin_of_box|>D<|end_of_box|>", "D"),
    ("The mug is located to the right of the table, as observed in the", None), ("The answer is C", "C"), ("E", None),
])
def test_parse_mcq_e0(text, want):
    assert parse_mcq_e0(text) == want


def test_parse_mcq_unchanged_for_mid_text_box():
    """原 parse_mcq 不改：句中的框起来的字母它不认，parse_mcq_e0 才认。"""
    t = "so it is <|begin_of_box|>A<|end_of_box|>"
    assert parse_mcq(t) is None and parse_mcq_e0(t) == "A"


def test_answer_mode_order_and_choice():
    assert answer_mode_order("GLM-4.6V-Flash") == ["text64", "letter_logit"]
    assert answer_mode_order("Qwen2.5-VL-7B-Instruct") == ["text16", "text64", "letter_logit"]
    assert choose_answer_mode("GLM-4.6V-Flash", {"text64": 0.97}) == "text64"
    assert choose_answer_mode("GLM-4.6V-Flash", {"text64": 0.90}) == "letter_logit"
    assert choose_answer_mode("InternVL3-8B", {"text16": 0.99}) == "text16"
    assert choose_answer_mode("InternVL3-8B", {"text16": 0.80}) is None            # 还要先测 text64
    assert choose_answer_mode("InternVL3-8B", {"text16": 0.80, "text64": 0.95}) == "text64"
    assert choose_answer_mode("InternVL3-8B", {"text16": 0.80, "text64": 0.94}) == "letter_logit"


class FakeTok:
    """手写的分词器：只有 vocab 里的字符串编码成单个 token，其余编码成两个 token（视为不存在）。"""

    def __init__(self, vocab):
        self.v = {t: i for i, t in enumerate(vocab)}

    def encode(self, text, add_special_tokens=False):
        return [self.v[text]] if text in self.v else [990, 991]


def test_variant_ids_skip_missing():
    tk = FakeTok(["A", " A", "B", "Yes", " no"])
    assert variant_token_ids(tk, ("A", " A")) == [0, 1]
    assert variant_token_ids(tk, ("B", " B")) == [2]
    assert variant_token_ids(tk, ("C", " C")) == []
    assert variant_token_ids(tk, ("Yes", "yes", " Yes", " yes")) == [3]


def test_letter_logit_answer_takes_max_over_variants():
    tk = FakeTok(["A", " A", "B", " B", "C", "D", " D"])
    lg = [0.0] * 8
    lg[0], lg[1] = 1.0, 5.0          # A：取 " A" 的 5
    lg[2], lg[3] = 4.0, 2.0          # B：取 "B" 的 4
    lg[4] = 4.5                       # C：只有 "C"
    lg[5], lg[6] = -1.0, 0.0
    assert letter_logit_answer(lg, tk, ["A", "B", "C", "D"]) == "A"
    lg[4] = 6.0
    assert letter_logit_answer(lg, tk, ["A", "B", "C", "D"]) == "C"
    assert letter_logit_answer(lg, tk, ["A", "B"]) == "A"
    assert letter_logit_answer(lg, FakeTok(["A"]), ["A", "B"]) is None


def test_yes_no_logit_answer():
    tk = FakeTok(["Yes", " yes", "No", " No"])          # "yes"、" Yes"、"no"、" no" 不存在 → 跳过
    lg = [0.0, 3.0, 2.5, 1.0]
    assert yes_no_logit_answer(lg, tk) == "yes"
    lg = [0.0, 1.0, 2.5, 1.0]
    assert yes_no_logit_answer(lg, tk) == "no"
    assert yes_no_logit_answer(lg, FakeTok(["Yes"])) is None
