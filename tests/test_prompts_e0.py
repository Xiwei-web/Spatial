"""E0 关系问法的模板与解析（手写字符串，不用任何模型输出）。"""
import pytest

from src.prompts import (E0_RELATION_OPTIONS, OPPOSITE, answer_mode_order, binary_question, choose_answer_mode, e0_mcq, parse_mcq, parse_mcq_e0,
                         parse_yes_no, relation_mcq)


def test_rotation0_equals_whatsup_template():
    for ds in ("whatsup_a", "whatsup_b"):
        assert e0_mcq(ds, "mug", "table", 0) == relation_mcq(ds, "mug", "table")


def test_rotations_are_cyclic_and_cover_each_letter():
    for ds, opts in E0_RELATION_OPTIONS.items():
        ref = None if ds.endswith("_one") else "chair"
        seen = [e0_mcq(ds, "cup", ref, r)[1] for r in range(4)]
        assert seen[0] == opts and seen[1] == opts[1:] + opts[:1] and e0_mcq(ds, "cup", ref, 4)[1] == opts
        for o in opts:                                   # 每个关系在 4 次轮换里恰好各占一次 A、B、C、D
            assert sorted(s.index(o) for s in seen) == [0, 1, 2, 3]


def test_two_object_text():
    p, o = e0_mcq("coco_spatial_two", "dog", "car", 1)
    assert p == ("Look at the image. Where is the dog relative to the car?\nA. The dog is to the right of the car.\nB. The dog is above the car.\n"
                 "C. The dog is below the car.\nD. The dog is to the left of the car.\nAnswer with the option's letter only.")
    assert o == ["right", "above", "below", "left"]
    p, _ = e0_mcq("gqa_spatial_two", "man", "horse", 2)
    assert "A. The man is in front of the horse." in p and "B. The man is behind the horse." in p


def test_one_object_text():
    p, o = e0_mcq("coco_spatial_one", "bus", None, 0)
    assert p == ("Look at the image. Where is the bus in the image?\nA. The bus is on the left side of the image.\nB. The bus is on the right side of the image.\n"
                 "C. The bus is at the top of the image.\nD. The bus is at the bottom of the image.\nAnswer with the option's letter only.")


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
    ("<think>no wait, maybe</think>Yes", "yes"), ("yes or no? no", "yes"), ("Yesterday", None), ("Yes-no", "yes"),
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
