"""提示词与解析。关系四选一提示词对所有模型相同；框提示词按模型家族的官方约定（见 configs/models/*.yaml 的 bbox 字段）。"""
from __future__ import annotations

import re

LETTERS = ["A", "B", "C", "D"]

# What's Up 两个子集各自的四个关系（顺序固定，选项字母不随样本变化）
RELATION_OPTIONS = {
    "whatsup_a": ["left", "right", "on", "under"],
    "whatsup_b": ["left", "right", "front", "behind"],
}
_PHRASE = {"left": "to the left of", "right": "to the right of", "on": "on", "under": "under",
           "front": "in front of", "behind": "behind"}


def relation_mcq(dataset: str, target: str, reference: str) -> tuple[str, list[str]]:
    opts = RELATION_OPTIONS[dataset]
    lines = [f"{L}. The {target} is {_PHRASE[o]} the {reference}." for L, o in zip(LETTERS, opts)]
    prompt = (f"Look at the image. Where is the {target} relative to the {reference}?\n"
              + "\n".join(lines) + "\nAnswer with the option's letter only.")
    return prompt, opts


# ---------------------------------------------------------------------------------------------------------------
# E0 用的关系问法（2026-11-26 写死，先于任何 dev 运行；planner 第 6 轮预审、第 7 轮修改；Xiwei 审批前不跑）。
# What's Up A/B：四选一（四个互斥选项）。COCO-spatial、GQA-spatial、COCO-Pairs：二选一——原始数据每条只标了一个轴
#（真关系与同轴反关系两个选项），四选一会出现两个正确答案（例如左上角的物体标 top，但 left 也成立），所以只给同轴的两个选项。
# 句式相同；选项顺序轮换：rotation=r 时选项列表循环左移 r 位（四选一 4 种、二选一 2 种），字母从 A 起；
# What's Up rotation=0 与原 relation_mcq 逐字相同。二选一的选项列表取该轴的固定顺序（见 AXES），不按真关系排，避免真答案总在 A。

E0_RELATION_OPTIONS = {        # 每个数据集允许的关系；不在其中的样本剔除
    "whatsup_a": ["left", "right", "on", "under"],
    "whatsup_b": ["left", "right", "front", "behind"],
    "coco_spatial_two": ["left", "right", "above", "below"],
    "coco_pairs": ["left", "right", "above", "below"],
    "gqa_spatial_two": ["left", "right", "front", "behind"],
    "coco_spatial_one": ["left", "right", "top", "bottom"],
    "gqa_spatial_one": ["left", "right", "top", "bottom"],
}
FOUR_WAY = {"whatsup_a", "whatsup_b"}
ONE_OBJECT = {"coco_spatial_one", "gqa_spatial_one"}
AXES = [("left", "right"), ("above", "below"), ("top", "bottom"), ("front", "behind"), ("on", "under")]
# 不在该数据集选项集合里的样本剔除，照实记录（planner 第 6 轮）：GQA one-object 的 1 条 behind（test）、GQA two-object 的 3 条 top（dev）
# GQA 的物体名照原样用（例如 "black trees" 这类复数带来的语法瑕疵不修，planner 第 7 轮）
PHRASE_TWO = {"left": "to the left of", "right": "to the right of", "on": "on", "under": "under", "front": "in front of", "behind": "behind",
              "above": "above", "below": "below"}
PHRASE_ONE = {"left": "on the left side of the image", "right": "on the right side of the image", "top": "at the top of the image", "bottom": "at the bottom of the image"}
OPPOSITE = {"left": "right", "right": "left", "on": "under", "under": "on", "front": "behind", "behind": "front", "above": "below", "below": "above",
            "top": "bottom", "bottom": "top"}
ANSWER_LETTER = "Answer with the option's letter only."
ANSWER_YES_NO = "Answer yes or no."


def n_choices(dataset: str) -> int:
    return 4 if dataset in FOUR_WAY else 2


def e0_options(dataset: str, relation: str | None = None) -> list[str]:
    """rotation=0 时的选项列表：What's Up 为四个关系；其余为真关系所在轴的两个关系（固定顺序）。"""
    if dataset in FOUR_WAY:
        return list(E0_RELATION_OPTIONS[dataset])
    if relation not in E0_RELATION_OPTIONS[dataset]:
        raise ValueError(f"{dataset}: 关系 {relation!r} 不在选项集合里（应已剔除）")
    return list(next(ax for ax in AXES if relation in ax))


def e0_mcq(dataset: str, target: str, reference: str | None, rotation: int = 0, relation: str | None = None) -> tuple[str, list[str]]:
    """→ (提示词, 本次的选项顺序 [A 对应的关系, B, …])。二选一的数据集必须给 relation（只用来确定轴）。"""
    opts = e0_options(dataset, relation)
    r = rotation % len(opts)
    opts = opts[r:] + opts[:r]
    if dataset in ONE_OBJECT:
        lines = [f"{L}. The {target} is {PHRASE_ONE[o]}." for L, o in zip(LETTERS, opts)]
        head = f"Look at the image. Where is the {target} in the image?"
    else:
        lines = [f"{L}. The {target} is {PHRASE_TWO[o]} the {reference}." for L, o in zip(LETTERS, opts)]
        head = f"Look at the image. Where is the {target} relative to the {reference}?"
    return head + "\n" + "\n".join(lines) + "\n" + ANSWER_LETTER, opts


def binary_question(dataset: str, target: str, reference: str | None, relation: str) -> str:
    """二元问法；relation 为被问的关系（真关系或 OPPOSITE[真关系]）。"""
    if dataset in ONE_OBJECT:
        return f"Is the {target} {PHRASE_ONE[relation]}? {ANSWER_YES_NO}"
    return f"Is the {target} {PHRASE_TWO[relation]} the {reference}? {ANSWER_YES_NO}"


def _unwrap(text: str) -> str:
    """去掉 <think>…</think>，以及 RoboBrain 的 <answer>…</answer>、GLM 的 <|begin_of_box|>…<|end_of_box|> 包装（取包装里的内容）。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    for a, b in ((r"<answer>", r"</answer>"), (r"<\|begin_of_box\|>", r"<\|end_of_box\|>")):
        m = re.search(a + r"\s*(.*?)\s*(?:" + b + r"|$)", t, flags=re.S)
        if m:
            return m.group(1).strip()
    return t


def parse_yes_no(text: str) -> str | None:
    """只认第一个独立的 yes / no 词（不区分大小写；“know”“nothing”“not” 不算）；先去掉 think 和包装，
    再删掉字面短语 “yes or no”（不区分大小写；模型复述问题时不会被判成 yes，planner 第 7 轮）。解析不出返回 None。"""
    t = re.sub(r"\byes\s+or\s+no\b", " ", _unwrap(text), flags=re.I)
    m = re.search(r"\b(yes|no)\b", t, flags=re.I)
    return m.group(1).lower() if m else None


def parse_mcq_e0(text: str) -> str | None:
    """E0 的四选一解析：先用 parse_mcq（与冒烟相同的规则，不改）；不行时再找文中任意位置的 <|begin_of_box|>X<|end_of_box|>（GLM 先推理、最后给框起来的字母）。"""
    r = parse_mcq(text)
    if r is not None:
        return r
    m = re.search(r"<\|begin_of_box\|>\s*\(?([ABCD])\)?\.?\s*<\|end_of_box\|>", text)
    return m.group(1) if m else None


# 关系问法的答案获取方式，按顺序尝试；dev 解析检查（每个数据集 50 个样本）上解析率 >= 0.95 就停在当前一步，否则进入下一步。
#   text16 = 生成 16 个新 token，parse_mcq_e0 / parse_yes_no；text64 = 生成 64 个新 token，同样的解析；
#   letter_logit = 第一步的 logit：每个字母取 "A"/" A" 变体的最大值后比较（letter_logit_answer）；二元比较 Yes 类与 No 类变体的最大值（yes_no_logit_answer）
# GLM-4.6V-Flash 从 text64 开始（冒烟时 16 个 token 内多半给不出字母）。所有模型都照样保存四个字母（及 Yes/No）的第一步 logit。
ANSWER_MODE_ORDER = {"default": ["text16", "text64", "letter_logit"], "GLM-4.6V-Flash": ["text64", "letter_logit"]}
PARSE_RATE_MIN = 0.95


YES_VARIANTS, NO_VARIANTS = ("Yes", "yes", " Yes", " yes"), ("No", "no", " No", " no")


def letter_variants(letter: str) -> tuple[str, str]:
    return (letter, " " + letter)


def variant_token_ids(tokenizer, variants) -> list[int]:
    """每个变体在分词器里恰好是一个 token 时取其 id；不存在（编码成多个 token）的变体跳过。去重，保持顺序。"""
    out = []
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def letter_logit_answer(first_logits, tokenizer, letters) -> str | None:
    """letter_logit 方式（四选一 / 二选一）：每个字母取 "A"、" A" 两种变体里 logit 最大的，再在字母之间取最大。某个字母一个变体都没有时返回 None。"""
    best = {}
    for L in letters:
        ids = variant_token_ids(tokenizer, letter_variants(L))
        if not ids:
            return None
        best[L] = max(float(first_logits[i]) for i in ids)
    return max(letters, key=lambda L: best[L])


def yes_no_logit_answer(first_logits, tokenizer) -> str | None:
    """letter_logit 方式（二元）：比较 {Yes, yes, " Yes", " yes"} 的最大 logit 与 {No, no, " No", " no"} 的最大 logit（不存在的变体跳过）。"""
    y, n = variant_token_ids(tokenizer, YES_VARIANTS), variant_token_ids(tokenizer, NO_VARIANTS)
    if not y or not n:
        return None
    return "yes" if max(float(first_logits[i]) for i in y) >= max(float(first_logits[i]) for i in n) else "no"


def answer_mode_order(model: str) -> list[str]:
    return ANSWER_MODE_ORDER.get(model, ANSWER_MODE_ORDER["default"])


def choose_answer_mode(model: str, dev_parse_rate: dict[str, float]) -> str | None:
    """dev_parse_rate：{方式: 解析率}，按 answer_mode_order 依次看，第一个 >= PARSE_RATE_MIN 的方式胜出；
    letter_logit 不需要解析，总能给出答案，排在最后；前面都没测或都不过时返回 letter_logit。"""
    for m in answer_mode_order(model):
        if m == "letter_logit":
            return m
        r = dev_parse_rate.get(m)
        if r is None:
            return None                       # 按顺序还没测到这一步
        if r >= PARSE_RATE_MIN:
            return m
    return None


def parse_mcq(text: str) -> str | None:
    """返回 A/B/C/D 或 None（解析不出就是 None，不猜）。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # 一些模型把答案包在固定的标记里：<answer>C</answer>（RoboBrain）、<|begin_of_box|>C<|end_of_box|>（GLM）
    w = re.match(r"^(?:<answer>|<\|begin_of_box\|>)\s*(.*?)\s*(?:</answer>|<\|end_of_box\|>|$)", t, flags=re.S)
    if w:
        t = w.group(1).strip()
    m = re.match(r"^\(?([ABCD])\)?(?:[\s\.\):,]|$)", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, flags=re.I)
    return m.group(1) if m else None


_NUM = r"-?\d+(?:\.\d+)?"


def parse_four_numbers(text: str) -> list[float] | None:
    """从生成文本里取出第一组四个坐标数字。先找括号/方括号里的四元组，再退回到 bbox_2d 之后的四个数。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(rf"[\[\(]\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*[\]\)]", t)
    if m:
        return [float(g) for g in m.groups()]
    # (x1,y1),(x2,y2) 形式
    m = re.search(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,?\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)", t)
    if m:
        return [float(g) for g in m.groups()]
    # <box>x1 y1 x2 y2</box>（空格分隔，MiniCPM-V）
    m = re.search(rf"<box>\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*</box>", t)
    if m:
        return [float(g) for g in m.groups()]
    # <loc0123> 形式
    locs = re.findall(r"<loc(\d{4})>", t)
    if len(locs) >= 4:
        return [float(x) for x in locs[:4]]
    return None


def to_original_xyxy(nums: list[float], convention: str, orig_wh: tuple[int, int],
                     model_wh: tuple[int, int] | None = None,
                     view_box: tuple[float, float, float, float] | None = None) -> list[float]:
    """把模型输出的四个数还原为原图像素坐标 [x1, y1, x2, y2]。
    convention:
      norm1000_xyxy  —— 0-1000 归一化，x1,y1,x2,y2
      norm1000_yxyx  —— 0-1000 归一化，y1,x1,y2,x2
      norm1_xyxy     —— 0-1 归一化
      abs_model_xyxy —— 模型实际看到的（缩放后）图像上的绝对像素，需要 model_wh
      abs_orig_xyxy  —— 原图绝对像素
      range_norm1_else_norm1000 —— 逐条按数值范围：都 <= 1 按 norm1_xyxy，否则 norm1000_xyxy
    """
    W, H = orig_wh
    a, b, c, d = nums
    if convention == "range_norm1_else_norm1000":     # 逐条：四个数都 <= 1 按 0-1 归一，否则按 0-1000 归一（dev 标定，SenseNova-SI-1.2/1.3）
        convention = "norm1_xyxy" if all(v <= 1 for v in nums) else "norm1000_xyxy"
    if convention == "norm1000_xyxy":
        return [a / 1000 * W, b / 1000 * H, c / 1000 * W, d / 1000 * H]
    if convention == "norm1000_yxyx":
        return [b / 1000 * W, a / 1000 * H, d / 1000 * W, c / 1000 * H]
    if convention == "norm1_xyxy":
        if view_box is not None:   # 模型只看到原图的一个子区域（如 CLIP 中心裁剪）：归一化坐标相对该区域
            x0, y0, x1, y1 = view_box
            return [x0 + a * (x1 - x0), y0 + b * (y1 - y0), x0 + c * (x1 - x0), y0 + d * (y1 - y0)]
        return [a * W, b * H, c * W, d * H]
    if convention == "abs_model_xyxy":
        if model_wh is None:
            raise ValueError("abs_model_xyxy 需要 model_wh")
        mw, mh = model_wh
        return [a / mw * W, b / mh * H, c / mw * W, d / mh * H]
    if convention == "abs_orig_xyxy":
        return [a, b, c, d]
    raise ValueError(f"未知坐标约定 {convention}")
