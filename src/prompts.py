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


def parse_mcq(text: str) -> str | None:
    """返回 A/B/C/D 或 None（解析不出就是 None，不猜）。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
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
    # <loc0123> 形式
    locs = re.findall(r"<loc(\d{4})>", t)
    if len(locs) >= 4:
        return [float(x) for x in locs[:4]]
    return None


def to_original_xyxy(nums: list[float], convention: str, orig_wh: tuple[int, int],
                     model_wh: tuple[int, int] | None = None) -> list[float]:
    """把模型输出的四个数还原为原图像素坐标 [x1, y1, x2, y2]。
    convention:
      norm1000_xyxy  —— 0-1000 归一化，x1,y1,x2,y2
      norm1000_yxyx  —— 0-1000 归一化，y1,x1,y2,x2
      norm1_xyxy     —— 0-1 归一化
      abs_model_xyxy —— 模型实际看到的（缩放后）图像上的绝对像素，需要 model_wh
      abs_orig_xyxy  —— 原图绝对像素
    """
    W, H = orig_wh
    a, b, c, d = nums
    if convention == "norm1000_xyxy":
        return [a / 1000 * W, b / 1000 * H, c / 1000 * W, d / 1000 * H]
    if convention == "norm1000_yxyx":
        return [b / 1000 * W, a / 1000 * H, d / 1000 * W, c / 1000 * H]
    if convention == "norm1_xyxy":
        return [a * W, b * H, c * W, d * H]
    if convention == "abs_model_xyxy":
        if model_wh is None:
            raise ValueError("abs_model_xyxy 需要 model_wh")
        mw, mh = model_wh
        return [a / mw * W, b / mh * H, c / mw * W, d / mh * H]
    if convention == "abs_orig_xyxy":
        return [a, b, c, d]
    raise ValueError(f"未知坐标约定 {convention}")
