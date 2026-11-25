"""SpatialRGPT-Bench（HF: a8cheng/SpatialRGPT-Bench，固定 commit，见 scripts/download_data.sh spatialrgpt）。
仅用于评测，不得再分发；HF 卡片未声明许可。

读取方式：标注来自快照里的 `SpatialRGPT-Bench_v1.json`，图像用快照里的 `images/<id>.png`（文件路径，不从 parquet 取 PIL）。
`images/` 覆盖全部 1,406 条（见 docs/parts/F7_spatialrgpt.md 的核对）。

本文件三部分：
  1. load_records()   全部 1,406 条的原始记录（问题两种写法、答案、类别、区域的框 / RLE mask、图像来源）。
  2. depth_samples()  front/behind 两个类别映射到统一的 `Sample`。
  3. rewrite_*        把 "Region [k]" / "<mask>" 换成物体名的规则（**只是设计稿**，没有对全量数据落盘；规则待确认）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src import paths
from src.annotate.masks import rle_bbox
from src.data.loaders import Sample

REVISION = "01b38d2c7aa805223518c061f2a7602ad8f491ef"
ANNOTATION_FILE = "SpatialRGPT-Bench_v1.json"
DEPTH_CATEGORIES = {"front_predicate": "front", "behind_predicate": "behind"}
_OPPOSITE = {"front": "behind", "behind": "front"}
_REGION = re.compile(r"Region \[(\d+)\]")


def root() -> Path:
    return paths.get("data") / "depth" / "spatialrgpt_bench"


@dataclass
class Record:
    id: str
    image_path: str                 # 本地 images/<id>.png
    image_source: str               # image_info.dataset：SUNRGBD / hypersim / nuScenes / ARKitScenes / KITTI_object
    image_wh: tuple[int, int]       # image_info 里的 (width, height)
    qa_type: str                    # qualitative / quantitative
    category: str                   # 例如 behind_predicate、left_choice、distance_data
    classes: list[str]              # qa_info.class 原样（单区域题里有 168 条给了 2 个类名）
    n_regions: int                  # 问题里引用的区域个数（= 框数 = mask 数）
    question_mask: str              # conversations[0] 去掉开头 "<image>\n " 之后的写法，区域写成 <mask>
    question_region: str            # text_q，区域写成 "Region [k]"
    answer: str                     # conversations[1]（答案里也可能出现 "Region [k]"）
    bboxes: list[list[float]]       # 每个区域一个 [x1, y1, x2, y2]，原图像素
    rles: list[dict]                # 每个区域一个 COCO 压缩 RLE：{"size": [h, w], "counts": str}
    extra: dict = field(default_factory=dict)

    def region_class(self, k: int) -> str:
        """第 k 个区域的类名：按 qa_info.class[k]（顺序对应是数据集的约定，本仓库只做了抽样目检）。"""
        return self.classes[k]


@lru_cache(maxsize=1)
def _raw() -> list[dict]:
    return json.load(open(root() / ANNOTATION_FILE))


def load_records() -> list[Record]:
    out = []
    for d in _raw():
        conv_q, conv_a = d["conversations"][0]["value"], d["conversations"][1]["value"]
        q_mask = conv_q.split("\n", 1)[1].strip() if conv_q.startswith("<image>") else conv_q
        info = d["image_info"]
        out.append(Record(
            id=d["id"], image_path=str(root() / "images" / f"{d['id']}.png"), image_source=info["dataset"],
            image_wh=(info["width"], info["height"]), qa_type=d["qa_info"]["type"], category=d["qa_info"]["category"],
            classes=list(d["qa_info"]["class"]), n_regions=len(set(_REGION.findall(d["text_q"]))),
            question_mask=q_mask, question_region=d["text_q"], answer=conv_a,
            bboxes=[[float(v) for v in b] for b in d["bbox"]], rles=list(d["rle"]),
            extra={"source_file_path": info["file_path"], "landmark": info.get("landmark"), "conversation_q_raw": conv_q},
        ))
    return out


# ---------- 标准答案的规范化 ----------
_YES = ("yes", "correct", "indeed", "that is true")
_NO = ("no", "incorrect", "actually", "in fact", "that is false")


def parse_yes_no(answer: str) -> str | None:
    """*_predicate 类的标准答案 → 'yes' / 'no'；只看开头的词，认不出返回 None（不猜）。"""
    a = answer.strip().lower()
    for label, heads in (("yes", _YES), ("no", _NO)):
        if any(re.match(re.escape(h) + r"(\W|$)", a) for h in heads):
            return label
    return None


def depth_relation(category: str, answer: str) -> tuple[str | None, str]:
    """front/behind 题的真实关系（Region [0] 相对 Region [1]）及其依据。

    问题模板都是「Region [0] 是否在 Region [1] 的 前面/后面（或更近/更远）」，答案是 yes/no：
      yes → 关系就是该类别问的那个（front_predicate → front，behind_predicate → behind），依据 'answer_yes'；
      no  → 取反。答案原文明说了相反关系（"No, it is in front of Region [1]."、"No, Region [0] is closer to the viewer."）
            时依据为 'answer_no_explicit_opposite'；只有 "No." / "That is False." / "No, it is not." 时依据为
            'answer_no_implied_opposite'（基准按深度二分，但原文没有明说；严格地说 “不在后面” 不等于 “在前面”）。
    yes/no 认不出时返回 (None, 'unparsed')。
    """
    asked = DEPTH_CATEGORIES[category]
    yn = parse_yes_no(answer)
    if yn is None:
        return None, "unparsed"
    if yn == "yes":
        return asked, "answer_yes"
    opp = _OPPOSITE[asked]
    words = {"front": ("in front of", "closer"), "behind": ("behind", "further")}[opp]
    explicit = any(w in answer.lower() for w in words)
    return opp, "answer_no_explicit_opposite" if explicit else "answer_no_implied_opposite"


_REL_TEXT = {"front": "in front of", "behind": "behind"}


def depth_samples(include_same_class: bool = False) -> list[Sample]:
    """front_predicate / behind_predicate → 统一 Sample。深度检验只用本数据集（What's Up B 的 front/behind 与垂直位置完全混淆）。

    用户 2026-11-25 的规则：
    - 真值框 = mask 外接框（bbox_*）；数据集自带的框在 extra["provided_bbox_target" / "provided_bbox_reference"]，只作对照；
    - 两个区域类名相同的 14 条跳过（include_same_class=True 时才返回）；
    - 标准答案只有 “No.” 一类、真实关系靠“否定即相反”推出的 34 条保留，extra["relation_inferred_from_bare_no"]=True，必须单独报告。

    target = Region [0] 的类名，reference = Region [1] 的类名（两类模板里主语都是 Region [0]，参照物都是 Region [1]）；
    relation ∈ {front, behind} 由 depth_relation 给出；bbox / mask 按区域下标取。
    caption 用区域写法（"Region [0] is behind Region [1]"），没有做名称改写。关系认不出的样本不返回（目前为 0 条）。
    """
    out = []
    for r in load_records():
        if r.category not in DEPTH_CATEGORIES:
            continue
        rel, src = depth_relation(r.category, r.answer)
        if rel is None or r.n_regions != 2:
            continue
        t, ref = r.region_class(0), r.region_class(1)
        same = t.strip().lower() == ref.strip().lower()
        if same and not include_same_class:
            continue
        cap = lambda x: f"Region [0] is {_REL_TEXT[x]} Region [1]"   # noqa: E731
        out.append(Sample(
            dataset="spatialrgpt_depth", sample_id=r.id, image_path=r.image_path, target=t, reference=ref, relation=rel,
            caption_true=cap(rel), caption_options=[cap(rel), cap(_OPPOSITE[rel])],
            bbox_target=rle_bbox(r.rles[0]), bbox_reference=rle_bbox(r.rles[1]), mask_target=r.rles[0], mask_reference=r.rles[1],
            extra={"category": r.category, "question_mask": r.question_mask, "question_region": r.question_region,
                   "answer": r.answer, "gold_yes_no": parse_yes_no(r.answer), "relation_source": src,
                   "relation_inferred_from_bare_no": src == "answer_no_implied_opposite",
                   "provided_bbox_target": r.bboxes[0], "provided_bbox_reference": r.bboxes[1],
                   "image_source": r.image_source, "same_class": same},
        ))
    return out


# ---------- 改写规则（设计稿）：Region [k] / <mask> → 物体名 ----------
# 88 个类名里的复数类名（固定名单，不靠 “以 s 结尾” 猜：glass / bus / soundsystem 是单数）。
PLURAL_CLASSES = frozenset({"books", "shelves", "blinds", "drawers", "clothes", "tissues", "toys", "plates",
                            "utensils", "shoes", "electronics"})
# 同类名时不能用左右位置来区分的类别（会直接泄露或干扰答案）
POSITION_LEAK_CATEGORIES = frozenset({"left_choice", "right_choice", "left_predicate", "right_predicate",
                                      "horizontal_distance_data", "direction"})
MIN_SEPARATION = 0.10      # 同类名用左右区分时，两个框中心的水平距离至少占图宽的比例
_PLACEHOLDER = re.compile(r"Region \[(\d+)\]|<mask>")
_AGREE = {"is": "are", "does": "do", "has": "have"}
_NON_SUBJECT_PREP = ("of", "from", "than", "between", "and", "to", "at", "find", "or")


def indefinite_article(name: str) -> str:
    """a / an（规则本身用定冠词 the，这个函数只在将来需要不定冠词的模板里用）。按首字母元音判断，带少量例外。"""
    w = name.strip().lower()
    if w.startswith(("hour", "honest", "heir")):
        return "an"
    if w.startswith(("uni", "use", "user", "utensil", "one", "euro")):
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def region_names(classes: list[str], n_regions: int, *, bboxes: list[list[float]] | None = None,
                 image_w: float | None = None, category: str | None = None,
                 same_class: str = "skip") -> tuple[list[str] | None, str]:
    """给每个区域一个带定冠词的名词短语。返回 (短语列表或 None, 状态)。

    状态：ok / ok_position（同类名，靠左右位置区分）/ ambiguous_same_class（同类名且不允许或无法区分）/
          missing_class（类名个数少于区域数）。
    same_class='skip'（默认）：两个区域类名相同就不改写；
    same_class='position'：改成 "the chair on the left" / "the chair on the right"，条件是该类别不在
      POSITION_LEAK_CATEGORIES 里，且两个框中心的水平距离 ≥ MIN_SEPARATION × 图宽；否则仍不改写。
    """
    if len(classes) < n_regions:
        return None, "missing_class"
    names = [c.strip() for c in classes[:n_regions]]
    if n_regions == 2 and names[0].lower() == names[1].lower():
        if same_class != "position" or category in POSITION_LEAK_CATEGORIES or not bboxes or not image_w:
            return None, "ambiguous_same_class"
        cx = [(b[0] + b[2]) / 2 for b in bboxes[:2]]
        if abs(cx[0] - cx[1]) < MIN_SEPARATION * image_w:
            return None, "ambiguous_same_class"
        sides = ["left", "right"] if cx[0] < cx[1] else ["right", "left"]
        return [f"the {n} on the {s}" for n, s in zip(names, sides)], "ok_position"
    return [f"the {n}" for n in names], "ok"


def substitute(text: str, names: list[str], classes: list[str]) -> str:
    """把 text 里的 "Region [k]"（按 k）或 "<mask>"（按出现顺序，第 k 个 ↔ 区域 k）换成 names[k]。

    - 句首大写；
    - 类名在 PLURAL_CLASSES 里时，紧挨着占位符的 is/does/has 改成 are/do/have
      （占位符前面是 of/from/than/and 等介词或连词时，它不是主语，后面的动词不动）；
    - 所有格：复数类名 "the books's" → "the books'"。
    答案里的代词（it / They）不改。
    """
    order = iter(range(len(names)))
    spans = []
    for m in _PLACEHOLDER.finditer(text):
        k = int(m.group(1)) if m.group(1) is not None else next(order)
        spans.append((m.start(), m.end(), k))
    out, pos = [], 0
    for s, e, k in spans:
        before, name = text[pos:s], names[k]
        plural = classes[k].strip().lower() in PLURAL_CLASSES
        after_start = e
        if plural:
            # 占位符前面紧挨的助动词（"Is R ..." / "How tall is R"）
            mb = re.search(r"\b(Is|is|Does|does|Has|has)(\s+)$", before)
            if mb:
                v = mb.group(1)
                new = _AGREE[v.lower()]
                before = before[: mb.start()] + (new.capitalize() if v[0].isupper() else new) + mb.group(2)
            # 占位符后面紧挨的动词（"R is ..." / "R has ..."），占位符不是主语时跳过
            prev_word = re.search(r"(\w+)\s+$", before)
            is_subject = not (prev_word and prev_word.group(1).lower() in _NON_SUBJECT_PREP)
            ma = re.match(r"(\s+)(is|does|has)\b", text[e:])
            if ma and is_subject:
                name += ma.group(1) + _AGREE[ma.group(2)]
                after_start = e + ma.end()
            elif text[e:e + 2] == "'s":
                name += "'"
                after_start = e + 2
        prefix = "".join(out) + before
        sentence_start = not prefix.strip() or re.search(r"[.?!]\s*$", prefix) is not None
        out.append(before + (name[0].upper() + name[1:] if sentence_start else name))
        pos = after_start
    out.append(text[pos:])
    return "".join(out)


def rewrite_record(r: Record, same_class: str = "skip") -> dict:
    """对一条记录应用改写规则。返回 status、改写后的问题和答案（不可改写时为 None）以及备注。"""
    names, status = region_names(r.classes, r.n_regions, bboxes=r.bboxes, image_w=r.image_wh[0],
                                 category=r.category, same_class=same_class)
    notes = []
    if len(r.classes) > r.n_regions:
        notes.append(f"class 有 {len(r.classes)} 个、区域只有 {r.n_regions} 个：按 class[0] 取")
    if any(c.strip().lower() in PLURAL_CLASSES for c in r.classes[: r.n_regions]):
        notes.append("含复数类名（紧邻的 is/does/has 按规则改成复数）")
    if names is None:
        return {"status": status, "question": None, "answer": None, "notes": notes}
    if re.search(r"\b(it|It|They|they)\b", r.answer):
        notes.append("答案含代词（it/They），未改")
    return {"status": status, "question": substitute(r.question_region, names, r.classes),
            "answer": substitute(r.answer, names, r.classes), "notes": notes}


# ---------- 统计 ----------
def stats() -> dict:
    import collections

    recs = load_records()
    per_cat = collections.Counter(r.category for r in recs)
    ok_counts = sum(len(r.bboxes) == len(r.rles) == r.n_regions == r.question_mask.count("<mask>") for r in recs)
    depth = depth_samples(include_same_class=True)
    used = depth_samples()
    by_cat_rel = collections.Counter((s.extra["category"], s.extra["gold_yes_no"], s.relation, s.extra["relation_source"]) for s in depth)
    rw = collections.Counter()
    rw_pos = collections.Counter()
    plural = pronoun_plural = extra_class = 0
    for r in recs:
        rw[rewrite_record(r)["status"]] += 1
        rw_pos[rewrite_record(r, same_class="position")["status"]] += 1
        is_pl = any(c.strip().lower() in PLURAL_CLASSES for c in r.classes[: r.n_regions])
        plural += is_pl
        pronoun_plural += bool(is_pl and re.search(r"\b(it|It)\b", r.answer))
        extra_class += len(r.classes) > r.n_regions
    same_all = [r for r in recs if r.n_regions == 2 and r.classes[0].strip().lower() == r.classes[1].strip().lower()]
    pred = [r for r in recs if r.category.endswith("_predicate")]
    return {
        "n_records": len(recs), "n_unique_ids": len({r.id for r in recs}),
        "n_unique_source_images": len({r.extra["source_file_path"] for r in recs}),
        "per_category": dict(sorted(per_cat.items())),
        "per_type": dict(collections.Counter(r.qa_type for r in recs)),
        "n_regions": dict(collections.Counter(r.n_regions for r in recs)),
        "n_bbox_mask_count_equals_regions": ok_counts,
        "n_with_bbox": sum(len(r.bboxes) > 0 for r in recs), "n_with_mask": sum(len(r.rles) > 0 for r in recs),
        "image_source_all": dict(collections.Counter(r.image_source for r in recs)),
        "depth": {
            "n": len(depth),
            "by_category": dict(collections.Counter(s.extra["category"] for s in depth)),
            "by_true_relation": dict(collections.Counter(s.relation for s in depth)),
            "by_category_answer_relation_source": {" | ".join(map(str, k)): v for k, v in sorted(by_cat_rel.items())},
            "relation_source": dict(collections.Counter(s.extra["relation_source"] for s in depth)),
            "image_source": dict(collections.Counter(s.extra["image_source"] for s in depth)),
            "same_class": sum(s.extra["same_class"] for s in depth),
            "used_after_skipping_same_class": {
                "n": len(used), "by_true_relation": dict(collections.Counter(s.relation for s in used)),
                "n_inferred_from_bare_no": sum(s.extra["relation_inferred_from_bare_no"] for s in used),
                "n_explicit": sum(not s.extra["relation_inferred_from_bare_no"] for s in used),
                "explicit_by_relation": dict(collections.Counter(s.relation for s in used if not s.extra["relation_inferred_from_bare_no"])),
                "inferred_by_relation": dict(collections.Counter(s.relation for s in used if s.extra["relation_inferred_from_bare_no"]))},
            "same_class_by_category": dict(collections.Counter(s.extra["category"] for s in depth if s.extra["same_class"])),
            "n_with_both_bbox": sum(s.bbox_target is not None and s.bbox_reference is not None for s in depth),
            "n_with_both_mask": sum(s.mask_target is not None and s.mask_reference is not None for s in depth),
        },
        "predicate_yes_no": {"n": len(pred), "parsed": sum(parse_yes_no(r.answer) is not None for r in pred)},
        "rewrite": {
            "status_same_class_skip": dict(rw), "status_same_class_position": dict(rw_pos),
            "same_class_two_region_items": len(same_all),
            "same_class_by_category": dict(sorted(collections.Counter(r.category for r in same_all).items())),
            "items_with_plural_class": plural, "items_with_plural_class_and_pronoun_it_in_answer": pronoun_plural,
            "single_region_items_with_two_classes": extra_class,
        },
    }


if __name__ == "__main__":   # python -m src.data.spatialrgpt  → 打印统计并写 manifests/spatialrgpt_stats.json
    s = stats()
    f = paths.REPO_ROOT / "manifests" / "spatialrgpt_stats.json"
    f.write_text(json.dumps(s, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(json.dumps(s, ensure_ascii=False, indent=1))
