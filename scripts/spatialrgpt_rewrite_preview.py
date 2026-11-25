"""SpatialRGPT-Bench 改写规则预览：固定种子抽 20 条，打印改写前后对照（Markdown，粘到 docs/parts/F7_spatialrgpt.md）。

**不改写全量数据，不落盘。** 规则在 src/data/spatialrgpt.py（region_names / substitute / rewrite_record）。
抽样（种子固定，按类别分层）：
  - front_predicate、behind_predicate 各 4 条：其中各 1 条取两个区域类名相同的（如果有），其余取类名不同的；
  - 单区域定量题 height_data / width_data 各 1 条（前者取 class 给了 2 个名字的，后者取只有 1 个名字的）；
  - 双区域定量题（distance / horizontal_distance / vertical_distance / direction）里随机 3 个类别各 1 条；
  - *_choice 里随机 3 个类别各 1 条；其余 *_predicate 里随机 4 个类别各 1 条。
用法：source scripts/env.sh tf5 && python scripts/spatialrgpt_rewrite_preview.py
"""
from __future__ import annotations

import random

from src.data import spatialrgpt as sr

SEED = "spatialrgpt_rewrite_preview:20261125"


def pick() -> list[sr.Record]:
    rng = random.Random(SEED)
    by_cat: dict[str, list[sr.Record]] = {}
    for r in sorted(sr.load_records(), key=lambda r: r.id):
        by_cat.setdefault(r.category, []).append(r)
    same = lambda r: r.n_regions == 2 and r.classes[0].strip().lower() == r.classes[1].strip().lower()  # noqa: E731
    chosen: list[sr.Record] = []
    for cat in ("front_predicate", "behind_predicate"):
        s, d = [r for r in by_cat[cat] if same(r)], [r for r in by_cat[cat] if not same(r)]
        n_same = min(1, len(s))
        chosen += rng.sample(d, 4 - n_same) + rng.sample(s, n_same)
    cats = sorted(by_cat)
    # 单区域题：height_data 取一条 class 给了 2 个名字的，width_data 取一条只有 1 个名字的
    chosen.append(rng.choice([r for r in by_cat["height_data"] if len(r.classes) == 2]))
    chosen.append(rng.choice([r for r in by_cat["width_data"] if len(r.classes) == 1]))
    fam = {
        "two_region_quant": (["direction", "distance_data", "horizontal_distance_data", "vertical_distance_data"], 3),
        "choice": ([c for c in cats if c.endswith("_choice")], 3),
        "predicate": ([c for c in cats if c.endswith("_predicate") and c not in sr.DEPTH_CATEGORIES], 4),
    }
    for cs, n in fam.values():
        for cat in rng.sample(cs, n):
            chosen.append(rng.choice(by_cat[cat]))
    assert len(chosen) == 20 and len({r.id for r in chosen}) == 20
    return chosen


def cell(s: str | None) -> str:
    return "—" if s is None else s.replace("|", "\\|").replace("<", "&lt;")


def main() -> None:
    print("| # | id / 类别 / class | 改写前（text_q ‖ &lt;mask&gt; 写法 ‖ 答案） | 改写后（问题 ‖ 答案） | 状态与备注 |")
    print("|---|---|---|---|---|")
    for i, r in enumerate(pick(), 1):
        a = sr.rewrite_record(r, same_class="skip")
        after = f"Q: {cell(a['question'])}<br>A: {cell(a['answer'])}"
        status = a["status"]
        if a["status"] == "ambiguous_same_class":
            b = sr.rewrite_record(r, same_class="position")
            after = "默认（skip）：不改写"
            if b["question"] is not None:
                after += f"<br>可选（position）：Q: {cell(b['question'])}<br>A: {cell(b['answer'])}"
            else:
                after += "<br>可选（position）：也不改写（类别会泄露左右，或两框水平距离不足）"
            status += f" → position: {b['status']}"
        notes = "；".join(a["notes"])
        before = f"Q: {cell(r.question_region)}<br>M: {cell(r.question_mask)}<br>A: {cell(r.answer)}"
        print(f"| {i} | `{r.id}`<br>{r.category}<br>{r.classes} | {before} | {after} | {status}{'；' + notes if notes else ''} |")


if __name__ == "__main__":
    main()
