"""CV-Bench 3D 的 Depth 子集（哪个物体离相机更近）：物体框可得性与垂直位置捷径（CPU，只用已下载的标注元数据，不下图像、不跑模型）。
规则：2D 框下边缘更低（y2 更大）的物体判为更近。输出 manifests/cvbench_depth_cues.json；
若与规则相反的样本 >= 150 条，另写平衡子集 splits/cvbench_depth_balanced.json（规则在子集上的准确率在 45%–55%）。"""
import json
import random
from collections import Counter

import numpy as np

from src import paths

SEED = 20261125
F = paths.get("data") / "depth_candidates" / "cv_bench" / "test_3d.jsonl"
R = [json.loads(l) for l in open(F)]
D = [r for r in R if r["task"] == "Depth"]
out = {"file": str(F.relative_to(paths.get("data"))), "n_3d": len(R), "n_by_task": dict(Counter(r["task"] for r in R)), "n_depth": len(D)}
rows, problems = [], Counter()
for i, r in enumerate(D):
    boxes = json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"]
    ch = eval(r["choices"], {"__builtins__": {}}) if isinstance(r["choices"], str) else r["choices"]
    q = r["question"]
    # 题干：“the X (highlighted by a red box) or the Y (highlighted by a blue box)”
    ok = q.index(f"the {ch[0]} (highlighted by a red box)") < q.index(f"the {ch[1]} (highlighted by a blue box)") if f"the {ch[0]} (highlighted by a red box)" in q and f"the {ch[1]} (highlighted by a blue box)" in q else None
    if not ok:
        problems["question_order_vs_choices"] += 1
    if ch[0] == ch[1]:
        problems["same_name"] += 1
    ans = {"(A)": 0, "(B)": 1}[r["answer"]]
    (a, b) = boxes
    rows.append({"id": r["filename"], "dataset": r["source_dataset"], "src": r["source_filename"], "ans": ans, "boxes": boxes, "names": ch, "ok": bool(ok),
                 "d_bottom": a[3] - b[3], "d_center": (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2, "log_area": float(np.log((a[2] - a[0]) * (a[3] - a[1]) / ((b[2] - b[0]) * (b[3] - b[1]))))})
out["problems"] = dict(problems)


def rule(x, key):        # 预测更近的是 0（值更大）还是 1；平局按错
    return -1 if x[key] == 0 else (0 if x[key] > 0 else 1)


for tag, rs in [("all", rows)] + [(d, [x for x in rows if x["dataset"] == d]) for d in sorted({x["dataset"] for x in rows})]:
    out[tag] = {"n": len(rs), "n_source_images": len({x["src"] for x in rs}), "answer_A_share": round(sum(x["ans"] == 0 for x in rs) / len(rs), 3),
                **{f"rule_{k}_acc": round(sum(rule(x, k) == x["ans"] for x in rs) / len(rs), 3) for k in ("d_bottom", "d_center", "log_area")},
                "n_contrary_to_bottom_rule": sum(rule(x, "d_bottom") != x["ans"] for x in rs)}
    print(tag, out[tag])
con = [x for x in rows if x["ok"] and rule(x, "d_bottom") != x["ans"]]
pro = [x for x in rows if x["ok"] and rule(x, "d_bottom") == x["ans"]]
out["balanced"] = {"n_contrary": len(con), "n_consistent": len(pro)}
if len(con) >= 150:
    # 在每个来源数据集内取相同数量的一致 / 相反样本（取该数据集较少一方的数量），固定种子
    pick = []
    for d in sorted({x["dataset"] for x in rows}):
        c, p = [x for x in con if x["dataset"] == d], [x for x in pro if x["dataset"] == d]
        n = min(len(c), len(p))
        rng = random.Random(f"{SEED}:cvbench_balanced:{d}")
        pick += rng.sample(sorted(c, key=lambda x: x["id"]), n) + rng.sample(sorted(p, key=lambda x: x["id"]), n)
    pick.sort(key=lambda x: x["id"])
    acc = lambda k: round(sum(rule(x, k) == x["ans"] for x in pick) / len(pick), 3)      # noqa: E731
    out["balanced"].update(built=True, n=len(pick), by_dataset=dict(Counter(x["dataset"] for x in pick)), n_source_images=len({x["src"] for x in pick}),
                           rule_d_bottom_acc=acc("d_bottom"), rule_d_center_acc=acc("d_center"), rule_log_area_acc=acc("log_area"), answer_A_share=round(sum(x["ans"] == 0 for x in pick) / len(pick), 3))
    assert 0.45 <= out["balanced"]["rule_d_bottom_acc"] <= 0.55
    json.dump({"dataset": "cvbench_depth_balanced", "seed": SEED, "status": "active", "made_by": "scripts/analysis_cvbench_depth_cues.py",
               "note": "只是题目 id 列表；图像未下载；图像上已画红/蓝框", "ids": [x["id"] for x in pick]}, open(paths.REPO_ROOT / "splits" / "cvbench_depth_balanced.json", "w"), indent=0)
else:
    out["balanced"]["built"] = False
print(out["balanced"], out["problems"])
json.dump(out, open(paths.REPO_ROOT / "manifests" / "cvbench_depth_cues.json", "w"), indent=1, ensure_ascii=False)
