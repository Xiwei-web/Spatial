"""teacher-forced 读出：|e_tree(bf16 eager) − e_tree(fp32 eager)| 最大的 10 个坐标（CPU，只读已有 JSON；lm_head 用模型自带）。
用法：python scripts/teacher_forced_top_diffs.py InternVL3-8B     输出 <outputs>/diagnostics/teacher_forced_top10_<模型>.json 并打印 markdown 表"""
import json
import sys

import numpy as np

from src import paths

name = sys.argv[1]
D = paths.get("outputs") / "diagnostics"
R = {p: json.load(open(D / f"teacher_forced_{name}_{p}.json"))["runs"] for p in ("bf16", "fp32")}
S = [("bf16", "eager"), ("bf16", "sdpa"), ("fp32", "eager"), ("fp32", "sdpa")]
E = {s: np.array([x["readout"]["native"]["e_tree"] for x in R[s[0]][s[1]]]) for s in S}
LP = {s: np.array([x["readout"]["native"]["lp_digits"] for x in R[s[0]][s[1]]]) for s in S}
meta = R["bf16"]["eager"]
BIG = {("pan_left_of_chair", "target"): "是（pan）", ("pillow_left_of_armchair", "target"): "是（pillow）"}      # 第 1 节自由生成差异 ≥ 50 的两条
d = np.abs(E[S[0]] - E[S[2]])
rows = []
for k, i in enumerate(np.argsort(-d.ravel())[:10]):
    r, c = divmod(int(i), 4)
    m = meta[r]
    rows.append({"rank": k + 1, "dataset": m["dataset"], "sample_id": m["sample_id"], "role": m["role"], "object": m["object"], "coord": ["x1", "y1", "x2", "y2"][c], "truth_det_box": m["truth1000"][c],
                 "abs_diff": float(d[r, c]), **{f"e_tree_{p}_{im}": float(E[(p, im)][r, c]) for p, im in S}, "lp_digits_fp32_eager": float(LP[S[2]][r, c]), "is_big_free_generation_diff_sample": BIG.get((m["sample_id"], m["role"]), "否")})
big = [{"sample_id": m["sample_id"], "role": m["role"], "truth_det_box": m["truth1000"], "abs_diff_bf16_vs_fp32_eager": [float(v) for v in d[r]], "rank_of_max_among_160": int((d > d[r].max()).sum()) + 1,
        **{f"e_tree_{p}_{im}": [float(v) for v in E[(p, im)][r]] for p, im in S}} for r, m in enumerate(meta) if (m["sample_id"], m["role"]) in BIG]
json.dump({"top10": rows, "big_free_generation_samples": big}, open(D / f"teacher_forced_top10_{name}.json", "w"), indent=1, ensure_ascii=False)
print("| # | 样本 / 角色 | 物体 | 坐标 | 真值（检测框） | bf16 eager | bf16 sdpa | fp32 eager | fp32 sdpa | 差（bf16 − fp32，eager，绝对值） | fp32 对真值的对数概率 | 是否第 1 节的 pan / pillow 样本 |\n|" + "---|" * 12)
for x in rows:
    print(f"| {x['rank']} | {x['sample_id']} / {x['role']} | {x['object']} | {x['coord']} | {x['truth_det_box']} | " + " | ".join(f"{x[f'e_tree_{p}_{im}']:.1f}" for p, im in S) + f" | {x['abs_diff']:.2f} | {x['lp_digits_fp32_eager']:.2f} | {x['is_big_free_generation_diff_sample']} |")
print()
for b in big:
    f = lambda v: "[" + ", ".join(f"{t:.1f}" for t in v) + "]"      # noqa: E731
    print(f"- {b['sample_id']} / {b['role']}：真值 {b['truth_det_box']}；e_tree bf16 eager {f(b['e_tree_bf16_eager'])}、bf16 sdpa {f(b['e_tree_bf16_sdpa'])}、fp32 eager {f(b['e_tree_fp32_eager'])}、fp32 sdpa {f(b['e_tree_fp32_sdpa'])}；bf16 对 fp32（eager）四个坐标的差 {f(b['abs_diff_bf16_vs_fp32_eager'])}，其中最大者在 160 个坐标里排第 {b['rank_of_max_among_160']}。")
