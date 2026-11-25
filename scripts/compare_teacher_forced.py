"""汇总 teacher-forced 读出在不同实现 / 精度之间的差异（CPU；只比较实现之间的差异，不与正确与否挂钩）。
用法：python scripts/compare_teacher_forced.py InternVL3-8B
输出：<outputs>/diagnostics/teacher_forced_compare_<模型>.json，并打印 markdown 表。"""
import json
import sys

import numpy as np

from src import paths

name = sys.argv[1]
D = paths.get("outputs") / "diagnostics"
R = {p: json.load(open(D / f"teacher_forced_{name}_{p}.json")) for p in ("bf16", "fp32")}
for p, r in R.items():
    assert r["switch_effective"], p
keys = [(x["sample_id"], x["role"]) for x in R["bf16"]["runs"]["eager"]]
for p in R:
    for im in ("eager", "sdpa"):
        assert [(x["sample_id"], x["role"]) for x in R[p]["runs"][im]] == keys
        assert [x["response"] for x in R[p]["runs"][im]] == [x["response"] for x in R["bf16"]["runs"]["eager"]]


def arr(p, im, head, m):           # (40, 4)
    return np.array([x["readout"][head][m] for x in R[p]["runs"][im]], dtype=float)


PAIRS = [("bf16 eager vs bf16 sdpa", ("bf16", "eager"), ("bf16", "sdpa")), ("fp32 eager vs fp32 sdpa", ("fp32", "eager"), ("fp32", "sdpa")),
         ("bf16 eager vs fp32 eager", ("bf16", "eager"), ("fp32", "eager")), ("bf16 sdpa vs fp32 sdpa", ("bf16", "sdpa"), ("fp32", "sdpa"))]
MET = {"e_tree": "期望坐标（精确枚举，0–1000 单位）", "e_chain": "期望坐标（逐位条件，0–1000 单位）", "lp_digits": "真值数字串对数概率（nat，每个坐标）"}
res = {"model": name, "n_prompts": len(keys), "n_coords": len(keys) * 4, "diffs": {}, "checks": {}, "per_sample": {}}
for p in R:                        # 互检：枚举路径与一次前向路径对同一个量（真值为三位数且非 100）
    for im in ("eager", "sdpa"):
        for h in ("native", "headfp32"):
            tr = np.array([x["truth1000"] for x in R[p]["runs"][im]])
            ok = (tr >= 101) & (tr <= 999)
            d = np.abs(arr(p, im, h, "lp_tree") - arr(p, im, h, "lp_digits"))[ok]
            mass = arr(p, im, h, "mass_tree")
            res["checks"][f"{p}/{im}/{h}"] = {"n": int(ok.sum()), "lp_tree_vs_lp_digits_median": float(np.median(d)), "lp_tree_vs_lp_digits_max": float(d.max()),
                                              "mass_min": float(mass.min()), "mass_median": float(np.median(mass))}
lines = ["| 读出量 | lm_head | 比较 | 中位数 | 95% 分位 | 最大 |", "|---|---|---|---|---|---|"]
for m, lab in MET.items():
    for head in ("native", "headfp32"):
        for pl, a, b in PAIRS:
            if head == "headfp32" and a[0] == b[0] == "fp32":
                continue           # float32 模型下两种 lm_head 相同
            d = np.abs(arr(*a, head, m) - arr(*b, head, m))
            res["diffs"][f"{m}/{head}/{pl}"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max()), "n_gt_5": int((d > 5).sum()), "n_gt_50": int((d > 50).sum())}
            lines.append(f"| {lab} | {'模型自带' if head == 'native' else 'float32'} | {pl} | {np.median(d):.3g} | {np.percentile(d, 95):.3g} | {d.max():.3g} |")
for k, key in enumerate(keys):
    res["per_sample"]["/".join(key)] = {"truth1000": R["bf16"]["runs"]["eager"][k]["truth1000"],
        **{f"{p}/{im}": {m: [round(v, 4) for v in R[p]["runs"][im][k]["readout"]["native"][m]] for m in MET} for p in R for im in ("eager", "sdpa")}}
json.dump(res, open(D / f"teacher_forced_compare_{name}.json", "w"), indent=1, ensure_ascii=False)
print("\n".join(lines))
print("\n互检:"); [print(k, v) for k, v in res["checks"].items()]
print("\ncounts:"); [print(k, v) for k, v in res["diffs"].items() if k.startswith("e_")]
