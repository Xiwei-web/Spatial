"""框读出精度的判定（planner 2026-11-26 给定标准；口径在看到框中心数字之前写死在这里，不事后修改）。CPU，只读已有 JSON。
用法：python scripts/tf_box_center_criterion.py <模型> <token 宽> <token 高>（坐标单位）   InternVL3-8B：62.5 62.5（1000/16）；Qwen2.5-VL：28 28（读自 JSON 的 model_wh / grid_hw，命令行值只用来核对）
比较：eager 下 bf16 − fp32；框中心 cx=(x1+x2)/2、cy=(y1+y2)/2，用 e_tree、模型自带 lm_head；除以 token 宽 / 高换成“视觉 token”。
标准：(1) |差| 的 95% 分位（numpy 线性插值） < 0.25 个视觉 token；(2) 有符号差均值的 95% bootstrap 区间（按图像重采样，10,000 次，百分位法，seed 20261126）包含 0。
主口径：cx、cy 两个分量各自判断，两个分量都同时满足 (1)(2) 才通过。另列两种读法仅供参考：两个分量合并（n=80，均值也合并）、欧氏距离（只有 (1)）；如果结论与主口径不同，如实写出，由 planner 裁定。"""
import json
import sys

import numpy as np

from src import paths

name, tw, th = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
D = paths.get("outputs") / "diagnostics"
R = {p: json.load(open(D / f"teacher_forced_{name}_{p}.json")) for p in ("bf16", "fp32")}
for p, d in {"bf16": "torch.bfloat16", "fp32": "torch.float32"}.items():
    assert R[p]["param_dtypes"] == [d] and R[p]["switch_effective"] if "switch_effective" in R[p] else R[p]["param_dtypes"] == [d]
X, Y = R["bf16"]["runs"]["eager"], R["fp32"]["runs"]["eager"]
assert [(a["sample_id"], a["role"], a["response"]) for a in X] == [(a["sample_id"], a["role"], a["response"]) for a in Y] and len(X) == 40
for a in X + Y:
    if "model_wh" in a:
        assert (a["model_wh"][0] / a["grid_hw"][1], a["model_wh"][1] / a["grid_hw"][0]) == (tw, th)
truth_kind = R["bf16"]["truth"]


def center(rows):
    e = np.array([a["readout"]["native"]["e_tree"] for a in rows], float)
    return np.stack([(e[:, 0] + e[:, 2]) / 2 / tw, (e[:, 1] + e[:, 3]) / 2 / th], 1)


d = center(X) - center(Y)                         # (40, 2)，单位：视觉 token
img = [(a["dataset"], a["sample_id"]) for a in X]
uniq = sorted(set(img))
idx = [np.array([i for i, k in enumerate(img) if k == u]) for u in uniq]
rng = np.random.default_rng(20261126)
draw = rng.integers(0, len(uniq), (10000, len(uniq)))


def boot(v):                                      # v: (40,) 或 (40, 2)（合并分量）
    s = np.array([v[i].sum() for i in idx]); n = np.array([v[i].size for i in idx])
    m = s[draw].sum(1) / n[draw].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = {"model": name, "truth": truth_kind, "token_size": [tw, th], "n_boxes": 40, "n_images": len(uniq), "components": {}}
for lab, v in (("cx", d[:, 0]), ("cy", d[:, 1]), ("合并", d)):
    a, ci = np.abs(v), boot(v)
    out["components"][lab] = {"n": int(v.size), "abs_mean": float(a.mean()), "abs_median": float(np.median(a)), "abs_p95": float(np.percentile(a, 95)), "abs_max": float(a.max()),
                              "signed_mean": float(v.mean()), "signed_mean_ci95": ci, "c1_p95_lt_0.25": bool(np.percentile(a, 95) < 0.25), "c2_ci_contains_0": bool(ci[0] <= 0 <= ci[1])}
e = np.hypot(d[:, 0], d[:, 1])
out["euclid"] = {"n": 40, "mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "max": float(e.max()), "p95_lt_0.25": bool(np.percentile(e, 95) < 0.25)}
C = out["components"]
out["pass_primary"] = all(C[k][c] for k in ("cx", "cy") for c in ("c1_p95_lt_0.25", "c2_ci_contains_0"))
out["pass_pooled"] = all(C["合并"][c] for c in ("c1_p95_lt_0.25", "c2_ci_contains_0"))
out["worst5"] = [{"sample_id": X[i]["sample_id"], "role": X[i]["role"], "dx": float(d[i, 0]), "dy": float(d[i, 1])} for i in np.argsort(-e)[:5]]
out["decision_by_primary"] = "bf16 + teacher-forced" if out["pass_primary"] else "fp32"
json.dump(out, open(D / f"tf_box_center_criterion_{name}.json", "w"), indent=1, ensure_ascii=False)
print(f"模型 {name}；真值 {truth_kind}；token {tw} x {th}；40 框 / {len(uniq)} 图")
print("| 分量 | n | \\|差\\| 均值 | 中位数 | 95% 分位 | 最大 | 有符号均值 | 95% bootstrap 区间 | (1) | (2) |\n|" + "---|" * 10)
for k, c in C.items():
    print(f"| {k} | {c['n']} | {c['abs_mean']:.4f} | {c['abs_median']:.4f} | {c['abs_p95']:.4f} | {c['abs_max']:.4f} | {c['signed_mean']:+.4f} | [{c['signed_mean_ci95'][0]:+.4f}, {c['signed_mean_ci95'][1]:+.4f}] | {'满足' if c['c1_p95_lt_0.25'] else '不满足'} | {'包含 0' if c['c2_ci_contains_0'] else '不含 0'} |")
E = out["euclid"]
print(f"| 欧氏距离 | 40 | {E['mean']:.4f} | {E['median']:.4f} | {E['p95']:.4f} | {E['max']:.4f} | — | — | {'满足' if E['p95_lt_0.25'] else '不满足'} | — |")
print("主口径：", "通过" if out["pass_primary"] else "不通过", "；合并口径：", "通过" if out["pass_pooled"] else "不通过", "；worst5", out["worst5"])
