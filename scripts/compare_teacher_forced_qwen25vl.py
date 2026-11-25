"""Qwen2.5-VL：teacher-forced 读出与自由生成在不同实现 / 精度之间的差异（CPU，只比较差异，不算正确率）。
用法：python scripts/compare_teacher_forced_qwen25vl.py Qwen2.5-VL-7B-Instruct
输出 <outputs>/diagnostics/{teacher_forced_compare,fp32_generate_compare}_<模型>.json，并打印 markdown。"""
import json
import sys

import numpy as np

from src import paths
from src.prompts import parse_four_numbers

name = sys.argv[1]
D = paths.get("outputs") / "diagnostics"
R = {p: json.load(open(D / f"teacher_forced_{name}_{p}.json")) for p in ("bf16", "fp32")}
S = [(p, i) for p in R for i in ("eager", "sdpa")]
K0 = [(x["dataset"], x["sample_id"], x["role"], x["response"]) for x in R["bf16"]["runs"]["eager"]]
for p, i in S:
    assert [(x["dataset"], x["sample_id"], x["role"], x["response"]) for x in R[p]["runs"][i]] == K0
meta = R["bf16"]["runs"]["eager"]
tok = np.array([[m["model_wh"][0] / m["grid_hw"][1], m["model_wh"][1] / m["grid_hw"][0]] for m in meta])       # 每条的 (token 宽, token 高)
truth = np.array([m["truth"] for m in meta])


def arr(s, head, m):
    return np.array([x["readout"][head][m] for x in R[s[0]]["runs"][s[1]]], dtype=float)


PAIRS = [("bf16 eager vs bf16 sdpa", ("bf16", "eager"), ("bf16", "sdpa")), ("fp32 eager vs fp32 sdpa", ("fp32", "eager"), ("fp32", "sdpa")),
         ("bf16 eager vs fp32 eager", ("bf16", "eager"), ("fp32", "eager")), ("bf16 sdpa vs fp32 sdpa", ("bf16", "sdpa"), ("fp32", "sdpa"))]
MET = {"e_tree": "期望坐标（精确枚举，模型输入像素）", "e_chain": "期望坐标（逐位条件，模型输入像素）", "lp_digits": "真值数字串对数概率（nat，每个坐标）"}
res = {"model": name, "n_prompts": len(K0), "n_coords": truth.size, "n_truth_ge_1000": int((truth >= 1000).sum()), "truth": R["bf16"]["truth"], "diffs": {}, "checks": {}}
st = lambda d: {"n": int(d.size), "n_nan": int(np.isnan(d).sum()), "mean": float(d.mean()), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}    # noqa: E731
print("| 读出量 | lm_head | 比较 | n | 均值 | 中位数 | 95% 分位 | 最大 | 框中心差，单位：视觉 token（cx、cy 合并 n=80：均值 / 中位数 / 95% 分位 / 最大） |\n|" + "---|" * 9)
for m, lab in MET.items():
    for head in ("native", "headfp32"):
        for pl, a, b in PAIRS:
            if head == "headfp32" and a[0] == b[0] == "fp32":
                continue
            A, B = arr(a, head, m), arr(b, head, m)
            d = np.abs(A - B)
            r = {**st(d), "n_gt_5": int((d > 5).sum()), "n_gt_50": int((d > 50).sum())}
            cell = "—"
            if m != "lp_digits":
                c = lambda E: np.stack([(E[:, 0] + E[:, 2]) / 2, (E[:, 1] + E[:, 3]) / 2], 1) / tok       # noqa: E731
                q = st(np.abs(c(A) - c(B)))
                r["center_tokens"] = q
                cell = f"{q['mean']:.3g} / {q['median']:.3g} / {q['p95']:.3g} / {q['max']:.3g}"
            res["diffs"][f"{m}/{head}/{pl}"] = r
            print(f"| {lab} | {'模型自带' if head == 'native' else 'float32'} | {pl} | {r['n']} | {r['mean']:.3g} | {r['median']:.3g} | {r['p95']:.3g} | {r['max']:.3g} | {cell} |")
print("\n互检：")
for p, i in S:
    for h in ("native", "headfp32"):
        lt, ld = arr((p, i), h, "lp_tree"), arr((p, i), h, "lp_digits")
        le = np.array([[np.nan if v is None else v for v in x["readout"][h]["lp_digits_end"]] for x in R[p]["runs"][i]], float)
        same = np.isnan(le)
        mass = arr((p, i), h, "mass_tree")
        d, d2 = np.abs(lt - ld)[same], np.abs(lt - le)[~same]
        res["checks"][f"{p}/{i}/{h}"] = {"n_same_quantity": int(same.sum()), "lp_tree_vs_lp_digits_median": float(np.median(d)), "lp_tree_vs_lp_digits_max": float(d.max()),
                                         "n_with_end_prob": int((~same).sum()), "lp_tree_vs_lp_digits_end_max": float(d2.max()) if d2.size else None, "mass_min": float(mass.min()), "mass_median": float(np.median(mass))}
        print(f"- {p}/{i}/{h}：", res["checks"][f"{p}/{i}/{h}"])
print("\n运行信息：")
for p in R:
    print(f"- {p}：", {k: v for k, v in R[p].items() if k not in ("runs",)})
json.dump(res, open(D / f"teacher_forced_compare_{name}.json", "w"), indent=1, ensure_ascii=False)

# ---- 自由生成 ----
g = json.load(open(D / f"fp32_generate_{name}.json"))
old = json.load(open(paths.get("outputs") / "smoke" / name / "default_vs_eager.json"))["runs"]
G = {"fp32 eager": g["runs"]["eager"], "fp32 sdpa": g["runs"]["sdpa"], "bf16 eager（已存）": old["observed"], "bf16 sdpa（已存）": old["reference"]}
key = [(x["task"], x["sample_id"]) for x in G["fp32 eager"]]
assert all([(x["task"], x["sample_id"]) for x in v] == key for v in G.values()) and len(key) == 60
GP = [("fp32 eager", "fp32 sdpa"), ("bf16 eager（已存）", "bf16 sdpa（已存）"), ("bf16 eager（已存）", "fp32 eager"), ("bf16 sdpa（已存）", "fp32 sdpa")]
gres = {"pairs": {}, "info": {k: v for k, v in g.items() if k != "runs"}}
print("\n| 比较 | 框逐 token 一致 | 框不可解析（前 / 后） | 坐标最大差（像素） | 不一致框的坐标差：中位数（每框取最大） | 关系逐 token 一致 | 关系首选项相同 | 选项 logit 差中位数 | 选项 logit 差最大 |\n|" + "---|" * 9)
for a, b in GP:
    bx = [(x, y) for x, y in zip(G[a], G[b]) if x["task"] == "bbox"]
    rl = [(x, y) for x, y in zip(G[a], G[b]) if x["task"] == "relation"]
    pa = [(parse_four_numbers(x["text"]), parse_four_numbers(y["text"])) for x, y in bx]
    cd = [max(abs(i - j) for i, j in zip(p, q)) for p, q in pa if p and q]
    nz = [v for (x, y), (p, q), v in zip([t for t, pp in zip(bx, pa) if pp[0] and pp[1]], [pp for pp in pa if pp[0] and pp[1]], cd) if x["tokens"] != y["tokens"]]
    ld = np.array([[abs(i - j) for i, j in zip(x["option_logits"], y["option_logits"])] for x, y in rl])
    r = {"bbox_identical": sum(x["tokens"] == y["tokens"] for x, y in bx), "bbox_n": len(bx), "unparsed": [sum(p is None for p, _ in pa), sum(q is None for _, q in pa)], "coord_max_diff": max(cd), "coord_max_diff_per_box_median_among_nonidentical": float(np.median(nz)) if nz else 0.0,
         "n_boxes_max_diff_gt_10": sum(v > 10 for v in cd), "relation_identical": sum(x["tokens"] == y["tokens"] for x, y in rl), "relation_n": len(rl), "relation_top_option_same": sum(int(np.argmax(x["option_logits"]) == np.argmax(y["option_logits"])) for x, y in rl),
         "option_logit_diff_median": float(np.median(ld)), "option_logit_diff_max": float(ld.max()),
         "worst_boxes": sorted(({"sample_id": x["sample_id"], "a": p, "b": q, "max_diff": max(abs(i - j) for i, j in zip(p, q))} for (x, y), (p, q) in zip(bx, pa) if p and q), key=lambda t: -t["max_diff"])[:3]}
    gres["pairs"][f"{a} vs {b}"] = r
    print(f"| {a} 对 {b} | {r['bbox_identical']}/{r['bbox_n']} | {r['unparsed'][0]} / {r['unparsed'][1]} | {r['coord_max_diff']:g} | {r['coord_max_diff_per_box_median_among_nonidentical']:g} | {r['relation_identical']}/20 | {r['relation_top_option_same']}/20 | {r['option_logit_diff_median']:.3g} | {r['option_logit_diff_max']:.3g} |")
json.dump(gres, open(D / f"fp32_generate_compare_{name}.json", "w"), indent=1, ensure_ascii=False)
print("\n", {k: v["worst_boxes"] for k, v in gres["pairs"].items()})
print("\n生成运行信息：", gres["info"])
