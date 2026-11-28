"""E1 dev 试跑的“讨厌参数”（planner 第 11 轮第 5 条；CPU，只读 outputs/synth/<模型>/e1_pilot200.jsonl，全部是 dev）。
只算：
  1. L 关于 f(真实 Δ) 的斜率与残差 sd（用来定功效模拟里的 κ、ν）；另用两个选项顺序之差估 ν（不依赖框）；
  2. 两物体场景的框中心有向误差分布（按 轴 × 尺寸档），作为功效模拟的误差池；
  3. 每个 token 档的直接回答正确率与框推正确率（各自的边际量）。
不算 β、OR、比值、一致率或任何 2×2（判定量）：src/e1_analysis.build_items 顺带产生的一致率字段读入后立即丢弃。
L 的定义：正确选项 − 错误选项的 fp32 logprob（各取 "A"/" A" 等变体的最大值）；取“产生答案的那一步”：letter_logit 方式取首步，文字方式取生成序列里最后一个答案 token 所在的步（fp32_final_answer）。
f(Δ) = Δ / s，Δ = 真框中心沿轴的有向差（token，真关系方向为正），s = 该（轴, 尺寸档）的 sd(e_t − e_r)（sd 版）或 1.4826·MAD(e_t − e_r)（MAD 版）。
功效模拟的智能体是 L = κ·S/s + ν（S 在 H1 下 = Δ̂）。无框信息时，S − Δ 的方差由框误差池给出，于是
  κ̂ = L 对 f(Δ) 的过原点最小二乘斜率；ν̂² = 残差方差 − κ̂²·Var((e_t − e_r)/s)（sd 版 Var = 1；MAD 版用池的实际方差除以 s²），< 0 时记 0。
实际用于功效模拟的是 L_local（饱和的局部线性模型，见 1b 段）：过原点直线拟合（L_fit）在试跑数据上不成立——L 几乎全部饱和、与距离近乎无关，
过原点拟合给出很小的 κ 和很大的 ν，按它模拟时最近一档的直接回答正确率接近 0.5，与实测（≈ 1）矛盾。L_fit 仍照实保存、写进报告。
用法：python scripts/e1_pilot_nuisance.py → manifests/e1_pilot_nuisance.json（含误差池原始数组，供 scripts/e1_power_sim_v3.py 读取）"""
import json
from collections import defaultdict

import numpy as np

from src import e0, e1, paths
from src.e1_analysis import build_items
from src.e1_alt import rcs_basis, rcs_knots
from src.e1_sim import robust_sd
from src.prompts import LETTERS, NO_VARIANTS, YES_VARIANTS, letter_variants

MODELS = e1.synth_cfg()["single_object_check"]["models"]
SIZES = (36, 72, 108, 161)


def best(lp, vs):
    v = [lp[x] for x in vs if x in lp]
    return max(v) if v else np.nan


def L_record(r, mode):
    fx = r.get("fp32_answer_step_flexible")
    if mode != "letter_logit" and fx and fx.get("fp32_logprob_by_option") and None not in fx["fp32_logprob_by_option"].values():
        lp = fx["fp32_logprob_by_option"]          # 通用定位（第 11 轮补跑的 RoboBrain）：每个选项已取候选 token 的最大值
        if r["task"] == "mcq":
            k = r["options"].index(r["relation"])
            return lp[LETTERS[k]] - lp[LETTERS[1 - k]]
        return (lp["yes"] - lp["no"]) if r["asked_relation"] == r["relation"] else (lp["no"] - lp["yes"])
    src = r.get("fp32_logprob") if mode == "letter_logit" else (r.get("fp32_final_answer") or {}).get("fp32_logprob")
    if not src:
        return np.nan
    if r["task"] == "mcq":
        k = r["options"].index(r["relation"])
        return best(src, letter_variants(LETTERS[k])) - best(src, letter_variants(LETTERS[1 - k]))
    y, n = best(src, YES_VARIANTS), best(src, NO_VARIANTS)
    return (y - n) if r["asked_relation"] == r["relation"] else (n - y)


def model_stats(name):
    f = paths.get("outputs") / "synth" / name / "e1_pilot200.jsonl"
    recs = list(e0.load_records(f).values())
    if len(recs) < 1200:
        return None
    f2 = f.with_name("e1_pilot200_ans2.jsonl")          # 直接回答的补跑（只有 RoboBrain）：用补跑的 mcq / binary 记录替换原记录
    if f2.is_file():
        r2 = [r for r in e0.load_records(f2).values()]
        if len(r2) < 800:
            return None
        recs = [r for r in recs if r["task"] == "box"] + r2
    assert all(r["split"] == "dev" for r in recs) and not any(r.get("error") for r in recs)
    modes = recs[0]["answer_modes"]
    cfg = {"variants": ["base"], "subset": "main", "truncate_to_model_bins": False}
    it, skipped = build_items(recs, cfg)
    for k in ("mcq_agree", "pair_agree"):
        it.pop(k)
    # 条目与场景对齐：build_items 按 (数据集, 样本) 排序、跳过缺框的条目
    by = defaultdict(list)
    for r in recs:
        by[(r["dataset"], r["sample_id"])].append(r)
    keys = [k for k in sorted(by) if {x["role"] for x in by[k] if x["task"] == "box"} == {"target", "reference"}]
    assert len(keys) == len(it["d_box"])
    axis = np.array(["x" if by[k][0]["axis"] == "horizontal" else "y" for k in keys])
    size = np.array([by[k][0]["sqrt_area_px"] for k in keys])
    Lm = np.array([[L_record(r, modes["mcq"]) for r in sorted([x for x in by[k] if x["task"] == "mcq"], key=lambda x: x["rotation"])] for k in keys])
    Lb = np.array([[L_record(r, modes["binary"]) for r in sorted([x for x in by[k] if x["task"] == "binary"], key=lambda x: x["asked_relation"] != x["relation"])] for k in keys])
    d, et, er = it["d_box"], it["e_t"], it["e_r"]
    parsed = it["box_parsed"].astype(bool)
    # 2. 误差池
    pools, pool_stats = {}, {}
    for a in ("x", "y"):
        for s in SIZES:
            m = (axis == a) & (size == s) & parsed
            e = np.concatenate([et[m], er[m]])
            dd = (et - er)[m]
            pools[f"{a}|{s}"] = [float(v) for v in e]
            pool_stats[f"{a}|{s}"] = {"n_items": int(m.sum()), "sd_e": float(e.std()) if len(e) else None, "mad_e": float(np.median(np.abs(e - np.median(e)))) if len(e) else None,
                                      "sd_diff": float(dd.std()) if len(dd) > 1 else None, "robust_sd_diff": robust_sd(dd) if len(dd) > 1 else None,
                                      "max_abs_e": float(np.abs(e).max()) if len(e) else None}
    # 1. L 的尺度与噪声
    Lfit = {}
    for sc in ("sd", "mad"):
        s_item = np.full(len(d), np.nan)
        var_ratio = np.full(len(d), np.nan)
        for a in ("x", "y"):
            dd_all = (et - er)[(axis == a) & parsed]
            for s in SIZES:
                m = (axis == a) & (size == s) & parsed
                dd = (et - er)[m]
                if len(dd) < 15:                                  # 小池并为该轴全部尺寸（与功效模拟的合并规则一致）
                    dd = dd_all
                sv = float(dd.std()) if sc == "sd" else robust_sd(dd)
                sv = max(sv, 1e-6)
                s_item[(axis == a) & (size == s)] = sv
                var_ratio[(axis == a) & (size == s)] = float(dd.var()) / sv ** 2
        out = {}
        for nm, L in (("mcq_mean_of_orders", np.nanmean(Lm, 1)), ("binary_mean_of_two", np.nanmean(Lb, 1))):
            f_ = d / s_item
            ok = np.isfinite(L) & np.isfinite(f_)
            if ok.sum() < 10:
                out[nm] = None
                continue
            k_ = float((f_[ok] * L[ok]).sum() / (f_[ok] ** 2).sum())
            res = L[ok] - k_ * f_[ok]
            A_ = np.column_stack([np.ones(ok.sum()), f_[ok]])
            ci, *_ = np.linalg.lstsq(A_, L[ok], rcond=None)
            nu2 = float(res.var() - k_ ** 2 * np.mean(var_ratio[ok]))
            out[nm] = {"n": int(ok.sum()), "kappa_through_origin": k_, "resid_sd": float(res.std()), "nu": float(np.sqrt(max(nu2, 0.0))), "nu2_raw": nu2,
                       "with_intercept": {"a": float(ci[0]), "kappa": float(ci[1]), "resid_sd": float((L[ok] - A_ @ ci).std())},
                       "f_quantiles": [float(np.percentile(f_[ok], q)) for q in (0, 10, 50, 90, 100)], "L_quantiles": [float(np.percentile(L[ok], q)) for q in (0, 10, 50, 90, 100)]}
        Lfit[sc] = out
    # 1b. 饱和的局部线性模型（功效模拟实际用的参数）：E[L | Δ] = m(Δ) 用 A 方案同样的限制性样条（log2 Δ，4 结）拟合；
    #     κ_loc(Δ) = dm/dΔ（每 token）；ν = 两个顺序之差的 sd / √2（扣掉平均差 = 字母位置偏好），不依赖框
    local = {}
    for nm, LL in (("mcq", Lm), ("binary", Lb)):
        Lbar = np.nanmean(LL, 1)
        ok = np.isfinite(Lbar) & (d > 0)
        if ok.sum() < 20:
            local[nm] = {"error": f"只有 {int(ok.sum())} 个条目有 L"}
            continue
        lx = np.log2(d[ok])
        kn = rcs_knots(lx, 4)
        coef, *_ = np.linalg.lstsq(rcs_basis(lx, kn), Lbar[ok], rcond=None)
        fit = rcs_basis(lx, kn) @ coef
        h = 1e-3
        slope_tok = (rcs_basis(lx + h, kn) @ coef - rcs_basis(lx - h, kn) @ coef) / (2 * h) / (d[ok] * np.log(2))
        df = LL[:, 0] - LL[:, 1]
        okd = np.isfinite(df)
        nu_o = float(np.std(df[okd] - df[okd].mean()) / np.sqrt(2))
        res = Lbar[ok] - fit
        local[nm] = {"knots": [float(v) for v in kn], "coef": [float(v) for v in coef], "log2_range": [float(lx.min()), float(lx.max())], "nu": nu_o,
                     "order_mean_difference": float(df[okd].mean()), "resid_sd_item_mean": float(res.std()),
                     "m_quantiles": [float(np.percentile(fit, q)) for q in (0, 10, 50, 90, 100)],
                     "kappa_loc_per_token_quantiles": [float(np.percentile(slope_tok, q)) for q in (0, 10, 50, 90, 100)],
                     "by_token_bin": {str(b): {"m_mean": float(fit[it["bin"][ok] == b].mean()), "kappa_loc_mean": float(slope_tok[it["bin"][ok] == b].mean())}
                                      for b in sorted(set(it["bin"][ok].tolist()))}}
    # 顺序之差估 ν（扣掉两个顺序的平均差，即字母位置偏好）
    dif = Lm[:, 0] - Lm[:, 1]
    okd = np.isfinite(dif)
    nu_order = float(np.std(dif[okd] - dif[okd].mean()) / np.sqrt(2)) if okd.sum() > 5 else None
    # 3. 边际正确率（按 token 档）
    tb = it["bin"]
    marg = {}
    for b in sorted(set(tb.tolist())):
        m = tb == b
        marg[str(b)] = {"n": int(m.sum()), "box_acc": float(it["box_ok"][m].mean()), "mcq_acc": float(np.nanmean(it["mcq_ok"][m])), "pair_acc": float(np.nanmean(it["pair_ok"][m])),
                        "frac_L_mcq_abs_gt10": float(np.nanmean(np.abs(Lm[m]) > 10))}
    return {"n_items": int(len(d)), "skipped": skipped, "answer_modes": modes, "box_parse_fail": float(1 - parsed.mean()), "box_error_pools": pools, "box_error_stats": pool_stats,
            "L_fit": Lfit, "L_local": local, "nu_from_order_difference": nu_order, "order_mean_difference": float(dif[okd].mean()) if okd.any() else None, "marginal_by_token_bin": marg,
            "L_mcq_quantiles": [float(np.nanpercentile(Lm, q)) for q in (0, 1, 10, 50, 90, 99, 100)], "frac_L_mcq_negative": float(np.nanmean(Lm < 0)),
            "frac_L_mcq_abs_gt10": float(np.nanmean(np.abs(Lm) > 10)), "L_binary_quantiles": [float(np.nanpercentile(Lb, q)) for q in (0, 1, 10, 50, 90, 99, 100)]}


if __name__ == "__main__":
    res = {"note": "只含讨厌参数（L 的尺度与噪声、框误差分布、边际正确率）；不含 β、OR、比值、一致率或 2×2", "models": {}}
    for m in MODELS:
        f = paths.get("outputs") / "synth" / m / "e1_pilot200.jsonl"
        if not f.is_file():
            print(m, "没有试跑记录"); continue
        st = model_stats(m)
        if st is None:
            print(m, "试跑还没跑完"); continue
        res["models"][m] = st
        print(m, json.dumps({"local_mcq": {k: st["L_local"]["mcq"][k] for k in ("nu", "m_quantiles", "kappa_loc_per_token_quantiles")}, "nu_order": st["nu_from_order_difference"], "L_mcq_q": [round(v, 2) for v in st["L_mcq_quantiles"]],
                             "box_parse_fail": st["box_parse_fail"]}, ensure_ascii=False), flush=True)
    import os
    out = paths.REPO_ROOT / "manifests" / "e1_pilot_nuisance.json"
    tmp = out.with_name(f".e1_pilot_nuisance.{os.getpid()}.tmp")
    json.dump(res, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, out)                    # 原子替换：几个功效模拟作业可能同时读这个文件
