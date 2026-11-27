"""E1 判定的备选方案（planner 第 10 轮补充，供 Xiwei 选；当前判定规则不变，configs/e1.yaml 仍是 placeholder）。只吃条目表（src/e1_analysis.build_items 或 src/e1_sim）。
  A 连续量耦合：L ~ 自然样条(log2 真实距离) + β·(Δ̂ − Δ)，按场景 bootstrap β 的 CI；标准化效应 β·sd(Δ̂ − Δ) / sd(残差)。
     L = 直接回答的 logit 差（正确选项 − 错误选项，二选一的两个顺序取平均）；Δ̂ = 两个预测框中心沿关系轴的有向差，Δ = 真框中心的有向差（token，真关系方向为正）。
  B OR 限定范围：只在 dev 上框推正确率 <= box_acc_max 的档里算 MH OR；某档四格任一格的期望 < min_expected 就不用；一档都不剩记“不可估”。另报按档分层的条件最大似然 OR（似然比 CI）作对比。
  C 比值限定条件：θ_IO 的 bootstrap CI 完全落在该模型的采样距离范围内才做比值判定，否则“不适用”，改报每档直接回答与框推正确率之差（配对，McNemar 精确检验）。
  另：logistic 回归 直接回答对错 ~ 样条(log2 距离) + 框推对错（档内连续距离协变量；参数恢复里看 H3 的 OR 偏高是否消失）。"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.stats import binomtest, chi2, nchypergeom_fisher

from src.e1_analysis import fit_psychometric, io_curve, mh_or, table_2x2


# ---------------- 样条 ----------------
def rcs_knots(x, n_knots=4):
    q = {3: [0.10, 0.50, 0.90], 4: [0.05, 0.35, 0.65, 0.95], 5: [0.05, 0.275, 0.5, 0.725, 0.95]}[n_knots]
    return np.quantile(x, q)


def rcs_basis(x, k):
    """限制性（自然）三次样条基（Harrell），含截距。"""
    K = len(k)
    p3 = lambda u: np.maximum(u, 0) ** 3      # noqa: E731
    cols = [np.ones_like(x), x]
    for j in range(K - 2):
        cols.append((p3(x - k[j]) - p3(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2]) + p3(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / (k[K - 1] - k[0]) ** 2)
    return np.column_stack(cols)


# ---------------- A：连续量耦合 ----------------
def _coupling_rows(items):
    L = np.nanmean(items["L"], axis=1) if items["L"].ndim == 2 else items["L"]
    err = items["dhat"] - items["d_box"]
    ok = ~np.isnan(L) & ~np.isnan(err) & ~np.isnan(items["x"])
    return ok, L, err


def coupling_point(items, w, knots):
    ok, L, err = _coupling_rows(items)
    ok = ok & (w > 0)
    X = np.column_stack([rcs_basis(np.log2(items["x"][ok]), knots), err[ok]])
    sw = np.sqrt(w[ok])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], L[ok] * sw, rcond=None)
    res = L[ok] - X @ coef
    ww = w[ok] / w[ok].sum()
    sd = lambda v: math.sqrt(max(float((ww * (v - (ww * v).sum()) ** 2).sum()), 1e-12))      # noqa: E731
    beta = float(coef[-1])
    return {"beta": beta, "std_effect": beta * sd(err[ok]) / sd(res), "n": int(ok.sum())}


def coupling(items, n_boot, rng, cfg_alt):
    ok, _, _ = _coupling_rows(items)
    knots = rcs_knots(np.log2(items["x"][ok]), cfg_alt["spline_knots"])
    est = coupling_point(items, np.ones(len(ok)), knots)
    keys = items["key"].astype(int)
    nk = keys.max() + 1
    bb, ss = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, nk, nk), minlength=nk).astype(float)
        p = coupling_point(items, cnt[keys], knots)
        bb.append(p["beta"]); ss.append(p["std_effect"])
    ci_b = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
    ci_s = [float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))]
    e = cfg_alt["equivalence_std_effect"]
    return {**est, "beta_ci": ci_b, "std_effect_ci": ci_s,
            "verdict": "coupled" if ci_b[0] > 0 else ("null_compatible" if ci_b[0] <= 0 <= ci_b[1] else "negative"),
            "equivalence_null": bool(ci_s[0] > -e and ci_s[1] < e)}


# ---------------- B：OR 限定范围 ----------------
def choose_bins_dev(dev_items, box_acc_max):
    """在 dev 条目上按档算框推正确率，<= box_acc_max 的档可用（不在 test 上挑）。"""
    out = []
    for b in np.unique(dev_items["bin"]):
        m = dev_items["bin"] == b
        if b >= 0 and m.sum() and dev_items["box_ok"][m].mean() <= box_acc_max:
            out.append(int(b))
    return out


def _expected_ok(box_ok, dir_ok, min_exp):
    a, b, c, d = table_2x2(box_ok, dir_ok, np.ones(len(box_ok)))
    n = a + b + c + d
    if n == 0:
        return False
    r1, r0, c1, c0 = a + b, c + d, a + c, b + d
    return min(r1 * c1, r1 * c0, r0 * c1, r0 * c0) / n >= min_exp


def cond_mle_or(tables):
    """按层的条件最大似然 OR（Fisher 非中心超几何），似然比 95% CI。tables：[(a, b, c, d), …]（整数）。"""
    T = [(int(a), int(a + b), int(a + c), int(a + b + c + d)) for a, b, c, d in tables if a + b + c + d > 0]

    def ll(lp):
        return float(sum(nchypergeom_fisher.logpmf(a, M, n, N, math.exp(lp)) for a, n, N, M in T))
    r = optimize.minimize_scalar(lambda lp: -ll(lp), bounds=(-12, 12), method="bounded")
    lmax, top = -r.fun, r.x
    f = lambda lp: 2 * (lmax - ll(lp)) - chi2.ppf(0.95, 1)      # noqa: E731
    lo = optimize.brentq(f, -12, top) if f(-12) > 0 else -math.inf
    hi = optimize.brentq(f, top, 12) if f(12) > 0 else math.inf
    return {"or": math.exp(top), "ci": [math.exp(lo) if lo > -math.inf else 0.0, math.exp(hi) if hi < math.inf else math.inf], "at_bound": abs(top) > 11.9}


def or_restricted(items, dev_items, n_boot, rng, cfg_alt, th):
    usable = choose_bins_dev(dev_items, cfg_alt["box_acc_max"])
    D = items["mcq_ok"]
    kept = [b for b in usable if _expected_ok(items["box_ok"][items["bin"] == b], D[items["bin"] == b], cfg_alt["min_expected"])]
    out = {"bins_dev_usable": usable, "bins_used": kept}
    if not kept:
        out["verdict"] = "not_estimable"
        return out
    m = np.isin(items["bin"], kept)
    sub = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in items.items()}
    keys = sub["key"].astype(int)
    uk, inv = np.unique(keys, return_inverse=True)
    est = mh_or(sub["box_ok"], sub["mcq_ok"], np.ones(len(keys)), sub["bin"])
    bs = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uk), len(uk)), minlength=len(uk)).astype(float)
        bs.append(mh_or(sub["box_ok"], sub["mcq_ok"], cnt[inv], sub["bin"]))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    tabs = [table_2x2(sub["box_ok"][sub["bin"] == b], sub["mcq_ok"][sub["bin"] == b][:, :1], np.ones(int((sub["bin"] == b).sum()))) for b in kept]
    out.update(or_mh=est, or_mh_ci=ci, cond_mle_order0=cond_mle_or(tabs),
               verdict="dependent" if ci[0] > th["or_dependent_min"] else ("independent_compatible" if ci[0] <= 1 <= ci[1] else "inconclusive"))
    return out


# ---------------- C：比值限定条件 ----------------
def ratio_restricted(res_main, items, th):
    """res_main：src.e1_analysis.analyze 的结果（含 θ_IO 的 bootstrap CI）。"""
    xs = items["x"][~np.isnan(items["x"])]
    lo, hi = float(xs.min()), float(xs.max())
    ci = res_main["ci"].get("theta_io")
    applicable = ci is not None and ci[0] >= lo and ci[1] <= hi
    out = {"applicable": bool(applicable), "x_range": [lo, hi], "theta_io_ci": ci}
    if applicable:
        out["verdict"] = res_main["decision"]["ratio"]
        return out
    out["verdict"] = "not_applicable"
    per = {}
    for b in np.unique(items["bin"]):
        m = items["bin"] == b
        if b < 0 or not m.any():
            continue
        dok = items["mcq_ok"][m, 0] > 0.5
        bok = items["box_ok"][m].astype(bool)
        n10, n01 = int((bok & ~dok).sum()), int((~bok & dok).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        per[int(b)] = {"n": int(m.sum()), "direct_minus_box_acc": float(dok.mean() - bok.mean()), "box_only_correct": n10, "direct_only_correct": n01, "mcnemar_p": float(p)}
    out["per_bin"] = per
    return out


# ---------------- 档内连续距离的 logistic 回归 ----------------
def logistic_box_effect(items, knots=None):
    """直接回答对错（顺序 0）~ 样条(log2 距离) + 框推对错；返回 OR 与 Wald 95% CI。"""
    ok = ~np.isnan(items["x"]) & ~np.isnan(items["mcq_ok"][:, 0])
    lx = np.log2(items["x"][ok])
    k = rcs_knots(lx, 4) if knots is None else knots
    X = np.column_stack([rcs_basis(lx, k), items["box_ok"][ok].astype(float)])
    y = items["mcq_ok"][ok, 0]

    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)), X.T @ (1 / (1 + np.exp(-z)) - y)
    r = optimize.minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    p = 1 / (1 + np.exp(-(X @ r.x)))
    H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
    se = math.sqrt(max(np.linalg.inv(H)[-1, -1], 0))
    g = float(r.x[-1])
    return {"or": math.exp(g), "ci": [math.exp(g - 1.96 * se), math.exp(g + 1.96 * se)], "coef": g, "se": se}


def analyze_all(items, dev_items, cfg, n_boot, seed):
    """主判定（src.e1_analysis.analyze）+ 备选 A、B、C + 档内连续距离的 logistic 回归。dev_items 只用于 B 选档。"""
    from src.e1_analysis import analyze
    rng = np.random.default_rng(seed)
    al = cfg["alternatives"]
    main = analyze(items, cfg, n_boot=n_boot, seed=seed)
    out = {"main": {k: main[k] for k in ("ratio", "or", "or_pooled", "theta_dir", "theta_io", "direct_acc", "box_acc", "decision") if k in main}, "main_ci": main["ci"]}
    out["A"] = coupling(items, n_boot, rng, al["A_coupling"])
    out["B"] = or_restricted(items, dev_items, n_boot, rng, al["B_or_restricted"], cfg["thresholds"])
    out["C"] = ratio_restricted(main, items, cfg["thresholds"])
    out["C"].pop("per_bin", None) if out["C"]["verdict"] != "not_applicable" else None
    out["logistic"] = logistic_box_effect(items)
    return out


EXPECT = {   # 各方案在各类真值下“判对”的定义（H2 的智能体与框误差方向部分相关，见 src/e1_sim.simulate_L）
    "A": {"H1": lambda r: r["A"]["verdict"] == "coupled", "H2": lambda r: r["A"]["verdict"] == "coupled", "H3": lambda r: r["A"]["verdict"] == "null_compatible"},
    "A_equiv": {"H3": lambda r: r["A"]["equivalence_null"]},
    "B": {"H1": lambda r: r["B"]["verdict"] == "dependent", "H2": lambda r: r["B"]["verdict"] == "dependent", "H3": lambda r: r["B"]["verdict"] == "independent_compatible"},
    "C": {"H1": lambda r: r["C"]["verdict"] == "equivalent", "H2": lambda r: r["C"]["verdict"] == "below", "H3": lambda r: r["C"]["verdict"] == "equivalent"},
    "main": {"H1": lambda r: r["main"]["decision"]["ratio"] == "equivalent" and r["main"]["decision"]["or"] == "dependent",
             "H2": lambda r: r["main"]["decision"]["ratio"] == "below", "H3": lambda r: r["main"]["decision"]["or"] == "independent_compatible"},
}


def summarize_alt(rows):
    """rows：[{agent, result}]；返回各方案在各真值下判对的概率，以及 B 不可估、C 不适用的比例、B 可用档数中位数。"""
    out = {}
    for ag in ("H1", "H2", "H3"):
        R = [r["result"] for r in rows if r["agent"] == ag]
        if not R:
            continue
        d = {"n": len(R)}
        for meth, ex in EXPECT.items():
            if ag in ex:
                d[f"p_correct_{meth}"] = float(np.mean([ex[ag](r) for r in R]))
        d["p_B_not_estimable"] = float(np.mean([r["B"]["verdict"] == "not_estimable" for r in R]))
        d["median_B_bins_used"] = float(np.median([len(r["B"]["bins_used"]) for r in R]))
        d["p_C_applicable"] = float(np.mean([r["C"]["applicable"] for r in R]))
        d["median_A_beta"] = float(np.median([r["A"]["beta"] for r in R]))
        d["median_A_std_effect"] = float(np.median([r["A"]["std_effect"] for r in R]))
        d["median_logistic_or"] = float(np.median([r["logistic"]["or"] for r in R]))
        d["p_logistic_or_ci_contains_1"] = float(np.mean([r["logistic"]["ci"][0] <= 1 <= r["logistic"]["ci"][1] for r in R]))
        d["median_main_or"] = float(np.median([r["main"]["or"] for r in R]))
        out[ag] = d
    return out

