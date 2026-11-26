"""E1 分析（planner 第 10 轮）：框推关系、一致率、2×2 与 OR（Haldane）、心理测量曲线 θ_dir、理想观察者 θ_IO、比值，按图像 bootstrap。
两层：build_items(records) 把 E0 / 合成集的 JSONL 记录整理成“条目表”（每个条目 = 一张图里的一对物体）；analyze(items, cfg) 只吃条目表。
模拟（src/e1_sim.py）直接产生条目表，所以参数恢复测试和功效模拟走的是同一段分析代码。
test 安全阀在 src/e1.py:require_prereg_if_test，入口脚本读入记录后先调用。"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import optimize
from scipy.stats import norm

from src import e0
from src.e1 import REL_AXIS
from src.e0_metrics import TOKEN_PX
from src.prompts import LETTERS, OPPOSITE, to_original_xyxy

FIELDS = ("key", "x", "bin", "d_box", "e_t", "e_r", "box_ok", "box_parsed", "mcq_ok", "mcq_agree", "pair_ok", "pair_agree")


# ---------------- 记录 → 条目表 ----------------
def pred_center(r):
    nums = r.get("parsed_numbers")
    if not nums or r.get("error"):
        return None
    vb = r.get("view_box")
    b = to_original_xyxy(nums, r["bbox_convention"], tuple(r["orig_wh"]), tuple(r["model_wh"]) if r.get("model_wh") else None, tuple(vb) if vb else None)
    if not (b[2] > b[0] and b[3] > b[1]):
        return None
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def token_scale(r):
    """原图坐标下一个视觉 token 的边长（x, y）。"""
    vb = r.get("view_box")
    W, H = r["orig_wh"]
    vw, vh = (vb[2] - vb[0], vb[3] - vb[1]) if vb else (W, H)
    mw, mh = r["model_wh"]
    tp = TOKEN_PX[r["model"]]
    return tp * vw / mw, tp * vh / mh


def build_items(records, cfg=None):
    """records：同一模型的记录。返回条目表（dict of np.ndarray）和计数。只保留关系能由框推出（REL_AXIS 里有）的两物体条目。"""
    cfg = cfg or {}
    groups = defaultdict(list)
    for r in records:
        if r.get("variant") is not None and cfg.get("variants") and r["variant"] not in cfg["variants"]:
            continue
        if r.get("subset") is not None and cfg.get("subset") and r["subset"] != cfg["subset"]:
            continue
        if r.get("reference") is None or r.get("relation") not in REL_AXIS:
            continue
        if cfg.get("truncate_to_model_bins") and r.get("dataset") == "synth_v2" and r.get("token_bin") is None:
            continue
        groups[(r["dataset"], r["sample_id"])].append(r)
    rows, keys, skipped = [], {}, defaultdict(int)
    modes = None
    for (ds, sid), R in sorted(groups.items()):
        box = {r["role"]: r for r in R if r["task"] == "box"}
        mcq = sorted([r for r in R if r["task"] == "mcq"], key=lambda r: r["rotation"])
        bins = {r["asked_relation"]: r for r in R if r["task"] == "binary"}
        if set(box) != {"target", "reference"}:
            skipped["box_missing"] += 1      # E0 里框按 图像 + 物体名 去重，同名物体在别的样本里问过；这类条目不进框相关的量
            continue
        r0 = R[0]
        modes = modes or r0["answer_modes"]
        rel = r0["relation"]
        ax, sg = REL_AXIS[rel]
        k = 0 if ax == "x" else 1
        sx, sy = token_scale(box["target"])
        tok = sx if k == 0 else sy
        tt, tr = box["target"]["truth_xyxy"], box["reference"]["truth_xyxy"]
        ct_true = ((tt[0] + tt[2]) / 2, (tt[1] + tt[3]) / 2)
        cr_true = ((tr[0] + tr[2]) / 2, (tr[1] + tr[3]) / 2)
        d_box = sg * (ct_true[k] - cr_true[k]) / tok          # 真框中心的有向差（token），真关系方向为正
        pt, pr = pred_center(box["target"]), pred_center(box["reference"])
        parsed = pt is not None and pr is not None
        e_t = sg * (pt[k] - ct_true[k]) / tok if pt else np.nan    # 有向误差：正 = 往真关系方向偏
        e_r = sg * (pr[k] - cr_true[k]) / tok if pr else np.nan
        if parsed and pt[k] != pr[k]:
            derived = rel if sg * (pt[k] - pr[k]) > 0 else OPPOSITE[rel]
        else:
            derived = None
        box_ok = derived == rel
        mo, ma = [np.nan, np.nan], [np.nan, np.nan]
        for j, r in enumerate(mcq[:2]):
            a = e0.answer(r, modes["mcq"])
            ans_rel = r["options"][LETTERS.index(a)] if a is not None else None
            mo[j] = float(ans_rel == rel)
            ma[j] = float(ans_rel is not None and ans_rel == derived)
        po, pa = np.nan, np.nan
        if rel in bins and OPPOSITE[rel] in bins:
            y1, y2 = e0.answer(bins[rel], modes["binary"]), e0.answer(bins[OPPOSITE[rel]], modes["binary"])
            pair_rel = rel if (y1, y2) == ("yes", "no") else (OPPOSITE[rel] if (y1, y2) == ("no", "yes") else None)
            po, pa = float(pair_rel == rel), float(pair_rel is not None and pair_rel == derived)
        x = r0.get("center_along_tokens")
        b = r0.get("token_bin")
        rows.append((keys.setdefault(r0["image_key"], len(keys)), np.nan if x is None else float(x), -1 if b is None else int(b), d_box, e_t, e_r,
                     box_ok, parsed, mo, ma, po, pa))
    it = {f: np.array([r[i] for r in rows]) for i, f in enumerate(FIELDS)} if rows else {f: np.array([]) for f in FIELDS}
    it["n_keys"] = len(keys)
    return it, dict(skipped)


# ---------------- 统计量 ----------------
def haldane_or(a, b, c, d):
    return (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))


def table_2x2(box_ok, dir_ok, w):
    """box_ok：条目；dir_ok：条目 × 观测（NaN = 缺）；w：条目权重（bootstrap 次数）。返回 a=都对, b=框对答错, c=框错答对, d=都错。"""
    m = ~np.isnan(dir_ok)
    D = np.nan_to_num(dir_ok)
    B = box_ok[:, None].astype(float)
    W = w[:, None] * m
    a = (W * B * D).sum(); b = (W * B * (1 - D)).sum(); c = (W * (1 - B) * D).sum(); d = (W * (1 - B) * (1 - D)).sum()
    return a, b, c, d


def mh_or(box_ok, dir_ok, w, strata):
    """按层（token 档）的 Mantel–Haenszel OR，每层每格加 0.5（Haldane）。不分层时距离同时影响两类对错，合并的 OR 在独立假设下也会偏大。"""
    num = den = 0.0
    for s_ in np.unique(strata):
        m = strata == s_
        a, b, c, d = table_2x2(box_ok[m], dir_ok[m], w[m])
        if a + b + c + d == 0:
            continue
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
    return num / den


def fit_psychometric(x, y, w, pc):
    """x：log2 token；y：命中比例（0/1 或模拟的概率）；w：权重。p = g + (1 - g - λ)·Φ((x - μ)/s)。返回 (θ_tokens, μ, s, λ)。"""
    g, tp = pc["guess"], pc["target_p"]
    lo, hi = (math.log2(v) for v in pc["theta_bounds_tokens"])
    s_lo, s_hi = pc["slope_bounds"]
    fit_l = pc.get("lapse", "fit") == "fit"
    lmax = pc.get("lapse_max", 0.0)
    ok = w > 0
    x, y, w = x[ok], y[ok], w[ok]

    def nll(p):
        mu, ls = p[0], p[1]
        lam = p[2] if fit_l else 0.0
        q = g + (1 - g - lam) * norm.cdf((x - mu) / math.exp(ls))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -float((w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum())
    bounds = [(lo, hi), (math.log(s_lo), math.log(s_hi))] + ([(0.0, lmax)] if fit_l else [])
    best = None
    for mu0 in (lo + (hi - lo) * 0.25, float(np.average(x, weights=w)), lo + (hi - lo) * 0.6):
        p0 = [min(max(mu0, lo), hi), 0.0] + ([lmax / 2] if fit_l else [])
        r = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, s = best.x[0], math.exp(best.x[1])
    lam = best.x[2] if fit_l else 0.0
    z = norm.ppf((tp - g) / (1 - g - lam))
    return 2 ** (mu + s * z), mu, s, lam


def io_curve(items, w, ic, rng):
    """理想观察者：每个条目的正确概率 = P(e_t - e_r > -d_box)，(e_t, e_r) 从该档（按权重）实测的有向误差池里独立抽。"""
    e = np.concatenate([items["e_t"], items["e_r"]])
    bn = np.concatenate([items["bin"], items["bin"]])
    ww = np.concatenate([w, w])
    ok = ~np.isnan(e) & (ww > 0)
    bins = sorted(set(items["bin"].tolist()))
    p = np.full(len(items["bin"]), np.nan)
    for b in bins:
        sel = ok & (bn == b)
        rad = 0
        while ww[sel].sum() < ic["min_pool"] and rad < 20:      # 并入相邻档
            rad += 1
            sel = ok & (np.abs(bn - b) <= rad)
        if ww[sel].sum() == 0:
            continue
        pool, pw = e[sel], ww[sel] / ww[sel].sum()
        dt = rng.choice(pool, ic["mc_draws"], p=pw) - rng.choice(pool, ic["mc_draws"], p=pw)
        dt.sort()
        idx = items["bin"] == b
        p[idx] = 1 - np.searchsorted(dt, -items["d_box"][idx], side="right") / len(dt)
    return p


def point_stats(items, w, cfg, rng):
    pc, ic = cfg["psychometric"], cfg["ideal_observer"]
    src = cfg.get("direct_source", "mcq")
    D = items["mcq_ok"] if src == "mcq" else items["pair_ok"][:, None]
    A = items["mcq_agree"] if src == "mcq" else items["pair_agree"][:, None]
    a, b, c, d = table_2x2(items["box_ok"], D, w)
    strata = items["bin"] if cfg.get("or_method", "mantel_haenszel_by_token_bin") == "mantel_haenszel_by_token_bin" else np.zeros(len(w), int)
    out = {"or": mh_or(items["box_ok"], D, w, strata), "or_pooled": haldane_or(a, b, c, d), "table": [a, b, c, d]}
    m = ~np.isnan(A)
    out["agreement"] = float((w[:, None] * np.nan_to_num(A)).sum() / max((w[:, None] * m).sum(), 1e-12))
    m = ~np.isnan(D)
    out["direct_acc"] = float((w[:, None] * np.nan_to_num(D)).sum() / max((w[:, None] * m).sum(), 1e-12))
    out["box_acc"] = float((w * items["box_ok"]).sum() / w.sum())
    has_x = ~np.isnan(items["x"])
    if has_x.sum() >= 10:
        lx = np.log2(items["x"][has_x])
        Dx, wx = D[has_x], w[has_x]
        mm = ~np.isnan(Dx)
        xr = np.repeat(lx[:, None], Dx.shape[1], 1)[mm]
        th_dir = fit_psychometric(xr, Dx[mm], np.repeat(wx[:, None], Dx.shape[1], 1)[mm], pc)
        sub = {k: (v[has_x] if isinstance(v, np.ndarray) else v) for k, v in items.items()}
        p_io = io_curve(sub, wx, ic, rng)
        ok = ~np.isnan(p_io)
        th_io = fit_psychometric(lx[ok], p_io[ok], wx[ok], {**pc, "lapse": ic.get("lapse", "fixed0")})
        out.update(theta_dir=th_dir[0], theta_io=th_io[0], ratio=th_dir[0] / th_io[0], fit_dir=th_dir[1:], fit_io=th_io[1:])
    return out


def analyze(items, cfg, n_boot=None, seed=None):
    """点估计 + 按图像 bootstrap 的 95% CI（OR 在 log 尺度取分位数，等价）。"""
    n_boot = cfg["n_boot"] if n_boot is None else n_boot
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    w1 = np.ones(len(items["key"]))
    est = point_stats(items, w1, cfg, rng)
    nk = int(items["key"].max()) + 1 if len(items["key"]) else 0
    keys = items["key"].astype(int)
    boots = defaultdict(list)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, nk, nk), minlength=nk).astype(float)
        s = point_stats(items, cnt[keys], cfg, rng)
        for k in ("or", "or_pooled", "agreement", "direct_acc", "box_acc", "theta_dir", "theta_io", "ratio"):
            if k in s:
                boots[k].append(s[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    res = {"n_items": int(len(keys)), "n_images": nk, "n_boot": n_boot, **{k: (float(v) if np.isscalar(v) else [float(t) for t in v]) for k, v in est.items()}, "ci": ci}
    res["decision"] = decide(res, cfg["thresholds"])
    return res


def decide(res, th):
    """按 configs/e1.yaml 的阈值分类（阈值是占位值）。"""
    out = {}
    if "ratio" in res.get("ci", {}):
        lo, hi = res["ci"]["ratio"]
        e_lo, e_hi = th["ratio_equivalence"]
        out["ratio"] = "equivalent" if (lo >= e_lo and hi <= e_hi) else ("below" if hi < th["ratio_h2_upper"] else ("above" if lo > e_hi else "inconclusive"))
    lo, hi = res["ci"]["or"]
    out["or"] = "dependent" if lo > th["or_dependent_min"] else ("independent_compatible" if lo <= 1 <= hi else "inconclusive")
    return out
