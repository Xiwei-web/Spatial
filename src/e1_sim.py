"""E1 的模拟“智能体”，直接产生 src/e1_analysis 的条目表（参数恢复测试与功效模拟共用）。
每个条目：质心沿轴距离 x（token），真框中心差 d_box = x；框中心误差 e_t、e_r 从误差池（或正态）独立抽。
  H1：直接回答 = 两个带噪框中心比较的结果（与框推关系完全一致）
  H2：直接回答读一个噪声更小的粗位置：d + η，η ~ N(0, k·sd(e_t - e_r))，k < 1，每个顺序独立
  H3：直接回答与框独立、精度相同：d + (e_t' - e_r')，(e_t', e_r') 从同一个误差池另抽，每个顺序独立"""
from __future__ import annotations

import numpy as np

from src.e1_analysis import FIELDS


def edges9(base=0.5, ratio=2 ** 0.5, n_edges=10):
    return [base * ratio ** k for k in range(n_edges)]


def token_bins(x, edges):
    b = np.searchsorted(edges, x, side="right") - 1
    return np.where((b >= 0) & (b < len(edges) - 1), b, -1)


def simulate(agent, x, rng, pool=None, sigma=0.5, k_h2=0.5, n_orders=2, axis=None):
    """x：每个条目的距离（token）；pool：误差池（token）——数组，或 {"x": 数组, "y": 数组} 配合 axis（每个条目 "x"/"y"）；None 时用 N(0, sigma)。一个条目一张图。
    H2 的 η 标准差 = k_h2 × sd(e_t - e_r)（按该条目所在轴的池算）。"""
    n = len(x)
    if isinstance(pool, dict):
        axis = np.asarray(axis)
        pools = {a: np.asarray(pool[a], float) for a in ("x", "y")}

        def draw(m):
            out = np.empty(n)
            for a_ in ("x", "y"):
                sel = axis == a_
                out[sel] = rng.choice(pools[a_], int(sel.sum()))
            return out
        sdD = np.where(axis == "x", pools["x"].std(), pools["y"].std()) * np.sqrt(2)
    elif pool is not None:
        draw = lambda m: rng.choice(pool, m)      # noqa: E731
        sdD = float(np.std(np.asarray(pool)) * np.sqrt(2))
    else:
        draw = lambda m: rng.normal(0, sigma, m)      # noqa: E731
        sdD = sigma * np.sqrt(2)
    e_t, e_r = draw(n), draw(n)
    d = np.asarray(x, float)
    box_ok = d + e_t - e_r > 0
    mcq = np.zeros((n, n_orders))
    for j in range(n_orders):
        if agent == "H1":
            mcq[:, j] = box_ok
        elif agent == "H2":
            mcq[:, j] = d + rng.normal(0, 1, n) * k_h2 * sdD > 0
        elif agent == "H3":
            mcq[:, j] = d + draw(n) - draw(n) > 0
        else:
            raise ValueError(agent)
    agree = (mcq == box_ok[:, None]).astype(float)
    it = {"key": np.arange(n), "x": d, "bin": token_bins(d, edges9()), "d_box": d, "e_t": e_t, "e_r": e_r, "box_ok": box_ok, "box_parsed": np.ones(n, bool),
          "mcq_ok": mcq, "mcq_agree": agree, "pair_ok": mcq[:, 0].copy(), "pair_agree": agree[:, 0].copy()}
    assert set(it) == set(FIELDS)
    it["n_keys"] = n
    return it


def design_x(rng, per_bin=160, lo_bin=1, edges=None):
    """各档 per_bin 个条目，档内 log 均匀（默认 0.71–11.3 token 的 8 档，模拟 28 px 模型的覆盖）。"""
    e = edges or edges9()
    return np.concatenate([np.exp(rng.uniform(np.log(e[k]), np.log(e[k + 1]), per_bin)) for k in range(lo_bin, len(e) - 1)])


def robust_sd(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def local_L(S, d, Lm, rng, n_orders):
    """饱和的局部线性 L（第 11 轮试跑估计，scripts/e1_pilot_nuisance.py 的 L_local）：
    L = sign(S)·m(|Δ|) + κ_loc(|Δ|)·(S − Δ) + ν，m 为 log2 距离上的限制性样条（超出试跑范围时截到边界），κ_loc = dm/dΔ，ν ~ N(0, Lm["nu"]) 每个顺序独立。
    回答 = sign(L)：S 与 Δ 同号时 L ≈ m（饱和的置信度），S 翻号（框推错，H1）时 L ≈ −m。"""
    from src.e1_alt import rcs_basis
    kn, coef = np.asarray(Lm["knots"]), np.asarray(Lm["coef"])
    lo, hi = Lm["log2_range"]
    ad = np.maximum(np.abs(d), 1e-3)
    lx = np.clip(np.log2(ad), lo, hi)
    m = rcs_basis(lx, kn) @ coef
    h = 1e-3
    k = (rcs_basis(lx + h, kn) @ coef - rcs_basis(lx - h, kn) @ coef) / (2 * h) / (2 ** lx * np.log(2))
    base = np.where(S >= 0, 1.0, -1.0) * m + k * (S - d)
    return base[:, None] + rng.normal(0, Lm["nu"], (len(d), n_orders))


def winsorize_pool(v, c=3.0):
    """把误差池截到 中位数 ± c·稳健 sd（1.4826·MAD），看离群值的影响（planner 第 11 轮第 5 条的稳健尺度版本）。"""
    v = np.asarray(v, float)
    med, rs = np.median(v), robust_sd(v)
    return np.clip(v, med - c * rs, med + c * rs) if rs > 0 else v


def simulate_L(agent, x, rng, pool=None, sigma=0.5, k_h2=0.5, quant=0.5, kappa=3.0, nu_sd=0.5, n_orders=2, axis=None, size=None, scale="sd", Lmodel=None):
    """带 logit 差 L 和预测框有向差 Δ̂ 的智能体（planner 第 10 轮补充第 4 条）。每个条目一个内部决策量 S（两个选项顺序共用），
    每个顺序的 L = kappa·S / sd(e_t − e_r) + ν，ν ~ N(0, nu_sd)；二选一的回答 = L 的符号（正 = 选对）。
      H1：S = Δ̂ = d + e_t − e_r（框推的有向距离本身）
      H2：S = 量化到 quant token 的 (d + k_h2·(e_t − e_r))：读一个更粗但噪声更小、与框误差方向部分相关的位置
      H3：S = d + (e_t' − e_r')，(e_t', e_r') 从同一误差池另抽，与框误差独立、方差相同
    pool：数组；或 {"x": 数组, "y": 数组}（配合 axis）；或 {(轴, 尺寸): 数组}（配合 axis、size，按尺寸档分池，两个物体同尺寸）；None 时 N(0, sigma)。
    scale："sd"（默认）：标准化尺度 sd(e_t − e_r) = 池的标准差 × √2；"mad"：改用稳健尺度 1.4826·MAD(池) × √2（planner 第 11 轮第 5 条，看离群值的影响）。"""
    sdf = (lambda v: float(np.std(np.asarray(v, float)))) if scale == "sd" else robust_sd
    n = len(x)
    if isinstance(pool, dict):
        axis = np.asarray(axis)
        size = np.asarray(size) if size is not None else None
        keyf = (lambda i: (axis[i], size[i])) if size is not None else (lambda i: axis[i])
        groups = {}
        for i in range(n):
            groups.setdefault(keyf(i), []).append(i)
        groups = {k: np.array(v) for k, v in groups.items()}

        def draw():
            out = np.empty(n)
            for k, idx in groups.items():
                out[idx] = rng.choice(np.asarray(pool[k], float), len(idx))
            return out
        sdD = np.empty(n)
        for k, idx in groups.items():
            sdD[idx] = sdf(pool[k]) * np.sqrt(2)
    elif pool is not None:
        draw = lambda: rng.choice(pool, n)      # noqa: E731
        sdD = np.full(n, sdf(pool) * np.sqrt(2))
    else:
        draw = lambda: rng.normal(0, sigma, n)      # noqa: E731
        sdD = np.full(n, sigma * np.sqrt(2))
    sdD = np.maximum(sdD, 1e-6)
    d = np.asarray(x, float)
    e_t, e_r = draw(), draw()
    dhat = d + e_t - e_r
    box_ok = dhat > 0
    if agent == "H1":
        S = dhat
    elif agent == "H2":
        S = quant * np.round((d + k_h2 * (e_t - e_r)) / quant)
    elif agent == "H3":
        S = d + draw() - draw()
    else:
        raise ValueError(agent)
    L = kappa * (S / sdD)[:, None] + rng.normal(0, nu_sd, (n, n_orders)) if Lmodel is None else local_L(S, d, Lmodel, rng, n_orders)
    mcq = (L > 0).astype(float)
    agree = (mcq == box_ok[:, None]).astype(float)
    it = {"key": np.arange(n), "x": d, "bin": token_bins(d, edges9()), "d_box": d, "e_t": e_t, "e_r": e_r, "box_ok": box_ok, "box_parsed": np.ones(n, bool),
          "mcq_ok": mcq, "mcq_agree": agree, "pair_ok": mcq[:, 0].copy(), "pair_agree": agree[:, 0].copy(), "dhat": dhat, "L": L, "n_keys": n}
    return it



def simulate_real(agent, delta, axis, rng, pools, kappa, nu_sd, tau, bin_edges, scale="sd", n_orders=2, Lmodel=None):
    """真实数据集的智能体（planner 第 11 轮第 6 条）。delta：每个条目的真框中心有向差（token，标注关系方向为正，可 <= 0）；axis："x"/"y"；
    pools：{"x": 数组, "y": 数组}（dev 实测的框中心有向误差）。Δ̂ = Δ + e_t − e_r，框推正确 = Δ̂ > 0。
      H1：S = Δ̂ + tau·sdD·z（直接回答读两个框的比较，外加与框无关的噪声 tau；tau 用来对上 dev 的直接回答正确率，>= 0）
      H3：S = Δ + tau·sdD·z（与框误差独立的高斯噪声；tau 用来对上 dev 的直接回答正确率）
    L = kappa·S/sdD + ν（每个顺序独立），回答 = sign(L)。距离档：|Δ| 按 dev 的三分位数 bin_edges 分 3 档（MH OR 的分层）。"""
    n = len(delta)
    d = np.asarray(delta, float)
    axis = np.asarray(axis)
    sdf = (lambda v: float(np.std(v))) if scale == "sd" else robust_sd
    P = {a: np.asarray(pools[a], float) for a in ("x", "y") if len(pools.get(a, []))}
    if len(P) == 1:                                   # 某个轴没有误差样本时借用另一个轴
        a0 = next(iter(P)); P = {"x": P[a0], "y": P[a0]}
    sdD = np.where(axis == "x", sdf(P["x"]), sdf(P["y"])) * np.sqrt(2)
    sdD = np.maximum(sdD, 1e-6)

    def draw():
        out = np.empty(n)
        for a in ("x", "y"):
            m = axis == a
            out[m] = rng.choice(P[a], int(m.sum()))
        return out
    e_t, e_r = draw(), draw()
    dhat = d + e_t - e_r
    box_ok = dhat > 0
    if agent == "H1":
        S = dhat + tau * sdD * rng.normal(0, 1, n)
    elif agent == "H3":
        S = d + tau * sdD * rng.normal(0, 1, n)
    else:
        raise ValueError(agent)
    L = kappa * (S / sdD)[:, None] + rng.normal(0, nu_sd, (n, n_orders)) if Lmodel is None else local_L(S, d, Lmodel, rng, n_orders)
    mcq = (L > 0).astype(float)
    ax_ = np.abs(d)
    b = np.searchsorted(np.asarray(bin_edges), ax_, side="right")
    return {"key": np.arange(n), "x": np.maximum(ax_, 1e-3), "bin": b, "d_box": d, "e_t": e_t, "e_r": e_r, "box_ok": box_ok, "box_parsed": np.ones(n, bool),
            "mcq_ok": mcq, "dhat": dhat, "L": L, "n_keys": n}      # n_orders = 1 时 mcq_ok 那一列代表成对二元的一次观测
