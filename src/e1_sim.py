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
