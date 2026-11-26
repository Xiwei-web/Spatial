"""参数恢复：模拟智能体（src/e1_sim.py）经过与真实数据相同的分析代码（src/e1_analysis.py），判定规则应能分出 H1 / H2 / H3。
阈值来自 configs/e1.yaml（占位值）。每种智能体跑 N_SEEDS 个种子，报告判定正确率，要求 >= 0.8。"""
import numpy as np
import pytest
import yaml

from src import e1_analysis as A
from src import e1_sim as S

CFG = yaml.safe_load(open("configs/e1.yaml"))
N_SEEDS, N_BOOT = 5, 150
SIGMA = 2.0          # 框中心沿轴误差 2 token：理想观察者的 75% 阈值约 1.9 token，落在 0.71–11.3 的采样范围中间
EXPECT = {"H1": lambda d: d["ratio"] == "equivalent" and d["or"] == "dependent",
          "H2": lambda d: d["ratio"] == "below",
          "H3": lambda d: d["or"] == "independent_compatible"}


def run(agent, seed):
    rng = np.random.default_rng(seed)
    it = S.simulate(agent, S.design_x(rng), rng, sigma=SIGMA)
    return A.analyze(it, CFG, n_boot=N_BOOT, seed=seed)


@pytest.mark.parametrize("agent", ["H1", "H2", "H3"])
def test_recovery(agent):
    res = [run(agent, 1000 + s) for s in range(N_SEEDS)]
    hit = [EXPECT[agent](r["decision"]) for r in res]
    rate = float(np.mean(hit))
    print(agent, "correct rate", rate, [(round(r["ratio"], 3), [round(v, 3) for v in r["ci"]["ratio"]], round(r["or"], 2), [round(v, 2) for v in r["ci"]["or"]], r["decision"]) for r in res])
    assert rate >= 0.8


def test_haldane_and_table():
    box = np.array([1, 1, 0, 0], bool)
    d = np.array([[1, 1], [0, 0], [1, 1], [0, 0]], float)
    a, b, c, dd = A.table_2x2(box, d, np.ones(4))
    assert (a, b, c, dd) == (2, 2, 2, 2)
    assert A.haldane_or(a, b, c, dd) == 1.0
    assert A.haldane_or(10, 0, 0, 10) == pytest.approx(10.5 * 10.5 / 0.25)


def test_psychometric_recovers_threshold():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 4, 20000)
    p = 0.5 + 0.5 * __import__("scipy.stats").stats.norm.cdf((x - 1.0) / 0.7)
    y = (rng.random(len(x)) < p).astype(float)
    th, mu, s, lam = A.fit_psychometric(x, y, np.ones(len(x)), {**CFG["psychometric"], "lapse": "fixed0"})
    assert abs(np.log2(th) - 1.0) < 0.08 and abs(s - 0.7) < 0.1
