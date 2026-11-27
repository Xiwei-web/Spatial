"""备选方案的单元测试（快）：样条基、条件 MLE OR、A/B/C 在中间噪声下的基本行为。完整的参数恢复在 scripts/e1_recovery_alt.py（CPU 作业）。"""
import numpy as np
import pytest
import yaml

from src import e1_alt as X
from src import e1_sim as S

CFG = yaml.safe_load(open("configs/e1.yaml"))


def test_rcs_linear_tail():
    x = np.linspace(0, 10, 200)
    B = X.rcs_basis(x, X.rcs_knots(x, 4))
    assert B.shape == (200, 4)
    d2 = np.diff(B[-30:, 2:], n=2, axis=0)          # 最后一个结之外是线性的
    assert np.abs(d2).max() < 1e-8


def test_cond_mle_or_null_and_strong():
    r = X.cond_mle_or([(25, 25, 25, 25), (30, 20, 30, 20)])
    assert r["ci"][0] < 1 < r["ci"][1]
    r = X.cond_mle_or([(40, 10, 10, 40)])
    assert r["or"] == pytest.approx(16, rel=0.15) and r["ci"][0] > 4


@pytest.mark.parametrize("agent,coupled", [("H1", True), ("H3", False)])
def test_coupling_direction(agent, coupled):
    rng = np.random.default_rng(3)
    it = S.simulate_L(agent, S.design_x(rng, 80), rng, sigma=2.0)
    c = X.coupling(it, 100, rng, CFG["alternatives"]["A_coupling"])
    assert (c["verdict"] == "coupled") == coupled


def test_b_not_estimable_near_ceiling():
    rng = np.random.default_rng(4)
    it = S.simulate_L("H3", S.design_x(rng, 80), rng, sigma=0.1)
    dv = S.simulate_L("H3", S.design_x(rng, 40), rng, sigma=0.1)
    o = X.or_restricted(it, dv, 50, rng, CFG["alternatives"]["B_or_restricted"], CFG["thresholds"])
    assert o["verdict"] == "not_estimable"
