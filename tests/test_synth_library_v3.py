"""抠图库 v3 新增特征（手写数据）。"""
import numpy as np
import pytest

from src.synth import library_v3 as L


def test_occ_contact_and_straight_run():
    m = np.zeros((60, 60), bool); m[10:40, 10:30] = True
    o = np.zeros_like(m); o[10:40, 31:50] = True                     # 右边紧挨着另一个实例
    near = L.near_others(o, 3)
    f = L.occ_contact_frac(m, near)
    assert 0.2 < f < 0.35                                            # 右边一条边（约占周长 30/100）
    poly = [[10, 10, 30, 10, 30, 40, 10, 40]]
    assert L.straight_runs(poly, 600, near, 5, 0.5) == pytest.approx(30 / np.sqrt(600))      # 只有接触的右边
    assert L.straight_runs(poly, 600, np.zeros_like(m), 5, 0.5) == 0
    assert L.straight_runs(poly, 600, None, 5, 0.5) == pytest.approx(30 / np.sqrt(600))
    # 两段共线的线段合并
    assert L.straight_runs([[0, 0, 10, 0, 20, 0, 20, 10, 0, 10]], 200, None, 5, 0.5) == pytest.approx(20 / np.sqrt(200))


def test_person_components_solidity():
    k = [0] * 51
    for i in (0, 15, 16):
        k[3 * i + 2] = 2
    assert L.person_visible(k)
    k[3 * 16 + 2] = 0
    assert not L.person_visible(k)
    m = np.zeros((40, 40), bool); m[5:15, 5:15] = True; m[25:35, 25:35] = True; m[0, 39] = True
    assert L.n_components(m, 0.01) == 2
    assert L.solidity(m) < 0.4
    sq = np.zeros((20, 20), bool); sq[2:12, 2:12] = True
    assert L.solidity(sq) == pytest.approx(1.0)


def test_rates():
    r = L.rates([True, False, True, False, True], ["BAD", "BAD", "OK", "OK", "UNSURE"])
    assert r == {"n_bad": 2, "n_ok": 2, "bad_recall": 0.5, "ok_loss": 0.5}
