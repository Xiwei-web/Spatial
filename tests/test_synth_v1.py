import math

import numpy as np
import pytest

from src.synth import generator_v1 as g1
from src.synth.generator import OPPOSITE, load_config


@pytest.fixture(scope="module")
def lib():
    rng = np.random.default_rng(0)
    E, A = [], {}
    for c in range(16):
        for k in range(8):
            h, w = int(rng.integers(150, 260)), int(rng.integers(150, 260))
            yy, xx = np.mgrid[0:h, 0:w]
            m = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1
            m &= ~((xx > w * 0.7) & (yy < h * 0.3 * (k % 3)))
            rgba = np.dstack([np.full((h, w, 3), 10 * c + k, np.uint8), m.astype(np.uint8) * 255])
            cid = f"c{c}_{k}"
            A[cid] = rgba
            E.append({"cutout_id": cid, "category": f"cat{c}", "split": "dev" if k < 4 else "test"})
    return g1.LibraryV1(entries=E, arrays=A)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/synth/trial_v1.yaml")


def test_design_balanced(cfg):
    d = g1.build_design(cfg)
    n = {l["name"]: sum(r["level"] == l["name"] for r in d) for l in g1.levels(cfg)}
    assert len(n) == 11 and max(n.values()) - min(n.values()) <= 1
    for l in n:
        s = [r["side"] for r in d if r["level"] == l]
        assert abs(s.count("dev") - s.count("test")) <= 1


@pytest.mark.parametrize("index", list(range(0, 33)))
def test_scene_constraints(cfg, lib, index):
    row = g1.build_design(cfg)[index]
    out = {r["meta"]["variant"]: r for r in g1.generate_scene(cfg, lib, index)}
    again = {r["meta"]["variant"]: r for r in g1.generate_scene(cfg, lib, index)}
    b, p, i = (out[k]["meta"] for k in ("base", "position_swap", "identity_swap"))
    assert all(out[k]["png"] == again[k]["png"] and out[k]["meta"] == again[k]["meta"] for k in out)
    assert p["relation"] == OPPOSITE[b["relation"]] and i["relation"] == b["relation"]
    for m in (b, p, i):
        for o in list(m["objects"].values()) + m["distractors"]:
            x1, y1, x2, y2 = o["box_xyxy"]
            assert min(x1, y1) >= 16 and max(x2, y2) <= 896 - 16 and o["upscale"] <= 1.5
            assert lib.by_id[o["cutout_id"]]["split"] == row["side"]
        for o in m["objects"].values():
            assert abs(o["sqrt_area"] - row["sqrt_area_px"]) <= 1.0
        if row["subset"] == "main":
            assert m["occlusion"]["intersection_px"] == 0 and m["gap"]["projection_gap_px"] >= 0
        else:
            assert row["tier"][0] <= m["occlusion"]["occluded_fraction"] < row["tier"][1]
    if row["subset"] == "main":
        assert b["gap"]["projection_gap_px"] == p["gap"]["projection_gap_px"] == row["gap_px"]
        assert b["gap"]["min_distance_px"] >= row["gap_px"] - 1e-6
    assert i["identity_swap"]["centroid_deviation_px"] <= 2.0 and abs(math.sqrt(i["identity_swap"]["area_ratio"]) - 1) < 0.02
    assert i["objects"]["reference"] == {**b["objects"]["reference"], **{k: i["objects"]["reference"][k] for k in ("visible_box_xyxy", "visible_area", "visible_mask_rle")}}
    assert i["distractors"] == b["distractors"] == p["distractors"]
