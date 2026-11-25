"""F4 合成场景生成器的单元测试。用内存里的假切图库（彩色几何形状），不依赖任何数据文件，CPU 上几秒跑完。"""
import copy
import io
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.synth import backgrounds, cutouts as cu
from src.synth.cutouts import CutoutLibrary
from src.synth.generator import OPPOSITE, build_design, dumps, generate_scene, load_config, scene_seed

CFG = load_config("configs/synth/trial_v0.yaml")
# 覆盖：最小 / 最大间距、三种大小、两个轴、0–6 个干扰物
INDICES = [0, 1, 2, 3, 10, 21, 32, 65, 131, 263, 264, 499]


def _fake_library() -> CutoutLibrary:
    rng = np.random.default_rng(0)
    entries, arrays = [], {}
    for c in range(14):
        for k in range(4):
            w, h = int(rng.integers(90, 220)), int(rng.integers(90, 220))
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            col = tuple(int(v) for v in rng.integers(0, 256, 3)) + (255,)
            shape = (c + k) % 3
            if shape == 0:
                dr.ellipse([0, 0, w - 1, h - 1], fill=col)
            elif shape == 1:
                dr.polygon([(0, h - 1), (w - 1, h - 1), (w // 3, 0)], fill=col)      # 质心不在框中心
            else:
                dr.rectangle([0, 0, w - 1, h - 1], fill=col)
            cid = f"fake_{c:02d}_{k}"
            arrays[cid] = cu.tight_crop(np.asarray(im))
            entries.append({"cutout_id": cid, "source": "fake", "category": f"cat{c:02d}", "split": "dev" if k < 2 else "test"})
    return CutoutLibrary(entries=entries, arrays=arrays)


@pytest.fixture(scope="module")
def scenes():
    lib = _fake_library()
    return {i: generate_scene(CFG, lib, i) for i in INDICES}


def _centroid(rle):
    ys, xs = np.nonzero(cu.rle_decode(rle))
    return xs.mean() + 0.5, ys.mean() + 0.5


def test_design_table():
    rows = build_design(CFG)
    f = CFG["factors"]
    assert len(rows) == 500
    n_cells = len(f["distance_px"]) * len(f["size_frac"]) * len(f["axis"]) * len(f["n_distractors"])
    cells = Counter((r["distance_px"], r["size_frac"], r["axis"], r["n_distractors"]) for r in rows)
    assert len(cells) == n_cells == 264 and max(cells.values()) <= 2          # 每个格子 1–2 个场景
    for key in ("distance_px", "size_frac", "axis", "n_distractors", "background", "top", "relation"):
        c = Counter(r[key] for r in rows)
        assert max(c.values()) - min(c.values()) <= 4, (key, c)
    assert set(Counter(r["distance_px"] for r in rows)) == set(f["distance_px"])
    assert build_design(copy.deepcopy(CFG)) == rows


def test_determinism(scenes):
    """同一个场景生成两次（全新的库对象、不同的生成顺序）：PNG 逐字节相同，metadata 完全相同。"""
    lib = _fake_library()
    for i in reversed(INDICES[:6]):
        again = generate_scene(CFG, lib, i)
        for a, b in zip(scenes[i], again):
            assert a["png"] == b["png"]
            assert dumps(a["meta"]) == dumps(b["meta"])
    assert scene_seed(CFG["global_seed"], 7) == scene_seed(CFG["global_seed"], 7) != scene_seed(CFG["global_seed"], 8)


def test_png_matches_array_and_hash(scenes):
    import hashlib
    for r in scenes[0]:
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(r["png"]))), r["image"])
        assert r["image"].shape == (896, 896, 3)
        assert hashlib.sha256(r["png"]).hexdigest() == r["meta"]["image_sha256"]
        json.loads(dumps(r["meta"]))


def test_position_twin_flips_label_and_swaps_positions(scenes):
    for i, (base, pos, _) in scenes.items():
        b, p = base["meta"], pos["meta"]
        assert p["relation"] == OPPOSITE[b["relation"]]
        assert b["twins"] == {"position_swap": p["scene_id"], "identity_swap": scenes[i][2]["meta"]["scene_id"]}
        assert p["twins"]["base"] == b["scene_id"] and p["base_id"] == b["scene_id"]
        for role in ("target", "reference"):
            assert p["objects"][role]["cutout_id"] == b["objects"][role]["cutout_id"]
        # 互换后的质心离对方原来的质心不超过 1 px（两次取整各 ≤ 0.5）
        for a, o in (("target", "reference"), ("reference", "target")):
            ca, co = p["objects"][a]["centroid_xy"], b["objects"][o]["centroid_xy"]
            assert abs(ca[0] - co[0]) <= 1.0 + 1e-6 and abs(ca[1] - co[1]) <= 1.0 + 1e-6
        # 其余完全相同：背景、干扰物、上下层
        assert p["background"] == b["background"] and p["distractors"] == b["distractors"]
        assert p["occlusion"]["top"] == b["occlusion"]["top"]


def test_distance_equals_requested(scenes):
    """由完整 mask（RLE 解码）独立重算质心：沿关系轴的间距与要求值相差 ≤ 0.5 px（整数摆放的理论上限），抖动同理。"""
    for res in scenes.values():
        for r in res:
            m = r["meta"]
            tx, ty = _centroid(m["objects"]["target"]["mask_rle"])
            rx, ry = _centroid(m["objects"]["reference"]["mask_rle"])
            along, perp = (tx - rx, ty - ry) if m["axis"] == "horizontal" else (ty - ry, tx - rx)
            assert abs(abs(along) - m["design"]["distance_px"]) <= 0.5 + 1e-6
            assert abs(perp - m["distance"]["jitter_requested_px"]) <= 0.5 + 1e-6
            assert abs(m["distance"]["jitter_requested_px"]) <= m["distance"]["jitter_max_px"] <= 0.25 * m["design"]["distance_px"]
            assert abs(abs(along) - m["distance"]["along_axis_px"]) < 1e-3
            want = {"horizontal": ("left", "right"), "vertical": ("above", "below")}[m["axis"]][int(along > 0)]
            assert m["relation"] == want


def test_masks_boxes_and_occlusion(scenes):
    for res in scenes.values():
        for r in res:
            m = r["meta"]
            full = {k: cu.rle_decode(o["mask_rle"]) for k, o in m["objects"].items()}
            vis = {k: cu.rle_decode(o["visible_mask_rle"]) for k, o in m["objects"].items()}
            occ = m["occlusion"]
            top, bot = occ["top"], occ["bottom"]
            assert {top, bot} == {"target", "reference"} and m["z_order"][-1] == top and m["z_order"][-2] == bot
            assert np.array_equal(vis[top], full[top])
            assert np.array_equal(vis[bot], full[bot] & ~full[top])
            inter = int((full["target"] & full["reference"]).sum())
            assert occ["intersection_px"] == inter and occ["occluded"] == (inter > 0)
            assert abs(occ["occluded_fraction"] - inter / full[bot].sum()) < 1e-5
            assert abs(occ["iou_complete"] - inter / (full["target"] | full["reference"]).sum()) < 1e-5
            assert 1 - occ["occluded_fraction"] >= CFG["occlusion"]["min_visible_fraction"]
            for k, o in m["objects"].items():
                assert o["box_xyxy"] == cu.mask_bbox(full[k]) and o["area"] == int(full[k].sum())
                assert o["visible_box_xyxy"] == cu.mask_bbox(vis[k]) and o["visible_area"] == int(vis[k].sum())
                assert abs(o["long_side_px"] - m["design"]["size_px"]) <= 1          # 大小 = 紧致框长边
                assert max(o["box_xyxy"][2] - o["box_xyxy"][0], o["box_xyxy"][3] - o["box_xyxy"][1]) == o["long_side_px"]


def test_small_distance_gives_occlusion(scenes):
    assert scenes[0][0]["meta"]["design"]["distance_px"] == 8
    assert scenes[0][0]["meta"]["occlusion"]["occluded"] and scenes[0][0]["meta"]["occlusion"]["occluded_fraction"] > 0.3


def test_everything_inside_canvas(scenes):
    mgn = CFG["placement"]["canvas_margin_px"]
    for res in scenes.values():
        for r in res:
            m = r["meta"]
            for o in list(m["objects"].values()) + m["distractors"]:
                x1, y1, x2, y2 = o["box_xyxy"]
                assert mgn <= x1 < x2 <= 896 - mgn and mgn <= y1 < y2 <= 896 - mgn
            px = m["pair_position"]
            assert px["feasible_x"][0] <= px["translation_xy"][0] <= px["feasible_x"][1]
            assert px["feasible_y"][0] <= px["translation_xy"][1] <= px["feasible_y"][1]


def test_distractors(scenes):
    gap = CFG["placement"]["distractor_gap_px"]

    def apart(a, b):
        return a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]

    seen = set()
    for i, res in scenes.items():
        mains = [o["box_xyxy"] for r in res for o in r["meta"]["objects"].values()]      # 三张图里主物体的所有位置
        for r in res:
            m = r["meta"]
            ds = m["distractors"]
            assert len(ds) == m["design"]["n_distractors"]
            seen.add(len(ds))
            cats = [o["category"] for o in m["objects"].values()] + [d["category"] for d in ds]
            assert len(set(cats)) == len(cats)                                            # 类别互不相同
            union = cu.rle_decode(m["objects"]["target"]["mask_rle"]) | cu.rle_decode(m["objects"]["reference"]["mask_rle"])
            for a, d in enumerate(ds):
                assert all(apart(d["box_xyxy"], b) for b in mains)
                assert all(apart(d["box_xyxy"], e["box_xyxy"]) for e in ds[a + 1:])
                dm = cu.rle_decode(d["mask_rle"])
                assert not (dm & union).any() and int(dm.sum()) == d["area"]
    assert seen == {0, 2, 4, 6}


def test_identity_twin(scenes):
    for base, _, idt in scenes.values():
        b, t = base["meta"], idt["meta"]
        assert t["relation"] == b["relation"] and t["variant"] == "identity_swap"
        assert t["objects"]["reference"]["cutout_id"] == b["objects"]["reference"]["cutout_id"]
        assert t["objects"]["reference"]["box_xyxy"] == b["objects"]["reference"]["box_xyxy"]
        assert t["objects"]["target"]["category"] not in (b["objects"]["target"]["category"], b["objects"]["reference"]["category"])
        assert t["replaced"]["original_cutout_id"] == b["objects"]["target"]["cutout_id"]
        assert abs(t["objects"]["target"]["long_side_px"] - b["objects"]["target"]["long_side_px"]) <= 1
        ct, cb = t["objects"]["target"]["centroid_xy"], b["objects"]["target"]["centroid_xy"]
        assert abs(ct[0] - cb[0]) <= 1.0 + 1e-6 and abs(ct[1] - cb[1]) <= 1.0 + 1e-6      # 位置不变（取整误差内）
        assert t["background"] == b["background"] and t["distractors"] == b["distractors"]


def test_only_requested_split_is_used(scenes):
    for res in scenes.values():
        for r in res:
            m = r["meta"]
            ids = [o["cutout_id"] for o in m["objects"].values()] + [d["cutout_id"] for d in m["distractors"]]
            assert all(i.endswith(("_0", "_1")) for i in ids)                              # 假库里 _0/_1 是 dev


def test_confusable_groups_never_cooccur():
    cfg = copy.deepcopy(CFG)
    cfg["confusable_groups"] = [[f"cat{c:02d}" for c in range(0, 5)]]
    lib = _fake_library()
    for i in (3, 7, 11):
        for r in generate_scene(cfg, lib, i):
            m = r["meta"]
            cats = [o["category"] for o in m["objects"].values()] + [d["category"] for d in m["distractors"]]
            assert sum(c in cfg["confusable_groups"][0] for c in cats) <= 1


def test_backgrounds():
    for t in backgrounds.TYPES:
        p = backgrounds.sample_background(np.random.default_rng(5), t)
        a = backgrounds.render_background(json.loads(json.dumps(p)), 64)                   # 只凭 JSON 参数就能重画
        assert a.shape == (64, 64, 3) and a.dtype == np.uint8 and p["type"] == t
        assert np.array_equal(a, backgrounds.render_background(p, 64))
        assert (len(np.unique(a.reshape(-1, 3), axis=0)) == 1) == (t == "solid")


# ---------------- 切图库 ----------------
def _disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


CU = CFG["cutouts"]


def test_check_mask_reasons():
    ok = _disk(480, 640, 240, 320, 80)
    assert cu.check_mask(ok, [], CU)[0] is None
    assert cu.check_mask(_disk(480, 640, 240, 320, 20), [], CU)[0] == "small_area"
    assert cu.check_mask(_disk(480, 640, 240, 80, 80), [], CU)[0] == "touches_border"
    two = _disk(480, 640, 150, 150, 60) | _disk(480, 640, 330, 450, 60)
    assert cu.check_mask(two, [], CU)[0] == "fragmented"
    thin = np.zeros((480, 640), bool)
    for k in range(-20, 21):
        thin[np.arange(50, 400) + k, np.arange(50, 400)] = True                            # 细斜条：填充率低
    assert cu.check_mask(thin, [], CU)[0] == "low_fill"
    assert cu.check_mask(ok, [_disk(480, 640, 240, 380, 70)], CU)[0] == "overlapped"
    speck = ok.copy()
    speck[10:14, 10:14] = True                                                             # 小碎块：通过，但只保留最大连通块
    reason, st, big = cu.check_mask(speck, [], CU)
    assert reason is None and st["n_components"] == 2 and np.array_equal(big, ok)


def test_make_and_resize_cutout():
    mask = _disk(480, 640, 240, 320, 150)
    img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    rgba = cu.make_cutout(img, mask, 256)
    assert rgba.shape[2] == 4 and max(rgba.shape[:2]) == 256
    small = cu.make_cutout(img, _disk(480, 640, 240, 320, 60), 256)                        # 不放大
    assert small.shape[:2] == (121, 121) and np.array_equal(small[..., 3] > 0, _disk(121, 121, 60, 60, 60))
    for s in (72, 108, 161):
        out = cu.resize_cutout(rgba, s)
        m = out[..., 3] >= 128
        assert max(out.shape[:2]) == s and m.any(axis=0).all() and m.any(axis=1).all()     # 紧致


def test_whatsup_path_with_synthetic_record(tmp_path):
    """What's Up 自动标注的代码路径：只取 auto_qc == pass 的图，来源标成 whatsup_auto_v1（自动标注，未经人工核对）。"""
    H, W = 960, 1280
    img = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
    f = tmp_path / "mug_left_of_plate.jpeg"
    Image.fromarray(img).save(f)
    mt, mr = _disk(H, W, 500, 300, 120), _disk(H, W, 500, 900, 150)

    def obj(name, m):
        x1, y1, x2, y2 = cu.mask_bbox(m)
        return {"name": name, "box_xyxy": [x1, y1, x2, y2], "score": 0.8, "mask_rle": cu._rle(m), "mask_area": int(m.sum()),
                "mask_centroid_xy": [0, 0], "mask_bbox_xyxy": [x1, y1, x2, y2]}

    def rec(sid, qc):
        return {"dataset": "whatsup_a", "sample_id": sid, "relation": "left", "width": W, "height": H, "auto_qc": qc,
                "objects": {"target": obj("Mug", mt), "reference": obj("plate", mr)}}

    recs = [rec("mug_left_of_plate", "pass"), rec("mug_right_of_plate", "review")]
    paths_ = {("whatsup_a", "mug_left_of_plate"): str(f), ("whatsup_a", "mug_right_of_plate"): str(f)}
    cands, funnel = cu.whatsup_candidates(recs, paths_, CU, {("whatsup_a", "mug_left_of_plate"): "dev"})
    assert funnel["whatsup_images_not_pass"] == 1 and funnel["pass"] == 2
    assert [c["cutout_id"] for c in cands] == ["wu_whatsup_a_mug_left_of_plate_target", "wu_whatsup_a_mug_left_of_plate_reference"]
    assert all(c["source"] == "whatsup_auto_v1" == cu.WHATSUP_SOURCE for c in cands)
    assert all("not human-verified" in c["src"]["annotation"] and c["src"]["source_split"] == "dev" for c in cands)
    assert cands[0]["category"] == "mug"
    rgba = cu.make_cutout(img, cu.rle_decode(cands[0]["_mask_rle"]), 256)
    assert rgba.shape == (241, 241, 4)
    assert np.array_equal(rgba[..., :3], img[380:621, 180:421])                            # RGB 原样来自原图
    assert np.array_equal(rgba[..., 3] == 255, mt[380:621, 180:421])                       # alpha 原样来自 mask
    # 遮挡近似：另一个物体压进框里太多 → 淘汰
    recs[0]["objects"]["reference"] = obj("plate", _disk(H, W, 500, 420, 150))
    cands, funnel = cu.whatsup_candidates(recs[:1], paths_, CU)
    assert funnel["overlapped"] == 2 and not cands


def test_cap_and_split():
    cands = [{"cutout_id": f"x_{c}_{i:03d}", "source": "s", "category": f"c{c}"} for c in range(5) for i in range(40 if c else 3)]
    kept, info = cu.cap_and_filter(cands, CU)
    assert info["dropped_categories_too_few"] == {"c0": 3}
    assert Counter(k["category"] for k in kept) == {f"c{c}": 30 for c in range(1, 5)}
    assert kept == cu.cap_and_filter(list(reversed(cands)), CU)[0]                          # 与输入顺序无关
    side = cu.make_split(kept)
    assert side == cu.make_split(list(reversed(kept)))
    for c in range(1, 5):
        assert Counter(side[k["cutout_id"]] for k in kept if k["category"] == f"c{c}") == {"dev": 15, "test": 15}


def test_real_library_split_is_disjoint_if_present():
    """正式切图库存在时：划分文件与 library.jsonl 一致，dev / test 没有共用的切图，类别数 ≥ 40。"""
    from src import paths
    from src.data.splits import split_path
    f_split = split_path("synth_cutouts")
    f_lib = cu.library_dir() / "library.jsonl"
    if not (f_split.is_file() and f_lib.is_file()):
        pytest.skip("正式切图库还没构建")
    sp = json.load(open(f_split))
    lib = CutoutLibrary()
    if lib.sha256 != sp["library_sha256"]:
        pytest.skip("切图库正在重建（library.jsonl 与划分文件的 sha256 暂不一致）")
    assert not set(sp["dev"]) & set(sp["test"])
    lib = CutoutLibrary(split_name="synth_cutouts")
    assert {e["cutout_id"]: e["split"] for e in lib.entries} == {**{i: "dev" for i in sp["dev"]}, **{i: "test" for i in sp["test"]}}
    assert len(lib.categories("both")) >= 40
    assert all(e["source"] in (cu.COCO_SOURCE, cu.WHATSUP_SOURCE) for e in lib.entries)
