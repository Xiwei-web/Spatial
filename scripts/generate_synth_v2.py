"""合成场景 v2（configs/synth/synth_v2.yaml）：几何干跑、全量生成、确定性检查、统计、联系表。CPU，计算节点上跑。
用法：python scripts/generate_synth_v2.py --dry-run [--workers 8]     只算几何不渲染 → manifests/synth_v2_dryrun.json（覆盖率与验收）
      python scripts/generate_synth_v2.py [--workers 8]               全量生成 → <data>/synth/synth_v2/{images, metadata.jsonl}、manifests/synth_v2.json、<outputs>/synth_review/synth_v2/
确定性：全量生成后重生成 20 个场景，逐字节比较。"""
import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import yaml

from src import paths
from src.synth import generator_v2 as G

REPO = paths.REPO_ROOT
CFG = yaml.safe_load(open(REPO / "configs" / "synth" / "synth_v2.yaml"))
OUT = paths.get("data") / "synth" / "synth_v2"
_G = {}


def _init(dry):
    _G.update(lib=G.LibraryV3(), dry=dry)


def _work(i):
    res = G.generate_scene(CFG, _G["lib"], i, dry_run=_G["dry"])
    if not _G["dry"] and res[0]["png"] is not None:
        for r in res:
            (OUT / r["meta"]["image"]).write_bytes(r["png"])
    return i, [r["meta"] for r in res], [len(r["png"]) if r["png"] else 0 for r in res]


def report(metas):
    base = [m for m in metas if m.get("variant") == "base"]
    failed = [m for m in metas if m.get("failed")]
    ids = {m["index"]: m for m in metas if m.get("variant") == "identity_swap"}
    edges = G.token_edges(CFG)
    labels = [f"{edges[k]:.2f}-{edges[k + 1]:.2f}" for k in range(len(edges) - 1)]
    acc = CFG["acceptance"]
    rep = {"n_base": len(base), "n_failed": len(failed), "failed": failed[:20], "main": {}, "overlap": {}, "size_x_stratum": {}, "identity_swap_flag": {}}
    ok = True
    dmin_obs = min(m["geometry"]["center_along_px"] for m in base if m["subset"] == "main")
    rep["main_min_center_along_px_observed"] = dmin_obs
    for tk, tv in CFG["token_px"].items():
        cnt = Counter((m["geometry"]["token_bin"][tk], m["side"]) for m in base if m["subset"] == "main")
        per = {}
        for k, lab in enumerate(labels):
            dev, test = cnt[(k, "dev")], cnt[(k, "test")]
            tok = [m["geometry"]["center_along_tokens"][tk] for m in base if m["subset"] == "main" and m["geometry"]["token_bin"][tk] == k]
            feasible = edges[k + 1] * tv["px"] > dmin_obs          # 档的上界高于实测能做到的最小质心距离，才算几何上可做
            meets = dev >= acc["main_per_model_bin"]["dev"] and test >= acc["main_per_model_bin"]["test"]
            per[lab] = {"dev": dev, "test": test, "actual_token_range": [round(min(tok), 3), round(max(tok), 3)] if tok else None,
                        "geometrically_feasible": feasible, "meets": meets if feasible else None}
            if feasible and not meets:
                ok = False
        rep["main"][tk] = {"models": tv["models"], "px": tv["px"], "bins": per}
    for lv in sorted({m["level"] for m in base if m["subset"] == "overlap"}):
        c = Counter(m["side"] for m in base if m["level"] == lv)
        meets = c["dev"] >= acc["overlap_per_tier"]["dev"] and c["test"] >= acc["overlap_per_tier"]["test"]
        ok &= meets
        rep["overlap"][lv] = {"dev": c["dev"], "test": c["test"], "meets": meets}
    ct = Counter((m["level"], m["sqrt_area_px"]) for m in base)
    rep["size_x_stratum"] = {lv: {str(s): ct[(lv, s)] for s in CFG["sizes_px"]} for lv in sorted({m["level"] for m in base})}
    for sub in ("main", "overlap"):
        fl = [ids[m["index"]]["identity_swap"]["identity_swap_flag"] for m in base if m["subset"] == sub and m["index"] in ids]
        rep["identity_swap_flag"][sub] = {"n": len(fl), "flagged": int(sum(fl)), "rate": round(float(np.mean(fl)), 4) if fl else None}
    eg = [m["geometry"]["edge_gap_px"] for m in base if m["subset"] == "main"]
    md = [m["geometry"]["min_distance_px"] for m in base if m["subset"] == "main"]
    rep["main_edge_gap_px"] = {"min": int(min(eg)), "median": float(np.median(eg)), "max": int(max(eg))}
    rep["main_min_distance_px"] = {"min": float(min(md)), "median": float(np.median(md)), "max": float(max(md))}
    dz = defaultdict(Counter)
    for m in base:
        for x in m["distractors"]:
            dz[m["level"]][x["sqrt_area_px"]] += 1
    rep["distractor_size_by_level"] = {lv: {str(k): v for k, v in sorted(c.items())} for lv, c in sorted(dz.items())}
    rep["distractor_le_pair_violations"] = sum(1 for m in base for x in m["distractors"] if x["sqrt_area_px"] > m["sqrt_area_px"])
    rep["n_extra_by_round"] = dict(Counter(m["design"].get("extra_round") for m in base if m["design"].get("extra_round")))
    rep["meets_acceptance"] = ok and not failed
    rep["n_base_by_side"] = dict(Counter(m["side"] for m in base))
    rep["within_3000"] = len(base) + len(failed) <= acc["max_base_scenes_without_asking"]
    return rep


def sheets(metas):
    """每个分层 / 遮挡档抽 4 个场景（固定种子），每行 base | 位置互换 | 身份互换。"""
    import random
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    rev = paths.get("outputs") / "synth_review" / "synth_v2"
    rev.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    by = defaultdict(dict)
    for m in metas:
        if m.get("variant"):
            by[m["index"]][m["variant"]] = m
    levels = sorted({v["base"]["level"] for v in by.values()}, key=lambda x: (x[0] != "d", int(x[1:]) if x[0] == "d" else int(x[3:])))
    T = 300
    for p in range(0, len(levels), 4):
        lv = levels[p:p + 4]
        rows = [(l, by[i]) for l in lv for i in random.Random(f"synth_v2_sheet:{l}").sample(sorted(k for k, v in by.items() if v["base"]["level"] == l), 2)]
        im = Image.new("RGB", (3 * (T + 6) + 330, len(rows) * (T + 8) + 24), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((6, 4), "synth_v2: base | position_swap | identity_swap (2 seeded scenes per level)", fill=(0, 0, 0), font=font)
        for k, (l, v) in enumerate(rows):
            y = 24 + k * (T + 8)
            for j, name in enumerate(("base", "position_swap", "identity_swap")):
                im.paste(Image.open(OUT / v[name]["image"]).convert("RGB").resize((T, T), Image.LANCZOS), (6 + j * (T + 6), y))
            b, g = v["base"], v["base"]["geometry"]
            fl = v["identity_swap"]["identity_swap"]["identity_swap_flag"]
            d.multiline_text((3 * (T + 6) + 10, y + 10), f"{b['scene_id']}\n{l} {b['side']} {b['relation']}\nsize {b['sqrt_area_px']} px\nalong {g['center_along_px']:.1f} px\n"
                             f"tokens 28/32/56: {g['center_along_tokens']['t28']:.2f}/{g['center_along_tokens']['t32']:.2f}/{g['center_along_tokens']['t56']:.2f}\n"
                             f"edge gap {g['edge_gap_px']} px\noccl {b['occlusion']['occluded_fraction']:.3f}\nid-swap flag {fl}", fill=(0, 0, 0), font=font)
        im.save(rev / f"sheet_{p // 4:02d}.jpg", quality=88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    A = ap.parse_args()
    n = len(G.build_design(CFG))
    if not A.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(A.workers, initializer=_init, initargs=(A.dry_run,)) as pool:
        res = sorted(pool.imap_unordered(_work, range(n), chunksize=4))
    metas = [m for _, ms, _ in res for m in ms]
    rep = report(metas)
    rep["seconds"] = round(time.time() - t0, 1)
    if A.dry_run:
        est = np.mean([sum(b) for _, _, b in res]) if False else None
        rep["estimated_images"] = 3 * rep["n_base"]
        rep["estimated_disk_gb"] = round(3 * rep["n_base"] * 0.22 / 1024, 2)      # trial_v1 实测平均每张约 0.22 MB（1,500 张 0.33 GB）
        json.dump(rep, open(REPO / "manifests" / "synth_v2_dryrun.json", "w"), indent=1, ensure_ascii=False)
    else:
        with open(OUT / "metadata.jsonl", "w") as fo:
            fo.writelines(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n" for m in metas)
        sizes = [s for _, _, b in res for s in b]
        rep["n_images"] = sum(1 for s in sizes if s)
        rep["disk_gb"] = round(sum(sizes) / 2 ** 30, 3)
        # 确定性：重生成 20 个场景
        lib = G.LibraryV3()
        pick = list(range(0, n, max(1, n // 20)))[:20]
        same = 0
        for i in pick:
            r = G.generate_scene(CFG, lib, i)
            same += all(x["png"] == (OUT / x["meta"]["image"]).read_bytes() for x in r)
        rep["determinism"] = {"n": len(pick), "identical": same}
        rep["metadata_sha256"] = hashlib.sha256((OUT / "metadata.jsonl").read_bytes()).hexdigest()
        sheets(metas)
        json.dump(rep, open(REPO / "manifests" / "synth_v2.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("main", "failed")}, ensure_ascii=False)[:3000])
    for tk, v in rep["main"].items():
        print(tk, {b: (x["dev"], x["test"], x["actual_token_range"], x["meets"]) for b, x in v["bins"].items()})
