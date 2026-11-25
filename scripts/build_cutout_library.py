"""构建合成场景用的切图库（F4）。筛选规则见 src/synth/cutouts.py，阈值见 configs/synth/trial_v0.yaml 的 cutouts 段。

输出：
  <data>/synth/cutouts/png/<cutout_id>.png   RGBA 小图（不进仓库）
  <data>/synth/cutouts/library.jsonl          每个切图一行 metadata（含 dev/test、来源、质量指标、png 的 sha256）
  splits/synth_cutouts.json                   dev / test 的 cutout id 列表（按类别分层、种子 20261125；进 Git）
  manifests/synth_cutouts.json                统计 + 筛选漏斗 + library.jsonl 的 sha256（进 Git）

来源是否纳入完全看构建时文件在不在，并如实写进 manifest：
  splits/coco_holdout.json 存在 → 其中的 COCO 图整张排除；不存在 → 不排除，manifest 里 holdout_exclusion_applied=false（之后必须重跑）
  <data>/whatsup_annotations/auto_v1/annotations.jsonl 存在 → 纳入 What's Up 自动标注（只取 auto_qc == pass）；不存在 → 只有 COCO

用法（全量构建请通过 Slurm 跑）：
  python scripts/build_cutout_library.py [--config configs/synth/trial_v0.yaml] [--force]
  python scripts/build_cutout_library.py --limit-images 300 --out-dir <临时目录>     # 试跑：所有输出都写到临时目录，不动仓库
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src import paths
from src.data import load_dataset
from src.synth import cutouts as cu
from src.synth.generator import load_config


def _coco_membership() -> tuple[dict[str, set[int]], dict]:
    """COCO-spatial / COCO-Pairs 用到的图像 id（只记录，不排除）。"""
    mem = {}
    for name in ("coco_spatial_one", "coco_spatial_two"):
        mem[f"in_{name}"] = {int(Path(s.image_path).stem) for s in load_dataset(name)}
    info = {"coco_pairs_split_present": False}
    from src.data.splits import split_path
    f = split_path("coco_pairs")
    if f.is_file():
        sp = json.load(open(f))
        mem["in_coco_pairs"] = {int(i) for i in sp["dev"] + sp["test"]}
        info = {"coco_pairs_split_present": True, "coco_pairs_split_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return mem, info


def _whatsup_inputs(ann_file: Path):
    records = [json.loads(line) for line in open(ann_file) if line.strip()]
    image_paths, source_split = {}, {}
    for name in ("whatsup_a", "whatsup_b"):
        for s in load_dataset(name):
            image_paths[(name, s.sample_id)] = s.image_path
        f = paths.REPO_ROOT / "splits" / f"{name}.json"
        if f.is_file():
            sp = json.load(open(f))
            for side in ("dev", "test"):
                source_split.update({(name, i): side for i in sp[side]})
    return records, image_paths, source_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth/trial_v0.yaml")
    ap.add_argument("--limit-images", type=int, default=None, help="只扫前 N 张 COCO 图（试跑）")
    ap.add_argument("--out-dir", type=Path, default=None, help="试跑输出目录；给了就不写仓库里的 splits / manifests")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的切图库和 splits/synth_cutouts.json")
    args = ap.parse_args()
    cfg = load_config(args.config)["cutouts"]
    trial = args.out_dir is not None
    if args.limit_images is not None and not trial:
        sys.exit("[error] --limit-images 只能和 --out-dir 一起用（不允许把试跑结果写进正式位置）")
    out = args.out_dir if trial else cu.library_dir()
    f_split = (out / "synth_cutouts.split.json") if trial else out / "synth_cutouts.embedded_split.json"     # 库内嵌划分，仅留档；现行划分 splits/synth_cutouts_v2.json 由 make_joint_coco_split.py 生成
    f_man = (out / "synth_cutouts.manifest.json") if trial else paths.REPO_ROOT / "manifests" / "synth_cutouts.json"
    if not trial and not args.force and ((out / "library.jsonl").exists() or f_split.exists()):
        sys.exit(f"[error] {out / 'library.jsonl'} 或 {f_split} 已存在；确认要重建请加 --force（dev/test 划分会随之改变）")
    (out / "png").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- COCO ----
    f_hold = paths.REPO_ROOT / "splits" / "coco_holdout.json"
    holdout = {int(i) for i in json.load(open(f_hold))["image_ids"]} if f_hold.is_file() else set()
    membership, mem_info = _coco_membership()
    cands, funnel_coco = cu.coco_candidates(cfg, holdout, args.limit_images, membership)
    sources = {cu.COCO_SOURCE: {"included": True, "funnel": dict(funnel_coco),
                                "holdout_exclusion_applied": f_hold.is_file(),
                                "holdout_sha256": hashlib.sha256(f_hold.read_bytes()).hexdigest() if f_hold.is_file() else None,
                                "n_holdout_images": len(holdout), **mem_info}}
    if not f_hold.is_file():
        print("[warn] splits/coco_holdout.json 不存在：没有做 holdout 排除，该文件产生后必须重跑本脚本（--force）")
    print(f"[coco] {dict(funnel_coco)}  ({time.time() - t0:.1f}s)")

    # ---- What's Up 自动标注 ----
    f_wu = paths.get("data") / cfg["whatsup_annotation"]
    if f_wu.is_file():
        records, image_paths, source_split = _whatsup_inputs(f_wu)
        wu, funnel_wu = cu.whatsup_candidates(records, image_paths, cfg, source_split)
        cands += wu
        sources[cu.WHATSUP_SOURCE] = {"included": True, "funnel": dict(funnel_wu), "human_verified": False,
                                      "annotations_sha256": hashlib.sha256(f_wu.read_bytes()).hexdigest()}
        print(f"[whatsup] {dict(funnel_wu)}")
    else:
        sources[cu.WHATSUP_SOURCE] = {"included": False, "human_verified": False,
                                      "reason": f"构建时 <data>/{cfg['whatsup_annotation']} 不存在"}
        print(f"[warn] {f_wu} 不存在：本次切图库不含 What's Up 来源")

    # ---- 截断、划分、裁剪 ----
    kept, cap_info = cu.cap_and_filter(cands, cfg)
    side = cu.make_split(kept, cfg["seed"])
    entries, total_bytes = [], 0
    wanted = set()
    by_image: dict[str, list[dict]] = {}
    for c in kept:
        by_image.setdefault(c["_image_path"], []).append(c)
    for img_path in sorted(by_image):
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        for c in by_image[img_path]:
            rgba = cu.make_cutout(rgb, cu.rle_decode(c["_mask_rle"]), cfg["max_side"])
            png = cu.png_bytes(rgba)
            (out / "png" / f"{c['cutout_id']}.png").write_bytes(png)
            wanted.add(f"{c['cutout_id']}.png")
            total_bytes += len(png)
            e = {k: v for k, v in c.items() if not k.startswith("_")}
            e.update(split=side[c["cutout_id"]], width=int(rgba.shape[1]), height=int(rgba.shape[0]),
                     alpha_area=int((rgba[..., 3] >= cu.ALPHA_THR).sum()), png_sha256=hashlib.sha256(png).hexdigest())
            entries.append(e)
    entries.sort(key=lambda e: e["cutout_id"])
    for p in (out / "png").iterdir():               # 重建时清掉不再属于库的旧小图（只动本脚本自己的输出目录）
        if p.name not in wanted:
            p.unlink()
    raw = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries).encode()
    (out / "library.jsonl").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()

    # ---- 划分文件 + manifest ----
    dev = sorted(i for i, s in side.items() if s == "dev")
    test = sorted(i for i, s in side.items() if s == "test")
    assert not set(dev) & set(test) and len(dev) + len(test) == len(entries) == len({e["cutout_id"] for e in entries})
    cats = sorted({e["category"] for e in entries})
    per_cat = {c: {"dev": 0, "test": 0, **{s: 0 for s in sources}} for c in cats}
    for e in entries:
        per_cat[e["category"]][e["split"]] += 1
        per_cat[e["category"]][e["source"]] += 1
    json.dump({"dataset": "synth_cutouts", "seed": cfg["seed"], "unit": "cutout", "stratify_by": "category",
               "n_total": len(entries), "n_dev": len(dev), "n_test": len(test), "n_categories": len(cats),
               "per_category": {c: [v["dev"], v["test"]] for c, v in per_cat.items()},
               "library_sha256": sha, "dev": dev, "test": test}, open(f_split, "w"), indent=0, ensure_ascii=False)
    coco_e = [e for e in entries if e["source"] == cu.COCO_SOURCE]
    man = {
        "task": "F4 synth cutout library", "config": args.config, "thresholds": cfg, "limit_images": args.limit_images,
        "library": {"paths_key": "data", "relative": "synth/cutouts/library.jsonl", "sha256": sha, "bytes": len(raw),
                    "n_cutouts": len(entries), "png_total_bytes": total_bytes},
        "sources": sources, "cap": cap_info,
        "n_categories": len(cats),
        "n_by_source": dict(Counter(e["source"] for e in entries)),
        "n_by_split": {"dev": len(dev), "test": len(test)},
        "n_by_source_split": {f"{s}:{sp}": sum(e["source"] == s and e["split"] == sp for e in entries)
                              for s in sources for sp in ("dev", "test")},
        "coco_membership_counts": {k: sum(bool(e["src"].get(k)) for e in coco_e) for k in sorted(membership)},
        "coco_n_source_images": len({e["src"]["image_id"] for e in coco_e}),
        "per_category": per_cat,
        "build_seconds": round(time.time() - t0, 1),
    }
    json.dump(man, open(f_man, "w"), indent=1, ensure_ascii=False)
    print(f"[ok] {len(entries)} cutouts / {len(cats)} categories  dev={len(dev)} test={len(test)}  by_source={man['n_by_source']}")
    print(f"[ok] {out / 'library.jsonl'} sha256={sha}\n[ok] {f_split}\n[ok] {f_man}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
