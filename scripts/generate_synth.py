"""按配置生成合成场景（F4）：每个基础场景 3 张图（base / position_swap / identity_swap）+ 每张图一行 metadata。

输出（<out> 默认 = <data>/synth/<version>/）：
  <out>/images/<scene_id>.png
  <out>/metadata.jsonl          每张图一行，按 (index, base → position_swap → identity_swap) 排序
  <out>/timing.json             每个场景的生成耗时（不进 metadata，否则 metadata 就不确定了）
  manifests/synth_<version>.json  设计表统计、遮挡统计、耗时 / 体积及全量外推、确定性检查结果、metadata 的 sha256

用法（完整试跑请通过 Slurm）：
  python scripts/generate_synth.py --config configs/synth/trial_v0.yaml --workers 8 --verify 20
  python scripts/generate_synth.py --limit 6 --out-dir <临时目录> [--library-dir <临时切图库>]   # 登录节点上的小试
  python scripts/generate_synth.py --only-index 123 --out-dir <临时目录>                          # 单独重生成一个场景
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from src import paths
from src.synth import CutoutLibrary, build_design, generate_scene, load_config
from src.synth.generator import dumps

_G: dict = {}


def _init(cfg, lib_dir, out_dir):
    _G.update(cfg=cfg, lib=CutoutLibrary(lib_dir, split_name=cfg.get("cutout_split_name")), out=Path(out_dir) if out_dir else None)


def _work(index: int):
    t = time.perf_counter()
    res = generate_scene(_G["cfg"], _G["lib"], index)
    t_gen = time.perf_counter() - t
    if _G["out"] is not None:
        for r in res:
            (_G["out"] / r["meta"]["image"]).write_bytes(r["png"])
    return index, [dumps(r["meta"]) for r in res], [len(r["png"]) for r in res], t_gen, time.perf_counter() - t


def _pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def summarize(metas: list[dict], cfg: dict) -> dict:
    base = [m for m in metas if m["variant"] == "base"]
    f = cfg["factors"]

    def count(key, rows=base):
        c = Counter(m["design"][key] for m in rows)
        return {str(k): c.get(k, 0) for k in f.get(key if key != "top" else "_", sorted(c, key=str))}

    cross = defaultdict(Counter)
    for m in base:
        d = m["design"]
        cross[str(d["distance_px"])][f"size={d['size_frac']}"] += 1
        cross[str(d["distance_px"])][f"axis={d['axis']}"] += 1
        cross[str(d["distance_px"])][f"n_dis={d['n_distractors']}"] += 1
        cross[str(d["distance_px"])][f"bg={d['background']}"] += 1
        cross[str(d["distance_px"])][f"top={d['top']}"] += 1
        cross[str(d["distance_px"])][f"rel={m['relation']}"] += 1

    def occ(rows):
        out = {}
        for dist in f["distance_px"]:
            sub = [m["occlusion"] for m in rows if m["design"]["distance_px"] == dist]
            o = [x["occluded_fraction"] for x in sub if x["occluded"]]
            out[str(dist)] = {"n": len(sub), "n_occluded": len(o), "share_occluded": round(len(o) / len(sub), 4) if sub else None,
                              "median_occluded_fraction_all": round(statistics.median(x["occluded_fraction"] for x in sub), 4) if sub else None,
                              "median_occluded_fraction_given_occluded": round(statistics.median(o), 4) if o else None,
                              "max_occluded_fraction": round(max(o), 4) if o else 0.0,
                              "median_iou_complete": round(statistics.median(x["iou_complete"] for x in sub), 4) if sub else None}
        return out

    occ_size = {}
    for dist in f["distance_px"]:
        for s in f["size_frac"]:
            sub = [m["occlusion"] for m in base if m["design"]["distance_px"] == dist and m["design"]["size_frac"] == s]
            if sub:
                occ_size[f"{dist}|{s}"] = {"n": len(sub), "share_occluded": round(sum(x["occluded"] for x in sub) / len(sub), 4),
                                           "median_occluded_fraction_all": round(statistics.median(x["occluded_fraction"] for x in sub), 4)}
    errs = [abs(m["distance"]["error_px"]) for m in metas]
    sides = [o["long_side_px"] - m["design"]["size_px"] for m in metas for o in m["objects"].values()]
    cats = Counter(o["category"] for m in base for o in m["objects"].values())
    return {
        "n_base_scenes": len(base), "n_images": len(metas),
        "n_cells_covered": len({m["design"]["cell"] for m in base}),
        "n_cells_total": len(f["distance_px"]) * len(f["size_frac"]) * len(f["axis"]) * len(f["n_distractors"]),
        "design_counts": {"distance_px": count("distance_px"), "size_frac": count("size_frac"), "axis": count("axis"),
                          "n_distractors": count("n_distractors"), "background": count("background"),
                          "top": dict(Counter(m["design"]["top"] for m in base)),
                          "relation": dict(sorted(Counter(m["relation"] for m in base).items()))},
        "design_cross_by_distance": {k: dict(sorted(v.items())) for k, v in cross.items()},
        "occlusion_by_distance": {"base": occ(base), "all_variants": occ(metas)},
        "occlusion_by_distance_and_size_base": occ_size,
        "distance_abs_error_px": {"max": max(errs), "mean": round(sum(errs) / len(errs), 4)},
        "long_side_minus_requested_px": dict(sorted(Counter(sides).items())),
        "attempts": dict(sorted(Counter(m["seed"]["attempt"] for m in base).items())),
        "rejection_reasons": dict(Counter(r for m in base for r in m["seed"]["rejected"])),
        "n_categories_used_as_main_object": len(cats),
        "object_sources_main": dict(Counter(o["source"] for m in metas for o in m["objects"].values())),
    }


def verify(cfg, lib_dir, out: Path, lines_by_index: dict[int, list[str]], n: int) -> dict:
    """在主进程里重新生成 n 个等间隔的场景，与磁盘上的 PNG 和 metadata 逐字节比较。"""
    lib = CutoutLibrary(lib_dir, split_name=cfg.get("cutout_split_name"))
    idx = sorted(lines_by_index)
    pick = [idx[int(i * (len(idx) - 1) / max(1, n - 1))] for i in range(min(n, len(idx)))]
    pick = sorted(set(pick))
    bad = []
    for i in pick:
        for r, line in zip(generate_scene(cfg, lib, i), lines_by_index[i]):
            disk = (out / r["meta"]["image"]).read_bytes()
            if disk != r["png"] or dumps(r["meta"]) != line or hashlib.sha256(disk).hexdigest() != r["meta"]["image_sha256"]:
                bad.append(r["meta"]["scene_id"])
    return {"n_scenes_checked": len(pick), "n_images_checked": 3 * len(pick), "indices": pick,
            "byte_identical_png_and_metadata": not bad, "mismatches": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth/trial_v0.yaml")
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--limit", type=int, default=None, help="只生成前 N 个场景（小试；需要 --out-dir）")
    ap.add_argument("--only-index", type=int, nargs="*", default=None, help="只生成这些序号（需要 --out-dir）")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--library-dir", type=Path, default=None)
    ap.add_argument("--verify", type=int, default=0, help="生成后重生成 N 个场景做逐字节比较")
    ap.add_argument("--full-n-scenes", type=int, default=5000, help="外推全量耗时 / 体积时用的基础场景数")
    args = ap.parse_args()
    cfg = load_config(args.config)
    partial = args.limit is not None or args.only_index is not None
    if partial and args.out_dir is None:
        sys.exit("[error] --limit / --only-index 只能和 --out-dir 一起用（不允许把不完整的结果写进正式位置）")
    out = args.out_dir or paths.get("data") / "synth" / cfg["version"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    design = build_design(cfg)
    indices = args.only_index if args.only_index is not None else list(range(len(design)))[: args.limit]

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init, initargs=(cfg, args.library_dir, out)) as pool:
        results = sorted(pool.imap_unordered(_work, indices, chunksize=4))
    wall = time.perf_counter() - t0

    lines_by_index = {i: lines for i, lines, *_ in results}
    raw = "".join(line + "\n" for i in sorted(lines_by_index) for line in lines_by_index[i]).encode()
    (out / "metadata.jsonl").write_bytes(raw)
    metas = [json.loads(line) for i in sorted(lines_by_index) for line in lines_by_index[i]]
    sizes = [b for _, _, bs, *_ in results for b in bs]
    t_gen = [r[3] for r in results]
    t_all = [r[4] for r in results]
    json.dump({"workers": args.workers, "wall_seconds": wall,
               "per_scene": [{"index": r[0], "generate_s": round(r[3], 4), "generate_and_write_s": round(r[4], 4)} for r in results]},
              open(out / "timing.json", "w"))
    n = len(results)
    N = args.full_n_scenes
    lib = CutoutLibrary(args.library_dir, split_name=cfg.get("cutout_split_name"))
    timing = {
        "hardware": f"{'Slurm 分区 ' + os.environ['SLURM_JOB_PARTITION'] + '，作业 ' + os.environ.get('SLURM_JOB_ID', '?') if 'SLURM_JOB_PARTITION' in os.environ else '非 Slurm（本地小试）'}，"
                    f"{args.workers} 个进程 / 可用 {len(os.sched_getaffinity(0))} 核，节点 {os.uname().nodename}",
        "workers": args.workers, "n_scenes": n, "wall_seconds_total": round(wall, 2),
        "wall_seconds_per_scene": round(wall / n, 4),
        "cpu_seconds_per_scene_generate": {"mean": round(sum(t_gen) / n, 4), "median": round(statistics.median(t_gen), 4), "p95": round(_pct(t_gen, 95), 4)},
        "cpu_seconds_per_scene_generate_and_write": {"mean": round(sum(t_all) / n, 4), "median": round(statistics.median(t_all), 4), "p95": round(_pct(t_all, 95), 4)},
        "note": "一个场景 = 3 张图；wall_seconds_total 含进程池启动和各进程首次读切图",
    }
    disk = {"png_bytes_per_image": {"mean": round(sum(sizes) / len(sizes), 1), "median": statistics.median(sizes),
                                    "min": min(sizes), "max": max(sizes)},
            "png_bytes_total": sum(sizes), "metadata_bytes": len(raw), "metadata_bytes_per_image": round(len(raw) / len(metas), 1)}
    projection = {
        "n_base_scenes": N, "n_images": 3 * N,
        "wall_seconds_same_hardware": round(wall / n * N, 1),
        "core_seconds": round(sum(t_all) / n * N, 1),
        "png_bytes": round(sum(sizes) / len(sizes) * 3 * N), "metadata_bytes": round(len(raw) / len(metas) * 3 * N),
        "method": "按试跑实测的 每场景墙钟时间 / 每张图平均字节数 线性外推",
    }
    man = {"task": "F4 synth scenes", "version": cfg["version"], "config": args.config, "partial": partial,
           "cutout_library": {"sha256": lib.sha256, "n_cutouts": len(lib.entries), "split_used": cfg["cutout_split"],
                              "n_categories_available": len(lib.categories(cfg["cutout_split"])),
                              "n_cutouts_available": sum(len(v) for v in lib.categories(cfg["cutout_split"]).values()),
                              "sources_in_library": dict(Counter(e["source"] for e in lib.entries))},
           "metadata": {"paths_key": "data", "relative": f"synth/{cfg['version']}/metadata.jsonl",
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "n_lines": len(metas)},
           "images": {"paths_key": "data", "relative": f"synth/{cfg['version']}/images/", "n": len(sizes)},
           "summary": summarize(metas, cfg), "timing": timing, "disk": disk, "projection_full_set": projection}
    if args.verify:
        man["determinism_check"] = verify(cfg, args.library_dir, out, lines_by_index, args.verify)
        print(f"[verify] {man['determinism_check']}")
    f_man = (out / "manifest.json") if args.out_dir else paths.REPO_ROOT / "manifests" / f"synth_{cfg['version']}.json"
    json.dump(man, open(f_man, "w"), indent=1, ensure_ascii=False)
    print(f"[ok] {n} scenes / {len(sizes)} images → {out}  wall={wall:.1f}s  mean_png={disk['png_bytes_per_image']['mean']:.0f}B")
    print(f"[ok] {f_man}")
    if args.verify and not man["determinism_check"]["byte_identical_png_and_metadata"]:
        sys.exit("[error] 确定性检查失败")


if __name__ == "__main__":
    main()
