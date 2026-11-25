"""数据核对与清单：统计文件数/体积、生成 sha256 清单、核对 What's Up 数量与图像是否齐全。
输出：<data>/<dataset>/SHA256SUMS（大文件旁）、manifests/data_summary.json（进 Git）。
数量对不上时以非零退出码结束，并把差异写进 summary（不修改判定标准）。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from src import paths
from src.data import DATASETS, load_dataset

EXPECT = {"whatsup_a_left_right": 206, "whatsup_b_total": 408, "coco_val2017_images": 5000}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def tree(root: Path, write_sums: bool = True) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    size = sum(p.stat().st_size for p in files)
    sums = root / "SHA256SUMS"
    if write_sums and files:
        with open(sums, "w") as f:
            for p in files:
                f.write(f"{sha256(p)}  {p.relative_to(root)}\n")
    return {"path_key": str(root.relative_to(paths.get('data'))), "n_files": len(files), "bytes": size,
            "sha256sums": str(sums.relative_to(paths.get('data'))) if files else None,
            "sha256_of_sums": sha256(sums) if files and sums.exists() else None}


def main() -> int:
    data = paths.get("data")
    summary = {"trees": {}, "datasets": {}, "checks": {}}
    for sub in ("whatsup", "coco", "depth_candidates", "_archives"):
        if (data / sub).is_dir():
            summary["trees"][sub] = tree(data / sub)
            print(sub, summary["trees"][sub], flush=True)
    ok = True
    for name in DATASETS:
        ss = load_dataset(name)
        missing = [s.sample_id for s in ss if not os.path.isfile(s.image_path)]
        summary["datasets"][name] = {
            "n_samples": len(ss), "n_unique_images": len({s.image_path for s in ss}), "n_missing_images": len(missing),
            "missing_examples": missing[:5], "per_relation": dict(sorted(Counter(s.relation for s in ss).items())),
            "n_with_target_box": sum(s.bbox_target is not None for s in ss),
            "n_with_reference_box": sum(s.bbox_reference is not None for s in ss),
            "n_with_both_boxes": sum(s.bbox_target is not None and s.bbox_reference is not None for s in ss)}
        ok &= not missing
    a = summary["datasets"]["whatsup_a"]["per_relation"]
    c = summary["checks"]
    c["whatsup_a_left_right"] = {"got": a.get("left", 0) + a.get("right", 0), "expected": EXPECT["whatsup_a_left_right"]}
    c["whatsup_b_total"] = {"got": summary["datasets"]["whatsup_b"]["n_samples"], "expected": EXPECT["whatsup_b_total"]}
    for sub, key in (("whatsup/controlled_images", "whatsup_a_image_files"), ("whatsup/controlled_clevr", "whatsup_b_image_files"),
                     ("whatsup/vg_images", "gqa_image_files"), ("coco/val2017", "coco_val2017_images")):
        d = data / sub
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if d.is_dir() else 0
        c[key] = {"got": n, **({"expected": EXPECT[key]} if key in EXPECT else {})}
    for k, v in c.items():
        if "expected" in v:
            v["ok"] = v["got"] == v["expected"]
            ok &= v["ok"]
    summary["all_ok"] = bool(ok)
    out = paths.REPO_ROOT / "manifests"
    out.mkdir(exist_ok=True)
    json.dump(summary, open(out / "data_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"checks": c, "all_ok": ok}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
