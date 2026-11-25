"""构建 COCO holdout 图像池（任务 F5）：从 COCO val2017 里选 1000 张**既不在 COCO-spatial（one / two）、
也不在任何一条 COCO-Pairs 记录（dev 和 test 都算）**里的图像。

用途：只用来算与评测无关的统计量（例如视觉 embedding 的均值）。绝不从评测图像里取。
剩余图像不足 1000 张时，写出全部剩余图像并如实报告数量。

输出：
  splits/coco_holdout.json          排序后的图像 id + 文件名 + 种子 + 各来源的排除数量；已存在时拒绝覆盖，除非 --force
  docs/parts/F5_coco_holdout.md     报告（追加写入）

用法：
  python scripts/build_coco_holdout.py [--force]
  python scripts/build_coco_holdout.py --pairs <试跑的 pairs.jsonl> --out-dir <临时目录>    # 试跑
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from src import paths
from src.data import load_dataset
from src.data import coco_pairs as cp

SEED = cp.SEED
N_HOLDOUT = 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 splits/coco_holdout.json")
    ap.add_argument("--pairs", type=Path, default=None, help="pairs.jsonl 路径（默认 <data>/coco_pairs/pairs.jsonl）")
    ap.add_argument("--out-dir", type=Path, default=None, help="把输出写到这个目录（试跑用）")
    args = ap.parse_args()
    if args.pairs is not None and args.out_dir is None:
        sys.exit("[error] 指定 --pairs 属于试跑，必须同时指定 --out-dir，避免把基于不完整 pairs 的结果写到正式位置")

    f_pairs = cp.pairs_file() if args.pairs is None else args.pairs
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        f_out, f_md = args.out_dir / "coco_holdout.json", args.out_dir / "F5_coco_holdout.md"
    else:
        f_out = paths.REPO_ROOT / "splits" / "coco_holdout.json"
        f_md = paths.REPO_ROOT / "docs" / "parts" / "F5_coco_holdout.md"
    if f_out.exists() and not args.force:
        sys.exit(f"[error] {f_out} 已存在；拒绝覆盖（确认要重做请加 --force）")
    if not f_pairs.is_file():
        sys.exit(f"[error] 找不到 {f_pairs}；请先完整跑完 scripts/build_coco_pairs.py")

    coco = cp.load_coco_instances()
    all_ids = set(coco["images"])
    in_one = {int(Path(s.image_path).stem) for s in load_dataset("coco_spatial_one")}
    in_two = {int(Path(s.image_path).stem) for s in load_dataset("coco_spatial_two")}
    raw = f_pairs.read_bytes()
    pairs_sha = hashlib.sha256(raw).hexdigest()
    in_pairs = {json.loads(line)["image_id"] for line in raw.decode().splitlines() if line.strip()}
    # 正式构建时：pairs.jsonl 必须就是 summary 里登记的那一份，且划分文件里的图像也一并排除（应与 pairs 的图像集合相同）
    in_split: set[int] = set()
    if args.pairs is None:
        summ = json.load(open(paths.REPO_ROOT / "manifests" / "coco_pairs_summary.json"))
        if summ["pairs_jsonl"]["sha256"] != pairs_sha or summ.get("limit") is not None:
            sys.exit("[error] pairs.jsonl 与 manifests/coco_pairs_summary.json 登记的 sha256 不一致（或 summary 来自试跑）")
        sp = json.load(open(cp.split_file()))
        in_split = set(sp["dev"]) | set(sp["test"])
        if in_split != in_pairs:
            sys.exit("[error] splits/coco_pairs.json 的图像集合与 pairs.jsonl 不一致")
    for name, ids in (("coco_spatial_one", in_one), ("coco_spatial_two", in_two), ("coco_pairs", in_pairs)):
        assert ids <= all_ids, f"{name} 里有不属于 val2017 的图像 id"

    excluded = in_one | in_two | in_pairs | in_split
    remaining = sorted(all_ids - excluded)
    n_sel = min(N_HOLDOUT, len(remaining))
    rng = random.Random(f"{SEED}:coco_holdout")
    selected = sorted(rng.sample(remaining, n_sel))
    assert not set(selected) & excluded and len(set(selected)) == n_sel
    file_names = [coco["images"][i]["file_name"] for i in selected]
    missing = [fn for fn in file_names if not cp.coco_image_path(fn).is_file()]
    if missing:
        sys.exit(f"[error] {len(missing)} 张选中的图像在磁盘上不存在，例如 {missing[:3]}")

    counts = {
        "n_val2017_images": len(all_ids),
        "excluded_coco_spatial_one": len(in_one),
        "excluded_coco_spatial_two": len(in_two),
        "excluded_coco_spatial_either": len(in_one | in_two),
        "excluded_coco_pairs": len(in_pairs),
        "excluded_coco_pairs_only": len(in_pairs - in_one - in_two),
        "excluded_coco_spatial_only": len((in_one | in_two) - in_pairs),
        "excluded_both_spatial_and_pairs": len((in_one | in_two) & in_pairs),
        "excluded_union": len(excluded),
        "n_remaining": len(remaining),
        "n_requested": N_HOLDOUT,
        "n_selected": n_sel,
    }
    out = {"dataset": "coco_holdout", "seed": SEED,
           "purpose": "只用于与评测无关的统计量（例如视觉 embedding 均值）；不得用于评测或任何选择类操作",
           "sampling": "对排序后的剩余图像 id 用 random.Random(f'{seed}:coco_holdout').sample 抽取，再排序",
           "pairs_jsonl_sha256": pairs_sha, "counts": counts, "image_ids": selected, "file_names": file_names}
    json.dump(out, open(f_out, "w"), indent=0, ensure_ascii=False)

    short = "" if n_sel == N_HOLDOUT else f"\n**剩余图像不足 {N_HOLDOUT} 张，实际只写出 {n_sel} 张。**\n"
    md = f"""
# F5 COCO holdout 图像池

由 `scripts/build_coco_holdout.py` 自动生成。输出 `splits/coco_holdout.json`（排序后的图像 id 与文件名）。

用途：只用来算与评测无关的统计量（例如视觉 embedding 的均值）。这些图像既不在 COCO-spatial（one / two），
也不在任何一条 COCO-Pairs 记录里（dev、test 都排除）。种子 {SEED}；
抽样方式：把剩余图像 id 排序后用 `random.Random("{SEED}:coco_holdout").sample` 抽取，再排序。
排除时所依据的 pairs.jsonl 的 sha256：`{pairs_sha}`。
{short}
| 项 | 图像数 |
|---|---|
| COCO val2017 全部图像 | {counts['n_val2017_images']} |
| 被 coco_spatial_one 排除 | {counts['excluded_coco_spatial_one']} |
| 被 coco_spatial_two 排除 | {counts['excluded_coco_spatial_two']} |
| 被 COCO-spatial（one ∪ two）排除 | {counts['excluded_coco_spatial_either']} |
| 被 COCO-Pairs（dev ∪ test）排除 | {counts['excluded_coco_pairs']} |
| 其中只被 COCO-Pairs 排除 | {counts['excluded_coco_pairs_only']} |
| 其中只被 COCO-spatial 排除 | {counts['excluded_coco_spatial_only']} |
| 其中两者都排除 | {counts['excluded_both_spatial_and_pairs']} |
| 排除合计（并集） | {counts['excluded_union']} |
| 剩余可用 | {counts['n_remaining']} |
| 选入 holdout | {counts['n_selected']} |

各来源之间有重叠，所以分项之和大于并集。
"""
    f_md.parent.mkdir(parents=True, exist_ok=True)
    with open(f_md, "a") as fh:          # 追加写入
        fh.write(md.lstrip("\n") if not f_md.stat().st_size else md)
    print(f"[ok] {f_out}\n[ok] {f_md}")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
