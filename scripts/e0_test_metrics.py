"""E0 test 主跑的指标（CPU，只读 test JSONL）→ manifests/e0_test.json 与 markdown 草表（模型 × 数据集 × 关系类型，95% 区间）。
只算 E0 的量；不算任何“框推关系”的量（见 src/e0_metrics.py 开头）。用法：python scripts/e0_test_metrics.py"""
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from src import e0, paths
from src.data import load_dataset
from src.data.whatsup_annotations import load_annotations
from src.e0_metrics import REL_GROUPS, impl_diff, summarize

O = paths.get("outputs") / "e0"
ann = {(a["dataset"], a["sample_id"]): a for a in load_annotations("auto_v3")}


def masks_for(records):
    """center-in-mask 用的 mask：What's Up 用标注里的原始 mask；COCO-spatial 用 COCO 分割；GQA 没有 mask（不算）。"""
    out = {}
    coco = {}
    for r in records:
        if r["task"] != "box":
            continue
        ds, k = r["dataset"], (r["dataset"], r["image_key"], r["box_object"])
        if k in out:
            continue
        if ds.startswith("whatsup"):
            o = ann[(ds, r["sample_id"])]["objects"][r["role"]]
            out[k] = mu.decode((o.get("mask_reference_raw") or o)["mask_rle"]).astype(bool)
        elif ds.startswith("coco"):
            if ds not in coco:
                coco[ds] = {s.sample_id: s for s in load_dataset(ds)}
            s = coco[ds][r["sample_id"]]
            seg = s.mask_target if r["role"] == "target" else s.mask_reference
            W, H = r["orig_wh"]
            from src.synth.cutouts import _coco_ann_mask
            out[k] = _coco_ann_mask({"segmentation": seg}, H, W)
    return out


res, rows = {}, []
for name, fam in e0.FAMILY.items():
    f = O / name / "test.jsonl"
    if not f.is_file():
        continue
    R = [r for r in e0.load_records(f).values() if r["stage"] == "test"]
    if {r["dataset"] for r in R} != set(e0.TEST_DATASETS):
        print(name, "还没跑完全部 test 数据集", sorted({r["dataset"] for r in R})); continue
    S = summarize(R, masks_for(R))
    d = {"family": fam, "test": S}
    fs = O / name / "test_sdpa.jsonl"
    if fs.is_file():
        RS = [r for r in e0.load_records(fs).values() if r["stage"] == "test"]
        d["default_impl_sdpa"] = summarize(RS, masks_for(RS))
        d["impl_diff_whatsup"] = impl_diff([r for r in R if r["dataset"].startswith("whatsup")], RS)
    d["seconds_total"] = sum(r.get("seconds") or 0 for r in R)
    res[name] = d
json.dump(res, open(paths.REPO_ROOT / "manifests" / "e0_test.json", "w"), indent=1, ensure_ascii=False)
f3 = lambda t: "—" if t is None else f"{t[0]:.3f} [{t[1][0]:.3f}, {t[1][1]:.3f}]"      # noqa: E731
print("| 模型 | 数据集 | 选择题（轮换后） | 各关系类型 | 成对二元 | IoU@0.5 | 平均 IoU | center-in-mask | 中心误差中位数（物体尺寸 / token） | 左右半区 | 解析失败 框 / 选择 |\n|" + "---|" * 11)
for n, d in res.items():
    for ds, x in d["test"].items():
        groups = "；".join(f"{lab.split('（')[0]} {x[f'mcq_acc[{lab}]'][0]:.3f}" for lab, _ in REL_GROUPS if f"mcq_acc[{lab}]" in x)
        ce = f"{x['center_err_obj_median']:.3f} / {x['center_err_tok_median']:.2f}" if x.get("center_err_obj_median") is not None else "—"
        print(f"| {n} | {ds} | {f3(x['mcq_acc'])} | {groups} | {f3(x['binary_pair_acc'])} | {f3(x.get('iou@0.5'))} | {f3(x.get('mean_iou'))} | {f3(x.get('center_in_mask'))} | {ce} | {f3(x.get('half_correct'))} | {x.get('box_parse_fail', 0):.3f} / {x['mcq_parse_fail']:.3f} |")
