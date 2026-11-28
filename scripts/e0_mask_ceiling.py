"""Table 1 的“上限”行（planner 第 11 轮第 1 条）：真值框中心落在 mask 内的比例，每个数据集一行。
与模型的 center-in-mask 用同一套 mask（scripts/e0_test_metrics.masks_for）和同一种判定（中心取整后落在 mask 内），框换成真值框；
框条目集合与 E0 test 的框题相同（按 图像 + 物体名 去重），只读记录里的真值框和图像尺寸，不读任何模型回答。
另报 What's Up 里 target / reference 分开的比例，以及该 mask 是否为“reference 减去 target 之后”的 mask。
用法：python scripts/e0_mask_ceiling.py [参考模型，默认 Qwen3-VL-8B-Instruct] → manifests/e0_mask_ceiling.json"""
import json
import sys

import numpy as np

from src import e0, paths
from pycocotools import mask as mu

from src.data import load_dataset
from src.data.whatsup_annotations import load_annotations

ann = {(a["dataset"], a["sample_id"]): a for a in load_annotations("auto_v3")}


def masks_for(records):
    """与 scripts/e0_test_metrics.masks_for 逐行相同（那个脚本没有 __main__ 保护，导入会重算全部指标，所以这里抄一份）。"""
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


name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-VL-8B-Instruct"
O = paths.get("outputs") / "e0" / name
recs = {}
for x in sorted(O.glob("test*.jsonl")):
    if not x.name.endswith("_sdpa.jsonl"):
        recs.update(e0.load_records(x))
R = [r for r in recs.values() if r["stage"] == "test" and r["task"] == "box"]
M = masks_for(R)
out = {"reference_model_for_item_list": name, "rule": "真值框中心 ((x0+x1)/2, (y0+y1)/2) 取整后落在 mask 内；mask 与模型的 center-in-mask 相同", "datasets": {}}
seen = set()
for ds in e0.TEST_DATASETS:
    vals, imgs, by_role = [], [], {}
    for r in R:
        if r["dataset"] != ds:
            continue
        k = (ds, r["image_key"], r["box_object"])
        if k in seen or k not in M:
            continue
        seen.add(k)
        x0, y0, x1, y1 = r["truth_xyxy"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        m = M[k]
        H, W = m.shape
        v = bool(0 <= cx < W and 0 <= cy < H and m[int(cy), int(cx)])
        vals.append(v); imgs.append(r["image_key"]); by_role.setdefault(r["role"], []).append(v)
    if not vals:
        out["datasets"][ds] = None
        continue
    x = np.array(vals, float)
    d = {"n_boxes": len(x), "ceiling": (float(x.mean()), e0.bootstrap_mean(x, imgs)), "by_role": {k: [float(np.mean(v)), len(v)] for k, v in by_role.items()}}
    if ds.startswith("whatsup"):
        objs = [ann[(ds, r["sample_id"])]["objects"] for r in R if r["dataset"] == ds]
        d["reference_mask_source"] = {"n_with_mask_reference_raw": int(sum(bool(o["reference"].get("mask_reference_raw")) for o in objs)), "n": len(objs),
                                      "note": "mask_reference_raw 只在 reference 与 target 的 mask 有交集、做过“减去 target”时才非空，存的是减之前的原 mask；masks_for 优先用它，所以 center-in-mask 用的是未减的原 mask"}
    out["datasets"][ds] = d
json.dump(out, open(paths.REPO_ROOT / "manifests" / "e0_mask_ceiling.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
