"""单物体定位检查的指标：每个模型 × 尺寸档的 center-in-mask 比例（预测框中心落在物体 mask 内），Wilson 95% 区间。
判定（configs/synth/synth_v2.yaml 的 single_object_check，运行前已提交）：点估计 >= center_in_mask_min 为通过；没通过的模型 × 尺寸档在 E1 里排除。
另报解析失败率、IoU 中位数、中心误差（token）中位数。输出 manifests/single_object_check.json，打印 markdown 表。"""
import json
import math

import numpy as np

from src import e0, e1, paths
from src.bbox_calib import iou
from src.e1_analysis import pred_center, token_scale
from src.prompts import to_original_xyxy
from src.synth.cutouts import rle_decode


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


cfg = e1.synth_cfg()
so = cfg["single_object_check"]
root = paths.get("data") / "synth" / "single_object_v2"
meta = {m["scene_id"]: m for m in map(json.loads, open(root / "metadata.jsonl"))}
res = {"rule": {"center_in_mask_min": so["center_in_mask_min"], "decision": so["decision"]}, "models": {}}
rows = []
for model in so["models"]:
    f = paths.get("outputs") / "synth" / model / "single_object.jsonl"
    if not f.is_file():
        continue
    R = list(e0.load_records(f).values())
    per = {}
    for s in cfg["sizes_px"]:
        rs = [r for r in R if r["sqrt_area_px"] == s]
        hit, ious, cerr, nf = [], [], [], 0
        for r in rs:
            m = meta[r["sample_id"]]
            c = pred_center(r)
            if c is None:
                nf += 1
                hit.append(False)
                continue
            mask = rle_decode(m["object"]["mask_rle"])
            x, y = int(c[0]), int(c[1])
            hit.append(bool(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]))
            b = to_original_xyxy(r["parsed_numbers"], r["bbox_convention"], tuple(r["orig_wh"]), tuple(r["model_wh"]) if r.get("model_wh") else None, tuple(r["view_box"]) if r.get("view_box") else None)
            ious.append(iou(b, r["truth_xyxy"], tuple(r["orig_wh"])))
            t = r["truth_xyxy"]
            sx, sy = token_scale(r)
            cerr.append(math.hypot((c[0] - (t[0] + t[2]) / 2) / sx, (c[1] - (t[1] + t[3]) / 2) / sy))
        k, n = int(sum(hit)), len(hit)
        p = k / n if n else None
        per[str(s)] = {"n": n, "center_in_mask": p, "wilson95": wilson(k, n), "pass": (p is not None and p >= so["center_in_mask_min"]), "parse_fail": nf,
                       "iou_median": float(np.median(ious)) if ious else None, "center_err_tok_median": float(np.median(cerr)) if cerr else None}
        rows.append(f"| {model} | {s} | {n} | {p:.3f} | [{per[str(s)]['wilson95'][0]:.3f}, {per[str(s)]['wilson95'][1]:.3f}] | {'通过' if per[str(s)]['pass'] else '**不通过**'} | {nf} | "
                    f"{per[str(s)]['iou_median'] if per[str(s)]['iou_median'] is None else round(per[str(s)]['iou_median'], 3)} | "
                    f"{per[str(s)]['center_err_tok_median'] if per[str(s)]['center_err_tok_median'] is None else round(per[str(s)]['center_err_tok_median'], 3)} |")
    res["models"][model] = per
json.dump(res, open(paths.REPO_ROOT / "manifests" / "single_object_check.json", "w"), indent=1, ensure_ascii=False)
print("| 模型 | 尺寸 px | n | center-in-mask | Wilson 95% | 判定 | 解析失败 | IoU 中位 | 中心误差中位（token） |\n|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
