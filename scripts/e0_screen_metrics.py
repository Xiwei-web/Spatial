"""What's Up dev 入选筛选的指标（CPU，只读 screen JSONL）。按图像 bootstrap 10,000 次给 95% 区间。不做入选决定，只列出按标准通过与否。
用法：python scripts/e0_screen_metrics.py → manifests/e0_screen_dev.json，并打印 markdown"""
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu

from src import e0, paths
from src.bbox_calib import iou
from src.data.whatsup_annotations import load_annotations
from src.prompts import to_original_xyxy

ann = {(a["dataset"], a["sample_id"]): a for a in load_annotations("auto_v3")}
_masks = {}


def orig_mask(ds, sid, role):
    k = (ds, sid, role)
    if k not in _masks:
        o = ann[(ds, sid)]["objects"][role]
        _masks[k] = mu.decode((o.get("mask_reference_raw") or o)["mask_rle"]).astype(bool)      # 与 box_gt 同一个原始 mask
    return _masks[k]


REL_COLS = [("left/right", {"left", "right"}), ("on/under（A）", {"on", "under"}), ("front/behind（B；与垂直位置完全混淆）", {"front", "behind"})]
out, rows = {}, []
for name, fam in e0.FAMILY.items():
    f = paths.get("outputs") / "e0" / name / "screen.jsonl"
    if not f.is_file():
        continue
    R = [r for r in e0.load_records(f).values() if r["stage"] == "screen"]
    modes = R[0]["answer_modes"]
    box = [r for r in R if r["task"] == "box"]
    mcq = [r for r in R if r["task"] == "mcq"]
    bi = [r for r in R if r["task"] == "binary"]
    if len(box) != 816 or len(mcq) != 1632 or len(bi) != 816:
        print(name, "还没跑完", len(box), len(mcq), len(bi)); continue
    m = {}
    iou_v, cim, img_b = [], [], []
    for r in box:
        nums = r.get("parsed_numbers")
        pred = to_original_xyxy(nums, r["bbox_convention"], tuple(r["orig_wh"]), tuple(r["model_wh"]) if r["model_wh"] else None, tuple(r["view_box"]) if r["view_box"] else None) if nums else None
        iou_v.append(iou(pred, r["truth_xyxy"], r["orig_wh"]))
        inside = False
        if pred is not None and pred[2] > pred[0] and pred[3] > pred[1]:
            cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
            W, H = r["orig_wh"]
            if 0 <= cx < W and 0 <= cy < H:
                inside = bool(orig_mask(r["dataset"], r["sample_id"], r["role"])[int(cy), int(cx)])
        cim.append(inside); img_b.append(r["image_key"])
    iou_v, cim = np.array(iou_v), np.array(cim, float)
    m["iou@0.5"] = (float((iou_v >= .5).mean()), e0.bootstrap_mean(iou_v >= .5, img_b))
    m["mean_iou"] = (float(iou_v.mean()), e0.bootstrap_mean(iou_v, img_b))
    m["center_in_mask"] = (float(cim.mean()), e0.bootstrap_mean(cim, img_b))
    m["box_parse_fail"] = float(np.mean([r.get("parsed_numbers") is None for r in box]))
    c = np.array([e0.correct(r, e0.answer(r, modes["mcq"])) for r in mcq], float)
    im = [r["image_key"] for r in mcq]
    m["mcq_rot_acc"] = (float(c.mean()), e0.bootstrap_mean(c, im))
    m["mcq_parse_fail"] = float(np.mean([e0.answer(r, modes["mcq"]) is None for r in mcq]))
    for lab, rels in REL_COLS:
        sel = np.array([r["relation"] in rels for r in mcq])
        m[f"mcq_acc[{lab}]"] = (float(c[sel].mean()), e0.bootstrap_mean(c[sel], [x for x, z in zip(im, sel) if z]))
    for ds in ("whatsup_a", "whatsup_b"):
        sel = np.array([r["dataset"] == ds for r in mcq])
        m[f"mcq_acc[{ds}]"] = (float(c[sel].mean()), e0.bootstrap_mean(c[sel], [x for x, z in zip(im, sel) if z]))
    pair = defaultdict(list)
    for r in bi:
        pair[(r["dataset"], r["sample_id"])].append(e0.correct(r, e0.answer(r, modes["binary"])))
    pk = sorted(pair)
    pv = np.array([all(pair[k]) and len(pair[k]) == 2 for k in pk], float)
    m["binary_pair_acc"] = (float(pv.mean()), e0.bootstrap_mean(pv, [f"{d}:{s}" for d, s in pk]))
    m["binary_parse_fail"] = float(np.mean([e0.answer(r, modes["binary"]) is None for r in bi]))
    m["n_errors"] = sum(bool(r.get("error")) for r in R)
    m["seconds_total"] = sum(r.get("seconds") or 0 for r in R)
    acc, i5 = m["mcq_rot_acc"][0], m["iou@0.5"][0]
    m["pass_strict"] = acc >= .8 and i5 >= .7
    m["pass_relaxed_iou60"] = acc >= .8 and i5 >= .6
    out[name] = {"family": fam, "answer_modes": modes, "metrics": m, "git_commit": R[0]["git_commit"], "calibrated_sha": R[0]["calibrated_sha"], "answer_modes_sha": R[0]["answer_modes_sha"]}
fams = lambda k: sorted({v["family"] for v in out.values() if v["family"] and v["metrics"][k]})      # noqa: E731
summary = {"families_pass_strict": fams("pass_strict"), "families_pass_relaxed_iou60": fams("pass_relaxed_iou60")}
json.dump({"models": out, "summary": summary, "note": "What's Up dev 筛选；入选决定由 planner / Xiwei 做，本脚本只列出按标准通过与否"}, open(paths.REPO_ROOT / "manifests" / "e0_screen_dev.json", "w"), indent=1, ensure_ascii=False)
f3 = lambda t: f"{t[0]:.3f} [{t[1][0]:.3f}, {t[1][1]:.3f}]"      # noqa: E731
print("| 模型 | 家族 | 四选一轮换后准确率 | IoU@0.5 | 平均 IoU | center-in-mask | 成对二元 | left/right | on/under（A） | front/behind（B，与垂直位置完全混淆） | 答案方式 mcq / binary | 解析失败 框 / mcq / 二元 | 严格标准 | IoU 放宽到 0.6 |\n|" + "---|" * 14)
for n, v in out.items():
    m = v["metrics"]
    print(f"| {n} | {v['family'] or '附录基线'} | {f3(m['mcq_rot_acc'])} | {f3(m['iou@0.5'])} | {f3(m['mean_iou'])} | {f3(m['center_in_mask'])} | {f3(m['binary_pair_acc'])} | "
          f"{f3(m['mcq_acc[left/right]'])} | {f3(m['mcq_acc[on/under（A）]'])} | {f3(m['mcq_acc[front/behind（B；与垂直位置完全混淆）]'])} | {v['answer_modes']['mcq']} / {v['answer_modes']['binary']} | "
          f"{m['box_parse_fail']:.3f} / {m['mcq_parse_fail']:.3f} / {m['binary_parse_fail']:.3f} | {'通过' if m['pass_strict'] else '不通过'} | {'通过' if m['pass_relaxed_iou60'] else '不通过'} |")
print("合格家族（严格）:", summary["families_pass_strict"], "；IoU 放宽到 0.6:", summary["families_pass_relaxed_iou60"])
