"""E0 test 指标（只算 E0 的量，不碰 E1 的 T1 检验）。
算：框指标（IoU@0.1/0.5/0.7/0.9、平均 IoU、center-in-mask、中心误差（以物体尺寸和视觉 token 为单位）、左右半区）、关系准确率（按关系类型分列）、成对二元。
**不算**：用两个预测框的中心推出关系、框推关系与直接回答的一致率、2×2 列联表、按间距分层的对比——这些属于 E1 的 T1 检验，
docs/PREREGISTRATION.md 提交之前不得在 test 上计算或查看（planner 第 9 轮硬性要求）。本模块里没有这类代码。"""
from __future__ import annotations

import math

import numpy as np

from src import e0
from src.bbox_calib import iou
from src.prompts import to_original_xyxy

# 视觉 token 在模型输入图像上的边长（像素）：Qwen2.5-VL 系 / GLM 为 14×2，Qwen3-VL / Qwen3.5 为 16×2，InternVL 系为 448/16，LLaVA-1.5 为 336/24
# （与 configs/synth/trial_v1.yaml 的 token_cell_px 一致：那里是 896 画布上的值）
TOKEN_PX = {"Qwen2.5-VL-7B-Instruct": 28, "VST-7B-RL": 28, "RoboBrain2.0-7B": 28, "GLM-4.6V-Flash": 28, "Qwen3-VL-8B-Instruct": 32, "Qwen3.5-2B": 32, "Qwen3.5-4B": 32,
            "Qwen3.5-9B": 32, "InternVL3-8B": 28, "InternVL3_5-8B": 28, "InternVL3-2B": 28, "SenseNova-SI-1.1-InternVL3-2B": 28, "SenseNova-SI-1.2-InternVL3-8B": 28,
            "SenseNova-SI-1.3-InternVL3-8B": 28, "LLaVA-1.5-7B": 14, "LLaVA-1.5-13B": 14}
REL_GROUPS = [("left/right", {"left", "right"}), ("on/under", {"on", "under"}), ("above/below", {"above", "below"}), ("top/bottom", {"top", "bottom"}),
              ("front/behind（与垂直位置完全混淆）", {"front", "behind"})]
FORBIDDEN_KEYS = ("box_relation", "derived_relation", "consistency", "contingency", "agreement_box", "gap_strat")


def box_record_metrics(r: dict, mask=None) -> dict:
    nums = r.get("parsed_numbers")
    W, H = r["orig_wh"]
    vb = r.get("view_box")
    pred = to_original_xyxy(nums, r["bbox_convention"], (W, H), tuple(r["model_wh"]) if r.get("model_wh") else None, tuple(vb) if vb else None) if nums and not r.get("error") else None
    t = r["truth_xyxy"]
    v = iou(pred, t, (W, H))
    out = {"iou": v, "parsed": pred is not None}
    tcx, tcy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
    if pred is not None and pred[2] > pred[0] and pred[3] > pred[1]:
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        out["center_err_obj"] = math.hypot(cx - tcx, cy - tcy) / math.sqrt(max((t[2] - t[0]) * (t[3] - t[1]), 1e-9))
        vw, vh = (vb[2] - vb[0], vb[3] - vb[1]) if vb else (W, H)
        mw, mh = r["model_wh"]
        tp = TOKEN_PX[r["model"]]
        out["center_err_tok"] = math.hypot((cx - tcx) / (tp * vw / mw), (cy - tcy) / (tp * vh / mh))
        out["half_ok"] = (cx < W / 2) == (tcx < W / 2)
        out["center_in_mask"] = None if mask is None else bool(0 <= cx < W and 0 <= cy < H and mask[int(cy), int(cx)])
    else:
        out.update(center_err_obj=None, center_err_tok=None, half_ok=False, center_in_mask=None if mask is None else False)
    return out


def summarize(model_records: list[dict], masks: dict | None = None) -> dict:
    """model_records：同一模型、同一 stage 的记录（按 e0.load_records 取最后一条）。masks：{(dataset, image_path, object): bool mask} 或 None。
    返回 {dataset: {...}}；CI 为按图像 bootstrap 10,000 次。"""
    out = {}
    for ds in sorted({r["dataset"] for r in model_records}):
        R = [r for r in model_records if r["dataset"] == ds]
        modes = R[0]["answer_modes"]
        d = {}
        box = [r for r in R if r["task"] == "box"]
        if box:
            bm = [box_record_metrics(r, (masks or {}).get((ds, r.get("image_key"), r.get("box_object")))) for r in box]
            img = [r["image_key"] for r in box]
            iv = np.array([m["iou"] for m in bm])
            for th in (0.1, 0.5, 0.7, 0.9):
                x = (iv >= th).astype(float)
                d[f"iou@{th}"] = (float(x.mean()), e0.bootstrap_mean(x, img))
            d["mean_iou"] = (float(iv.mean()), e0.bootstrap_mean(iv, img))
            hx = np.array([float(m["half_ok"]) for m in bm])
            d["half_correct"] = (float(hx.mean()), e0.bootstrap_mean(hx, img))
            for k in ("center_err_obj", "center_err_tok"):
                vals = [m[k] for m in bm if m[k] is not None]
                d[f"{k}_median"] = float(np.median(vals)) if vals else None
            cim = [(m["center_in_mask"], i) for m, i in zip(bm, img) if m["center_in_mask"] is not None]
            if cim:
                x = np.array([float(v) for v, _ in cim])
                d["center_in_mask"] = (float(x.mean()), e0.bootstrap_mean(x, [i for _, i in cim]))
            d["box_parse_fail"] = float(np.mean([not m["parsed"] for m in bm]))
            d["n_box"] = len(box)
        mcq = [r for r in R if r["task"] == "mcq"]
        c = np.array([e0.correct(r, e0.answer(r, modes["mcq"])) for r in mcq], float)
        im = [r["image_key"] for r in mcq]
        d["mcq_acc"] = (float(c.mean()), e0.bootstrap_mean(c, im))
        d["mcq_parse_fail"] = float(np.mean([e0.answer(r, modes["mcq"]) is None for r in mcq]))
        d["n_choices"] = len(mcq[0]["options"])
        for lab, rels in REL_GROUPS:
            sel = np.array([r["relation"] in rels for r in mcq])
            if sel.any():
                d[f"mcq_acc[{lab}]"] = (float(c[sel].mean()), e0.bootstrap_mean(c[sel], [x for x, z in zip(im, sel) if z]))
        pair = {}
        for r in [r for r in R if r["task"] == "binary"]:
            pair.setdefault((r["sample_id"], r["image_key"]), []).append(e0.correct(r, e0.answer(r, modes["binary"])))
        pk = sorted(pair)
        pv = np.array([float(len(pair[k]) == 2 and all(pair[k])) for k in pk])
        d["binary_pair_acc"] = (float(pv.mean()), e0.bootstrap_mean(pv, [k[1] for k in pk]))
        d["n_samples"] = len(pk)
        d["n_errors"] = sum(bool(r.get("error")) for r in R)
        out[ds] = d
    for ds, d in out.items():
        assert not any(f in k for k in d for f in FORBIDDEN_KEYS), "E0 test 指标里不得出现框推关系一类的量"
    return out


def impl_diff(eager: list[dict], sdpa: list[dict]) -> dict:
    """默认实现（sdpa）与 eager 补丁在同一批提示上的差异：逐 token 一致率、答案一致率（按题型）。"""
    E = {e0.record_key(r)[1:]: r for r in eager}
    S = {e0.record_key(r)[1:]: r for r in sdpa}
    keys = sorted(set(E) & set(S))
    out = {"n_common": len(keys), "n_eager_only": len(set(E) - set(S)), "n_sdpa_only": len(set(S) - set(E))}
    for task in ("box", "mcq", "binary"):
        ks = [k for k in keys if E[k]["task"] == task]
        if not ks:
            continue
        same_text = [E[k].get("text_raw") == S[k].get("text_raw") for k in ks]
        out[f"{task}_identical_text"] = float(np.mean(same_text))
        if task == "box":
            out["box_identical_numbers"] = float(np.mean([E[k].get("parsed_numbers") == S[k].get("parsed_numbers") for k in ks]))
        else:
            m = E[ks[0]]["answer_modes"][task]
            out[f"{task}_same_answer"] = float(np.mean([e0.answer(E[k], m) == e0.answer(S[k], m) for k in ks]))
    return out
