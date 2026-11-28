"""真实数据集功效模拟的输入（planner 第 11 轮第 6 条；CPU，只读 dev）。每个模型 × 数据集只取“边际量”，不算框推与直接回答的一致率或 OR：
  - 几何：dev 条目的真框中心有向差 Δ（token，标注关系方向为正）与轴；按 |Δ| 的三分位数定 3 个距离档（MH OR 的分层）；
  - 框误差池：两个物体框中心沿关系轴的有向误差（token），按轴分池；框推正确率（边际）；
  - 直接回答正确率（边际）。
数据来源（都是 dev）：
  What's Up A / B：E0 dev 入选筛选（outputs/e0/<模型>/screen.jsonl），框 + 二选一（四选一的前两个轮换，与 src/e1_analysis.build_items 相同）。B 只用 left/right。
  COCO-spatial two：框误差 = 框坐标标定的 dev 记录（outputs/bbox_calib/<模型>/records.jsonl，该模型标定选中的提示词）；直接回答 = E0 dev 解析检查的 50 条二选一。
  GQA-spatial two：dev 上没有问过框。框误差池借用同一模型的 COCO-spatial two 标定误差（代用，报告里写明）；几何 = 解析检查 50 个样本的真值框；直接回答 = 解析检查的 50 条二选一。只用 left/right。
test 样本量 = E0 test 里关系能由框推出的条目数（只数样本，不读回答）。
用法：python scripts/e1_real_inputs.py → <outputs>/e1_power_v3/real_inputs.json 与 manifests/e1_real_inputs_summary.json"""
import json
from collections import defaultdict

import numpy as np

from src import e0, e1, paths
from src.data import load_split
from src.e0_metrics import TOKEN_PX
from src.e1 import REL_AXIS
from src.e1_analysis import build_items
from src.prompts import to_original_xyxy

MODELS = e1.synth_cfg()["single_object_check"]["models"]
DS = ["whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two"]
OUT = paths.get("outputs") / "e1_power_v3"


def tok_scale(model, orig_wh, model_wh, vb=None):
    W, H = orig_wh
    vw, vh = (vb[2] - vb[0], vb[3] - vb[1]) if vb else (W, H)
    tp = TOKEN_PX[model]
    return tp * vw / model_wh[0], tp * vh / model_wh[1]


def signed_delta(rel, bt, br, sc):
    ax, sg = REL_AXIS[rel]
    k = 0 if ax == "x" else 1
    ct, cr = (bt[k] + bt[k + 2]) / 2, (br[k] + br[k + 2]) / 2
    return ax, sg * (ct - cr) / sc[k]


def direct_acc_parse_check(model, ds):
    modes = e0.answer_modes(model)
    R = [json.loads(l) for l in open(paths.get("outputs") / "e0" / model / "parse_check.jsonl")]
    R = [r for r in R if r["dataset"] == ds and r["task"] == "mcq" and r["relation"] in REL_AXIS]
    ok = [e0.correct(r, e0.answer(r, modes["mcq"])) for r in R]
    return R, float(np.mean(ok)), len(ok)


def pair_acc_parse_check(model, ds):
    """成对二元（同一样本问关系与其反关系，回答须为 是/否 或 否/是 且指向标注关系）的 dev 边际正确率。"""
    modes = e0.answer_modes(model)
    R = [json.loads(l) for l in open(paths.get("outputs") / "e0" / model / "parse_check.jsonl")]
    by = defaultdict(dict)
    for r in R:
        if r["dataset"] == ds and r["task"] == "binary" and r["relation"] in REL_AXIS:
            by[r["sample_id"]][r["asked_relation"]] = e0.answer(r, modes["binary"])
    ok = []
    for sid, a in by.items():
        rel = next(r for r in R if r["dataset"] == ds and r["sample_id"] == sid)["relation"]
        opp = [k for k in a if k != rel]
        if rel in a and opp:
            ok.append(a[rel] == "yes" and a[opp[0]] == "no")
    return float(np.mean(ok)), len(ok)


def whatsup(model, ds):
    R = [json.loads(l) for l in open(paths.get("outputs") / "e0" / model / "screen.jsonl")]
    R = [r for r in R if r["dataset"] == ds and r["relation"] in REL_AXIS]
    it, skipped = build_items(R, {})
    for k in ("mcq_agree", "pair_agree"):          # build_items 顺带算的一致率字段，立即丢弃、不读（planner：dev 上只取边际量）
        it.pop(k)
    axis = []
    groups = defaultdict(list)
    for r in R:
        groups[(r["dataset"], r["sample_id"])].append(r)
    for k in sorted(groups):
        rr = groups[k]
        if {x["role"] for x in rr if x["task"] == "box"} == {"target", "reference"}:
            axis.append(REL_AXIS[rr[0]["relation"]][0])
    assert len(axis) == len(it["d_box"])
    e_t, e_r = it["e_t"], it["e_r"]
    pools = {a: [float(v) for v in np.concatenate([e_t[np.array(axis) == a], e_r[np.array(axis) == a]]) if np.isfinite(v)] for a in ("x", "y")}
    D = it["mcq_ok"]
    po = it["pair_ok"]
    return {"p_pair_dev": float(np.nanmean(po)), "n_pair_dev": int(np.isfinite(po).sum()), "delta": [float(v) for v in it["d_box"]], "axis": axis, "pools": pools, "p_box_dev": float(np.mean(it["box_ok"])), "n_box_items_dev": int(len(it["box_ok"])),
            "box_parse_fail_dev": float(1 - np.mean(it["box_parsed"])), "p_dir_dev": float(np.nanmean(D)), "n_dir_obs_dev": int(np.isfinite(D).sum()),
            "source": "E0 dev screen.jsonl（框 + 四选一前两个轮换）", "skipped": skipped}


def coco_calib(model):
    cal = e0.calibrated(model)
    R = [json.loads(l) for l in open(paths.get("outputs") / "bbox_calib" / model / "records.jsonl")]
    R = [r for r in R if r["dataset"] == "coco_spatial_two" and r["prompt_id"] == cal["prompt_id"] and not r.get("error")]
    by = defaultdict(dict)
    for r in R:
        by[r["sample_id"]][r["role"]] = r
    S = {s.sample_id: s for s in load_split("coco_spatial_two", "dev")}
    delta, axis, pools, box_ok, nf = [], [], {"x": [], "y": []}, [], 0
    for sid, rr in sorted(by.items()):
        if set(rr) != {"target", "reference"} or S[sid].relation not in REL_AXIS:
            continue
        rel = S[sid].relation
        t, r_ = rr["target"], rr["reference"]
        sc = tok_scale(model, t["orig_wh"], t["model_wh"], t.get("view_box"))
        ax, d = signed_delta(rel, t["truth_xyxy"], r_["truth_xyxy"], sc)
        _, sg = REL_AXIS[rel]
        k = 0 if ax == "x" else 1
        es = []
        for o in (t, r_):
            if not o.get("parsed"):
                es.append(None); continue
            b = to_original_xyxy(o["parsed"], cal["convention"], tuple(o["orig_wh"]), tuple(o["model_wh"]) if o.get("model_wh") else None, tuple(o["view_box"]) if o.get("view_box") else None)
            tb = o["truth_xyxy"]
            es.append(sg * ((b[k] + b[k + 2]) / 2 - (tb[k] + tb[k + 2]) / 2) / sc[k])
        delta.append(d); axis.append(ax)
        if None in es:
            nf += 1; box_ok.append(False); continue
        pools[ax] += es
        box_ok.append(d + es[0] - es[1] > 0)
    return {"delta": delta, "axis": axis, "pools": pools, "p_box_dev": float(np.mean(box_ok)), "n_box_items_dev": len(box_ok), "box_parse_fail_dev": nf / max(len(box_ok), 1),
            "source_box": f"bbox_calib dev（提示词 {cal['prompt_id']}）"}


def parse_check_geometry(model, ds, R):
    S = {s.sample_id: s for s in load_split(ds, "dev")}
    delta, axis = [], []
    for r in R:
        s = S[r["sample_id"]]
        if s.bbox_target is None or s.bbox_reference is None:          # GQA 有的物体框不唯一 / 缺失，E0 不对它问框，E1 也没有这个条目
            continue
        sc = tok_scale(model, r["orig_wh"], r["model_wh"], r.get("view_box"))
        ax, d = signed_delta(s.relation, s.bbox_target, s.bbox_reference, sc)
        delta.append(d); axis.append(ax)
    return delta, axis


def test_n(ds):
    if ds.startswith("whatsup"):                     # What's Up 的框来自标注 auto_v3（每个样本都有），不在样本对象上
        return sum(1 for s in e0.test_samples(ds) if s.relation in REL_AXIS)
    return sum(1 for s in e0.test_samples(ds) if s.relation in REL_AXIS and s.bbox_target is not None and s.bbox_reference is not None)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, summ = {}, {}
    tn = {ds: test_n(ds) for ds in DS}
    for m in MODELS:
        res[m] = {}
        res[m]["whatsup_a"] = whatsup(m, "whatsup_a")
        res[m]["whatsup_b"] = whatsup(m, "whatsup_b")
        cc = coco_calib(m)
        R, p, n = direct_acc_parse_check(m, "coco_spatial_two")
        pp, pn = pair_acc_parse_check(m, "coco_spatial_two")
        res[m]["coco_spatial_two"] = {**cc, "p_dir_dev": p, "n_dir_obs_dev": n, "p_pair_dev": pp, "n_pair_dev": pn, "source_dir": "E0 dev parse_check 二选一 / 成对二元"}
        R, p, n = direct_acc_parse_check(m, "gqa_spatial_two")
        dg, ag = parse_check_geometry(m, "gqa_spatial_two", R)
        pp, pn = pair_acc_parse_check(m, "gqa_spatial_two")
        res[m]["gqa_spatial_two"] = {"delta": dg, "axis": ag, "pools": cc["pools"], "p_box_dev": None, "n_box_items_dev": 0, "p_dir_dev": p, "n_dir_obs_dev": n, "p_pair_dev": pp, "n_pair_dev": pn,
                                     "source_box": "借用 COCO-spatial two 的标定框误差（GQA dev 没有框）", "source_dir": "E0 dev parse_check 二选一"}
        for ds in DS:
            d = res[m][ds]
            ad = np.abs(np.array(d["delta"]))
            d["bin_edges_abs_delta"] = [float(v) for v in np.quantile(ad, [1 / 3, 2 / 3])]
            d["n_test"] = tn[ds]
            pe = {a: np.array(v) for a, v in d["pools"].items()}
            summ.setdefault(m, {})[ds] = {"n_test": tn[ds], "n_geom_dev": len(ad), "p_box_dev": d["p_box_dev"], "n_box_items_dev": d["n_box_items_dev"], "p_dir_dev": d["p_dir_dev"],
                                          "n_dir_obs_dev": d["n_dir_obs_dev"], "p_pair_dev": d["p_pair_dev"], "n_pair_dev": d["n_pair_dev"], "frac_delta_nonpositive": float(np.mean(np.array(d["delta"]) <= 0)),
                                          "abs_delta_median_tok": float(np.median(ad)), "bin_edges_abs_delta": d["bin_edges_abs_delta"],
                                          "pool_sd": {a: float(v.std()) if len(v) else None for a, v in pe.items()},
                                          "pool_mad": {a: float(np.median(np.abs(v - np.median(v)))) if len(v) else None for a, v in pe.items()},
                                          "pool_n": {a: len(v) for a, v in pe.items()}}
        print(m, {ds: (round(summ[m][ds]["p_box_dev"], 3) if summ[m][ds]["p_box_dev"] is not None else None, round(summ[m][ds]["p_dir_dev"], 3), round(summ[m][ds]["p_pair_dev"], 3)) for ds in DS}, flush=True)
    json.dump(res, open(OUT / "real_inputs.json", "w"))
    json.dump({"note": "只含边际量；不含一致率、OR 或任何 2×2", "test_n": tn, "models": summ}, open(paths.REPO_ROOT / "manifests" / "e1_real_inputs_summary.json", "w"), indent=1, ensure_ascii=False)
