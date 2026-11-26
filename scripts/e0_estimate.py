"""E0 的 GPU 用量估算（CPU，只读：划分文件里的条数、bbox_calib 与冒烟记录里每条的实测秒数；不跑模型，不读 test 的任何模型输出）。
用法：python scripts/e0_estimate.py → manifests/e0_estimate.json，并打印 markdown 表
秒数来源：
  框（What's Up）  = bbox_calib 里该模型选定提示词在 What's Up 子集的平均秒数（1280×960 原图）
  框（其余数据集）  = 同上，COCO 子集的平均秒数（COCO 640 宽；GQA 的 VG 图尺寸相近，借用这个数）
  关系选择题 / 二元 = 冒烟 relation_mcq 的平均秒数（What's Up dev 20 张，max_new_tokens 16）；其余数据集乘以该模型框的 COCO/What's Up 耗时比（≤ 1 时才乘）
  每个模型每个作业另加 0.05 小时加载。默认实现那一遍按与主跑相同的耗时计（上界）。"""
import json
import statistics as st
from collections import Counter

import yaml

from src import paths
from src.data import load_split
from src.data.coco_pairs import load_coco_pairs
from src.data.whatsup_annotations import load_annotations
from src.prompts import n_choices

MODELS = {1: ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "InternVL3-8B", "InternVL3_5-8B", "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B", "GLM-4.6V-Flash", "VST-7B-RL", "RoboBrain2.0-7B"],
          2: ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "InternVL3-2B", "SenseNova-SI-1.1-InternVL3-2B", "LLaVA-1.5-7B", "LLaVA-1.5-13B"]}
OPTS = {"whatsup_a": ["left", "right", "on", "under"], "whatsup_b": ["left", "right", "front", "behind"], "coco_spatial_one": ["left", "right", "top", "bottom"],
        "coco_spatial_two": ["left", "right", "above", "below"], "gqa_spatial_one": ["left", "right", "top", "bottom"], "gqa_spatial_two": ["left", "right", "front", "behind"],
        "coco_pairs": ["left", "right", "above", "below"]}
LOAD_H = 0.05


def counts(split):
    ann = {(a["dataset"], a["sample_id"]) for a in load_annotations()}
    out = {}
    for n in ("whatsup_a", "whatsup_b", "coco_spatial_one", "coco_spatial_two", "gqa_spatial_one", "gqa_spatial_two"):
        S = load_split(n, split)
        rel = Counter(s.relation for s in S)
        keep = [s for s in S if s.relation in OPTS[n]]
        if n.startswith("whatsup"):
            box = {(s.sample_id, r) for s in S if (n, s.sample_id) in ann for r in ("t", "r")}
        else:
            box = {(s.image_path, s.target) for s in S if s.bbox_target is not None} | {(s.image_path, s.reference) for s in S if s.reference and s.bbox_reference is not None}
        out[n] = {"samples": len(S), "images": len({s.image_path for s in S}), "relations": dict(rel), "excluded_relation_not_in_options": len(S) - len(keep),
                  "box": len(box), "mcq": n_choices(n) * len(keep), "binary": 2 * len(keep)}
    P = load_coco_pairs(split)
    objs = {(p.extra["image_id"], p.extra["target_ann_id"]) for p in P} | {(p.extra["image_id"], p.extra["reference_ann_id"]) for p in P}
    out["coco_pairs"] = {"samples": len(P), "images": len({p.extra["image_id"] for p in P}), "relations": dict(Counter(p.relation for p in P)), "excluded_relation_not_in_options": 0,
                         "box": len(objs), "mcq": n_choices("coco_pairs") * len(P), "binary": 2 * len(P)}
    return out


def speeds():
    cal = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "bbox_conventions.yaml"))["calibrated"]
    O = paths.get("outputs")
    sp = {}
    for m in MODELS[1] + MODELS[2]:
        last = {}
        for l in open(O / "bbox_calib" / m / "records.jsonl"):
            r = json.loads(l)
            last[(r["prompt_id"], r["item_id"])] = r
        pid = cal[m]["prompt_id"]
        b = {s: st.mean(r["seconds"] for k, r in last.items() if k[0] == pid and r["subset"] == s) for s in ("whatsup", "coco")}
        sm = [json.loads(l) for l in open(O / "smoke" / m / "records.jsonl")]
        rel = st.mean(r["seconds"] for r in sm if r["task"] == "relation_mcq" and r.get("seconds"))
        ratio = min(1.0, b["coco"] / b["whatsup"])
        sp[m] = {"box_whatsup": b["whatsup"], "box_other": b["coco"], "rel_whatsup": rel, "rel_other": rel * ratio, "n_box_calib": sum(1 for k in last if k[0] == pid), "n_rel_smoke": 20}
    return sp


def hours(C, sp, datasets, tasks=("box", "mcq", "binary")):
    per = {}
    for m, s in sp.items():
        tot = 0.0
        for d in datasets:
            wu = d.startswith("whatsup")
            for t in tasks:
                sec = s["box_whatsup" if wu else "box_other"] if t == "box" else s["rel_whatsup" if wu else "rel_other"]
                tot += C[d][t] * sec
        per[m] = tot / 3600 + LOAD_H
    return per


if __name__ == "__main__":
    C, Cd, sp = counts("test"), counts("dev"), speeds()
    ALL = list(C)
    MAIN = ["whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two"]
    plans = {"full_main": hours(C, sp, ALL), "full_default_impl": hours(C, sp, ALL), "lean_main": hours(C, sp, MAIN), "lean_default_impl_whatsup_only": hours(C, sp, ["whatsup_a", "whatsup_b"]),
             "screen_dev_whatsup": hours(Cd, sp, ["whatsup_a", "whatsup_b"])}
    # dev 上的提示词 / 解析检查：每个数据集 dev 取 50 个样本，1 次选择题（固定一种顺序）+ 2 次二元，只看解析率
    chk = {d: {"box": 0, "mcq": 50, "binary": 100} for d in ALL}
    plans["dev_parse_check"] = hours(chk, sp, ALL)
    # 合成 / 深度：每 1,000 张图（两个物体：框 2 + 选择题按四选一 4 次计（上界）+ 二元 2），按 What's Up 的秒数（上界）；深度每 100 条
    syn = {"x": {"box": 2000, "mcq": 4000, "binary": 2000}}
    plans["synthetic_per_1000_images"] = {m: v - LOAD_H for m, v in hours(syn, {m: {**s, "box_other": s["box_whatsup"], "rel_other": s["rel_whatsup"]} for m, s in sp.items()}, ["x"]).items()}
    dep = {"x": {"box": 200, "mcq": 400, "binary": 200}}
    plans["depth_per_100_items"] = {m: v - LOAD_H for m, v in hours(dep, {m: {**s, "box_other": s["box_whatsup"], "rel_other": s["rel_whatsup"]} for m, s in sp.items()}, ["x"]).items()}
    tot = {k: sum(v.values()) for k, v in plans.items()}
    json.dump({"counts_test": C, "counts_dev": Cd, "speeds_seconds": sp, "plans_gpu_hours_per_model": plans, "plans_gpu_hours_total": tot, "load_hours_per_model_job": LOAD_H}, open(paths.REPO_ROOT / "manifests" / "e0_estimate.json", "w"), indent=1, ensure_ascii=False)
    print("| 数据集（test） | 样本 | 图 | 关系 | 问法 | 不在选项里而剔除 | 框提问（去重，有真值） | 选择题（× 轮换次数） | 二元（真、反） |\n|" + "---|" * 9)
    for d, c in C.items():
        print(f"| {d} | {c['samples']} | {c['images']} | {'/'.join(OPTS[d])} | {'四选一 ×4' if n_choices(d) == 4 else '二选一（同轴）×2'} | {c['excluded_relation_not_in_options']} | {c['box']} | {c['mcq']} | {c['binary']} |")
    s = {t: sum(c[t] for c in C.values()) for t in ("box", "mcq", "binary")}
    print(f"| 合计 | {sum(c['samples'] for c in C.values())} | | | | {s['box']} | {s['mcq']} | {s['binary']} |\n")
    print("| 模型 | 框秒数 WU / 其他 | 关系秒数 WU / 其他 | 全量主跑 | 默认实现一遍（全量） | 省钱主跑 | 省钱默认实现（只 WU） | dev 筛选（WU） | dev 解析检查 | 合成 / 1000 图 | 深度 / 100 条 |\n|" + "---|" * 11)
    for m in sp:
        x = sp[m]
        print(f"| {m} | {x['box_whatsup']:.2f} / {x['box_other']:.2f} | {x['rel_whatsup']:.2f} / {x['rel_other']:.2f} | " + " | ".join(f"{plans[k][m]:.2f}" for k in ("full_main", "full_default_impl", "lean_main", "lean_default_impl_whatsup_only", "screen_dev_whatsup", "dev_parse_check", "synthetic_per_1000_images", "depth_per_100_items")) + " |")
    print("| **合计（GPU 小时）** | | | " + " | ".join(f"**{tot[k]:.1f}**" for k in ("full_main", "full_default_impl", "lean_main", "lean_default_impl_whatsup_only", "screen_dev_whatsup", "dev_parse_check", "synthetic_per_1000_images", "depth_per_100_items")) + " |")
    for g in (1, 2):
        print(f"第 {g} 组全量主跑合计 {sum(plans['full_main'][m] for m in MODELS[g]):.1f} 小时")
    print("per-dataset share (full main, all models):", {d: round(sum(hours({d: C[d]}, sp, [d])[m] - LOAD_H for m in sp), 1) for d in ALL})
