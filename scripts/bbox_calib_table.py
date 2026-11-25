"""由 outputs/bbox_calib/<模型>/records.jsonl 生成标定表（CPU，只读；表只来自 JSONL，不碰 test）。
用法：python scripts/bbox_calib_table.py <模型1> [<模型2> …]   输出 manifests/bbox_calib_<模型>.json（每个模型）并打印 markdown"""
import json
import sys

import numpy as np

from scripts.bbox_calib_run import prompt_set
from scripts.smoke_test import bbox_convention
from src import paths
from src.bbox_calib import select

items = {i["item_id"]: i for i in json.load(open(paths.REPO_ROOT / "splits" / "bbox_calib_v1.json"))["items"]}


def load(name):
    last = {}
    for l in open(paths.get("outputs") / "bbox_calib" / name / "records.jsonl"):
        r = json.loads(l)
        last[(r["prompt_id"], r["item_id"])] = r
    prompts = list(prompt_set(name))
    recs = []
    for p in prompts:
        for iid, i in items.items():
            r = last.get((p, iid))
            assert r is not None, f"{name} {p} {iid} 缺失，还没跑完"
            assert r["truth_xyxy"] == i["truth_xyxy"] and r["subset"] == i["subset"] and r["image_key"] == i["image_key"]
            # parsed=None 或 error：model_wh 可能为 None，用占位值，不影响（换算在 parsed=None 时返回 None）
            recs.append({"item_id": iid, "prompt_id": p, "parsed": r["parsed"], "orig_wh": i["orig_wh"], "model_wh": r["model_wh"] or [1, 1], "view_box": r["view_box"], "truth": i["truth_xyxy"],
                         "subset": i["subset"], "image_key": i["image_key"], "error": r.get("error"), "text_raw": r.get("text_raw") or ""})
    return recs, prompts, last


def main(names):
    rows = []
    for name in names:
        recs, prompts, last = load(name)
        s = select(recs, prompts)
        cur = bbox_convention(name)
        s["model"], s["current_config_convention"], s["prompts"] = name, cur["convention"], prompt_set(name)
        s["revision"] = next(iter(last.values()))["revision"]
        s["n_3d_output"] = {p: int(sum("bbox_3d" in r["text_raw"] for r in recs if r["prompt_id"] == p)) for p in prompts}
        s["n_hit_max_tokens"] = {p: sum(bool(r.get("hit_max_tokens")) for k, r in last.items() if k[0] == p) for p in prompts}
        s["n_items_per_prompt"] = {p: len(items) for p in prompts}
        s["current_matches_first"] = None
        lab = {"norm1000_xyxy": "norm1000_xyxy", "abs_model_xyxy": "model_xyxy", "abs_orig_xyxy": "orig_xyxy", "norm1_xyxy": "norm1_xyxy", "norm1000_yxyx": "norm1000_yxyx"}.get(cur["convention"])
        if lab:
            s["current_matches_first"] = s["first"][1] == lab
        json.dump(s, open(paths.REPO_ROOT / "manifests" / f"bbox_calib_{name}.json", "w"), indent=1, ensure_ascii=False)
        rows.append(s)
    print("| 模型 | 首选（提示词, 约定） | 平均 IoU | IoU@0.5 | 第二名（平均 IoU） | 差值 [95% 区间] | 解析失败率 | 越界比例（首选） | What's Up 平均 IoU / IoU@0.5 | COCO 平均 IoU / IoU@0.5 | 未定 |\n|" + "---|" * 11)
    for s in rows:
        w, c = s["first_subset"]["whatsup"], s["first_subset"]["coco"]
        pf = s["parse_fail"][s["first"][0]]
        print(f"| {s['model']} | {s['first'][0]}, {s['first'][1]} | {s['first_mean_iou']:.3f} | {s['first_iou_at_0.5']:.3f} | {s['second'][0]}, {s['second'][1]}（{s['second_mean_iou']:.3f}） | {s['diff_mean']:+.3f} [{s['diff_ci95'][0]:+.3f}, {s['diff_ci95'][1]:+.3f}] | "
              f"{pf:.3f} | {('%.3f' % s['first_out_of_bounds']) if s['first_out_of_bounds'] is not None else '—'} | {w['mean_iou']:.3f} / {w['iou_at_0.5']:.3f} | {c['mean_iou']:.3f} / {c['iou_at_0.5']:.3f} | {'；'.join(s['undecided_reasons']) if s['undecided'] else '否'} |")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
