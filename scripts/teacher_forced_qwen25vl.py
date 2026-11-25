"""Qwen2.5-VL：40 条框提示的 teacher-forced 读出（eager / sdpa），以及（仅 fp32）40 条框 + 20 条关系的自由生成。只比较实现与精度之间的差异，不算正确率。
用法：python scripts/teacher_forced_qwen25vl.py Qwen2.5-VL-7B-Instruct <fp32|bf16> --chunk 10 [--generate]
真值：What's Up auto_v3 的 box_gt_xyxy（原始 mask 外接框；自动标注，经模型辅助预审和部分人工复核），换算到模型输入像素并取整。
门槛（fp32，每种实现跑完就检查，不过就退出码 3，后面的不跑；不放宽）：
  lp_tree 与 lp_digits 最大差 < 1e-3 nat（只取枚举里不带结束概率的坐标，两者是同一个量）；枚举质量 > 0.999；
  拆分 embeds / 位置后的 logits 与官方完整 forward 最大差 < 1e-3（前 2 条）。
OOM 退出码 42。输出 <outputs>/diagnostics/teacher_forced_<模型>_<精度>.json、fp32_generate_<模型>.json"""
import argparse
import json
import subprocess
import sys
import time

import torch
from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.data.whatsup_annotations import load_annotations, truth_box
from src.models.adapter import load_adapter
from src.models.attn_switch import AttnSwitch
from src.prompts import LETTERS, relation_mcq
from src.readout import Qwen25VLBoxReadout, digit_trie

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("prec", choices=["bf16", "fp32"])
ap.add_argument("--chunk", type=int, default=10); ap.add_argument("--generate", action="store_true")
A = ap.parse_args()
if not torch.cuda.is_available():
    sys.exit("没有可用的 GPU")
name, prec, t_all = A.name, A.prec, time.time()
D = paths.get("outputs") / "diagnostics"
ad = load_adapter(name, {"bf16": torch.bfloat16, "fp32": torch.float32}[prec])
assert {p.device.type for p in ad.model.parameters()} == {"cuda"}
sw, ro = AttnSwitch(ad), Qwen25VLBoxReadout(ad, A.chunk)
bc = bbox_convention(name)
assert bc["convention"] == "abs_model_xyxy"
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
ann = {(d["dataset"], d["sample_id"]): d for d in load_annotations("auto_v3")}
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
head = {"model": name, "precision": prec, "chunk": A.chunk, "git_commit": commit, "eager_patch": ad.eager_fp32, "param_dtypes": sorted({str(p.dtype) for p in ad.model.parameters()}),
        "n_gpu": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0), "switch_note": "Qwen2.5-VL 的视觉塔与语言模型一起切换；lm_attention_calls 只数语言模型层，other_attention_calls 为其余（视觉塔）"}
assert head["param_dtypes"] == [{"bf16": "torch.bfloat16", "fp32": "torch.float32"}[prec]], head["param_dtypes"]
peak = lambda: [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]      # noqa: E731
peak_r = lambda: [round(torch.cuda.max_memory_reserved(i) / 2**30, 2) for i in range(torch.cuda.device_count())]     # noqa: E731


def gate(rows, N):
    """返回 (是否通过, 明细)；同一个量：枚举里不带结束概率的坐标（lp_digits_end 为 None）。"""
    d, mass = [], []
    for x in rows:
        for k in ("native", "headfp32"):
            r = x["readout"][k]
            d += [abs(a - b) for a, b, e in zip(r["lp_tree"], r["lp_digits"], r["lp_digits_end"]) if e is None]
            mass += r["mass_tree"]
    off = [x["official_forward_max_abs_logit_diff"] for x in rows if "official_forward_max_abs_logit_diff" in x]
    info = {"n_compared(2 heads)": len(d), "lp_tree_vs_lp_digits_max": max(d), "mass_min": min(mass), "official_forward_max_abs_logit_diff": max(off)}
    return max(d) < 1e-3 and min(mass) > 0.999 and max(off) < 1e-3, info


def switch_ok(impl, c):
    return c[impl] > 0 and c["sdpa" if impl == "eager" else "eager"] == 0


res = {**head, "truth": "auto_v3 box_gt_xyxy（换算到模型输入像素取整）", "runs": {}, "lm_attention_calls": {}, "other_attention_calls": {}, "seconds": {}, "gate": {}, "errors": []}
out_f = D / f"teacher_forced_{name}_{prec}.json"
try:
    for impl in ("eager", "sdpa"):
        sw.use(impl)
        t0, out = time.time(), []
        for it in items:
            key = (it["dataset"], it["sample_id"])
            s, a = by_id[key], ann[key]
            for role in ("target", "reference"):
                o = a["objects"][role]
                assert o["name"] == getattr(s, role)
                prep = ad.prepare(Image.open(s.image_path), bc["prompt_template"].format(object=o["name"]))
                assert prep.orig_wh == (a["width"], a["height"])
                (mw, mh), (W, H) = prep.model_wh, prep.orig_wh
                x1, y1, x2, y2 = truth_box(o)
                box = [min(mw, max(0, round(x1 * mw / W))), min(mh, max(0, round(y1 * mh / H))), min(mw, max(0, round(x2 * mw / W))), min(mh, max(0, round(y2 * mh / H)))]
                out.append({"dataset": key[0], "sample_id": key[1], "image": str(s.image_path.name if hasattr(s.image_path, "name") else s.image_path).split("/")[-1], "role": role, "object": o["name"], "truth": box,
                            **ro.run(prep, o["name"], box, official_check=len(out) < 2)})
                if len(out) == 2:
                    print(impl, "first 2:", {k: out[0][k] for k in ("official_forward_max_abs_logit_diff", "max_value", "model_wh", "grid_hw", "n_input", "truth")}, "peak", peak(), flush=True)
        ok, info = gate(out, 0)
        res["runs"][impl], res["lm_attention_calls"][impl], res["other_attention_calls"][impl], res["seconds"][impl], res["gate"][impl] = out, dict(sw.calls), dict(sw.calls_other), round(time.time() - t0, 1), info
        res["peak_allocated_gb"], res["peak_reserved_gb"] = peak(), peak_r()
        json.dump(res, open(out_f, "w"), ensure_ascii=False)
        print(impl, "done", res["seconds"][impl], "s; lm calls", sw.calls, "other", sw.calls_other, "gate", ok, info, flush=True)
        if not switch_ok(impl, sw.calls):
            sys.exit("注意力实现切换未生效")
        if prec == "fp32" and not ok:
            print("GATE FAILED", info)
            sys.exit(3)
    print("TF SUMMARY", {k: v for k, v in res.items() if k != "runs"}, flush=True)
    if A.generate:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        opt = {L: ad.option_token_ids()[L]["bare"] for L in LETTERS}
        jobs = []                                       # 顺序、提示词、max_new_tokens 与 check_default_vs_eager.py 相同，便于与已存 bf16 结果逐条对照
        for it in items:
            s = by_id[(it["dataset"], it["sample_id"])]
            jobs += [("bbox", s, "target", bc["prompt_template"].format(object=s.target), 96), ("bbox", s, "reference", bc["prompt_template"].format(object=s.reference), 96),
                     ("relation", s, None, relation_mcq(s.dataset, s.target, s.reference)[0], 16)]
        g_res = {**head, "runs": {}, "lm_attention_calls": {}, "other_attention_calls": {}, "seconds": {}}
        for impl in ("eager", "sdpa"):
            sw.use(impl)
            t0, out = time.time(), []
            for task, s, role, prompt, mnt in jobs:
                t1 = time.time()
                g = ad.generate(ad.prepare(Image.open(s.image_path), prompt), mnt)
                out.append({"task": task, "sample_id": s.sample_id, "role": role, "tokens": g["token_ids"], "text": g["text_raw"], "seconds": round(time.time() - t1, 3),
                            "option_logits": [float(g["first_logits"][opt[L]]) for L in LETTERS] if task == "relation" else None})
            g_res["runs"][impl], g_res["lm_attention_calls"][impl], g_res["other_attention_calls"][impl], g_res["seconds"][impl] = out, dict(sw.calls), dict(sw.calls_other), round(time.time() - t0, 1)
            assert switch_ok(impl, sw.calls)
            print("generate", impl, g_res["seconds"][impl], "s", sw.calls, sw.calls_other, flush=True)
        g_res["peak_allocated_gb"], g_res["peak_reserved_gb"] = peak(), peak_r()
        json.dump(g_res, open(D / f"fp32_generate_{name}.json", "w"), indent=1, ensure_ascii=False)
except torch.cuda.OutOfMemoryError as e:
    print("OOM", e, "peak", peak(), flush=True)
    sys.exit(42)
print("ALL", round(time.time() - t_all, 1), "s")
