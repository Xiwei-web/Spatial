"""框的 teacher-forced 读出：同样 40 条框提示，比较 eager 与 sdpa（用户 2026-11-25 要求；只比较实现之间的差异，不算正确率）。
用法：python scripts/teacher_forced_readout.py InternVL3-8B <bf16|fp32>
真值框：What's Up 标注 auto_v3 的检测框 box_det_xyxy（自动标注，经模型辅助预审和部分人工复核；默认真值框尚未由用户选定），按 0-1000 取整。
输出：<outputs>/diagnostics/teacher_forced_<模型>_<精度>.json；汇总用 scripts/compare_teacher_forced.py"""
import json
import subprocess
import sys
import time

import torch
from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import load_adapter
from src.models.attn_switch import AttnSwitch
from src.readout import InternVLBoxReadout

name, prec = sys.argv[1:3]
ad = load_adapter(name, {"bf16": torch.bfloat16, "fp32": torch.float32}[prec])
sw, ro = AttnSwitch(ad), InternVLBoxReadout(ad)
bc = bbox_convention(name)
assert bc["convention"] == "norm1000_xyxy"
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
ann = {(d["dataset"], d["sample_id"]): d for d in map(json.loads, open(paths.get("data") / "whatsup_annotations" / "auto_v3" / "annotations.jsonl"))}
res = {"model": name, "precision": prec, "truth": "auto_v3 box_det_xyxy", "eager_patch": ad.eager_fp32,
       "param_dtypes": sorted({str(p.dtype) for p in ad.model.parameters()}), "runs": {}, "lm_attention_calls": {}, "seconds": {}}
for impl in ("eager", "sdpa"):
    sw.use(impl)
    t0, out = time.time(), []
    for it in items:
        key = (it["dataset"], it["sample_id"])
        s, a = by_id[key], ann[key]
        assert a["usable"]
        for role in ("target", "reference"):
            o = a["objects"][role]
            assert o["name"] == getattr(s, role)
            x1, y1, x2, y2 = o["box_det_xyxy"]
            W, H = a["width"], a["height"]
            box = [min(1000, max(0, round(v))) for v in (x1 / W * 1000, y1 / H * 1000, x2 / W * 1000, y2 / H * 1000)]
            prep = ad.prepare(Image.open(s.image_path), bc["prompt_template"].format(object=o["name"]))
            out.append({"dataset": key[0], "sample_id": key[1], "role": role, "object": o["name"], "truth1000": box, **ro.run(prep, o["name"], box)})
    res["runs"][impl], res["lm_attention_calls"][impl], res["seconds"][impl] = out, dict(sw.calls), round(time.time() - t0, 1)
    print(impl, "done", res["seconds"][impl], "s; calls", sw.calls, flush=True)
c = res["lm_attention_calls"]
res["switch_effective"] = c["eager"]["eager"] > 0 == c["eager"]["sdpa"] and c["sdpa"]["sdpa"] > 0 == c["sdpa"]["eager"]
res["peak_mem_gb_per_gpu"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
res["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
json.dump(res, open(paths.get("outputs") / "diagnostics" / f"teacher_forced_{name}_{prec}.json", "w"), ensure_ascii=False)
print("SUMMARY", {k: v for k, v in res.items() if k != "runs"})
