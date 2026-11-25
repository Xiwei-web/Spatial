"""float32 下 sdpa 与 eager 是否逐 token 一致（用户 2026-11-25 要求；只做 InternVL3-8B，不是评测，不算正确率）。
用法（2 张卡）：python scripts/check_fp32_sdpa_vs_eager.py InternVL3-8B
输入与 scripts/check_default_vs_eager.py 的 40 条框提示完全相同（同样的图、提示词、max_new_tokens=96、贪心）。
切换的是语言模型的注意力实现；InternVL 的视觉塔是远程代码自带的 _naive_attn，两种设置下相同。
每种实现跑两遍：第二遍只用来确认同一设置自身可复现。
另外与已保存的 bf16 结果（default_vs_eager.json）逐 token 对照，不重跑 bf16。
输出：<outputs>/diagnostics/fp32_sdpa_vs_eager_<模型>.json"""
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
from src.prompts import parse_four_numbers

name = sys.argv[1]
ad = load_adapter(name, torch.float32)
dtypes = {}
for p in ad.model.parameters():
    dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
assert set(dtypes) == {"torch.float32"}, dtypes
cfgs = {}
for m in ad.model.modules():
    c = getattr(m, "config", None)
    if c is not None and hasattr(c, "_attn_implementation"):
        cfgs[id(c)] = c
orig = {k: c._attn_implementation for k, c in cfgs.items()}


def use(impl):
    for k, c in cfgs.items():
        c._attn_implementation = orig[k] if impl == "eager" else "sdpa"


items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
jobs = [(s, role, bc["prompt_template"].format(object=getattr(s, role))) for s in samples for role in ("target", "reference")]
assert len(jobs) == 40
runs, probe, secs = {}, {}, {}
for impl in ("eager", "sdpa"):
    use(impl)
    try:
        a = ad.forward(ad.prepare(Image.open(jobs[0][0].image_path), jobs[0][2]), output_attentions=True).attentions
        probe[impl] = "none" if a is None or all(x is None for x in a) else "returned"
        del a
    except Exception as e:
        probe[impl] = f"error: {type(e).__name__}: {e}"[:200]
    for rep in (1, 2):
        t0, out = time.time(), []
        for s, role, prompt in jobs:
            g = ad.generate(ad.prepare(Image.open(s.image_path), prompt), 96)
            out.append({"dataset": s.dataset, "sample_id": s.sample_id, "role": role, "tokens": g["token_ids"], "text": g["text_raw"]})
        runs[f"{impl}_{rep}"], secs[f"{impl}_{rep}"] = out, round(time.time() - t0, 1)
        print(impl, rep, "done", secs[f"{impl}_{rep}"], "s; attentions:", probe[impl], flush=True)


def ident(x, y):
    return sum(a["tokens"] == b["tokens"] for a, b in zip(x, y))


def diffs(x, y):
    out = []
    for a, b in zip(x, y):
        if a["tokens"] == b["tokens"]:
            continue
        pa, pb = parse_four_numbers(a["text"]), parse_four_numbers(b["text"])
        out.append({"sample_id": a["sample_id"], "role": a.get("role"), "x": a["text"], "y": b["text"],
                    "max_coord_diff": max(abs(i - j) for i, j in zip(pa, pb)) if pa and pb else None})
    return out


e, d = runs["eager_1"], runs["sdpa_1"]
old = json.load(open(paths.get("outputs") / "smoke" / name / "default_vs_eager.json"))["runs"]
bf = {k: [r for r in v if r["task"] == "bbox"] for k, v in old.items()}     # observed=eager+fp32 打分，reference=sdpa
for v in bf.values():
    assert [r["sample_id"] for r in v] == [r["sample_id"] for r in e]
res = {"model": name, "dtype": "float32", "param_dtypes": dtypes, "n": 40,
       "attentions_available": probe, "switch_effective": probe["eager"] == "returned" and probe["sdpa"] != "returned",
       "fp32_eager_vs_sdpa_identical": ident(e, d), "fp32_eager_vs_sdpa_diffs": diffs(e, d),
       "fp32_eager_repeat_identical": ident(e, runs["eager_2"]), "fp32_sdpa_repeat_identical": ident(d, runs["sdpa_2"]),
       "bf16_eager_vs_bf16_sdpa_identical(saved)": ident(bf["observed"], bf["reference"]),
       "fp32_eager_vs_bf16_eager_identical": ident(e, bf["observed"]), "fp32_eager_vs_bf16_eager_diffs": diffs(e, bf["observed"]),
       "fp32_sdpa_vs_bf16_sdpa_identical": ident(d, bf["reference"]), "fp32_sdpa_vs_bf16_sdpa_diffs": diffs(d, bf["reference"]),
       "seconds_per_40": secs, "peak_mem_gb_per_gpu": [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())],
       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip(),
       "runs": runs}
out = paths.get("outputs") / "diagnostics" / f"fp32_sdpa_vs_eager_{name}.json"
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print("SUMMARY", name, {k: (len(v) if k.endswith("_diffs") else v) for k, v in res.items() if k != "runs"})
