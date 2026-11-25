"""float32 下 sdpa 与 eager 是否逐 token 一致（用户 2026-11-25 要求；只做 InternVL3-8B，不是评测，不算正确率）。
用法（2 张卡）：python scripts/check_fp32_sdpa_vs_eager.py InternVL3-8B
输入与 scripts/check_default_vs_eager.py 的 40 条框提示完全相同（同样的图、提示词、max_new_tokens=96、贪心）。
切换的是语言模型的注意力实现；InternVL 的视觉塔是远程代码自带的 _naive_attn，两种设置下相同。
每种实现跑两遍：第二遍只用来确认同一设置自身可复现。
另外与已保存的 bf16 结果（default_vs_eager.json）逐 token 对照，不重跑 bf16。
切换是否生效用直接证据判断：统计语言模型层里 eager_attention_forward 与 F.scaled_dot_product_attention 的实际调用次数。
（不用 output_attentions 判断：多卡切分时 InternVL 远程代码的 forward 会因跨设备索引报错，generate 不受影响。）
输出：<outputs>/diagnostics/fp32_sdpa_vs_eager_<模型>[_标签].json（第二个命令行参数为标签）"""
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


# ---- 调用计数 ----
calls = {"eager": 0, "sdpa": 0}
in_lm = {"on": False}
lm_mod = sys.modules[type(ad.lm_layers[0].self_attn).__module__]
_eager, _sdpa = lm_mod.eager_attention_forward, torch.nn.functional.scaled_dot_product_attention


def eager_counted(*a, **k):
    calls["eager"] += 1
    return _eager(*a, **k)


def sdpa_counted(*a, **k):
    calls["sdpa"] += in_lm["on"]
    return _sdpa(*a, **k)


lm_mod.eager_attention_forward = eager_counted
torch.nn.functional.scaled_dot_product_attention = sdpa_counted
for l in ad.lm_layers:       # 只统计语言模型层里的 sdpa（视觉塔不计）
    l.self_attn.register_forward_pre_hook(lambda m, i: in_lm.__setitem__("on", True))
    l.self_attn.register_forward_hook(lambda m, i, o: in_lm.__setitem__("on", False))

items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
jobs = [(s, role, bc["prompt_template"].format(object=getattr(s, role))) for s in samples for role in ("target", "reference")]
assert len(jobs) == 40
runs, secs, ncalls, logits = {}, {}, {}, {}
for impl in ("eager", "sdpa"):
    use(impl)
    for rep in (1, 2):
        k = f"{impl}_{rep}"
        calls.update(eager=0, sdpa=0)
        t0, out, lg = time.time(), [], []
        for s, role, prompt in jobs:
            g = ad.generate(ad.prepare(Image.open(s.image_path), prompt), 96)
            lg.append(g["first_logits"])
            out.append({"dataset": s.dataset, "sample_id": s.sample_id, "role": role, "tokens": g["token_ids"], "text": g["text_raw"]})
        runs[k], secs[k], ncalls[k], logits[k] = out, round(time.time() - t0, 1), dict(calls), torch.stack(lg)
        print(k, "done", secs[k], "s; calls:", ncalls[k], flush=True)


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
       "lm_attention_calls": ncalls,
       "switch_effective": all(ncalls[f"eager_{r}"]["eager"] > 0 and ncalls[f"eager_{r}"]["sdpa"] == 0 and ncalls[f"sdpa_{r}"]["sdpa"] > 0
                               and ncalls[f"sdpa_{r}"]["eager"] == 0 for r in (1, 2)),
       "fp32_first_logits_max_abs_diff_eager_vs_sdpa": float((logits["eager_1"] - logits["sdpa_1"]).abs().max()),
       "fp32_first_logits_bit_identical_eager_vs_sdpa": bool(torch.equal(logits["eager_1"], logits["sdpa_1"])),
       "fp32_eager_vs_sdpa_identical": ident(e, d), "fp32_eager_vs_sdpa_diffs": diffs(e, d),
       "fp32_eager_repeat_identical": ident(e, runs["eager_2"]), "fp32_sdpa_repeat_identical": ident(d, runs["sdpa_2"]),
       "bf16_eager_vs_bf16_sdpa_identical(saved)": ident(bf["observed"], bf["reference"]),
       "fp32_eager_vs_bf16_eager_identical": ident(e, bf["observed"]), "fp32_eager_vs_bf16_eager_diffs": diffs(e, bf["observed"]),
       "fp32_sdpa_vs_bf16_sdpa_identical": ident(d, bf["reference"]), "fp32_sdpa_vs_bf16_sdpa_diffs": diffs(d, bf["reference"]),
       "seconds_per_40": secs, "peak_mem_gb_per_gpu": [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())],
       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip(),
       "runs": runs}
out = paths.get("outputs") / "diagnostics" / ("fp32_sdpa_vs_eager_" + "_".join(sys.argv[1:3]) + ".json")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print("SUMMARY", name, {k: (len(v) if k.endswith("_diffs") else v) for k, v in res.items() if k != "runs"})
