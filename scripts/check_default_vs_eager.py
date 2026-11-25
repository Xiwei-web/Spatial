"""“带观测的模型”与“大家实际使用的模型”是否是同一个模型（用户 2026-11-25 要求的检查；不是评测，不算正确率）。
用法：python scripts/check_default_vs_eager.py <模型名>
同一份权重、同一批输入，比较两种注意力实现下的贪心生成：
  observed  = 本仓库做机制分析用的设置：eager 注意力 + float32 打分补丁（能取注意力权重）
  reference = 不取权重时的常规实现 sdpa（transformers 原生模型的默认实现；InternVL 远程代码的默认是
              flash-attn（装了的话）否则 eager，这里统一用 sdpa 作参照并如实注明）
输入：冒烟的 40 条框提示 + 20 条关系四选一提示。
报告：逐 token 完全一致的条数（框 /40、关系 /20）；关系提示上四个选项 token 的 logit 最大绝对差、
      以及四选项 argmax 是否一致；并报告 reference 设置下确实取不到注意力权重（证明切换生效）。
输出：<outputs>/smoke/<模型>/default_vs_eager.json"""
import json
import sys

from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import load_adapter
from src.prompts import LETTERS, relation_mcq

name = sys.argv[1]
ad = load_adapter(name)                                   # observed：eager + fp32 打分
cfgs = {}
for n, m in ad.model.named_modules():
    c = getattr(m, "config", None)
    if c is not None and hasattr(c, "_attn_implementation"):
        cfgs[id(c)] = c
for c in (getattr(ad.model, "config", None), getattr(getattr(ad.model, "language_model", None), "config", None)):
    if c is not None and hasattr(c, "_attn_implementation"):
        cfgs[id(c)] = c
orig = {k: c._attn_implementation for k, c in cfgs.items()}


def use(impl):
    for k, c in cfgs.items():
        c._attn_implementation = orig[k] if impl == "observed" else "sdpa"


items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
jobs = []
for s in samples:
    jobs += [("bbox", s, bc["prompt_template"].format(object=s.target), 96), ("bbox", s, bc["prompt_template"].format(object=s.reference), 96),
             ("relation", s, relation_mcq(s.dataset, s.target, s.reference)[0], 16)]
opt = {L: ad.option_token_ids()[L]["bare"] for L in LETTERS}
runs, attn_probe = {}, {}
for impl in ("observed", "reference"):
    use(impl)
    prep = ad.prepare(Image.open(jobs[2][1].image_path), jobs[2][2])
    try:
        a = ad.forward(prep, output_attentions=True).attentions
        attn_probe[impl] = "none" if a is None or all(x is None for x in a) else "returned"
    except Exception as e:
        attn_probe[impl] = f"error: {type(e).__name__}"
    out = []
    for task, s, prompt, mnt in jobs:
        g = ad.generate(ad.prepare(Image.open(s.image_path), prompt), mnt)
        out.append({"task": task, "sample_id": s.sample_id, "tokens": g["token_ids"], "text": g["text_raw"],
                    "option_logits": [float(g["first_logits"][opt[L]]) for L in LETTERS] if task == "relation" else None})
    runs[impl] = out
    print(impl, "done; attentions:", attn_probe[impl], flush=True)
use("observed")
o, r = runs["observed"], runs["reference"]


def same(t):
    return sum(a["tokens"] == b["tokens"] for a, b in zip(o, r) if a["task"] == t)


def amax(v):
    return max(range(4), key=v.__getitem__)


rel = [(a, b) for a, b in zip(o, r) if a["task"] == "relation"]
diffs = [max(abs(x - y) for x, y in zip(a["option_logits"], b["option_logits"])) for a, b in rel]
res = {"model": name, "observed": "eager + fp32 scoring patch", "observed_patch": ad.eager_fp32, "reference": "sdpa",
       "attentions_available": attn_probe,
       "switch_effective": attn_probe.get("observed") == "returned" and attn_probe.get("reference") != "returned",
       "bbox_token_identical": same("bbox"), "bbox_n": 40, "relation_token_identical": same("relation"), "relation_n": 20,
       "option_logit_max_abs_diff": max(diffs), "option_logit_mean_abs_diff": sum(diffs) / len(diffs),
       "option_argmax_identical": sum(amax(a["option_logits"]) == amax(b["option_logits"]) for a, b in rel),
       "first_differing_examples": [{"task": a["task"], "sample_id": a["sample_id"], "observed": a["text"][:120], "reference": b["text"][:120]}
                                    for a, b in zip(o, r) if a["tokens"] != b["tokens"]][:5],
       "runs": runs}
json.dump(res, open(paths.get("outputs") / "smoke" / name / "default_vs_eager.json", "w"), indent=1, ensure_ascii=False)
print("SUMMARY", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                        if k not in ("runs", "first_differing_examples", "observed_patch", "model")})
