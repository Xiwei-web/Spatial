"""诊断：eager 注意力 + bf16 下框输出错乱的来源定位（不是评测）。
用法：python scripts/diagnostics/eager_localize.py <hf_chat 模型名>
同一模型只加载一次（bf16），通过改各注意力模块读取的 config._attn_implementation 切换实现：
  all_sdpa / all_eager / vision_eager+lm_sdpa / vision_sdpa+lm_eager / vision_sdpa+lm_eager_fp32
其中 lm_eager_fp32 = 语言模型的 eager 注意力把 q·k 的打分放到 float32 里计算（权重仍可取出）。
对冒烟的 40 条框提示词统计能解析出四个数的次数；并在第一条输入上比较各设置与 all_sdpa 的末位置 logits 差异。"""
import json
import sys

import torch
from PIL import Image
from torch import nn

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import GREEDY, load_adapter
from src.prompts import parse_four_numbers

name = sys.argv[1]
ad = load_adapter(name)                      # eager, bf16
mod = sys.modules[type(ad.lm_layers[0].self_attn).__module__]
orig_eager = mod.eager_attention_forward
lm_attn_ids = {id(l.self_attn) for l in ad.lm_layers}


def eager_fp32(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if id(module) not in lm_attn_ids:        # 视觉塔保持原实现
        return orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    g = module.num_key_value_groups
    k = mod.repeat_kv(key, g).float()
    v = mod.repeat_kv(value, g)
    w = torch.matmul(query.float(), k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask.float()
    w = nn.functional.softmax(w, dim=-1)
    out = torch.matmul(w.to(query.dtype), v).transpose(1, 2).contiguous()
    return out, w.to(query.dtype)


cfgs = {}
for n, m in ad.model.named_modules():
    c = getattr(m, "config", None)
    if c is not None and hasattr(c, "_attn_implementation"):
        tag = "lm" if (id(m) in lm_attn_ids or "language_model" in n) else ("vision" if any(s in n for s in ("visual", "vision")) else "other")
        cfgs.setdefault(id(c), {"cfg": c, "tags": set(), "type": type(c).__name__})["tags"].add(tag)
print("config objects:", [(v["type"], sorted(v["tags"])) for v in cfgs.values()], flush=True)
shared = any({"lm", "vision"} <= v["tags"] for v in cfgs.values())
print("LM 与视觉塔是否共用同一个 config 对象:", shared, flush=True)


def set_impl(vision, lm, fp32=False):
    for v in cfgs.values():
        if "lm" in v["tags"] and "vision" not in v["tags"]:
            v["cfg"]._attn_implementation = lm
        elif "vision" in v["tags"] and "lm" not in v["tags"]:
            v["cfg"]._attn_implementation = vision
        else:                                 # 顶层/共用：跟随 LM
            v["cfg"]._attn_implementation = lm
    mod.eager_attention_forward = eager_fp32 if fp32 else orig_eager


items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
jobs = [(s.image_path, bc["prompt_template"].format(object=o)) for s in samples for o in (s.target, s.reference)]
settings = {"all_sdpa": ("sdpa", "sdpa", False), "all_eager": ("eager", "eager", False), "vision_eager+lm_sdpa": ("eager", "sdpa", False),
            "vision_sdpa+lm_eager": ("sdpa", "eager", False), "vision_sdpa+lm_eager_fp32": ("sdpa", "eager", True),
            "all_eager+lm_fp32": ("eager", "eager", True)}
out, ref_logits, ref_texts = {}, None, None
for tag, (vi, lm, fp) in settings.items():
    set_impl(vi, lm, fp)
    prep = ad.prepare(Image.open(jobs[0][0]), jobs[0][1])
    with torch.no_grad():
        lg = ad.model(**prep.inputs).logits[0, -1].float()
    texts = []
    for path, prompt in jobs:
        prep = ad.prepare(Image.open(path), prompt)
        with torch.no_grad():
            g = ad.model.generate(**prep.inputs, max_new_tokens=96, **GREEDY)
        texts.append(ad.tokenizer.decode(g[0][prep.inputs["input_ids"].shape[1]:], skip_special_tokens=False))
    if ref_logits is None:
        ref_logits, ref_texts = lg, texts
    out[tag] = {"n_parsed": sum(parse_four_numbers(t) is not None for t in texts), "n": len(texts),
                "identical_text_to_all_sdpa": sum(a == b for a, b in zip(texts, ref_texts)),
                "last_logits_max_abs_diff_vs_all_sdpa": float((lg - ref_logits).abs().max()),
                "top1_same_as_all_sdpa": bool(lg.argmax() == ref_logits.argmax()), "texts": texts}
    print(f"{tag:28s}", {k: v for k, v in out[tag].items() if k != "texts"}, flush=True)
d = paths.get("outputs") / "diagnostics"
d.mkdir(parents=True, exist_ok=True)
json.dump({"model": name, "shared_config": shared, "results": out}, open(d / f"eager_localize_{name}.json", "w"), indent=1, ensure_ascii=False)
