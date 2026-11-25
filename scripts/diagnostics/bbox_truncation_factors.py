"""1.3-1 / 1.3-3 的原因排查（不是评测，不改判定标准）。
用法：python scripts/diagnostics/bbox_truncation_factors.py <模型名>
对冒烟的 40 条框提示词，在“冒烟设置”基础上每次只改一个因素重新贪心生成，统计能解析出四个数的次数：
  baseline            冒烟设置：max_new_tokens=96，EOS=模型 generation_config 默认，repetition_penalty=1.0
  max_new_tokens=256  只改生成长度上限
  eos=<|im_end|> only 只改 EOS：只把 <|im_end|> 当作结束符
  eos=disabled        只改 EOS：不设结束符（看截断处之后模型会继续生成什么）
  repetition_penalty=shipped  只改重复惩罚：用仓库 generation_config 里自带的值
另外把 baseline 重复一次检查可复现性，并保存每条的完整输入 token 与输出 token。"""
import json
import sys

import torch
from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import GREEDY, load_adapter
from src.prompts import parse_four_numbers

name = sys.argv[1]
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
ad = load_adapter(name)
gc = ad.model.generation_config
shipped = {"eos_token_id": gc.eos_token_id, "repetition_penalty": gc.repetition_penalty, "do_sample": gc.do_sample,
           "temperature": gc.temperature, "top_p": gc.top_p, "top_k": gc.top_k, "pad_token_id": gc.pad_token_id}
im_end = ad.tokenizer.convert_tokens_to_ids("<|im_end|>")
variants = {
    "baseline": {},
    "baseline_repeat": {},
    "max_new_tokens=256": {"max_new_tokens": 256},
    "eos=<|im_end|> only": {"eos_token_id": im_end},
    "eos=disabled": {"eos_token_id": None, "min_new_tokens": 0, "stop_disabled": True},
    "repetition_penalty=shipped": {"repetition_penalty": shipped["repetition_penalty"]},
}
out = {"model": name, "shipped_generation_config": shipped, "im_end_id": im_end, "prompt_template": bc["prompt_template"], "variants": {}}
preps = []
for s in samples:
    for role, obj in (("target", s.target), ("reference", s.reference)):
        preps.append((s.sample_id, role, bc["prompt_template"].format(object=obj), s.image_path))
for vname, v in variants.items():
    kw = dict(GREEDY, max_new_tokens=96)
    kw.update({k: val for k, val in v.items() if k != "stop_disabled"})
    recs = []
    for sid, role, prompt, path in preps:
        prep = ad.prepare(Image.open(path), prompt)
        gen_kw = dict(kw)
        if v.get("stop_disabled"):
            gen_kw.pop("eos_token_id")
            gen_kw["eos_token_id"] = -1          # 不存在的 id：永不触发停止
            gen_kw["pad_token_id"] = ad.tokenizer.pad_token_id or 0
        with torch.no_grad():
            g = ad.model.generate(**prep.inputs, **gen_kw)
        new = g[0][prep.inputs["input_ids"].shape[1]:].tolist() if "input_ids" in prep.inputs and ad._sequences_include_prompt else g[0].tolist()
        text = ad.tokenizer.decode(new, skip_special_tokens=False)
        recs.append({"sample_id": sid, "role": role, "prompt": prompt, "output_token_ids": new, "raw_text": text,
                     "n_new_tokens": len(new), "parsed": parse_four_numbers(text),
                     "input_ids": prep.inputs["input_ids"][0].tolist() if vname == "baseline" and "input_ids" in prep.inputs else None})
    n_ok = sum(r["parsed"] is not None for r in recs)
    out["variants"][vname] = {"settings": {k: (str(val) if val is None else val) for k, val in v.items()}, "n": len(recs), "n_parsed": n_ok, "records": recs}
    print(f"{vname:32s} parsed {n_ok}/{len(recs)}  mean_new_tokens {sum(r['n_new_tokens'] for r in recs) / len(recs):.1f}", flush=True)
b, r = out["variants"]["baseline"]["records"], out["variants"]["baseline_repeat"]["records"]
print("baseline reproducible (token-identical):", sum(x["output_token_ids"] == y["output_token_ids"] for x, y in zip(b, r)), "/", len(b))
bad = [x for x in b if x["parsed"] is None]
print("baseline unparsed:", len(bad))
for x in bad[:6]:
    print("   ", x["sample_id"], x["role"], repr(x["raw_text"][:120]))
    for vn in list(variants)[2:]:
        y = next(z for z in out["variants"][vn]["records"] if (z["sample_id"], z["role"]) == (x["sample_id"], x["role"]))
        print(f"        {vn:28s} -> parsed={y['parsed'] is not None} {repr(y['raw_text'][:110])}")
d = paths.get("outputs") / "diagnostics"
d.mkdir(parents=True, exist_ok=True)
json.dump(out, open(d / f"bbox_truncation_{name}.json", "w"), indent=1, ensure_ascii=False)
