"""诊断：框预测输出中途截断/数字错乱是否与 (注意力实现, 精度) 有关。
对同一模型、同一批冒烟图片的框提示词，在几种 (attn_implementation, dtype) 组合下贪心生成，统计能解析出四个数的次数。
只用于排查管线问题，不是评测。用法：python scripts/diagnostics/attn_dtype_check.py <hf_chat 模型名>"""
import json
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import GREEDY, HFChatAdapter
from src.prompts import parse_four_numbers

name = sys.argv[1]
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
bc = bbox_convention(name)
out = {}
ad = HFChatAdapter(name)
for impl, dtype in [("eager", torch.bfloat16), ("sdpa", torch.bfloat16), ("eager", torch.float32), ("sdpa", torch.float32)]:
    key = f"{impl}/{str(dtype).split('.')[-1]}"
    try:
        ad.load()
        del ad.model
        torch.cuda.empty_cache()
        ad.model = AutoModelForImageTextToText.from_pretrained(ad.path, dtype=dtype, device_map="auto", attn_implementation=impl).eval()
        texts = []
        for s in samples:
            for obj in (s.target, s.reference):
                prep = ad.prepare(Image.open(s.image_path), bc["prompt_template"].format(object=obj))
                with torch.no_grad():
                    g = ad.model.generate(**prep.inputs, max_new_tokens=96, **GREEDY)
                texts.append(ad.tokenizer.decode(g[0][prep.inputs["input_ids"].shape[1]:], skip_special_tokens=False))
        out[key] = {"n": len(texts), "n_parsed": sum(parse_four_numbers(t) is not None for t in texts), "texts": texts}
    except Exception as e:
        out[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    print(key, {k: v for k, v in out[key].items() if k != "texts"}, flush=True)
    del ad.model
    torch.cuda.empty_cache()
keys = [k for k in out if "texts" in out[k]]
for a in keys[1:]:
    same = sum(x == y for x, y in zip(out[keys[0]]["texts"], out[a]["texts"]))
    print(f"identical outputs {keys[0]} vs {a}: {same}/{len(out[a]['texts'])}")
d = paths.get("outputs") / "diagnostics"
d.mkdir(parents=True, exist_ok=True)
json.dump(out, open(d / f"attn_dtype_{name}.json", "w"), indent=1, ensure_ascii=False)
