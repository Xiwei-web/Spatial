"""关系问法复用图像前缀 KV cache 的可行性与节省估算（CPU；只用分词器 / 处理器和已有记录，不加载模型、不实现复用）。
用法：python scripts/e0_prefix_reuse_estimate.py → manifests/e0_prefix_reuse_estimate.json
1. 可行性：对每个模型，把同一张图配两个不同的关系问题，比较两条输入 token 序列的公共前缀是否覆盖全部视觉 token（公共前缀长度 ≥ 最后一个视觉 token 的下标 + 1）。
2. 估算：每条关系问法的耗时 t_rel（冒烟）≈ 前缀（视觉编码 + 图像前缀 prefill）+ 解码。每个新 token 的解码耗时 t_tok 由 bbox_calib 与冒烟的差估计：
   t_tok = (t_box − t_rel) / (N_box − N_rel)，N 为平均新 token 数；前缀耗时 ≈ t_rel − N_rel × t_tok（问题文字本身的 prefill 忽略不计，偏乐观）。
   一个样本 6 次问法（4 次四选一 + 2 次二元）共用一次前缀，节省 5/6 × 前缀耗时 × 问法次数。"""
import json
import statistics as st

import torch
import yaml
from PIL import Image

from scripts.e0_estimate import MODELS, counts, speeds
from src import paths
from src.data import load_dataset
from src.models.adapter import adapter_spec
from src.prompts import binary_question, relation_mcq

O = paths.get("outputs")
cal = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "bbox_conventions.yaml"))["calibrated"]
s0 = load_dataset("whatsup_a")[0]
img = Image.open(s0.image_path).convert("RGB")
q1 = relation_mcq("whatsup_a", s0.target, s0.reference)[0]
q2 = binary_question("whatsup_a", s0.target, s0.reference, "left")
res = {}


def tok_hf(name, q):
    from transformers import AutoProcessor
    sp = adapter_spec(name)
    p = AutoProcessor.from_pretrained(paths.get(sp["root"]) / name)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    kw = {}
    if sp["grid"] == "qwen_thw" and sp.get("max_pixels"):
        kw = {"processor_kwargs": {"images_kwargs": ({"size": dict(sp["image_size"])} if sp.get("image_size") else {"max_pixels": int(sp["max_pixels"])})}}
    im2 = img
    if sp.get("pad_to_square"):
        S = max(img.size); im2 = Image.new("RGB", (S, S)); im2.paste(img, ((S - img.size[0]) // 2, (S - img.size[1]) // 2))
        msgs[0]["content"][0]["image"] = im2
    out = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **(sp.get("chat_kwargs") or {}), **kw)
    ids = out["input_ids"][0].tolist()
    imtok = p.tokenizer.convert_tokens_to_ids(getattr(p, "image_token", None) or "<image>")
    return ids, imtok, p.tokenizer


def tok_internvl(name, q):
    from transformers import AutoTokenizer
    sp = adapter_spec(name)
    from src.models.adapter import model_facts
    # code_revision 与 InternVLRemoteAdapter 相同：SenseNova 的 auto_map 带仓库前缀，远程代码从 Hub 取，必须固定到权重同一 commit
    t = AutoTokenizer.from_pretrained(paths.get(sp["root"]) / name, trust_remote_code=True, use_fast=False, code_revision=model_facts(name).get("revision"))
    # 与 InternVLRemoteAdapter.prepare 相同的拼法：<img> + 256 × <IMG_CONTEXT> + </img>，后接 "\n" + 问题（对话模板的系统消息在前，对两个问题相同）
    query = "<|im_start|>user\n" + "<img>" + "<IMG_CONTEXT>" * 256 + "</img>\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
    return t(query)["input_ids"], t.convert_tokens_to_ids("<IMG_CONTEXT>"), t


sp_all = speeds()
C = counts("test")
for name in MODELS[1] + MODELS[2]:
    kind = adapter_spec(name)["kind"]
    f = tok_internvl if kind == "internvl_remote" else tok_hf
    a, imtok, tk = f(name, q1)
    b, _, _ = f(name, q2)
    lcp = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    last_img = max(i for i, x in enumerate(a) if x == imtok)
    # 平均新 token 数
    last = {}
    for l in open(O / "bbox_calib" / name / "records.jsonl"):
        r = json.loads(l)
        last[(r["prompt_id"], r["item_id"])] = r
    nb = st.mean(r["n_new_tokens"] for k, r in last.items() if k[0] == cal[name]["prompt_id"] and r["subset"] == "whatsup")
    sm = [json.loads(l) for l in open(O / "smoke" / name / "records.jsonl")]
    nr = st.mean(len(tk.encode(r["raw_text"], add_special_tokens=False)) for r in sm if r["task"] == "relation_mcq" and r.get("raw_text") is not None)
    s = sp_all[name]
    t_tok = max(0.0, (s["box_whatsup"] - s["rel_whatsup"]) / max(nb - nr, 1e-9))
    pre = max(0.0, s["rel_whatsup"] - nr * t_tok)
    frac = pre / s["rel_whatsup"]
    rel_h = sum((C[d]["mcq"] + C[d]["binary"]) * (s["rel_whatsup"] if d.startswith("whatsup") else s["rel_other"]) for d in C) / 3600
    rel_h_lean = sum((C[d]["mcq"] + C[d]["binary"]) * (s["rel_whatsup"] if d.startswith("whatsup") else s["rel_other"]) for d in ("whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two")) / 3600
    res[name] = {"n_input_tokens": len(a), "last_visual_token_index": last_img, "common_prefix_len": lcp, "prefix_covers_all_visual_tokens": lcp > last_img,
                 "mean_new_tokens_box": nb, "mean_new_tokens_rel_smoke": nr, "t_tok_s": t_tok, "t_prefix_s": pre, "prefix_fraction_of_rel": frac,
                 "rel_hours_full": rel_h, "saved_hours_full": rel_h * frac * 5 / 6, "rel_hours_lean": rel_h_lean, "saved_hours_lean": rel_h_lean * frac * 5 / 6}
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}))
tot = {k: sum(r[k] for r in res.values()) for k in ("rel_hours_full", "saved_hours_full", "rel_hours_lean", "saved_hours_lean")}
print("TOTAL", {k: round(v, 1) for k, v in tot.items()})
json.dump({"per_model": res, "total": tot, "questions_used": [q1, q2]}, open(paths.REPO_ROOT / "manifests" / "e0_prefix_reuse_estimate.json", "w"), indent=1, ensure_ascii=False)
