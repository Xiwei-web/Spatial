"""显存与耗时测试（用户 2026-11-25 要求；不算正确率）：float32 + eager 下单样本前向，取出全部层注意力权重。
用法：python scripts/bench_attention_memory.py <模型> <标签>      可见的 GPU 数由 CUDA_VISIBLE_DEVICES 决定
对冒烟 20 张里的前 5 张（target 的框提示）各测四种：
  forward            前向，不保留注意力权重
  attn_on_gpu        前向，全部层注意力权重同时留在 GPU（output_attentions=True 的常规用法）
  attn_to_cpu        前向，逐层用 hook 取出注意力权重并立即搬到 CPU
  generate           框提示贪心生成（max_new_tokens=96，不取注意力）
每种先预热 1 次；记录每卡峰值 max_memory_allocated / reserved 和耗时中位数。OOM 如实记录。
输出：<outputs>/diagnostics/bench_attn_<模型>_<标签>.json"""
import json
import statistics
import subprocess
import sys
import time
import traceback

import torch
from PIL import Image

from scripts.smoke_test import bbox_convention
from src import paths
from src.data import load_dataset
from src.models.adapter import load_adapter
from src.readout import internvl_embeds

name, tag = sys.argv[1:3]
N = torch.cuda.device_count()
res = {"model": name, "tag": tag, "dtype": "float32", "attn": "eager", "n_gpu": N,
       "gpu": [torch.cuda.get_device_name(i) for i in range(N)], "gpu_total_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2) for i in range(N)]}
OUT = paths.get("outputs") / "diagnostics" / f"bench_attn_{name}_{tag}.json"


def save():
    res["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("SUMMARY", json.dumps(res, ensure_ascii=False), flush=True)


def peaks(f):
    return [round(f(i) / 2**30, 2) for i in range(N)]


try:
    ad = load_adapter(name, torch.float32)
except Exception as e:
    res["load_error"] = f"{type(e).__name__}: {e}"[:500]
    save()
    raise SystemExit(0)
devs = {str(p.device) for p in ad.model.parameters()}
res.update(param_devices=sorted(devs), offloaded=any(d in ("cpu", "meta") for d in devs), mem_after_load_gb=peaks(torch.cuda.memory_allocated),
           param_dtypes=sorted({str(p.dtype) for p in ad.model.parameters()}))
if res["offloaded"]:
    res["note"] = "accelerate 把部分权重放到了 CPU/磁盘：该卡数下放不下，后面的数字不代表纯 GPU 运行"
    save()
    raise SystemExit(0)
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"][:5]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
bc = bbox_convention(name)
preps = []
for it in items:
    s = by_id[(it["dataset"], it["sample_id"])]
    preps.append(ad.prepare(Image.open(s.image_path), bc["prompt_template"].format(object=s.target)))
res["n_visual_tokens"] = [len(p.visual_positions) for p in preps]
res["n_input_tokens"] = [int(p.inputs["input_ids"].shape[1]) for p in preps]
internvl = ad.kind == "internvl_remote"


@torch.no_grad()
def fwd(p, **kw):
    if internvl:          # 远程 forward 多卡下报错，改为与 generate 相同的嵌入构造后直接驱动语言模型
        return ad.model.language_model(inputs_embeds=internvl_embeds(ad, p), attention_mask=p.inputs["attention_mask"], return_dict=True, use_cache=False, **kw)
    return ad.model(**p.inputs, return_dict=True, use_cache=False, **kw)


def run_forward(p):
    fwd(p)
    return {}


def run_gpu(p):
    a = fwd(p, output_attentions=True).attentions
    return {"n_layers": len(a), "shape": list(a[0].shape), "dtype": str(a[0].dtype), "devices": sorted({str(x.device) for x in a}),
            "attn_total_gb": round(sum(x.numel() * x.element_size() for x in a) / 2**30, 2)}


def run_cpu(p):
    got, hs = [], []
    for l in ad.lm_layers:
        hs.append(l.self_attn.register_forward_hook(lambda m, i, o: got.append(o[1].cpu())))
    try:
        fwd(p)
    finally:
        for h in hs:
            h.remove()
    return {"n_layers": len(got), "shape": list(got[0].shape), "dtype": str(got[0].dtype)}


def run_gen(p):
    return {"n_new_tokens": len(ad.generate(p, 96)["token_ids"])}


for mode, fn in (("forward", run_forward), ("attn_on_gpu", run_gpu), ("attn_to_cpu", run_cpu), ("generate", run_gen)):
    r = {}
    try:
        torch.cuda.empty_cache()
        fn(preps[0])
        for i in range(N):
            torch.cuda.reset_peak_memory_stats(i)
        ts, info = [], None
        for p in preps:
            torch.cuda.synchronize()
            t0 = time.time()
            info = fn(p)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        r = {"ok": True, "seconds_median": round(statistics.median(ts), 3), "seconds": [round(t, 3) for t in ts],
             "peak_allocated_gb": peaks(torch.cuda.max_memory_allocated), "peak_reserved_gb": peaks(torch.cuda.max_memory_reserved), **info}
    except torch.cuda.OutOfMemoryError as e:
        r = {"ok": False, "error": f"OutOfMemoryError: {e}"[:400], "peak_allocated_gb": peaks(torch.cuda.max_memory_allocated)}
    except Exception as e:
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "traceback": traceback.format_exc()[-1200:]}
    res[mode] = r
    torch.cuda.empty_cache()
    print(mode, r, flush=True)
save()
