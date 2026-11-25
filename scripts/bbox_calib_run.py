"""框坐标约定标定：生成（GPU）。每个模型、每个提示词、每个 item 一条 JSONL，断点续跑；只记录，不算 IoU。
用法：python scripts/bbox_calib_run.py <模型> [--max-new 96]   模型的提示词集合（P0，或 P0+P1）由本脚本的 PROMPT_SET 决定
输出 <outputs>/bbox_calib/<模型>/records.jsonl：原始文本、解析结果、输入尺寸、提示词、模型 revision、git commit 等。已存在（非 error）的 item 跳过；error 的会重跑一次并追加新行，读取时取最后一行。"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from scripts.smoke_test import GENERIC_BBOX_PROMPT, bbox_convention, git_commit
from src import paths
from src.data import load_dataset
from src.models.adapter import adapter_spec, config_hash, load_adapter, model_facts
from src.prompts import parse_four_numbers

QWEN_OFFICIAL = "Locate {object}, output its bbox coordinates using JSON format."
INTERNVL_OFFICIAL = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{object}</ref>"
P1 = {**{m: INTERNVL_OFFICIAL for m in ("SenseNova-SI-1.1-InternVL3-2B", "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B")},
      "VST-7B-RL": QWEN_OFFICIAL, "RoboBrain2.0-7B": QWEN_OFFICIAL,
      "Qwen3-VL-8B-Instruct": QWEN_OFFICIAL + " Output only bbox_2d in JSON. Do not output 3D boxes."}


def prompt_set(name):
    """返回 {提示词编号: 模板}；P1 与 P0 逐字相同就跳过。"""
    p0 = bbox_convention(name)["prompt_template"]
    out = {"P0": p0}
    if name in P1 and P1[name] != p0:
        out["P1"] = P1[name]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--max-new", type=int, default=96)
    A = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 GPU")
    name = A.name
    items = json.load(open(paths.REPO_ROOT / "splits" / "bbox_calib_v1.json"))["items"]
    by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
    ps = prompt_set(name)
    out_dir = paths.get("outputs") / "bbox_calib" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "records.jsonl"
    done = set()
    if f.is_file():
        for l in open(f):
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["prompt_id"], r["item_id"]))
            else:
                done.discard((r["prompt_id"], r["item_id"]))
    todo = [(pid, i) for pid in ps for i in items if (pid, i["item_id"]) not in done]
    print(name, "prompts", list(ps), "todo", len(todo), "done", len(done), flush=True)
    if todo:
        import transformers
        ad = load_adapter(name)
        meta = {"model": name, "revision": model_facts(name).get("revision"), "config_hash": config_hash(name), "git_commit": git_commit(), "env": adapter_spec(name).get("env"),
                "torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(ad.dtype), "eager_fp32_patch": ad.eager_fp32, "decoding": "greedy", "max_new_tokens": A.max_new,
                "slurm_job": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0)}
        t0 = time.time()
        with open(f, "a") as fout:
            for n, (pid, i) in enumerate(todo):
                s = by_id[(i["dataset"], i["sample_id"])]
                prompt = ps[pid].format(object=i["object"])
                rec = {**meta, "prompt_id": pid, "prompt": prompt, "item_id": i["item_id"], "dataset": i["dataset"], "subset": i["subset"], "sample_id": i["sample_id"], "role": i["role"], "object": i["object"],
                       "image_key": i["image_key"], "truth_xyxy": i["truth_xyxy"], "orig_wh": i["orig_wh"], "error": None}
                try:
                    t = time.time()
                    img = Image.open(s.image_path).convert("RGB")
                    assert list(img.size) == i["orig_wh"]
                    prep = ad.prepare(img, prompt)
                    g = ad.generate(prep, A.max_new)
                    nums = parse_four_numbers(g["text"]) or parse_four_numbers(g["text_raw"])
                    vb = prep.extra.get("view_box")
                    rec.update(text=g["text"], text_raw=g["text_raw"], n_new_tokens=len(g["token_ids"]), hit_max_tokens=len(g["token_ids"]) >= A.max_new, parsed=nums, model_wh=list(prep.model_wh) if prep.model_wh else None,
                               view_box=list(vb) if vb else None, n_input_tokens=int(prep.inputs["attention_mask"].shape[1]), n_visual_tokens=len(prep.visual_positions), seconds=round(time.time() - t, 3))
                except Exception as e:
                    rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", parsed=None, model_wh=None, view_box=None)
                    print("ERROR", pid, i["item_id"], rec["error"], flush=True)
                    torch.cuda.empty_cache()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                if n % 50 == 0:
                    print(name, pid, n, "/", len(todo), round(time.time() - t0), "s", flush=True)
        print(name, "finished", round(time.time() - t0, 1), "s", "peak_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
    print("done", name)
